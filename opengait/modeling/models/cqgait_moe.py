"""
CQGaitMoE: Condition-Quality Guided Part-MoE for RGB-PC gait recognition.

Drop this file into:
    OpenGait/opengait/modeling/models/cqgait_moe.py

The model keeps GaitOracle's RGB/PC/text-disentanglement backbone style, and adds:
  - Appearance Part-MoE for RGB part tokens
  - Structure Part-MoE for PC part tokens
  - Supervised ConditionQualityEncoder from raw RGB/PC statistics
  - Token-attentive Fusion Part-MoE with hybrid part queries
  - MoE importance/load/diversity/z-loss and quality-aligned routing

Recognition inputs at inference: RGB + point cloud.
Text/Qwen nuisance features and masks are only used as training supervision.
"""

import os
import math
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# ============================================================
# Basic utilities
# ============================================================

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


def l2n(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _safe_zero_like(x):
    return x.new_tensor(0.0)


def _cv_squared(x, eps=1e-10):
    x = x.float()
    mean = x.mean().clamp_min(eps)
    return x.var(unbiased=False) / (mean * mean + eps)


def symmetric_infonce(z1, z2, temperature=0.07):
    if z1 is None or z2 is None:
        base = z1 if z1 is not None else z2
        return _safe_zero_like(base)
    if z1.size(0) <= 1 or z2.size(0) <= 1:
        return _safe_zero_like(z1)
    z1 = l2n(z1.float(), dim=-1)
    z2 = l2n(z2.float(), dim=-1)
    logits = z1 @ z2.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
    if z_img is None or z_txt is None:
        base = z_img if z_img is not None else z_txt
        return _safe_zero_like(base)

    if mask is not None:
        mask = mask.float().view(mask.size(0), -1).mean(-1)
        valid = mask > 0
        if valid.sum() <= 1:
            return z_img.new_tensor(0.0)
        z_img = z_img[valid]
        z_txt = z_txt[valid]
        weights = mask[valid].clamp(0, 1)
    else:
        if z_img.size(0) <= 1:
            return z_img.new_tensor(0.0)
        weights = None

    z_img = l2n(z_img.float(), dim=-1)
    z_txt = l2n(z_txt.float(), dim=-1)
    logits = z_img @ z_txt.t() / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)

    if weights is None:
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    loss_i = F.cross_entropy(logits, labels, reduction='none')
    loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
    weights = weights / weights.mean().clamp_min(1e-6)
    return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


def ramp_value(step: int, start: int, length: int, max_value: float = 1.0):
    if length <= 0:
        return max_value if step >= start else 0.0
    if step < start:
        return 0.0
    return float(max_value) * min(1.0, max(0.0, (step - start) / float(length)))


class RunningNorm(nn.Module):
    """Running normalization for handcrafted quality statistics."""

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('initialized', torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float()
        if x.numel() == 0:
            return
        bmean = x.mean(dim=0)
        bvar = x.var(dim=0, unbiased=False).clamp_min(self.eps)
        if not bool(self.initialized.item()):
            self.mean.copy_(bmean)
            self.var.copy_(bvar)
            self.initialized.fill_(True)
        else:
            self.mean.mul_(1.0 - self.momentum).add_(self.momentum * bmean)
            self.var.mul_(1.0 - self.momentum).add_(self.momentum * bvar)

    def forward(self, x: torch.Tensor, update: bool = False):
        if self.training and update:
            self.update(x)
        return (x.float() - self.mean) / torch.sqrt(self.var + self.eps)


# ============================================================
# Text modules inherited from GaitOracle design
# ============================================================

class TextEmbeddingAdapter(nn.Module):
    """Compress high-dimensional MLLM text embeddings, e.g. Qwen2.5-VL 3584-d."""

    def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024, dropout=0.1, l2_norm=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.code_dim = int(code_dim)
        self.l2_norm = bool(l2_norm)
        hidden_dim = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
        self.in_norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.code_dim, bias=False),
            nn.LayerNorm(self.code_dim),
        )

    def _fit_dim(self, x):
        d = x.size(-1)
        if d == self.input_dim:
            return x
        if d > self.input_dim:
            return x[..., :self.input_dim]
        pad = x.new_zeros(*x.shape[:-1], self.input_dim - d)
        return torch.cat([x, pad], dim=-1)

    def forward(self, x):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.mean(1)
        elif x.dim() > 3:
            x = x.view(x.size(0), -1)
        x = self._fit_dim(x.float())
        z = self.net(self.in_norm(x))
        return l2n(z, dim=-1) if self.l2_norm else z


class PositiveTextGate(nn.Module):
    """Fixed gait-positive prompt bank -> channel/spatial gait gate for RGB features."""

    def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
                 trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024, dropout=0.1,
                 enabled=True):
        super().__init__()
        self.channels = int(channels)
        self.enabled = bool(enabled)

        if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
            arr = np.load(prompt_feat_path).astype('float32')
            if arr.ndim == 1:
                arr = arr[None, :]
            prompt = torch.from_numpy(arr)
            text_dim = int(prompt.shape[-1])
        else:
            prompt = torch.randn(num_prompts, text_dim) * 0.02

        self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
        self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim, dropout=dropout, l2_norm=False)
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_code_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        mid = max(channels // 4, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_map):
        # feat_map: [B,T,C,H,W]
        if not self.enabled:
            B, T, _, H, W = feat_map.shape
            return feat_map, feat_map.new_ones(B, T, 1, H, W)

        B, T, C, H, W = feat_map.shape
        prompt = self.prompt_feat.to(device=feat_map.device)
        pos_code = self.text_adapter(prompt).mean(0)
        pos = self.text_to_channel(pos_code)
        ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)
        x = feat_map * (1.0 + ch_gate)
        sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
        return guided, sp_gate


class StagedNuisanceRemoval(nn.Module):
    """Training-time nuisance disentanglement using Qwen text/mask supervision."""

    def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256, text_hidden_dim=1024,
                 sub_dim=256, factors=None, grl_lambda=0.08, temperature=0.07,
                 warmup_iters=5000, probe_iters=10000, adv_ramp_iters=10000,
                 reliability_start_iter=15000, reliability_ramp_iters=10000,
                 probe_loss_weight=0.03, adv_loss_weight=0.03, mask_loss_weight=0.01,
                 max_suppression=0.35, detach_reliability_gate=True, factor_weights=None,
                 dropout=0.1):
        super().__init__()
        self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
        self.parts_num = int(parts_num)
        self.grl_lambda = float(grl_lambda)
        self.temperature = float(temperature)
        self.warmup_iters = int(warmup_iters)
        self.probe_iters = int(probe_iters)
        self.adv_ramp_iters = int(adv_ramp_iters)
        self.reliability_start_iter = int(reliability_start_iter)
        self.reliability_ramp_iters = int(reliability_ramp_iters)
        self.probe_loss_weight = float(probe_loss_weight)
        self.adv_loss_weight = float(adv_loss_weight)
        self.mask_loss_weight = float(mask_loss_weight)
        self.max_suppression = float(max_suppression)
        self.detach_reliability_gate = bool(detach_reliability_gate)
        self.factor_weights = {k: 1.0 for k in self.factors}
        if factor_weights:
            self.factor_weights.update({str(k): float(v) for k, v in factor_weights.items()})

        self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim, dropout=dropout, l2_norm=True)
        self.txt_heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim))
            for k in self.factors
        })
        self.visual_probe = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(in_dim, sub_dim),
                nn.LayerNorm(sub_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(sub_dim, sub_dim),
                nn.LayerNorm(sub_dim),
            ) for k in self.factors
        })
        self.risk_heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(sub_dim, max(sub_dim // 2, 16)), nn.GELU(), nn.Linear(max(sub_dim // 2, 16), 1))
            for k in self.factors
        })
        self.part_heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(sub_dim, sub_dim), nn.GELU(), nn.Linear(sub_dim, self.parts_num))
            for k in self.factors
        })

    def schedule(self, step: int):
        probe_start = self.warmup_iters
        adv_start = self.warmup_iters + self.probe_iters
        return {
            'probe': ramp_value(step, probe_start, max(1, self.probe_iters), 1.0),
            'adv': ramp_value(step, adv_start, max(1, self.adv_ramp_iters), 1.0),
            'rel': ramp_value(step, self.reliability_start_iter, max(1, self.reliability_ramp_iters), 1.0),
        }

    def _target_mask(self, masks, factor):
        if masks is None or factor not in masks or masks[factor] is None:
            return None
        m = masks[factor].float()
        if m.dim() > 1:
            m = m.view(m.size(0), -1).mean(-1)
        return m.clamp(0, 1)

    def _text_z(self, text_feats, factor):
        if text_feats is None or factor not in text_feats or text_feats[factor] is None:
            return None
        code = self.text_adapter(text_feats[factor])
        return l2n(self.txt_heads[factor](code), dim=-1)

    def pre_temporal(self, visual_global, text_feats=None, masks=None, step: int = 0):
        sched = self.schedule(step)
        B = visual_global.size(0)
        losses = {}
        bad_parts = visual_global.new_zeros(B, self.parts_num)

        for f in self.factors:
            z_probe = l2n(self.visual_probe[f](visual_global.detach().float()), dim=-1)
            target = self._target_mask(masks, f)
            z_txt = self._text_z(text_feats, f)

            if z_txt is not None and sched['probe'] > 0 and self.probe_loss_weight > 0:
                losses[f'probe_{f}'] = masked_symmetric_infonce(
                    z_probe, z_txt, mask=target, temperature=self.temperature
                ) * (self.probe_loss_weight * sched['probe'])

            risk = torch.sigmoid(self.risk_heads[f](z_probe)).squeeze(-1)
            if target is not None and sched['probe'] > 0 and self.mask_loss_weight > 0:
                losses[f'risk_{f}'] = F.mse_loss(risk, target.float()) * (self.mask_loss_weight * sched['probe'])

            part_bad = torch.sigmoid(self.part_heads[f](z_probe))
            factor_w = float(self.factor_weights.get(f, 1.0))
            bad_parts = bad_parts + factor_w * risk.unsqueeze(-1) * part_bad

        if sched['rel'] <= 0:
            reliability = visual_global.new_ones(B, 1, 1, self.parts_num)
        else:
            denom = max(1.0, sum(abs(self.factor_weights.get(f, 1.0)) for f in self.factors))
            bad_parts = (bad_parts / denom).clamp(0, 1)
            if self.detach_reliability_gate:
                bad_parts = bad_parts.detach()
            reliability = 1.0 - self.max_suppression * sched['rel'] * bad_parts
            reliability = reliability.clamp(min=1.0 - self.max_suppression, max=1.0).view(B, 1, 1, self.parts_num)

        diag = {
            'sched_probe': visual_global.new_tensor(sched['probe']),
            'sched_adv': visual_global.new_tensor(sched['adv']),
            'sched_rel': visual_global.new_tensor(sched['rel']),
            'mean_reliability': reliability.mean().detach(),
        }
        return reliability, losses, diag

    def post_temporal(self, id_global, text_feats=None, masks=None, step: int = 0):
        sched = self.schedule(step)
        losses = {}
        if sched['adv'] <= 0 or self.adv_loss_weight <= 0:
            return losses
        lambd = self.grl_lambda * sched['adv']
        id_rev = grad_reverse(id_global.float(), lambd)
        for f in self.factors:
            z_txt = self._text_z(text_feats, f)
            if z_txt is None:
                continue
            target = self._target_mask(masks, f)
            z_id = l2n(self.visual_probe[f](id_rev), dim=-1)
            losses[f'adv_{f}'] = masked_symmetric_infonce(
                z_id, z_txt, mask=target, temperature=self.temperature
            ) * (self.adv_loss_weight * sched['adv'])
        return losses


# ============================================================
# Backbones inherited from GaitOracle design
# ============================================================

class TinyDenseRGBEncoder(nn.Module):
    """Dependency-free dense RGB encoder; replace with timm if needed."""

    def __init__(self, out_channels=512, in_channels=3):
        super().__init__()
        c1 = max(out_channels // 8, 32)
        c2 = max(out_channels // 4, 64)
        c3 = max(out_channels // 2, 128)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B * T, C, H, W))
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


class TimmDenseRGBEncoder(nn.Module):
    """Optional dense encoder via timm features_only models."""

    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', out_channels=512, pretrained=True, freeze=True):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise ImportError("Install timm or set rgb_backbone.type='tiny'.") from exc
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        in_c = self.backbone.feature_info.channels()[-1]
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.out_channels = out_channels

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.backbone(x.reshape(B * T, C, H, W))[-1]
        y = self.proj(y)
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


class PointCloudAnchorEncoder(nn.Module):
    """Simple z-part point-cloud encoder, output [B,T,C,P]."""

    def __init__(self, out_channels=512, parts_num=16, in_dim=3):
        super().__init__()
        self.parts_num = int(parts_num)
        self.out_channels = int(out_channels)
        self.in_dim = int(in_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_channels)

    def _fit_dim(self, pc):
        d = pc.size(-1)
        if d == self.in_dim:
            return pc
        if d > self.in_dim:
            return pc[..., :self.in_dim]
        pad = pc.new_zeros(*pc.shape[:-1], self.in_dim - d)
        return torch.cat([pc, pad], dim=-1)

    def forward(self, pc):
        # pc: [B,T,N,C]
        B, T, N, _ = pc.shape
        pc = self._fit_dim(pc.float())
        xyz = pc[..., :min(3, self.in_dim)]
        z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
        z_min = z.min(dim=-1, keepdim=True).values
        z_max = z.max(dim=-1, keepdim=True).values
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)
        feat = self.point_mlp(pc.reshape(B * T * N, self.in_dim)).reshape(B, T, N, self.out_channels)
        feat = self.norm(feat)

        part_feats = []
        for p in range(self.parts_num):
            lo = p / self.parts_num
            hi = (p + 1) / self.parts_num
            if p == self.parts_num - 1:
                mask = ((z_norm >= lo) & (z_norm <= hi)).float().unsqueeze(-1)
            else:
                mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
            denom = mask.sum(dim=2).clamp_min(1.0)
            pooled = (feat * mask).sum(dim=2) / denom
            part_feats.append(pooled)
        return torch.stack(part_feats, dim=-1)


# ============================================================
# Condition-quality encoder
# ============================================================

class ConditionQualityEncoder(nn.Module):
    """Supervised condition and modality-quality encoder from raw RGB/PC statistics.

    It outputs:
      cond_emb:        [B,Dc]
      quality_pair:    [B,2], softmax probabilities for RGB/PC reliability
      stat_quality:    [B,2], detached pseudo target from handcrafted statistics
      condition heads: optional risk predictions from nuisance masks/meta labels
    """

    def __init__(self, parts_num=16, stats_dim=20, cond_dim=128, hidden_dim=256,
                 dropout=0.1, norm_momentum=0.01,
                 stat_quality_weight=0.02, oracle_quality_weight=0.05,
                 condition_loss_weight=0.05, branch_identity_weight=0.15,
                 oracle_temperature=1.0, use_condition_heads=True):
        super().__init__()
        self.parts_num = int(parts_num)
        self.stats_dim = int(stats_dim)
        self.cond_dim = int(cond_dim)
        self.stat_quality_weight = float(stat_quality_weight)
        self.oracle_quality_weight = float(oracle_quality_weight)
        self.condition_loss_weight = float(condition_loss_weight)
        self.branch_identity_weight = float(branch_identity_weight)
        self.oracle_temperature = float(oracle_temperature)
        self.use_condition_heads = bool(use_condition_heads)

        self.norm = RunningNorm(stats_dim, momentum=norm_momentum)
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cond = nn.Linear(hidden_dim, cond_dim)
        self.quality_head = nn.Linear(hidden_dim, 2)  # RGB vs PC reliability
        if self.use_condition_heads:
            self.condition_heads = nn.ModuleDict({
                'occlusion': nn.Linear(hidden_dim, 1),
                'carrying': nn.Linear(hidden_dim, 1),
                'clothing': nn.Linear(hidden_dim, 1),
                'background': nn.Linear(hidden_dim, 1),
                'day_night': nn.Linear(hidden_dim, 2),
                'near_far': nn.Linear(hidden_dim, 2),
            })
        else:
            self.condition_heads = nn.ModuleDict()

    @staticmethod
    def _batch_minmax(x, eps=1e-6):
        xmin = x.min(dim=0, keepdim=True).values
        xmax = x.max(dim=0, keepdim=True).values
        return ((x - xmin) / (xmax - xmin + eps)).clamp(0, 1)

    def _rgb_stats(self, rgb: torch.Tensor):
        # rgb: [B,T,3,H,W], possibly already normalized. Use distributional stats, not absolute camera calibration.
        x = rgb.float()
        B, T, C, H, W = x.shape
        if C > 3:
            x = x[:, :, :3]
        if C == 1:
            x = x.repeat(1, 1, 3, 1, 1)
        gray = x.mean(dim=2)  # [B,T,H,W]
        lum_mean = gray.mean(dim=(1, 2, 3))
        lum_std = gray.std(dim=(1, 2, 3), unbiased=False)

        # Relative low/high exposure ratios. Thresholds are intentionally soft for normalized inputs.
        g_center = gray - gray.mean(dim=(2, 3), keepdim=True)
        low_ratio = (g_center < -1.0 * g_center.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)).float().mean(dim=(1, 2, 3))
        high_ratio = (g_center > 1.0 * g_center.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)).float().mean(dim=(1, 2, 3))

        dh = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs().mean(dim=(1, 2, 3)) if H > 1 else gray.new_zeros(B)
        dw = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)) if W > 1 else gray.new_zeros(B)
        edge = 0.5 * (dh + dw)

        # Laplacian-like sharpness proxy.
        if H > 2 and W > 2:
            center = gray[:, :, 1:-1, 1:-1]
            lap = -4 * center + gray[:, :, :-2, 1:-1] + gray[:, :, 2:, 1:-1] + gray[:, :, 1:-1, :-2] + gray[:, :, 1:-1, 2:]
            blur_proxy = lap.abs().mean(dim=(1, 2, 3))
        else:
            blur_proxy = edge

        if T > 1:
            temporal_diff = (gray[:, 1:] - gray[:, :-1]).abs().mean(dim=(1, 2, 3))
        else:
            temporal_diff = gray.new_zeros(B)

        channel_std = x.std(dim=2, unbiased=False).mean(dim=(1, 2, 3))
        rg = (x[:, :, 0] - x[:, :, 1]).std(dim=(1, 2, 3), unbiased=False)
        yb = (0.5 * (x[:, :, 0] + x[:, :, 1]) - x[:, :, 2]).std(dim=(1, 2, 3), unbiased=False)
        colorfulness = torch.sqrt(rg * rg + yb * yb + 1e-6)

        return torch.stack([
            lum_mean, lum_std, low_ratio, high_ratio, dh, dw, edge, blur_proxy, temporal_diff, channel_std + colorfulness,
        ], dim=-1)

    def _pc_stats(self, pc: Optional[torch.Tensor]):
        if pc is None:
            raise ValueError('CQGaitMoE requires point cloud input for ConditionQualityEncoder.')
        x = pc.float()
        B, T, N, D = x.shape
        xyz = x[..., :min(3, D)]
        if xyz.size(-1) < 3:
            pad = xyz.new_zeros(B, T, N, 3 - xyz.size(-1))
            xyz = torch.cat([xyz, pad], dim=-1)
        finite = torch.isfinite(xyz).all(dim=-1)
        nonzero = xyz.abs().sum(dim=-1) > 1e-8
        valid = (finite & nonzero).float()
        denom = valid.sum(dim=(1, 2)).clamp_min(1.0)
        valid_ratio = valid.mean(dim=(1, 2))
        point_count_ratio = valid.sum(dim=(1, 2)) / float(max(T * N, 1))

        xyz_clean = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
        radius = xyz_clean.norm(dim=-1)
        mean_range = (radius * valid).sum(dim=(1, 2)) / denom
        range_var = (((radius - mean_range.view(B, 1, 1)) ** 2) * valid).sum(dim=(1, 2)) / denom
        range_std = torch.sqrt(range_var.clamp_min(1e-8))

        xyz_mean = (xyz_clean * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denom.unsqueeze(-1)
        xyz_var = (((xyz_clean - xyz_mean.view(B, 1, 1, 3)) ** 2) * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denom.unsqueeze(-1)
        xyz_spread = torch.sqrt(xyz_var.clamp_min(1e-8))  # [B,3]
        height_coverage = xyz_spread[:, 2]

        # Per-part density over normalized z.
        z = xyz_clean[..., 2]
        z_min = z.masked_fill(valid <= 0, float('inf')).amin(dim=2, keepdim=True)
        z_max = z.masked_fill(valid <= 0, float('-inf')).amax(dim=2, keepdim=True)
        z_min = torch.where(torch.isfinite(z_min), z_min, z.new_zeros(z_min.shape))
        z_max = torch.where(torch.isfinite(z_max), z_max, z_min + 1.0)
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)
        densities = []
        for p in range(self.parts_num):
            lo = p / self.parts_num
            hi = (p + 1) / self.parts_num
            if p == self.parts_num - 1:
                m = ((z_norm >= lo) & (z_norm <= hi)).float() * valid
            else:
                m = ((z_norm >= lo) & (z_norm < hi)).float() * valid
            densities.append(m.mean(dim=2))  # [B,T]
        dens = torch.stack(densities, dim=-1)  # [B,T,P]
        dens_mean = dens.mean(dim=(1, 2))
        dens_std = dens.std(dim=(1, 2), unbiased=False)
        part_prob = dens.mean(dim=1)
        part_prob = part_prob / part_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(part_prob * (part_prob + 1e-8).log()).sum(dim=-1) / math.log(max(self.parts_num, 2))
        missing_ratio = (dens.mean(dim=1) <= 1.0 / max(N * 4, 1)).float().mean(dim=-1)

        return torch.stack([
            valid_ratio, point_count_ratio, mean_range, range_std,
            xyz_spread[:, 0], xyz_spread[:, 1], xyz_spread[:, 2],
            height_coverage, entropy, missing_ratio + dens_std + dens_mean,
        ], dim=-1)

    def extract_stats(self, rgb: torch.Tensor, pc: torch.Tensor):
        stats = torch.cat([self._rgb_stats(rgb), self._pc_stats(pc)], dim=-1)
        if stats.size(-1) != self.stats_dim:
            raise RuntimeError(f'Expected stats_dim={self.stats_dim}, got {stats.size(-1)}')
        return stats.detach()

    def stat_quality_target(self, stats: torch.Tensor):
        # stats: [B,20]. Build relative quality pseudo labels from current mini-batch.
        # RGB quality: contrast, edge, sharpness, temporal cue, less extreme exposure.
        rgb_good = torch.stack([
            self._batch_minmax(stats[:, 1]),      # contrast/luminance std
            self._batch_minmax(stats[:, 6]),      # edge
            self._batch_minmax(stats[:, 7]),      # sharpness
            self._batch_minmax(stats[:, 8]),      # temporal difference
            1.0 - self._batch_minmax(stats[:, 2] + stats[:, 3]),
        ], dim=-1).mean(dim=-1)

        # PC quality: valid points, reasonable density entropy, height coverage, lower missing ratio.
        pc_good = torch.stack([
            stats[:, 10].clamp(0, 1),
            stats[:, 11].clamp(0, 1),
            self._batch_minmax(stats[:, 16]),
            stats[:, 18].clamp(0, 1),
            1.0 - stats[:, 19].clamp(0, 1),
        ], dim=-1).mean(dim=-1)

        q = torch.stack([rgb_good, pc_good], dim=-1).clamp_min(1e-4)
        return (q / q.sum(dim=-1, keepdim=True).clamp_min(1e-6)).detach()

    def forward(self, rgb: torch.Tensor, pc: torch.Tensor, update_norm: bool = False):
        stats = self.extract_stats(rgb, pc)
        stat_quality = self.stat_quality_target(stats)
        x = self.norm(stats, update=update_norm)
        h = self.net(x)
        cond_emb = self.cond(h)
        quality_pair = torch.softmax(self.quality_head(h), dim=-1)
        out = {
            'stats': stats,
            'stat_quality_pair': stat_quality,
            'cond_emb': cond_emb,
            'quality_pair': quality_pair,
        }
        for k, head in self.condition_heads.items():
            out[f'{k}_logits'] = head(h)
        return out

    def supervision_losses(self, cqe_out: Dict[str, torch.Tensor], labels: Optional[torch.Tensor] = None,
                           masks: Optional[Dict[str, torch.Tensor]] = None,
                           rgb_logits: Optional[torch.Tensor] = None,
                           pc_logits: Optional[torch.Tensor] = None,
                           meta_labels: Optional[Dict[str, torch.Tensor]] = None):
        losses = {}
        q = cqe_out['quality_pair'].float().clamp_min(1e-8)
        q_stat = cqe_out['stat_quality_pair'].float().clamp_min(1e-8)

        if self.stat_quality_weight > 0:
            losses['cqe_stat_quality'] = F.smooth_l1_loss(q, q_stat) * self.stat_quality_weight

        if labels is not None and rgb_logits is not None and pc_logits is not None:
            labels = labels.long().view(-1)
            if self.branch_identity_weight > 0:
                losses['rgb_branch_ce'] = F.cross_entropy(rgb_logits.float(), labels) * self.branch_identity_weight
                losses['pc_branch_ce'] = F.cross_entropy(pc_logits.float(), labels) * self.branch_identity_weight
            if self.oracle_quality_weight > 0:
                rgb_ce = F.cross_entropy(rgb_logits.float(), labels, reduction='none')
                pc_ce = F.cross_entropy(pc_logits.float(), labels, reduction='none')
                oracle = torch.softmax(-torch.stack([rgb_ce, pc_ce], dim=-1) / max(self.oracle_temperature, 1e-6), dim=-1).detach()
                losses['cqe_oracle_quality'] = F.kl_div(q.log(), oracle, reduction='batchmean') * self.oracle_quality_weight

        if self.condition_loss_weight > 0 and self.use_condition_heads:
            # Text masks supervise risk heads during training. Text is not used by router/inference.
            if masks is not None:
                for factor in ['occlusion', 'carrying', 'clothing', 'background']:
                    key = f'{factor}_logits'
                    if key in cqe_out and factor in masks and masks[factor] is not None:
                        tgt = masks[factor].float()
                        if tgt.dim() > 1:
                            tgt = tgt.view(tgt.size(0), -1).mean(-1)
                        losses[f'cqe_{factor}_risk'] = F.binary_cross_entropy_with_logits(
                            cqe_out[key].view(-1).float(), tgt.clamp(0, 1).float()
                        ) * self.condition_loss_weight
            # Optional parsed meta labels.
            if meta_labels is not None:
                if 'day_night' in meta_labels and 'day_night_logits' in cqe_out:
                    losses['cqe_day_night'] = F.cross_entropy(cqe_out['day_night_logits'].float(), meta_labels['day_night'].long()) * self.condition_loss_weight
                if 'near_far' in meta_labels and 'near_far_logits' in cqe_out:
                    losses['cqe_near_far'] = F.cross_entropy(cqe_out['near_far_logits'].float(), meta_labels['near_far'].long()) * self.condition_loss_weight
        return losses


# ============================================================
# Part-MoE blocks
# ============================================================

class PartSelfAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, ffn_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = int(ffn_dim or channels * 2)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [N,C,P]
        t = x.permute(0, 2, 1).contiguous()
        q = self.ln1(t)
        y, _ = self.attn(q, q, q, need_weights=False)
        t = t + y
        t = t + self.ffn(self.ln2(t))
        return t.permute(0, 2, 1).contiguous()


class PartExpertBlock(nn.Module):
    """Generic per-modality part expert. Input/output: [B,T,C,P]."""

    def __init__(self, channels=512, heads=4, dropout=0.05, use_film=False, cond_dim=128, groups=1):
        super().__init__()
        self.use_film = bool(use_film)
        g = max(1, min(int(groups), int(channels)))
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, 1, groups=g, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.attn = PartSelfAttention(channels, heads=heads, ffn_dim=channels * 2, dropout=dropout)
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )
        if self.use_film:
            self.film = nn.Sequential(nn.Linear(cond_dim, channels * 2), nn.Tanh())

    def forward(self, x, cond_emb=None):
        B, T, C, P = x.shape
        y = x.reshape(B * T, C, P)
        if self.use_film and cond_emb is not None:
            gb = self.film(cond_emb).view(B, 2, C, 1)
            gamma, beta = gb[:, 0], gb[:, 1]
            y = y.view(B, T, C, P)
            y = y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
            y = y.reshape(B * T, C, P)
        res = y
        y = self.pre(y)
        y = self.attn(y)
        y = self.out(y) + res
        return y.reshape(B, T, C, P)


class TopKPartRouter(nn.Module):
    """Part-aware router, output [B,P,E]."""

    def __init__(self, channels=512, cond_dim=128, num_experts=8, hidden_dim=256,
                 dropout=0.05, quality_dim=2, temperature=1.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        in_dim = channels + cond_dim + quality_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def forward(self, part_summary, cond_emb=None, quality=None):
        # part_summary: [B,P,C]
        B, P, C = part_summary.shape
        xs = [part_summary]
        if cond_emb is not None:
            xs.append(cond_emb.unsqueeze(1).expand(B, P, -1))
        if quality is not None:
            if quality.dim() == 2:
                q = quality.unsqueeze(1).expand(B, P, -1)
            else:
                q = quality
            xs.append(q)
        x = torch.cat(xs, dim=-1)
        logits = self.net(x) / max(self.temperature, 1e-6)
        prob = torch.softmax(logits, dim=-1)
        return logits, prob


class TopKPartMoE(nn.Module):
    """Part-aware top-k MoE for RGB appearance or PC structure branch."""

    def __init__(self, channels=512, parts_num=16, cond_dim=128, num_experts=8, top_k=2,
                 expert_type='appearance', heads=4, dropout=0.05, router_hidden=256,
                 router_temperature=1.0, balance_weight=0.01, load_weight=0.01,
                 diversity_weight=0.002, z_loss_weight=1e-4):
        super().__init__()
        self.channels = int(channels)
        self.parts_num = int(parts_num)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.balance_weight = float(balance_weight)
        self.load_weight = float(load_weight)
        self.diversity_weight = float(diversity_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.expert_type = str(expert_type)

        self.router = TopKPartRouter(
            channels=channels, cond_dim=cond_dim, num_experts=num_experts,
            hidden_dim=router_hidden, dropout=dropout, quality_dim=2, temperature=router_temperature,
        )
        use_film = self.expert_type.lower() in ('structure', 'pc')
        groups = 4 if use_film else 1
        self.experts = nn.ModuleList([
            PartExpertBlock(channels, heads=heads, dropout=dropout, use_film=use_film, cond_dim=cond_dim, groups=groups)
            for _ in range(num_experts)
        ])
        self.out_norm = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def _aux_loss(self, logits, prob, topk_idx, expert_out):
        losses = {}
        B, P, E = prob.shape
        if self.balance_weight > 0:
            importance = prob.mean(dim=(0, 1))
            losses['importance'] = _cv_squared(importance) * self.balance_weight
        if self.load_weight > 0:
            onehot = F.one_hot(topk_idx, num_classes=E).float().sum(dim=2).clamp(0, 1)  # [B,P,E]
            load = onehot.mean(dim=(0, 1))
            losses['load'] = _cv_squared(load) * self.load_weight
        if self.z_loss_weight > 0:
            losses['z_loss'] = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * self.z_loss_weight
        if self.diversity_weight > 0 and E > 1:
            # expert_out: [B,T,C,P,E]. Summarize each expert, penalize off-diagonal cosine similarity.
            z = expert_out.float().mean(dim=(1, 3))  # [B,C,E]
            z = F.normalize(z.permute(0, 2, 1), dim=-1)  # [B,E,C]
            gram = torch.matmul(z, z.transpose(1, 2))
            off = gram - torch.eye(E, device=gram.device, dtype=gram.dtype).unsqueeze(0)
            losses['diversity'] = off.pow(2).mean() * self.diversity_weight
        total = sum(losses.values()) if losses else logits.new_tensor(0.0)
        diag = {
            'importance_entropy': (-(prob.mean(dim=(0, 1)) * (prob.mean(dim=(0, 1)) + 1e-8).log()).sum()).detach(),
            'dominant_expert': prob.mean(dim=(0, 1)).argmax().detach().float(),
            'aux_loss': total.detach(),
        }
        return total, losses, diag

    def forward(self, x, cond_emb=None, quality_pair=None):
        # x: [B,T,C,P]
        B, T, C, P = x.shape
        part_summary = x.mean(dim=1).permute(0, 2, 1).contiguous()  # [B,P,C]
        logits, prob = self.router(part_summary, cond_emb=cond_emb, quality=quality_pair)
        k = min(self.top_k, self.num_experts)
        topk_val, topk_idx = torch.topk(prob, k=k, dim=-1)
        topk_weight = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        combine = torch.zeros_like(prob)
        combine.scatter_add_(-1, topk_idx, topk_weight)

        outs = []
        for expert in self.experts:
            outs.append(expert(x, cond_emb=cond_emb))
        expert_out = torch.stack(outs, dim=-1)  # [B,T,C,P,E]
        y = (expert_out * combine[:, None, None, :, :]).sum(dim=-1)
        y = self.out_norm(y.reshape(B * T, C, P)).reshape(B, T, C, P)
        aux_total, aux_losses, diag = self._aux_loss(logits, prob, topk_idx, expert_out)
        return y, {
            'loss': aux_total,
            'losses': aux_losses,
            'router_logits': logits,
            'router_prob': prob,
            'topk_idx': topk_idx,
            'topk_weight': topk_weight,
            'diag': diag,
        }


class PartSequenceProjector(nn.Module):
    def __init__(self, channels=512, out_channels=512, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # [B,T,C,P]
        B, T, C, P = x.shape
        y = self.proj(x.reshape(B * T, C, P))
        C2 = y.size(1)
        return y.reshape(B, T, C2, P)


# ============================================================
# Token-attentive fusion Part-MoE
# ============================================================

class TokenAttentionFusionExpert(nn.Module):
    """One fusion expert: hybrid part query attends to RGB+PC part tokens."""

    def __init__(self, channels=512, parts_num=16, cond_dim=128, heads=4, dropout=0.05):
        super().__init__()
        self.channels = int(channels)
        self.parts_num = int(parts_num)
        self.part_query = nn.Parameter(torch.randn(parts_num, channels) * 0.02)
        dyn_in = channels * 3 + cond_dim
        self.dynamic_query = nn.Sequential(
            nn.Linear(dyn_in, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.q_norm = nn.LayerNorm(channels)
        self.kv_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, rgb_tokens, pc_tokens, cond_emb):
        # rgb/pc: [B,T,C,P]
        B, T, C, P = rgb_tokens.shape
        rgb = rgb_tokens.permute(0, 1, 3, 2).contiguous()  # [B,T,P,C]
        pc = pc_tokens.permute(0, 1, 3, 2).contiguous()
        cond = cond_emb.view(B, 1, 1, -1).expand(B, T, P, -1)
        dyn = self.dynamic_query(torch.cat([rgb, pc, (rgb - pc).abs(), cond], dim=-1))
        q = dyn + self.part_query.view(1, 1, P, C)
        kv = torch.cat([rgb, pc], dim=2)  # [B,T,2P,C]

        q_flat = q.reshape(B * T, P, C)
        kv_flat = kv.reshape(B * T, 2 * P, C)
        attn_out, attn_w = self.attn(
            self.q_norm(q_flat), self.kv_norm(kv_flat), self.kv_norm(kv_flat),
            need_weights=True, average_attn_weights=False,
        )
        y = q_flat + attn_out
        y = y + self.ffn(y)
        y = self.out_norm(y).reshape(B, T, P, C).permute(0, 1, 3, 2).contiguous()

        # attn_w: [B*T, heads, P, 2P]. Summarize attention mass to RGB/PC tokens.
        if attn_w is not None:
            aw = attn_w.float().reshape(B, T, attn_w.size(1), P, 2 * P).mean(dim=(1, 2))  # [B,P,2P]
            rgb_mass = aw[..., :P].sum(dim=-1)
            pc_mass = aw[..., P:].sum(dim=-1)
            mass = torch.stack([rgb_mass, pc_mass], dim=-1)
            mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            mass = rgb_tokens.new_full((B, P, 2), 0.5)
        return y, mass


class FusionPartMoE(nn.Module):
    """Token-attentive Fusion Part-MoE with quality-aligned routing."""

    def __init__(self, channels=512, parts_num=16, cond_dim=128, num_experts=8, top_k=2,
                 heads=4, dropout=0.05, router_hidden=256, router_temperature=1.0,
                 balance_weight=0.02, load_weight=0.02, diversity_weight=0.003,
                 z_loss_weight=1e-4, router_quality_weight=0.01):
        super().__init__()
        self.channels = int(channels)
        self.parts_num = int(parts_num)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.balance_weight = float(balance_weight)
        self.load_weight = float(load_weight)
        self.diversity_weight = float(diversity_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.router_quality_weight = float(router_quality_weight)

        # Fusion router sees RGB, PC, difference, cond, and q_rgb/q_pc.
        in_dim = channels * 3 + cond_dim + 2
        self.router = nn.Sequential(
            nn.Linear(in_dim, router_hidden),
            nn.LayerNorm(router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts),
        )
        self.router_temperature = float(router_temperature)
        self.experts = nn.ModuleList([
            TokenAttentionFusionExpert(channels, parts_num, cond_dim, heads=heads, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def _route(self, rgb_tokens, pc_tokens, cond_emb, quality_pair):
        B, T, C, P = rgb_tokens.shape
        rgb_sum = rgb_tokens.mean(dim=1).permute(0, 2, 1).contiguous()  # [B,P,C]
        pc_sum = pc_tokens.mean(dim=1).permute(0, 2, 1).contiguous()
        diff = (rgb_sum - pc_sum).abs()
        cond = cond_emb.unsqueeze(1).expand(B, P, -1)
        q = quality_pair.unsqueeze(1).expand(B, P, -1)
        r_in = torch.cat([rgb_sum, pc_sum, diff, cond, q], dim=-1)
        logits = self.router(r_in) / max(self.router_temperature, 1e-6)
        prob = torch.softmax(logits, dim=-1)
        return logits, prob

    def _aux_loss(self, logits, prob, topk_idx, expert_out, modality_mass, quality_pair):
        losses = {}
        B, P, E = prob.shape
        if self.balance_weight > 0:
            importance = prob.mean(dim=(0, 1))
            losses['importance'] = _cv_squared(importance) * self.balance_weight
        if self.load_weight > 0:
            onehot = F.one_hot(topk_idx, num_classes=E).float().sum(dim=2).clamp(0, 1)
            load = onehot.mean(dim=(0, 1))
            losses['load'] = _cv_squared(load) * self.load_weight
        if self.z_loss_weight > 0:
            losses['z_loss'] = torch.logsumexp(logits.float(), dim=-1).pow(2).mean() * self.z_loss_weight
        if self.diversity_weight > 0 and E > 1:
            z = expert_out.float().mean(dim=(1, 3))  # [B,C,E]
            z = F.normalize(z.permute(0, 2, 1), dim=-1)
            gram = torch.matmul(z, z.transpose(1, 2))
            off = gram - torch.eye(E, device=gram.device, dtype=gram.dtype).unsqueeze(0)
            losses['diversity'] = off.pow(2).mean() * self.diversity_weight
        if self.router_quality_weight > 0 and modality_mass is not None and quality_pair is not None:
            m = modality_mass.float().mean(dim=1).clamp_min(1e-8)  # [B,2]
            m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            q = quality_pair.detach().float().clamp_min(1e-8)
            q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            losses['router_quality'] = F.kl_div(m.log(), q, reduction='batchmean') * self.router_quality_weight
        total = sum(losses.values()) if losses else logits.new_tensor(0.0)
        diag = {
            'modality_rgb_mass': modality_mass[..., 0].mean().detach() if modality_mass is not None else logits.new_tensor(0.0),
            'modality_pc_mass': modality_mass[..., 1].mean().detach() if modality_mass is not None else logits.new_tensor(0.0),
            'dominant_expert': prob.mean(dim=(0, 1)).argmax().detach().float(),
            'aux_loss': total.detach(),
        }
        return total, losses, diag

    def forward(self, rgb_tokens, pc_tokens, cond_emb, quality_pair):
        B, T, C, P = rgb_tokens.shape
        logits, prob = self._route(rgb_tokens, pc_tokens, cond_emb, quality_pair)
        k = min(self.top_k, self.num_experts)
        topk_val, topk_idx = torch.topk(prob, k=k, dim=-1)
        topk_weight = topk_val / topk_val.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        combine = torch.zeros_like(prob)
        combine.scatter_add_(-1, topk_idx, topk_weight)

        expert_outs = []
        expert_masses = []
        for expert in self.experts:
            y, mass = expert(rgb_tokens, pc_tokens, cond_emb)
            expert_outs.append(y)
            expert_masses.append(mass)
        expert_out = torch.stack(expert_outs, dim=-1)      # [B,T,C,P,E]
        expert_mass = torch.stack(expert_masses, dim=-2)   # [B,P,E,2]
        fused = (expert_out * combine[:, None, None, :, :]).sum(dim=-1)
        modality_mass = (expert_mass * combine[:, :, :, None]).sum(dim=-2)  # [B,P,2]
        modality_mass = modality_mass / modality_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)
        aux_total, aux_losses, diag = self._aux_loss(logits, prob, topk_idx, expert_out, modality_mass, quality_pair)
        return fused, {
            'loss': aux_total,
            'losses': aux_losses,
            'router_logits': logits,
            'router_prob': prob,
            'topk_idx': topk_idx,
            'topk_weight': topk_weight,
            'modality_mass': modality_mass,
            'diag': diag,
        }


class EarlyPartTokenFusion(nn.Module):
    """Fallback fusion if FusionPartMoE is disabled."""

    def __init__(self, channels=512, dropout=0.05):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1),
        )
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, rgb_part_seq, pc_part_seq=None):
        B, T, C, P = rgb_part_seq.shape
        if pc_part_seq is None:
            x = rgb_part_seq.reshape(B * T, C, P)
            return self.out(x).reshape(B, T, C, P), None
        pair = torch.cat([rgb_part_seq, pc_part_seq], dim=2).reshape(B * T, C * 2, P)
        gate = torch.sigmoid(self.gate(pair)).reshape(B, T, 1, P)
        fused = rgb_part_seq * (1.0 - gate) + pc_part_seq * gate
        fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)
        return fused, gate


# ============================================================
# Temporal encoder and verifier
# ============================================================

class CycleAwareTemporalEncoder(nn.Module):
    """Temporal encoder with frame, difference and low-frequency gait cycle branches."""

    def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1, future_pred=True):
        super().__init__()
        self.channels = int(channels)
        self.parts_num = int(parts_num)
        self.future_pred = bool(future_pred)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.attn = nn.Sequential(nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(), nn.Linear(max(channels // 4, 16), 1))
        if self.future_pred:
            self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, part_seq):
        # part_seq: [B,T,C,P]
        B, T, C, P = part_seq.shape
        x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
        x_tem = self.temporal(x)
        if T > 1:
            diff = x[:, 1:] - x[:, :-1]
            diff = F.pad(diff, (0, 0, 1, 0))
        else:
            diff = torch.zeros_like(x)
        diff = self.diff_proj(diff)
        if T > 1:
            freq = torch.fft.rfft(x.float(), dim=1).abs()
            freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
        else:
            freq = x.float().mean(1)
        freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)
        y = x_tem + diff + freq
        att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)
        pooled = (y * att.unsqueeze(-1)).sum(1)
        pooled = pooled.reshape(B, P, C).permute(0, 2, 1).contiguous()
        future_loss = pooled.new_tensor(0.0)
        if self.future_pred and T > 2:
            pred = self.future_head(y[:, :-1])
            target = x[:, 1:].detach()
            future_loss = F.mse_loss(pred, target)
        att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()
        return pooled, future_loss, att


class RuleBasedVerifier(nn.Module):
    def __init__(self, attention_weight=0.0):
        super().__init__()
        self.attention_weight = float(attention_weight)

    def forward(self, gait_attn, rgb_seq=None):
        if self.attention_weight <= 0 or gait_attn is None:
            return gait_attn.new_tensor(0.0)
        # Weak regularizer: avoid degenerate all-zero/all-one attention.
        mean = gait_attn.float().mean()
        entropy = -(gait_attn.float().clamp(1e-6, 1 - 1e-6) * gait_attn.float().clamp(1e-6, 1 - 1e-6).log()).mean()
        return ((mean - 0.5).abs() + 0.1 * entropy) * self.attention_weight


# ============================================================
# CQGaitMoE model
# ============================================================

class CQGaitMoE(BaseModel):
    """Condition-Quality Guided Part-MoE for RGB-PC gait recognition."""

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        C = int(model_cfg.get('channels', 512))
        P = int(model_cfg.get('parts_num', 16))
        self.channels = C
        self.parts_num = P
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))
        self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

        text_cfg = model_cfg.get('text_adapter', {})
        text_dim = int(text_cfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        text_code_dim = int(text_cfg.get('code_dim', 256))
        text_hidden_dim = int(text_cfg.get('hidden_dim', 1024))
        text_dropout = float(text_cfg.get('dropout', 0.1))

        rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
        if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
            self.rgb_encoder = TimmDenseRGBEncoder(
                model_name=rgb_backbone_cfg.get('model_name', 'vit_base_patch14_dinov2.lvd142m'),
                out_channels=C,
                pretrained=rgb_backbone_cfg.get('pretrained', True),
                freeze=rgb_backbone_cfg.get('freeze', True),
            )
        else:
            self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

        pos_cfg = model_cfg.get('positive_text', {})
        text_guidance_cfg = model_cfg.get('text_guidance', {})
        self.positive_gate = PositiveTextGate(
            text_dim=pos_cfg.get('text_dim', text_dim),
            channels=C,
            prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
            num_prompts=pos_cfg.get('num_prompts', 4),
            trainable_prompts=pos_cfg.get('trainable', False),
            text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
            text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
            dropout=pos_cfg.get('dropout', text_dropout),
            enabled=text_guidance_cfg.get('positive_gate_enabled', True),
        )

        self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3))
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        # Condition-quality encoder.
        cq_cfg = model_cfg.get('condition_quality', {})
        self.use_cqe = bool(cq_cfg.get('enabled', True))
        cond_dim = int(cq_cfg.get('cond_dim', 128))
        if self.use_cqe:
            self.cqe = ConditionQualityEncoder(
                parts_num=P,
                stats_dim=cq_cfg.get('stats_dim', 20),
                cond_dim=cond_dim,
                hidden_dim=cq_cfg.get('hidden_dim', 256),
                dropout=cq_cfg.get('dropout', 0.1),
                norm_momentum=cq_cfg.get('norm_momentum', 0.01),
                stat_quality_weight=cq_cfg.get('stat_quality_weight', 0.02),
                oracle_quality_weight=cq_cfg.get('oracle_quality_weight', 0.05),
                condition_loss_weight=cq_cfg.get('condition_loss_weight', 0.05),
                branch_identity_weight=cq_cfg.get('branch_identity_weight', 0.15),
                oracle_temperature=cq_cfg.get('oracle_temperature', 1.0),
                use_condition_heads=cq_cfg.get('use_condition_heads', True),
            )
        else:
            self.cqe = None
            cond_dim = 1

        # MoE modules.
        moe_cfg = model_cfg.get('moe', {})
        app_cfg = moe_cfg.get('appearance', {})
        str_cfg = moe_cfg.get('structure', {})
        fuse_cfg = moe_cfg.get('fusion', {})
        self.use_app_moe = bool(app_cfg.get('enabled', True))
        self.use_str_moe = bool(str_cfg.get('enabled', True))
        self.use_fusion_moe = bool(fuse_cfg.get('enabled', True))

        if self.use_app_moe:
            self.appearance_moe = TopKPartMoE(
                channels=C, parts_num=P, cond_dim=cond_dim,
                num_experts=app_cfg.get('num_experts', 8), top_k=app_cfg.get('top_k', 2),
                expert_type='appearance', heads=app_cfg.get('heads', 4), dropout=app_cfg.get('dropout', 0.05),
                router_hidden=app_cfg.get('router_hidden', 256), router_temperature=app_cfg.get('router_temperature', 1.0),
                balance_weight=app_cfg.get('balance_weight', 0.01), load_weight=app_cfg.get('load_weight', 0.01),
                diversity_weight=app_cfg.get('diversity_weight', 0.002), z_loss_weight=app_cfg.get('z_loss_weight', 1e-4),
            )
        else:
            self.appearance_moe = None

        if self.use_str_moe:
            self.structure_moe = TopKPartMoE(
                channels=C, parts_num=P, cond_dim=cond_dim,
                num_experts=str_cfg.get('num_experts', 8), top_k=str_cfg.get('top_k', 2),
                expert_type='structure', heads=str_cfg.get('heads', 4), dropout=str_cfg.get('dropout', 0.05),
                router_hidden=str_cfg.get('router_hidden', 256), router_temperature=str_cfg.get('router_temperature', 1.0),
                balance_weight=str_cfg.get('balance_weight', 0.01), load_weight=str_cfg.get('load_weight', 0.01),
                diversity_weight=str_cfg.get('diversity_weight', 0.002), z_loss_weight=str_cfg.get('z_loss_weight', 1e-4),
            )
        else:
            self.structure_moe = None

        proj_cfg = model_cfg.get('projector', {})
        self.rgb_projector = PartSequenceProjector(C, C, dropout=proj_cfg.get('dropout', 0.0))
        self.pc_projector = PartSequenceProjector(C, C, dropout=proj_cfg.get('dropout', 0.0))

        if self.use_fusion_moe:
            self.fusion_moe = FusionPartMoE(
                channels=C, parts_num=P, cond_dim=cond_dim,
                num_experts=fuse_cfg.get('num_experts', 8), top_k=fuse_cfg.get('top_k', 2),
                heads=fuse_cfg.get('heads', 4), dropout=fuse_cfg.get('dropout', 0.05),
                router_hidden=fuse_cfg.get('router_hidden', 256), router_temperature=fuse_cfg.get('router_temperature', 1.0),
                balance_weight=fuse_cfg.get('balance_weight', 0.02), load_weight=fuse_cfg.get('load_weight', 0.02),
                diversity_weight=fuse_cfg.get('diversity_weight', 0.003), z_loss_weight=fuse_cfg.get('z_loss_weight', 1e-4),
                router_quality_weight=fuse_cfg.get('router_quality_weight', 0.01),
            )
        else:
            early_cfg = model_cfg.get('early_fusion', {})
            self.fusion_moe = None
            self.early_fusion = EarlyPartTokenFusion(channels=C, dropout=early_cfg.get('dropout', 0.05))

        # Temporal.
        temporal_cfg = model_cfg.get('cycle_temporal', model_cfg.get('temporal', {}))
        self.use_temporal = bool(temporal_cfg.get('enabled', True))
        if self.use_temporal:
            self.temporal = CycleAwareTemporalEncoder(
                channels=C, parts_num=P,
                heads=temporal_cfg.get('heads', 4),
                layers=temporal_cfg.get('layers', 1),
                dropout=temporal_cfg.get('dropout', 0.1),
                future_pred=temporal_cfg.get('future_pred', True),
            )
        else:
            self.temporal = None
            self.temporal_fallback = temporal_cfg.get('fallback', 'max_pool')

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        class_num = int(model_cfg['SeparateBNNecks'].get('class_num'))
        self.rgb_aux_classifier = nn.Linear(C, class_num)
        self.pc_aux_classifier = nn.Linear(C, class_num)

        # Nuisance text disentanglement.
        nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
        self.use_nuisance = bool(nuis_cfg.get('enabled', False)) and bool(text_guidance_cfg.get('nuisance_detach_enabled', True))
        legacy_loss_w = float(nuis_cfg.get('loss_weight', 0.03))
        if self.use_nuisance:
            self.nuisance_removal = StagedNuisanceRemoval(
                in_dim=C, parts_num=P,
                text_dim=nuis_cfg.get('text_dim', text_dim),
                text_code_dim=nuis_cfg.get('text_code_dim', text_code_dim),
                text_hidden_dim=nuis_cfg.get('text_hidden_dim', text_hidden_dim),
                sub_dim=nuis_cfg.get('sub_dim', 256),
                factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
                grl_lambda=nuis_cfg.get('grl_lambda', 0.08),
                temperature=nuis_cfg.get('temperature', 0.07),
                warmup_iters=nuis_cfg.get('warmup_iters', 5000),
                probe_iters=nuis_cfg.get('probe_iters', 10000),
                adv_ramp_iters=nuis_cfg.get('adv_ramp_iters', 10000),
                reliability_start_iter=nuis_cfg.get('reliability_start_iter', 15000),
                reliability_ramp_iters=nuis_cfg.get('reliability_ramp_iters', 10000),
                probe_loss_weight=nuis_cfg.get('probe_loss_weight', legacy_loss_w),
                adv_loss_weight=nuis_cfg.get('adv_loss_weight', legacy_loss_w),
                mask_loss_weight=nuis_cfg.get('mask_loss_weight', legacy_loss_w * 0.5),
                max_suppression=nuis_cfg.get('max_suppression', 0.35),
                detach_reliability_gate=nuis_cfg.get('detach_reliability_gate', True),
                factor_weights=nuis_cfg.get('factor_weights', None),
                dropout=nuis_cfg.get('dropout', text_dropout),
            )
        else:
            self.nuisance_removal = None

        self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.05))
        self.future_weight = float(model_cfg.get('future_weight', 0.01))
        ver_cfg = model_cfg.get('verifier', {})
        self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))

    def _current_step(self):
        if self.training:
            self._forward_iter += 1
        for attr in ('iteration', 'iter', 'current_iter', 'global_step'):
            if hasattr(self, attr):
                try:
                    val = getattr(self, attr)
                    if torch.is_tensor(val):
                        return int(val.detach().item())
                    return int(val)
                except Exception:
                    pass
        return int(self._forward_iter.detach().item())

    @staticmethod
    def _sum_losses(losses: Dict[str, torch.Tensor], base: torch.Tensor):
        total = base.new_tensor(0.0)
        for v in losses.values():
            if v is not None:
                total = total + v
        return total

    def _to_rgb(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif x.dim() == 5:
            if x.shape[2] == 1:
                x = x.repeat(1, 1, 3, 1, 1)
            elif x.shape[-1] == 3 and x.shape[2] != 3:
                x = x.permute(0, 1, 4, 2, 3)
        return x.float()

    def _to_text_feat(self, x):
        if x is None:
            return None
        x = x.float()
        if x.dim() == 3:
            return x.mean(1)
        if x.dim() == 2:
            return x
        return x.view(x.size(0), -1)

    def _get_optional(self, ipts, idx):
        if idx is None:
            return None
        if isinstance(idx, str):
            if idx.lower() in ('none', 'null'):
                return None
            idx = int(idx)
        if idx < 0 or idx >= len(ipts):
            return None
        return ipts[idx]

    def _dense_to_part_seq(self, feat_map):
        B, T, C, H, W = feat_map.shape
        part = self.HPP(feat_map.reshape(B * T, C, H, W))
        return part.reshape(B, T, C, -1)

    def _collect_nuisance_inputs(self, ipts):
        text_feats, masks = {}, {}
        for k, idx in self.nuisance_indices.items():
            text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
        for k, idx in self.nuisance_mask_indices.items():
            masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
        return text_feats, masks

    def _parse_meta_condition_labels(self, typs, views, device):
        # Optional heuristic parser. Disabled by default unless labels are clearly present.
        # It is deliberately conservative: returns an empty dict when no reliable keywords exist.
        labels = {}
        if typs is None:
            return labels
        try:
            strings = [str(t).lower() for t in typs]
        except Exception:
            return labels
        if len(strings) == 0:
            return labels
        day = []
        dist = []
        has_day = False
        has_dist = False
        for s in strings:
            if any(k in s for k in ['night', 'dark', 'lowlight', 'low-light']):
                day.append(1); has_day = True
            elif any(k in s for k in ['day', 'light', 'normal']):
                day.append(0); has_day = True
            else:
                day.append(0)
            if any(k in s for k in ['far', 'remote', 'long']):
                dist.append(1); has_dist = True
            elif any(k in s for k in ['near', 'close', 'short']):
                dist.append(0); has_dist = True
            else:
                dist.append(0)
        if has_day:
            labels['day_night'] = torch.tensor(day, device=device, dtype=torch.long)
        if has_dist:
            labels['near_far'] = torch.tensor(dist, device=device, dtype=torch.long)
        return labels

    def _branch_logits(self, tokens, classifier):
        # tokens: [B,T,C,P] -> logits [B,num_classes]
        feat = tokens.mean(dim=(1, 3))
        return classifier(feat.float())

    def _temporal_forward(self, fused_seq):
        if self.use_temporal:
            return self.temporal(fused_seq)
        if getattr(self, 'temporal_fallback', 'max_pool') == 'mean_pool':
            pooled = fused_seq.mean(dim=1)
        else:
            pooled = fused_seq.max(dim=1).values
        return pooled, fused_seq.new_tensor(0.0), None

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._current_step()
        rgb = self._to_rgb(ipts[self.rgb_index])
        pc = self._get_optional(ipts, self.pc_index)
        if pc is None:
            raise ValueError('CQGaitMoE requires point-cloud input. Check pc_index/data_in_use.')
        pc = pc.float()

        # Raw condition/quality branch. It never consumes text as router input.
        if self.use_cqe:
            cqe_out = self.cqe(rgb, pc, update_norm=True)
            cond_emb = cqe_out['cond_emb']
            quality_pair = cqe_out['quality_pair']
        else:
            cond_emb = rgb.new_zeros(rgb.size(0), 1)
            quality_pair = rgb.new_full((rgb.size(0), 2), 0.5)
            cqe_out = {'cond_emb': cond_emb, 'quality_pair': quality_pair, 'stat_quality_pair': quality_pair.detach()}

        # RGB branch.
        rgb_map = self.rgb_encoder(rgb)
        rgb_guided, gait_attn = self.positive_gate(rgb_map)
        rgb_part_seq = self._dense_to_part_seq(rgb_guided)

        # PC branch.
        pc_part_seq = self.pc_encoder(pc)

        # Text nuisance supervision/gate is training-time auxiliary only.
        text_feats, nuisance_masks = ({}, {})
        nuisance_pre_losses, nuisance_post_losses, nuisance_diag = {}, {}, {}
        if self.use_nuisance:
            text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
            pre_global = rgb_part_seq.mean(dim=(1, 3))
            reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
                pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            rgb_part_seq = rgb_part_seq * reliability
        else:
            # Still collect masks for CQE risk supervision if indices exist.
            _, nuisance_masks = self._collect_nuisance_inputs(ipts)

        # Branch Part-MoE.
        aux_losses = {}
        moe_info = {}
        if self.use_app_moe:
            rgb_moe, app_info = self.appearance_moe(rgb_part_seq, cond_emb=cond_emb, quality_pair=quality_pair)
            aux_losses['moe_app'] = app_info['loss']
            moe_info['appearance'] = app_info
        else:
            rgb_moe = rgb_part_seq

        if self.use_str_moe:
            pc_moe, str_info = self.structure_moe(pc_part_seq, cond_emb=cond_emb, quality_pair=quality_pair)
            aux_losses['moe_structure'] = str_info['loss']
            moe_info['structure'] = str_info
        else:
            pc_moe = pc_part_seq

        rgb_proj = self.rgb_projector(rgb_moe)
        pc_proj = self.pc_projector(pc_moe)

        # Branch classifiers provide single-modality oracle supervision to CQE.
        rgb_aux_logits = self._branch_logits(rgb_proj, self.rgb_aux_classifier)
        pc_aux_logits = self._branch_logits(pc_proj, self.pc_aux_classifier)

        # Fusion.
        if self.use_fusion_moe:
            fused_seq, fuse_info = self.fusion_moe(rgb_proj, pc_proj, cond_emb=cond_emb, quality_pair=quality_pair)
            aux_losses['moe_fusion'] = fuse_info['loss']
            moe_info['fusion'] = fuse_info
            modality_gate = fuse_info.get('modality_mass', None)
        else:
            fused_seq, modality_gate = self.early_fusion(rgb_proj, pc_proj)

        # Temporal and head.
        fused_feat, future_loss, temporal_attn = self._temporal_forward(fused_seq)
        embed_1 = self.FCs(fused_feat)
        embed_2, logits = self.BNNecks(embed_1)

        if not self.training:
            return {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {'embeddings': embed_1},
            }

        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }

        # Core auxiliary losses.
        if self.rgb_anchor_weight > 0:
            rgb_g = rgb_proj.mean(dim=(1, 3))
            pc_g = pc_proj.mean(dim=(1, 3))
            aux_losses['rgb_pc_anchor'] = symmetric_infonce(rgb_g, pc_g) * self.rgb_anchor_weight
        if self.future_weight > 0:
            aux_losses['future_pred'] = future_loss * self.future_weight

        if self.use_cqe:
            meta_labels = self._parse_meta_condition_labels(typs, views, device=embed_1.device)
            cqe_losses = self.cqe.supervision_losses(
                cqe_out,
                labels=labs,
                masks=nuisance_masks,
                rgb_logits=rgb_aux_logits,
                pc_logits=pc_aux_logits,
                meta_labels=meta_labels,
            )
            aux_losses.update(cqe_losses)

        if self.use_nuisance:
            id_global = fused_feat.mean(-1)
            nuisance_post_losses = self.nuisance_removal.post_temporal(
                id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_pre_losses.items()})
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_post_losses.items()})

        attn_loss = self.verifier(gait_attn, rgb_seq=rgb)
        if torch.is_tensor(attn_loss) and (attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0):
            aux_losses['attention_verifier'] = attn_loss

        if self.use_scalar_aux_loss:
            aux_total = self._sum_losses(aux_losses, embed_1)
            training_feat['cqgait_aux'] = {'loss': aux_total}
        else:
            for k, v in aux_losses.items():
                training_feat[f'loss_{k}'] = {'loss': v}

        # Keep visual_summary light; no heavy visualizations in this first performance-focused version.
        visual_summary = {}
        if self.training:
            visual_summary['scalar/cqe_q_rgb'] = quality_pair[:, 0].mean().detach()
            visual_summary['scalar/cqe_q_pc'] = quality_pair[:, 1].mean().detach()
            if 'fusion' in moe_info:
                visual_summary['scalar/fusion_rgb_mass'] = moe_info['fusion']['diag']['modality_rgb_mass']
                visual_summary['scalar/fusion_pc_mass'] = moe_info['fusion']['diag']['modality_pc_mass']
            if self.use_nuisance:
                for k, v in nuisance_diag.items():
                    visual_summary[f'scalar/nuisance_{k}'] = v.detach()

        return {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embed_1},
        }
