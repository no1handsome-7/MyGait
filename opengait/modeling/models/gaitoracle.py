
import os
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# ============================================================
# Utilities
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
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def _scalar_like(x, value=0.0):
    return x.new_tensor(float(value))


def symmetric_infonce(z1, z2, temperature=0.07):
    if z1 is None or z2 is None or z1.size(0) <= 1 or z2.size(0) <= 1:
        base = z1 if z1 is not None else z2
        return _scalar_like(base, 0.0)
    z1 = l2n(z1.float(), dim=-1)
    z2 = l2n(z2.float(), dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
    if z_img is None or z_txt is None:
        base = z_img if z_img is not None else z_txt
        return _scalar_like(base, 0.0)

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
    logits = z_img @ z_txt.t() / temperature
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


# ============================================================
# Text compression: keep old Qwen 3584-dim features compatible
# ============================================================

class TextEmbeddingAdapter(nn.Module):
    """Compress high-dimensional MLLM text embeddings to a gait-sized code.

    Existing Qwen2.5-VL features such as 3584-dim prompt/nuisance embeddings can be
    used directly.  The adapter crops/pads unexpected dimensions, then learns a
    low-rank bottleneck.  No offline feature regeneration is required.
    """

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


# ============================================================
# Module A: RGB encoder
# ============================================================

class TinyDenseRGBEncoder(nn.Module):
    """Dependency-free dense RGB encoder; use timm backbone if available/needed."""

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
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B * T, C, H, W))
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


class TimmDenseRGBEncoder(nn.Module):
    """Optional dense foundation encoder via timm features_only models."""

    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', out_channels=512, pretrained=True, freeze=True):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        chs = self.backbone.feature_info.channels()
        in_c = chs[-1]
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


# ============================================================
# Module B1: positive gait prompt gate
# ============================================================

class PositiveTextGate(nn.Module):
    """Fixed gait-positive prompt bank -> channel/spatial gait gate.

    This preserves motion-biometric cues before early RGB-anchor fusion.  The
    prompt bank can still be the old Qwen .npy file; the high-dimensional text
    vectors are compressed by TextEmbeddingAdapter inside the module.
    """

    def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
                 trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.channels = channels

        if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
            arr = np.load(prompt_feat_path).astype('float32')
            if arr.ndim == 1:
                arr = arr[None, :]
            prompt = torch.from_numpy(arr)
            text_dim = int(prompt.shape[-1])
        else:
            prompt = torch.randn(num_prompts, text_dim) * 0.02

        self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
        self.text_adapter = TextEmbeddingAdapter(
            input_dim=text_dim,
            code_dim=text_code_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
            l2_norm=False,
        )
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_code_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        mid = max(channels // 4, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_map):
        # feat_map: [B, T, C, H, W]
        B, T, C, H, W = feat_map.shape
        prompt = self.prompt_feat.to(device=feat_map.device)
        pos_code = self.text_adapter(prompt).mean(0)  # [Dtxt]
        pos = self.text_to_channel(pos_code)          # [C]
        ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

        x = feat_map * (1.0 + ch_gate)
        sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
        return guided, sp_gate


# ============================================================
# Module B2: staged nuisance removal
# ============================================================

class StagedNuisanceRemoval(nn.Module):
    """Four-stage nuisance disentanglement/removal.

    Stage 0 warmup: no nuisance gradient.
    Stage 1 probe: learn visual nuisance probes and risk prediction using detached visual features.
    Stage 2 adversarial: gradually apply GRL to identity features to remove nuisance information.
    Stage 3 refinement: turn on a visual-only part reliability gate before temporal modeling.

    Text inputs are training supervision only; at inference the reliability gate uses only visual
    features predicted by the nuisance probes, so existing retrieval remains text-free.
    """

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
            k: nn.Sequential(nn.Linear(sub_dim, sub_dim // 2), nn.GELU(), nn.Linear(sub_dim // 2, 1))
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
        """Return visual-only part reliability and pre-temporal nuisance losses.

        visual_global: [B, C], computed from RGB part tokens before temporal modeling.
        """
        sched = self.schedule(step)
        B = visual_global.size(0)
        device = visual_global.device
        losses = {}
        bad_parts = visual_global.new_zeros(B, self.parts_num)

        for f in self.factors:
            # Probe branch learns to extract nuisance from detached visual features.
            z_probe = l2n(self.visual_probe[f](visual_global.detach().float()), dim=-1)
            target = self._target_mask(masks, f)
            z_txt = self._text_z(text_feats, f)

            if z_txt is not None and sched['probe'] > 0 and self.probe_loss_weight > 0:
                losses[f'probe_{f}'] = masked_symmetric_infonce(
                    z_probe, z_txt, mask=target, temperature=self.temperature
                ) * (self.probe_loss_weight * sched['probe'])

            risk = torch.sigmoid(self.risk_heads[f](z_probe)).squeeze(-1)  # [B]
            if target is not None and sched['probe'] > 0 and self.mask_loss_weight > 0:
                losses[f'risk_{f}'] = F.mse_loss(risk, target.float()) * (self.mask_loss_weight * sched['probe'])

            part_bad = torch.sigmoid(self.part_heads[f](z_probe))          # [B,P]
            factor_w = float(self.factor_weights.get(f, 1.0))
            bad_parts = bad_parts + factor_w * risk.unsqueeze(-1) * part_bad

        if sched['rel'] <= 0:
            reliability = visual_global.new_ones(B, 1, 1, self.parts_num)
        else:
            bad_parts = bad_parts / max(1.0, sum(abs(self.factor_weights.get(f, 1.0)) for f in self.factors))
            bad_parts = bad_parts.clamp(0, 1)
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


# Backward-compatible alias for old imports/checkpoints that refer to this name.
NuisanceDetachHead = StagedNuisanceRemoval


# ============================================================
# Module C: geometry / silhouette / point-cloud anchors
# ============================================================

class SilhouetteAnchorEncoder(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

    def forward(self, sil):
        # sil: [B,T,H,W] or [B,T,1,H,W]
        if sil.dim() == 4:
            sil = sil.unsqueeze(2)
        elif sil.dim() == 5 and sil.shape[2] != 1:
            sil = sil[:, :, :1]
        return self.net(sil.float())


class PointCloudSimpleEncoder(nn.Module):
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
        return torch.stack(part_feats, dim=-1)  # [B,T,C,P]
    
class LidarGaitPPEncoder(nn.Module):
    def __init__(self, channel=16, npoints=None, nsample=8, sampling='knn',
                 scale_aware=True, normalize_dp=True, out_channels=256, parts_num=16):
        super().__init__()
        from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling
        npoints = npoints or [256, 128, 64]
        ic = 4 if scale_aware else 3
        C = channel
        self.sa1 = PointNetSetAbstraction(npoints[0], 0.1, nsample, ic, [2*C, 2*C, 4*C], False, sampling, scale_aware, normalize_dp)
        self.sa2 = PointNetSetAbstraction(npoints[1], 0.2, nsample, 4*C+ic, [4*C, 4*C, 8*C], False, sampling, scale_aware, normalize_dp)
        self.sa3 = PointNetSetAbstraction(npoints[2], 0.4, nsample, 8*C+ic, [8*C, 8*C, 16*C], False, sampling, scale_aware, normalize_dp)
        self.sa4 = PointNetSetAbstraction(None, None, None, 16*C+ic, [16*C, 16*C, out_channels], True, sampling, scale_aware, normalize_dp)
        self.pool = PPPooling(scale_aware=scale_aware, bin_num=[parts_num])

    def forward(self, pc):
        B, T, N, C = pc.shape
        xyz = rearrange(pc, 'B T N C -> (B T) C N')
        l1x, l1p = self.sa1(xyz, None); l1p = l1p.max(-2)[0]
        l2x, l2p = self.sa2(l1x, l1p); l2p = l2p.max(-2)[0]
        l3x, l3p = self.sa3(l2x, l2p); l3p = l3p.max(-2)[0]
        l4x, l4p = self.sa4(l3x, l3p)
        x = self.pool(l4p, l3x)
        return rearrange(x, '(B T) C P -> B T C P', B=B)


# ============================================================
# Module D: early part-token fusion + unified temporal encoder
# ============================================================

class EarlyPartTokenFusion(nn.Module):
    """Fuse RGB and anchor part tokens before temporal modeling.

    This fixes the previous late-fusion design where each modality was temporally
    encoded independently before interaction.
    """

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

    def forward(self, rgb_part_seq, anchor_part_seq=None):
        # [B,T,C,P]
        B, T, C, P = rgb_part_seq.shape
        if anchor_part_seq is None:
            x = rgb_part_seq.reshape(B * T, C, P)
            return self.out(x).reshape(B, T, C, P), None
        if anchor_part_seq.size(-1) != P:
            raise ValueError(f'Anchor parts {anchor_part_seq.size(-1)} != RGB parts {P}')
        pair = torch.cat([rgb_part_seq, anchor_part_seq], dim=2).reshape(B * T, C * 2, P)
        gate = torch.sigmoid(self.gate(pair)).reshape(B, T, 1, P)
        fused = rgb_part_seq * (1.0 - gate) + anchor_part_seq * gate
        fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)
        return fused, gate


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



class MaxPoolTemporalEncoder(nn.Module):
    """Temporal ablation fallback: max-pool part tokens across frames.

    The interface matches CycleAwareTemporalEncoder so the downstream FC/BNNeck
    path does not change during ablations.
    """

    def forward(self, part_seq):
        # part_seq: [B,T,C,P]
        B, T, C, P = part_seq.shape
        pooled = part_seq.max(dim=1).values
        future_loss = pooled.new_tensor(0.0)
        temporal_attn = pooled.new_full((B, T, P), 1.0 / max(T, 1))
        return pooled, future_loss, temporal_attn


# ============================================================
# Module E: lightweight verifier regularization
# ============================================================

class RuleBasedVerifier(nn.Module):
    def __init__(self, attention_weight=0.0):
        super().__init__()
        self.attention_weight = float(attention_weight)

    def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
        if self.attention_weight <= 0 or gait_attn is None:
            return gait_attn.new_tensor(0.0)
        B, T, _, Hf, Wf = gait_attn.shape
        target = None
        if sil_seq is not None:
            sil = sil_seq.unsqueeze(2) if sil_seq.dim() == 4 else sil_seq[:, :, :1]
            target = F.interpolate(
                sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
                size=(Hf, Wf), mode='bilinear', align_corners=False,
            ).reshape(B, T, 1, Hf, Wf)
            target = (target > 0.05).float()
        elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
            gray = rgb_seq.float().mean(2, keepdim=True)
            diff = torch.zeros_like(gray)
            diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
            target = F.interpolate(
                diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
                size=(Hf, Wf), mode='bilinear', align_corners=False,
            ).reshape(B, T, 1, Hf, Wf)
            target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        if target is None:
            return gait_attn.new_tensor(0.0)
        # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
        if self.attention_weight <= 0:
            return gait_attn.new_tensor(0.0)

        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(
                gait_attn.float().clamp(1e-6, 1 - 1e-6),
                target.float()
            )

        return loss * self.attention_weight


# ============================================================
# Main OpenGait model
# ============================================================

class GaitOracle(BaseModel):
    """GaitOracle strong version.

    Key changes against the previous version:
      1) RGB/geometry anchor are fused at part-token level before temporal encoding.
      2) Qwen 3584-d text embeddings are compressed to gait-sized codes on the fly.
      3) Nuisance removal is staged: warmup -> probe -> adversarial -> reliability refinement.
      4) Existing positive prompt .npy and qwen25vl_*_feat.pkl / mask.pkl files remain compatible.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        C = model_cfg.get('channels', 512)
        P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
        self.channels = int(C)
        self.parts_num = int(P)
        self.rgb_index = model_cfg.get('rgb_index', 0)
        self.sil_index = model_cfg.get('sil_index', None)
        self.pc_index = model_cfg.get('pc_index', None)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

        # Centralized ablation switches.  Section-level `enabled` flags are still
        # supported for backward compatibility, but `module_switches` is easier
        # to edit when running module ablations.
        switch_cfg = model_cfg.get('module_switches', {})

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
        self.use_text_guidance = bool(switch_cfg.get(
            'text_guidance', pos_cfg.get('enabled', True)
        )) and bool(pos_cfg.get('enabled', True))
        self.positive_gate = PositiveTextGate(
            text_dim=pos_cfg.get('text_dim', text_dim),
            channels=C,
            prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
            num_prompts=pos_cfg.get('num_prompts', 4),
            trainable_prompts=pos_cfg.get('trainable', False),
            text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
            text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
            dropout=pos_cfg.get('dropout', text_dropout),
        ) if self.use_text_guidance else None

        self.use_sil_anchor = self.sil_index is not None
        self.use_pc_anchor = self.pc_index is not None
        self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
        # self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None
        # --- PC encoder ---
        pc_type = str(model_cfg.get('pc_encoder_type', 'simple'))
        pc_in_dim = int(model_cfg.get('pc_in_dim', 3))
        if pc_type == 'lidargaitpp':
            pc_cfg = model_cfg.get('pc_model_cfg', {})
            self.pc_encoder = LidarGaitPPEncoder(
                channel=pc_cfg.get('channel', 16),
                npoints=pc_cfg.get('npoints', [256, 128, 64]),
                nsample=pc_cfg.get('nsample', 8),
                sampling=pc_cfg.get('sampling', 'knn'),
                scale_aware=pc_cfg.get('scale_aware', True),
                normalize_dp=pc_cfg.get('normalize_dp', True),
                out_channels=C, parts_num=P,
            )
        else:
            self.pc_encoder = PointCloudSimpleEncoder(
                out_channels=C, parts_num=P, in_dim=pc_in_dim,
            )
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        early_cfg = model_cfg.get('early_fusion', {})
        self.early_fusion = EarlyPartTokenFusion(channels=C, dropout=early_cfg.get('dropout', 0.05))

        temporal_cfg = model_cfg.get('cycle_temporal', {})
        self.use_temporal = bool(switch_cfg.get(
            'temporal', temporal_cfg.get('enabled', True)
        )) and bool(temporal_cfg.get('enabled', True))
        if self.use_temporal:
            self.temporal = CycleAwareTemporalEncoder(
                channels=C,
                parts_num=P,
                heads=temporal_cfg.get('heads', 4),
                layers=temporal_cfg.get('layers', 1),
                dropout=temporal_cfg.get('dropout', 0.1),
                future_pred=temporal_cfg.get('future_pred', True),
            )
        else:
            self.temporal = MaxPoolTemporalEncoder()

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        pc_aux_cfg = model_cfg.get('pc_aux', {})
        self.use_pc_aux = bool(switch_cfg.get(
            'pc_aux', pc_aux_cfg.get('enabled', False)
        )) and bool(pc_aux_cfg.get('enabled', False))
        self.pc_aux_log_embedding_norm = bool(pc_aux_cfg.get('log_embedding_norm', True))
        if self.use_pc_aux:
            # PC-only auxiliary branch:
            #   point cloud -> part tokens -> temporal/maxpool -> separate FC/BNNeck -> Triplet + CE.
            # It gives the PC encoder direct identity supervision instead of relying only on
            # gradients coming through the RGB/PC fusion gate.
            pc_aux_temporal_cfg = dict(temporal_cfg)
            pc_aux_temporal_cfg.update(pc_aux_cfg.get('temporal', {}))
            self.pc_aux_use_temporal = bool(pc_aux_cfg.get('use_temporal', self.use_temporal))
            if self.pc_aux_use_temporal:
                self.pc_aux_temporal = CycleAwareTemporalEncoder(
                    channels=C,
                    parts_num=P,
                    heads=pc_aux_temporal_cfg.get('heads', 4),
                    layers=pc_aux_temporal_cfg.get('layers', 1),
                    dropout=pc_aux_temporal_cfg.get('dropout', 0.1),
                    # The auxiliary future-prediction loss is intentionally disabled;
                    # this branch is only for identity supervision.
                    future_pred=False,
                )
            else:
                self.pc_aux_temporal = MaxPoolTemporalEncoder()

            pc_aux_fc_cfg = dict(model_cfg['SeparateFCs'])
            pc_aux_fc_cfg.update(pc_aux_cfg.get('SeparateFCs', {}))
            pc_aux_bn_cfg = dict(model_cfg['SeparateBNNecks'])
            pc_aux_bn_cfg.update(pc_aux_cfg.get('SeparateBNNecks', {}))
            self.pc_aux_FCs = SeparateFCs(**pc_aux_fc_cfg)
            self.pc_aux_BNNecks = SeparateBNNecks(**pc_aux_bn_cfg)
        else:
            self.pc_aux_use_temporal = False
            self.pc_aux_temporal = None
            self.pc_aux_FCs = None
            self.pc_aux_BNNecks = None

        nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
        self.use_nuisance = bool(switch_cfg.get(
            'text_disentangle', nuis_cfg.get('enabled', False)
        )) and bool(nuis_cfg.get('enabled', False))
        legacy_loss_w = float(nuis_cfg.get('loss_weight', 0.03))
        if self.use_nuisance:
            self.nuisance_removal = StagedNuisanceRemoval(
                in_dim=C,
                parts_num=P,
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

        ver_cfg = model_cfg.get('verifier', {})
        self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))
        self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.future_weight = float(model_cfg.get('future_weight', 0.01))
        self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))

        gate_monitor_cfg = model_cfg.get('fusion_gate_monitor', {})
        self.monitor_fusion_gate = bool(switch_cfg.get(
            'fusion_gate_monitor', gate_monitor_cfg.get('enabled', True)
        )) and bool(gate_monitor_cfg.get('enabled', True))
        self.monitor_gate_parts = bool(gate_monitor_cfg.get('log_part_mean', False))

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

    def _sum_losses(self, losses: Dict[str, torch.Tensor], base: torch.Tensor):
        total = base.new_tensor(0.0)
        for v in losses.values():
            if v is not None:
                total = total + v
        return total

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._current_step()
        rgb = self._to_rgb(ipts[self.rgb_index])
        sil = self._get_optional(ipts, self.sil_index)
        pc = self._get_optional(ipts, self.pc_index)

        rgb_map = self.rgb_encoder(rgb)
        if self.use_text_guidance:
            rgb_guided, gait_attn = self.positive_gate(rgb_map)
        else:
            # Text-guidance ablation: use raw RGB feature maps and skip
            # attention/verifier losses tied to the text-guided gate.
            rgb_guided, gait_attn = rgb_map, None
        rgb_part_seq = self._dense_to_part_seq(rgb_guided)

        anchor_part_seq = None
        anchor_source = None
        sil_for_verifier = None
        if pc is not None and self.pc_encoder is not None:
            anchor_part_seq = self.pc_encoder(pc.float())
            anchor_source = 'pc'
        elif sil is not None and self.sil_encoder is not None:
            sil_for_verifier = sil
            sil_map = self.sil_encoder(sil.float())
            anchor_part_seq = self._dense_to_part_seq(sil_map)
            anchor_source = 'sil'

        nuisance_pre_losses = {}
        nuisance_post_losses = {}
        nuisance_diag = {}
        text_feats, nuisance_masks = ({}, {})
        if self.use_nuisance:
            text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
            pre_global = rgb_part_seq.mean(dim=(1, 3))
            reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
                pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            rgb_part_seq = rgb_part_seq * reliability

        # Early fusion happens before temporal encoding.
        fused_seq, modality_gate = self.early_fusion(rgb_part_seq, anchor_part_seq)
        fused_feat, future_loss, temporal_attn = self.temporal(fused_seq)

        embed_1 = self.FCs(fused_feat)
        embed_2, logits = self.BNNecks(embed_1)

        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }

        pc_aux_embed_1 = None
        if self.training and self.use_pc_aux and anchor_source == 'pc' and anchor_part_seq is not None:
            pc_aux_feat, _, _ = self.pc_aux_temporal(anchor_part_seq)
            pc_aux_embed_1 = self.pc_aux_FCs(pc_aux_feat)
            _, pc_aux_logits = self.pc_aux_BNNecks(pc_aux_embed_1)
            training_feat['pc_triplet'] = {'embeddings': pc_aux_embed_1, 'labels': labs}
            training_feat['pc_softmax'] = {'logits': pc_aux_logits, 'labels': labs}


        aux_losses = {}
        if anchor_part_seq is not None and self.rgb_anchor_weight > 0:
            rgb_g = rgb_part_seq.mean(dim=(1, 3))
            anc_g = anchor_part_seq.mean(dim=(1, 3))
            aux_losses['rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

        if self.use_temporal and self.future_weight > 0:
            aux_losses['future_pred'] = future_loss * self.future_weight

        if self.use_nuisance:
            id_global = fused_feat.mean(-1)
            nuisance_post_losses = self.nuisance_removal.post_temporal(
                id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_pre_losses.items()})
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_post_losses.items()})

        if gait_attn is not None:
            attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
            if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
                aux_losses['attention_verifier'] = attn_loss

        if self.use_scalar_aux_loss:
            aux_total = self._sum_losses(aux_losses, embed_1)
            training_feat['gaitoracle_aux'] = {'loss': aux_total}
        else:
            for k, v in aux_losses.items():
                training_feat[f'loss_{k}'] = v

        visual_summary = {
            'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
        }
        if gait_attn is not None:
            visual_summary['image/gait_attention'] = rearrange(
                gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'
            )

        if self.monitor_fusion_gate and modality_gate is not None and anchor_source == 'pc':
            gate_detached = modality_gate.detach().float()
            # `gate` is the coefficient of anchor_part_seq in
            # fused = rgb * (1-gate) + pc * gate, so its mean is the PC ratio.
            visual_summary['scalar/fusion_gate_pc_ratio'] = gate_detached.mean()
            visual_summary['scalar/fusion_gate_pc_percent'] = gate_detached.mean() * 100.0
            visual_summary['scalar/fusion_gate_pc_min'] = gate_detached.amin()
            visual_summary['scalar/fusion_gate_pc_max'] = gate_detached.amax()
            visual_summary['scalar/fusion_gate_pc_std'] = gate_detached.std(unbiased=False)
            if self.monitor_gate_parts:
                part_gate = gate_detached.mean(dim=(0, 1, 2))
                for part_idx, part_value in enumerate(part_gate):
                    visual_summary[f'scalar/fusion_gate_pc_part_{part_idx:02d}'] = part_value

        visual_summary['scalar/module_text_guidance_enabled'] = rgb_map.new_tensor(float(self.use_text_guidance))
        visual_summary['scalar/module_text_disentangle_enabled'] = rgb_map.new_tensor(float(self.use_nuisance))
        visual_summary['scalar/module_temporal_enabled'] = rgb_map.new_tensor(float(self.use_temporal))
        visual_summary['scalar/module_pc_aux_enabled'] = rgb_map.new_tensor(float(self.use_pc_aux))

        if self.training and pc_aux_embed_1 is not None and self.pc_aux_log_embedding_norm:
            visual_summary['scalar/pc_aux_embedding_norm'] = (
                pc_aux_embed_1.detach().float().norm(dim=1).mean()
            )

        if self.training and self.use_nuisance:
            # Scalars are intentionally kept detached for loggers that inspect visual_summary.
            for k, v in nuisance_diag.items():
                visual_summary[f'scalar/{k}'] = v.detach()

        return {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embed_1},
        }


    



# import os
# import math
# from typing import Dict, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# # ============================================================
# # Utilities
# # ============================================================

# class GradReverseFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd):
#         ctx.lambd = float(lambd)
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.lambd * grad_output, None


# def grad_reverse(x, lambd=1.0):
#     return GradReverseFn.apply(x, lambd)


# def l2n(x, dim=-1, eps=1e-12):
#     return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# def _scalar_like(x, value=0.0):
#     return x.new_tensor(float(value))


# def symmetric_infonce(z1, z2, temperature=0.07):
#     if z1 is None or z2 is None or z1.size(0) <= 1 or z2.size(0) <= 1:
#         base = z1 if z1 is not None else z2
#         return _scalar_like(base, 0.0)
#     z1 = l2n(z1.float(), dim=-1)
#     z2 = l2n(z2.float(), dim=-1)
#     logits = z1 @ z2.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)
#     return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
#     if z_img is None or z_txt is None:
#         base = z_img if z_img is not None else z_txt
#         return _scalar_like(base, 0.0)

#     if mask is not None:
#         mask = mask.float().view(mask.size(0), -1).mean(-1)
#         valid = mask > 0
#         if valid.sum() <= 1:
#             return z_img.new_tensor(0.0)
#         z_img = z_img[valid]
#         z_txt = z_txt[valid]
#         weights = mask[valid].clamp(0, 1)
#     else:
#         if z_img.size(0) <= 1:
#             return z_img.new_tensor(0.0)
#         weights = None

#     z_img = l2n(z_img.float(), dim=-1)
#     z_txt = l2n(z_txt.float(), dim=-1)
#     logits = z_img @ z_txt.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)

#     if weights is None:
#         return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

#     loss_i = F.cross_entropy(logits, labels, reduction='none')
#     loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
#     weights = weights / weights.mean().clamp_min(1e-6)
#     return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


# def ramp_value(step: int, start: int, length: int, max_value: float = 1.0):
#     if length <= 0:
#         return max_value if step >= start else 0.0
#     if step < start:
#         return 0.0
#     return float(max_value) * min(1.0, max(0.0, (step - start) / float(length)))


# # ============================================================
# # Text compression: keep old Qwen 3584-dim features compatible
# # ============================================================

# class TextEmbeddingAdapter(nn.Module):
#     """Compress high-dimensional MLLM text embeddings to a gait-sized code.

#     Existing Qwen2.5-VL features such as 3584-dim prompt/nuisance embeddings can be
#     used directly.  The adapter crops/pads unexpected dimensions, then learns a
#     low-rank bottleneck.  No offline feature regeneration is required.
#     """

#     def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024, dropout=0.1, l2_norm=True):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.code_dim = int(code_dim)
#         self.l2_norm = bool(l2_norm)
#         hidden_dim = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
#         self.in_norm = nn.LayerNorm(self.input_dim)
#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, hidden_dim, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.code_dim, bias=False),
#             nn.LayerNorm(self.code_dim),
#         )

#     def _fit_dim(self, x):
#         d = x.size(-1)
#         if d == self.input_dim:
#             return x
#         if d > self.input_dim:
#             return x[..., :self.input_dim]
#         pad = x.new_zeros(*x.shape[:-1], self.input_dim - d)
#         return torch.cat([x, pad], dim=-1)

#     def forward(self, x):
#         if x is None:
#             return None
#         if x.dim() == 3:
#             x = x.mean(1)
#         elif x.dim() > 3:
#             x = x.view(x.size(0), -1)
#         x = self._fit_dim(x.float())
#         z = self.net(self.in_norm(x))
#         return l2n(z, dim=-1) if self.l2_norm else z


# # ============================================================
# # Module A: RGB encoder
# # ============================================================

# class TinyDenseRGBEncoder(nn.Module):
#     """Dependency-free dense RGB encoder; use timm backbone if available/needed."""

#     def __init__(self, out_channels=512, in_channels=3):
#         super().__init__()
#         c1 = max(out_channels // 8, 32)
#         c2 = max(out_channels // 4, 64)
#         c3 = max(out_channels // 2, 128)
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.GELU(),
#             nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.GELU(),
#             nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c3),
#             nn.GELU(),
#             nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.out_channels = out_channels

#     def forward(self, x):
#         # x: [B, T, C, H, W]
#         B, T, C, H, W = x.shape
#         y = self.net(x.reshape(B * T, C, H, W))
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# class TimmDenseRGBEncoder(nn.Module):
#     """Optional dense foundation encoder via timm features_only models."""

#     def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', out_channels=512, pretrained=True, freeze=True):
#         super().__init__()
#         try:
#             import timm
#         except Exception as exc:
#             raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

#         self.model_name = model_name
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
#         chs = self.backbone.feature_info.channels()
#         in_c = chs[-1]
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_c, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         if freeze:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         self.out_channels = out_channels

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         y = self.backbone(x.reshape(B * T, C, H, W))[-1]
#         y = self.proj(y)
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# # ============================================================
# # Module B1: positive gait prompt gate
# # ============================================================

# class PositiveTextGate(nn.Module):
#     """Fixed gait-positive prompt bank -> channel/spatial gait gate.

#     This preserves motion-biometric cues before early RGB-anchor fusion.  The
#     prompt bank can still be the old Qwen .npy file; the high-dimensional text
#     vectors are compressed by TextEmbeddingAdapter inside the module.
#     """

#     def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
#                  trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024, dropout=0.1):
#         super().__init__()
#         self.channels = channels

#         if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
#             arr = np.load(prompt_feat_path).astype('float32')
#             if arr.ndim == 1:
#                 arr = arr[None, :]
#             prompt = torch.from_numpy(arr)
#             text_dim = int(prompt.shape[-1])
#         else:
#             prompt = torch.randn(num_prompts, text_dim) * 0.02

#         self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
#         self.text_adapter = TextEmbeddingAdapter(
#             input_dim=text_dim,
#             code_dim=text_code_dim,
#             hidden_dim=text_hidden_dim,
#             dropout=dropout,
#             l2_norm=False,
#         )
#         self.text_to_channel = nn.Sequential(
#             nn.Linear(text_code_dim, channels),
#             nn.LayerNorm(channels),
#             nn.GELU(),
#         )
#         self.channel_gate = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.Sigmoid(),
#         )
#         mid = max(channels // 4, 16)
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(channels, mid, 1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.GELU(),
#             nn.Conv2d(mid, 1, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, feat_map):
#         # feat_map: [B, T, C, H, W]
#         B, T, C, H, W = feat_map.shape
#         prompt = self.prompt_feat.to(device=feat_map.device)
#         pos_code = self.text_adapter(prompt).mean(0)  # [Dtxt]
#         pos = self.text_to_channel(pos_code)          # [C]
#         ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

#         x = feat_map * (1.0 + ch_gate)
#         sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
#         guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
#         return guided, sp_gate


# # ============================================================
# # Module B2: staged nuisance removal
# # ============================================================

# class StagedNuisanceRemoval(nn.Module):
#     """Four-stage nuisance disentanglement/removal.

#     Stage 0 warmup: no nuisance gradient.
#     Stage 1 probe: learn visual nuisance probes and risk prediction using detached visual features.
#     Stage 2 adversarial: gradually apply GRL to identity features to remove nuisance information.
#     Stage 3 refinement: turn on a visual-only part reliability gate before temporal modeling.

#     Text inputs are training supervision only; at inference the reliability gate uses only visual
#     features predicted by the nuisance probes, so existing retrieval remains text-free.
#     """

#     def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256, text_hidden_dim=1024,
#                  sub_dim=256, factors=None, grl_lambda=0.08, temperature=0.07,
#                  warmup_iters=5000, probe_iters=10000, adv_ramp_iters=10000,
#                  reliability_start_iter=15000, reliability_ramp_iters=10000,
#                  probe_loss_weight=0.03, adv_loss_weight=0.03, mask_loss_weight=0.01,
#                  max_suppression=0.35, detach_reliability_gate=True, factor_weights=None,
#                  dropout=0.1):
#         super().__init__()
#         self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
#         self.parts_num = int(parts_num)
#         self.grl_lambda = float(grl_lambda)
#         self.temperature = float(temperature)
#         self.warmup_iters = int(warmup_iters)
#         self.probe_iters = int(probe_iters)
#         self.adv_ramp_iters = int(adv_ramp_iters)
#         self.reliability_start_iter = int(reliability_start_iter)
#         self.reliability_ramp_iters = int(reliability_ramp_iters)
#         self.probe_loss_weight = float(probe_loss_weight)
#         self.adv_loss_weight = float(adv_loss_weight)
#         self.mask_loss_weight = float(mask_loss_weight)
#         self.max_suppression = float(max_suppression)
#         self.detach_reliability_gate = bool(detach_reliability_gate)

#         self.factor_weights = {k: 1.0 for k in self.factors}
#         if factor_weights:
#             self.factor_weights.update({str(k): float(v) for k, v in factor_weights.items()})

#         self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim, dropout=dropout, l2_norm=True)
#         self.txt_heads = nn.ModuleDict({
#             k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim))
#             for k in self.factors
#         })
#         self.visual_probe = nn.ModuleDict({
#             k: nn.Sequential(
#                 nn.Linear(in_dim, sub_dim),
#                 nn.LayerNorm(sub_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(sub_dim, sub_dim),
#                 nn.LayerNorm(sub_dim),
#             ) for k in self.factors
#         })
#         self.risk_heads = nn.ModuleDict({
#             k: nn.Sequential(nn.Linear(sub_dim, sub_dim // 2), nn.GELU(), nn.Linear(sub_dim // 2, 1))
#             for k in self.factors
#         })
#         self.part_heads = nn.ModuleDict({
#             k: nn.Sequential(nn.Linear(sub_dim, sub_dim), nn.GELU(), nn.Linear(sub_dim, self.parts_num))
#             for k in self.factors
#         })

#     def schedule(self, step: int):
#         probe_start = self.warmup_iters
#         adv_start = self.warmup_iters + self.probe_iters
#         return {
#             'probe': ramp_value(step, probe_start, max(1, self.probe_iters), 1.0),
#             'adv': ramp_value(step, adv_start, max(1, self.adv_ramp_iters), 1.0),
#             'rel': ramp_value(step, self.reliability_start_iter, max(1, self.reliability_ramp_iters), 1.0),
#         }

#     def _target_mask(self, masks, factor):
#         if masks is None or factor not in masks or masks[factor] is None:
#             return None
#         m = masks[factor].float()
#         if m.dim() > 1:
#             m = m.view(m.size(0), -1).mean(-1)
#         return m.clamp(0, 1)

#     def _text_z(self, text_feats, factor):
#         if text_feats is None or factor not in text_feats or text_feats[factor] is None:
#             return None
#         code = self.text_adapter(text_feats[factor])
#         return l2n(self.txt_heads[factor](code), dim=-1)

#     def pre_temporal(self, visual_global, text_feats=None, masks=None, step: int = 0):
#         """Return visual-only part reliability and pre-temporal nuisance losses.

#         visual_global: [B, C], computed from RGB part tokens before temporal modeling.
#         """
#         sched = self.schedule(step)
#         B = visual_global.size(0)
#         device = visual_global.device
#         losses = {}
#         bad_parts = visual_global.new_zeros(B, self.parts_num)

#         for f in self.factors:
#             # Probe branch learns to extract nuisance from detached visual features.
#             z_probe = l2n(self.visual_probe[f](visual_global.detach().float()), dim=-1)
#             target = self._target_mask(masks, f)
#             z_txt = self._text_z(text_feats, f)

#             if z_txt is not None and sched['probe'] > 0 and self.probe_loss_weight > 0:
#                 losses[f'probe_{f}'] = masked_symmetric_infonce(
#                     z_probe, z_txt, mask=target, temperature=self.temperature
#                 ) * (self.probe_loss_weight * sched['probe'])

#             risk = torch.sigmoid(self.risk_heads[f](z_probe)).squeeze(-1)  # [B]
#             if target is not None and sched['probe'] > 0 and self.mask_loss_weight > 0:
#                 losses[f'risk_{f}'] = F.mse_loss(risk, target.float()) * (self.mask_loss_weight * sched['probe'])

#             part_bad = torch.sigmoid(self.part_heads[f](z_probe))          # [B,P]
#             factor_w = float(self.factor_weights.get(f, 1.0))
#             bad_parts = bad_parts + factor_w * risk.unsqueeze(-1) * part_bad

#         if sched['rel'] <= 0:
#             reliability = visual_global.new_ones(B, 1, 1, self.parts_num)
#         else:
#             bad_parts = bad_parts / max(1.0, sum(abs(self.factor_weights.get(f, 1.0)) for f in self.factors))
#             bad_parts = bad_parts.clamp(0, 1)
#             if self.detach_reliability_gate:
#                 bad_parts = bad_parts.detach()
#             reliability = 1.0 - self.max_suppression * sched['rel'] * bad_parts
#             reliability = reliability.clamp(min=1.0 - self.max_suppression, max=1.0).view(B, 1, 1, self.parts_num)

#         diag = {
#             'sched_probe': visual_global.new_tensor(sched['probe']),
#             'sched_adv': visual_global.new_tensor(sched['adv']),
#             'sched_rel': visual_global.new_tensor(sched['rel']),
#             'mean_reliability': reliability.mean().detach(),
#         }
#         return reliability, losses, diag

#     def post_temporal(self, id_global, text_feats=None, masks=None, step: int = 0):
#         sched = self.schedule(step)
#         losses = {}
#         if sched['adv'] <= 0 or self.adv_loss_weight <= 0:
#             return losses
#         lambd = self.grl_lambda * sched['adv']
#         id_rev = grad_reverse(id_global.float(), lambd)
#         for f in self.factors:
#             z_txt = self._text_z(text_feats, f)
#             if z_txt is None:
#                 continue
#             target = self._target_mask(masks, f)
#             z_id = l2n(self.visual_probe[f](id_rev), dim=-1)
#             losses[f'adv_{f}'] = masked_symmetric_infonce(
#                 z_id, z_txt, mask=target, temperature=self.temperature
#             ) * (self.adv_loss_weight * sched['adv'])
#         return losses


# # Backward-compatible alias for old imports/checkpoints that refer to this name.
# NuisanceDetachHead = StagedNuisanceRemoval


# # ============================================================
# # Module C: geometry / silhouette / point-cloud anchors
# # ============================================================

# class SilhouetteAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512):
#         super().__init__()
#         self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

#     def forward(self, sil):
#         # sil: [B,T,H,W] or [B,T,1,H,W]
#         if sil.dim() == 4:
#             sil = sil.unsqueeze(2)
#         elif sil.dim() == 5 and sil.shape[2] != 1:
#             sil = sil[:, :, :1]
#         return self.net(sil.float())


# class PointCloudAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512, parts_num=16, in_dim=3):
#         super().__init__()
#         self.parts_num = int(parts_num)
#         self.out_channels = int(out_channels)
#         self.in_dim = int(in_dim)
#         self.point_mlp = nn.Sequential(
#             nn.Linear(self.in_dim, 64),
#             nn.GELU(),
#             nn.Linear(64, 128),
#             nn.GELU(),
#             nn.Linear(128, out_channels),
#             nn.GELU(),
#         )
#         self.norm = nn.LayerNorm(out_channels)

#     def _fit_dim(self, pc):
#         d = pc.size(-1)
#         if d == self.in_dim:
#             return pc
#         if d > self.in_dim:
#             return pc[..., :self.in_dim]
#         pad = pc.new_zeros(*pc.shape[:-1], self.in_dim - d)
#         return torch.cat([pc, pad], dim=-1)

#     def forward(self, pc):
#         # pc: [B,T,N,C]
#         B, T, N, _ = pc.shape
#         pc = self._fit_dim(pc.float())
#         xyz = pc[..., :min(3, self.in_dim)]
#         z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
#         z_min = z.min(dim=-1, keepdim=True).values
#         z_max = z.max(dim=-1, keepdim=True).values
#         z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

#         feat = self.point_mlp(pc.reshape(B * T * N, self.in_dim)).reshape(B, T, N, self.out_channels)
#         feat = self.norm(feat)

#         part_feats = []
#         for p in range(self.parts_num):
#             lo = p / self.parts_num
#             hi = (p + 1) / self.parts_num
#             if p == self.parts_num - 1:
#                 mask = ((z_norm >= lo) & (z_norm <= hi)).float().unsqueeze(-1)
#             else:
#                 mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
#             denom = mask.sum(dim=2).clamp_min(1.0)
#             pooled = (feat * mask).sum(dim=2) / denom
#             part_feats.append(pooled)
#         return torch.stack(part_feats, dim=-1)  # [B,T,C,P]


# # ============================================================
# # Module D: early part-token fusion + unified temporal encoder
# # ============================================================

# class EarlyPartTokenFusion(nn.Module):
#     """Fuse RGB and anchor part tokens before temporal modeling.

#     This fixes the previous late-fusion design where each modality was temporally
#     encoded independently before interaction.
#     """

#     def __init__(self, channels=512, dropout=0.05):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Conv1d(channels * 2, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(channels, 1, 1),
#         )
#         self.out = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#         )

#     def forward(self, rgb_part_seq, anchor_part_seq=None):
#         # [B,T,C,P]
#         B, T, C, P = rgb_part_seq.shape
#         if anchor_part_seq is None:
#             x = rgb_part_seq.reshape(B * T, C, P)
#             return self.out(x).reshape(B, T, C, P), None
#         if anchor_part_seq.size(-1) != P:
#             raise ValueError(f'Anchor parts {anchor_part_seq.size(-1)} != RGB parts {P}')
#         pair = torch.cat([rgb_part_seq, anchor_part_seq], dim=2).reshape(B * T, C * 2, P)
#         gate = torch.sigmoid(self.gate(pair)).reshape(B, T, 1, P)
#         fused = rgb_part_seq * (1.0 - gate) + anchor_part_seq * gate
#         fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)
#         return fused, gate


# class CycleAwareTemporalEncoder(nn.Module):
#     """Temporal encoder with frame, difference and low-frequency gait cycle branches."""

#     def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1, future_pred=True):
#         super().__init__()
#         self.channels = int(channels)
#         self.parts_num = int(parts_num)
#         self.future_pred = bool(future_pred)
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=channels,
#             nhead=heads,
#             dim_feedforward=channels * 4,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu',
#             norm_first=True,
#         )
#         self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
#         self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
#         self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
#         self.attn = nn.Sequential(nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(), nn.Linear(max(channels // 4, 16), 1))
#         if self.future_pred:
#             self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

#     def forward(self, part_seq):
#         # part_seq: [B,T,C,P]
#         B, T, C, P = part_seq.shape
#         x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
#         x_tem = self.temporal(x)
#         if T > 1:
#             diff = x[:, 1:] - x[:, :-1]
#             diff = F.pad(diff, (0, 0, 1, 0))
#         else:
#             diff = torch.zeros_like(x)
#         diff = self.diff_proj(diff)
#         if T > 1:
#             freq = torch.fft.rfft(x.float(), dim=1).abs()
#             freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
#         else:
#             freq = x.float().mean(1)
#         freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)
#         y = x_tem + diff + freq
#         att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)
#         pooled = (y * att.unsqueeze(-1)).sum(1)
#         pooled = pooled.reshape(B, P, C).permute(0, 2, 1).contiguous()
#         future_loss = pooled.new_tensor(0.0)
#         if self.future_pred and T > 2:
#             pred = self.future_head(y[:, :-1])
#             target = x[:, 1:].detach()
#             future_loss = F.mse_loss(pred, target)
#         att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()
#         return pooled, future_loss, att



# class MaxPoolTemporalEncoder(nn.Module):
#     """Temporal ablation fallback: max-pool part tokens across frames.

#     The interface matches CycleAwareTemporalEncoder so the downstream FC/BNNeck
#     path does not change during ablations.
#     """

#     def forward(self, part_seq):
#         # part_seq: [B,T,C,P]
#         B, T, C, P = part_seq.shape
#         pooled = part_seq.max(dim=1).values
#         future_loss = pooled.new_tensor(0.0)
#         temporal_attn = pooled.new_full((B, T, P), 1.0 / max(T, 1))
#         return pooled, future_loss, temporal_attn


# # ============================================================
# # Module E: lightweight verifier regularization
# # ============================================================

# class RuleBasedVerifier(nn.Module):
#     def __init__(self, attention_weight=0.0):
#         super().__init__()
#         self.attention_weight = float(attention_weight)

#     def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
#         if self.attention_weight <= 0 or gait_attn is None:
#             return gait_attn.new_tensor(0.0)
#         B, T, _, Hf, Wf = gait_attn.shape
#         target = None
#         if sil_seq is not None:
#             sil = sil_seq.unsqueeze(2) if sil_seq.dim() == 4 else sil_seq[:, :, :1]
#             target = F.interpolate(
#                 sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
#                 size=(Hf, Wf), mode='bilinear', align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = (target > 0.05).float()
#         elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
#             gray = rgb_seq.float().mean(2, keepdim=True)
#             diff = torch.zeros_like(gray)
#             diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
#             target = F.interpolate(
#                 diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
#                 size=(Hf, Wf), mode='bilinear', align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
#         if target is None:
#             return gait_attn.new_tensor(0.0)
#         # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
#         if self.attention_weight <= 0:
#             return gait_attn.new_tensor(0.0)

#         with torch.cuda.amp.autocast(enabled=False):
#             loss = F.binary_cross_entropy(
#                 gait_attn.float().clamp(1e-6, 1 - 1e-6),
#                 target.float()
#             )

#         return loss * self.attention_weight


# # ============================================================
# # Main OpenGait model
# # ============================================================

# class GaitOracle(BaseModel):
#     """GaitOracle strong version.

#     Key changes against the previous version:
#       1) RGB/geometry anchor are fused at part-token level before temporal encoding.
#       2) Qwen 3584-d text embeddings are compressed to gait-sized codes on the fly.
#       3) Nuisance removal is staged: warmup -> probe -> adversarial -> reliability refinement.
#       4) Existing positive prompt .npy and qwen25vl_*_feat.pkl / mask.pkl files remain compatible.
#     """

#     def build_network(self, model_cfg):
#         self.model_cfg = model_cfg
#         C = model_cfg.get('channels', 512)
#         P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
#         self.channels = int(C)
#         self.parts_num = int(P)
#         self.rgb_index = model_cfg.get('rgb_index', 0)
#         self.sil_index = model_cfg.get('sil_index', None)
#         self.pc_index = model_cfg.get('pc_index', None)
#         self.nuisance_indices = model_cfg.get('nuisance_indices', {})
#         self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
#         self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

#         # Centralized ablation switches.  Section-level `enabled` flags are still
#         # supported for backward compatibility, but `module_switches` is easier
#         # to edit when running module ablations.
#         switch_cfg = model_cfg.get('module_switches', {})

#         text_cfg = model_cfg.get('text_adapter', {})
#         text_dim = int(text_cfg.get('input_dim', model_cfg.get('text_dim', 3584)))
#         text_code_dim = int(text_cfg.get('code_dim', 256))
#         text_hidden_dim = int(text_cfg.get('hidden_dim', 1024))
#         text_dropout = float(text_cfg.get('dropout', 0.1))

#         rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
#         if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
#             self.rgb_encoder = TimmDenseRGBEncoder(
#                 model_name=rgb_backbone_cfg.get('model_name', 'vit_base_patch14_dinov2.lvd142m'),
#                 out_channels=C,
#                 pretrained=rgb_backbone_cfg.get('pretrained', True),
#                 freeze=rgb_backbone_cfg.get('freeze', True),
#             )
#         else:
#             self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

#         pos_cfg = model_cfg.get('positive_text', {})
#         self.use_text_guidance = bool(switch_cfg.get(
#             'text_guidance', pos_cfg.get('enabled', True)
#         )) and bool(pos_cfg.get('enabled', True))
#         self.positive_gate = PositiveTextGate(
#             text_dim=pos_cfg.get('text_dim', text_dim),
#             channels=C,
#             prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
#             num_prompts=pos_cfg.get('num_prompts', 4),
#             trainable_prompts=pos_cfg.get('trainable', False),
#             text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
#             text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
#             dropout=pos_cfg.get('dropout', text_dropout),
#         ) if self.use_text_guidance else None

#         self.use_sil_anchor = self.sil_index is not None
#         self.use_pc_anchor = self.pc_index is not None
#         self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
#         self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None
#         self.HPP = HorizontalPoolingPyramid(bin_num=[P])

#         early_cfg = model_cfg.get('early_fusion', {})
#         self.early_fusion = EarlyPartTokenFusion(channels=C, dropout=early_cfg.get('dropout', 0.05))

#         temporal_cfg = model_cfg.get('cycle_temporal', {})
#         self.use_temporal = bool(switch_cfg.get(
#             'temporal', temporal_cfg.get('enabled', True)
#         )) and bool(temporal_cfg.get('enabled', True))
#         if self.use_temporal:
#             self.temporal = CycleAwareTemporalEncoder(
#                 channels=C,
#                 parts_num=P,
#                 heads=temporal_cfg.get('heads', 4),
#                 layers=temporal_cfg.get('layers', 1),
#                 dropout=temporal_cfg.get('dropout', 0.1),
#                 future_pred=temporal_cfg.get('future_pred', True),
#             )
#         else:
#             self.temporal = MaxPoolTemporalEncoder()

#         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
#         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

#         nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
#         self.use_nuisance = bool(switch_cfg.get(
#             'text_disentangle', nuis_cfg.get('enabled', False)
#         )) and bool(nuis_cfg.get('enabled', False))
#         legacy_loss_w = float(nuis_cfg.get('loss_weight', 0.03))
#         if self.use_nuisance:
#             self.nuisance_removal = StagedNuisanceRemoval(
#                 in_dim=C,
#                 parts_num=P,
#                 text_dim=nuis_cfg.get('text_dim', text_dim),
#                 text_code_dim=nuis_cfg.get('text_code_dim', text_code_dim),
#                 text_hidden_dim=nuis_cfg.get('text_hidden_dim', text_hidden_dim),
#                 sub_dim=nuis_cfg.get('sub_dim', 256),
#                 factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
#                 grl_lambda=nuis_cfg.get('grl_lambda', 0.08),
#                 temperature=nuis_cfg.get('temperature', 0.07),
#                 warmup_iters=nuis_cfg.get('warmup_iters', 5000),
#                 probe_iters=nuis_cfg.get('probe_iters', 10000),
#                 adv_ramp_iters=nuis_cfg.get('adv_ramp_iters', 10000),
#                 reliability_start_iter=nuis_cfg.get('reliability_start_iter', 15000),
#                 reliability_ramp_iters=nuis_cfg.get('reliability_ramp_iters', 10000),
#                 probe_loss_weight=nuis_cfg.get('probe_loss_weight', legacy_loss_w),
#                 adv_loss_weight=nuis_cfg.get('adv_loss_weight', legacy_loss_w),
#                 mask_loss_weight=nuis_cfg.get('mask_loss_weight', legacy_loss_w * 0.5),
#                 max_suppression=nuis_cfg.get('max_suppression', 0.35),
#                 detach_reliability_gate=nuis_cfg.get('detach_reliability_gate', True),
#                 factor_weights=nuis_cfg.get('factor_weights', None),
#                 dropout=nuis_cfg.get('dropout', text_dropout),
#             )
#         else:
#             self.nuisance_removal = None

#         ver_cfg = model_cfg.get('verifier', {})
#         self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))
#         self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.1))
#         self.future_weight = float(model_cfg.get('future_weight', 0.01))
#         self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))

#         gate_monitor_cfg = model_cfg.get('fusion_gate_monitor', {})
#         self.monitor_fusion_gate = bool(switch_cfg.get(
#             'fusion_gate_monitor', gate_monitor_cfg.get('enabled', True)
#         )) and bool(gate_monitor_cfg.get('enabled', True))
#         self.monitor_gate_parts = bool(gate_monitor_cfg.get('log_part_mean', False))

#     def _current_step(self):
#         if self.training:
#             self._forward_iter += 1
#         for attr in ('iteration', 'iter', 'current_iter', 'global_step'):
#             if hasattr(self, attr):
#                 try:
#                     val = getattr(self, attr)
#                     if torch.is_tensor(val):
#                         return int(val.detach().item())
#                     return int(val)
#                 except Exception:
#                     pass
#         return int(self._forward_iter.detach().item())

#     def _to_rgb(self, x):
#         if x.dim() == 4:
#             x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
#         elif x.dim() == 5:
#             if x.shape[2] == 1:
#                 x = x.repeat(1, 1, 3, 1, 1)
#             elif x.shape[-1] == 3 and x.shape[2] != 3:
#                 x = x.permute(0, 1, 4, 2, 3)
#         return x.float()

#     def _to_text_feat(self, x):
#         if x is None:
#             return None
#         x = x.float()
#         if x.dim() == 3:
#             return x.mean(1)
#         if x.dim() == 2:
#             return x
#         return x.view(x.size(0), -1)

#     def _get_optional(self, ipts, idx):
#         if idx is None:
#             return None
#         if isinstance(idx, str):
#             if idx.lower() in ('none', 'null'):
#                 return None
#             idx = int(idx)
#         if idx < 0 or idx >= len(ipts):
#             return None
#         return ipts[idx]

#     def _dense_to_part_seq(self, feat_map):
#         B, T, C, H, W = feat_map.shape
#         part = self.HPP(feat_map.reshape(B * T, C, H, W))
#         return part.reshape(B, T, C, -1)

#     def _collect_nuisance_inputs(self, ipts):
#         text_feats, masks = {}, {}
#         for k, idx in self.nuisance_indices.items():
#             text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
#         for k, idx in self.nuisance_mask_indices.items():
#             masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
#         return text_feats, masks

#     def _sum_losses(self, losses: Dict[str, torch.Tensor], base: torch.Tensor):
#         total = base.new_tensor(0.0)
#         for v in losses.values():
#             if v is not None:
#                 total = total + v
#         return total

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs
#         step = self._current_step()
#         rgb = self._to_rgb(ipts[self.rgb_index])
#         sil = self._get_optional(ipts, self.sil_index)
#         pc = self._get_optional(ipts, self.pc_index)

#         rgb_map = self.rgb_encoder(rgb)
#         if self.use_text_guidance:
#             rgb_guided, gait_attn = self.positive_gate(rgb_map)
#         else:
#             # Text-guidance ablation: use raw RGB feature maps and skip
#             # attention/verifier losses tied to the text-guided gate.
#             rgb_guided, gait_attn = rgb_map, None
#         rgb_part_seq = self._dense_to_part_seq(rgb_guided)

#         anchor_part_seq = None
#         anchor_source = None
#         sil_for_verifier = None
#         if pc is not None and self.pc_encoder is not None:
#             anchor_part_seq = self.pc_encoder(pc.float())
#             anchor_source = 'pc'
#         elif sil is not None and self.sil_encoder is not None:
#             sil_for_verifier = sil
#             sil_map = self.sil_encoder(sil.float())
#             anchor_part_seq = self._dense_to_part_seq(sil_map)
#             anchor_source = 'sil'

#         nuisance_pre_losses = {}
#         nuisance_post_losses = {}
#         nuisance_diag = {}
#         text_feats, nuisance_masks = ({}, {})
#         if self.use_nuisance:
#             text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
#             pre_global = rgb_part_seq.mean(dim=(1, 3))
#             reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
#                 pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
#             )
#             rgb_part_seq = rgb_part_seq * reliability

#         # Early fusion happens before temporal encoding.
#         fused_seq, modality_gate = self.early_fusion(rgb_part_seq, anchor_part_seq)
#         fused_feat, future_loss, temporal_attn = self.temporal(fused_seq)

#         embed_1 = self.FCs(fused_feat)
#         embed_2, logits = self.BNNecks(embed_1)

#         training_feat = {
#             'triplet': {'embeddings': embed_1, 'labels': labs},
#             'softmax': {'logits': logits, 'labels': labs},
#         }


#         aux_losses = {}
#         if anchor_part_seq is not None and self.rgb_anchor_weight > 0:
#             rgb_g = rgb_part_seq.mean(dim=(1, 3))
#             anc_g = anchor_part_seq.mean(dim=(1, 3))
#             aux_losses['rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

#         if self.use_temporal and self.future_weight > 0:
#             aux_losses['future_pred'] = future_loss * self.future_weight

#         if self.use_nuisance:
#             id_global = fused_feat.mean(-1)
#             nuisance_post_losses = self.nuisance_removal.post_temporal(
#                 id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
#             )
#             aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_pre_losses.items()})
#             aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_post_losses.items()})

#         if gait_attn is not None:
#             attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
#             if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
#                 aux_losses['attention_verifier'] = attn_loss

#         if self.use_scalar_aux_loss:
#             aux_total = self._sum_losses(aux_losses, embed_1)
#             training_feat['gaitoracle_aux'] = {'loss': aux_total}
#         else:
#             for k, v in aux_losses.items():
#                 training_feat[f'loss_{k}'] = v

#         visual_summary = {
#             'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
#         }
#         if gait_attn is not None:
#             visual_summary['image/gait_attention'] = rearrange(
#                 gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'
#             )

#         if self.monitor_fusion_gate and modality_gate is not None and anchor_source == 'pc':
#             gate_detached = modality_gate.detach().float()
#             # `gate` is the coefficient of anchor_part_seq in
#             # fused = rgb * (1-gate) + pc * gate, so its mean is the PC ratio.
#             visual_summary['scalar/fusion_gate_pc_ratio'] = gate_detached.mean()
#             visual_summary['scalar/fusion_gate_pc_percent'] = gate_detached.mean() * 100.0
#             visual_summary['scalar/fusion_gate_pc_min'] = gate_detached.amin()
#             visual_summary['scalar/fusion_gate_pc_max'] = gate_detached.amax()
#             visual_summary['scalar/fusion_gate_pc_std'] = gate_detached.std(unbiased=False)
#             if self.monitor_gate_parts:
#                 part_gate = gate_detached.mean(dim=(0, 1, 2))
#                 for part_idx, part_value in enumerate(part_gate):
#                     visual_summary[f'scalar/fusion_gate_pc_part_{part_idx:02d}'] = part_value

#         visual_summary['scalar/module_text_guidance_enabled'] = rgb_map.new_tensor(float(self.use_text_guidance))
#         visual_summary['scalar/module_text_disentangle_enabled'] = rgb_map.new_tensor(float(self.use_nuisance))
#         visual_summary['scalar/module_temporal_enabled'] = rgb_map.new_tensor(float(self.use_temporal))

#         if self.training and self.use_nuisance:
#             # Scalars are intentionally kept detached for loggers that inspect visual_summary.
#             for k, v in nuisance_diag.items():
#                 visual_summary[f'scalar/{k}'] = v.detach()

#         return {
#             'training_feat': training_feat,
#             'visual_summary': visual_summary,
#             'inference_feat': {'embeddings': embed_1},
#         }

    



