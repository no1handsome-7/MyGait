"""GaitMoEOracle — Integrated multimodal MoE gait recognition model.

Combines GaitOracle (text-guided disentanglement, cycle-aware temporal) with
hierarchical MoE (per-modality experts, condition-aware fusion) into a single
configurable model.

Architecture
============
RGB  -> TinyDenseRGBEncoder -> [PositiveTextGate] -> HPP -> rgb_part_seq [B,T,C,P]
PC   -> PC_Encoder (simple / LidarGait++) -> pc_part_seq [B,T,C,P]

Per-frame (flatten B*T):
  rgb_flat [B*T,C,P]  -> [AppearanceMoE(E, top-k)] -> AppearanceProjector -> [B*T,C',P]
  pc_flat  [B*T,C,P]  -> [StructureMoE(E, top-k)]  -> StructureProjector  -> [B*T,C',P]
                                        ↓
              Stack -> [B*T, 2P, C'] -> [FusionMoE(E, top-k, condition-aware)] -> [B*T,C',P]

Reshape back -> [B,T,C',P]
  -> [CycleAwareTemporalEncoder / MaxPool] -> [B,C',P]
  -> SeparateFCs -> SeparateBNNecks
  -> [StagedNuisanceRemoval.post_temporal]

All MoE layers have balance + diversity losses. Every module supports config
ablation switches.
"""

import os
import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────

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
    if z1 is None or z2 is None or z1.size(0) <= 1:
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
        z_img, z_txt = z_img[valid], z_txt[valid]
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


# ──────────────────────────────────────────────────────────────
# Text modules (from GaitOracle)
# ──────────────────────────────────────────────────────────────

class TextEmbeddingAdapter(nn.Module):
    """Compress high-dim MLLM text embeddings to gait-sized codes."""

    def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024,
                 dropout=0.1, l2_norm=True):
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
    """Fixed gait-positive prompt bank -> channel/spatial gait gate."""

    def __init__(self, text_dim, channels, prompt_feat_path=None,
                 num_prompts=4, trainable_prompts=False, text_code_dim=256,
                 text_hidden_dim=1024, dropout=0.1):
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
            input_dim=text_dim, code_dim=text_code_dim,
            hidden_dim=text_hidden_dim, dropout=dropout, l2_norm=False,
        )
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_code_dim, channels), nn.LayerNorm(channels), nn.GELU(),
        )
        self.channel_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        mid = max(channels // 4, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid(),
        )

    def forward(self, feat_map):
        B, T, C, H, W = feat_map.shape
        prompt = self.prompt_feat.to(device=feat_map.device)
        pos_code = self.text_adapter(prompt).mean(0)
        pos = self.text_to_channel(pos_code)
        ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)
        x = feat_map * (1.0 + ch_gate)
        sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
        return guided, sp_gate


# ──────────────────────────────────────────────────────────────
# RGB encoder (from GaitOracle)
# ──────────────────────────────────────────────────────────────

class TinyDenseRGBEncoder(nn.Module):
    def __init__(self, out_channels=512, in_channels=3):
        super().__init__()
        c1, c2, c3 = max(out_channels // 8, 32), max(out_channels // 4, 64), max(out_channels // 2, 128)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, 1, 1, bias=False), nn.BatchNorm2d(c1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.GELU(),
            nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.GELU(),
            nn.Conv2d(c3, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B * T, C, H, W))
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


# ──────────────────────────────────────────────────────────────
# Point-cloud encoders
# ──────────────────────────────────────────────────────────────

class PointCloudSimpleEncoder(nn.Module):
    """Lightweight MLP encoder with height-based part binning (from GaitOracle)."""

    def __init__(self, out_channels=512, parts_num=16, in_dim=3):
        super().__init__()
        self.parts_num, self.out_channels, self.in_dim = int(parts_num), int(out_channels), int(in_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.in_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, out_channels), nn.GELU(),
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
        """pc: [B, T, N, C] -> [B, T, C_out, P]"""
        B, T, N, _ = pc.shape
        pc = self._fit_dim(pc.float())
        xyz = pc[..., :min(3, self.in_dim)]
        z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
        z_min = z.min(dim=-1, keepdim=True).values
        z_max = z.max(dim=-1, keepdim=True).values
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)
        feat = self.point_mlp(pc.reshape(B * T * N, self.in_dim)).reshape(B, T, N, self.out_channels)
        feat = self.norm(feat)
        parts = []
        for p in range(self.parts_num):
            lo, hi = p / self.parts_num, (p + 1) / self.parts_num
            mask = ((z_norm >= lo) & (z_norm < hi if p < self.parts_num - 1 else z_norm <= hi)).float().unsqueeze(-1)
            pooled = (feat * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
            parts.append(pooled)
        return torch.stack(parts, dim=-1)  # [B,T,C,P]


class LidarGaitPPEncoder(nn.Module):
    """LidarGait++ encoder wrapper (PointNet SA layers + PPPooling).

    Extracts part features without identity heads (FCs / BNNecks).
    """

    def __init__(self, channel=16, npoints=None, nsample=8,
                 sampling='knn', scale_aware=True, normalize_dp=True,
                 out_channels=256, parts_num=16):
        super().__init__()
        from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling
        npoints = npoints or [256, 128, 64]
        in_channel = 4 if scale_aware else 3
        C = channel
        self.sa1 = PointNetSetAbstraction(npoints[0], 0.1, nsample, in_channel,
                                          [2*C, 2*C, 4*C], False, sampling, scale_aware, normalize_dp)
        self.sa2 = PointNetSetAbstraction(npoints[1], 0.2, nsample, 4*C + in_channel,
                                          [4*C, 4*C, 8*C], False, sampling, scale_aware, normalize_dp)
        self.sa3 = PointNetSetAbstraction(npoints[2], 0.4, nsample, 8*C + in_channel,
                                          [8*C, 8*C, 16*C], False, sampling, scale_aware, normalize_dp)
        self.sa4 = PointNetSetAbstraction(None, None, None, 16*C + in_channel,
                                          [16*C, 16*C, out_channels], True, sampling, scale_aware, normalize_dp)
        self.pool = PPPooling(scale_aware=scale_aware, bin_num=[parts_num])
        self.out_channels = out_channels
        self.parts_num = parts_num

    def forward(self, pc):
        """pc: [B, T, N, C] -> [B, T, C_out, P]"""
        B, T, N, C = pc.shape
        xyz = rearrange(pc, 'B T N C -> (B T) C N')
        l1_xyz, l1_pts = self.sa1(xyz, None)
        l1_pts = torch.max(l1_pts, dim=-2)[0]
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l2_pts = torch.max(l2_pts, dim=-2)[0]
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        l3_pts = torch.max(l3_pts, dim=-2)[0]
        l4_xyz, l4_pts = self.sa4(l3_xyz, l3_pts)
        x = self.pool(l4_pts, l3_xyz)  # [(B*T), C, P]
        return rearrange(x, '(B T) C P -> B T C P', B=B)


# ──────────────────────────────────────────────────────────────
# Nuisance removal (from GaitOracle)
# ──────────────────────────────────────────────────────────────

class StagedNuisanceRemoval(nn.Module):
    """Four-stage nuisance disentanglement/removal.

    Text inputs are training supervision only; inference is text-free.
    """

    def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256,
                 text_hidden_dim=1024, sub_dim=256, factors=None,
                 grl_lambda=0.08, temperature=0.07,
                 warmup_iters=5000, probe_iters=10000, adv_ramp_iters=10000,
                 reliability_start_iter=15000, reliability_ramp_iters=10000,
                 probe_loss_weight=0.03, adv_loss_weight=0.03, mask_loss_weight=0.01,
                 max_suppression=0.35, detach_reliability_gate=True,
                 factor_weights=None, dropout=0.1):
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

        self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim,
                                                 dropout=dropout, l2_norm=True)
        self.txt_heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim))
            for k in self.factors
        })
        self.visual_probe = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(in_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim),
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
        sched = self.schedule(step)
        B, device = visual_global.size(0), visual_global.device
        losses, bad_parts = {}, visual_global.new_zeros(B, self.parts_num)
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
            fw = float(self.factor_weights.get(f, 1.0))
            bad_parts = bad_parts + fw * risk.unsqueeze(-1) * part_bad
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


# ──────────────────────────────────────────────────────────────
# MoE building blocks
# ──────────────────────────────────────────────────────────────

class PartExpertBlock(nn.Module):
    """Lightweight expert: learnable part-attention bias + LoRA residual.

    Each expert learns to attend to different body-part subsets via
    ``part_logit``, and applies a low-rank residual transform for diversity.
    Parameters per expert: ``2 * rank * dim + parts_num`` (typically ~8 K).
    """

    def __init__(self, dim: int, parts_num: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.part_logit = nn.Parameter(torch.zeros(parts_num))
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C, P] -> [N, C, P]"""
        part_attn = torch.sigmoid(self.part_logit)                   # [P]
        x_m = x * part_attn.unsqueeze(0).unsqueeze(0)                # soft part mask
        h = x_m.permute(0, 2, 1)                                     # [N, P, C]
        delta = self.up(self.drop(self.act(self.down(self.norm(h)))))  # LoRA
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()                       # [N, C, P]


class TopKMoELayer(nn.Module):
    """Generic sample-level top-k MoE for part features.

    Used for appearance experts (on RGB parts) and structure experts (on PC parts).
    Returns output, router logits, router weights for loss computation.
    """

    def __init__(self, dim: int, parts_num: int, num_experts: int = 8,
                 top_k: int = 2, expert_rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            PartExpertBlock(dim, parts_num, rank=expert_rank, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, num_experts),
        )

    def forward(self, x: torch.Tensor):
        """x: [N, C, P] -> (output [N,C,P], logits [N,E], weights [N,E])"""
        N, C, P = x.shape
        router_in = x.mean(dim=-1)                                    # [N, C]
        logits = self.router(router_in)                                # [N, E]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)         # [N, k]
        # Sparse softmax
        mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights = F.softmax(logits.masked_fill(mask == 0, float('-inf')), dim=-1).to(dtype=x.dtype)  # [N, E]
        # Compute selected expert outputs only (efficiency)
        output = torch.zeros_like(x)
        for e_idx in range(self.num_experts):
            sel = mask[:, e_idx] > 0                                  # [N] bool
            if not sel.any():
                continue
            # e_out = self.experts[e_idx](x[sel])
            # output[sel] = output[sel] + weights[sel, e_idx].unsqueeze(-1).unsqueeze(-1) * e_out

            e_out = self.experts[e_idx](x[sel])
            w_e = weights[sel, e_idx].to(dtype=output.dtype).unsqueeze(-1).unsqueeze(-1)
            e_out = e_out.to(dtype=output.dtype)
            output[sel] = output[sel] + w_e * e_out

        # Residual connection for training stability
        output = output + x
        return output, logits, weights


# ──────────────────────────────────────────────────────────────
# Condition-aware Fusion MoE (token-level)
# ──────────────────────────────────────────────────────────────

class ConditionEncoder(nn.Module):
    """Encode handcrafted stats into a condition embedding with pseudo-label
    supervision (explicit gradient for condition awareness).

    Inputs 6-dim stats:  [rgb_brightness, rgb_contrast, rgb_edge,
                          pc_distance, pc_density, pc_spread]
    """

    def __init__(self, stats_dim: int = 6, cond_dim: int = 32,
                 hidden_dim: int = 64, num_illum_bins: int = 3,
                 num_dist_bins: int = 3):
        super().__init__()
        self.stats_dim = stats_dim
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.illum_head = nn.Linear(cond_dim, num_illum_bins)
        self.dist_head = nn.Linear(cond_dim, num_dist_bins)
        self.num_illum_bins = num_illum_bins
        self.num_dist_bins = num_dist_bins

    def forward(self, stats: torch.Tensor):
        """stats: [B, 6] -> cond_emb [B, cond_dim]"""
        cond = self.net(stats)
        illum_logits = self.illum_head(cond)
        dist_logits = self.dist_head(cond)
        return cond, illum_logits, dist_logits

    @staticmethod
    @torch.no_grad()
    def compute_pseudo_labels(stats: torch.Tensor, num_bins: int = 3):
        """Batch-adaptive binning into ``num_bins`` classes."""
        B = stats.size(0)
        if B < num_bins:
            return stats.new_zeros(B, dtype=torch.long), stats.new_zeros(B, dtype=torch.long)
        brightness = stats[:, 0]
        distance = stats[:, 3]

        def _bin(x):
            sorted_x = x.sort().values
            boundaries = [sorted_x[(len(sorted_x) * i) // num_bins].item()
                          for i in range(1, num_bins)]
            return torch.bucketize(x.contiguous(),
                                   torch.tensor(boundaries, device=x.device, dtype=x.dtype))

        return _bin(brightness), _bin(distance)


class FusionExpertBlock(nn.Module):
    """Expert for fusing multi-modal tokens via cross-attention + LoRA.

    Each expert has unique query offsets and token-selection biases that
    encourage different experts to attend to different input tokens.
    Backbone cross-attention is **shared** across experts (see FusionMoELayer).
    """

    def __init__(self, dim: int, output_parts: int, num_input_tokens: int,
                 rank: int = 16):
        super().__init__()
        self.query_offset = nn.Parameter(torch.randn(output_parts, dim) * 0.01)
        self.token_logit = nn.Parameter(torch.zeros(num_input_tokens))
        self.down = nn.Linear(dim, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(dim)

    def forward(self, shared_attn_out: torch.Tensor) -> torch.Tensor:
        """shared_attn_out: [N, P, C] (already cross-attended).  Apply LoRA."""
        h = self.norm(shared_attn_out)
        delta = self.up(self.act(self.down(h)))
        return shared_attn_out + self.scale * delta  # [N, P, C]


class FusionMoELayer(nn.Module):
    """Token-level fusion MoE with condition-aware routing.

    Shared cross-attention backbone + per-expert LoRA + token biases.
    Router is conditioned on both feature statistics and condition embeddings.
    """

    def __init__(self, dim: int, output_parts: int, num_input_tokens: int,
                 num_experts: int = 8, top_k: int = 2, num_heads: int = 4,
                 cond_dim: int = 32, expert_rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_parts = output_parts

        # Shared cross-attention: P queries attend to 2P input tokens
        self.shared_queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

        # Per-expert modules
        self.experts = nn.ModuleList([
            FusionExpertBlock(dim, output_parts, num_input_tokens, rank=expert_rank)
            for _ in range(num_experts)
        ])

        # Condition-aware router
        self.router = nn.Sequential(
            nn.Linear(dim + cond_dim, dim // 2), nn.ReLU(inplace=True),
            nn.Linear(dim // 2, num_experts),
        )

    def forward(self, tokens: torch.Tensor, cond_emb: torch.Tensor):
        """
        tokens:   [N, 2P, C]  -- stacked rgb + pc part features
        cond_emb: [N, cond_dim]
        Returns:  (output [N,C,P], logits [N,E], weights [N,E])
        """
        N = tokens.size(0)

        # --- Router (sample-level) ---
        feat_pool = tokens.mean(dim=1)                                 # [N, C]
        router_in = torch.cat([feat_pool, cond_emb], dim=-1)           # [N, C+cond]
        logits = self.router(router_in)                                # [N, E]
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights = F.softmax(logits.masked_fill(mask == 0, float('-inf')), dim=-1).to(dtype=tokens.dtype)

        # --- Shared cross-attention backbone ---
        kv = self.cross_norm_kv(tokens)                                # [N, 2P, C]
        output = torch.zeros(N, self.output_parts, tokens.size(-1),
                             device=tokens.device, dtype=tokens.dtype)

        for e_idx in range(self.num_experts):
            sel = mask[:, e_idx] > 0
            if not sel.any():
                continue
            expert = self.experts[e_idx]
            kv_e = kv[sel]
            N_sel = kv_e.size(0)

            # Apply per-expert token bias to keys/values
            tok_attn = torch.sigmoid(expert.token_logit)               # [2P]
            kv_e = kv_e * tok_attn.unsqueeze(0).unsqueeze(-1)

            # Per-expert query = shared + offset
            q = (self.shared_queries + expert.query_offset).unsqueeze(0).expand(N_sel, -1, -1)
            q = self.cross_norm_q(q)

            attn_out, _ = self.cross_attn(q, kv_e, kv_e)              # [N_sel, P, C]
            attn_out = attn_out + self.ffn(self.ffn_norm(attn_out))    # shared FFN
            attn_out = expert(attn_out)                                 # per-expert LoRA

            # w_e = weights[sel, e_idx].unsqueeze(-1).unsqueeze(-1)      # [N_sel,1,1]
            # output[sel] = output[sel] + w_e * attn_out

            w_e = weights[sel, e_idx].to(dtype=output.dtype).unsqueeze(-1).unsqueeze(-1)
            attn_out = attn_out.to(dtype=output.dtype)
            output[sel] = output[sel] + w_e * attn_out


        return output.permute(0, 2, 1).contiguous(), logits, weights   # [N,C,P]


# ──────────────────────────────────────────────────────────────
# Fallback fusion when MoE is disabled
# ──────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """Simple cross-attention fusion (ablation fallback for FusionMoE)."""

    def __init__(self, dim: int, output_parts: int, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim),
        )

    def forward(self, tokens: torch.Tensor, cond_emb=None):
        N = tokens.size(0)
        q = self.queries.unsqueeze(0).expand(N, -1, -1)
        kv = self.norm(tokens)
        out, _ = self.attn(q, kv, kv)
        out = out + self.ffn(out)
        return out.permute(0, 2, 1).contiguous(), None, None


class MLPFusion(nn.Module):
    """Concat-MLP fusion (ablation fallback)."""

    def __init__(self, dim: int, output_parts: int, num_input_tokens: int):
        super().__init__()
        # Pool 2P tokens to P via learned linear
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.output_parts = output_parts

    def forward(self, tokens: torch.Tensor, cond_emb=None):
        # tokens: [N, 2P, C] -> split and concat channel-wise
        P = self.output_parts
        rgb_t, pc_t = tokens[:, :P, :], tokens[:, P:, :]               # [N,P,C] each
        cat = torch.cat([rgb_t, pc_t], dim=-1)                         # [N,P,2C]
        out = self.proj(cat)                                            # [N,P,C]
        return out.permute(0, 2, 1).contiguous(), None, None


# ──────────────────────────────────────────────────────────────
# MoE auxiliary losses (computed inside model)
# ──────────────────────────────────────────────────────────────

def moe_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer style load-balance loss.

    ``L = E * sum(f_i * p_i)`` where f_i is the dispatch fraction and
    p_i is the mean router probability.
    """
    if logits is None or logits.dim() != 2:
        return logits.new_tensor(0.0) if logits is not None else torch.tensor(0.0)
    E = logits.size(1)
    probs = F.softmax(logits, dim=-1)                                   # [N, E]
    # f_i: fraction of samples where expert i has the highest probability
    top1 = logits.argmax(dim=-1)                                        # [N]
    f = torch.zeros(E, device=logits.device)
    for i in range(E):
        f[i] = (top1 == i).float().mean()
    p = probs.mean(dim=0)                                               # [E]
    return E * (f * p).sum()


def moe_diversity_loss(experts: nn.ModuleList, key: str = 'part_logit',
                       threshold: float = 0.5) -> torch.Tensor:
    """Pairwise cosine-similarity penalty on expert part-attention biases."""
    biases = []
    for exp in experts:
        if hasattr(exp, key):
            biases.append(torch.sigmoid(getattr(exp, key)))
    if len(biases) < 2:
        return experts[0].part_logit.new_tensor(0.0)
    stack = torch.stack(biases, dim=0)                                  # [E, P]
    stack_n = F.normalize(stack, dim=-1)
    sim = stack_n @ stack_n.t()                                         # [E, E]
    E = sim.size(0)
    mask = ~torch.eye(E, dtype=torch.bool, device=sim.device)
    offdiag = sim[mask]
    penalty = F.relu(offdiag - threshold)
    return penalty.mean()


def moe_fusion_diversity_loss(experts: nn.ModuleList,
                              threshold: float = 0.5) -> torch.Tensor:
    """Diversity loss on fusion-expert token biases."""
    return moe_diversity_loss(experts, key='token_logit', threshold=threshold)


# ──────────────────────────────────────────────────────────────
# Temporal encoder (from GaitOracle)
# ──────────────────────────────────────────────────────────────

class CycleAwareTemporalEncoder(nn.Module):
    """Temporal encoder with frame + diff + low-frequency branches."""

    def __init__(self, channels=512, parts_num=16, heads=4, layers=1,
                 dropout=0.1, future_pred=True):
        super().__init__()
        self.channels, self.parts_num = int(channels), int(parts_num)
        self.future_pred = bool(future_pred)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=heads, dim_feedforward=channels * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.attn = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(),
            nn.Linear(max(channels // 4, 16), 1),
        )
        if self.future_pred:
            self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, part_seq):
        """part_seq: [B, T, C, P] -> (pooled [B,C,P], future_loss, attn)"""
        B, T, C, P = part_seq.shape
        x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
        x_tem = self.temporal(x)
        diff = torch.zeros_like(x)
        if T > 1:
            diff[:, 1:] = x[:, 1:] - x[:, :-1]
        diff = self.diff_proj(diff)
        if T > 1:
            freq = torch.fft.rfft(x.float(), dim=1).abs()
            freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
        else:
            freq = x.float().mean(1)
        freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)
        y = x_tem + diff + freq
        att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)
        pooled = (y * att.unsqueeze(-1)).sum(1).reshape(B, P, C).permute(0, 2, 1).contiguous()
        future_loss = pooled.new_tensor(0.0)
        if self.future_pred and T > 2:
            pred = self.future_head(y[:, :-1])
            future_loss = F.mse_loss(pred, x[:, 1:].detach())
        att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()
        return pooled, future_loss, att


# ──────────────────────────────────────────────────────────────
# Verifier (from GaitOracle)
# ──────────────────────────────────────────────────────────────

class RuleBasedVerifier(nn.Module):
    def __init__(self, attention_weight=0.0):
        super().__init__()
        self.attention_weight = float(attention_weight)

    def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
        if self.attention_weight <= 0 or gait_attn is None:
            return gait_attn.new_tensor(0.0)
        B, T, _, Hf, Wf = gait_attn.shape
        target = None
        if rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
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
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(
                gait_attn.float().clamp(1e-6, 1 - 1e-6), target.float()
            )
        return loss * self.attention_weight


# ──────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────

class GaitMoEOracle(BaseModel):
    """Integrated MoE + text-disentangled gait model.

    Unifies GaitOracle's text-guided nuisance removal with hierarchical MoE
    expert routing.  All sub-modules are configurable via ``model_cfg.ablation``.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        C = int(model_cfg.get('channels', 256))
        P = int(model_cfg.get('parts_num', 16))
        proj_dim = int(model_cfg.get('proj_dim', C))
        self.channels, self.parts_num = C, P


        # --- Data indices ---
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

        # --- Ablation switches ---
        abl = model_cfg.get('ablation', {})
        self.appearance_moe_enabled = bool(abl.get('appearance_moe_enabled', True))
        self.structure_moe_enabled = bool(abl.get('structure_moe_enabled', True))
        self.fusion_moe_enabled = bool(abl.get('fusion_moe_enabled', True))
        self.fusion_fallback = str(abl.get('fusion_fallback', 'cross_attention'))
        self.fusion_condition_aware = bool(abl.get('fusion_condition_aware', True))
        self.temporal_type = str(abl.get('temporal_type', 'cycle_aware'))
        self.text_guidance_enabled = bool(abl.get('text_guidance_enabled', True))
        self.nuisance_detach_enabled = bool(abl.get('nuisance_detach_enabled', True))

        # MoE hyper-params
        app_E = int(abl.get('appearance_num_experts', 8))
        app_K = int(abl.get('appearance_topk', 2))
        str_E = int(abl.get('structure_num_experts', 8))
        str_K = int(abl.get('structure_topk', 2))
        fus_E = int(abl.get('fusion_num_experts', 8))
        fus_K = int(abl.get('fusion_topk', 2))
        expert_rank = int(abl.get('expert_rank', 16))
        expert_dropout = float(abl.get('expert_dropout', 0.0))

        # Loss weights
        self.balance_loss_weight = float(abl.get('balance_loss_weight', 0.01))
        self.diversity_loss_weight = float(abl.get('diversity_loss_weight', 0.01))
        self.diversity_threshold = float(abl.get('diversity_threshold', 0.5))
        self.condition_cls_weight = float(abl.get('condition_cls_weight', 0.05))

        # --- Text config ---
        text_cfg = model_cfg.get('text_adapter', {})
        text_dim = int(text_cfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        text_code_dim = int(text_cfg.get('code_dim', 256))
        text_hidden_dim = int(text_cfg.get('hidden_dim', 1024))
        text_dropout = float(text_cfg.get('dropout', 0.1))

        # --- RGB encoder ---
        rgb_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
        self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

        # --- Positive text gate ---
        if self.text_guidance_enabled:
            pos_cfg = model_cfg.get('positive_text', {})
            self.positive_gate = PositiveTextGate(
                text_dim=pos_cfg.get('text_dim', text_dim), channels=C,
                prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
                num_prompts=pos_cfg.get('num_prompts', 4),
                trainable_prompts=pos_cfg.get('trainable', False),
                text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
                text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
                dropout=pos_cfg.get('dropout', text_dropout),
            )
        else:
            self.positive_gate = None

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

        # --- HPP ---
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        # --- Nuisance removal ---
        if self.nuisance_detach_enabled:
            nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
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
                probe_loss_weight=nuis_cfg.get('probe_loss_weight', 0.03),
                adv_loss_weight=nuis_cfg.get('adv_loss_weight', 0.03),
                mask_loss_weight=nuis_cfg.get('mask_loss_weight', 0.015),
                max_suppression=nuis_cfg.get('max_suppression', 0.35),
                detach_reliability_gate=nuis_cfg.get('detach_reliability_gate', True),
                factor_weights=nuis_cfg.get('factor_weights', None),
                dropout=nuis_cfg.get('dropout', text_dropout),
            )
        else:
            self.nuisance_removal = None

        # --- Appearance MoE (on RGB part features) ---
        if self.appearance_moe_enabled:
            self.appearance_moe = TopKMoELayer(
                C, P, num_experts=app_E, top_k=app_K,
                expert_rank=expert_rank, dropout=expert_dropout,
            )
        else:
            self.appearance_moe = None

        # --- Structure MoE (on PC part features) ---
        if self.structure_moe_enabled:
            self.structure_moe = TopKMoELayer(
                C, P, num_experts=str_E, top_k=str_K,
                expert_rank=expert_rank, dropout=expert_dropout,
            )
        else:
            self.structure_moe = None

        # --- Projectors (align modalities to shared dim) ---
        # proj_dim = int(model_cfg.get('proj_dim', C))
        self.rgb_projector = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU(),
        )
        self.pc_projector = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU(),
        )
        self.post_nuisance_adapter = (
            nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity()
        )

        # --- Condition encoder ---
        cond_dim = int(abl.get('cond_dim', 32))
        self.cond_stats_dim = 6
        if self.fusion_condition_aware:
            self.condition_encoder = ConditionEncoder(
                stats_dim=self.cond_stats_dim, cond_dim=cond_dim,
                hidden_dim=int(abl.get('cond_hidden_dim', 64)),
            )
        else:
            self.condition_encoder = None

        # --- Fusion MoE / fallback ---
        num_input_tokens = 2 * P
        if self.fusion_moe_enabled:
            self.fusion_moe = FusionMoELayer(
                dim=proj_dim, output_parts=P, num_input_tokens=num_input_tokens,
                num_experts=fus_E, top_k=fus_K, num_heads=4,
                cond_dim=cond_dim if self.fusion_condition_aware else 0,
                expert_rank=expert_rank, dropout=expert_dropout,
            )
        else:
            if self.fusion_fallback == 'cross_attention':
                self.fusion_fallback_module = CrossAttentionFusion(proj_dim, P)
            else:
                self.fusion_fallback_module = MLPFusion(proj_dim, P, num_input_tokens)
            self.fusion_moe = None

        # --- Temporal encoder ---
        if self.temporal_type == 'cycle_aware':
            temp_cfg = model_cfg.get('cycle_temporal', {})
            self.temporal = CycleAwareTemporalEncoder(
                channels=proj_dim, parts_num=P,
                heads=temp_cfg.get('heads', 4),
                layers=temp_cfg.get('layers', 1),
                dropout=temp_cfg.get('dropout', 0.1),
                future_pred=temp_cfg.get('future_pred', True),
            )
        else:
            self.temporal = None  # max pooling fallback

        # --- Identity heads ---
        fc_cfg = model_cfg.get('SeparateFCs', {})
        bn_cfg = model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(
            parts_num=fc_cfg.get('parts_num', P),
            in_channels=fc_cfg.get('in_channels', proj_dim),
            out_channels=fc_cfg.get('out_channels', 256),
        )
        self.BNNecks = SeparateBNNecks(
            parts_num=bn_cfg.get('parts_num', P),
            in_channels=bn_cfg.get('in_channels', 256),
            class_num=bn_cfg.get('class_num', 250),
        )

        # --- Verifier ---
        ver_cfg = model_cfg.get('verifier', {})
        self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))

        # --- Aux weights ---
        self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.future_weight = float(model_cfg.get('future_weight', 0.01))
        self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))

    # ──────── helpers ────────

    def _current_step(self):
        if self.training:
            self._forward_iter += 1
        for attr in ('iteration', 'iter', 'current_iter', 'global_step'):
            if hasattr(self, attr):
                try:
                    val = getattr(self, attr)
                    return int(val.detach().item()) if torch.is_tensor(val) else int(val)
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
        if idx is None or (isinstance(idx, str) and idx.lower() in ('none', 'null')):
            return None
        idx = int(idx)
        return ipts[idx] if 0 <= idx < len(ipts) else None

    def _dense_to_part_seq(self, feat_map):
        B, T, C, H, W = feat_map.shape
        part = self.HPP(feat_map.reshape(B * T, C, H, W))
        return part.reshape(B, T, C, -1)                               # [B,T,C,P]

    def _collect_nuisance_inputs(self, ipts):
        text_feats, masks = {}, {}
        for k, idx in self.nuisance_indices.items():
            text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
        for k, idx in self.nuisance_mask_indices.items():
            masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
        return text_feats, masks

    @torch.no_grad()
    def _compute_condition_stats(self, rgb: torch.Tensor, pc: torch.Tensor) -> torch.Tensor:
        """6-dim handcrafted stats from raw RGB and PC inputs."""
        # RGB stats (from [B,T,C,H,W])
        rgb_f = rgb.float()
        brightness = rgb_f.mean(dim=(1, 2, 3, 4))
        contrast = rgb_f.std(dim=(1, 2, 3, 4), unbiased=False)
        # Edge strength
        dh = (rgb_f[:, :, :, 1:, :] - rgb_f[:, :, :, :-1, :]).abs().mean(dim=(1, 2, 3, 4))
        dw = (rgb_f[:, :, :, :, 1:] - rgb_f[:, :, :, :, :-1]).abs().mean(dim=(1, 2, 3, 4))
        edge = 0.5 * (dh + dw)
        # PC stats (from [B,T,N,C])
        pc_f = pc.float()
        xyz = pc_f[..., :3]
        xyz_clean = torch.nan_to_num(xyz, 0.0, 0.0, 0.0)
        distance = xyz_clean.norm(dim=-1).mean(dim=(1, 2))
        density = torch.isfinite(xyz).all(dim=-1).float().mean(dim=(1, 2))
        spread = xyz_clean.std(dim=2, unbiased=False).mean(dim=(1, 2))
        stats = torch.stack([brightness, contrast, edge, distance, density, spread], dim=1)
        return torch.nan_to_num(stats, 0.0, 0.0, 0.0)

    def _sum_losses(self, losses, ref):
        total = ref.new_tensor(0.0)
        for v in losses.values():
            if v is not None:
                total = total + v
        return total

    # ──────── forward ────────

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._current_step()
        device = ipts[0].device

        # ── 1. Encode RGB ──
        rgb = self._to_rgb(ipts[self.rgb_index])                        # [B,T,3,H,W]
        rgb_map = self.rgb_encoder(rgb)                                 # [B,T,C,H,W]
        gait_attn = None
        if self.positive_gate is not None:
            rgb_map, gait_attn = self.positive_gate(rgb_map)
        rgb_part_seq = self._dense_to_part_seq(rgb_map)                 # [B,T,C,P]

        # ── 2. Encode PC ──
        pc = ipts[self.pc_index].float()                                # [B,T,N,D]
        pc_part_seq = self.pc_encoder(pc)                               # [B,T,C,P]

        B, T, C, P = rgb_part_seq.shape

        # ── 3. Nuisance pre-temporal (on RGB) ──
        nuisance_pre_losses, nuisance_post_losses, nuisance_diag = {}, {}, {}
        text_feats, nuisance_masks = {}, {}
        if self.nuisance_removal is not None:
            text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
            pre_global = rgb_part_seq.mean(dim=(1, 3))                  # [B, C]
            reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
                pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            rgb_part_seq = rgb_part_seq * reliability                   # [B,T,C,P] * [B,1,1,P]

        # ── 4. Flatten temporal for per-frame MoE ──
        rgb_flat = rgb_part_seq.reshape(B * T, C, P)                    # [BT, C, P]
        pc_flat = pc_part_seq.reshape(B * T, C, P)

        # ── 5. Appearance MoE (RGB) ──
        app_logits, app_weights = None, None
        if self.appearance_moe is not None:
            rgb_flat, app_logits, app_weights = self.appearance_moe(rgb_flat)
        # ── 6. Structure MoE (PC) ──
        str_logits, str_weights = None, None
        if self.structure_moe is not None:
            pc_flat, str_logits, str_weights = self.structure_moe(pc_flat)

        # ── 7. Project to shared dim ──
        rgb_proj = self.rgb_projector(rgb_flat)                         # [BT, C', P]
        pc_proj = self.pc_projector(pc_flat)                            # [BT, C', P]

        # ── 8. Fusion ──
        tokens = torch.cat([
            rgb_proj.permute(0, 2, 1),                                  # [BT, P, C']
            pc_proj.permute(0, 2, 1),                                   # [BT, P, C']
        ], dim=1)                                                       # [BT, 2P, C']

        fus_logits, fus_weights = None, None
        cond_cls_loss = tokens.new_tensor(0.0)

        if self.fusion_moe is not None:
            # Condition encoding
            if self.condition_encoder is not None:
                cond_stats = self._compute_condition_stats(rgb, pc)      # [B, 6]
                cond_emb, illum_logits, dist_logits = self.condition_encoder(cond_stats)
                # Expand to [BT, cond_dim]
                cond_emb_exp = cond_emb.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
                # Pseudo-label loss
                if self.training and self.condition_cls_weight > 0:
                    illum_lbl, dist_lbl = ConditionEncoder.compute_pseudo_labels(cond_stats)
                    cond_cls_loss = (
                        F.cross_entropy(illum_logits, illum_lbl) +
                        F.cross_entropy(dist_logits, dist_lbl)
                    ) * self.condition_cls_weight
            else:
                cond_emb_exp = tokens.new_zeros(B * T, 32)

            fused_flat, fus_logits, fus_weights = self.fusion_moe(tokens, cond_emb_exp)
        else:
            fused_flat, _, _ = self.fusion_fallback_module(tokens)

        # ── 9. Reshape back & temporal encoding ──
        C_proj = fused_flat.size(1)
        fused_seq = fused_flat.reshape(B, T, C_proj, P)                 # [B,T,C',P]

        future_loss = fused_flat.new_tensor(0.0)
        if self.temporal is not None:
            feat, future_loss, temporal_attn = self.temporal(fused_seq)  # [B,C',P]
        else:
            feat = fused_seq.max(dim=1).values                          # [B,C',P]

        # ── 10. Identity heads ──
        embed_1 = self.FCs(feat)                                        # [B, emb_dim, P]
        embed_2, logits = self.BNNecks(embed_1)

        if not self.training:
            return {
                'training_feat': {},
                'visual_summary': {},
                'inference_feat': {'embeddings': embed_1},
            }

        # ── 11. Losses ──
        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }

        aux_losses = {}

        # RGB-PC anchor consistency
        if self.rgb_anchor_weight > 0:
            rgb_g = rgb_part_seq.mean(dim=(1, 3))
            pc_g = pc_part_seq.mean(dim=(1, 3))
            aux_losses['rgb_anchor'] = symmetric_infonce(rgb_g, pc_g) * self.rgb_anchor_weight

        # Future prediction
        if self.future_weight > 0:
            aux_losses['future_pred'] = future_loss * self.future_weight

        # MoE balance losses
        if self.training and self.balance_loss_weight > 0:
            bal = fused_flat.new_tensor(0.0)
            count = 0
            for tag_logits in [app_logits, str_logits, fus_logits]:
                if tag_logits is not None:
                    bal = bal + moe_balance_loss(tag_logits)
                    count += 1
            if count > 0:
                aux_losses['moe_balance'] = bal / count * self.balance_loss_weight

        # MoE diversity losses
        if self.training and self.diversity_loss_weight > 0:
            div = fused_flat.new_tensor(0.0)
            count = 0
            if self.appearance_moe is not None:
                div = div + moe_diversity_loss(self.appearance_moe.experts, 'part_logit', self.diversity_threshold)
                count += 1
            if self.structure_moe is not None:
                div = div + moe_diversity_loss(self.structure_moe.experts, 'part_logit', self.diversity_threshold)
                count += 1
            if self.fusion_moe is not None:
                div = div + moe_fusion_diversity_loss(self.fusion_moe.experts, self.diversity_threshold)
                count += 1
            if count > 0:
                aux_losses['moe_diversity'] = div / count * self.diversity_loss_weight

        # Condition classification loss
        if self.training and cond_cls_loss.abs() > 0:
            aux_losses['condition_cls'] = cond_cls_loss

        # Verifier
        if gait_attn is not None:
            attn_loss = self.verifier(gait_attn, rgb_seq=rgb)
            if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
                aux_losses['attention_verifier'] = attn_loss

        # Nuisance post-temporal
        if self.nuisance_removal is not None:
            id_global = feat.mean(-1)                                   # [B, proj_dim]
            id_global = self.post_nuisance_adapter(id_global.float())    # [B, C]
            nuisance_post_losses = self.nuisance_removal.post_temporal(
                id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )


        # Pack aux losses
        if self.use_scalar_aux_loss:
            aux_total = self._sum_losses(aux_losses, embed_1)
            training_feat['gaitoracle_aux'] = {'loss': aux_total}
        else:
            for k, v in aux_losses.items():
                training_feat[f'loss_{k}'] = v

        # ── Visual summary ──
        visual_summary = {}
        if rgb.dim() == 5:
            visual_summary['image/rgb'] = rearrange(
                rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w')
        if gait_attn is not None:
            visual_summary['image/gait_attention'] = rearrange(
                gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w')
        # MoE diagnostics
        if app_weights is not None:
            for i in range(app_weights.size(1)):
                visual_summary[f'scalar/app_expert_{i}_mean_w'] = app_weights[:, i].mean().detach()
        if str_weights is not None:
            for i in range(str_weights.size(1)):
                visual_summary[f'scalar/str_expert_{i}_mean_w'] = str_weights[:, i].mean().detach()
        if fus_weights is not None:
            for i in range(fus_weights.size(1)):
                visual_summary[f'scalar/fus_expert_{i}_mean_w'] = fus_weights[:, i].mean().detach()
        if self.nuisance_removal is not None and self.training:
            for k, v in nuisance_diag.items():
                visual_summary[f'scalar/{k}'] = v.detach()

        return {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embed_1},
        }
