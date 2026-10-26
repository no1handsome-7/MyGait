"""GaitMoEOracle shared building blocks (V2 & V3).

All four fixes applied:
  Fix-1: Sequence-level routing (TopKMoELayer accepts seq_len)
  Fix-2: Stronger experts (part_logit=2.0, scale=0.5, learnable residual gate)
  Fix-3: Per-expert independent low-rank attention in FusionMoE
  Fix-4: proj_dim defaults to channels (no premature bottleneck)

V3.1 additions:
  - ModalityReliabilityProbe: end-to-end learned modality quality assessment
  - probe_balance_loss: prevents modality weight collapse
"""

import os
import math
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# ================================================================
#  Utilities
# ================================================================

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


def _scalar(x, v=0.0):
    return x.new_tensor(float(v))


def symmetric_infonce(z1, z2, temperature=0.07):
    if z1 is None or z2 is None or z1.size(0) <= 1:
        return _scalar(z1 if z1 is not None else z2)
    z1, z2 = l2n(z1.float()), l2n(z2.float())
    logits = z1 @ z2.t() / temperature
    lbl = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))


def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
    if z_img is None or z_txt is None:
        return _scalar(z_img if z_img is not None else z_txt)
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
    z_img, z_txt = l2n(z_img.float()), l2n(z_txt.float())
    logits = z_img @ z_txt.t() / temperature
    lbl = torch.arange(logits.size(0), device=logits.device)
    if weights is None:
        return 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
    li = F.cross_entropy(logits, lbl, reduction='none')
    lt = F.cross_entropy(logits.t(), lbl, reduction='none')
    w = weights / weights.mean().clamp_min(1e-6)
    return 0.5 * ((li * w).mean() + (lt * w).mean())


def ramp_value(step, start, length, max_val=1.0):
    if length <= 0:
        return max_val if step >= start else 0.0
    if step < start:
        return 0.0
    return float(max_val) * min(1.0, (step - start) / float(length))


# ================================================================
#  Text modules
# ================================================================

class TextEmbeddingAdapter(nn.Module):
    def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024,
                 dropout=0.1, l2_norm=True):
        super().__init__()
        self.input_dim, self.code_dim, self.l2_norm = int(input_dim), int(code_dim), bool(l2_norm)
        hid = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
        self.in_norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hid, bias=False), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, self.code_dim, bias=False), nn.LayerNorm(self.code_dim),
        )

    def _fit(self, x):
        d = x.size(-1)
        if d == self.input_dim:
            return x
        if d > self.input_dim:
            return x[..., :self.input_dim]
        return torch.cat([x, x.new_zeros(*x.shape[:-1], self.input_dim - d)], -1)

    def forward(self, x):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.mean(1)
        elif x.dim() > 3:
            x = x.view(x.size(0), -1)
        z = self.net(self.in_norm(self._fit(x.float())))
        return l2n(z) if self.l2_norm else z


class PositiveTextGate(nn.Module):
    def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
                 trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024,
                 dropout=0.1):
        super().__init__()
        self.channels = channels
        if prompt_feat_path and os.path.exists(prompt_feat_path):
            arr = np.load(prompt_feat_path).astype('float32')
            prompt = torch.from_numpy(arr if arr.ndim == 2 else arr[None])
            text_dim = int(prompt.shape[-1])
        else:
            prompt = torch.randn(num_prompts, text_dim) * 0.02
        self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
        self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim,
                                                 dropout, l2_norm=False)
        self.text_to_ch = nn.Sequential(nn.Linear(text_code_dim, channels),
                                        nn.LayerNorm(channels), nn.GELU())
        self.ch_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        mid = max(channels // 4, 16)
        self.sp_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, 1, 1), nn.Sigmoid(),
        )

    def forward(self, feat_map):
        B, T, C, H, W = feat_map.shape
        pos = self.text_to_ch(self.text_adapter(self.prompt_feat.to(feat_map.device)).mean(0))
        cg = self.ch_gate(pos).view(1, 1, C, 1, 1)
        x = feat_map * (1.0 + cg)
        sg = self.sp_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        return feat_map * (1.0 + cg) * (1.0 + sg), sg


# ================================================================
#  Encoders
# ================================================================

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

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B * T, C, H, W))
        return y.reshape(B, T, *y.shape[1:])


class PointCloudSimpleEncoder(nn.Module):
    def __init__(self, out_channels=512, parts_num=16, in_dim=3):
        super().__init__()
        self.P, self.C, self.D = int(parts_num), int(out_channels), int(in_dim)
        self.mlp = nn.Sequential(nn.Linear(self.D, 64), nn.GELU(), nn.Linear(64, 128),
                                 nn.GELU(), nn.Linear(128, out_channels), nn.GELU())
        self.norm = nn.LayerNorm(out_channels)

    def _fit(self, pc):
        d = pc.size(-1)
        if d == self.D: return pc
        if d > self.D: return pc[..., :self.D]
        return torch.cat([pc, pc.new_zeros(*pc.shape[:-1], self.D - d)], -1)

    def forward(self, pc):
        B, T, N, _ = pc.shape
        pc = self._fit(pc.float())
        z = (pc[..., :min(3, self.D)])[..., -1]
        zn = (z - z.min(-1, True).values) / (z.max(-1, True).values - z.min(-1, True).values).clamp_min(1e-6)
        feat = self.norm(self.mlp(pc.reshape(-1, self.D)).reshape(B, T, N, self.C))
        parts = []
        for p in range(self.P):
            lo, hi = p / self.P, (p + 1) / self.P
            m = ((zn >= lo) & (zn < hi if p < self.P - 1 else zn <= hi)).float().unsqueeze(-1)
            parts.append((feat * m).sum(2) / m.sum(2).clamp_min(1.0))
        return torch.stack(parts, -1)


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


# ================================================================
#  Nuisance removal
# ================================================================

class StagedNuisanceRemoval(nn.Module):
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
        self.grl_lambda, self.temperature = float(grl_lambda), float(temperature)
        self.warmup_iters, self.probe_iters = int(warmup_iters), int(probe_iters)
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
        self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim, dropout, True)
        self.txt_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim)) for k in self.factors})
        self.visual_probe = nn.ModuleDict({k: nn.Sequential(nn.Linear(in_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim)) for k in self.factors})
        self.risk_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(sub_dim, sub_dim // 2), nn.GELU(), nn.Linear(sub_dim // 2, 1)) for k in self.factors})
        self.part_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(sub_dim, sub_dim), nn.GELU(), nn.Linear(sub_dim, self.parts_num)) for k in self.factors})

    def schedule(self, step):
        ps = self.warmup_iters
        ad = self.warmup_iters + self.probe_iters
        return {'probe': ramp_value(step, ps, max(1, self.probe_iters)),
                'adv': ramp_value(step, ad, max(1, self.adv_ramp_iters)),
                'rel': ramp_value(step, self.reliability_start_iter, max(1, self.reliability_ramp_iters))}

    def _tmask(self, masks, f):
        if masks is None or f not in masks or masks[f] is None: return None
        m = masks[f].float()
        return m.view(m.size(0), -1).mean(-1).clamp(0, 1) if m.dim() > 1 else m.clamp(0, 1)

    def _tz(self, text_feats, f):
        if text_feats is None or f not in text_feats or text_feats[f] is None: return None
        return l2n(self.txt_heads[f](self.text_adapter(text_feats[f])))

    def pre_temporal(self, vis_g, text_feats=None, masks=None, step=0):
        sc = self.schedule(step)
        B, dev = vis_g.size(0), vis_g.device
        losses, bad = {}, vis_g.new_zeros(B, self.parts_num)
        for f in self.factors:
            zp = l2n(self.visual_probe[f](vis_g.detach().float()))
            tgt, zt = self._tmask(masks, f), self._tz(text_feats, f)
            if zt is not None and sc['probe'] > 0 and self.probe_loss_weight > 0:
                losses[f'probe_{f}'] = masked_symmetric_infonce(zp, zt, tgt, self.temperature) * self.probe_loss_weight * sc['probe']
            risk = torch.sigmoid(self.risk_heads[f](zp)).squeeze(-1)
            if tgt is not None and sc['probe'] > 0 and self.mask_loss_weight > 0:
                losses[f'risk_{f}'] = F.mse_loss(risk, tgt.float()) * self.mask_loss_weight * sc['probe']
            pb = torch.sigmoid(self.part_heads[f](zp))
            bad = bad + float(self.factor_weights.get(f, 1.0)) * risk.unsqueeze(-1) * pb
        if sc['rel'] <= 0:
            rel = vis_g.new_ones(B, 1, 1, self.parts_num)
        else:
            bad = (bad / max(1.0, sum(abs(self.factor_weights.get(f, 1.0)) for f in self.factors))).clamp(0, 1)
            if self.detach_reliability_gate: bad = bad.detach()
            rel = (1.0 - self.max_suppression * sc['rel'] * bad).clamp(1.0 - self.max_suppression, 1.0).view(B, 1, 1, self.parts_num)
        diag = {k: vis_g.new_tensor(v) for k, v in sc.items()}
        diag['mean_reliability'] = rel.mean().detach()
        return rel, losses, diag

    def post_temporal(self, id_g, text_feats=None, masks=None, step=0):
        sc = self.schedule(step)
        losses = {}
        if sc['adv'] <= 0 or self.adv_loss_weight <= 0: return losses
        lam = self.grl_lambda * sc['adv']
        ir = grad_reverse(id_g.float(), lam)
        for f in self.factors:
            zt = self._tz(text_feats, f)
            if zt is None: continue
            tgt = self._tmask(masks, f)
            zi = l2n(self.visual_probe[f](ir))
            losses[f'adv_{f}'] = masked_symmetric_infonce(zi, zt, tgt, self.temperature) * self.adv_loss_weight * sc['adv']
        return losses


# ================================================================
#  MoE — Part experts  [Fix-2: stronger experts]
# ================================================================

class PartExpertBlock(nn.Module):
    """Lightweight expert with learned part-attention + LoRA residual.

    Fix-2: part_logit init 2.0 (sigmoid~0.88), scale init 0.5.
    """
    def __init__(self, dim, parts_num, rank=16, dropout=0.0):
        super().__init__()
        self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
        self.transform = nn.Sequential(
            nn.Linear(dim, rank, bias=False), nn.GELU(), nn.Linear(rank, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """x: [N, C, P] -> [N, C, P]"""
        pa = torch.sigmoid(self.part_logit)
        h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)       # [N, P, C]
        delta = self.drop(self.transform(self.norm(h)))
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()


class TopKMoELayer(nn.Module):
    """Sample/sequence-level top-k MoE.

    Fix-1: accepts ``seq_len`` for sequence-level routing.
    Fix-2: learnable residual gate instead of hard ``output + x``.
    """
    def __init__(self, dim, parts_num, num_experts=8, top_k=2,
                 expert_rank=16, dropout=0.0):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList([
            PartExpertBlock(dim, parts_num, expert_rank, dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(True), nn.Linear(dim // 2, num_experts),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.7))

    def forward(self, x, seq_len=1):
        """x: [N, C, P]; seq_len: frames per sample (1 = already pooled).
        Returns (output, logits_B, weights_B).
        """
        N, C, P = x.shape

        # --- routing at sequence level ---
        if seq_len > 1:
            B = N // seq_len
            r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))        # [B, C]
        else:
            B = N
            r_in = x.mean(dim=-1)                                      # [B, C]
        logits = self.router(r_in)                                      # [B, E]
        _, topk_idx = logits.topk(self.top_k, dim=-1)
        mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights_B = F.softmax(logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

        # expand to frame level
        if seq_len > 1:
            mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mask_N, wgt_N = mask_B, weights_B

        # --- expert computation ---
        expert_out = torch.zeros_like(x)
        for i in range(self.num_experts):
            sel = mask_N[:, i] > 0
            if not sel.any():
                continue
            e = self.experts[i](x[sel]).to(dtype=expert_out.dtype)
            w = wgt_N[sel, i].to(dtype=expert_out.dtype).unsqueeze(-1).unsqueeze(-1)
            expert_out[sel] = expert_out[sel] + w * e

        alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
        output = alpha * x + (1.0 - alpha) * expert_out
        return output, logits, weights_B


# ================================================================
#  Condition encoder  [LEGACY — kept for backward compatibility]
# ================================================================

class ConditionEncoder(nn.Module):
    def __init__(self, stats_dim=6, cond_dim=32, hidden_dim=64,
                 num_illum_bins=3, num_dist_bins=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim), nn.ReLU(True),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, cond_dim),
        )
        self.illum_head = nn.Linear(cond_dim, num_illum_bins)
        self.dist_head = nn.Linear(cond_dim, num_dist_bins)

    def forward(self, stats):
        c = self.net(stats)
        return c, self.illum_head(c), self.dist_head(c)

    @staticmethod
    @torch.no_grad()
    def pseudo_labels(stats, n=3):
        B = stats.size(0)
        if B < n:
            return stats.new_zeros(B, dtype=torch.long), stats.new_zeros(B, dtype=torch.long)
        def _b(x):
            s = x.sort().values
            bd = [s[(len(s) * i) // n].item() for i in range(1, n)]
            return torch.bucketize(x.contiguous(), torch.tensor(bd, device=x.device, dtype=x.dtype))
        return _b(stats[:, 0]), _b(stats[:, 3])


# ================================================================
#  Modality Reliability Probe  [NEW — replaces ConditionEncoder]
# ================================================================

class ModalityReliabilityProbe(nn.Module):
    """End-to-end learnable probe that assesses modality quality from encoded
    features, replacing the hand-crafted ``compute_condition_stats`` +
    ``ConditionEncoder`` pipeline.

    Inputs:  rgb_feat [B, C, P] and pc_feat [B, C, P]  (post-temporal, post-projection)
    Outputs:
        cond_emb     [B, cond_dim]  — scene condition embedding for MoE routing
        mod_weights  [B, 2]         — soft reliability weights (rgb, pc)
        diag         dict           — diagnostic scalars for TensorBoard
    """

    def __init__(self, channels, cond_dim=32, hidden_dim=64):
        super().__init__()
        stat_dim = channels * 3  # mean + std + range per channel

        # Per-modality feature probes
        self.rgb_probe = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.pc_probe = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim * 2, bias=False),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )

        # Cross-modal interaction  (+1 dim for cross-modal cosine similarity)
        self.cross_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.GELU(), nn.LayerNorm(hidden_dim),
        )

        # Output heads
        self.cond_head = nn.Linear(hidden_dim, cond_dim)
        self.weight_head = nn.Linear(hidden_dim, 2)

    def _feat_stats(self, feat):
        """Extract differentiable per-channel statistics from [B, C, P].

        Returns [B, 3C]:
          - mean over parts   : overall activation strength
          - std over parts    : part discriminability (high = informative)
          - range over parts  : dynamic range (sensitive to occlusion)
        """
        feat = feat.float()
        return torch.cat([
            feat.mean(dim=-1),                              # [B, C]
            feat.std(dim=-1).clamp_min(1e-6),               # [B, C]
            feat.amax(dim=-1) - feat.amin(dim=-1),          # [B, C]
        ], dim=-1)                                           # [B, 3C]

    def forward(self, rgb_feat, pc_feat, temperature=1.0):
        """
        Args:
            rgb_feat: [B, C, P] — projected RGB features (post-temporal)
            pc_feat:  [B, C, P] — projected PC  features (post-temporal)
            temperature: float  — softmax temperature for weight sharpness
        Returns:
            cond_emb:    [B, cond_dim]
            mod_weights: [B, 2]   (column 0 = rgb, column 1 = pc)
            diag:        dict of detached diagnostic scalars
        """
        # 1) Per-modality probing
        rgb_s = self.rgb_probe(self._feat_stats(rgb_feat))       # [B, H]
        pc_s = self.pc_probe(self._feat_stats(pc_feat))          # [B, H]

        # 2) Cross-modal consistency signal
        cross_sim = F.cosine_similarity(
            rgb_feat.float().mean(-1),
            pc_feat.float().mean(-1),
            dim=-1,
        ).unsqueeze(-1)                                           # [B, 1]

        # 3) Joint reasoning
        joint = self.cross_net(
            torch.cat([rgb_s, pc_s, cross_sim], dim=-1)
        )                                                         # [B, H]

        # 4) Outputs
        cond_emb = self.cond_head(joint)                          # [B, cond_dim]
        mod_weights = F.softmax(
            self.weight_head(joint) / max(temperature, 1e-3),
            dim=-1,
        )                                                         # [B, 2]

        diag = {
            'mod_rgb_w': mod_weights[:, 0].mean().detach(),
            'mod_pc_w': mod_weights[:, 1].mean().detach(),
            'mod_cross_sim': cross_sim.mean().detach(),
        }
        return cond_emb, mod_weights, diag


# ================================================================
#  Fusion MoE  [Fix-3: per-expert independent low-rank attention]
# ================================================================

class FusionExpertBlock(nn.Module):
    """Single fusion expert with independent low-rank cross-attention."""
    def __init__(self, dim, output_parts, num_input_tokens, rank=32):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.token_logit = nn.Parameter(torch.zeros(num_input_tokens))
        # Independent low-rank attention
        self.q_proj = nn.Linear(dim, rank, bias=False)
        self.k_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.attn_scale = rank ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        """tokens: [N, 2P, C] -> [N, P, C]"""
        N = tokens.size(0)
        ta = torch.sigmoid(self.token_logit)
        kv = self.norm_kv(tokens) * ta.unsqueeze(0).unsqueeze(-1)
        q = self.norm_q(self.queries.unsqueeze(0).expand(N, -1, -1))
        Q, K = self.q_proj(q), self.k_proj(kv)
        V = self.v_proj(kv)
        attn = (Q @ K.transpose(-1, -2)) * self.attn_scale
        out = self.o_proj(attn.softmax(-1) @ V)
        return out + self.ffn(self.ffn_norm(out))


class FusionMoELayer(nn.Module):
    """Token-level fusion MoE with condition-aware routing.

    Fix-3: each expert has its own low-rank attention (no shared weights).
    Supports ``seq_len`` for V2 sequence-level routing.
    """
    def __init__(self, dim, output_parts, num_input_tokens, num_experts=8,
                 top_k=2, cond_dim=32, expert_rank=32, dropout=0.0):
        super().__init__()
        self.num_experts, self.top_k, self.output_parts = num_experts, top_k, output_parts
        self.experts = nn.ModuleList([
            FusionExpertBlock(dim, output_parts, num_input_tokens, rank=expert_rank)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim + cond_dim, dim // 2), nn.ReLU(True),
            nn.Linear(dim // 2, num_experts),
        )
        self.residual_proj = nn.Sequential(
            nn.Linear(num_input_tokens * dim, output_parts * dim),
        )  # linear shortcut from 2P tokens to P tokens
        self.residual_gate = nn.Parameter(torch.tensor(0.6))

    def forward(self, tokens, cond_emb, seq_len=1):
        """tokens [N,2P,C], cond_emb [B,cond] -> (out [N,C,P], logits, weights)"""
        N, TK, Cd = tokens.shape  # TK = 2P
        if seq_len > 1:
            B = N // seq_len
            fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
            if cond_emb is None:
                cond_for_router = fp.new_zeros(B, 0)
            elif cond_emb.size(0) == N:
                cond_for_router = cond_emb.reshape(B, seq_len, -1).mean(dim=1)
            elif cond_emb.size(0) == B:
                cond_for_router = cond_emb
            else:
                raise RuntimeError(
                    f"cond_emb batch mismatch: got {cond_emb.size(0)}, expected B={B} or N={N}"
                )
        else:
            B = N
            fp = tokens.mean(dim=1)
            if cond_emb is None:
                cond_for_router = fp.new_zeros(B, 0)
            elif cond_emb.size(0) == B:
                cond_for_router = cond_emb
            else:
                raise RuntimeError(
                    f"cond_emb batch mismatch: got {cond_emb.size(0)}, expected B={B}"
                )
        cond_for_router = cond_for_router.to(device=fp.device, dtype=fp.dtype)
        ri = torch.cat([fp, cond_for_router], -1)
        logits = self.router(ri)
        _, ti = logits.topk(self.top_k, -1)
        mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
        wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), -1)
        if seq_len > 1:
            mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mN, wN = mB, wB

        out = torch.zeros(N, self.output_parts, Cd, device=tokens.device, dtype=tokens.dtype)
        for i in range(self.num_experts):
            sel = mN[:, i] > 0
            if not sel.any(): continue
            eo = self.experts[i](tokens[sel]).to(dtype=out.dtype)
            w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
            out[sel] = out[sel] + w * eo

        # Learnable residual shortcut (flatten 2P -> P via linear)
        alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
        res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, self.output_parts, Cd)
        res = res.to(dtype=out.dtype)
        out = alpha * res + (1.0 - alpha) * out
        return out.permute(0, 2, 1).contiguous(), logits, wB


class CrossAttentionFusion(nn.Module):
    """Fallback (no MoE)."""
    def __init__(self, dim, output_parts, num_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens, cond_emb=None, seq_len=1):
        N = tokens.size(0)
        q = self.q.unsqueeze(0).expand(N, -1, -1)
        o, _ = self.attn(q, self.norm(tokens), self.norm(tokens))
        return (o + self.ffn(o)).permute(0, 2, 1).contiguous(), None, None


class MLPFusion(nn.Module):
    """Fallback (no MoE)."""
    def __init__(self, dim, output_parts, num_input_tokens):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.P = output_parts

    def forward(self, tokens, cond_emb=None, seq_len=1):
        rgb_t, pc_t = tokens[:, :self.P], tokens[:, self.P:]
        return self.proj(torch.cat([rgb_t, pc_t], -1)).permute(0, 2, 1).contiguous(), None, None

class EarlyPartTokenFusion(nn.Module):
    """Fuse RGB and anchor part tokens before temporal modeling."""

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


# ================================================================
#  Temporal encoder
# ================================================================

class CycleAwareTemporalEncoder(nn.Module):
    def __init__(self, channels=512, parts_num=16, heads=4, layers=1,
                 dropout=0.1, future_pred=True):
        super().__init__()
        self.future_pred = bool(future_pred)
        enc = nn.TransformerEncoderLayer(channels, heads, channels * 4, dropout,
                                         batch_first=True, activation='gelu', norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, layers)
        self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.attn = nn.Sequential(nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(),
                                  nn.Linear(max(channels // 4, 16), 1))
        if future_pred:
            self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, part_seq):
        B, T, C, P = part_seq.shape
        x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
        xt = self.temporal(x)
        d = torch.zeros_like(x)
        if T > 1: d[:, 1:] = x[:, 1:] - x[:, :-1]
        d = self.diff_proj(d)
        if T > 1:
            fr = torch.fft.rfft(x.float(), dim=1).abs()
            fr = fr[:, 1:min(fr.size(1), 4)].mean(1) if fr.size(1) > 1 else fr.mean(1)
        else:
            fr = x.float().mean(1)
        fr = self.freq_proj(fr).unsqueeze(1).expand(-1, T, -1)
        y = xt + d + fr
        a = torch.softmax(self.attn(y).squeeze(-1), 1)
        pooled = (y * a.unsqueeze(-1)).sum(1).reshape(B, P, C).permute(0, 2, 1).contiguous()
        fl = pooled.new_tensor(0.0)
        if self.future_pred and T > 2:
            fl = F.mse_loss(self.future_head(y[:, :-1]), x[:, 1:].detach())
        return pooled, fl, a.reshape(B, P, T).permute(0, 2, 1).contiguous()


# ================================================================
#  Verifier
# ================================================================

class RuleBasedVerifier(nn.Module):
    def __init__(self, attention_weight=0.0):
        super().__init__()
        self.aw = float(attention_weight)

    def forward(self, ga, rgb_seq=None):
        if self.aw <= 0 or ga is None: return ga.new_tensor(0.0)
        B, T, _, Hf, Wf = ga.shape
        if rgb_seq is None or rgb_seq.dim() != 5 or T <= 1: return ga.new_tensor(0.0)
        g = rgb_seq.float().mean(2, True)
        d = torch.zeros_like(g); d[:, 1:] = (g[:, 1:] - g[:, :-1]).abs()
        tgt = F.interpolate(d.reshape(B * T, 1, d.shape[-2], d.shape[-1]),
                            (Hf, Wf), mode='bilinear', align_corners=False).reshape(B, T, 1, Hf, Wf)
        tgt = tgt / tgt.amax((-1, -2), True).clamp_min(1e-6)
        with torch.cuda.amp.autocast(False):
            return F.binary_cross_entropy(ga.float().clamp(1e-6, 1 - 1e-6), tgt.float()) * self.aw


# ================================================================
#  MoE loss helpers
# ================================================================

def moe_balance_loss(logits):
    if logits is None or logits.dim() != 2: return torch.tensor(0.0)
    E = logits.size(1)
    p = F.softmax(logits, -1).mean(0)
    f = torch.zeros(E, device=logits.device)
    top1 = logits.argmax(-1)
    for i in range(E): f[i] = (top1 == i).float().mean()
    return E * (f * p).sum()


def moe_diversity_loss(experts, key='part_logit', thr=0.5):
    bs = [torch.sigmoid(getattr(e, key)) for e in experts if hasattr(e, key)]
    if len(bs) < 2: return torch.tensor(0.0, device=bs[0].device) if bs else torch.tensor(0.0)
    s = F.normalize(torch.stack(bs), dim=-1)
    sim = s @ s.t()
    E = sim.size(0)
    return F.relu(sim[~torch.eye(E, dtype=torch.bool, device=sim.device)] - thr).mean()


def moe_token_diversity_loss(experts, thr=0.5):
    return moe_diversity_loss(experts, 'token_logit', thr)


def probe_balance_loss(mod_weights):
    """Prevent modality weight collapse to a single modality.

    Penalizes batch-mean weights deviating from uniform (0.5, 0.5).
    Args:
        mod_weights: [B, 2] — soft modality reliability weights.
    Returns:
        scalar loss.
    """
    mean_w = mod_weights.mean(0)        # [2]
    return (mean_w - 0.5).pow(2).sum()


# ================================================================
#  Condition stats helper  [LEGACY — kept for backward compatibility]
# ================================================================

@torch.no_grad()
def compute_condition_stats(rgb, pc):
    """6-dim: [brightness, contrast, edge, distance, density, spread]"""
    r = rgb.float()
    bri = r.mean((1, 2, 3, 4))
    con = r.std((1, 2, 3, 4), unbiased=False)
    dh = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().mean((1, 2, 3, 4))
    dw = (r[:, :, :, :, 1:] - r[:, :, :, :, :-1]).abs().mean((1, 2, 3, 4))
    edge = 0.5 * (dh + dw)
    p = pc.float()[..., :3]
    p = torch.nan_to_num(p, 0., 0., 0.)
    dist = p.norm(dim=-1).mean((1, 2))
    dens = torch.isfinite(pc.float()[..., :3]).all(-1).float().mean((1, 2))
    sprd = p.std(2, unbiased=False).mean((1, 2))
    return torch.nan_to_num(torch.stack([bri, con, edge, dist, dens, sprd], 1), 0., 0., 0.)





# """GaitMoEOracle shared building blocks (V2 & V3).

# All four fixes applied:
#   Fix-1: Sequence-level routing (TopKMoELayer accepts seq_len)
#   Fix-2: Stronger experts (part_logit=2.0, scale=0.5, learnable residual gate)
#   Fix-3: Per-expert independent low-rank attention in FusionMoE
#   Fix-4: proj_dim defaults to channels (no premature bottleneck)
# """

# import os
# import math
# from typing import Dict, Optional, List

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# # ================================================================
# #  Utilities
# # ================================================================

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
#     return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# def _scalar(x, v=0.0):
#     return x.new_tensor(float(v))


# def symmetric_infonce(z1, z2, temperature=0.07):
#     if z1 is None or z2 is None or z1.size(0) <= 1:
#         return _scalar(z1 if z1 is not None else z2)
#     z1, z2 = l2n(z1.float()), l2n(z2.float())
#     logits = z1 @ z2.t() / temperature
#     lbl = torch.arange(logits.size(0), device=logits.device)
#     return 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))


# def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
#     if z_img is None or z_txt is None:
#         return _scalar(z_img if z_img is not None else z_txt)
#     if mask is not None:
#         mask = mask.float().view(mask.size(0), -1).mean(-1)
#         valid = mask > 0
#         if valid.sum() <= 1:
#             return z_img.new_tensor(0.0)
#         z_img, z_txt = z_img[valid], z_txt[valid]
#         weights = mask[valid].clamp(0, 1)
#     else:
#         if z_img.size(0) <= 1:
#             return z_img.new_tensor(0.0)
#         weights = None
#     z_img, z_txt = l2n(z_img.float()), l2n(z_txt.float())
#     logits = z_img @ z_txt.t() / temperature
#     lbl = torch.arange(logits.size(0), device=logits.device)
#     if weights is None:
#         return 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
#     li = F.cross_entropy(logits, lbl, reduction='none')
#     lt = F.cross_entropy(logits.t(), lbl, reduction='none')
#     w = weights / weights.mean().clamp_min(1e-6)
#     return 0.5 * ((li * w).mean() + (lt * w).mean())


# def ramp_value(step, start, length, max_val=1.0):
#     if length <= 0:
#         return max_val if step >= start else 0.0
#     if step < start:
#         return 0.0
#     return float(max_val) * min(1.0, (step - start) / float(length))


# # ================================================================
# #  Text modules
# # ================================================================

# class TextEmbeddingAdapter(nn.Module):
#     def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024,
#                  dropout=0.1, l2_norm=True):
#         super().__init__()
#         self.input_dim, self.code_dim, self.l2_norm = int(input_dim), int(code_dim), bool(l2_norm)
#         hid = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
#         self.in_norm = nn.LayerNorm(self.input_dim)
#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, hid, bias=False), nn.GELU(), nn.Dropout(dropout),
#             nn.Linear(hid, self.code_dim, bias=False), nn.LayerNorm(self.code_dim),
#         )

#     def _fit(self, x):
#         d = x.size(-1)
#         if d == self.input_dim:
#             return x
#         if d > self.input_dim:
#             return x[..., :self.input_dim]
#         return torch.cat([x, x.new_zeros(*x.shape[:-1], self.input_dim - d)], -1)

#     def forward(self, x):
#         if x is None:
#             return None
#         if x.dim() == 3:
#             x = x.mean(1)
#         elif x.dim() > 3:
#             x = x.view(x.size(0), -1)
#         z = self.net(self.in_norm(self._fit(x.float())))
#         return l2n(z) if self.l2_norm else z


# class PositiveTextGate(nn.Module):
#     def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
#                  trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024,
#                  dropout=0.1):
#         super().__init__()
#         self.channels = channels
#         if prompt_feat_path and os.path.exists(prompt_feat_path):
#             arr = np.load(prompt_feat_path).astype('float32')
#             prompt = torch.from_numpy(arr if arr.ndim == 2 else arr[None])
#             text_dim = int(prompt.shape[-1])
#         else:
#             prompt = torch.randn(num_prompts, text_dim) * 0.02
#         self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
#         self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim,
#                                                  dropout, l2_norm=False)
#         self.text_to_ch = nn.Sequential(nn.Linear(text_code_dim, channels),
#                                         nn.LayerNorm(channels), nn.GELU())
#         self.ch_gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
#         mid = max(channels // 4, 16)
#         self.sp_gate = nn.Sequential(
#             nn.Conv2d(channels, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.GELU(),
#             nn.Conv2d(mid, 1, 1), nn.Sigmoid(),
#         )

#     def forward(self, feat_map):
#         B, T, C, H, W = feat_map.shape
#         pos = self.text_to_ch(self.text_adapter(self.prompt_feat.to(feat_map.device)).mean(0))
#         cg = self.ch_gate(pos).view(1, 1, C, 1, 1)
#         x = feat_map * (1.0 + cg)
#         sg = self.sp_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
#         return feat_map * (1.0 + cg) * (1.0 + sg), sg


# # ================================================================
# #  Encoders
# # ================================================================

# class TinyDenseRGBEncoder(nn.Module):
#     def __init__(self, out_channels=512, in_channels=3):
#         super().__init__()
#         c1, c2, c3 = max(out_channels // 8, 32), max(out_channels // 4, 64), max(out_channels // 2, 128)
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, c1, 3, 1, 1, bias=False), nn.BatchNorm2d(c1), nn.GELU(),
#             nn.Conv2d(c1, c2, 3, 2, 1, bias=False), nn.BatchNorm2d(c2), nn.GELU(),
#             nn.Conv2d(c2, c3, 3, 2, 1, bias=False), nn.BatchNorm2d(c3), nn.GELU(),
#             nn.Conv2d(c3, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
#         )

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         y = self.net(x.reshape(B * T, C, H, W))
#         return y.reshape(B, T, *y.shape[1:])


# class PointCloudSimpleEncoder(nn.Module):
#     def __init__(self, out_channels=512, parts_num=16, in_dim=3):
#         super().__init__()
#         self.P, self.C, self.D = int(parts_num), int(out_channels), int(in_dim)
#         self.mlp = nn.Sequential(nn.Linear(self.D, 64), nn.GELU(), nn.Linear(64, 128),
#                                  nn.GELU(), nn.Linear(128, out_channels), nn.GELU())
#         self.norm = nn.LayerNorm(out_channels)

#     def _fit(self, pc):
#         d = pc.size(-1)
#         if d == self.D: return pc
#         if d > self.D: return pc[..., :self.D]
#         return torch.cat([pc, pc.new_zeros(*pc.shape[:-1], self.D - d)], -1)

#     def forward(self, pc):
#         B, T, N, _ = pc.shape
#         pc = self._fit(pc.float())
#         z = (pc[..., :min(3, self.D)])[..., -1]
#         zn = (z - z.min(-1, True).values) / (z.max(-1, True).values - z.min(-1, True).values).clamp_min(1e-6)
#         feat = self.norm(self.mlp(pc.reshape(-1, self.D)).reshape(B, T, N, self.C))
#         parts = []
#         for p in range(self.P):
#             lo, hi = p / self.P, (p + 1) / self.P
#             m = ((zn >= lo) & (zn < hi if p < self.P - 1 else zn <= hi)).float().unsqueeze(-1)
#             parts.append((feat * m).sum(2) / m.sum(2).clamp_min(1.0))
#         return torch.stack(parts, -1)


# class LidarGaitPPEncoder(nn.Module):
#     def __init__(self, channel=16, npoints=None, nsample=8, sampling='knn',
#                  scale_aware=True, normalize_dp=True, out_channels=256, parts_num=16):
#         super().__init__()
#         from .lidargaitv2_utils import PointNetSetAbstraction, PPPooling
#         npoints = npoints or [256, 128, 64]
#         ic = 4 if scale_aware else 3
#         C = channel
#         self.sa1 = PointNetSetAbstraction(npoints[0], 0.1, nsample, ic, [2*C, 2*C, 4*C], False, sampling, scale_aware, normalize_dp)
#         self.sa2 = PointNetSetAbstraction(npoints[1], 0.2, nsample, 4*C+ic, [4*C, 4*C, 8*C], False, sampling, scale_aware, normalize_dp)
#         self.sa3 = PointNetSetAbstraction(npoints[2], 0.4, nsample, 8*C+ic, [8*C, 8*C, 16*C], False, sampling, scale_aware, normalize_dp)
#         self.sa4 = PointNetSetAbstraction(None, None, None, 16*C+ic, [16*C, 16*C, out_channels], True, sampling, scale_aware, normalize_dp)
#         self.pool = PPPooling(scale_aware=scale_aware, bin_num=[parts_num])

#     def forward(self, pc):
#         B, T, N, C = pc.shape
#         xyz = rearrange(pc, 'B T N C -> (B T) C N')
#         l1x, l1p = self.sa1(xyz, None); l1p = l1p.max(-2)[0]
#         l2x, l2p = self.sa2(l1x, l1p); l2p = l2p.max(-2)[0]
#         l3x, l3p = self.sa3(l2x, l2p); l3p = l3p.max(-2)[0]
#         l4x, l4p = self.sa4(l3x, l3p)
#         x = self.pool(l4p, l3x)
#         return rearrange(x, '(B T) C P -> B T C P', B=B)


# # ================================================================
# #  Nuisance removal
# # ================================================================

# class StagedNuisanceRemoval(nn.Module):
#     def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256,
#                  text_hidden_dim=1024, sub_dim=256, factors=None,
#                  grl_lambda=0.08, temperature=0.07,
#                  warmup_iters=5000, probe_iters=10000, adv_ramp_iters=10000,
#                  reliability_start_iter=15000, reliability_ramp_iters=10000,
#                  probe_loss_weight=0.03, adv_loss_weight=0.03, mask_loss_weight=0.01,
#                  max_suppression=0.35, detach_reliability_gate=True,
#                  factor_weights=None, dropout=0.1):
#         super().__init__()
#         self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
#         self.parts_num = int(parts_num)
#         self.grl_lambda, self.temperature = float(grl_lambda), float(temperature)
#         self.warmup_iters, self.probe_iters = int(warmup_iters), int(probe_iters)
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
#         self.text_adapter = TextEmbeddingAdapter(text_dim, text_code_dim, text_hidden_dim, dropout, True)
#         self.txt_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim)) for k in self.factors})
#         self.visual_probe = nn.ModuleDict({k: nn.Sequential(nn.Linear(in_dim, sub_dim), nn.LayerNorm(sub_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(sub_dim, sub_dim), nn.LayerNorm(sub_dim)) for k in self.factors})
#         self.risk_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(sub_dim, sub_dim // 2), nn.GELU(), nn.Linear(sub_dim // 2, 1)) for k in self.factors})
#         self.part_heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(sub_dim, sub_dim), nn.GELU(), nn.Linear(sub_dim, self.parts_num)) for k in self.factors})

#     def schedule(self, step):
#         ps = self.warmup_iters
#         ad = self.warmup_iters + self.probe_iters
#         return {'probe': ramp_value(step, ps, max(1, self.probe_iters)),
#                 'adv': ramp_value(step, ad, max(1, self.adv_ramp_iters)),
#                 'rel': ramp_value(step, self.reliability_start_iter, max(1, self.reliability_ramp_iters))}

#     def _tmask(self, masks, f):
#         if masks is None or f not in masks or masks[f] is None: return None
#         m = masks[f].float()
#         return m.view(m.size(0), -1).mean(-1).clamp(0, 1) if m.dim() > 1 else m.clamp(0, 1)

#     def _tz(self, text_feats, f):
#         if text_feats is None or f not in text_feats or text_feats[f] is None: return None
#         return l2n(self.txt_heads[f](self.text_adapter(text_feats[f])))

#     def pre_temporal(self, vis_g, text_feats=None, masks=None, step=0):
#         sc = self.schedule(step)
#         B, dev = vis_g.size(0), vis_g.device
#         losses, bad = {}, vis_g.new_zeros(B, self.parts_num)
#         for f in self.factors:
#             zp = l2n(self.visual_probe[f](vis_g.detach().float()))
#             tgt, zt = self._tmask(masks, f), self._tz(text_feats, f)
#             if zt is not None and sc['probe'] > 0 and self.probe_loss_weight > 0:
#                 losses[f'probe_{f}'] = masked_symmetric_infonce(zp, zt, tgt, self.temperature) * self.probe_loss_weight * sc['probe']
#             risk = torch.sigmoid(self.risk_heads[f](zp)).squeeze(-1)
#             if tgt is not None and sc['probe'] > 0 and self.mask_loss_weight > 0:
#                 losses[f'risk_{f}'] = F.mse_loss(risk, tgt.float()) * self.mask_loss_weight * sc['probe']
#             pb = torch.sigmoid(self.part_heads[f](zp))
#             bad = bad + float(self.factor_weights.get(f, 1.0)) * risk.unsqueeze(-1) * pb
#         if sc['rel'] <= 0:
#             rel = vis_g.new_ones(B, 1, 1, self.parts_num)
#         else:
#             bad = (bad / max(1.0, sum(abs(self.factor_weights.get(f, 1.0)) for f in self.factors))).clamp(0, 1)
#             if self.detach_reliability_gate: bad = bad.detach()
#             rel = (1.0 - self.max_suppression * sc['rel'] * bad).clamp(1.0 - self.max_suppression, 1.0).view(B, 1, 1, self.parts_num)
#         diag = {k: vis_g.new_tensor(v) for k, v in sc.items()}
#         diag['mean_reliability'] = rel.mean().detach()
#         return rel, losses, diag

#     def post_temporal(self, id_g, text_feats=None, masks=None, step=0):
#         sc = self.schedule(step)
#         losses = {}
#         if sc['adv'] <= 0 or self.adv_loss_weight <= 0: return losses
#         lam = self.grl_lambda * sc['adv']
#         ir = grad_reverse(id_g.float(), lam)
#         for f in self.factors:
#             zt = self._tz(text_feats, f)
#             if zt is None: continue
#             tgt = self._tmask(masks, f)
#             zi = l2n(self.visual_probe[f](ir))
#             losses[f'adv_{f}'] = masked_symmetric_infonce(zi, zt, tgt, self.temperature) * self.adv_loss_weight * sc['adv']
#         return losses


# # ================================================================
# #  MoE — Part experts  [Fix-2: stronger experts]
# # ================================================================

# class PartExpertBlock(nn.Module):
#     """Lightweight expert with learned part-attention + LoRA residual.

#     Fix-2: part_logit init 2.0 (sigmoid≈0.88), scale init 0.5.
#     """
#     def __init__(self, dim, parts_num, rank=16, dropout=0.0):
#         super().__init__()
#         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
#         self.transform = nn.Sequential(
#             nn.Linear(dim, rank, bias=False), nn.GELU(), nn.Linear(rank, dim, bias=False),
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.scale = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         """x: [N, C, P] -> [N, C, P]"""
#         pa = torch.sigmoid(self.part_logit)
#         h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)       # [N, P, C]
#         delta = self.drop(self.transform(self.norm(h)))
#         h = h + self.scale * delta
#         return h.permute(0, 2, 1).contiguous()


# class TopKMoELayer(nn.Module):
#     """Sample/sequence-level top-k MoE.

#     Fix-1: accepts ``seq_len`` for sequence-level routing.
#     Fix-2: learnable residual gate instead of hard ``output + x``.
#     """
#     def __init__(self, dim, parts_num, num_experts=8, top_k=2,
#                  expert_rank=16, dropout=0.0):
#         super().__init__()
#         self.num_experts, self.top_k = num_experts, top_k
#         self.experts = nn.ModuleList([
#             PartExpertBlock(dim, parts_num, expert_rank, dropout)
#             for _ in range(num_experts)
#         ])
#         self.router = nn.Sequential(
#             nn.Linear(dim, dim // 2), nn.ReLU(True), nn.Linear(dim // 2, num_experts),
#         )
#         self.residual_gate = nn.Parameter(torch.tensor(0.7))

#     def forward(self, x, seq_len=1):
#         """x: [N, C, P]; seq_len: frames per sample (1 = already pooled).
#         Returns (output, logits_B, weights_B).
#         """
#         N, C, P = x.shape

#         # --- routing at sequence level ---
#         if seq_len > 1:
#             B = N // seq_len
#             r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))        # [B, C]
#         else:
#             B = N
#             r_in = x.mean(dim=-1)                                      # [B, C]
#         logits = self.router(r_in)                                      # [B, E]
#         _, topk_idx = logits.topk(self.top_k, dim=-1)
#         mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
#         weights_B = F.softmax(logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

#         # expand to frame level
#         if seq_len > 1:
#             mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#             wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#         else:
#             mask_N, wgt_N = mask_B, weights_B

#         # --- expert computation ---
#         expert_out = torch.zeros_like(x)
#         for i in range(self.num_experts):
#             sel = mask_N[:, i] > 0
#             if not sel.any():
#                 continue
#             e = self.experts[i](x[sel]).to(dtype=expert_out.dtype)
#             w = wgt_N[sel, i].to(dtype=expert_out.dtype).unsqueeze(-1).unsqueeze(-1)
#             expert_out[sel] = expert_out[sel] + w * e

#         alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
#         output = alpha * x + (1.0 - alpha) * expert_out
#         return output, logits, weights_B


# # ================================================================
# #  Condition encoder
# # ================================================================

# class ConditionEncoder(nn.Module):
#     def __init__(self, stats_dim=6, cond_dim=32, hidden_dim=64,
#                  num_illum_bins=3, num_dist_bins=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(stats_dim, hidden_dim), nn.ReLU(True),
#             nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, cond_dim),
#         )
#         self.illum_head = nn.Linear(cond_dim, num_illum_bins)
#         self.dist_head = nn.Linear(cond_dim, num_dist_bins)

#     def forward(self, stats):
#         c = self.net(stats)
#         return c, self.illum_head(c), self.dist_head(c)

#     @staticmethod
#     @torch.no_grad()
#     def pseudo_labels(stats, n=3):
#         B = stats.size(0)
#         if B < n:
#             return stats.new_zeros(B, dtype=torch.long), stats.new_zeros(B, dtype=torch.long)
#         def _b(x):
#             s = x.sort().values
#             bd = [s[(len(s) * i) // n].item() for i in range(1, n)]
#             return torch.bucketize(x.contiguous(), torch.tensor(bd, device=x.device, dtype=x.dtype))
#         return _b(stats[:, 0]), _b(stats[:, 3])


# # ================================================================
# #  Fusion MoE  [Fix-3: per-expert independent low-rank attention]
# # ================================================================

# class FusionExpertBlock(nn.Module):
#     """Single fusion expert with independent low-rank cross-attention."""
#     def __init__(self, dim, output_parts, num_input_tokens, rank=32):
#         super().__init__()
#         self.queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
#         self.token_logit = nn.Parameter(torch.zeros(num_input_tokens))
#         # Independent low-rank attention
#         self.q_proj = nn.Linear(dim, rank, bias=False)
#         self.k_proj = nn.Linear(dim, rank, bias=False)
#         self.v_proj = nn.Linear(dim, dim, bias=False)
#         self.o_proj = nn.Linear(dim, dim, bias=False)
#         self.attn_scale = rank ** -0.5
#         self.norm_q = nn.LayerNorm(dim)
#         self.norm_kv = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
#         self.ffn_norm = nn.LayerNorm(dim)

#     def forward(self, tokens):
#         """tokens: [N, 2P, C] -> [N, P, C]"""
#         N = tokens.size(0)
#         ta = torch.sigmoid(self.token_logit)
#         kv = self.norm_kv(tokens) * ta.unsqueeze(0).unsqueeze(-1)
#         q = self.norm_q(self.queries.unsqueeze(0).expand(N, -1, -1))
#         Q, K = self.q_proj(q), self.k_proj(kv)
#         V = self.v_proj(kv)
#         attn = (Q @ K.transpose(-1, -2)) * self.attn_scale
#         out = self.o_proj(attn.softmax(-1) @ V)
#         return out + self.ffn(self.ffn_norm(out))


# class FusionMoELayer(nn.Module):
#     """Token-level fusion MoE with condition-aware routing.

#     Fix-3: each expert has its own low-rank attention (no shared weights).
#     Supports ``seq_len`` for V2 sequence-level routing.
#     """
#     def __init__(self, dim, output_parts, num_input_tokens, num_experts=8,
#                  top_k=2, cond_dim=32, expert_rank=32, dropout=0.0):
#         super().__init__()
#         self.num_experts, self.top_k, self.output_parts = num_experts, top_k, output_parts
#         self.experts = nn.ModuleList([
#             FusionExpertBlock(dim, output_parts, num_input_tokens, rank=expert_rank)
#             for _ in range(num_experts)
#         ])
#         self.router = nn.Sequential(
#             nn.Linear(dim + cond_dim, dim // 2), nn.ReLU(True),
#             nn.Linear(dim // 2, num_experts),
#         )
#         self.residual_proj = nn.Sequential(
#             nn.Linear(num_input_tokens * dim, output_parts * dim),
#         )  # linear shortcut from 2P tokens to P tokens
#         self.residual_gate = nn.Parameter(torch.tensor(0.6))

#     def forward(self, tokens, cond_emb, seq_len=1):
#         """tokens [N,2P,C], cond_emb [B,cond] -> (out [N,C,P], logits, weights)"""
#         N, TK, Cd = tokens.shape  # TK = 2P
#         if seq_len > 1:
#             B = N // seq_len
#             fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
#             if cond_emb is None:
#                 cond_for_router = fp.new_zeros(B, 0)
#             elif cond_emb.size(0) == N:
#                 cond_for_router = cond_emb.reshape(B, seq_len, -1).mean(dim=1)
#             elif cond_emb.size(0) == B:
#                 cond_for_router = cond_emb
#             else:
#                 raise RuntimeError(
#                     f"cond_emb batch mismatch: got {cond_emb.size(0)}, expected B={B} or N={N}"
#                 )
#         else:
#             B = N
#             fp = tokens.mean(dim=1)
#             if cond_emb is None:
#                 cond_for_router = fp.new_zeros(B, 0)
#             elif cond_emb.size(0) == B:
#                 cond_for_router = cond_emb
#             else:
#                 raise RuntimeError(
#                     f"cond_emb batch mismatch: got {cond_emb.size(0)}, expected B={B}"
#                 )
#         cond_for_router = cond_for_router.to(device=fp.device, dtype=fp.dtype)
#         ri = torch.cat([fp, cond_for_router], -1)
#         logits = self.router(ri)
#         _, ti = logits.topk(self.top_k, -1)
#         mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
#         wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), -1)
#         if seq_len > 1:
#             mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#             wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#         else:
#             mN, wN = mB, wB

#         out = torch.zeros(N, self.output_parts, Cd, device=tokens.device, dtype=tokens.dtype)
#         for i in range(self.num_experts):
#             sel = mN[:, i] > 0
#             if not sel.any(): continue
#             eo = self.experts[i](tokens[sel]).to(dtype=out.dtype)
#             w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
#             out[sel] = out[sel] + w * eo

#         # Learnable residual shortcut (flatten 2P -> P via linear)
#         alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
#         res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, self.output_parts, Cd)
#         res = res.to(dtype=out.dtype)
#         out = alpha * res + (1.0 - alpha) * out
#         return out.permute(0, 2, 1).contiguous(), logits, wB


# class CrossAttentionFusion(nn.Module):
#     """Fallback (no MoE)."""
#     def __init__(self, dim, output_parts, num_heads=4, dropout=0.0):
#         super().__init__()
#         self.q = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
#         self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

#     def forward(self, tokens, cond_emb=None, seq_len=1):
#         N = tokens.size(0)
#         q = self.q.unsqueeze(0).expand(N, -1, -1)
#         o, _ = self.attn(q, self.norm(tokens), self.norm(tokens))
#         return (o + self.ffn(o)).permute(0, 2, 1).contiguous(), None, None


# class MLPFusion(nn.Module):
#     """Fallback (no MoE)."""
#     def __init__(self, dim, output_parts, num_input_tokens):
#         super().__init__()
#         self.proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
#         self.P = output_parts

#     def forward(self, tokens, cond_emb=None, seq_len=1):
#         rgb_t, pc_t = tokens[:, :self.P], tokens[:, self.P:]
#         return self.proj(torch.cat([rgb_t, pc_t], -1)).permute(0, 2, 1).contiguous(), None, None

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


# # ================================================================
# #  Temporal encoder
# # ================================================================

# class CycleAwareTemporalEncoder(nn.Module):
#     def __init__(self, channels=512, parts_num=16, heads=4, layers=1,
#                  dropout=0.1, future_pred=True):
#         super().__init__()
#         self.future_pred = bool(future_pred)
#         enc = nn.TransformerEncoderLayer(channels, heads, channels * 4, dropout,
#                                          batch_first=True, activation='gelu', norm_first=True)
#         self.temporal = nn.TransformerEncoder(enc, layers)
#         self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
#         self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
#         self.attn = nn.Sequential(nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(),
#                                   nn.Linear(max(channels // 4, 16), 1))
#         if future_pred:
#             self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

#     def forward(self, part_seq):
#         B, T, C, P = part_seq.shape
#         x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
#         xt = self.temporal(x)
#         d = torch.zeros_like(x)
#         if T > 1: d[:, 1:] = x[:, 1:] - x[:, :-1]
#         d = self.diff_proj(d)
#         if T > 1:
#             fr = torch.fft.rfft(x.float(), dim=1).abs()
#             fr = fr[:, 1:min(fr.size(1), 4)].mean(1) if fr.size(1) > 1 else fr.mean(1)
#         else:
#             fr = x.float().mean(1)
#         fr = self.freq_proj(fr).unsqueeze(1).expand(-1, T, -1)
#         y = xt + d + fr
#         a = torch.softmax(self.attn(y).squeeze(-1), 1)
#         pooled = (y * a.unsqueeze(-1)).sum(1).reshape(B, P, C).permute(0, 2, 1).contiguous()
#         fl = pooled.new_tensor(0.0)
#         if self.future_pred and T > 2:
#             fl = F.mse_loss(self.future_head(y[:, :-1]), x[:, 1:].detach())
#         return pooled, fl, a.reshape(B, P, T).permute(0, 2, 1).contiguous()


# # ================================================================
# #  Verifier
# # ================================================================

# class RuleBasedVerifier(nn.Module):
#     def __init__(self, attention_weight=0.0):
#         super().__init__()
#         self.aw = float(attention_weight)

#     def forward(self, ga, rgb_seq=None):
#         if self.aw <= 0 or ga is None: return ga.new_tensor(0.0)
#         B, T, _, Hf, Wf = ga.shape
#         if rgb_seq is None or rgb_seq.dim() != 5 or T <= 1: return ga.new_tensor(0.0)
#         g = rgb_seq.float().mean(2, True)
#         d = torch.zeros_like(g); d[:, 1:] = (g[:, 1:] - g[:, :-1]).abs()
#         tgt = F.interpolate(d.reshape(B * T, 1, d.shape[-2], d.shape[-1]),
#                             (Hf, Wf), mode='bilinear', align_corners=False).reshape(B, T, 1, Hf, Wf)
#         tgt = tgt / tgt.amax((-1, -2), True).clamp_min(1e-6)
#         with torch.cuda.amp.autocast(False):
#             return F.binary_cross_entropy(ga.float().clamp(1e-6, 1 - 1e-6), tgt.float()) * self.aw


# # ================================================================
# #  MoE loss helpers
# # ================================================================

# def moe_balance_loss(logits):
#     if logits is None or logits.dim() != 2: return torch.tensor(0.0)
#     E = logits.size(1)
#     p = F.softmax(logits, -1).mean(0)
#     f = torch.zeros(E, device=logits.device)
#     top1 = logits.argmax(-1)
#     for i in range(E): f[i] = (top1 == i).float().mean()
#     return E * (f * p).sum()


# def moe_diversity_loss(experts, key='part_logit', thr=0.5):
#     bs = [torch.sigmoid(getattr(e, key)) for e in experts if hasattr(e, key)]
#     if len(bs) < 2: return torch.tensor(0.0, device=bs[0].device) if bs else torch.tensor(0.0)
#     s = F.normalize(torch.stack(bs), dim=-1)
#     sim = s @ s.t()
#     E = sim.size(0)
#     return F.relu(sim[~torch.eye(E, dtype=torch.bool, device=sim.device)] - thr).mean()


# def moe_token_diversity_loss(experts, thr=0.5):
#     return moe_diversity_loss(experts, 'token_logit', thr)


# # ================================================================
# #  Condition stats helper
# # ================================================================

# @torch.no_grad()
# def compute_condition_stats(rgb, pc):
#     """6-dim: [brightness, contrast, edge, distance, density, spread]"""
#     r = rgb.float()
#     bri = r.mean((1, 2, 3, 4))
#     con = r.std((1, 2, 3, 4), unbiased=False)
#     dh = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().mean((1, 2, 3, 4))
#     dw = (r[:, :, :, :, 1:] - r[:, :, :, :, :-1]).abs().mean((1, 2, 3, 4))
#     edge = 0.5 * (dh + dw)
#     p = pc.float()[..., :3]
#     p = torch.nan_to_num(p, 0., 0., 0.)
#     dist = p.norm(dim=-1).mean((1, 2))
#     dens = torch.isfinite(pc.float()[..., :3]).all(-1).float().mean((1, 2))
#     sprd = p.std(2, unbiased=False).mean((1, 2))
#     return torch.nan_to_num(torch.stack([bri, con, edge, dist, dens, sprd], 1), 0., 0., 0.)
