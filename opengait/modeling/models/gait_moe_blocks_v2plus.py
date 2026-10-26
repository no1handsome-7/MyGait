"""GaitMoEOracle V2Plus — Enhanced building blocks.

Key improvements over V2 blocks:
  Fix-5: Learned Modality Quality Estimator (MQE) for condition-aware routing
  Fix-6: Enhanced ConditionEncoder with learned features + finer pseudo labels
  Fix-7: Quality-aware FusionMoE with modality reliability signals
  Fix-8: Stronger anti-collapse (z-loss, load-capping, usage tracking)
  Fix-9: Modality dropout for robust cross-modal learning
  Fix-10: Lower residual gate init for stronger expert contribution
"""

import os
import math
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Re-export unchanged blocks from original
from .gait_moe_blocks import (
    GradReverseFn, grad_reverse, l2n, _scalar,
    symmetric_infonce, masked_symmetric_infonce, ramp_value,
    TextEmbeddingAdapter,
    PositiveTextGate,
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    StagedNuisanceRemoval,
    PartExpertBlock,
    CrossAttentionFusion, MLPFusion,
    CycleAwareTemporalEncoder,
    RuleBasedVerifier,
    compute_condition_stats,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
)


# ================================================================
#  Fix-5: Modality Quality Estimator
# ================================================================

class ModalityQualityEstimator(nn.Module):
    """Learns per-modality quality scores from encoded features.

    Operates on global (spatially-pooled) features from each modality.
    Produces a scalar reliability score in [0, 1] for each modality.
    """

    def __init__(self, channels: int, hidden_dim: int = 128):
        super().__init__()
        self.rgb_head = nn.Sequential(
            nn.Linear(channels, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.pc_head = nn.Sequential(
            nn.Linear(channels, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Cross-modal comparator: compares both modalities jointly
        self.cross_head = nn.Sequential(
            nn.Linear(channels * 2, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),  # [rgb_quality_offset, pc_quality_offset]
        )

    def forward(self, rgb_global: torch.Tensor, pc_global: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_global: [B, C] global RGB features (mean-pooled over T, P)
            pc_global:  [B, C] global PC features
        Returns:
            rgb_quality: [B, 1] in (0, 1)
            pc_quality:  [B, 1] in (0, 1)
        """
        rq_base = self.rgb_head(rgb_global.float())      # [B, 1]
        pq_base = self.pc_head(pc_global.float())         # [B, 1]
        cross = self.cross_head(torch.cat([rgb_global, pc_global], -1).float())
        rq = torch.sigmoid(rq_base + cross[:, :1])
        pq = torch.sigmoid(pq_base + cross[:, 1:])
        return rq, pq


# ================================================================
#  Fix-6: Enhanced Condition Encoder
# ================================================================

class EnhancedConditionEncoder(nn.Module):
    """Condition encoder using both hand-crafted stats AND learned features.

    Improvements:
    - Receives learned modality quality scores as additional input
    - Uses learned global features from both modalities
    - Finer pseudo-label bins (5 instead of 3)
    - Explicit quality classification head
    - Larger hidden/output dimensions
    """

    def __init__(self, channels: int, cond_dim: int = 64, hidden_dim: int = 128,
                 stats_dim: int = 6, num_illum_bins: int = 5, num_dist_bins: int = 5):
        super().__init__()
        self.num_illum_bins = num_illum_bins
        self.num_dist_bins = num_dist_bins
        # Branch 1: hand-crafted statistics
        self.stats_net = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim // 2), nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        # Branch 2: learned global features + quality scores
        # Input: rgb_global [C] + pc_global [C] + rgb_quality [1] + pc_quality [1]
        feat_in = channels * 2 + 2
        self.feat_net = nn.Sequential(
            nn.Linear(feat_in, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
        )
        # Combine both branches
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, cond_dim),
        )
        # Classification heads for auxiliary supervision
        self.illum_head = nn.Linear(cond_dim, num_illum_bins)
        self.dist_head = nn.Linear(cond_dim, num_dist_bins)
        self.quality_head = nn.Linear(cond_dim, 3)  # low / medium / high

    def forward(self, stats, rgb_global, pc_global, rgb_quality, pc_quality):
        """
        Args:
            stats: [B, 6] hand-crafted stats from compute_condition_stats
            rgb_global: [B, C]
            pc_global:  [B, C]
            rgb_quality: [B, 1]
            pc_quality:  [B, 1]
        Returns:
            cond_emb: [B, cond_dim]
            illum_logits, dist_logits, quality_logits
        """
        s = self.stats_net(stats.float())
        feat_input = torch.cat([
            rgb_global.float(), pc_global.float(),
            rgb_quality.float(), pc_quality.float()
        ], dim=-1)
        f = self.feat_net(feat_input)
        c = self.combine(torch.cat([s, f], dim=-1))
        return c, self.illum_head(c), self.dist_head(c), self.quality_head(c)

    @staticmethod
    @torch.no_grad()
    def pseudo_labels(stats, rgb_quality, pc_quality, n_illum=5, n_dist=5):
        """Finer pseudo labels with more bins.

        Returns: illum_labels, dist_labels, quality_labels
        """
        B = stats.size(0)
        dev = stats.device

        def _bucketize(x, n):
            if B < n:
                return torch.zeros(B, dtype=torch.long, device=dev)
            s = x.sort().values
            boundaries = [s[(len(s) * i) // n].item() for i in range(1, n)]
            return torch.bucketize(
                x.contiguous(),
                torch.tensor(boundaries, device=dev, dtype=x.dtype)
            )

        illum_lbl = _bucketize(stats[:, 0], n_illum)   # brightness
        dist_lbl = _bucketize(stats[:, 3], n_dist)      # distance

        # Quality labels based on RGB quality score
        rq = rgb_quality.squeeze(-1)
        qual_lbl = _bucketize(rq, 3)

        return illum_lbl, dist_lbl, qual_lbl


# ================================================================
#  Fix-7: Quality-Aware Fusion Expert & MoE
# ================================================================

class QualityAwareFusionExpertBlock(nn.Module):
    """Fusion expert with modality quality modulation.

    Each expert receives quality signals and can modulate attention
    differently for RGB vs PC tokens based on quality.
    """

    def __init__(self, dim, output_parts, num_input_tokens, rank=32):
        super().__init__()
        self.output_parts = output_parts
        half_tokens = num_input_tokens // 2  # P for RGB, P for PC

        self.queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.token_logit = nn.Parameter(torch.zeros(num_input_tokens))

        # Quality modulation: learnable bias based on quality difference
        self.quality_gate_rgb = nn.Sequential(
            nn.Linear(2, rank), nn.GELU(), nn.Linear(rank, half_tokens),
        )
        self.quality_gate_pc = nn.Sequential(
            nn.Linear(2, rank), nn.GELU(), nn.Linear(rank, half_tokens),
        )

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

    def forward(self, tokens, quality_scores=None):
        """
        Args:
            tokens: [N, 2P, C]
            quality_scores: [N, 2] — (rgb_quality, pc_quality) or None
        Returns:
            [N, P, C]
        """
        N, TK, C = tokens.shape
        P = TK // 2

        # Compute token attention weights with quality modulation
        ta = torch.sigmoid(self.token_logit)  # [2P]

        if quality_scores is not None:
            qs = quality_scores.float()  # [N, 2]
            rgb_mod = torch.sigmoid(self.quality_gate_rgb(qs))   # [N, P]
            pc_mod = torch.sigmoid(self.quality_gate_pc(qs))     # [N, P]
            quality_mod = torch.cat([rgb_mod, pc_mod], dim=-1)   # [N, 2P]
            ta = ta.unsqueeze(0) * quality_mod                   # [N, 2P]
        else:
            ta = ta.unsqueeze(0)

        kv = self.norm_kv(tokens) * ta.unsqueeze(-1)
        q = self.norm_q(self.queries.unsqueeze(0).expand(N, -1, -1))
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        attn = (Q @ K.transpose(-1, -2)) * self.attn_scale
        out = self.o_proj(attn.softmax(-1) @ V)
        return out + self.ffn(self.ffn_norm(out))

    def get_token_preference(self, quality_scores=None):
        """Return RGB vs PC token preference for visualization."""
        P = self.token_logit.size(0) // 2
        ta = torch.sigmoid(self.token_logit)
        rgb_pref = ta[:P].mean().item()
        pc_pref = ta[P:].mean().item()
        return rgb_pref, pc_pref


class QualityAwareFusionMoELayer(nn.Module):
    """Fusion MoE with explicit modality quality signals in routing.

    Fix-7: Router receives quality scores; experts modulate token attention.
    Fix-8: Lower residual gate init (0.4 vs 0.6) for stronger expert contribution.
    """

    def __init__(self, dim, output_parts, num_input_tokens, num_experts=8,
                 top_k=2, cond_dim=64, expert_rank=32, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_parts = output_parts

        self.experts = nn.ModuleList([
            QualityAwareFusionExpertBlock(dim, output_parts, num_input_tokens, rank=expert_rank)
            for _ in range(num_experts)
        ])
        # Router input: token_mean [dim] + cond_emb [cond_dim] + quality [2]
        router_in = dim + cond_dim + 2
        self.router = nn.Sequential(
            nn.Linear(router_in, dim // 2), nn.ReLU(True),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, num_experts),
        )
        # Residual projection
        self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
        # Fix-10: lower residual gate init for stronger expert contribution
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens, cond_emb, quality_scores, seq_len=1):
        """
        Args:
            tokens: [N, 2P, C]
            cond_emb: [B, cond_dim] or None
            quality_scores: [B, 2] — (rgb_quality, pc_quality)
            seq_len: frames per sample
        Returns:
            (output [N, C, P], logits [B, E], weights [B, E])
        """
        N, TK, Cd = tokens.shape

        # --- Compute routing at sequence level ---
        if seq_len > 1:
            B = N // seq_len
            fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))  # [B, Cd]
        else:
            B = N
            fp = tokens.mean(dim=1)  # [B, Cd]

        # Build router input
        parts = [fp]
        if cond_emb is not None and cond_emb.size(-1) > 0:
            if cond_emb.size(0) == N and seq_len > 1:
                ce = cond_emb.reshape(B, seq_len, -1).mean(dim=1)
            elif cond_emb.size(0) == B:
                ce = cond_emb
            else:
                ce = fp.new_zeros(B, 0)
            parts.append(ce.to(device=fp.device, dtype=fp.dtype))

        if quality_scores is not None and quality_scores.size(-1) > 0:
            qs = quality_scores
            if qs.size(0) == N and seq_len > 1:
                qs = qs.reshape(B, seq_len, -1).mean(dim=1)
            parts.append(qs.to(device=fp.device, dtype=fp.dtype))
        else:
            parts.append(fp.new_zeros(B, 2))

        ri = torch.cat(parts, dim=-1)
        logits = self.router(ri)                        # [B, E]
        _, ti = logits.topk(self.top_k, dim=-1)
        mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
        wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), dim=-1)

        # Expand to frame level
        if seq_len > 1:
            mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            qs_N = quality_scores.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1) \
                if quality_scores is not None else None
        else:
            mN, wN = mB, wB
            qs_N = quality_scores

        # --- Expert computation ---
        out = torch.zeros(N, self.output_parts, Cd, device=tokens.device, dtype=tokens.dtype)
        for i in range(self.num_experts):
            sel = mN[:, i] > 0
            if not sel.any():
                continue
            qs_sel = qs_N[sel] if qs_N is not None else None
            eo = self.experts[i](tokens[sel], qs_sel).to(dtype=out.dtype)
            w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
            out[sel] = out[sel] + w * eo

        # Residual
        alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
        res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, self.output_parts, Cd)
        res = res.to(dtype=out.dtype)
        out = alpha * res + (1.0 - alpha) * out

        return out.permute(0, 2, 1).contiguous(), logits, wB


# ================================================================
#  Fix-8: Enhanced Anti-Collapse Losses
# ================================================================

def moe_z_loss(logits):
    """Z-loss from ST-MoE: prevents router logits from growing unbounded.

    Large logits reduce exploration and cause collapse.
    """
    if logits is None or logits.dim() != 2:
        return torch.tensor(0.0)
    return torch.logsumexp(logits, dim=-1).square().mean()


def moe_load_balance_loss_v2(logits, top_k=2):
    """Improved balance loss with importance and load terms (from Switch Transformer).

    More aggressive than the V2 version to prevent collapse.
    """
    if logits is None or logits.dim() != 2:
        return torch.tensor(0.0)
    E = logits.size(1)
    # Importance: mean probability assigned to each expert
    probs = F.softmax(logits, dim=-1)
    importance = probs.mean(dim=0)  # [E]
    # Load: fraction of samples selecting each expert in top-k
    _, topk_idx = logits.topk(top_k, dim=-1)
    load = torch.zeros(E, device=logits.device)
    for k in range(top_k):
        for i in range(E):
            load[i] += (topk_idx[:, k] == i).float().mean()
    load = load / top_k
    # Loss: minimize CV (coefficient of variation) of both
    return E * (importance * load).sum()


def expert_utilization_stats(logits, top_k=2):
    """Compute per-expert utilization for monitoring/visualization.

    Returns dict with per-expert usage fraction and entropy.
    """
    if logits is None or logits.dim() != 2:
        return {}
    E = logits.size(1)
    _, topk_idx = logits.topk(top_k, dim=-1)
    usage = torch.zeros(E, device=logits.device)
    for k in range(top_k):
        for i in range(E):
            usage[i] += (topk_idx[:, k] == i).float().sum()
    usage = usage / (logits.size(0) * top_k)
    # Entropy of usage distribution (higher = more balanced)
    usage_clamp = usage.clamp(min=1e-8)
    entropy = -(usage_clamp * usage_clamp.log()).sum().item()
    max_entropy = math.log(E)
    return {
        'usage': usage.detach().cpu(),
        'entropy': entropy,
        'max_entropy': max_entropy,
        'balance_ratio': entropy / max_entropy if max_entropy > 0 else 1.0,
    }


# ================================================================
#  Fix-10: Stronger Expert TopKMoE (lower residual gate)
# ================================================================

class TopKMoELayerV2Plus(nn.Module):
    """TopK MoE with lower residual gate init and z-loss support.

    Fix-10: residual_gate init 0.4 (expert contributes 60% instead of 30%).
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
            nn.Linear(dim, dim // 2), nn.ReLU(True),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, num_experts),
        )
        # Fix-10: lower init so expert output dominates
        self.residual_gate = nn.Parameter(torch.tensor(0.5)) #0.4

    def forward(self, x, seq_len=1):
        N, C, P = x.shape
        if seq_len > 1:
            B = N // seq_len
            r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))
        else:
            B = N
            r_in = x.mean(dim=-1)

        logits = self.router(r_in)
        _, topk_idx = logits.topk(self.top_k, dim=-1)
        mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights_B = F.softmax(logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

        if seq_len > 1:
            mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mask_N, wgt_N = mask_B, weights_B

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
#  Fix-9: Modality Dropout
# ================================================================

class ModalityDropout(nn.Module):
    """Randomly drops one modality during training to force robust fusion.

    During training, with probability `p`, zeros out either RGB or PC
    projected tokens. This forces the fusion experts to handle degraded
    input and learn meaningful modality-dependent routing.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, rgb_tokens: torch.Tensor, pc_tokens: torch.Tensor):
        """
        Args:
            rgb_tokens: [N, C, P] projected RGB features
            pc_tokens:  [N, C, P] projected PC features
        Returns:
            (rgb_tokens, pc_tokens) — potentially with one zeroed
        """
        if not self.training or self.p <= 0:
            return rgb_tokens, pc_tokens

        if torch.rand(1).item() < self.p:
            # 50/50 chance to drop RGB or PC
            if torch.rand(1).item() < 0.5:
                rgb_tokens = torch.zeros_like(rgb_tokens)
            else:
                pc_tokens = torch.zeros_like(pc_tokens)

        return rgb_tokens, pc_tokens


# ================================================================
#  Modality Quality Supervision Loss
# ================================================================

def modality_quality_contrastive_loss(rgb_quality, pc_quality, stats, temperature=0.5):
    """Auxiliary loss to train the quality estimator.

    Intuition: when brightness is low (night), RGB quality should be low.
    When point density is low, PC quality should be low.
    Uses soft margin ranking: quality should correlate with input stats.
    """
    B = stats.size(0)
    if B < 2:
        return stats.new_tensor(0.0)

    rq = rgb_quality.squeeze(-1)  # [B]
    pq = pc_quality.squeeze(-1)   # [B]

    # RGB quality should correlate with brightness + contrast
    rgb_proxy = (stats[:, 0] + stats[:, 1]) / 2.0  # brightness + contrast
    rgb_proxy = (rgb_proxy - rgb_proxy.mean()) / rgb_proxy.std().clamp(min=1e-6)

    # PC quality should correlate with density + spread
    pc_proxy = (stats[:, 4] + stats[:, 5]) / 2.0   # density + spread
    pc_proxy = (pc_proxy - pc_proxy.mean()) / pc_proxy.std().clamp(min=1e-6)

    # Soft ranking loss: if proxy_i > proxy_j, quality_i should > quality_j
    loss = torch.tensor(0.0, device=stats.device)
    # Use efficient pairwise computation
    rq_diff = rq.unsqueeze(0) - rq.unsqueeze(1)        # [B, B]
    rgb_proxy_diff = rgb_proxy.unsqueeze(0) - rgb_proxy.unsqueeze(1)
    # Margin ranking: sign should agree
    loss = loss + F.soft_margin_loss(rq_diff, rgb_proxy_diff.sign())

    pq_diff = pq.unsqueeze(0) - pq.unsqueeze(1)
    pc_proxy_diff = pc_proxy.unsqueeze(0) - pc_proxy.unsqueeze(1)
    loss = loss + F.soft_margin_loss(pq_diff, pc_proxy_diff.sign())

    return loss * 0.5


# ================================================================
#  Visualization Data Collection Helpers
# ================================================================

@torch.no_grad()
def collect_expert_usage(logits_dict: Dict[str, torch.Tensor], top_k: int = 2) -> Dict:
    """Collect expert usage statistics from routing logits.

    Args:
        logits_dict: {'app': [B, E], 'str': [B, E], 'fus': [B, E]}
        top_k: number of experts selected per sample
    Returns:
        Dict with per-expert usage counts and fractions
    """
    result = {}
    for name, logits in logits_dict.items():
        if logits is None:
            continue
        stats = expert_utilization_stats(logits, top_k)
        result[name] = stats
    return result


@torch.no_grad()
def collect_appearance_expert_parts(app_moe) -> Dict[str, torch.Tensor]:
    """Extract part-attention patterns from appearance experts.

    Returns sigmoid(part_logit) for each expert — shows which body parts
    each expert attends to.
    """
    if app_moe is None:
        return {}
    result = {}
    for i, expert in enumerate(app_moe.experts):
        if hasattr(expert, 'part_logit'):
            result[f'expert_{i}'] = torch.sigmoid(expert.part_logit).detach().cpu()
    return result


@torch.no_grad()
def collect_fusion_token_preference(fus_moe, quality_scores=None) -> Dict:
    """Analyze fusion experts' preference for RGB vs PC tokens.

    Returns per-expert mean attention on RGB tokens vs PC tokens.
    """
    if fus_moe is None:
        return {}
    result = {}
    for i, expert in enumerate(fus_moe.experts):
        if hasattr(expert, 'token_logit'):
            logit = expert.token_logit.detach()
            P = logit.size(0) // 2
            rgb_attn = torch.sigmoid(logit[:P]).mean().item()
            pc_attn = torch.sigmoid(logit[P:]).mean().item()
            result[f'expert_{i}'] = {
                'rgb_preference': rgb_attn,
                'pc_preference': pc_attn,
                'ratio_rgb_over_pc': rgb_attn / max(pc_attn, 1e-8),
            }
            # Quality-modulated preference if available
            if quality_scores is not None and hasattr(expert, 'quality_gate_rgb'):
                qs = quality_scores.float().mean(dim=0, keepdim=True)  # avg quality
                rgb_mod = torch.sigmoid(expert.quality_gate_rgb(qs)).mean().item()
                pc_mod = torch.sigmoid(expert.quality_gate_pc(qs)).mean().item()
                result[f'expert_{i}']['quality_modulated_rgb'] = rgb_attn * rgb_mod
                result[f'expert_{i}']['quality_modulated_pc'] = pc_attn * pc_mod
    return result
