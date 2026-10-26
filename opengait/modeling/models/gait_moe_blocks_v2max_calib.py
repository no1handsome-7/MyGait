"""GaitMoEOracle V2Max-Calib — calibrated QACT + condition-aware MoE blocks.

NEW over V2Pro:
  (1) Multi-scale heterogeneous experts (Pointwise / LocalConv / GlobalAttn)
  (2) Shared Expert Isolation (DeepSeekMoE, ACL'24)
  (3) Capacity-heterogeneous experts (mixed rank)
  (4) Branch-level auxiliary supervision heads
  (5) Lightweight pre-fusion temporal aggregation
  (6) Modality-aware dynamic router
  (7) Identity-nuisance orthogonality loss
  (8) Hard-negative cross-modal contrastive loss
  (9) Quality-weighted modality dropout
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Re-export everything from V2Pro blocks
from .gait_moe_blocks_v2pro import (
    PartLevelAlignBeforeFuse,
    cross_modal_alignment_loss,
    part_level_cross_modal_contrastive_loss,
    intra_modal_separation_loss,
)
from .gait_moe_blocks_v2plus import (
    ModalityQualityEstimator, EnhancedConditionEncoder,
    QualityAwareFusionExpertBlock,
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss,
    expert_utilization_stats,
    collect_expert_usage, collect_appearance_expert_parts,
    collect_fusion_token_preference,
    ModalityDropout,
)
from .gait_moe_blocks import (
    l2n, _scalar, symmetric_infonce, masked_symmetric_infonce,
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    CrossAttentionFusion, MLPFusion,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    compute_condition_stats,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
    PartExpertBlock,
)
from ..modules import SeparateFCs, SeparateBNNecks


# ================================================================
#  Multi-Scale Heterogeneous Experts (Method A)
# ================================================================

class PointwiseExpert(nn.Module):
    """Pointwise (per-part independent) expert.
    Inductive bias: captures channel-level patterns within individual parts.
    Suitable for texture/appearance per body region.
    """
    def __init__(self, dim, parts_num, rank=16, dropout=0.0):
        super().__init__()
        self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
        self.transform = nn.Sequential(
            nn.Linear(dim, rank, bias=False), nn.GELU(),
            nn.Linear(rank, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """x: [N, C, P] -> [N, C, P]"""
        pa = torch.sigmoid(self.part_logit)
        h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
        delta = self.drop(self.transform(self.norm(h)))
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()


class LocalConvExpert(nn.Module):
    """Local-neighbor expert using Conv1d over parts dimension.
    Inductive bias: captures relationships between adjacent body parts.
    Suitable for limb coordination patterns.
    """
    def __init__(self, dim, parts_num, rank=16, kernel_size=3, dropout=0.0):
        super().__init__()
        self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
        # Depthwise conv over parts + pointwise projection
        self.local_conv = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2,
            groups=min(dim, 32), bias=False)
        self.transform = nn.Sequential(
            nn.Linear(dim, rank, bias=False), nn.GELU(),
            nn.Linear(rank, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """x: [N, C, P] -> [N, C, P]"""
        pa = torch.sigmoid(self.part_logit)
        h = x * pa.unsqueeze(0).unsqueeze(0)
        # Local conv over parts: [N, C, P]
        h = self.local_conv(h)
        h = h.permute(0, 2, 1)  # [N, P, C]
        delta = self.drop(self.transform(self.norm(h)))
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()


class GlobalAttentionExpert(nn.Module):
    """Global attention expert attending over all parts.
    Inductive bias: captures holistic body structure and viewpoint patterns.
    Suitable for overall silhouette/contour and view-dependent features.
    """
    def __init__(self, dim, parts_num, rank=32, num_heads=2, dropout=0.0):
        super().__init__()
        self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.transform = nn.Sequential(
            nn.Linear(dim, rank, bias=False), nn.GELU(),
            nn.Linear(rank, dim, bias=False),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """x: [N, C, P] -> [N, C, P]"""
        pa = torch.sigmoid(self.part_logit)
        h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
        h_normed = self.norm1(h)
        attn_out, _ = self.attn(h_normed, h_normed, h_normed)
        h = h + attn_out
        delta = self.drop(self.transform(self.norm2(h)))
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()


class SharedExpert(nn.Module):
    """Always-on shared expert (DeepSeekMoE).
    Captures universal patterns common to all inputs.
    """
    def __init__(self, dim, parts_num, rank=16):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(dim, rank, bias=False), nn.GELU(),
            nn.LayerNorm(rank),
            nn.Linear(rank, dim, bias=False),
        )
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        """x: [N, C, P] -> [N, C, P]"""
        h = x.permute(0, 2, 1)  # [N, P, C]
        delta = self.transform(self.norm(h))
        h = h + self.scale * delta
        return h.permute(0, 2, 1).contiguous()


# ================================================================
#  Heterogeneous TopK MoE Layer
# ================================================================

def build_expert(expert_type, dim, parts_num, rank, dropout, kernel_size=3):
    """Factory for building heterogeneous experts."""
    if expert_type == 'pointwise':
        return PointwiseExpert(dim, parts_num, rank, dropout)
    elif expert_type == 'local_conv':
        return LocalConvExpert(dim, parts_num, rank, kernel_size, dropout)
    elif expert_type == 'global_attn':
        return GlobalAttentionExpert(dim, parts_num, rank, 2, dropout)
    elif expert_type == 'standard':
        return PartExpertBlock(dim, parts_num, rank, dropout)
    else:
        raise ValueError(f'Unknown expert type: {expert_type}')


class HeterogeneousMoELayer(nn.Module):
    """MoE with shared expert + heterogeneous routed experts.

    V2Max-Calib adds optional condition / quality inputs to the branch router.
    This lets appearance / structure experts see explicit degradation cues
    (brightness, distance, density, modality reliability) instead of relying
    only on pooled features.  Defaults keep backward compatibility.
    """
    def __init__(self, dim, parts_num, expert_configs, top_k=2,
                 shared_rank=16, dropout=0.0, shared_enabled=True,
                 cond_dim=0, quality_dim=0):
        """
        Args:
            expert_configs: list of dicts, each with keys:
                'type': 'pointwise'|'local_conv'|'global_attn'|'standard'
                'rank': int
                'kernel_size': int (for local_conv)
            cond_dim: optional condition embedding size for routing.
            quality_dim: optional quality vector size for routing.
        """
        super().__init__()
        self.num_experts = len(expert_configs)
        self.top_k = top_k
        self.shared_enabled = shared_enabled
        self.cond_dim = int(cond_dim or 0)
        self.quality_dim = int(quality_dim or 0)

        # Build heterogeneous experts
        self.experts = nn.ModuleList()
        for cfg in expert_configs:
            self.experts.append(build_expert(
                cfg.get('type', 'standard'), dim, parts_num,
                cfg.get('rank', 16), dropout,
                cfg.get('kernel_size', 3)))

        # Shared expert
        if shared_enabled:
            self.shared_expert = SharedExpert(dim, parts_num, shared_rank)
        else:
            self.shared_expert = None

        # Router
        router_in = dim + self.cond_dim + self.quality_dim
        self.router = nn.Sequential(
            nn.Linear(router_in, dim // 2), nn.ReLU(True),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, self.num_experts),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, seq_len=1, cond_emb=None, quality_scores=None):
        """x: [N, C, P]; Returns (output, logits, weights)."""
        N, C, P = x.shape

        # --- Shared expert (always-on) ---
        shared_out = self.shared_expert(x) if self.shared_expert is not None else None

        # --- Routing at sequence level ---
        if seq_len > 1:
            B = N // seq_len
            r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))
        else:
            B = N
            r_in = x.mean(dim=-1)

        router_parts = [r_in]
        if self.cond_dim > 0:
            if cond_emb is not None and cond_emb.size(-1) == self.cond_dim:
                ce = cond_emb
                if ce.size(0) == N and seq_len > 1:
                    ce = ce.reshape(B, seq_len, -1).mean(dim=1)
                elif ce.size(0) != B:
                    ce = r_in.new_zeros(B, self.cond_dim)
            else:
                ce = r_in.new_zeros(B, self.cond_dim)
            router_parts.append(ce.to(device=r_in.device, dtype=r_in.dtype))
        if self.quality_dim > 0:
            if quality_scores is not None and quality_scores.size(-1) == self.quality_dim:
                qs = quality_scores
                if qs.size(0) == N and seq_len > 1:
                    qs = qs.reshape(B, seq_len, -1).mean(dim=1)
                elif qs.size(0) != B:
                    qs = r_in.new_zeros(B, self.quality_dim)
            else:
                qs = r_in.new_zeros(B, self.quality_dim)
            router_parts.append(qs.to(device=r_in.device, dtype=r_in.dtype))
        logits = self.router(torch.cat(router_parts, dim=-1))
        _, topk_idx = logits.topk(self.top_k, dim=-1)
        mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights_B = F.softmax(
            logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

        if seq_len > 1:
            mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mask_N, wgt_N = mask_B, weights_B

        # --- Expert computation ---
        expert_out = torch.zeros_like(x)
        for i in range(self.num_experts):
            sel = mask_N[:, i] > 0
            if not sel.any():
                continue
            e = self.experts[i](x[sel]).to(dtype=expert_out.dtype)
            w = wgt_N[sel, i].to(dtype=expert_out.dtype).unsqueeze(-1).unsqueeze(-1)
            expert_out[sel] = expert_out[sel] + w * e

        # --- Combine: residual + routed + shared ---
        alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
        output = alpha * x + (1.0 - alpha) * expert_out
        if shared_out is not None:
            output = output + shared_out

        return output, logits, weights_B


# ================================================================
#  Heterogeneous Quality-Aware Fusion MoE
# ================================================================

class HeterogeneousFusionMoELayer(nn.Module):
    """Fusion MoE with shared expert + heterogeneous routed fusion experts.

    Routed fusion experts have different attention ranks for capacity heterogeneity.
    """
    def __init__(self, dim, output_parts, num_input_tokens,
                 expert_ranks, top_k=2, cond_dim=64,
                 dropout=0.0, shared_enabled=True, shared_rank=32,
                 modality_aware_router=False):
        """
        Args:
            expert_ranks: list of int, one per expert. E.g. [16,16,32,32,32,64,64,64]
        """
        super().__init__()
        self.num_experts = len(expert_ranks)
        self.top_k = top_k
        self.output_parts = output_parts
        self.shared_enabled = shared_enabled
        self.modality_aware_router = modality_aware_router

        # Heterogeneous fusion experts
        self.experts = nn.ModuleList([
            QualityAwareFusionExpertBlock(
                dim, output_parts, num_input_tokens, rank=r)
            for r in expert_ranks
        ])

        # Shared fusion expert
        if shared_enabled:
            self.shared_fusion = QualityAwareFusionExpertBlock(
                dim, output_parts, num_input_tokens, rank=shared_rank)
            self.shared_scale = nn.Parameter(torch.tensor(0.3))
        else:
            self.shared_fusion = None

        # Router
        router_in = dim + cond_dim + 2
        if modality_aware_router:
            self.mod_embed = nn.Embedding(2, dim // 8)
            router_in += dim // 8
        self.router = nn.Sequential(
            nn.Linear(router_in, dim // 2), nn.ReLU(True),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, self.num_experts),
        )
        # Residual
        self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, tokens, cond_emb, quality_scores, seq_len=1):
        N, TK, Cd = tokens.shape
        P = self.output_parts
        half = TK // 2

        # --- Shared fusion (always-on) ---
        shared_out = None
        if self.shared_fusion is not None:
            qs_for_shared = quality_scores if quality_scores is not None else None
            if qs_for_shared is not None and qs_for_shared.size(0) != N and seq_len > 1:
                B_s = N // seq_len
                qs_for_shared = qs_for_shared.unsqueeze(1).expand(
                    -1, seq_len, -1).reshape(N, -1)
            shared_out = self.shared_fusion(tokens, qs_for_shared)
            shared_out = shared_out * self.shared_scale

        # --- Routing at sequence level ---
        if seq_len > 1:
            B = N // seq_len
            fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
        else:
            B = N
            fp = tokens.mean(dim=1)

        parts = [fp]
        if cond_emb is not None and cond_emb.size(-1) > 0:
            ce = cond_emb
            if ce.size(0) == N and seq_len > 1:
                ce = ce.reshape(B, seq_len, -1).mean(dim=1)
            elif ce.size(0) != B:
                ce = fp.new_zeros(B, cond_emb.size(-1))
            parts.append(ce.to(device=fp.device, dtype=fp.dtype))

        qs = quality_scores
        if qs is not None and qs.size(-1) > 0:
            if qs.size(0) == N and seq_len > 1:
                qs_r = qs.reshape(B, seq_len, -1).mean(dim=1)
            elif qs.size(0) == B:
                qs_r = qs
            else:
                qs_r = fp.new_zeros(B, 2)
            parts.append(qs_r.to(device=fp.device, dtype=fp.dtype))
        else:
            parts.append(fp.new_zeros(B, 2))

        if self.modality_aware_router:
            # Average modality embedding
            mod_ids = torch.zeros(B, TK, dtype=torch.long, device=fp.device)
            mod_ids[:, half:] = 1
            if seq_len > 1:
                mod_ids_s = mod_ids[:1].expand(B, -1)
            else:
                mod_ids_s = mod_ids
            me = self.mod_embed(mod_ids_s).mean(dim=1)  # [B, dim//8]
            parts.append(me)

        ri = torch.cat(parts, dim=-1)
        logits = self.router(ri)
        _, ti = logits.topk(self.top_k, dim=-1)
        mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
        wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), dim=-1)

        if seq_len > 1:
            mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            qs_N = quality_scores
            if qs_N is not None and qs_N.size(0) == B:
                qs_N = qs_N.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mN, wN = mB, wB
            qs_N = quality_scores

        # --- Expert computation ---
        out = torch.zeros(N, P, Cd, device=tokens.device, dtype=tokens.dtype)
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
        res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, P, Cd)
        res = res.to(dtype=out.dtype)
        out = alpha * res + (1.0 - alpha) * out

        # Add shared
        if shared_out is not None:
            out = out + shared_out.to(dtype=out.dtype)

        return out.permute(0, 2, 1).contiguous(), logits, wB


# ================================================================
#  Lightweight Pre-Fusion Temporal Aggregation
# ================================================================

class LightTemporalAgg(nn.Module):
    """Lightweight temporal context via depthwise Conv1d.
    < 2K params. Gated residual ensures no degradation.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())
        # Init gate bias to negative so gate starts near 0
        nn.init.constant_(self.gate[0].bias, -2.0)

    def forward(self, x):
        """x: [B, T, C, P] -> [B, T, C, P]"""
        B, T, C, P = x.shape
        if T <= 1:
            return x
        xp = x.permute(0, 3, 2, 1).reshape(B * P, C, T)
        h = self.conv(xp).reshape(B, P, C, T).permute(0, 3, 2, 1)
        h_n = self.norm(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        g = self.gate(h_n.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return x + g * h_n


# ================================================================
#  Branch Auxiliary Head
# ================================================================

class BranchAuxHead(nn.Module):
    """Lightweight auxiliary classification head for branch-level supervision.
    Discarded at inference.
    """
    def __init__(self, in_channels, out_channels, parts_num, class_num):
        super().__init__()
        self.fc = SeparateFCs(parts_num, in_channels, out_channels)
        self.bn = SeparateBNNecks(parts_num, out_channels, class_num)

    def forward(self, x):
        """x: [B, C, P] -> (embed [B, out_c, P], logits [B, class_num, P])"""
        e = self.fc(x)
        _, logits = self.bn(e)
        return e, logits


# ================================================================
#  Quality-Weighted Modality Dropout
# ================================================================

class QualityWeightedModalityDropout(nn.Module):
    """Drop the lower-quality modality more frequently."""
    def __init__(self, p=0.1, quality_weighted=True):
        super().__init__()
        self.p = p
        self.quality_weighted = quality_weighted

    def forward(self, rgb_tokens, pc_tokens,
                rgb_quality=None, pc_quality=None):
        if not self.training or self.p <= 0:
            return rgb_tokens, pc_tokens

        if torch.rand(1).item() >= self.p:
            return rgb_tokens, pc_tokens

        # Decide which modality to drop
        if self.quality_weighted and rgb_quality is not None and pc_quality is not None:
            rq = rgb_quality.mean().item()
            pq = pc_quality.mean().item()
            # Higher quality -> lower drop probability
            total = rq + pq + 1e-8
            drop_rgb_prob = pq / total  # drop RGB more when PC is better
        else:
            drop_rgb_prob = 0.5

        if torch.rand(1).item() < drop_rgb_prob:
            rgb_tokens = torch.zeros_like(rgb_tokens)
        else:
            pc_tokens = torch.zeros_like(pc_tokens)

        return rgb_tokens, pc_tokens


# ================================================================
#  Losses
# ================================================================

def identity_nuisance_orthogonality_loss(id_feat, nuisance_feats):
    """Explicit orthogonality between identity and nuisance features.

    Args:
        id_feat: [B, D] identity features
        nuisance_feats: dict {factor_name: [B, D_sub]} visual probe outputs
    Returns:
        scalar loss
    """
    loss = id_feat.new_tensor(0.0)
    id_norm = F.normalize(id_feat.float(), dim=-1)
    cnt = 0
    for k, z_f in nuisance_feats.items():
        if z_f is None:
            continue
        z_f_norm = F.normalize(z_f.detach().float(), dim=-1)
        if z_f_norm.size(-1) != id_norm.size(-1):
            continue
        cos_sim = (id_norm * z_f_norm).sum(dim=-1)
        loss = loss + cos_sim.pow(2).mean()
        cnt += 1
    return loss / max(cnt, 1)


def part_level_cross_modal_contrastive_loss_v2(
        rgb_align, pc_align, labels=None, temperature=0.07,
        hard_negative=True, hard_ratio=0.5):
    """Part-level cross-modal contrastive with hard negative mining."""
    B, P, D = rgb_align.shape
    if B <= 1:
        return rgb_align.new_tensor(0.0)

    loss = rgb_align.new_tensor(0.0)
    for p in range(P):
        rp = rgb_align[:, p, :]
        pp = pc_align[:, p, :]
        logits = rp @ pp.t() / temperature

        if labels is not None:
            lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            neg_mask = 1.0 - lbl_eq

            if hard_negative and neg_mask.sum() > 0:
                pos_sim = logits * lbl_eq + (-1e9) * neg_mask
                hardest_pos_val = pos_sim.min(dim=1).values
                semi_hard = (
                    logits > (hardest_pos_val.unsqueeze(1) - 0.3)
                ).float() * neg_mask
                semi_hard = torch.where(
                    semi_hard.sum(1, keepdim=True) > 0, semi_hard, neg_mask)
                weight = semi_hard * hard_ratio + neg_mask * (1 - hard_ratio)
            else:
                weight = neg_mask

            pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
            log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
            loss_p = -(pos_mask * log_prob).sum(dim=1).mean()
            # Symmetric
            log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
            loss_p = loss_p - (pos_mask.t() * log_prob_t).sum(dim=1).mean()
            loss = loss + loss_p * 0.5
        else:
            targets = torch.arange(B, device=rgb_align.device)
            loss = loss + 0.5 * (
                F.cross_entropy(logits, targets) +
                F.cross_entropy(logits.t(), targets))

    return loss / P


def cross_modal_alignment_loss_v2(
        rgb_align, pc_align, labels=None,
        plc_weight=1.0, ims_weight=0.3, temperature=0.07,
        hard_negative=True, hard_ratio=0.5):
    """Combined alignment loss with hard negative mining."""
    plc = part_level_cross_modal_contrastive_loss_v2(
        rgb_align, pc_align, labels, temperature,
        hard_negative, hard_ratio)
    ims_rgb = intra_modal_separation_loss(rgb_align, labels)
    ims_pc = intra_modal_separation_loss(pc_align, labels)
    return plc_weight * plc + ims_weight * 0.5 * (ims_rgb + ims_pc)


# ================================================================
#  Heterogeneous Expert Diversity Loss
# ================================================================

def heterogeneous_diversity_loss(experts, thr=0.5):
    """Diversity loss adapted for heterogeneous experts.
    Only compares experts that share the same attribute (part_logit or token_logit).
    """
    bs = []
    for e in experts:
        if hasattr(e, 'part_logit'):
            bs.append(torch.sigmoid(e.part_logit))
    if len(bs) < 2:
        return torch.tensor(0.0, device=bs[0].device) if bs else torch.tensor(0.0)
    s = F.normalize(torch.stack(bs), dim=-1)
    sim = s @ s.t()
    E = sim.size(0)
    mask = ~torch.eye(E, dtype=torch.bool, device=sim.device)
    return F.relu(sim[mask] - thr).mean()


# ================================================================
#  Quality-Aware Context Tokens (QACT)
#
#  Replaces: compute_condition_stats, ModalityQualityEstimator,
#            EnhancedConditionEncoder, modality_quality_contrastive_loss
#
#  Theory: Nagrani et al. "Attention Bottlenecks" (NeurIPS 2022),
#          Zhang et al. "CMNeXt Self-Query Hub" (CVPR 2023),
#          Wang et al. "TokenFusion" (CVPR 2022).
# ================================================================

class QualityContextTokens(nn.Module):
    """Calibrated Quality-Aware Context Tokens (QACT-Calib).

    Changes vs. original entropy-only QACT:
      1) Quality is predicted from context tokens + explicit condition stats +
         cross-modal agreement, then softly blended with entropy quality.
      2) cond_emb also receives stats / quality cues so routers can react to
         brightness, contrast, distance and point density.
      3) Forward returns diagnostic tensors for visualization and losses.

    stats is expected to be compute_condition_stats(rgb, pc):
      [brightness, contrast, rgb_edge/sharpness, distance, density, spread].
    """

    def __init__(self, channels, num_tokens=4, num_heads=4,
                 cond_dim=64, dropout=0.1,
                 cross_probe_enabled=True, joint_refine_enabled=True,
                 entropy_quality_blend=0.20):
        super().__init__()
        self.channels = channels
        self.num_tokens = num_tokens
        self.cond_dim = cond_dim
        self.cross_probe_enabled = cross_probe_enabled
        self.joint_refine_enabled = joint_refine_enabled
        self.entropy_quality_blend = float(entropy_quality_blend)

        # Learnable context tokens
        self.rgb_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)
        self.pc_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)

        # Stage 1: Self-probe
        self.rgb_self_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.pc_self_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.rgb_self_norm_q = nn.LayerNorm(channels)
        self.rgb_self_norm_kv = nn.LayerNorm(channels)
        self.pc_self_norm_q = nn.LayerNorm(channels)
        self.pc_self_norm_kv = nn.LayerNorm(channels)

        # Stage 2: Cross-probe
        if cross_probe_enabled:
            self.rgb_cross_attn = nn.MultiheadAttention(
                channels, num_heads, dropout=dropout, batch_first=True)
            self.pc_cross_attn = nn.MultiheadAttention(
                channels, num_heads, dropout=dropout, batch_first=True)
            self.rgb_cross_norm = nn.LayerNorm(channels)
            self.pc_cross_norm = nn.LayerNorm(channels)
            self.rgb_cross_gate = nn.Sequential(
                nn.Linear(channels * 2, channels), nn.Sigmoid())
            self.pc_cross_gate = nn.Sequential(
                nn.Linear(channels * 2, channels), nn.Sigmoid())

        # Stage 3: Joint refinement
        if joint_refine_enabled:
            self.refine_attn = nn.MultiheadAttention(
                channels, num_heads, dropout=dropout, batch_first=True)
            self.refine_norm = nn.LayerNorm(channels)
            self.refine_ffn = nn.Sequential(
                nn.Linear(channels, channels * 2), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(channels * 2, channels))
            self.refine_ffn_norm = nn.LayerNorm(channels)

        # Explicit condition features: 6 stats + 2 qualities.
        self.stats_norm = nn.LayerNorm(6)
        quality_in_dim = channels * 2 + 2 + 6 + 1  # ctxs + entropy q + stats + agreement
        self.quality_head = nn.Sequential(
            nn.Linear(quality_in_dim, channels // 2), nn.GELU(),
            nn.LayerNorm(channels // 2),
            nn.Dropout(dropout),
            nn.Linear(channels // 2, 2),
        )

        cond_in_dim = channels * num_tokens * 2 + 6 + 2
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, channels), nn.GELU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
            nn.Linear(channels, cond_dim))

    def forward(self, rgb_parts, pc_parts, stats=None):
        """
        Args:
            rgb_parts: [B, T, C, P] or [B, C, P]
            pc_parts:  [B, T, C, P] or [B, C, P]
            stats: optional [B, 6] handcrafted condition statistics.
        """
        # Temporal mean → [B, P, C]
        if rgb_parts.dim() == 4:
            rgb_kv = rgb_parts.mean(dim=1).permute(0, 2, 1)
            pc_kv = pc_parts.mean(dim=1).permute(0, 2, 1)
        else:
            rgb_kv = rgb_parts.permute(0, 2, 1)
            pc_kv = pc_parts.permute(0, 2, 1)

        B, P, C = rgb_kv.shape
        K = self.num_tokens
        if stats is None:
            stats = rgb_kv.new_zeros(B, 6)
        stats = torch.nan_to_num(stats.to(device=rgb_kv.device, dtype=rgb_kv.dtype), 0.0, 0.0, 0.0)
        stats_n = self.stats_norm(stats.float()).to(dtype=rgb_kv.dtype)

        rq = self.rgb_tokens.unsqueeze(0).expand(B, -1, -1)
        pq = self.pc_tokens.unsqueeze(0).expand(B, -1, -1)

        # ---- Stage 1: Self-probe ----
        rq_self, r_self_w = self.rgb_self_attn(
            self.rgb_self_norm_q(rq),
            self.rgb_self_norm_kv(rgb_kv), rgb_kv)
        rq = rq + rq_self

        pq_self, p_self_w = self.pc_self_attn(
            self.pc_self_norm_q(pq),
            self.pc_self_norm_kv(pc_kv), pc_kv)
        pq = pq + pq_self

        # ---- Stage 2: Cross-probe ----
        if self.cross_probe_enabled:
            rq_cross, r_cross_w = self.rgb_cross_attn(
                self.rgb_cross_norm(rq), pc_kv, pc_kv)
            rg = self.rgb_cross_gate(torch.cat([rq, rq_cross], dim=-1))
            rq = rq + rg * rq_cross

            pq_cross, p_cross_w = self.pc_cross_attn(
                self.pc_cross_norm(pq), rgb_kv, rgb_kv)
            pg = self.pc_cross_gate(torch.cat([pq, pq_cross], dim=-1))
            pq = pq + pg * pq_cross
        else:
            r_cross_w, p_cross_w = None, None

        # ---- Stage 3: Joint refinement ----
        if self.joint_refine_enabled:
            all_ctx = torch.cat([rq, pq], dim=1)
            all_refined, refine_w = self.refine_attn(
                self.refine_norm(all_ctx), self.refine_norm(all_ctx), all_ctx)
            all_ctx = all_ctx + all_refined
            all_ctx = all_ctx + self.refine_ffn(self.refine_ffn_norm(all_ctx))
            rgb_ctx = all_ctx[:, :K]
            pc_ctx = all_ctx[:, K:]
        else:
            rgb_ctx = rq
            pc_ctx = pq
            refine_w = None

        # ---- Calibrated quality ----
        rgb_ent_q, pc_ent_q = self._quality_from_entropy(r_self_w, p_self_w, P)
        ent_q = torch.cat([rgb_ent_q, pc_ent_q], dim=-1)
        rgb_g = rgb_ctx.mean(dim=1)
        pc_g = pc_ctx.mean(dim=1)
        agreement = F.cosine_similarity(rgb_g.float(), pc_g.float(), dim=-1).unsqueeze(-1)
        agreement_01 = ((agreement + 1.0) * 0.5).clamp(0, 1).to(dtype=rgb_kv.dtype)
        q_in = torch.cat([rgb_g, pc_g, ent_q.to(dtype=rgb_g.dtype), stats_n, agreement_01], dim=-1)
        learned_q = torch.sigmoid(self.quality_head(q_in.float())).to(dtype=rgb_kv.dtype)
        b = min(max(self.entropy_quality_blend, 0.0), 1.0)
        quality_scores = ((1.0 - b) * learned_q + b * ent_q.to(dtype=learned_q.dtype)).clamp(0, 1)

        # ---- Condition embedding ----
        cond_emb = self.cond_proj(torch.cat([
            rgb_ctx.reshape(B, -1), pc_ctx.reshape(B, -1),
            stats_n, quality_scores.detach().to(dtype=stats_n.dtype)
        ], dim=-1).float()).to(dtype=rgb_kv.dtype)

        return {
            'rgb_ctx': rgb_ctx,
            'pc_ctx': pc_ctx,
            'cond_emb': cond_emb,
            'quality_scores': quality_scores,
            'entropy_quality_scores': ent_q.detach(),
            'learned_quality_scores': learned_q.detach(),
            'cross_agreement': agreement_01.detach(),
            'condition_stats': stats.detach(),
            'quality_pseudo_targets': self._pseudo_quality_targets(stats).detach(),
            'rgb_self_attn_w': r_self_w.detach(),
            'pc_self_attn_w': p_self_w.detach(),
            'rgb_cross_attn_w': None if r_cross_w is None else r_cross_w.detach(),
            'pc_cross_attn_w': None if p_cross_w is None else p_cross_w.detach(),
            'refine_attn_w': None if refine_w is None else refine_w.detach(),
        }

    @staticmethod
    def _quality_from_entropy(rgb_attn_w, pc_attn_w, P):
        """Derive [B,1] quality from attention entropy.
        Low entropy → focused → high quality.  This is now only a weak prior.
        """
        max_ent = math.log(max(P, 2))
        def _ent(w):
            e = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
            return e.mean(dim=-1) / max_ent
        return (1 - _ent(rgb_attn_w)).unsqueeze(-1).clamp(0, 1), \
               (1 - _ent(pc_attn_w)).unsqueeze(-1).clamp(0, 1)

    @staticmethod
    @torch.no_grad()
    def _rank01(x):
        """Batch-rank normalize to [0, 1]; robust to unknown data scale."""
        x = torch.nan_to_num(x.float(), 0.0, 0.0, 0.0)
        if x.numel() <= 1:
            return torch.full_like(x, 0.5)
        order = x.argsort()
        ranks = torch.zeros_like(x, dtype=torch.float)
        ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float)
        return ranks / max(1, x.numel() - 1)

    @classmethod
    @torch.no_grad()
    def _pseudo_quality_targets(cls, stats):
        """Weak targets from brightness / contrast / sharpness / distance / density.

        Relative ranks are used because LRGait preprocessing may normalize RGB.
        rgb target: brighter, higher contrast and sharper edges → higher quality.
        pc target: denser, nearer and geometrically stable points → higher quality.
        """
        s = torch.nan_to_num(stats.float(), 0.0, 0.0, 0.0)
        bri = cls._rank01(s[:, 0])
        con = cls._rank01(s[:, 1])
        edge = cls._rank01(s[:, 2])
        dist = cls._rank01(s[:, 3])
        dens = cls._rank01(s[:, 4])
        spread = cls._rank01(s[:, 5])
        rgb_t = (0.45 * bri + 0.30 * con + 0.25 * edge).clamp(0, 1)
        pc_t = (0.50 * dens + 0.35 * (1.0 - dist) + 0.15 * spread).clamp(0, 1)
        return torch.stack([rgb_t, pc_t], dim=-1)


class ContextAwareModalityDropout(nn.Module):
    """Modality dropout using attention entropy instead of scalar quality.

    Low entropy → sharp focus → structured info → high quality → low drop prob.
    High entropy → uniform attention → noisy input → low quality → high drop prob.
    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, rgb_tokens, pc_tokens,
                rgb_self_attn_w=None, pc_self_attn_w=None,
                rgb_quality=None, pc_quality=None):
        if not self.training or self.p <= 0:
            return rgb_tokens, pc_tokens
        if torch.rand(1).item() >= self.p:
            return rgb_tokens, pc_tokens

        if rgb_quality is not None and pc_quality is not None:
            # Lower calibrated quality -> higher probability of being dropped.
            rq = float(rgb_quality.detach().float().mean().clamp(0, 1).item())
            pq = float(pc_quality.detach().float().mean().clamp(0, 1).item())
            rgb_bad, pc_bad = 1.0 - rq, 1.0 - pq
            drop_rgb_prob = rgb_bad / (rgb_bad + pc_bad + 1e-8)
        elif rgb_self_attn_w is not None and pc_self_attn_w is not None:
            def _ent(w):
                return -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()
                         ).sum(dim=-1).mean().item()
            re, pe = _ent(rgb_self_attn_w), _ent(pc_self_attn_w)
            drop_rgb_prob = re / (re + pe + 1e-8)
        else:
            drop_rgb_prob = 0.5

        if torch.rand(1).item() < drop_rgb_prob:
            rgb_tokens = torch.zeros_like(rgb_tokens)
        else:
            pc_tokens = torch.zeros_like(pc_tokens)
        return rgb_tokens, pc_tokens


# ================================================================
#  QACT Self-supervised Losses
# ================================================================

def context_cross_modal_consistency_loss(rgb_ctx, pc_ctx, labels,
                                         temperature=0.1):
    """Same-identity context tokens should encode similar scene conditions
    across modalities. Replaces modality_quality_contrastive_loss."""
    B = rgb_ctx.size(0)
    if B <= 1:
        return rgb_ctx.new_tensor(0.0)
    rgb_r = F.normalize(rgb_ctx.mean(dim=1), dim=-1)
    pc_r = F.normalize(pc_ctx.mean(dim=1), dim=-1)
    logits = rgb_r @ pc_r.t() / temperature
    if labels is not None:
        lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
        log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
        loss_fwd = -(pos_mask * log_prob).sum(dim=1).mean()
        log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
        loss_bwd = -(pos_mask.t() * log_prob_t).sum(dim=1).mean()
        return 0.5 * (loss_fwd + loss_bwd)
    tgt = torch.arange(B, device=rgb_ctx.device)
    return 0.5 * (F.cross_entropy(logits, tgt) +
                   F.cross_entropy(logits.t(), tgt))


def context_diversity_loss(rgb_ctx, pc_ctx, threshold=0.5):
    """Prevent context token collapse — each token should capture
    a different aspect of quality/condition."""
    loss = rgb_ctx.new_tensor(0.0)
    for ctx in [rgb_ctx, pc_ctx]:
        tk = F.normalize(ctx.mean(dim=0), dim=-1)
        sim = tk @ tk.t()
        K = sim.size(0)
        mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
        loss = loss + F.relu(sim[mask] - threshold).mean()
    return loss * 0.5


def qact_quality_calibration_loss(quality_scores, qact_out, stats,
                                  weight_entropy=0.10, entropy_target=0.70):
    """Calibrate QACT quality with weak condition pseudo labels.

    The pseudo labels are intentionally batch-relative, which makes the loss
    usable when RGB normalization / point-cloud scaling differ across setups.
    """
    if quality_scores is None:
        return torch.tensor(0.0, device=stats.device if stats is not None else 'cpu')
    if stats is None:
        return quality_scores.new_tensor(0.0)
    targets = qact_out.get('quality_pseudo_targets', None) if qact_out else None
    if targets is None:
        targets = QualityContextTokens._pseudo_quality_targets(stats)
    targets = targets.to(device=quality_scores.device, dtype=quality_scores.dtype)
    loss = F.mse_loss(quality_scores.float(), targets.float())

    # Do not let attention stay perfectly uniform forever; target a moderate
    # entropy regime instead of pushing it to be one-hot.
    if qact_out is not None and weight_entropy > 0:
        ent_terms = []
        for key in ['rgb_self_attn_w', 'pc_self_attn_w']:
            w = qact_out.get(key, None)
            if w is None:
                continue
            P = w.size(-1)
            max_ent = math.log(max(P, 2))
            ent = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
            ent = ent.mean(dim=-1) / max_ent
            ent_terms.append((ent - float(entropy_target)).pow(2).mean())
        if ent_terms:
            loss = loss + float(weight_entropy) * sum(ent_terms) / len(ent_terms)
    return loss


def context_decorrelation_loss(rgb_ctx, pc_ctx):
    """Per-sample token decorrelation to reduce context-token collapse."""
    loss = rgb_ctx.new_tensor(0.0)
    for ctx in [rgb_ctx, pc_ctx]:
        if ctx.size(1) <= 1:
            continue
        z = F.normalize(ctx.float(), dim=-1)
        sim = torch.bmm(z, z.transpose(1, 2))
        K = sim.size(1)
        eye = torch.eye(K, dtype=torch.bool, device=sim.device).unsqueeze(0)
        loss = loss + sim.masked_select(~eye).pow(2).mean()
    return loss * 0.5


def expert_structure_specialization_loss(experts, parts_num,
                                         sparse_weight=0.10,
                                         smooth_weight=0.10,
                                         global_weight=0.10,
                                         orth_weight=0.20):
    """Encourage heterogeneous experts to express different part priors.

    - Pointwise/standard experts: lower entropy part gates (local focus).
    - LocalConv experts: smooth adjacent part gates.
    - GlobalAttention experts: higher entropy part gates (global coverage).
    - All experts: pairwise part-gate decorrelation.
    """
    if experts is None:
        return torch.tensor(0.0)
    part_probs, typed_losses = [], []
    ref = None
    for e in experts:
        if not hasattr(e, 'part_logit'):
            continue
        pa = torch.sigmoid(e.part_logit.float()).clamp(1e-6, 1 - 1e-6)
        ref = pa
        p = pa / pa.sum().clamp_min(1e-6)
        ent = -(p * p.log()).sum() / math.log(max(parts_num, 2))
        name = type(e).__name__.lower()
        if 'global' in name:
            typed_losses.append(global_weight * (1.0 - ent).pow(2))
        elif 'local' in name:
            tv = (pa[1:] - pa[:-1]).pow(2).mean()
            typed_losses.append(smooth_weight * tv)
        else:
            typed_losses.append(sparse_weight * ent.pow(2))
        part_probs.append(F.normalize(pa, dim=0))
    if ref is None:
        return torch.tensor(0.0)
    loss = ref.new_tensor(0.0)
    if typed_losses:
        loss = loss + sum(typed_losses) / len(typed_losses)
    if len(part_probs) > 1 and orth_weight > 0:
        M = torch.stack(part_probs, dim=0)
        sim = M @ M.t()
        mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        loss = loss + orth_weight * sim.masked_select(mask).pow(2).mean()
    return loss


# ================================================================
#  Default Expert Configurations
# ================================================================

def default_appearance_expert_configs(num_experts=8):
    """Method A + C: multi-scale + mixed rank."""
    configs = [
        # 2x Pointwise, small rank (simple local patterns)
        {'type': 'pointwise', 'rank': 8},
        {'type': 'pointwise', 'rank': 8},
        # 2x LocalConv k=3, medium rank (adjacent part coordination)
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
        # 2x LocalConv k=5, medium rank (wider part interactions)
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
        # 2x GlobalAttn, large rank (holistic structure/viewpoint)
        {'type': 'global_attn', 'rank': 32},
        {'type': 'global_attn', 'rank': 32},
    ]
    return configs[:num_experts]


def default_structure_expert_configs(num_experts=8):
    """Structure experts for point cloud branch."""
    configs = [
        {'type': 'pointwise', 'rank': 8},
        {'type': 'pointwise', 'rank': 8},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
        {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
        {'type': 'global_attn', 'rank': 32},
        {'type': 'global_attn', 'rank': 32},
    ]
    return configs[:num_experts]


def default_fusion_expert_ranks(num_experts=8):
    """Mixed-rank fusion experts (capacity heterogeneity)."""
    # 2 small + 3 medium + 3 large
    ranks = [16, 16, 32, 32, 32, 64, 64, 64]
    return ranks[:num_experts]



# """GaitMoEOracle V2Max-Calib — calibrated QACT + condition-aware MoE blocks.

# NEW over V2Pro:
#   (1) Multi-scale heterogeneous experts (Pointwise / LocalConv / GlobalAttn)
#   (2) Shared Expert Isolation (DeepSeekMoE, ACL'24)
#   (3) Capacity-heterogeneous experts (mixed rank)
#   (4) Branch-level auxiliary supervision heads
#   (5) Lightweight pre-fusion temporal aggregation
#   (6) Modality-aware dynamic router
#   (7) Identity-nuisance orthogonality loss
#   (8) Hard-negative cross-modal contrastive loss
#   (9) Quality-weighted modality dropout
# """

# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# # Re-export everything from V2Pro blocks
# from .gait_moe_blocks_v2pro import (
#     PartLevelAlignBeforeFuse,
#     cross_modal_alignment_loss,
#     part_level_cross_modal_contrastive_loss,
#     intra_modal_separation_loss,
# )
# from .gait_moe_blocks_v2plus import (
#     ModalityQualityEstimator, EnhancedConditionEncoder,
#     QualityAwareFusionExpertBlock,
#     moe_z_loss, moe_load_balance_loss_v2,
#     modality_quality_contrastive_loss,
#     expert_utilization_stats,
#     collect_expert_usage, collect_appearance_expert_parts,
#     collect_fusion_token_preference,
#     ModalityDropout,
# )
# from .gait_moe_blocks import (
#     l2n, _scalar, symmetric_infonce, masked_symmetric_infonce,
#     TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
#     PositiveTextGate, StagedNuisanceRemoval,
#     CrossAttentionFusion, MLPFusion,
#     CycleAwareTemporalEncoder, RuleBasedVerifier,
#     compute_condition_stats,
#     moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
#     PartExpertBlock,
# )
# from ..modules import SeparateFCs, SeparateBNNecks


# # ================================================================
# #  Multi-Scale Heterogeneous Experts (Method A)
# # ================================================================

# class PointwiseExpert(nn.Module):
#     """Pointwise (per-part independent) expert.
#     Inductive bias: captures channel-level patterns within individual parts.
#     Suitable for texture/appearance per body region.
#     """
#     def __init__(self, dim, parts_num, rank=16, dropout=0.0):
#         super().__init__()
#         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
#         self.transform = nn.Sequential(
#             nn.Linear(dim, rank, bias=False), nn.GELU(),
#             nn.Linear(rank, dim, bias=False),
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.scale = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         """x: [N, C, P] -> [N, C, P]"""
#         pa = torch.sigmoid(self.part_logit)
#         h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
#         delta = self.drop(self.transform(self.norm(h)))
#         h = h + self.scale * delta
#         return h.permute(0, 2, 1).contiguous()


# class LocalConvExpert(nn.Module):
#     """Local-neighbor expert using Conv1d over parts dimension.
#     Inductive bias: captures relationships between adjacent body parts.
#     Suitable for limb coordination patterns.
#     """
#     def __init__(self, dim, parts_num, rank=16, kernel_size=3, dropout=0.0):
#         super().__init__()
#         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
#         # Depthwise conv over parts + pointwise projection
#         self.local_conv = nn.Conv1d(
#             dim, dim, kernel_size, padding=kernel_size // 2,
#             groups=min(dim, 32), bias=False)
#         self.transform = nn.Sequential(
#             nn.Linear(dim, rank, bias=False), nn.GELU(),
#             nn.Linear(rank, dim, bias=False),
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.scale = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         """x: [N, C, P] -> [N, C, P]"""
#         pa = torch.sigmoid(self.part_logit)
#         h = x * pa.unsqueeze(0).unsqueeze(0)
#         # Local conv over parts: [N, C, P]
#         h = self.local_conv(h)
#         h = h.permute(0, 2, 1)  # [N, P, C]
#         delta = self.drop(self.transform(self.norm(h)))
#         h = h + self.scale * delta
#         return h.permute(0, 2, 1).contiguous()


# class GlobalAttentionExpert(nn.Module):
#     """Global attention expert attending over all parts.
#     Inductive bias: captures holistic body structure and viewpoint patterns.
#     Suitable for overall silhouette/contour and view-dependent features.
#     """
#     def __init__(self, dim, parts_num, rank=32, num_heads=2, dropout=0.0):
#         super().__init__()
#         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
#         self.attn = nn.MultiheadAttention(
#             dim, num_heads, dropout=dropout, batch_first=True)
#         self.transform = nn.Sequential(
#             nn.Linear(dim, rank, bias=False), nn.GELU(),
#             nn.Linear(rank, dim, bias=False),
#         )
#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.scale = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x):
#         """x: [N, C, P] -> [N, C, P]"""
#         pa = torch.sigmoid(self.part_logit)
#         h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
#         h_normed = self.norm1(h)
#         attn_out, _ = self.attn(h_normed, h_normed, h_normed)
#         h = h + attn_out
#         delta = self.drop(self.transform(self.norm2(h)))
#         h = h + self.scale * delta
#         return h.permute(0, 2, 1).contiguous()


# class SharedExpert(nn.Module):
#     """Always-on shared expert (DeepSeekMoE).
#     Captures universal patterns common to all inputs.
#     """
#     def __init__(self, dim, parts_num, rank=16):
#         super().__init__()
#         self.transform = nn.Sequential(
#             nn.Linear(dim, rank, bias=False), nn.GELU(),
#             nn.LayerNorm(rank),
#             nn.Linear(rank, dim, bias=False),
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.scale = nn.Parameter(torch.tensor(0.3))

#     def forward(self, x):
#         """x: [N, C, P] -> [N, C, P]"""
#         h = x.permute(0, 2, 1)  # [N, P, C]
#         delta = self.transform(self.norm(h))
#         h = h + self.scale * delta
#         return h.permute(0, 2, 1).contiguous()


# # ================================================================
# #  Heterogeneous TopK MoE Layer
# # ================================================================

# def build_expert(expert_type, dim, parts_num, rank, dropout, kernel_size=3):
#     """Factory for building heterogeneous experts."""
#     if expert_type == 'pointwise':
#         return PointwiseExpert(dim, parts_num, rank, dropout)
#     elif expert_type == 'local_conv':
#         return LocalConvExpert(dim, parts_num, rank, kernel_size, dropout)
#     elif expert_type == 'global_attn':
#         return GlobalAttentionExpert(dim, parts_num, rank, 2, dropout)
#     elif expert_type == 'standard':
#         return PartExpertBlock(dim, parts_num, rank, dropout)
#     else:
#         raise ValueError(f'Unknown expert type: {expert_type}')


# class HeterogeneousMoELayer(nn.Module):
#     """MoE with shared expert + heterogeneous routed experts.

#     Combines:
#       - 1 shared expert (always-on, DeepSeekMoE)
#       - N routed experts with different structures/capacities
#     """
#     def __init__(self, dim, parts_num, expert_configs, top_k=2,
#                  shared_rank=16, dropout=0.0, shared_enabled=True):
#         """
#         Args:
#             expert_configs: list of dicts, each with keys:
#                 'type': 'pointwise'|'local_conv'|'global_attn'|'standard'
#                 'rank': int
#                 'kernel_size': int (for local_conv)
#         """
#         super().__init__()
#         self.num_experts = len(expert_configs)
#         self.top_k = top_k
#         self.shared_enabled = shared_enabled

#         # Build heterogeneous experts
#         self.experts = nn.ModuleList()
#         for cfg in expert_configs:
#             self.experts.append(build_expert(
#                 cfg.get('type', 'standard'), dim, parts_num,
#                 cfg.get('rank', 16), dropout,
#                 cfg.get('kernel_size', 3)))

#         # Shared expert
#         if shared_enabled:
#             self.shared_expert = SharedExpert(dim, parts_num, shared_rank)
#         else:
#             self.shared_expert = None

#         # Router
#         self.router = nn.Sequential(
#             nn.Linear(dim, dim // 2), nn.ReLU(True),
#             nn.LayerNorm(dim // 2),
#             nn.Linear(dim // 2, self.num_experts),
#         )
#         self.residual_gate = nn.Parameter(torch.tensor(0.5))

#     def forward(self, x, seq_len=1):
#         """x: [N, C, P]; Returns (output, logits, weights)."""
#         N, C, P = x.shape

#         # --- Shared expert (always-on) ---
#         shared_out = self.shared_expert(x) if self.shared_expert is not None else None

#         # --- Routing at sequence level ---
#         if seq_len > 1:
#             B = N // seq_len
#             r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))
#         else:
#             B = N
#             r_in = x.mean(dim=-1)

#         logits = self.router(r_in)
#         _, topk_idx = logits.topk(self.top_k, dim=-1)
#         mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
#         weights_B = F.softmax(
#             logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

#         if seq_len > 1:
#             mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#             wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#         else:
#             mask_N, wgt_N = mask_B, weights_B

#         # --- Expert computation ---
#         expert_out = torch.zeros_like(x)
#         for i in range(self.num_experts):
#             sel = mask_N[:, i] > 0
#             if not sel.any():
#                 continue
#             e = self.experts[i](x[sel]).to(dtype=expert_out.dtype)
#             w = wgt_N[sel, i].to(dtype=expert_out.dtype).unsqueeze(-1).unsqueeze(-1)
#             expert_out[sel] = expert_out[sel] + w * e

#         # --- Combine: residual + routed + shared ---
#         alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
#         output = alpha * x + (1.0 - alpha) * expert_out
#         if shared_out is not None:
#             output = output + shared_out

#         return output, logits, weights_B


# # ================================================================
# #  Heterogeneous Quality-Aware Fusion MoE
# # ================================================================

# class HeterogeneousFusionMoELayer(nn.Module):
#     """Fusion MoE with shared expert + heterogeneous routed fusion experts.

#     Routed fusion experts have different attention ranks for capacity heterogeneity.
#     """
#     def __init__(self, dim, output_parts, num_input_tokens,
#                  expert_ranks, top_k=2, cond_dim=64,
#                  dropout=0.0, shared_enabled=True, shared_rank=32,
#                  modality_aware_router=False):
#         """
#         Args:
#             expert_ranks: list of int, one per expert. E.g. [16,16,32,32,32,64,64,64]
#         """
#         super().__init__()
#         self.num_experts = len(expert_ranks)
#         self.top_k = top_k
#         self.output_parts = output_parts
#         self.shared_enabled = shared_enabled
#         self.modality_aware_router = modality_aware_router

#         # Heterogeneous fusion experts
#         self.experts = nn.ModuleList([
#             QualityAwareFusionExpertBlock(
#                 dim, output_parts, num_input_tokens, rank=r)
#             for r in expert_ranks
#         ])

#         # Shared fusion expert
#         if shared_enabled:
#             self.shared_fusion = QualityAwareFusionExpertBlock(
#                 dim, output_parts, num_input_tokens, rank=shared_rank)
#             self.shared_scale = nn.Parameter(torch.tensor(0.3))
#         else:
#             self.shared_fusion = None

#         # Router
#         router_in = dim + cond_dim + 2
#         if modality_aware_router:
#             self.mod_embed = nn.Embedding(2, dim // 8)
#             router_in += dim // 8
#         self.router = nn.Sequential(
#             nn.Linear(router_in, dim // 2), nn.ReLU(True),
#             nn.LayerNorm(dim // 2),
#             nn.Linear(dim // 2, self.num_experts),
#         )
#         # Residual
#         self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
#         self.residual_gate = nn.Parameter(torch.tensor(0.5))

#     def forward(self, tokens, cond_emb, quality_scores, seq_len=1):
#         N, TK, Cd = tokens.shape
#         P = self.output_parts
#         half = TK // 2

#         # --- Shared fusion (always-on) ---
#         shared_out = None
#         if self.shared_fusion is not None:
#             qs_for_shared = quality_scores if quality_scores is not None else None
#             if qs_for_shared is not None and qs_for_shared.size(0) != N and seq_len > 1:
#                 B_s = N // seq_len
#                 qs_for_shared = qs_for_shared.unsqueeze(1).expand(
#                     -1, seq_len, -1).reshape(N, -1)
#             shared_out = self.shared_fusion(tokens, qs_for_shared)
#             shared_out = shared_out * self.shared_scale

#         # --- Routing at sequence level ---
#         if seq_len > 1:
#             B = N // seq_len
#             fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
#         else:
#             B = N
#             fp = tokens.mean(dim=1)

#         parts = [fp]
#         if cond_emb is not None and cond_emb.size(-1) > 0:
#             ce = cond_emb
#             if ce.size(0) == N and seq_len > 1:
#                 ce = ce.reshape(B, seq_len, -1).mean(dim=1)
#             elif ce.size(0) != B:
#                 ce = fp.new_zeros(B, cond_emb.size(-1))
#             parts.append(ce.to(device=fp.device, dtype=fp.dtype))

#         qs = quality_scores
#         if qs is not None and qs.size(-1) > 0:
#             if qs.size(0) == N and seq_len > 1:
#                 qs_r = qs.reshape(B, seq_len, -1).mean(dim=1)
#             elif qs.size(0) == B:
#                 qs_r = qs
#             else:
#                 qs_r = fp.new_zeros(B, 2)
#             parts.append(qs_r.to(device=fp.device, dtype=fp.dtype))
#         else:
#             parts.append(fp.new_zeros(B, 2))

#         if self.modality_aware_router:
#             # Average modality embedding
#             mod_ids = torch.zeros(B, TK, dtype=torch.long, device=fp.device)
#             mod_ids[:, half:] = 1
#             if seq_len > 1:
#                 mod_ids_s = mod_ids[:1].expand(B, -1)
#             else:
#                 mod_ids_s = mod_ids
#             me = self.mod_embed(mod_ids_s).mean(dim=1)  # [B, dim//8]
#             parts.append(me)

#         ri = torch.cat(parts, dim=-1)
#         logits = self.router(ri)
#         _, ti = logits.topk(self.top_k, dim=-1)
#         mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
#         wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), dim=-1)

#         if seq_len > 1:
#             mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#             wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#             qs_N = quality_scores
#             if qs_N is not None and qs_N.size(0) == B:
#                 qs_N = qs_N.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
#         else:
#             mN, wN = mB, wB
#             qs_N = quality_scores

#         # --- Expert computation ---
#         out = torch.zeros(N, P, Cd, device=tokens.device, dtype=tokens.dtype)
#         for i in range(self.num_experts):
#             sel = mN[:, i] > 0
#             if not sel.any():
#                 continue
#             qs_sel = qs_N[sel] if qs_N is not None else None
#             eo = self.experts[i](tokens[sel], qs_sel).to(dtype=out.dtype)
#             w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
#             out[sel] = out[sel] + w * eo

#         # Residual
#         alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
#         res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, P, Cd)
#         res = res.to(dtype=out.dtype)
#         out = alpha * res + (1.0 - alpha) * out

#         # Add shared
#         if shared_out is not None:
#             out = out + shared_out.to(dtype=out.dtype)

#         return out.permute(0, 2, 1).contiguous(), logits, wB


# # ================================================================
# #  Lightweight Pre-Fusion Temporal Aggregation
# # ================================================================

# class LightTemporalAgg(nn.Module):
#     """Lightweight temporal context via depthwise Conv1d.
#     < 2K params. Gated residual ensures no degradation.
#     """
#     def __init__(self, channels, kernel_size=3):
#         super().__init__()
#         self.conv = nn.Conv1d(
#             channels, channels, kernel_size,
#             padding=kernel_size // 2, groups=channels, bias=False)
#         self.norm = nn.LayerNorm(channels)
#         self.gate = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())
#         # Init gate bias to negative so gate starts near 0
#         nn.init.constant_(self.gate[0].bias, -2.0)

#     def forward(self, x):
#         """x: [B, T, C, P] -> [B, T, C, P]"""
#         B, T, C, P = x.shape
#         if T <= 1:
#             return x
#         xp = x.permute(0, 3, 2, 1).reshape(B * P, C, T)
#         h = self.conv(xp).reshape(B, P, C, T).permute(0, 3, 2, 1)
#         h_n = self.norm(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
#         g = self.gate(h_n.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
#         return x + g * h_n


# # ================================================================
# #  Branch Auxiliary Head
# # ================================================================

# class BranchAuxHead(nn.Module):
#     """Lightweight auxiliary classification head for branch-level supervision.
#     Discarded at inference.
#     """
#     def __init__(self, in_channels, out_channels, parts_num, class_num):
#         super().__init__()
#         self.fc = SeparateFCs(parts_num, in_channels, out_channels)
#         self.bn = SeparateBNNecks(parts_num, out_channels, class_num)

#     def forward(self, x):
#         """x: [B, C, P] -> (embed [B, out_c, P], logits [B, class_num, P])"""
#         e = self.fc(x)
#         _, logits = self.bn(e)
#         return e, logits


# # ================================================================
# #  Quality-Weighted Modality Dropout
# # ================================================================

# class QualityWeightedModalityDropout(nn.Module):
#     """Drop the lower-quality modality more frequently."""
#     def __init__(self, p=0.1, quality_weighted=True):
#         super().__init__()
#         self.p = p
#         self.quality_weighted = quality_weighted

#     def forward(self, rgb_tokens, pc_tokens,
#                 rgb_quality=None, pc_quality=None):
#         if not self.training or self.p <= 0:
#             return rgb_tokens, pc_tokens

#         if torch.rand(1).item() >= self.p:
#             return rgb_tokens, pc_tokens

#         # Decide which modality to drop
#         if self.quality_weighted and rgb_quality is not None and pc_quality is not None:
#             rq = rgb_quality.mean().item()
#             pq = pc_quality.mean().item()
#             # Higher quality -> lower drop probability
#             total = rq + pq + 1e-8
#             drop_rgb_prob = pq / total  # drop RGB more when PC is better
#         else:
#             drop_rgb_prob = 0.5

#         if torch.rand(1).item() < drop_rgb_prob:
#             rgb_tokens = torch.zeros_like(rgb_tokens)
#         else:
#             pc_tokens = torch.zeros_like(pc_tokens)

#         return rgb_tokens, pc_tokens


# # ================================================================
# #  Losses
# # ================================================================

# def identity_nuisance_orthogonality_loss(id_feat, nuisance_feats):
#     """Explicit orthogonality between identity and nuisance features.

#     Args:
#         id_feat: [B, D] identity features
#         nuisance_feats: dict {factor_name: [B, D_sub]} visual probe outputs
#     Returns:
#         scalar loss
#     """
#     loss = id_feat.new_tensor(0.0)
#     id_norm = F.normalize(id_feat.float(), dim=-1)
#     cnt = 0
#     for k, z_f in nuisance_feats.items():
#         if z_f is None:
#             continue
#         z_f_norm = F.normalize(z_f.detach().float(), dim=-1)
#         if z_f_norm.size(-1) != id_norm.size(-1):
#             continue
#         cos_sim = (id_norm * z_f_norm).sum(dim=-1)
#         loss = loss + cos_sim.pow(2).mean()
#         cnt += 1
#     return loss / max(cnt, 1)


# def part_level_cross_modal_contrastive_loss_v2(
#         rgb_align, pc_align, labels=None, temperature=0.07,
#         hard_negative=True, hard_ratio=0.5):
#     """Part-level cross-modal contrastive with hard negative mining."""
#     B, P, D = rgb_align.shape
#     if B <= 1:
#         return rgb_align.new_tensor(0.0)

#     loss = rgb_align.new_tensor(0.0)
#     for p in range(P):
#         rp = rgb_align[:, p, :]
#         pp = pc_align[:, p, :]
#         logits = rp @ pp.t() / temperature

#         if labels is not None:
#             lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#             neg_mask = 1.0 - lbl_eq

#             if hard_negative and neg_mask.sum() > 0:
#                 pos_sim = logits * lbl_eq + (-1e9) * neg_mask
#                 hardest_pos_val = pos_sim.min(dim=1).values
#                 semi_hard = (
#                     logits > (hardest_pos_val.unsqueeze(1) - 0.3)
#                 ).float() * neg_mask
#                 semi_hard = torch.where(
#                     semi_hard.sum(1, keepdim=True) > 0, semi_hard, neg_mask)
#                 weight = semi_hard * hard_ratio + neg_mask * (1 - hard_ratio)
#             else:
#                 weight = neg_mask

#             pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
#             log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
#             loss_p = -(pos_mask * log_prob).sum(dim=1).mean()
#             # Symmetric
#             log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
#             loss_p = loss_p - (pos_mask.t() * log_prob_t).sum(dim=1).mean()
#             loss = loss + loss_p * 0.5
#         else:
#             targets = torch.arange(B, device=rgb_align.device)
#             loss = loss + 0.5 * (
#                 F.cross_entropy(logits, targets) +
#                 F.cross_entropy(logits.t(), targets))

#     return loss / P


# def cross_modal_alignment_loss_v2(
#         rgb_align, pc_align, labels=None,
#         plc_weight=1.0, ims_weight=0.3, temperature=0.07,
#         hard_negative=True, hard_ratio=0.5):
#     """Combined alignment loss with hard negative mining."""
#     plc = part_level_cross_modal_contrastive_loss_v2(
#         rgb_align, pc_align, labels, temperature,
#         hard_negative, hard_ratio)
#     ims_rgb = intra_modal_separation_loss(rgb_align, labels)
#     ims_pc = intra_modal_separation_loss(pc_align, labels)
#     return plc_weight * plc + ims_weight * 0.5 * (ims_rgb + ims_pc)


# # ================================================================
# #  Heterogeneous Expert Diversity Loss
# # ================================================================

# def heterogeneous_diversity_loss(experts, thr=0.5):
#     """Diversity loss adapted for heterogeneous experts.
#     Only compares experts that share the same attribute (part_logit or token_logit).
#     """
#     bs = []
#     for e in experts:
#         if hasattr(e, 'part_logit'):
#             bs.append(torch.sigmoid(e.part_logit))
#     if len(bs) < 2:
#         return torch.tensor(0.0, device=bs[0].device) if bs else torch.tensor(0.0)
#     s = F.normalize(torch.stack(bs), dim=-1)
#     sim = s @ s.t()
#     E = sim.size(0)
#     mask = ~torch.eye(E, dtype=torch.bool, device=sim.device)
#     return F.relu(sim[mask] - thr).mean()


# # ================================================================
# #  Quality-Aware Context Tokens (QACT)
# #
# #  Replaces: compute_condition_stats, ModalityQualityEstimator,
# #            EnhancedConditionEncoder, modality_quality_contrastive_loss
# #
# #  Theory: Nagrani et al. "Attention Bottlenecks" (NeurIPS 2022),
# #          Zhang et al. "CMNeXt Self-Query Hub" (CVPR 2023),
# #          Wang et al. "TokenFusion" (CVPR 2022).
# # ================================================================

# class QualityContextTokens(nn.Module):
#     """Learnable context tokens that probe each modality's part features.

#     Outputs rich [B, K, C] context per modality instead of scalar quality.
#     Each downstream consumer (router, expert, dropout) projects what IT needs.

#     Architecture:
#       Stage 1: Self-probe   — each modality's tokens attend to own parts
#       Stage 2: Cross-probe  — each modality's tokens attend to OTHER's parts
#       Stage 3: Refinement   — joint self-attention among all 2K tokens
#     """

#     def __init__(self, channels, num_tokens=4, num_heads=4,
#                  cond_dim=64, dropout=0.1):
#         super().__init__()
#         self.channels = channels
#         self.num_tokens = num_tokens
#         self.cond_dim = cond_dim

#         # Learnable context tokens
#         self.rgb_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)
#         self.pc_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)

#         # Stage 1: Self-probe
#         self.rgb_self_attn = nn.MultiheadAttention(
#             channels, num_heads, dropout=dropout, batch_first=True)
#         self.pc_self_attn = nn.MultiheadAttention(
#             channels, num_heads, dropout=dropout, batch_first=True)
#         self.rgb_self_norm_q = nn.LayerNorm(channels)
#         self.rgb_self_norm_kv = nn.LayerNorm(channels)
#         self.pc_self_norm_q = nn.LayerNorm(channels)
#         self.pc_self_norm_kv = nn.LayerNorm(channels)

#         # Stage 2: Cross-probe
#         self.rgb_cross_attn = nn.MultiheadAttention(
#             channels, num_heads, dropout=dropout, batch_first=True)
#         self.pc_cross_attn = nn.MultiheadAttention(
#             channels, num_heads, dropout=dropout, batch_first=True)
#         self.rgb_cross_norm = nn.LayerNorm(channels)
#         self.pc_cross_norm = nn.LayerNorm(channels)
#         self.rgb_cross_gate = nn.Sequential(
#             nn.Linear(channels * 2, channels), nn.Sigmoid())
#         self.pc_cross_gate = nn.Sequential(
#             nn.Linear(channels * 2, channels), nn.Sigmoid())

#         # Stage 3: Joint refinement
#         self.refine_attn = nn.MultiheadAttention(
#             channels, num_heads, dropout=dropout, batch_first=True)
#         self.refine_norm = nn.LayerNorm(channels)
#         self.refine_ffn = nn.Sequential(
#             nn.Linear(channels, channels * 2), nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(channels * 2, channels))
#         self.refine_ffn_norm = nn.LayerNorm(channels)

#         # Output: condition embedding for MoE router
#         self.cond_proj = nn.Sequential(
#             nn.Linear(channels * num_tokens * 2, channels), nn.GELU(),
#             nn.LayerNorm(channels),
#             nn.Linear(channels, cond_dim))

#     def forward(self, rgb_parts, pc_parts):
#         """
#         Args:
#             rgb_parts: [B, T, C, P]
#             pc_parts:  [B, T, C, P]
#         Returns:
#             dict with rgb_ctx, pc_ctx, cond_emb, quality_scores, attn weights
#         """
#         # Temporal mean → [B, P, C]
#         if rgb_parts.dim() == 4:
#             rgb_kv = rgb_parts.mean(dim=1).permute(0, 2, 1)
#             pc_kv = pc_parts.mean(dim=1).permute(0, 2, 1)
#         else:
#             rgb_kv = rgb_parts.permute(0, 2, 1)
#             pc_kv = pc_parts.permute(0, 2, 1)

#         B, P, C = rgb_kv.shape
#         K = self.num_tokens

#         rq = self.rgb_tokens.unsqueeze(0).expand(B, -1, -1)
#         pq = self.pc_tokens.unsqueeze(0).expand(B, -1, -1)

#         # ---- Stage 1: Self-probe ----
#         rq_self, r_self_w = self.rgb_self_attn(
#             self.rgb_self_norm_q(rq),
#             self.rgb_self_norm_kv(rgb_kv), rgb_kv)
#         rq = rq + rq_self

#         pq_self, p_self_w = self.pc_self_attn(
#             self.pc_self_norm_q(pq),
#             self.pc_self_norm_kv(pc_kv), pc_kv)
#         pq = pq + pq_self

#         # ---- Stage 2: Cross-probe ----
#         rq_cross, _ = self.rgb_cross_attn(
#             self.rgb_cross_norm(rq), pc_kv, pc_kv)
#         rg = self.rgb_cross_gate(torch.cat([rq, rq_cross], dim=-1))
#         rq = rq + rg * rq_cross

#         pq_cross, _ = self.pc_cross_attn(
#             self.pc_cross_norm(pq), rgb_kv, rgb_kv)
#         pg = self.pc_cross_gate(torch.cat([pq, pq_cross], dim=-1))
#         pq = pq + pg * pq_cross

#         # ---- Stage 3: Joint refinement ----
#         all_ctx = torch.cat([rq, pq], dim=1)
#         all_refined, _ = self.refine_attn(
#             self.refine_norm(all_ctx), self.refine_norm(all_ctx), all_ctx)
#         all_ctx = all_ctx + all_refined
#         all_ctx = all_ctx + self.refine_ffn(self.refine_ffn_norm(all_ctx))

#         rgb_ctx = all_ctx[:, :K]
#         pc_ctx = all_ctx[:, K:]

#         # ---- Outputs ----
#         cond_emb = self.cond_proj(torch.cat([
#             rgb_ctx.reshape(B, -1), pc_ctx.reshape(B, -1)], dim=-1))

#         # Quality scores from attention entropy (no scalar bottleneck —
#         # this is a derived diagnostic, not the primary signal)
#         rgb_q, pc_q = self._quality_from_entropy(r_self_w, p_self_w, P)

#         return {
#             'rgb_ctx': rgb_ctx,
#             'pc_ctx': pc_ctx,
#             'cond_emb': cond_emb,
#             'quality_scores': torch.cat([rgb_q, pc_q], dim=-1),
#             'rgb_self_attn_w': r_self_w.detach(),
#             'pc_self_attn_w': p_self_w.detach(),
#         }

#     @staticmethod
#     def _quality_from_entropy(rgb_attn_w, pc_attn_w, P):
#         """Derive [B,1] quality from attention entropy.
#         Low entropy → focused → high quality.
#         """
#         max_ent = math.log(max(P, 2))
#         def _ent(w):
#             e = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
#             return e.mean(dim=-1) / max_ent
#         return (1 - _ent(rgb_attn_w)).unsqueeze(-1).clamp(0, 1), \
#                (1 - _ent(pc_attn_w)).unsqueeze(-1).clamp(0, 1)


# class ContextAwareModalityDropout(nn.Module):
#     """Modality dropout using attention entropy instead of scalar quality.

#     Low entropy → sharp focus → structured info → high quality → low drop prob.
#     High entropy → uniform attention → noisy input → low quality → high drop prob.
#     """

#     def __init__(self, p=0.1):
#         super().__init__()
#         self.p = p

#     def forward(self, rgb_tokens, pc_tokens,
#                 rgb_self_attn_w=None, pc_self_attn_w=None):
#         if not self.training or self.p <= 0:
#             return rgb_tokens, pc_tokens
#         if torch.rand(1).item() >= self.p:
#             return rgb_tokens, pc_tokens

#         if rgb_self_attn_w is not None and pc_self_attn_w is not None:
#             def _ent(w):
#                 return -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()
#                          ).sum(dim=-1).mean().item()
#             re, pe = _ent(rgb_self_attn_w), _ent(pc_self_attn_w)
#             drop_rgb_prob = re / (re + pe + 1e-8)
#         else:
#             drop_rgb_prob = 0.5

#         if torch.rand(1).item() < drop_rgb_prob:
#             rgb_tokens = torch.zeros_like(rgb_tokens)
#         else:
#             pc_tokens = torch.zeros_like(pc_tokens)
#         return rgb_tokens, pc_tokens


# # ================================================================
# #  QACT Self-supervised Losses
# # ================================================================

# def context_cross_modal_consistency_loss(rgb_ctx, pc_ctx, labels,
#                                          temperature=0.1):
#     """Same-identity context tokens should encode similar scene conditions
#     across modalities. Replaces modality_quality_contrastive_loss."""
#     B = rgb_ctx.size(0)
#     if B <= 1:
#         return rgb_ctx.new_tensor(0.0)
#     rgb_r = F.normalize(rgb_ctx.mean(dim=1), dim=-1)
#     pc_r = F.normalize(pc_ctx.mean(dim=1), dim=-1)
#     logits = rgb_r @ pc_r.t() / temperature
#     if labels is not None:
#         lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
#         log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
#         loss_fwd = -(pos_mask * log_prob).sum(dim=1).mean()
#         log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
#         loss_bwd = -(pos_mask.t() * log_prob_t).sum(dim=1).mean()
#         return 0.5 * (loss_fwd + loss_bwd)
#     tgt = torch.arange(B, device=rgb_ctx.device)
#     return 0.5 * (F.cross_entropy(logits, tgt) +
#                    F.cross_entropy(logits.t(), tgt))


# def context_diversity_loss(rgb_ctx, pc_ctx, threshold=0.5):
#     """Prevent context token collapse — each token should capture
#     a different aspect of quality/condition."""
#     loss = rgb_ctx.new_tensor(0.0)
#     for ctx in [rgb_ctx, pc_ctx]:
#         tk = F.normalize(ctx.mean(dim=0), dim=-1)
#         sim = tk @ tk.t()
#         K = sim.size(0)
#         mask = ~torch.eye(K, dtype=torch.bool, device=sim.device)
#         loss = loss + F.relu(sim[mask] - threshold).mean()
#     return loss * 0.5


# # ================================================================
# #  Default Expert Configurations
# # ================================================================

# def default_appearance_expert_configs(num_experts=8):
#     """Method A + C: multi-scale + mixed rank."""
#     configs = [
#         # 2x Pointwise, small rank (simple local patterns)
#         {'type': 'pointwise', 'rank': 8},
#         {'type': 'pointwise', 'rank': 8},
#         # 2x LocalConv k=3, medium rank (adjacent part coordination)
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
#         # 2x LocalConv k=5, medium rank (wider part interactions)
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
#         # 2x GlobalAttn, large rank (holistic structure/viewpoint)
#         {'type': 'global_attn', 'rank': 32},
#         {'type': 'global_attn', 'rank': 32},
#     ]
#     return configs[:num_experts]


# def default_structure_expert_configs(num_experts=8):
#     """Structure experts for point cloud branch."""
#     configs = [
#         {'type': 'pointwise', 'rank': 8},
#         {'type': 'pointwise', 'rank': 8},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
#         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
#         {'type': 'global_attn', 'rank': 32},
#         {'type': 'global_attn', 'rank': 32},
#     ]
#     return configs[:num_experts]


# def default_fusion_expert_ranks(num_experts=8):
#     """Mixed-rank fusion experts (capacity heterogeneity)."""
#     # 2 small + 3 medium + 3 large
#     ranks = [16, 16, 32, 32, 32, 64, 64, 64]
#     return ranks[:num_experts]






# # """GaitMoEOracle V2Max-Calib — calibrated QACT + condition-aware MoE blocks.

# # NEW over V2Pro:
# #   (1) Multi-scale heterogeneous experts (Pointwise / LocalConv / GlobalAttn)
# #   (2) Shared Expert Isolation (DeepSeekMoE, ACL'24)
# #   (3) Capacity-heterogeneous experts (mixed rank)
# #   (4) Branch-level auxiliary supervision heads
# #   (5) Lightweight pre-fusion temporal aggregation
# #   (6) Modality-aware dynamic router
# #   (7) Identity-nuisance orthogonality loss
# #   (8) Hard-negative cross-modal contrastive loss
# #   (9) Quality-weighted modality dropout
# # """

# # import math
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from einops import rearrange


# # from .gait_moe_blocks_v2pro import (
# #     PartLevelAlignBeforeFuse,
# #     cross_modal_alignment_loss,
# #     part_level_cross_modal_contrastive_loss,
# #     intra_modal_separation_loss,
# # )
# # from .gait_moe_blocks_v2plus import (
# #     ModalityQualityEstimator, EnhancedConditionEncoder,
# #     QualityAwareFusionExpertBlock,
# #     moe_z_loss, moe_load_balance_loss_v2,
# #     modality_quality_contrastive_loss,
# #     expert_utilization_stats,
# #     collect_expert_usage, collect_appearance_expert_parts,
# #     collect_fusion_token_preference,
# #     ModalityDropout,
# # )
# # from .gait_moe_blocks import (
# #     l2n, _scalar, symmetric_infonce, masked_symmetric_infonce,
# #     TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
# #     PositiveTextGate, StagedNuisanceRemoval,
# #     CrossAttentionFusion, MLPFusion,
# #     CycleAwareTemporalEncoder, RuleBasedVerifier,
# #     compute_condition_stats,
# #     moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
# #     PartExpertBlock,
# # )
# # from ..modules import SeparateFCs, SeparateBNNecks


# # # ================================================================
# # #  Multi-Scale Heterogeneous Experts (Method A)
# # # ================================================================

# # class PointwiseExpert(nn.Module):
# #     """Pointwise (per-part independent) expert.
# #     Inductive bias: captures channel-level patterns within individual parts.
# #     Suitable for texture/appearance per body region.
# #     """
# #     def __init__(self, dim, parts_num, rank=16, dropout=0.0):
# #         super().__init__()
# #         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
# #         self.transform = nn.Sequential(
# #             nn.Linear(dim, rank, bias=False), nn.GELU(),
# #             nn.Linear(rank, dim, bias=False),
# #         )
# #         self.norm = nn.LayerNorm(dim)
# #         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
# #         self.scale = nn.Parameter(torch.tensor(0.5))

# #     def forward(self, x):
# #         """x: [N, C, P] -> [N, C, P]"""
# #         pa = torch.sigmoid(self.part_logit)
# #         h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
# #         delta = self.drop(self.transform(self.norm(h)))
# #         h = h + self.scale * delta
# #         return h.permute(0, 2, 1).contiguous()


# # class LocalConvExpert(nn.Module):
# #     """Local-neighbor expert using Conv1d over parts dimension.
# #     Inductive bias: captures relationships between adjacent body parts.
# #     Suitable for limb coordination patterns.
# #     """
# #     def __init__(self, dim, parts_num, rank=16, kernel_size=3, dropout=0.0):
# #         super().__init__()
# #         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
# #         # Depthwise conv over parts + pointwise projection
# #         self.local_conv = nn.Conv1d(
# #             dim, dim, kernel_size, padding=kernel_size // 2,
# #             groups=min(dim, 32), bias=False)
# #         self.transform = nn.Sequential(
# #             nn.Linear(dim, rank, bias=False), nn.GELU(),
# #             nn.Linear(rank, dim, bias=False),
# #         )
# #         self.norm = nn.LayerNorm(dim)
# #         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
# #         self.scale = nn.Parameter(torch.tensor(0.5))

# #     def forward(self, x):
# #         """x: [N, C, P] -> [N, C, P]"""
# #         pa = torch.sigmoid(self.part_logit)
# #         h = x * pa.unsqueeze(0).unsqueeze(0)
# #         # Local conv over parts: [N, C, P]
# #         h = self.local_conv(h)
# #         h = h.permute(0, 2, 1)  # [N, P, C]
# #         delta = self.drop(self.transform(self.norm(h)))
# #         h = h + self.scale * delta
# #         return h.permute(0, 2, 1).contiguous()


# # class GlobalAttentionExpert(nn.Module):
# #     """Global attention expert attending over all parts.
# #     Inductive bias: captures holistic body structure and viewpoint patterns.
# #     Suitable for overall silhouette/contour and view-dependent features.
# #     """
# #     def __init__(self, dim, parts_num, rank=32, num_heads=2, dropout=0.0):
# #         super().__init__()
# #         self.part_logit = nn.Parameter(torch.full((parts_num,), 2.0))
# #         self.attn = nn.MultiheadAttention(
# #             dim, num_heads, dropout=dropout, batch_first=True)
# #         self.transform = nn.Sequential(
# #             nn.Linear(dim, rank, bias=False), nn.GELU(),
# #             nn.Linear(rank, dim, bias=False),
# #         )
# #         self.norm1 = nn.LayerNorm(dim)
# #         self.norm2 = nn.LayerNorm(dim)
# #         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
# #         self.scale = nn.Parameter(torch.tensor(0.5))

# #     def forward(self, x):
# #         """x: [N, C, P] -> [N, C, P]"""
# #         pa = torch.sigmoid(self.part_logit)
# #         h = (x * pa.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)  # [N, P, C]
# #         h_normed = self.norm1(h)
# #         attn_out, _ = self.attn(h_normed, h_normed, h_normed)
# #         h = h + attn_out
# #         delta = self.drop(self.transform(self.norm2(h)))
# #         h = h + self.scale * delta
# #         return h.permute(0, 2, 1).contiguous()


# # class SharedExpert(nn.Module):
# #     """Always-on shared expert (DeepSeekMoE).
# #     Captures universal patterns common to all inputs.
# #     """
# #     def __init__(self, dim, parts_num, rank=16):
# #         super().__init__()
# #         self.transform = nn.Sequential(
# #             nn.Linear(dim, rank, bias=False), nn.GELU(),
# #             nn.LayerNorm(rank),
# #             nn.Linear(rank, dim, bias=False),
# #         )
# #         self.norm = nn.LayerNorm(dim)
# #         self.scale = nn.Parameter(torch.tensor(0.3))

# #     def forward(self, x):
# #         """x: [N, C, P] -> [N, C, P]"""
# #         h = x.permute(0, 2, 1)  # [N, P, C]
# #         delta = self.transform(self.norm(h))
# #         h = h + self.scale * delta
# #         return h.permute(0, 2, 1).contiguous()


# # # ================================================================
# # #  Heterogeneous TopK MoE Layer
# # # ================================================================

# # def build_expert(expert_type, dim, parts_num, rank, dropout, kernel_size=3):
# #     """Factory for building heterogeneous experts."""
# #     if expert_type == 'pointwise':
# #         return PointwiseExpert(dim, parts_num, rank, dropout)
# #     elif expert_type == 'local_conv':
# #         return LocalConvExpert(dim, parts_num, rank, kernel_size, dropout)
# #     elif expert_type == 'global_attn':
# #         return GlobalAttentionExpert(dim, parts_num, rank, 2, dropout)
# #     elif expert_type == 'standard':
# #         return PartExpertBlock(dim, parts_num, rank, dropout)
# #     else:
# #         raise ValueError(f'Unknown expert type: {expert_type}')


# # class HeterogeneousMoELayer(nn.Module):
# #     """MoE with shared expert + heterogeneous routed experts.

# #     Combines:
# #       - 1 shared expert (always-on, DeepSeekMoE)
# #       - N routed experts with different structures/capacities
# #     """
# #     def __init__(self, dim, parts_num, expert_configs, top_k=2,
# #                  shared_rank=16, dropout=0.0, shared_enabled=True):
# #         """
# #         Args:
# #             expert_configs: list of dicts, each with keys:
# #                 'type': 'pointwise'|'local_conv'|'global_attn'|'standard'
# #                 'rank': int
# #                 'kernel_size': int (for local_conv)
# #         """
# #         super().__init__()
# #         self.num_experts = len(expert_configs)
# #         self.top_k = top_k
# #         self.shared_enabled = shared_enabled

# #         # Build heterogeneous experts
# #         self.experts = nn.ModuleList()
# #         for cfg in expert_configs:
# #             self.experts.append(build_expert(
# #                 cfg.get('type', 'standard'), dim, parts_num,
# #                 cfg.get('rank', 16), dropout,
# #                 cfg.get('kernel_size', 3)))

# #         # Shared expert
# #         if shared_enabled:
# #             self.shared_expert = SharedExpert(dim, parts_num, shared_rank)
# #         else:
# #             self.shared_expert = None

# #         # Router
# #         self.router = nn.Sequential(
# #             nn.Linear(dim, dim // 2), nn.ReLU(True),
# #             nn.LayerNorm(dim // 2),
# #             nn.Linear(dim // 2, self.num_experts),
# #         )
# #         self.residual_gate = nn.Parameter(torch.tensor(0.5))

# #     def forward(self, x, seq_len=1):
# #         """x: [N, C, P]; Returns (output, logits, weights)."""
# #         N, C, P = x.shape

# #         # --- Shared expert (always-on) ---
# #         shared_out = self.shared_expert(x) if self.shared_expert is not None else None

# #         # --- Routing at sequence level ---
# #         if seq_len > 1:
# #             B = N // seq_len
# #             r_in = x.reshape(B, seq_len, C, P).mean(dim=(1, 3))
# #         else:
# #             B = N
# #             r_in = x.mean(dim=-1)

# #         logits = self.router(r_in)
# #         _, topk_idx = logits.topk(self.top_k, dim=-1)
# #         mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
# #         weights_B = F.softmax(
# #             logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

# #         if seq_len > 1:
# #             mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
# #             wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
# #         else:
# #             mask_N, wgt_N = mask_B, weights_B

# #         # --- Expert computation ---
# #         expert_out = torch.zeros_like(x)
# #         for i in range(self.num_experts):
# #             sel = mask_N[:, i] > 0
# #             if not sel.any():
# #                 continue
# #             e = self.experts[i](x[sel]).to(dtype=expert_out.dtype)
# #             w = wgt_N[sel, i].to(dtype=expert_out.dtype).unsqueeze(-1).unsqueeze(-1)
# #             expert_out[sel] = expert_out[sel] + w * e

# #         # --- Combine: residual + routed + shared ---
# #         alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
# #         output = alpha * x + (1.0 - alpha) * expert_out
# #         if shared_out is not None:
# #             output = output + shared_out

# #         return output, logits, weights_B


# # # ================================================================
# # #  Heterogeneous Quality-Aware Fusion MoE
# # # ================================================================

# # class HeterogeneousFusionMoELayer(nn.Module):
# #     """Fusion MoE with shared expert + heterogeneous routed fusion experts.

# #     Routed fusion experts have different attention ranks for capacity heterogeneity.
# #     """
# #     def __init__(self, dim, output_parts, num_input_tokens,
# #                  expert_ranks, top_k=2, cond_dim=64,
# #                  dropout=0.0, shared_enabled=True, shared_rank=32,
# #                  modality_aware_router=False):
# #         """
# #         Args:
# #             expert_ranks: list of int, one per expert. E.g. [16,16,32,32,32,64,64,64]
# #         """
# #         super().__init__()
# #         self.num_experts = len(expert_ranks)
# #         self.top_k = top_k
# #         self.output_parts = output_parts
# #         self.shared_enabled = shared_enabled
# #         self.modality_aware_router = modality_aware_router

# #         # Heterogeneous fusion experts
# #         self.experts = nn.ModuleList([
# #             QualityAwareFusionExpertBlock(
# #                 dim, output_parts, num_input_tokens, rank=r)
# #             for r in expert_ranks
# #         ])

# #         # Shared fusion expert
# #         if shared_enabled:
# #             self.shared_fusion = QualityAwareFusionExpertBlock(
# #                 dim, output_parts, num_input_tokens, rank=shared_rank)
# #             self.shared_scale = nn.Parameter(torch.tensor(0.3))
# #         else:
# #             self.shared_fusion = None

# #         # Router
# #         router_in = dim + cond_dim + 2
# #         if modality_aware_router:
# #             self.mod_embed = nn.Embedding(2, dim // 8)
# #             router_in += dim // 8
# #         self.router = nn.Sequential(
# #             nn.Linear(router_in, dim // 2), nn.ReLU(True),
# #             nn.LayerNorm(dim // 2),
# #             nn.Linear(dim // 2, self.num_experts),
# #         )
# #         # Residual
# #         self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
# #         self.residual_gate = nn.Parameter(torch.tensor(0.5))

# #     def forward(self, tokens, cond_emb, quality_scores, seq_len=1):
# #         N, TK, Cd = tokens.shape
# #         P = self.output_parts
# #         half = TK // 2

# #         # --- Shared fusion (always-on) ---
# #         shared_out = None
# #         if self.shared_fusion is not None:
# #             qs_for_shared = quality_scores if quality_scores is not None else None
# #             if qs_for_shared is not None and qs_for_shared.size(0) != N and seq_len > 1:
# #                 B_s = N // seq_len
# #                 qs_for_shared = qs_for_shared.unsqueeze(1).expand(
# #                     -1, seq_len, -1).reshape(N, -1)
# #             shared_out = self.shared_fusion(tokens, qs_for_shared)
# #             shared_out = shared_out * self.shared_scale

# #         # --- Routing at sequence level ---
# #         if seq_len > 1:
# #             B = N // seq_len
# #             fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
# #         else:
# #             B = N
# #             fp = tokens.mean(dim=1)

# #         parts = [fp]
# #         if cond_emb is not None and cond_emb.size(-1) > 0:
# #             ce = cond_emb
# #             if ce.size(0) == N and seq_len > 1:
# #                 ce = ce.reshape(B, seq_len, -1).mean(dim=1)
# #             elif ce.size(0) != B:
# #                 ce = fp.new_zeros(B, cond_emb.size(-1))
# #             parts.append(ce.to(device=fp.device, dtype=fp.dtype))

# #         qs = quality_scores
# #         if qs is not None and qs.size(-1) > 0:
# #             if qs.size(0) == N and seq_len > 1:
# #                 qs_r = qs.reshape(B, seq_len, -1).mean(dim=1)
# #             elif qs.size(0) == B:
# #                 qs_r = qs
# #             else:
# #                 qs_r = fp.new_zeros(B, 2)
# #             parts.append(qs_r.to(device=fp.device, dtype=fp.dtype))
# #         else:
# #             parts.append(fp.new_zeros(B, 2))

# #         if self.modality_aware_router:
# #             # Average modality embedding
# #             mod_ids = torch.zeros(B, TK, dtype=torch.long, device=fp.device)
# #             mod_ids[:, half:] = 1
# #             if seq_len > 1:
# #                 mod_ids_s = mod_ids[:1].expand(B, -1)
# #             else:
# #                 mod_ids_s = mod_ids
# #             me = self.mod_embed(mod_ids_s).mean(dim=1)  # [B, dim//8]
# #             parts.append(me)

# #         ri = torch.cat(parts, dim=-1)
# #         logits = self.router(ri)
# #         _, ti = logits.topk(self.top_k, dim=-1)
# #         mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
# #         wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), dim=-1)

# #         if seq_len > 1:
# #             mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
# #             wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
# #             qs_N = quality_scores
# #             if qs_N is not None and qs_N.size(0) == B:
# #                 qs_N = qs_N.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
# #         else:
# #             mN, wN = mB, wB
# #             qs_N = quality_scores

# #         # --- Expert computation ---
# #         out = torch.zeros(N, P, Cd, device=tokens.device, dtype=tokens.dtype)
# #         for i in range(self.num_experts):
# #             sel = mN[:, i] > 0
# #             if not sel.any():
# #                 continue
# #             qs_sel = qs_N[sel] if qs_N is not None else None
# #             eo = self.experts[i](tokens[sel], qs_sel).to(dtype=out.dtype)
# #             w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
# #             out[sel] = out[sel] + w * eo

# #         # Residual
# #         alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
# #         res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, P, Cd)
# #         res = res.to(dtype=out.dtype)
# #         out = alpha * res + (1.0 - alpha) * out

# #         # Add shared
# #         if shared_out is not None:
# #             out = out + shared_out.to(dtype=out.dtype)

# #         return out.permute(0, 2, 1).contiguous(), logits, wB


# # # ================================================================
# # #  Lightweight Pre-Fusion Temporal Aggregation
# # # ================================================================

# # class LightTemporalAgg(nn.Module):
# #     """Lightweight temporal context via depthwise Conv1d.
# #     < 2K params. Gated residual ensures no degradation.
# #     """
# #     def __init__(self, channels, kernel_size=3):
# #         super().__init__()
# #         self.conv = nn.Conv1d(
# #             channels, channels, kernel_size,
# #             padding=kernel_size // 2, groups=channels, bias=False)
# #         self.norm = nn.LayerNorm(channels)
# #         self.gate = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())
# #         # Init gate bias to negative so gate starts near 0
# #         nn.init.constant_(self.gate[0].bias, -2.0)

# #     def forward(self, x):
# #         """x: [B, T, C, P] -> [B, T, C, P]"""
# #         B, T, C, P = x.shape
# #         if T <= 1:
# #             return x
# #         xp = x.permute(0, 3, 2, 1).reshape(B * P, C, T)
# #         h = self.conv(xp).reshape(B, P, C, T).permute(0, 3, 2, 1)
# #         h_n = self.norm(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
# #         g = self.gate(h_n.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
# #         return x + g * h_n


# # # ================================================================
# # #  Branch Auxiliary Head
# # # ================================================================

# # class BranchAuxHead(nn.Module):
# #     """Lightweight auxiliary classification head for branch-level supervision.
# #     Discarded at inference.
# #     """
# #     def __init__(self, in_channels, out_channels, parts_num, class_num):
# #         super().__init__()
# #         self.fc = SeparateFCs(parts_num, in_channels, out_channels)
# #         self.bn = SeparateBNNecks(parts_num, out_channels, class_num)

# #     def forward(self, x):
# #         """x: [B, C, P] -> (embed [B, out_c, P], logits [B, class_num, P])"""
# #         e = self.fc(x)
# #         _, logits = self.bn(e)
# #         return e, logits


# # # ================================================================
# # #  Quality-Weighted Modality Dropout
# # # ================================================================

# # class QualityWeightedModalityDropout(nn.Module):
# #     """Drop the lower-quality modality more frequently."""
# #     def __init__(self, p=0.1, quality_weighted=True):
# #         super().__init__()
# #         self.p = p
# #         self.quality_weighted = quality_weighted

# #     def forward(self, rgb_tokens, pc_tokens,
# #                 rgb_quality=None, pc_quality=None):
# #         if not self.training or self.p <= 0:
# #             return rgb_tokens, pc_tokens

# #         if torch.rand(1).item() >= self.p:
# #             return rgb_tokens, pc_tokens

# #         # Decide which modality to drop
# #         if self.quality_weighted and rgb_quality is not None and pc_quality is not None:
# #             rq = rgb_quality.mean().item()
# #             pq = pc_quality.mean().item()
# #             # Higher quality -> lower drop probability
# #             total = rq + pq + 1e-8
# #             drop_rgb_prob = pq / total  # drop RGB more when PC is better
# #         else:
# #             drop_rgb_prob = 0.5

# #         if torch.rand(1).item() < drop_rgb_prob:
# #             rgb_tokens = torch.zeros_like(rgb_tokens)
# #         else:
# #             pc_tokens = torch.zeros_like(pc_tokens)

# #         return rgb_tokens, pc_tokens


# # # ================================================================
# # #  Losses
# # # ================================================================

# # def identity_nuisance_orthogonality_loss(id_feat, nuisance_feats):
# #     """Explicit orthogonality between identity and nuisance features.

# #     Args:
# #         id_feat: [B, D] identity features
# #         nuisance_feats: dict {factor_name: [B, D_sub]} visual probe outputs
# #     Returns:
# #         scalar loss
# #     """
# #     loss = id_feat.new_tensor(0.0)
# #     id_norm = F.normalize(id_feat.float(), dim=-1)
# #     cnt = 0
# #     for k, z_f in nuisance_feats.items():
# #         if z_f is None:
# #             continue
# #         z_f_norm = F.normalize(z_f.detach().float(), dim=-1)
# #         if z_f_norm.size(-1) != id_norm.size(-1):
# #             continue
# #         cos_sim = (id_norm * z_f_norm).sum(dim=-1)
# #         loss = loss + cos_sim.pow(2).mean()
# #         cnt += 1
# #     return loss / max(cnt, 1)


# # def part_level_cross_modal_contrastive_loss_v2(
# #         rgb_align, pc_align, labels=None, temperature=0.07,
# #         hard_negative=True, hard_ratio=0.5):
# #     """Part-level cross-modal contrastive with hard negative mining."""
# #     B, P, D = rgb_align.shape
# #     if B <= 1:
# #         return rgb_align.new_tensor(0.0)

# #     loss = rgb_align.new_tensor(0.0)
# #     for p in range(P):
# #         rp = rgb_align[:, p, :]
# #         pp = pc_align[:, p, :]
# #         logits = rp @ pp.t() / temperature

# #         if labels is not None:
# #             lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
# #             neg_mask = 1.0 - lbl_eq

# #             if hard_negative and neg_mask.sum() > 0:
# #                 pos_sim = logits * lbl_eq + (-1e9) * neg_mask
# #                 hardest_pos_val = pos_sim.min(dim=1).values
# #                 semi_hard = (
# #                     logits > (hardest_pos_val.unsqueeze(1) - 0.3)
# #                 ).float() * neg_mask
# #                 semi_hard = torch.where(
# #                     semi_hard.sum(1, keepdim=True) > 0, semi_hard, neg_mask)
# #                 weight = semi_hard * hard_ratio + neg_mask * (1 - hard_ratio)
# #             else:
# #                 weight = neg_mask

# #             pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
# #             log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
# #             loss_p = -(pos_mask * log_prob).sum(dim=1).mean()
# #             # Symmetric
# #             log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
# #             loss_p = loss_p - (pos_mask.t() * log_prob_t).sum(dim=1).mean()
# #             loss = loss + loss_p * 0.5
# #         else:
# #             targets = torch.arange(B, device=rgb_align.device)
# #             loss = loss + 0.5 * (
# #                 F.cross_entropy(logits, targets) +
# #                 F.cross_entropy(logits.t(), targets))

# #     return loss / P


# # def cross_modal_alignment_loss_v2(
# #         rgb_align, pc_align, labels=None,
# #         plc_weight=1.0, ims_weight=0.3, temperature=0.07,
# #         hard_negative=True, hard_ratio=0.5):
# #     """Combined alignment loss with hard negative mining."""
# #     plc = part_level_cross_modal_contrastive_loss_v2(
# #         rgb_align, pc_align, labels, temperature,
# #         hard_negative, hard_ratio)
# #     ims_rgb = intra_modal_separation_loss(rgb_align, labels)
# #     ims_pc = intra_modal_separation_loss(pc_align, labels)
# #     return plc_weight * plc + ims_weight * 0.5 * (ims_rgb + ims_pc)


# # # ================================================================
# # #  Heterogeneous Expert Diversity Loss
# # # ================================================================

# # def heterogeneous_diversity_loss(experts, thr=0.5):
# #     """Diversity loss adapted for heterogeneous experts.
# #     Only compares experts that share the same attribute (part_logit or token_logit).
# #     """
# #     bs = []
# #     for e in experts:
# #         if hasattr(e, 'part_logit'):
# #             bs.append(torch.sigmoid(e.part_logit))
# #     if len(bs) < 2:
# #         return torch.tensor(0.0, device=bs[0].device) if bs else torch.tensor(0.0)
# #     s = F.normalize(torch.stack(bs), dim=-1)
# #     sim = s @ s.t()
# #     E = sim.size(0)
# #     mask = ~torch.eye(E, dtype=torch.bool, device=sim.device)
# #     return F.relu(sim[mask] - thr).mean()


# # # ================================================================
# # #  Default Expert Configurations
# # # ================================================================

# # def default_appearance_expert_configs(num_experts=8):
# #     """Method A + C: multi-scale + mixed rank."""
# #     configs = [
# #         # 2x Pointwise, small rank (simple local patterns)
# #         {'type': 'pointwise', 'rank': 8},
# #         {'type': 'pointwise', 'rank': 8},
# #         # 2x LocalConv k=3, medium rank (adjacent part coordination)
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
# #         # 2x LocalConv k=5, medium rank (wider part interactions)
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
# #         # 2x GlobalAttn, large rank (holistic structure/viewpoint)
# #         {'type': 'global_attn', 'rank': 32},
# #         {'type': 'global_attn', 'rank': 32},
# #     ]
# #     return configs[:num_experts]


# # def default_structure_expert_configs(num_experts=8):
# #     """Structure experts for point cloud branch."""
# #     configs = [
# #         {'type': 'pointwise', 'rank': 8},
# #         {'type': 'pointwise', 'rank': 8},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 3},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
# #         {'type': 'local_conv', 'rank': 16, 'kernel_size': 5},
# #         {'type': 'global_attn', 'rank': 32},
# #         {'type': 'global_attn', 'rank': 32},
# #     ]
# #     return configs[:num_experts]


# # def default_fusion_expert_ranks(num_experts=8):
# #     """Mixed-rank fusion experts (capacity heterogeneity)."""
# #     # 2 small + 3 medium + 3 large
# #     ranks = [16, 16, 32, 32, 32, 64, 64, 64]
# #     return ranks[:num_experts]
