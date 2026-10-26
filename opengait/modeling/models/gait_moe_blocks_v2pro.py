"""GaitMoEOracle V2Pro — Cross-Modal Alignment building blocks.

NEW over V2Plus:
  Fix-11: Part-Level Align-Before-Fuse (PLABF) module
          Inspired by ALBEF (NeurIPS 2021), CrossPoint (CVPR 2022),
          Cross-BERT (PIC+UMC), Mind the Gap (NeurIPS 2022),
          Mitigate the Gap (ICLR 2025), CL-Gait (ECCV 2024).

Design rationale:
  RGB and PC live in very different feature spaces (modality gap).
  Directly fusing them forces the MoE experts to simultaneously
  handle domain bridging AND gait discrimination — too much to ask.
  By aligning part-level features BEFORE fusion, the MoE experts
  only need to focus on gait discrimination in a shared space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Re-export everything from V2Plus blocks
from .gait_moe_blocks_v2plus import (
    ModalityQualityEstimator, EnhancedConditionEncoder,
    QualityAwareFusionMoELayer, TopKMoELayerV2Plus,
    ModalityDropout,
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss,
    expert_utilization_stats,
    collect_expert_usage, collect_appearance_expert_parts,
    collect_fusion_token_preference,
)
from .gait_moe_blocks import (
    l2n, _scalar, symmetric_infonce,
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    CrossAttentionFusion, MLPFusion,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    compute_condition_stats,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
)


# ================================================================
#  Fix-11: Part-Level Align-Before-Fuse (PLABF)
# ================================================================

class PartLevelAlignBeforeFuse(nn.Module):
    """Aligns RGB and PC part-level features before fusion.

    Three sub-mechanisms (following top-conference best practices):
      (a) Dual projection into shared alignment space
          → Part-level cross-modal contrastive loss (CrossPoint, PIC)
      (b) Cross-modal attention bridge
          → Each modality attends to the other for information exchange
      (c) Gated residual
          → Original features enhanced with cross-modal context

    References:
      - ALBEF (NeurIPS 2021): align representations before fusing
      - CrossPoint (CVPR 2022): 3D-2D cross-modal contrastive alignment
      - Cross-BERT: PIC + UMC (point-image + unimodal contrastive)
      - Mitigate the Gap (ICLR 2025): intra-modal separation + shared proj
    """

    def __init__(self, channels: int, align_dim: int = 128,
                 parts_num: int = 16, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.align_dim = align_dim
        self.parts_num = parts_num

        # (a) Dual projection heads into shared alignment space
        self.rgb_align_proj = nn.Sequential(
            nn.Linear(channels, align_dim * 2), nn.GELU(),
            nn.LayerNorm(align_dim * 2),
            nn.Linear(align_dim * 2, align_dim),
            nn.LayerNorm(align_dim),
        )
        self.pc_align_proj = nn.Sequential(
            nn.Linear(channels, align_dim * 2), nn.GELU(),
            nn.LayerNorm(align_dim * 2),
            nn.Linear(align_dim * 2, align_dim),
            nn.LayerNorm(align_dim),
        )

        # (b) Cross-modal attention bridge
        self.rgb_to_pc_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.pc_to_rgb_attn = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.rgb_norm = nn.LayerNorm(channels)
        self.pc_norm = nn.LayerNorm(channels)

        # (c) Gated residual: control how much cross-modal info to mix in
        self.rgb_gate = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.Sigmoid(),
        )
        self.pc_gate = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.Sigmoid(),
        )

    def forward(self, rgb_parts, pc_parts):
        """
        Args:
            rgb_parts: [B, T, C, P] — RGB part features after HPP
            pc_parts:  [B, T, C, P] — PC part features
        Returns:
            rgb_enhanced: [B, T, C, P] — RGB features enhanced with PC context
            pc_enhanced:  [B, T, C, P] — PC features enhanced with RGB context
            rgb_align:    [B, P, D_align] — L2-normalized, for contrastive loss
            pc_align:     [B, P, D_align] — L2-normalized, for contrastive loss
        """
        B, T, C, P = rgb_parts.shape

        # ---- (a) Project to alignment space (temporal mean for loss) ----
        rgb_mean = rgb_parts.mean(dim=1).permute(0, 2, 1)  # [B, P, C]
        pc_mean = pc_parts.mean(dim=1).permute(0, 2, 1)    # [B, P, C]
        rgb_align = F.normalize(self.rgb_align_proj(rgb_mean), dim=-1)  # [B, P, D]
        pc_align = F.normalize(self.pc_align_proj(pc_mean), dim=-1)     # [B, P, D]

        # ---- (b) Cross-modal attention bridge ----
        # Flatten temporal: [B*T, P, C]
        rgb_flat = rgb_parts.reshape(B * T, C, P).permute(0, 2, 1)
        pc_flat = pc_parts.reshape(B * T, C, P).permute(0, 2, 1)

        # RGB queries, PC provides K/V (and vice versa)
        rgb_from_pc, _ = self.rgb_to_pc_attn(
            self.rgb_norm(rgb_flat), self.pc_norm(pc_flat), pc_flat)
        pc_from_rgb, _ = self.pc_to_rgb_attn(
            self.pc_norm(pc_flat), self.rgb_norm(rgb_flat), rgb_flat)

        # ---- (c) Gated residual fusion ----
        rg = self.rgb_gate(torch.cat([rgb_flat, rgb_from_pc], dim=-1))
        pg = self.pc_gate(torch.cat([pc_flat, pc_from_rgb], dim=-1))

        rgb_enh = rgb_flat + rg * rgb_from_pc    # [B*T, P, C]
        pc_enh = pc_flat + pg * pc_from_rgb       # [B*T, P, C]

        rgb_enh = rgb_enh.permute(0, 2, 1).reshape(B, T, C, P)
        pc_enh = pc_enh.permute(0, 2, 1).reshape(B, T, C, P)

        return rgb_enh, pc_enh, rgb_align, pc_align


# ================================================================
#  Cross-Modal Alignment Losses
# ================================================================

def part_level_cross_modal_contrastive_loss(
        rgb_align, pc_align, labels=None, temperature=0.07):
    """Part-level cross-modal contrastive loss (PIC from Cross-BERT).

    For each body part, the RGB and PC aligned features of the same
    sample (or same identity) should be close; different identities far.

    Args:
        rgb_align: [B, P, D] — L2-normalized RGB alignment features
        pc_align:  [B, P, D] — L2-normalized PC alignment features
        labels:    [B] identity labels (optional; if None uses sample index)
        temperature: InfoNCE temperature
    Returns:
        scalar loss
    """
    B, P, D = rgb_align.shape
    if B <= 1:
        return rgb_align.new_tensor(0.0)

    loss = rgb_align.new_tensor(0.0)
    for p in range(P):
        rp = rgb_align[:, p, :]    # [B, D]
        pp = pc_align[:, p, :]     # [B, D]

        # Cross-modal logits: [B, B]
        logits = rp @ pp.t() / temperature

        if labels is not None:
            # Positive: same identity (not just diagonal)
            lbl_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            # Soft positive mask
            pos_mask = lbl_eq / lbl_eq.sum(dim=1, keepdim=True).clamp(min=1)
            # Multi-positive cross-entropy
            log_prob = logits - logits.logsumexp(dim=1, keepdim=True)
            loss_p = -(pos_mask * log_prob).sum(dim=1).mean()
            log_prob_t = logits.t() - logits.t().logsumexp(dim=1, keepdim=True)
            loss_p = loss_p - (pos_mask.t() * log_prob_t).sum(dim=1).mean()
            loss = loss + loss_p * 0.5
        else:
            targets = torch.arange(B, device=rgb_align.device)
            loss = loss + 0.5 * (F.cross_entropy(logits, targets) +
                                 F.cross_entropy(logits.t(), targets))

    return loss / P


def intra_modal_separation_loss(features, labels=None, temperature=0.3):
    """Intra-modal uniformity / separation loss (IMSep from ICLR 2025).

    Ensures features within the same modality are well-separated
    for different identities, preventing modality collapse.

    Args:
        features: [B, P, D] — alignment features from one modality
        labels:   [B] identity labels
    """
    B, P, D = features.shape
    if B <= 1:
        return features.new_tensor(0.0)

    # Global feature per sample
    fg = F.normalize(features.mean(dim=1), dim=-1)    # [B, D]
    sim = fg @ fg.t() / temperature                     # [B, B]

    if labels is not None:
        # Push apart different-identity pairs
        diff_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        if diff_mask.sum() < 1:
            return features.new_tensor(0.0)
        # Uniformity: log(mean(exp(sim))) over negative pairs
        neg_sim = sim * diff_mask + (-1e9) * (1 - diff_mask)
        return torch.logsumexp(neg_sim.flatten(), dim=0) - math.log(diff_mask.sum().clamp(min=1))
    else:
        mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
        return torch.logsumexp(sim[mask], dim=0) - math.log(max(mask.sum().item(), 1))


def cross_modal_alignment_loss(
        rgb_align, pc_align, labels=None,
        plc_weight=1.0, ims_weight=0.3, temperature=0.07):
    """Combined alignment loss: PLC + IMSep for both modalities.

    Args:
        rgb_align: [B, P, D]
        pc_align:  [B, P, D]
        labels: [B]
        plc_weight: weight for part-level contrastive
        ims_weight: weight for intra-modal separation
    Returns:
        total alignment loss
    """
    plc = part_level_cross_modal_contrastive_loss(
        rgb_align, pc_align, labels, temperature)
    ims_rgb = intra_modal_separation_loss(rgb_align, labels)
    ims_pc = intra_modal_separation_loss(pc_align, labels)
    return plc_weight * plc + ims_weight * 0.5 * (ims_rgb + ims_pc)
