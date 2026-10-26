"""GaitMoEOracleV2Titan blocks.

V2Titan keeps the stable V2Max/V2Sol backbone and changes the context-token
path from quality-score driven QACT to condition-driven CACT.  The module here
is intentionally drop-in friendly: it reuses V2Max heterogeneous experts and
adds only the pieces that are new for Titan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gait_moe_blocks_v2max import *  # reuse V2Max experts, PLABF, temporal and losses


class ConditionAwareContextTokens(QualityContextTokens):
    """CACT: condition-aware context tokens without scalar quality scores.

    It preserves the robust self/cross/joint token probing design of QACT but
    removes the entropy-derived quality-score branch from the public output.
    Downstream modules must consume `cond_emb`, `rgb_ctx`, `pc_ctx`, token
    activations and attention maps, rather than scalar RGB/PC quality.
    """

    def forward(self, rgb_parts, pc_parts):
        out = super().forward(rgb_parts, pc_parts)
        out.pop('quality_scores', None)

        rgb_attn = out.get('rgb_self_attn_w', None)
        pc_attn = out.get('pc_self_attn_w', None)
        if rgb_attn is not None:
            # [B, K, P] -> [B, K], high value means a token strongly selects parts.
            out['rgb_token_activation'] = rgb_attn.max(dim=-1).values.detach()
            out['rgb_token_entropy'] = (-(rgb_attn.clamp_min(1e-8) * rgb_attn.clamp_min(1e-8).log()).sum(dim=-1)).detach()
        else:
            out['rgb_token_activation'] = out['rgb_ctx'].norm(dim=-1).detach()
        if pc_attn is not None:
            out['pc_token_activation'] = pc_attn.max(dim=-1).values.detach()
            out['pc_token_entropy'] = (-(pc_attn.clamp_min(1e-8) * pc_attn.clamp_min(1e-8).log()).sum(dim=-1)).detach()
        else:
            out['pc_token_activation'] = out['pc_ctx'].norm(dim=-1).detach()

        # Aliases used by the visualization script.
        out['cact_cond_emb'] = out['cond_emb']
        out['cact_rgb_ctx'] = out['rgb_ctx']
        out['cact_pc_ctx'] = out['pc_ctx']
        out['cact_rgb_token_activation'] = out['rgb_token_activation']
        out['cact_pc_token_activation'] = out['pc_token_activation']
        return out


class ConditionGatedEmbeddingEnsemble(nn.Module):
    """Training-time and test-time branch ensemble controlled by CACT.

    Instead of using a fixed test-time concatenation such as [fused, 0.5*rgb,
    0.5*pc], V2Titan learns a soft gate from `cond_emb` and applies the same
    gated recognition path during training.  The weighted sum keeps the output
    dimension identical to the main embedding, so the original BNNeck can be
    reused for ID supervision.
    """

    def __init__(self, cond_dim=64, hidden_dim=128, dropout=0.1, init_fused_bias=1.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0)
        with torch.no_grad():
            self.gate[-1].bias[0] = float(init_fused_bias)

    def forward(self, cond_emb, fused_emb, rgb_aux_emb=None, pc_aux_emb=None):
        if rgb_aux_emb is None or pc_aux_emb is None:
            w = fused_emb.new_zeros(fused_emb.size(0), 3)
            w[:, 0] = 1.0
            return fused_emb, w
        logits = self.gate(cond_emb.to(dtype=fused_emb.dtype))
        weights = F.softmax(logits, dim=-1)
        w0 = weights[:, 0].view(-1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1)
        w2 = weights[:, 2].view(-1, 1, 1)
        emb = w0 * fused_emb + w1 * rgb_aux_emb + w2 * pc_aux_emb
        return emb, weights


def supervised_contrastive_loss(features, labels, temperature=0.1, eps=1e-12):
    """AMP-safe supervised contrastive loss.

    features: [B, C] or [B, ...]
    labels:   [B]
    """
    if features is None or labels is None:
        return features.new_tensor(0.0)

    if features.dim() > 2:
        features = features.flatten(1)

    device = features.device
    labels = labels.view(-1).to(device)
    batch_size = features.size(0)

    if batch_size <= 1:
        return features.sum() * 0.0

    # Very important: compute SupCon in fp32 under AMP.
    feats = F.normalize(features.float(), dim=1)
    logits = torch.matmul(feats, feats.t()) / float(temperature)

    eye = torch.eye(batch_size, dtype=torch.bool, device=device)

    # Safe because logits is fp32 now.
    logits = logits.masked_fill(eye, -1e9)

    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    pos_mask = pos_mask & (~eye)

    pos_count = pos_mask.sum(dim=1)
    valid_anchor = pos_count > 0

    if not valid_anchor.any():
        return features.sum() * 0.0

    log_prob = F.log_softmax(logits, dim=1)

    loss = -(log_prob * pos_mask.float()).sum(dim=1)
    loss = loss[valid_anchor] / pos_count[valid_anchor].float()
    loss = loss.mean()

    return loss


def context_variance_loss(rgb_ctx, pc_ctx, min_std=0.05):
    """Encourages CACT tokens not to collapse to a constant representation."""
    ctx = torch.cat([rgb_ctx, pc_ctx], dim=1)
    std = ctx.float().reshape(ctx.size(0), -1).std(dim=0)
    return F.relu(float(min_std) - std).mean()
