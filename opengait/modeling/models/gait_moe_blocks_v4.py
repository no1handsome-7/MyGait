"""GaitMoEOracle V3 — Token-level building blocks.

Key changes from V2Plus:
  - No HPP: spatial/point tokens preserved throughout
  - SpatialExpertBlock: channel + spatial attention (replaces PartExpertBlock)
  - SpatialMoELayer: TopK MoE for spatial tokens with seq-level routing
  - PointCloudTokenEncoder: per-point features without pooling
  - ModalityPositionalEncoding: 2D sincos (RGB) + xyz projection (PC) + modality tokens
  - LearnableTokenFusion: learnable queries, condition/quality-aware top-k, cross+self attention
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Re-export unchanged blocks from V2 / V2Plus ────────────────────────
from .gait_moe_blocks import (
    GradReverseFn, grad_reverse, l2n, _scalar,
    symmetric_infonce, masked_symmetric_infonce, ramp_value,
    TextEmbeddingAdapter,
    PositiveTextGate,
    TinyDenseRGBEncoder,
    StagedNuisanceRemoval,
    CycleAwareTemporalEncoder,
    RuleBasedVerifier,
    compute_condition_stats,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
)

from .gait_moe_blocks_v2plus import (
    ModalityQualityEstimator,
    EnhancedConditionEncoder,
    ModalityDropout,
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss,
    expert_utilization_stats,
)


# ================================================================
#  Spatial Expert Block (replaces PartExpertBlock)
# ================================================================

class SpatialExpertBlock(nn.Module):
    """Expert block for variable-length spatial token sequences.

    Applies channel attention (SE-style) and learnable spatial attention,
    followed by a low-rank bottleneck transform.  The spatial attention
    map is extractable for visualization.
    """

    def __init__(self, dim: int, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        # Channel attention (squeeze-excitation)
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(dim, rank), nn.GELU(), nn.Linear(rank, dim), nn.Sigmoid(),
        )
        # Spatial attention (1×1 conv → scalar per position)
        self.spatial_attn = nn.Sequential(
            nn.Conv1d(dim, rank, 1), nn.GELU(), nn.Conv1d(rank, 1, 1), nn.Sigmoid(),
        )
        # Low-rank transform
        self.down = nn.Conv1d(dim, rank, 1, bias=False)
        self.up = nn.Conv1d(rank, dim, 1, bias=False)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, return_spatial_attn: bool = False):
        """
        Args:
            x: [N, C, S]  (S = H'*W' for RGB or N_pts for PC)
            return_spatial_attn: if True, also return [N, S] attention map
        Returns:
            out: [N, C, S]
            (optional) sa: [N, S]
        """
        # Channel gating
        cg = self.se_fc(self.se_pool(x).squeeze(-1)).unsqueeze(-1)   # [N, C, 1]
        x = x * cg
        # Spatial gating
        sa = self.spatial_attn(x)          # [N, 1, S]
        x = x * sa
        # Low-rank bottleneck
        out = self.norm(self.up(self.act(self.drop(self.down(x)))))
        if return_spatial_attn:
            return out, sa.squeeze(1)      # [N, S]
        return out


# ================================================================
#  Spatial MoE Layer (seq-level routing, spatial tokens)
# ================================================================

class SpatialMoELayer(nn.Module):
    """TopK MoE with SpatialExpertBlock and sequence-level routing.

    Same routing logic as TopKMoELayerV2Plus but operates on spatial
    tokens [N, C, S] instead of part tokens [N, C, P].
    """

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 expert_rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            SpatialExpertBlock(dim, expert_rank, dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(True),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, num_experts),
        )
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    
    
    # Add this method inside class SpatialMoELayer in opengait/modeling/models/gait_moe_blocks_v4.py.
    # It must return dense all-expert attention for the same input samples:
    # input x: [B, C, S]
    # output:  [B, E, S]
    #
    # You need to adapt _get_one_expert_spatial_attn() to your actual expert implementation.
    # The important rule is: do NOT use routing/top-k here; run every expert on the same x.

    @torch.no_grad()
    def get_all_expert_spatial_attn(self, x, Hf=None, Wf=None):
        """Return all experts' spatial attention on the same samples.

        Args:
            x: [B, C, S], normally the first-frame RGB tokens before AppearanceMoE.
        Returns:
            Tensor [B, E, S]
        """
        attns = []
        for expert in self.experts:
            # Preferred: if your expert already has a method returning [B,S].
            if hasattr(expert, "get_spatial_attn"):
                a = expert.get_spatial_attn(x)
            # Common alternative: expert.forward(..., return_spatial_attn=True)
            elif hasattr(expert, "forward"):
                out = expert(x, return_spatial_attn=True)
                if isinstance(out, tuple):
                    a = out[-1]
                else:
                    raise RuntimeError(
                        "Expert does not return spatial attention. "
                        "Expose the attention map used inside the expert as [B,S]."
                    )
            else:
                raise RuntimeError("Cannot extract spatial attention from expert.")

            if a.dim() == 3 and a.size(1) == 1:
                a = a[:, 0]
            if a.dim() == 3 and Hf is not None and Wf is not None:
                a = a.reshape(a.size(0), -1)
            if a.dim() != 2:
                raise RuntimeError(f"Expected one expert attention [B,S], got {tuple(a.shape)}")
            attns.append(a.detach().float())

        return torch.stack(attns, dim=1)  # [B, E, S]


    def forward(self, x: torch.Tensor, seq_len: int = 1,
                return_spatial_attn: bool = False):
        """
        Args:
            x: [N, C, S]
            seq_len: T  (N = B*T)
            return_spatial_attn: collect per-expert spatial attention
        Returns:
            output: [N, C, S]
            logits: [B, E]
            weights: [B, E]
            (optional) spatial_attns: dict  expert_i → [N_sel, S]
        """
        N, C, S = x.shape
        B = N // seq_len if seq_len > 1 else N

        # Sequence-level routing
        if seq_len > 1:
            r_in = x.reshape(B, seq_len, C, S).mean(dim=(1, 3))   # [B, C]
        else:
            r_in = x.mean(dim=-1)                                   # [B, C]

        logits = self.router(r_in)                                   # [B, E]
        _, topk_idx = logits.topk(self.top_k, dim=-1)
        mask_B = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        weights_B = F.softmax(logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)

        if seq_len > 1:
            mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wgt_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mask_N, wgt_N = mask_B, weights_B

        expert_out = torch.zeros_like(x)
        spatial_attns = {} if return_spatial_attn else None

        for i in range(self.num_experts):
            sel = mask_N[:, i] > 0
            if not sel.any():
                continue
            if return_spatial_attn:
                eo, sa = self.experts[i](x[sel], return_spatial_attn=True)
                spatial_attns[f'expert_{i}'] = sa.detach().cpu()
            else:
                eo = self.experts[i](x[sel])
            eo = eo.to(dtype=expert_out.dtype)
            w = wgt_N[sel, i].to(dtype=eo.dtype).unsqueeze(-1).unsqueeze(-1)
            expert_out[sel] = expert_out[sel] + w * eo

        alpha = torch.sigmoid(self.residual_gate).to(dtype=x.dtype)
        output = alpha * x + (1.0 - alpha) * expert_out

        if return_spatial_attn:
            return output, logits, weights_B, spatial_attns
        return output, logits, weights_B


# ================================================================
#  Point Cloud Token Encoder (no HPP)
# ================================================================

class PointCloudTokenEncoder(nn.Module):
    """Shared-MLP point cloud encoder that preserves per-point features.

    Input:  [B, T, N, D]
        D can be 3 (xyz) or 4+ when PointCloudsTransform(scale_aware=True)
        appends extra geometry channels.
    Output: [B, T, C, N] per-point feature map
    Also returns xyz coordinates for positional encoding.
    """

    def __init__(self, channels: int, in_dim: int = 3):
        super().__init__()
        self.D = int(in_dim)
        c1, c2 = channels // 4, channels // 2
        self.conv1 = nn.Conv1d(self.D, c1, 1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, channels, 1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def _fit(self, pc: torch.Tensor) -> torch.Tensor:
        """Match runtime point dimension to configured pc_in_dim.

        This mirrors PointCloudSimpleEncoder behavior:
        - if transform outputs more dims than pc_in_dim, use the first pc_in_dim;
        - if transform outputs fewer dims, pad zeros.
        """
        d = pc.size(-1)
        if d == self.D:
            return pc
        if d > self.D:
            return pc[..., :self.D]
        return torch.cat([pc, pc.new_zeros(*pc.shape[:-1], self.D - d)], dim=-1)

    def forward(self, pc: torch.Tensor):
        """
        Args:
            pc: [B, T, N, D]
        Returns:
            feat: [B, T, C, N]
            xyz:  [B, T, N, 3]  (first 3 dims of original input, for positional encoding)
        """
        B, T, N, _ = pc.shape

        # Keep original xyz for positional encoding even if extra channels exist.
        xyz = pc[..., :3]

        # Make Conv1d input channel count consistent with pc_in_dim.
        pc = self._fit(pc.float())
        D = pc.size(-1)

        x = pc.reshape(B * T, N, D).permute(0, 2, 1)           # [BT, D, N]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))                            # [BT, C, N]
        return x.reshape(B, T, -1, N), xyz


# ================================================================
#  Modality & Positional Encoding
# ================================================================

class ModalityPositionalEncoding(nn.Module):
    """Adds modality embedding + positional encoding to RGB / PC tokens
    before they enter the fusion module.

    RGB tokens receive 2-D sinusoidal PE (based on feature-map grid).
    PC  tokens receive a learned projection of their xyz coordinates.
    Both receive a learned modality embedding.
    """

    def __init__(self, dim: int, pc_coord_dim: int = 3):
        super().__init__()
        self.dim = dim
        # Modality embeddings
        self.rgb_modal = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pc_modal = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        # PC positional encoding: project xyz → dim
        self.pc_pos_proj = nn.Sequential(
            nn.Linear(pc_coord_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim),
        )

    # ---- 2-D sinusoidal PE for RGB ----
    @staticmethod
    def _sincos_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
        """Returns [H*W, dim] positional encoding."""
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        half = dim // 4
        freq = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / half)
        )
        pos_h = torch.arange(h, device=device, dtype=torch.float32).unsqueeze(1) * freq
        pos_w = torch.arange(w, device=device, dtype=torch.float32).unsqueeze(1) * freq
        pe_h = torch.cat([pos_h.sin(), pos_h.cos()], dim=-1)           # [H, dim/2]
        pe_w = torch.cat([pos_w.sin(), pos_w.cos()], dim=-1)           # [W, dim/2]
        pe = torch.cat([
            pe_h.unsqueeze(1).expand(-1, w, -1),
            pe_w.unsqueeze(0).expand(h, -1, -1),
        ], dim=-1)                                                      # [H, W, dim]
        return pe.reshape(h * w, dim)

    def encode_rgb(self, rgb_tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            rgb_tokens: [N, C, H*W]  (C = dim)
        Returns:
            [N, H*W, C]  with PE + modality embedding, transposed for attention
        """
        N, C, S = rgb_tokens.shape
        pe = self._sincos_2d(h, w, C, rgb_tokens.device)               # [S, C]
        out = rgb_tokens.permute(0, 2, 1) + pe.unsqueeze(0) + self.rgb_modal
        return out                                                       # [N, S, C]

    def encode_pc(self, pc_tokens: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pc_tokens: [N, C, N_pts]
            xyz:       [N, N_pts, 3]
        Returns:
            [N, N_pts, C]
        """
        pe = self.pc_pos_proj(xyz.float())                              # [N, N_pts, C]
        out = pc_tokens.permute(0, 2, 1) + pe + self.pc_modal
        return out                                                       # [N, N_pts, C]


# ================================================================
#  Learnable Token Fusion
# ================================================================

class _CrossAttentionLayer(nn.Module):
    """Single cross-attention + self-attention sub-layer."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_kv = nn.LayerNorm(dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.cross_ffn_norm = nn.LayerNorm(dim)

        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        self.self_ffn_norm = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                return_cross_attn: bool = False):
        """
        Args:
            q:  [N, K, C]   (selected learnable queries)
            kv: [N, S, C]   (concatenated RGB + PC tokens)
        Returns:
            q:  [N, K, C]
            (optional) cross_attn_weights: [N, K, S]
        """
        # Cross-attention
        q_n = self.cross_norm_q(q)
        kv_n = self.cross_norm_kv(kv)
        ca_out, ca_w = self.cross_attn(q_n, kv_n, kv_n,
                                       need_weights=return_cross_attn,
                                       average_attn_weights=True)
        q = q + ca_out
        q = q + self.cross_ffn(self.cross_ffn_norm(q))
        # Self-attention among queries
        q_n = self.self_norm(q)
        sa_out, _ = self.self_attn(q_n, q_n, q_n)
        q = q + sa_out
        q = q + self.self_ffn(self.self_ffn_norm(q))
        return (q, ca_w) if return_cross_attn else (q, None)


class LearnableTokenFusion(nn.Module):
    """Fusion via learnable query tokens with condition/quality-aware top-k.

    Architecture:
        1. Maintain a pool of ``num_queries`` learnable tokens.
        2. Score each query using global context + condition + quality.
        3. Select top-K queries.
        4. Selected queries attend to concatenated RGB+PC tokens
           via multi-layer cross-attention + self-attention.
        5. Output ``K`` fused tokens  → [N, C, K].

    Replaces the MoE-based fusion from V2Plus with a more flexible
    attention-based mechanism inspired by Q-Former / Perceiver.
    """

    def __init__(self, dim: int, num_queries: int = 64, output_k: int = 16,
                 num_heads: int = 8, cond_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.output_k = output_k
        self.dim = dim

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        # Condition/quality-aware query scorer
        self.query_scorer = nn.Sequential(
            nn.Linear(dim + cond_dim + 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_queries),
        )

        # Cross+self attention layers
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection (keeps dim but adds nonlinearity)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, tokens: torch.Tensor, cond_emb: torch.Tensor,
                quality_scores: torch.Tensor, seq_len: int = 1,
                return_cross_attn: bool = False):
        """
        Args:
            tokens: [N, S, C]  concatenated RGB + PC with PE/modal encoding
            cond_emb: [B, cond_dim]
            quality_scores: [B, 2]
            seq_len: T
            return_cross_attn: if True, return last-layer attention weights
        Returns:
            output: [N, C, K]
            query_scores: [B, num_queries]
            selected_idx: [B, K]
            (optional) cross_attn_w: [N, K, S]
        """
        N, S, C = tokens.shape
        B = N // seq_len if seq_len > 1 else N

        # ---- Global context for query scoring ----
        if seq_len > 1:
            ctx = tokens.reshape(B, seq_len, S, C).mean(dim=(1, 2))    # [B, C]
        else:
            ctx = tokens.mean(dim=1)

        parts = [ctx]
        if cond_emb is not None and cond_emb.size(-1) > 0:
            ce = cond_emb
            if ce.size(0) == N and seq_len > 1:
                ce = ce.reshape(B, seq_len, -1).mean(1)
            parts.append(ce.to(device=ctx.device, dtype=ctx.dtype))
        qs = quality_scores
        if qs is not None and qs.size(-1) > 0:
            if qs.size(0) == N and seq_len > 1:
                qs = qs.reshape(B, seq_len, -1).mean(1)
            parts.append(qs.to(device=ctx.device, dtype=ctx.dtype))
        else:
            parts.append(ctx.new_zeros(B, 2))

        scorer_in = torch.cat(parts, dim=-1)
        q_scores = self.query_scorer(scorer_in)                         # [B, num_queries]

        # ---- Top-K selection ----
        _, topk_idx = q_scores.topk(self.output_k, dim=-1)             # [B, K]
        all_q = self.queries.unsqueeze(0).expand(B, -1, -1)            # [B, Q, C]
        selected = torch.gather(
            all_q, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))        # [B, K, C]

        # Expand to frame level
        if seq_len > 1:
            selected = (selected.unsqueeze(1)
                        .expand(-1, seq_len, -1, -1)
                        .reshape(N, self.output_k, C))

        # ---- Cross + self attention ----
        q = selected
        ca_w = None
        for i, layer in enumerate(self.layers):
            need_w = return_cross_attn and (i == len(self.layers) - 1)
            q, ca_w = layer(q, tokens, return_cross_attn=need_w)

        q = self.out_proj(q)                                            # [N, K, C]
        output = q.permute(0, 2, 1).contiguous()                       # [N, C, K]
        return output, q_scores, topk_idx, ca_w


# ================================================================
#  Losses for V3
# ================================================================

def query_balance_loss(query_scores: torch.Tensor) -> torch.Tensor:
    """Encourage balanced query selection across the batch.

    Maximises the entropy of the average selection probability over
    queries so that all queries are used roughly equally.

    Args:
        query_scores: [B, num_queries]  (raw logits from the scorer)
    Returns:
        scalar loss in [0, 1]  (lower is more balanced)
    """
    probs = F.softmax(query_scores, dim=-1)             # [B, Q]
    avg = probs.mean(dim=0)                              # [Q]
    entropy = -(avg.clamp(min=1e-8) * avg.clamp(min=1e-8).log()).sum()
    max_ent = math.log(query_scores.size(1))
    return 1.0 - entropy / max_ent


def query_diversity_loss(queries: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Penalise high cosine similarity between learnable query embeddings.

    Args:
        queries: [Q, C]
        threshold: only penalise pairs with sim > threshold
    """
    q = F.normalize(queries, dim=-1)
    sim = q @ q.T                                        # [Q, Q]
    mask = ~torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
    sim = sim[mask]
    excess = F.relu(sim - threshold)
    return excess.mean()
