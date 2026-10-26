"""GaitMoEOracleV2Sol building blocks.

V2Sol replaces QACT's scalar quality readout with CACT
(Condition-Aware Context Tokens). The context tokens are not forced to predict
RGB/PC quality; they are used as condition/control signals for MoE routing,
fusion-token transmission, and test-time branch ensemble.
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse stable V2Max blocks.
from .gait_moe_blocks_v2max import (
    HeterogeneousMoELayer,
    LightTemporalAgg,
    BranchAuxHead,
    QualityWeightedModalityDropout,
    cross_modal_alignment_loss_v2,
    identity_nuisance_orthogonality_loss,
    heterogeneous_diversity_loss,
    default_appearance_expert_configs,
    default_structure_expert_configs,
    default_fusion_expert_ranks,
    ContextAwareModalityDropout,
    QualityContextTokens,
)
from .gait_moe_blocks import symmetric_infonce


def _fit_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad/crop the last dimension to dim without creating new parameters."""
    if dim <= 0:
        return x.new_zeros(*x.shape[:-1], 0)
    d = x.size(-1)
    if d == dim:
        return x
    if d > dim:
        return x[..., :dim]
    return torch.cat([x, x.new_zeros(*x.shape[:-1], dim - d)], dim=-1)


class ConditionAwareContextTokens(nn.Module):
    """CACT: Condition-Aware Context Tokens.

    Difference from QACT:
      - no scalar `quality_scores` output;
      - no entropy-quality bottleneck;
      - context tokens and `cond_emb` are the primary signals;
      - downstream router/expert/ensemble modules consume condition context.

    Stages:
      1. self-probe: RGB tokens attend RGB parts; PC tokens attend PC parts.
      2. cross-probe: RGB tokens attend PC parts; PC tokens attend RGB parts.
      3. joint refinement: all context tokens interact to form `cond_emb`.
    """

    def __init__(self, channels: int, num_tokens: int = 8,
                 num_heads: int = 4, cond_dim: int = 64,
                 dropout: float = 0.1,
                 cross_probe_enabled: bool = True,
                 joint_refine_enabled: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.num_tokens = int(num_tokens)
        self.cond_dim = int(cond_dim)
        self.cross_probe_enabled = bool(cross_probe_enabled)
        self.joint_refine_enabled = bool(joint_refine_enabled)

        self.rgb_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)
        self.pc_tokens = nn.Parameter(torch.randn(num_tokens, channels) * 0.02)

        self.rgb_self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.pc_self_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.rgb_self_norm_q = nn.LayerNorm(channels)
        self.rgb_self_norm_kv = nn.LayerNorm(channels)
        self.pc_self_norm_q = nn.LayerNorm(channels)
        self.pc_self_norm_kv = nn.LayerNorm(channels)

        if self.cross_probe_enabled:
            self.rgb_cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            self.pc_cross_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            self.rgb_cross_norm_q = nn.LayerNorm(channels)
            self.pc_cross_norm_q = nn.LayerNorm(channels)
            self.rgb_cross_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
            self.pc_cross_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())

        if self.joint_refine_enabled:
            self.refine_norm = nn.LayerNorm(channels)
            self.refine_attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
            self.refine_ffn_norm = nn.LayerNorm(channels)
            self.refine_ffn = nn.Sequential(
                nn.Linear(channels, channels * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(channels * 2, channels))

        self.cond_proj = nn.Sequential(
            nn.Linear(channels * num_tokens * 2, channels), nn.GELU(),
            nn.LayerNorm(channels), nn.Linear(channels, cond_dim))

    def forward(self, rgb_parts: torch.Tensor, pc_parts: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb_parts.dim() == 4:
            rgb_kv = rgb_parts.mean(dim=1).permute(0, 2, 1)  # [B,P,C]
            pc_kv = pc_parts.mean(dim=1).permute(0, 2, 1)
        else:
            rgb_kv = rgb_parts.permute(0, 2, 1)
            pc_kv = pc_parts.permute(0, 2, 1)

        B, P, C = rgb_kv.shape
        K = self.num_tokens
        rq = self.rgb_tokens.unsqueeze(0).expand(B, -1, -1)
        pq = self.pc_tokens.unsqueeze(0).expand(B, -1, -1)

        rq_self, r_self_w = self.rgb_self_attn(
            self.rgb_self_norm_q(rq), self.rgb_self_norm_kv(rgb_kv), rgb_kv)
        pq_self, p_self_w = self.pc_self_attn(
            self.pc_self_norm_q(pq), self.pc_self_norm_kv(pc_kv), pc_kv)
        rq = rq + rq_self
        pq = pq + pq_self

        r_cross_w, p_cross_w = None, None
        if self.cross_probe_enabled:
            rq_cross, r_cross_w = self.rgb_cross_attn(self.rgb_cross_norm_q(rq), pc_kv, pc_kv)
            pq_cross, p_cross_w = self.pc_cross_attn(self.pc_cross_norm_q(pq), rgb_kv, rgb_kv)
            rq = rq + self.rgb_cross_gate(torch.cat([rq, rq_cross], dim=-1)) * rq_cross
            pq = pq + self.pc_cross_gate(torch.cat([pq, pq_cross], dim=-1)) * pq_cross

        if self.joint_refine_enabled:
            all_ctx = torch.cat([rq, pq], dim=1)
            all_refined, refine_w = self.refine_attn(
                self.refine_norm(all_ctx), self.refine_norm(all_ctx), all_ctx)
            all_ctx = all_ctx + all_refined
            all_ctx = all_ctx + self.refine_ffn(self.refine_ffn_norm(all_ctx))
            rgb_ctx, pc_ctx = all_ctx[:, :K], all_ctx[:, K:]
        else:
            rgb_ctx, pc_ctx = rq, pq
            refine_w = None

        cond_emb = self.cond_proj(torch.cat([rgb_ctx.reshape(B, -1), pc_ctx.reshape(B, -1)], dim=-1))
        token_energy = torch.cat([
            rgb_ctx.detach().float().norm(dim=-1),
            pc_ctx.detach().float().norm(dim=-1)], dim=-1)

        return {
            'rgb_ctx': rgb_ctx,
            'pc_ctx': pc_ctx,
            'cond_emb': cond_emb,
            'token_energy': token_energy,
            'rgb_self_attn_w': r_self_w.detach(),
            'pc_self_attn_w': p_self_w.detach(),
            'rgb_cross_attn_w': None if r_cross_w is None else r_cross_w.detach(),
            'pc_cross_attn_w': None if p_cross_w is None else p_cross_w.detach(),
            'joint_refine_attn_w': None if refine_w is None else refine_w.detach(),
        }


class ConditionContextModalityDropout(nn.Module):
    """Condition-aware modality dropout driven by CACT cond_emb.

    This is intentionally mild: it regularizes modality reliance without using
    scalar quality. In evaluation it is identity.
    """
    def __init__(self, p: float = 0.1, cond_dim: int = 64):
        super().__init__()
        self.p = float(p)
        self.gate = nn.Sequential(nn.Linear(cond_dim, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, rgb_tokens: torch.Tensor, pc_tokens: torch.Tensor,
                cond_emb: Optional[torch.Tensor] = None, seq_len: int = 1):
        if (not self.training) or self.p <= 0:
            return rgb_tokens, pc_tokens
        N = rgb_tokens.size(0)
        if cond_emb is None or cond_emb.numel() == 0:
            pr = pc = rgb_tokens.new_full((N, 1, 1), self.p)
        else:
            B = cond_emb.size(0)
            logits = self.gate(cond_emb.float()).sigmoid().to(rgb_tokens.dtype)
            probs = (self.p * logits).clamp(0.0, 0.8)
            if B != N and seq_len > 1:
                probs = probs.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, 2)
            elif B != N:
                probs = probs[:1].expand(N, -1)
            pr, pc = probs[:, 0:1, None], probs[:, 1:2, None]
        mr = (torch.rand(N, 1, 1, device=rgb_tokens.device) > pr).to(rgb_tokens.dtype)
        mp = (torch.rand(N, 1, 1, device=pc_tokens.device) > pc).to(pc_tokens.dtype)
        return rgb_tokens * mr / (1 - pr).clamp_min(1e-6), pc_tokens * mp / (1 - pc).clamp_min(1e-6)


class ConditionAwareFusionExpertBlock(nn.Module):
    """Fusion expert whose token transmission is controlled by CACT context."""
    def __init__(self, dim: int, output_parts: int, num_input_tokens: int,
                 rank: int = 32, context_dim: int = 0):
        super().__init__()
        self.output_parts = int(output_parts)
        self.num_input_tokens = int(num_input_tokens)
        self.context_dim = int(context_dim)
        self.queries = nn.Parameter(torch.randn(output_parts, dim) * 0.02)
        self.token_logit = nn.Parameter(torch.zeros(num_input_tokens))
        if self.context_dim > 0:
            self.context_gate = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, rank), nn.GELU(),
                nn.Linear(rank, num_input_tokens))
        else:
            self.context_gate = None

        self.q_proj = nn.Linear(dim, rank, bias=False)
        self.k_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.attn_scale = rank ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, tokens: torch.Tensor, context: Optional[torch.Tensor] = None):
        N, TK, C = tokens.shape
        base_gate = torch.sigmoid(self.token_logit).unsqueeze(0).expand(N, -1)
        if self.context_gate is not None and context is not None and context.numel() > 0:
            context = _fit_last_dim(context.to(tokens.dtype), self.context_dim)
            dyn_gate = torch.sigmoid(self.context_gate(context))
            token_gate = base_gate * dyn_gate
        else:
            token_gate = base_gate
        kv = self.norm_kv(tokens) * token_gate.unsqueeze(-1)
        q = self.norm_q(self.queries.unsqueeze(0).expand(N, -1, -1))
        attn = (self.q_proj(q) @ self.k_proj(kv).transpose(-1, -2)) * self.attn_scale
        out = self.o_proj(attn.softmax(-1) @ self.v_proj(kv))
        return out + self.ffn(self.ffn_norm(out))

    def get_token_preference(self, context: Optional[torch.Tensor] = None):
        P = self.token_logit.numel() // 2
        ta = torch.sigmoid(self.token_logit.detach())
        return float(ta[:P].mean()), float(ta[P:].mean())


class ConditionAwareFusionMoELayer(nn.Module):
    """Fusion MoE controlled by CACT condition/context rather than quality scores."""
    def __init__(self, dim: int, output_parts: int, num_input_tokens: int,
                 expert_ranks, top_k: int = 2, cond_dim: int = 64,
                 context_dim: int = 0, dropout: float = 0.0,
                 shared_enabled: bool = True, shared_rank: int = 32,
                 modality_aware_router: bool = False):
        super().__init__()
        self.num_experts = len(expert_ranks)
        self.top_k = int(top_k)
        self.output_parts = int(output_parts)
        self.context_dim = int(context_dim)
        self.shared_enabled = bool(shared_enabled)
        self.modality_aware_router = bool(modality_aware_router)
        self.experts = nn.ModuleList([
            ConditionAwareFusionExpertBlock(dim, output_parts, num_input_tokens, rank=int(r), context_dim=self.context_dim)
            for r in expert_ranks
        ])
        if shared_enabled:
            self.shared_fusion = ConditionAwareFusionExpertBlock(
                dim, output_parts, num_input_tokens, rank=int(shared_rank), context_dim=self.context_dim)
            self.shared_scale = nn.Parameter(torch.tensor(0.3))
        else:
            self.shared_fusion = None

        router_in = dim + self.context_dim
        if modality_aware_router:
            self.mod_embed = nn.Embedding(2, max(dim // 8, 1))
            router_in += max(dim // 8, 1)
        self.router = nn.Sequential(
            nn.Linear(router_in, dim // 2), nn.ReLU(True), nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, self.num_experts))
        self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def _make_context(self, cact_info, B: int, N: int, seq_len: int, ref: torch.Tensor):
        if self.context_dim <= 0:
            return ref.new_zeros(B, 0), ref.new_zeros(N, 0)
        parts = []
        if isinstance(cact_info, dict):
            ce = cact_info.get('cond_emb', None)
            if ce is not None:
                if ce.size(0) == N and seq_len > 1:
                    ce = ce.reshape(B, seq_len, -1).mean(dim=1)
                elif ce.size(0) != B:
                    ce = ce[:1].expand(B, -1)
                parts.append(ce.to(device=ref.device, dtype=ref.dtype))
            for key in ('rgb_ctx', 'pc_ctx'):
                ctx = cact_info.get(key, None)
                if ctx is not None:
                    if ctx.dim() == 3:
                        m = ctx.mean(dim=1)
                    else:
                        m = ctx.reshape(ctx.size(0), -1)
                    if m.size(0) == N and seq_len > 1:
                        m = m.reshape(B, seq_len, -1).mean(dim=1)
                    elif m.size(0) != B:
                        m = m[:1].expand(B, -1)
                    parts.append(m.to(device=ref.device, dtype=ref.dtype))
        elif torch.is_tensor(cact_info):
            ce = cact_info
            if ce.size(0) == N and seq_len > 1:
                ce = ce.reshape(B, seq_len, -1).mean(dim=1)
            elif ce.size(0) != B:
                ce = ce[:1].expand(B, -1)
            parts.append(ce.to(device=ref.device, dtype=ref.dtype))
        if parts:
            ctx_B = _fit_last_dim(torch.cat(parts, dim=-1), self.context_dim)
        else:
            ctx_B = ref.new_zeros(B, self.context_dim)
        if seq_len > 1:
            ctx_N = ctx_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            ctx_N = ctx_B
        return ctx_B, ctx_N

    def forward(self, tokens: torch.Tensor, cact_info=None, seq_len: int = 1):
        N, TK, Cd = tokens.shape
        P = self.output_parts
        if seq_len > 1:
            B = N // seq_len
            fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
        else:
            B = N
            fp = tokens.mean(dim=1)
        ctx_B, ctx_N = self._make_context(cact_info, B, N, seq_len, tokens)
        parts = [fp, ctx_B]
        if self.modality_aware_router:
            half = TK // 2
            mod_ids = torch.zeros(B, TK, dtype=torch.long, device=tokens.device)
            mod_ids[:, half:] = 1
            parts.append(self.mod_embed(mod_ids).mean(dim=1).to(fp.dtype))
        logits = self.router(torch.cat(parts, dim=-1))
        _, top_idx = logits.topk(self.top_k, dim=-1)
        mask_B = torch.zeros_like(logits).scatter_(1, top_idx, 1.0)
        weights_B = F.softmax(logits.masked_fill(mask_B == 0, float('-inf')), dim=-1)
        if seq_len > 1:
            mask_N = mask_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            weights_N = weights_B.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mask_N, weights_N = mask_B, weights_B

        shared_out = None
        if self.shared_fusion is not None:
            shared_out = self.shared_fusion(tokens, ctx_N) * self.shared_scale

        out = torch.zeros(N, P, Cd, device=tokens.device, dtype=tokens.dtype)
        for i, expert in enumerate(self.experts):
            sel = mask_N[:, i] > 0
            if not sel.any():
                continue
            eo = expert(tokens[sel], ctx_N[sel]).to(out.dtype)
            w = weights_N[sel, i].to(out.dtype).view(-1, 1, 1)
            out[sel] = out[sel] + w * eo
        alpha = torch.sigmoid(self.residual_gate).to(out.dtype)
        res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, P, Cd).to(out.dtype)
        out = alpha * res + (1.0 - alpha) * out
        if shared_out is not None:
            out = out + shared_out.to(out.dtype)
        return out.permute(0, 2, 1).contiguous(), logits, weights_B


def cact_cross_modal_consistency_loss(rgb_ctx: torch.Tensor, pc_ctx: torch.Tensor, labels=None,
                                      temperature: float = 0.07):
    if rgb_ctx is None or pc_ctx is None:
        ref = rgb_ctx if rgb_ctx is not None else pc_ctx
        return ref.new_tensor(0.0)
    r = rgb_ctx.mean(dim=1)
    p = pc_ctx.mean(dim=1)
    if r.size(0) <= 1:
        return r.new_tensor(0.0)
    return symmetric_infonce(r, p, temperature=temperature)


def cact_token_diversity_loss(rgb_ctx: torch.Tensor, pc_ctx: torch.Tensor,
                              margin: float = 0.15):
    if rgb_ctx is None or pc_ctx is None:
        ref = rgb_ctx if rgb_ctx is not None else pc_ctx
        return ref.new_tensor(0.0)
    def _one(x):
        # x [B,K,C], penalize high off-diagonal cosine similarity among tokens.
        if x.size(1) <= 1:
            return x.new_tensor(0.0)
        z = F.normalize(x.float(), dim=-1)
        sim = z @ z.transpose(1, 2)
        eye = torch.eye(sim.size(-1), device=sim.device, dtype=torch.bool).unsqueeze(0)
        off = sim.masked_select(~eye)
        return F.relu(off.abs() - margin).mean()
    return 0.5 * (_one(rgb_ctx) + _one(pc_ctx))


def cact_token_expert_correlation(token_energy: torch.Tensor, router_logits: torch.Tensor):
    """Diagnostic-only differentiable-free correlation matrix [2K,E]."""
    if token_energy is None or router_logits is None:
        return None
    with torch.no_grad():
        x = token_energy.float()
        y = router_logits.softmax(dim=-1).float()
        n = min(x.size(0), y.size(0))
        x, y = x[:n], y[:n]
        x = (x - x.mean(0, keepdim=True)) / x.std(0, keepdim=True).clamp_min(1e-6)
        y = (y - y.mean(0, keepdim=True)) / y.std(0, keepdim=True).clamp_min(1e-6)
        return (x.t() @ y) / max(n - 1, 1)
