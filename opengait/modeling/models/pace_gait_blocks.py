
"""PACE-Gait configurable blocks.

This file intentionally keeps only the pieces that are new for the cleaned
PACE-Gait entry point. The heavy experts are reused from the previous
GaitMoEOracle V2Max implementation to minimize risk when integrating into
OpenGait.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gait_moe_blocks_v2plus import QualityAwareFusionExpertBlock


class ConfigurableHeterogeneousFusionMoELayer(nn.Module):
    """Fusion MoE with true router-level switches for condition and quality.

    Compared with the previous ``HeterogeneousFusionMoELayer``, the router input
    dimension is constructed from the enabled signals only. Therefore, when
    ``use_quality_router=False`` or ``cond_dim=0``, the corresponding signals are
    not merely zero-filled - they are absent from the router MLP.
    """

    def __init__(self, dim, output_parts, num_input_tokens, expert_ranks,
                 top_k=2, cond_dim=64, dropout=0.0, shared_enabled=True,
                 shared_rank=32, modality_aware_router=False,
                 use_quality_router=True, use_quality_expert=True):
        super().__init__()
        self.num_experts = len(expert_ranks)
        self.top_k = int(top_k)
        self.output_parts = int(output_parts)
        self.cond_dim = int(cond_dim or 0)
        self.use_quality_router = bool(use_quality_router)
        self.use_quality_expert = bool(use_quality_expert)
        self.modality_aware_router = bool(modality_aware_router)

        self.experts = nn.ModuleList([
            QualityAwareFusionExpertBlock(dim, output_parts, num_input_tokens, rank=r)
            for r in expert_ranks
        ])
        if shared_enabled:
            self.shared_fusion = QualityAwareFusionExpertBlock(
                dim, output_parts, num_input_tokens, rank=shared_rank)
            self.shared_scale = nn.Parameter(torch.tensor(0.3))
        else:
            self.shared_fusion = None

        router_in = dim + self.cond_dim
        if self.use_quality_router:
            router_in += 2
        if self.modality_aware_router:
            self.mod_embed = nn.Embedding(2, max(1, dim // 8))
            router_in += max(1, dim // 8)

        self.router = nn.Sequential(
            nn.Linear(router_in, max(1, dim // 2)), nn.ReLU(True),
            nn.LayerNorm(max(1, dim // 2)),
            nn.Linear(max(1, dim // 2), self.num_experts),
        )
        self.residual_proj = nn.Linear(num_input_tokens * dim, output_parts * dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

    def _seq_reduce(self, x, B, N, seq_len, fallback_dim=0):
        if x is None or x.size(-1) == 0:
            return None
        if x.size(0) == B:
            return x
        if x.size(0) == N and seq_len > 1:
            return x.reshape(B, seq_len, -1).mean(dim=1)
        return x.new_zeros(B, fallback_dim or x.size(-1))

    def forward(self, tokens, cond_emb=None, quality_scores=None, seq_len=1):
        """Args:
        tokens: [N, 2P, C]
        cond_emb: [B, D] or [N, D], only used when cond_dim > 0
        quality_scores: [B, 2] or [N, 2], only used when quality switches are on
        """
        N, TK, Cd = tokens.shape
        P = self.output_parts
        half = TK // 2
        B = N // seq_len if seq_len > 1 else N

        if seq_len > 1:
            fp = tokens.reshape(B, seq_len, TK, Cd).mean(dim=(1, 2))
        else:
            fp = tokens.mean(dim=1)

        parts = [fp]
        if self.cond_dim > 0:
            ce = self._seq_reduce(cond_emb, B, N, seq_len, self.cond_dim)
            if ce is None:
                ce = fp.new_zeros(B, self.cond_dim)
            parts.append(ce.to(device=fp.device, dtype=fp.dtype))

        qs_r = None
        if self.use_quality_router:
            qs_r = self._seq_reduce(quality_scores, B, N, seq_len, 2)
            if qs_r is None:
                qs_r = fp.new_zeros(B, 2)
            parts.append(qs_r.to(device=fp.device, dtype=fp.dtype))

        if self.modality_aware_router:
            mod_ids = torch.zeros(B, TK, dtype=torch.long, device=fp.device)
            mod_ids[:, half:] = 1
            parts.append(self.mod_embed(mod_ids).mean(dim=1))

        logits = self.router(torch.cat(parts, dim=-1))
        top_k = min(self.top_k, self.num_experts)
        _, ti = logits.topk(top_k, dim=-1)
        mB = torch.zeros_like(logits).scatter_(1, ti, 1.0)
        wB = F.softmax(logits.masked_fill(mB == 0, float('-inf')), dim=-1)

        if seq_len > 1:
            mN = mB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            wN = wB.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            qs_N = quality_scores
            if qs_N is not None and qs_N.size(0) == B:
                qs_N = qs_N.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
        else:
            mN, wN, qs_N = mB, wB, quality_scores

        expert_quality = qs_N if self.use_quality_expert else None

        shared_out = None
        if self.shared_fusion is not None:
            q_shared = expert_quality
            if q_shared is not None and q_shared.size(0) != N and seq_len > 1:
                q_shared = q_shared.unsqueeze(1).expand(-1, seq_len, -1).reshape(N, -1)
            shared_out = self.shared_fusion(tokens, q_shared) * self.shared_scale

        out = torch.zeros(N, P, Cd, device=tokens.device, dtype=tokens.dtype)
        for i in range(self.num_experts):
            sel = mN[:, i] > 0
            if not sel.any():
                continue
            qs_sel = expert_quality[sel] if expert_quality is not None else None
            eo = self.experts[i](tokens[sel], qs_sel).to(dtype=out.dtype)
            w = wN[sel, i].to(dtype=out.dtype).unsqueeze(-1).unsqueeze(-1)
            out[sel] = out[sel] + w * eo

        alpha = torch.sigmoid(self.residual_gate).to(dtype=out.dtype)
        res = self.residual_proj(tokens.reshape(N, -1)).reshape(N, P, Cd).to(dtype=out.dtype)
        out = alpha * res + (1.0 - alpha) * out
        if shared_out is not None:
            out = out + shared_out.to(dtype=out.dtype)
        return out.permute(0, 2, 1).contiguous(), logits, wB


def collect_part_attention(moe_module):
    """Return expert part gates for visualization."""
    if moe_module is None or not hasattr(moe_module, 'experts'):
        return {}
    info = {}
    for i, e in enumerate(moe_module.experts):
        if hasattr(e, 'part_logit'):
            info[f'expert_{i}'] = {
                'type': type(e).__name__,
                'part_attention': torch.sigmoid(e.part_logit).detach().cpu().numpy(),
            }
    if getattr(moe_module, 'shared_expert', None) is not None:
        # SharedExpert has no explicit part gate; mark it for legend completeness.
        info['shared'] = {'type': type(moe_module.shared_expert).__name__}
    return info


def collect_fusion_token_preference(fusion_moe):
    """Collect per-fusion-expert RGB/PC token preference when available."""
    if fusion_moe is None or not hasattr(fusion_moe, 'experts'):
        return {}
    out = {}
    for i, e in enumerate(fusion_moe.experts):
        if hasattr(e, 'token_logit'):
            ta = torch.sigmoid(e.token_logit).detach().cpu()
            P = ta.numel() // 2
            out[f'expert_{i}'] = {
                'rgb': ta[:P].numpy(),
                'pc': ta[P:].numpy(),
                'rgb_mean': float(ta[:P].mean()),
                'pc_mean': float(ta[P:].mean()),
            }
    return out
