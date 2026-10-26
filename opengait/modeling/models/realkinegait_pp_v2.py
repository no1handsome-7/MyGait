"""RealKineGait++ v2 — Further optimisations targeting CVPR 2027.

Changes from v1 (realkinegait_pp.py):
  A. Pose-Guided Part Query Initialisation — pose joints directly seed part
     attention, creating explicit anatomy↔token correspondence.
  B. Sinusoidal Temporal Positional Encoding in KineTF — frames are aware of
     their temporal position for proper phase modelling.
  C. Phase-Canonical Inference — at test time, identity is compared per phase
     bin then aggregated, rather than naive averaging.
  D. Part-Joint Alignment Loss support — output anchors for aligning learned
     parts with anatomical pose joints.
  E. Intra-Cycle vs Inter-Cycle decomposition — separate within-cycle kinematic
     identity from across-cycle gait style.
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateBNNecks, SeparateFCs


# ======================== Backbones (unchanged) ========================

class TinyFrameBackbone(nn.Module):
    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(out_dim), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        y = self.net(x)
        tokens = y.flatten(2).transpose(1, 2).contiguous()
        cls = tokens.mean(1, keepdim=True)
        return [torch.cat([cls, tokens], 1) for _ in range(4)]


class FrozenDINOBackbone(nn.Module):
    """Optional HuggingFace DINO/DINOv2 backbone with a Tiny CNN fallback.

    The original v2 draft decorated ``forward`` with ``@torch.no_grad()``, which
    silently disabled gradients even when ``freeze_lvm=False``.  This OpenGait
    version makes freezing explicit and keeps the fallback backbone importable on
    machines without ``transformers`` or local DINO weights.
    """

    def __init__(self, name_or_path, out_dim, selected_layers, freeze=True):
        super().__init__()
        self.use_hf = False
        self.freeze = bool(freeze)
        self.out_dim = out_dim
        self.selected_layers = list(selected_layers)
        if name_or_path:
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
                self.use_hf = True
                self.out_dim = int(getattr(self.model.config, 'hidden_size', out_dim))
            except Exception as exc:
                warnings.warn(f"LVM load failed: {exc}. Using TinyFrameBackbone.")
                self.model = TinyFrameBackbone(out_dim)
        else:
            self.model = TinyFrameBackbone(out_dim)
        if self.freeze:
            self.model.eval()
            self.model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def _select(self, hiddens):
        hiddens = list(hiddens)
        mx = len(hiddens) - 1
        if mx < 0:
            raise RuntimeError('FrozenDINOBackbone received no hidden states.')
        return [hiddens[max(0, min(mx, i if i >= 0 else len(hiddens) + i))]
                for i in self.selected_layers]

    def _forward_impl(self, x):
        if not self.use_hf:
            return self._select(self.model(x))
        try:
            outs = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        except TypeError:
            outs = self.model(x, output_hidden_states=True)
        return self._select(list(outs.hidden_states[1:]))

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self._forward_impl(x)
        return self._forward_impl(x)

class LayerWiseGaitAdapter(nn.Module):
    def __init__(self, lvm_dim, adapter_dim, out_hw, num_layers):
        super().__init__()
        self.out_hw = out_hw
        self.projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(lvm_dim), nn.Linear(lvm_dim, adapter_dim), nn.GELU())
            for _ in range(num_layers)])
        self.fuse = nn.Sequential(
            nn.Conv2d(adapter_dim * num_layers, adapter_dim, 1, bias=False),
            nn.BatchNorm2d(adapter_dim), nn.GELU(),
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(adapter_dim), nn.GELU())

    @staticmethod
    def tokens_to_map(tokens, out_hw):
        if tokens.size(1) > 1:
            tokens = tokens[:, 1:]
        b, l, c = tokens.shape
        h = int(round(math.sqrt(l * 2)))
        w = max(1, l // max(1, h))
        if h * w != l:
            h = int(math.sqrt(l))
            while h > 1 and l % h != 0:
                h -= 1
            w = l // h
        return F.interpolate(tokens.transpose(1, 2).reshape(b, c, h, w),
                             out_hw, mode='bilinear', align_corners=False)

    def forward(self, hidden_states, batch, frames):
        maps = [self.tokens_to_map(proj(h), self.out_hw)
                for h, proj in zip(hidden_states, self.projs)]
        x = self.fuse(torch.cat(maps, 1))
        return rearrange(x, '(n s) c h w -> n c s h w', n=batch, s=frames).contiguous()


# =======================================================================
# [UPGRADE A] Pose-Guided Part Query GauPE
# =======================================================================

class PoseGuidedPartQueryGauPE(nn.Module):
    """Part queries seeded by 2D pose joint locations.

    When pose is available:
        1. Project pose joints to part-space → generate per-frame spatial
           priors for each part query.
        2. Blend learned queries with pose-guided queries via a gating
           mechanism → adapts when pose is unreliable.

    When pose is absent: falls back to purely learned queries (identical
    to v0.1 behaviour).
    """

    def __init__(self, dim: int, parts_num: int, num_joints: int = 17,
                 temperature: float = 0.07, eps: float = 1e-6):
        super().__init__()
        self.parts_num = parts_num
        self.num_joints = num_joints
        self.temperature = temperature
        self.eps = eps

        # Purely learned queries (fallback)
        self.queries = nn.Parameter(torch.randn(parts_num, dim) * 0.02)

        # Joint-to-Part mapping: learns which joints correspond to which parts
        # [num_joints, 3] → [num_joints, dim]
        self.joint_embed = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, dim))
        # Soft assignment: [num_joints] → [parts_num] (which joints feed which part)
        self.joint_to_part = nn.Linear(num_joints, parts_num, bias=False)
        # Gating: blend learned vs pose-guided
        self.pose_gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Linear(dim, 1), nn.Sigmoid())

        self.part_proj = nn.Linear(dim, dim)
        self.gaupe_proj = nn.Sequential(
            nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))

    def _pose_to_part_query(self, pose: torch.Tensor) -> torch.Tensor:
        """Generate dynamic part queries from pose joints.

        Args:
            pose: [N, T, J, 3] with normalized x/y/confidence.

        Returns:
            pose_query: [N*T, P, C], confidence-weighted anatomical query tokens.

        This makes ``joint_embed`` an active branch instead of a dead module: joint
        coordinates are embedded, softly assigned to learned parts, then used as
        frame-specific part-query residuals.
        """
        n, t, j, _ = pose.shape
        pose_in = pose
        if j < self.num_joints:
            pad = torch.zeros(n, t, self.num_joints - j, 3,
                              device=pose.device, dtype=pose.dtype)
            pose_in = torch.cat([pose_in, pad], dim=2)
        pose_in = pose_in[:, :, :self.num_joints]
        conf = pose_in[..., 2:3].clamp(0, 1)
        joint_tokens = self.joint_embed(pose_in) * conf
        w_jp = torch.softmax(self.joint_to_part.weight, dim=-1)  # [P, J]
        pose_query = torch.einsum('pj,ntjd->ntpd', w_jp, joint_tokens)
        norm = torch.einsum('pj,ntj->ntp', w_jp, conf.squeeze(-1)).clamp_min(self.eps)
        pose_query = pose_query / norm.unsqueeze(-1)
        return pose_query.reshape(n * t, self.parts_num, -1)

    def _pose_to_spatial_prior(self, pose, h, w, device, dtype):
        """Convert 2D pose joints to spatial attention priors.

        Args:
            pose: [N, T, J, 3] (x, y, confidence)

        Returns:
            spatial_prior: [N*T, P, H*W] soft attention maps from joint locations
        """
        n, t, j, _ = pose.shape
        if j < self.num_joints:
            pad = torch.zeros(n, t, self.num_joints - j, 3,
                              device=pose.device, dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=2)
        pose = pose[:, :, :self.num_joints]
        j = self.num_joints
        # Normalise coordinates to [-1, 1]
        xy = pose[:, :, :, :2].clone()  # [N, T, J, 2]
        conf = pose[:, :, :, 2:3].clamp(0, 1)  # [N, T, J, 1]

        # Create spatial grid
        yy = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xx = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], -1).reshape(1, 1, 1, h * w, 2)

        # Joint locations
        xy_flat = xy.reshape(n * t, j, 1, 2)  # [N*T, J, 1, 2]
        # Gaussian attention around each joint
        dist = ((grid.reshape(1, 1, h * w, 2) - xy_flat) ** 2).sum(-1)  # [N*T, J, H*W]
        sigma = 0.15  # Controls spread
        joint_attn = torch.exp(-dist / (2 * sigma ** 2))  # [N*T, J, H*W]
        # Weight by confidence
        conf_flat = conf.reshape(n * t, j, 1)
        joint_attn = joint_attn * conf_flat

        # Map joints → parts via learnable assignment
        # joint_to_part weight: [P, J]
        w_jp = torch.softmax(self.joint_to_part.weight, dim=-1)  # [P, J]
        part_attn = torch.einsum('pj,bjl->bpl', w_jp, joint_attn)  # [N*T, P, H*W]

        return part_attn

    def forward(self, feat: torch.Tensor,
                pose: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        n, c, t, h, w = feat.shape
        x = rearrange(feat, 'n c t h w -> (n t) (h w) c')

        # --- Attention computation ---
        q_learned = F.normalize(self.queries, dim=-1)
        k = F.normalize(x, dim=-1)
        attn_learned = torch.einsum('pc,blc->bpl', q_learned, k) / max(
            self.temperature, self.eps)
        attn_learned = torch.softmax(attn_learned, dim=-1)

        # Pose-guided blending
        if pose is not None and pose.dim() == 4 and pose.size(2) >= 1:
            pose_prior = self._pose_to_spatial_prior(
                pose, h, w, feat.device, feat.dtype)  # [N*T, P, H*W]
            # Normalise pose prior to probability
            pose_prior = pose_prior / pose_prior.sum(-1, keepdim=True).clamp_min(1e-8)

            # Dynamic pose queries from embedded joints.  They complement the
            # Gaussian spatial prior: the prior says *where* to look, while the
            # embedded query says *which anatomical part* is being represented.
            q_pose = self._pose_to_part_query(pose).to(dtype=x.dtype)
            q_dynamic = F.normalize(self.queries.unsqueeze(0) + q_pose, dim=-1)
            attn_pose_query = torch.einsum('bpc,blc->bpl', q_dynamic, k) / max(
                self.temperature, self.eps)
            attn_pose_query = torch.softmax(attn_pose_query, dim=-1)

            # Generate pose-guided query features
            pose_feat = torch.einsum('bpl,blc->bpc', pose_prior, x)  # [N*T, P, C]
            query_feat = torch.einsum('bpl,blc->bpc', attn_pose_query, x)
            learned_feat = torch.einsum('bpl,blc->bpc', attn_learned, x)
            pose_feat = 0.5 * (pose_feat + query_feat)

            # Gate: how much to trust pose vs learned
            gate_input = torch.cat([learned_feat, pose_feat], dim=-1)
            gate = self.pose_gate(gate_input)  # [N*T, P, 1]

            # Blend attention maps.  The pose side itself blends the embedded
            # pose-query attention and the Gaussian joint heatmap prior.
            attn_pose = 0.5 * pose_prior + 0.5 * attn_pose_query
            attn = gate * attn_pose + (1 - gate) * attn_learned
            attn = attn / attn.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            attn = attn_learned

        part_feat = torch.einsum('bpl,blc->bpc', attn, x)
        attn_map = attn.reshape(n, t, self.parts_num, h, w)

        # GauPE geometry
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).reshape(1, 1, 1, h * w, 2)
        prob = attn.reshape(n, t, self.parts_num, h * w, 1)
        mu = (prob * grid).sum(3)
        delta = grid - mu.unsqueeze(3)
        var = (prob * delta ** 2).sum(3)
        cov = (prob[..., 0] * delta[..., 0] * delta[..., 1]).sum(3, keepdim=True)
        theta = 0.5 * torch.atan2(
            2 * cov, (var[..., 0:1] - var[..., 1:2]).clamp_min(self.eps))
        geom = torch.cat([mu, var, cov, theta], -1)

        vel = torch.zeros_like(mu)
        vel[:, 1:] = mu[:, 1:] - mu[:, :-1]
        acc = torch.zeros_like(mu)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        geom12 = torch.cat([geom, vel, acc], -1)
        pad = torch.zeros(*geom12.shape[:-1], 2, device=feat.device, dtype=feat.dtype)
        geom12 = torch.cat([geom12, pad], -1)

        part_feat = self.part_proj(part_feat).reshape(n, t, self.parts_num, c)
        part_feat = part_feat + self.gaupe_proj(geom12)

        return {
            'part_feat': part_feat, 'attn_map': attn_map,
            'mu': mu, 'vel': vel, 'acc': acc, 'geom': geom12,
        }


# =======================================================================
# [UPGRADE B] Temporal Positional Encoding in KineTF
# =======================================================================

class SinusoidalTemporalPE(nn.Module):
    """Sinusoidal positional encoding for temporal dimension."""

    def __init__(self, dim: int, max_len: int = 300):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:dim // 2]) if dim % 2 else torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, t: int) -> torch.Tensor:
        return self.pe[:, :t]


class GraphTemporalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, dim))

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        return x + self.mlp(self.norm2(x))


class CrossPartKinematicTransformerV2(nn.Module):
    """KineTF with sinusoidal temporal PE."""

    def __init__(self, dim, parts_num, depth=2, heads=4,
                 graph_bias_strength=0.25, max_frames=300):
        super().__init__()
        self.parts_num = parts_num
        self.graph_bias_strength = graph_bias_strength
        self.blocks = nn.ModuleList([GraphTemporalBlock(dim, heads) for _ in range(depth)])
        self.part_type = nn.Parameter(torch.randn(1, parts_num, dim) * 0.02)
        self.temporal_pe = SinusoidalTemporalPE(dim, max_frames)

    def _mask(self, frames, device, dtype):
        p = self.parts_num
        edges = {(i, i) for i in range(p)}

        def add(a, b):
            if a < p and b < p:
                edges.add((a, b)); edges.add((b, a))

        add(1, 2); add(3, 6); add(4, 5); add(5, 6)
        add(5, 7); add(6, 8); add(2, 5); add(2, 6)
        base = torch.full((p, p), -float(self.graph_bias_strength), device=device, dtype=dtype)
        for a, b in edges:
            base[a, b] = 0.0
        mask = torch.zeros(frames * p, frames * p, device=device, dtype=dtype)
        for tt in range(frames):
            mask[tt * p:(tt + 1) * p, tt * p:(tt + 1) * p] = base
        return mask

    def forward(self, part_feat):
        n, t, p, c = part_feat.shape
        # Add part type embedding
        x = part_feat + self.part_type[:, :p].unsqueeze(1)
        # Add temporal positional encoding (broadcast across parts)
        tpe = self.temporal_pe(t)  # [1, T, C]
        x = x + tpe.unsqueeze(2)  # [N, T, P, C] + [1, T, 1, C]
        x = rearrange(x, 'n t p c -> n (t p) c')
        mask = self._mask(t, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        return rearrange(x, 'n (t p) c -> n t p c', t=t, p=p).contiguous()


# ======================== Phase-Canonical Module (unchanged from v1) ========================

class PhaseCanonicalModule(nn.Module):
    def __init__(self, parts_num, dim, num_bins=16,
                 foot_indices=(7, 8), phase_temperature=1.0):
        super().__init__()
        self.parts_num = parts_num
        self.dim = dim
        self.num_bins = num_bins
        self.foot_indices = foot_indices
        self.phase_net = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / num_bins), num_bins)
        self.register_buffer('bin_sin', angles.sin())
        self.register_buffer('bin_cos', angles.cos())
        self.phase_temperature = float(phase_temperature)
        self.bin_logit_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.cycle_proj = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def _foot_signal(self, mu):
        n, t, p, _ = mu.shape
        li = min(self.foot_indices[0], p - 1)
        ri = min(self.foot_indices[1], p - 1)
        lx, rx = mu[:, :, li, 0], mu[:, :, ri, 0]
        lv = torch.zeros_like(lx); rv = torch.zeros_like(rx)
        lv[:, 1:] = lx[:, 1:] - lx[:, :-1]
        rv[:, 1:] = rx[:, 1:] - rx[:, :-1]
        la = torch.zeros_like(lx); ra = torch.zeros_like(rx)
        la[:, 1:] = lv[:, 1:] - lv[:, :-1]
        ra[:, 1:] = rv[:, 1:] - rv[:, :-1]
        return torch.stack([lx, rx, lv, rv, la, ra], dim=-1)

    def forward(self, mu, kine_seq):
        n, t, p, c = kine_seq.shape
        foot_feat = self._foot_signal(mu)
        phase_sc = self.phase_net(foot_feat)
        phase_sc = phase_sc / phase_sc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        bin_vecs = torch.stack([self.bin_sin, self.bin_cos], dim=-1)
        sim = torch.einsum('ntd,kd->ntk', phase_sc, bin_vecs)
        temperature = max(self.phase_temperature, 1e-6)
        phase_weight = torch.softmax(sim * self.bin_logit_scale.clamp_min(0.1) / temperature, dim=-1)
        cycle_tokens = torch.einsum('ntk,ntpc->nkpc', phase_weight, kine_seq)
        weight_sum = phase_weight.sum(1).clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
        cycle_tokens = cycle_tokens / weight_sum
        cycle_tokens = cycle_tokens + self.cycle_proj(cycle_tokens)
        return {
            'cycle_tokens': cycle_tokens, 'phase_code': phase_sc,
            'phase_weight': phase_weight, 'foot_signal': foot_feat,
        }


# =======================================================================
# [UPGRADE E] Intra-Cycle / Inter-Cycle Decomposition
# =======================================================================

class CycleDecomposer(nn.Module):
    """Learnably decompose cycle tokens into identity and style components.

    The previous draft defined ``decompose_gate`` but never used it.  Here, the
    gate explicitly controls how much global cycle style is removed from each
    phase-part token before producing intra-cycle identity features.
    """

    def __init__(self, dim: int, num_bins: int):
        super().__init__()
        self.intra_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.inter_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.style_token_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.decompose_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid())

    def forward(self, cycle_tokens: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cycle_tokens: [N, K, P, C]
        Returns:
            intra: [N, K, P, C]  — style-suppressed phase identity features
            inter: [N, C]        — global cycle style/speed embedding
            gate:  [N, K, P, C]  — learned identity-vs-style gate
        """
        global_feat = cycle_tokens.mean((1, 2))
        inter = self.inter_proj(global_feat)
        style_token = self.style_token_proj(global_feat).view(cycle_tokens.size(0), 1, 1, -1)
        style_token = style_token.expand_as(cycle_tokens)

        gate = self.decompose_gate(torch.cat([cycle_tokens, style_token], dim=-1))
        identity_input = cycle_tokens - gate * style_token
        intra = self.intra_proj(identity_input)
        return {'intra': intra, 'inter': inter, 'gate': gate}


# ======================== Silhouette Module (from v1) ========================

class SilhouetteAwareModule(nn.Module):
    def __init__(self, adapter_dim, feature_hw):
        super().__init__()
        self.feature_hw = feature_hw
        self.bg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(adapter_dim, adapter_dim // 2), nn.GELU(),
            nn.Linear(adapter_dim // 2, adapter_dim))

    def _resize_sil(self, sil, n, s):
        if sil.dim() == 5:
            sil = sil[:, :, 0]
        sil_flat = rearrange(sil, 'n s h w -> (n s) 1 h w').float()
        sil_flat = F.interpolate(sil_flat, self.feature_hw, mode='bilinear', align_corners=False)
        return rearrange(sil_flat, '(n s) 1 h w -> n 1 s h w', n=n, s=s).clamp(0, 1)

    def forward(self, feat5d, sil, n, s):
        out = {}
        if sil is None:
            out['fg_feat5d'] = feat5d
            out['bg_feat5d'] = torch.zeros_like(feat5d)
            out['sil_mask'] = torch.ones(n, 1, s, *self.feature_hw,
                                         device=feat5d.device, dtype=feat5d.dtype)
            out['bg_embed'] = torch.zeros(n, feat5d.size(1),
                                          device=feat5d.device, dtype=feat5d.dtype)
            return out
        sil_mask = self._resize_sil(sil, n, s)
        out['fg_feat5d'] = feat5d * sil_mask
        out['bg_feat5d'] = feat5d * (1.0 - sil_mask)
        out['sil_mask'] = sil_mask
        out['bg_embed'] = self.bg_head(out['bg_feat5d'].mean(2))
        return out


# ======================== Uncertainty Teachers (from v1, compact) ========================

class UncertaintyHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class UncertaintySMPLTeacher(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
                                  nn.GELU(), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)

    def forward(self, x):
        if x.dim() > 2: x = x.mean(1)
        z = self.proj(x)
        return z, self.unc(z)


class UncertaintyPCTeacher(nn.Module):
    def __init__(self, out_dim, in_dim=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(),
                                 nn.Linear(64, 128), nn.GELU(), nn.Linear(128, out_dim))
        self.proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)
        self.density_unc = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, pc):
        if pc.dim() == 5: pc = pc.flatten(2, 3)
        if pc.size(-1) < 4:
            pc = torch.cat([pc, torch.zeros(*pc.shape[:-1], 4 - pc.size(-1),
                                            device=pc.device, dtype=pc.dtype)], -1)
        pc = pc[..., :4]
        x = self.mlp(pc).max(dim=2)[0]
        z = self.proj(x.mean(1))
        n_pts = (pc[..., :3].abs().sum(-1) > 1e-6).float()
        pt_count = n_pts.sum(-1)
        ds = torch.stack([pt_count.mean(1), pt_count.std(1).clamp_min(1e-6), pt_count.min(1)[0]], -1)
        return z, self.unc(z) + self.density_unc(ds).squeeze(-1)


class UncertaintyEventTeacher(nn.Module):
    def __init__(self, adapter_dim, in_channels=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, adapter_dim, 3, padding=1, bias=False), nn.BatchNorm3d(adapter_dim), nn.GELU())
        self.unc = UncertaintyHead(adapter_dim)
        self.rate_unc = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, ev):
        z = self.encoder(ev).mean((2, 3, 4))
        rate = ev.abs().mean((1, 3, 4))
        rs = torch.stack([rate.mean(1), rate.std(1).clamp_min(1e-6), rate.min(1)[0]], -1)
        return z, self.unc(z) + self.rate_unc(rs).squeeze(-1)


class PoseKinematicTeacher(nn.Module):
    def __init__(self, out_dim, num_joints=17):
        super().__init__()
        self.joint_embed = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(128 * num_joints, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.GELU())
        self.proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)
        self.num_joints = num_joints

    def forward(self, pose):
        if pose.dim() == 3: pose = pose.reshape(pose.size(0), pose.size(1), -1, 3)
        n, t, j, _ = pose.shape
        if j < self.num_joints:
            pose = torch.cat([pose, torch.zeros(n, t, self.num_joints - j, 3,
                                                device=pose.device, dtype=pose.dtype)], 2)
        pose = pose[:, :, :self.num_joints]
        x = rearrange(self.joint_embed(pose), 'n t j c -> n (j c) t')
        z = self.proj(self.temporal(x).mean(-1))
        conf = pose[:, :, :, 2].clamp(0, 1).mean((1, 2))
        return z, self.unc(z) - conf


class SequenceFeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
                                 nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        if x.dim() > 2: x = x.mean(1)
        return self.net(x)


# =======================================================================
# [UPGRADE C] Phase-Canonical Inference Head
# =======================================================================

class PhaseCanonicalInferenceHead(nn.Module):
    """At test time, produce per-phase-bin identity descriptors.

    Instead of simply averaging cycle_tokens, this head produces:
      1. A flat cycle descriptor: [N, K*P*C_proj] for direct distance computation
      2. Per-bin embeddings for bin-wise matching then aggregation
    """

    def __init__(self, dim: int, num_bins: int, parts_num: int, out_dim: int):
        super().__init__()
        self.num_bins = num_bins
        self.parts_num = parts_num
        # Per-bin projection
        self.bin_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, out_dim))
        # Cross-bin attention for final aggregation
        self.bin_attn = nn.MultiheadAttention(out_dim, 4, batch_first=True)
        self.bin_norm = nn.LayerNorm(out_dim)

    def forward(self, cycle_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            cycle_tokens: [N, K, P, C]
        Returns:
            cycle_id: [N, K, out_dim]  — per-bin identity descriptors
            cycle_agg: [N, out_dim]     — aggregated via cross-bin attention
        """
        n, k, p, c = cycle_tokens.shape
        # Pool over parts → [N, K, C]
        x = cycle_tokens.mean(2)
        x = self.bin_proj(x)  # [N, K, out_dim]

        # Cross-bin self-attention for context-aware aggregation
        h = self.bin_norm(x)
        y, _ = self.bin_attn(h, h, h, need_weights=False)
        x = x + y
        agg = x.mean(1)  # [N, out_dim]

        return {'cycle_id': x, 'cycle_agg': agg}


# =======================================================================
# MAIN MODEL: RealKineGaitPP_v2
# =======================================================================

class RealKineGaitPP_v2(BaseModel):
    """RealKineGait++ v2 with all upgrades for CVPR 2027."""

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        self.input_indices = model_cfg.get('input_indices', {'rgb': 0})
        self.rgb_index = self.input_indices.get('rgb', 0)
        self.sil_index = self.input_indices.get('sil', None)
        self.pose_index = self.input_indices.get('pose', None)
        self.motion_text_index = self.input_indices.get('motion_text', None)
        self.nuisance_text_index = self.input_indices.get('nuisance_text', None)
        self.smpl_index = self.input_indices.get('smpl', None)
        self.pc_index = self.input_indices.get('pointcloud', None)
        # Event data is intentionally disabled for this version.  The codebase
        # keeps the EventTeacher class for future experiments, but the forward
        # path, quality gate, and distillation stack do not consume event tensors.
        self.event_index = None

        self.image_hw = tuple(model_cfg.get('image_hw', [224, 112]))
        self.feature_hw = tuple(model_cfg.get('feature_hw', [32, 16]))
        D = self.adapter_dim = int(model_cfg.get('adapter_dim', 256))
        P = self.parts_num = int(model_cfg.get('parts_num', 10))
        self.hpp_bins = model_cfg.get('bin_num', [16, 8, 4, 2, 1])
        self.hpp_parts = sum(self.hpp_bins)
        K = self.num_phase_bins = int(model_cfg.get('num_phase_bins', 16))
        self.use_sil = bool(model_cfg.get('use_sil', True))
        self.use_pose_teacher = bool(model_cfg.get('use_pose_teacher', True))
        self.use_pose_guided_pq = bool(model_cfg.get('use_pose_guided_pq', True))
        self.pose_parts_num = int(model_cfg.get('pose_parts_num', 1 if self.use_pose_teacher else 0))
        self.total_parts = self.hpp_parts + P + self.pose_parts_num

        # Backbone
        sel = model_cfg.get('lvm_layers', [3, 7, 11])
        lvm_d = int(model_cfg.get('lvm_dim', 384))
        self.LVM = FrozenDINOBackbone(model_cfg.get('lvm_name_or_path'), lvm_d, sel,
                                      bool(model_cfg.get('freeze_lvm', True)))
        self.Adapter = LayerWiseGaitAdapter(self.LVM.out_dim, D, self.feature_hw, len(sel))

        # [UPGRADE A] Pose-Guided Part Query
        nj = int(model_cfg.get('num_pose_joints', 17))
        # PoseGuidedPartQueryGauPE is backward-compatible with pose=None,
        # so it also serves as the learned-query fallback.  This avoids importing
        # a non-existent v1 module in a clean OpenGait checkout.
        self.PartQuery = PoseGuidedPartQueryGauPE(
            D, P, nj, model_cfg.get('part_temperature', 0.07))

        # [UPGRADE B] KineTF with temporal PE
        self.KineTF = CrossPartKinematicTransformerV2(
            D, P, depth=int(model_cfg.get('kinematic_depth', 2)),
            heads=int(model_cfg.get('kinematic_heads', 4)),
            graph_bias_strength=float(model_cfg.get('graph_bias_strength', 0.25)))

        # Phase-Canonical
        self.Phase = PhaseCanonicalModule(
            P, D, K, tuple(model_cfg.get('foot_indices', [7, 8])),
            float(model_cfg.get('phase_temperature', 1.0)))

        # [UPGRADE E] Cycle decomposer
        self.CycleDecomp = CycleDecomposer(D, K)

        # [UPGRADE C] Phase-Canonical Inference Head
        self.CycleInfHead = PhaseCanonicalInferenceHead(D, K, P, D)

        # Silhouette
        if self.use_sil:
            self.SilModule = SilhouetteAwareModule(D, self.feature_hw)

        # Heads
        self.HPP = HorizontalPoolingPyramid(self.hpp_bins)
        fc_cfg = dict(model_cfg['SeparateFCs'])
        fc_cfg.pop('parts_num', None)
        self.FCs = SeparateFCs(parts_num=self.total_parts, **fc_cfg)
        bn_cfg = dict(model_cfg['SeparateBNNecks'])
        bn_cfg.pop('parts_num', None)
        bn_cfg['parts_num'] = self.total_parts
        self.BNNecks = SeparateBNNecks(**bn_cfg)

        self.cycle_pool_fc = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))

        # Text
        td = int(model_cfg.get('text_dim', 512))
        self.MotionTextProj = SequenceFeatureProjector(td, D)
        self.NuisanceTextProj = SequenceFeatureProjector(td, D)

        # Teachers
        sd = int(model_cfg.get('smpl_dim', 256))
        self.SMPLTeacher = UncertaintySMPLTeacher(sd, D)
        self.PCTeacher = UncertaintyPCTeacher(D, int(model_cfg.get('pc_in_dim', 4)))
        if self.use_pose_teacher:
            self.PoseTeacher = PoseKinematicTeacher(D, nj)
        if self.pose_parts_num > 0:
            self.PosePartProj = nn.Sequential(
                nn.LayerNorm(D), nn.Linear(D, D * self.pose_parts_num))

        # Gate: semantic / kinematic / optional pose branch.  Event is excluded.
        ng = 3 if self.use_pose_teacher and self.pose_parts_num > 0 else 2
        self.QualityGate = nn.Sequential(nn.Linear(D * ng, D), nn.GELU(), nn.Linear(D, ng))
        self._ng = ng

    def init_parameters(self):
        for name, m in self.named_modules():
            if name.startswith('LVM.model'): continue
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        np_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.msg_mgr.log_info(f'RealKineGait++ v2 trainable params: {np_ / 1e6:.3f}M')

    def _get(self, ipts, idx):
        if idx is None or idx >= len(ipts):
            return None
        return ipts[idx]

    def _restore_one_from_seqL(self, x, seqL):
        """Restore OpenGait's all_ordered concatenated layout.

        For fixed sampling OpenGait gives tensors as [N, T, ...] and seqL=None.
        For all_ordered/unfixed sampling it gives [1, sum(T_i), ...] plus
        seqL=[[T_1, ..., T_N]].  Baseline models call PackSequenceWrapper;
        this model needs explicit restoration before frame-wise LVM processing.
        """
        if x is None or seqL is None or (not torch.is_tensor(x)) or x.dim() < 2:
            return x
        lengths = seqL[0].detach().cpu().long().tolist() if torch.is_tensor(seqL) else list(seqL[0])
        lengths = [int(l) for l in lengths]
        if len(lengths) <= 1:
            return x
        if x.size(0) == len(lengths):
            return x
        total = sum(lengths)
        if x.size(0) != 1 or x.size(1) < total:
            return x
        max_len = max(lengths)
        out_shape = (len(lengths), max_len) + tuple(x.shape[2:])
        out = x.new_zeros(out_shape)
        start = 0
        for i, length in enumerate(lengths):
            if length > 0:
                out[i, :length] = x[0, start:start + length]
            start += length
        return out

    def _restore_ipts_from_seqL(self, ipts, seqL):
        if seqL is None:
            return ipts
        return [self._restore_one_from_seqL(x, seqL) for x in ipts]

    def _prep_rgb(self, rgb):
        if rgb.dim() == 4: rgb = rgb.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        if rgb.size(2) == 1: rgb = rgb.repeat(1, 1, 3, 1, 1)
        n, s, c, h, w = rgb.shape
        x = rearrange(rgb, 'n s c h w -> (n s) c h w').contiguous()
        return F.interpolate(x, self.image_hw, mode='bilinear', align_corners=False), n, s

    def _prep_event(self, event, n, s):
        if event is None or event.dim() != 5: return None
        ec = int(self.model_cfg.get('event_in_channels', 2))
        if event.size(2) != ec: event = event[:, :, :ec]
        ev = rearrange(event, 'n s c h w -> (n s) c h w').contiguous()
        ev = F.interpolate(ev, self.feature_hw, mode='bilinear', align_corners=False)
        return rearrange(ev, '(n s) c h w -> n c s h w', n=n, s=s).contiguous()

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        ipts = self._restore_ipts_from_seqL(ipts, seqL)
        rgb = self._get(ipts, self.rgb_index)
        if rgb is None:
            raise ValueError('RealKineGaitPP_v2 requires an RGB stream. Check model_cfg.input_indices.rgb and data_cfg.data_in_use.')
        x, n, s = self._prep_rgb(rgb)

        # === Backbone ===
        with (torch.no_grad() if self.model_cfg.get('freeze_lvm', True)
              else torch.enable_grad()):
            hidden = self.LVM(x)
        feat5d = self.Adapter(hidden, n, s)

        # === Silhouette ===
        sil = self._get(ipts, self.sil_index)
        if self.use_sil and hasattr(self, 'SilModule'):
            sil_out = self.SilModule(feat5d, sil, n, s)
            fg_feat5d = sil_out['fg_feat5d']
            bg_embed = sil_out['bg_embed']
            sil_mask = sil_out['sil_mask']
        else:
            fg_feat5d = feat5d
            bg_embed = torch.zeros(n, self.adapter_dim, device=feat5d.device, dtype=feat5d.dtype)
            sil_mask = None

        # === [UPGRADE A] Pose-Guided Part Query ===
        pose = self._get(ipts, self.pose_index)
        pq_pose = pose if self.use_pose_guided_pq else None
        pq = self.PartQuery(fg_feat5d, pq_pose)

        # === [UPGRADE B] KineTF with temporal PE ===
        kine_seq = self.KineTF(pq['part_feat'])

        # === Phase-Canonical ===
        phase_out = self.Phase(pq['mu'], kine_seq)
        cycle_tokens = phase_out['cycle_tokens']

        # === [UPGRADE E] Cycle decomposition ===
        decomp = self.CycleDecomp(cycle_tokens)
        intra_cycle = decomp['intra']  # [N, K, P, C] — identity
        inter_cycle = decomp['inter']  # [N, C] — style

        # Kinematic identity from intra-cycle
        cycle_pooled = self.cycle_pool_fc(intra_cycle.mean(1))  # [N, P, C]
        kine_parts = cycle_pooled.permute(0, 2, 1).contiguous()
        z_kin = intra_cycle.mean((1, 2))

        # === [UPGRADE C] Phase-canonical inference head ===
        cycle_inf = self.CycleInfHead(intra_cycle)

        # Semantic branch
        sem_map = feat5d.mean(2)
        sem_parts = self.HPP(sem_map)

        # === Teachers ===
        if self.use_pose_teacher and hasattr(self, 'PoseTeacher') and pose is not None and pose.dim() >= 3:
            z_pose, pose_lv = self.PoseTeacher(pose)
        else:
            z_pose = torch.zeros_like(z_kin)
            pose_lv = torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype)

        # Gate. Event is deliberately excluded.  Pose is no longer only an
        # indirect kinematic amplifier: when available it becomes a dedicated
        # identity part that goes through SeparateFCs/BNNecks and the final metric.
        z_sem = sem_map.mean((2, 3))
        if self._ng == 3:
            gate_inp = torch.cat([z_sem, z_kin, z_pose], -1)
        else:
            gate_inp = torch.cat([z_sem, z_kin], -1)
        gate = torch.softmax(self.QualityGate(gate_inp), -1)
        sem_parts = sem_parts * gate[:, 0].view(n, 1, 1)
        kine_parts = kine_parts * gate[:, 1].view(n, 1, 1)

        part_list = [sem_parts, kine_parts]
        if self.pose_parts_num > 0 and hasattr(self, 'PosePartProj'):
            pose_parts = self.PosePartProj(z_pose).view(n, self.pose_parts_num, self.adapter_dim)
            pose_parts = pose_parts.permute(0, 2, 1).contiguous()
            pose_gate = gate[:, 2].view(n, 1, 1) if self._ng == 3 else 0.0
            pose_parts = pose_parts * pose_gate
            part_list.append(pose_parts)

        feat_parts = torch.cat(part_list, -1)
        embed_1 = self.FCs(feat_parts)
        embed_2, logits = self.BNNecks(embed_1)
        z_id = embed_1.mean(-1)

        # Text
        motion_text = self._get(ipts, self.motion_text_index)
        nuisance_text = self._get(ipts, self.nuisance_text_index)
        z_mt = self.MotionTextProj(motion_text) if motion_text is not None else torch.zeros_like(z_kin)
        z_nt = self.NuisanceTextProj(nuisance_text) if nuisance_text is not None else torch.zeros_like(z_kin)

        # Uncertainty distillation
        smpl = self._get(ipts, self.smpl_index)
        pc = self._get(ipts, self.pc_index)
        t_feats, t_lvs, t_valids = [], [], []
        for mod, enc_fn in [
            (smpl, lambda m: self.SMPLTeacher(m)),
            (pc if pc is not None and pc.dim() >= 4 else None, lambda m: self.PCTeacher(m)),
        ]:
            if mod is not None:
                zz, lv = enc_fn(mod)
                t_feats.append(zz); t_lvs.append(lv)
                t_valids.append(torch.ones(n, device=z_kin.device, dtype=z_kin.dtype))
            else:
                t_feats.append(torch.zeros_like(z_kin))
                t_lvs.append(torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype))
                t_valids.append(torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype))
        if self.use_pose_teacher:
            t_feats.append(z_pose); t_lvs.append(pose_lv)
            t_valids.append(torch.ones(n, device=z_kin.device, dtype=z_kin.dtype) if pose is not None
                            else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype))

        teacher_stack = torch.stack(t_feats, 1).detach()
        lv_stack = torch.stack(t_lvs, 1).detach()
        valid_stack = torch.stack(t_valids, 1)

        # === [UPGRADE D] Part-Joint Alignment output ===
        # pq['mu']: [N,T,P,2] — learned part centroids
        # pose: [N,T,J,3] — joint locations
        # We output both for PartJointAlignmentLoss to consume.

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                'cd': {'embeddings': z_id, 'nuisance_embeddings': z_nt, 'labels': labs},
                'motion_align': {'kin_embeddings': z_kin, 'motion_embeddings': z_mt},
                'counterfactual': {'embeddings': z_id, 'labels': labs, 'nuisance_embeddings': z_nt},
                'phase': {'phase_code': phase_out['phase_code'], 'part_mu': pq['mu'],
                          'cycle_tokens': cycle_tokens},
                'distill_unc': {'student_embeddings': z_kin, 'teacher_stack': teacher_stack,
                                'log_var_stack': lv_stack, 'valid_stack': valid_stack},
                'sil_fg': {'attn_map': pq['attn_map'], 'sil_mask': sil_mask},
                'sil_bg_adv': {'bg_embed': bg_embed, 'labels': labs},
                'pose_align': {'kin_embeddings': z_kin, 'pose_embeddings': z_pose},
                # [UPGRADE D] Part-Joint alignment
                'part_joint': {
                    'part_mu': pq['mu'],         # [N, T, P, 2]
                    'pose': pose,                # [N, T, J, 3] or None
                },
            },
            'visual_summary': {
                'image/rgb': (rgb[:, :min(4, rgb.size(1))].flatten(0, 1)
                              if rgb.dim() == 5
                              else rgb[:, :min(4, rgb.size(1))].unsqueeze(2).flatten(0, 1)),
                'scalar/gate_semantic': gate[:, 0].mean().detach(),
                'scalar/gate_kinematic': gate[:, 1].mean().detach(),
            },
            'inference_feat': {
                'embeddings': embed_1,
                'kin_embeddings': z_kin,
                'phase_code': phase_out['phase_code'].mean(1),
                # [UPGRADE C] Phase-canonical inference descriptors
                'cycle_id': cycle_inf['cycle_id'],      # [N, K, D]
                'cycle_agg': cycle_inf['cycle_agg'],     # [N, D]
                'inter_cycle': inter_cycle,              # [N, D] style
            },
        }
        if self._ng == 3:
            retval['visual_summary']['scalar/gate_pose'] = gate[:, 2].mean().detach()
        return retval


# OpenGait model registry aliases. Both config names are supported.
RealKineGaitPPv2 = RealKineGaitPP_v2
RealKineGaitPPV2 = RealKineGaitPP_v2
