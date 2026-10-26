"""RealKineGait++ — Phase-Canonical Causal Kinematic Identity Recognition.

Upgrades over v0.1 (RealKineGait):
  1. Phase-Canonical Kinematic Tokens — maps irregular walking sequences into a
     normalised gait-cycle coordinate system so identity is compared at matched
     gait phases instead of matched frame indices.
  2. Causal Nuisance Intervention support — foreground/background causal streams
     via silhouette; cross-identity same-clothing hard negatives; explicit
     do-operations on clothing, viewpoint and background.
  3. Uncertainty-aware Privileged Distillation — SMPL, point-cloud and event
     teachers each produce features *and* uncertainty scores; the student is
     supervised proportionally to each teacher's reliability.
  4. Silhouette integration — sil-guided RGB masking, attention-inside-foreground
     loss, and contour perturbation augmentation.

Inference still only requires RGB.
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


# ---------------------------------------------------------------------------
# Backbone — identical to v0.1 (kept for self-containedness)
# ---------------------------------------------------------------------------

class TinyFrameBackbone(nn.Module):
    """Light fallback when transformer weights are unavailable."""

    def __init__(self, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(out_dim), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y = self.net(x)
        tokens = y.flatten(2).transpose(1, 2).contiguous()
        cls = tokens.mean(1, keepdim=True)
        return [torch.cat([cls, tokens], 1) for _ in range(4)]


class FrozenDINOBackbone(nn.Module):
    def __init__(self, name_or_path: Optional[str], out_dim: int,
                 selected_layers: List[int], freeze: bool = True):
        super().__init__()
        self.name_or_path = name_or_path
        self.selected_layers = selected_layers
        self.out_dim = out_dim
        self.use_hf = False
        if name_or_path:
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    name_or_path, trust_remote_code=True)
                self.use_hf = True
                hidden = getattr(self.model.config, 'hidden_size', out_dim)
                self.out_dim = int(hidden)
                if freeze:
                    self.model.eval()
                    self.model.requires_grad_(False)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load LVM from {name_or_path}. "
                    f"Falling back to TinyFrameBackbone. Error: {exc}")
                self.model = TinyFrameBackbone(out_dim)
        else:
            self.model = TinyFrameBackbone(out_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_hf:
            self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if not self.use_hf:
            return self.model(x)
        try:
            outs = self.model(
                x, output_hidden_states=True, interpolate_pos_encoding=True)
        except TypeError:
            outs = self.model(x, output_hidden_states=True)
        hiddens = list(outs.hidden_states[1:])
        max_idx = len(hiddens) - 1
        selected = []
        for idx in self.selected_layers:
            real = max(0, min(max_idx,
                              idx if idx >= 0 else len(hiddens) + idx))
            selected.append(hiddens[real])
        return selected


# ---------------------------------------------------------------------------
# Layer-wise Gait Adapter (unchanged)
# ---------------------------------------------------------------------------

class LayerWiseGaitAdapter(nn.Module):
    def __init__(self, lvm_dim: int, adapter_dim: int,
                 out_hw: Tuple[int, int], num_layers: int):
        super().__init__()
        self.out_hw = out_hw
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(lvm_dim), nn.Linear(lvm_dim, adapter_dim),
                nn.GELU())
            for _ in range(num_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(adapter_dim * num_layers, adapter_dim, 1, bias=False),
            nn.BatchNorm2d(adapter_dim), nn.GELU(),
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(adapter_dim), nn.GELU(),
        )

    @staticmethod
    def tokens_to_map(tokens: torch.Tensor,
                      out_hw: Tuple[int, int]) -> torch.Tensor:
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
        feat = tokens.transpose(1, 2).reshape(b, c, h, w)
        return F.interpolate(feat, out_hw, mode='bilinear',
                             align_corners=False)

    def forward(self, hidden_states: List[torch.Tensor],
                batch: int, frames: int) -> torch.Tensor:
        maps = []
        for h, proj in zip(hidden_states, self.projs):
            z = proj(h)
            maps.append(self.tokens_to_map(z, self.out_hw))
        x = self.fuse(torch.cat(maps, 1))
        return rearrange(x, '(n s) c h w -> n c s h w',
                         n=batch, s=frames).contiguous()


# ---------------------------------------------------------------------------
# PartQuery GauPE++ (unchanged from v0.1)
# ---------------------------------------------------------------------------

class PartQueryGauPE(nn.Module):
    def __init__(self, dim: int, parts_num: int,
                 temperature: float = 0.07, eps: float = 1e-6):
        super().__init__()
        self.parts_num = parts_num
        self.temperature = temperature
        self.eps = eps
        self.queries = nn.Parameter(torch.randn(parts_num, dim) * 0.02)
        self.part_proj = nn.Linear(dim, dim)
        self.gaupe_proj = nn.Sequential(
            nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        n, c, t, h, w = feat.shape
        x = rearrange(feat, 'n c t h w -> (n t) (h w) c')
        q = F.normalize(self.queries, dim=-1)
        k = F.normalize(x, dim=-1)
        attn = torch.einsum('pc,blc->bpl', q, k) / max(
            self.temperature, self.eps)
        attn = torch.softmax(attn, dim=-1)
        part_feat = torch.einsum('bpl,blc->bpc', attn, x)
        attn_map = attn.reshape(n, t, self.parts_num, h, w)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).reshape(1, 1, 1, h * w, 2)
        prob = attn.reshape(n, t, self.parts_num, h * w, 1)
        mu = (prob * grid).sum(3)                     # [N,T,P,2]
        delta = grid - mu.unsqueeze(3)
        var = (prob * delta ** 2).sum(3)
        cov = (prob[..., 0] * delta[..., 0] * delta[..., 1]).sum(
            3, keepdim=True)
        theta = 0.5 * torch.atan2(
            2 * cov,
            (var[..., 0:1] - var[..., 1:2]).clamp_min(self.eps))
        geom = torch.cat([mu, var, cov, theta], -1)   # 6 dims

        vel = torch.zeros_like(mu)
        vel[:, 1:] = mu[:, 1:] - mu[:, :-1]
        acc = torch.zeros_like(mu)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        geom12 = torch.cat([geom, vel, acc], -1)       # [N,T,P,10]
        pad = torch.zeros(*geom12.shape[:-1], 2,
                          device=feat.device, dtype=feat.dtype)
        geom12 = torch.cat([geom12, pad], -1)

        part_feat = self.part_proj(part_feat).reshape(n, t, self.parts_num, c)
        part_feat = part_feat + self.gaupe_proj(geom12)
        return {
            'part_feat': part_feat,
            'attn_map': attn_map,
            'mu': mu, 'vel': vel, 'acc': acc, 'geom': geom12,
        }


# ---------------------------------------------------------------------------
# Cross-Part Kinematic Transformer (unchanged)
# ---------------------------------------------------------------------------

class GraphTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossPartKinematicTransformer(nn.Module):
    def __init__(self, dim: int, parts_num: int, depth: int = 2,
                 heads: int = 4, graph_bias_strength: float = 0.25):
        super().__init__()
        self.parts_num = parts_num
        self.graph_bias_strength = graph_bias_strength
        self.blocks = nn.ModuleList(
            [GraphTemporalBlock(dim, heads) for _ in range(depth)])
        self.part_type = nn.Parameter(
            torch.randn(1, parts_num, dim) * 0.02)

    def _mask(self, frames: int, device, dtype) -> torch.Tensor:
        p = self.parts_num
        edges = {(i, i) for i in range(p)}

        def add(a, b):
            if a < p and b < p:
                edges.add((a, b))
                edges.add((b, a))

        add(1, 2); add(3, 6); add(4, 5); add(5, 6)
        add(5, 7); add(6, 8); add(2, 5); add(2, 6)
        base = torch.full(
            (p, p), -float(self.graph_bias_strength),
            device=device, dtype=dtype)
        for a, b in edges:
            base[a, b] = 0.0
        mask = torch.zeros(
            frames * p, frames * p, device=device, dtype=dtype)
        for tt in range(frames):
            mask[tt * p:(tt + 1) * p, tt * p:(tt + 1) * p] = base
        return mask

    def forward(self, part_feat: torch.Tensor) -> torch.Tensor:
        n, t, p, c = part_feat.shape
        x = part_feat + self.part_type[:, :p].unsqueeze(1)
        x = rearrange(x, 'n t p c -> n (t p) c')
        mask = self._mask(t, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        return rearrange(x, 'n (t p) c -> n t p c', t=t, p=p).contiguous()


# ===================================================================
# INNOVATION 1 : Phase-Canonical Kinematic Tokens
# ===================================================================

class PhaseCanonicalModule(nn.Module):
    """Map irregular walking sequences into K canonical gait-cycle phase bins.

    Pipeline:
        mu (part centroids) ──► continuous phase θ(t) ∈ [0, 2π)
                              ──► soft assignment to K phase bins
                              ──► weighted aggregation of part-kinematic
                                  features into canonical cycle tokens
                                  Z_cycle: [N, K, P, C]

    The module also outputs a smooth phase code for the auxiliary
    PhaseConsistencyLoss and the raw phase values for downstream use.
    """

    def __init__(self, parts_num: int, dim: int, num_bins: int = 16,
                 foot_indices: Tuple[int, int] = (7, 8),
                 phase_temperature: float = 1.0):
        super().__init__()
        self.parts_num = parts_num
        self.dim = dim
        self.num_bins = num_bins
        self.foot_indices = foot_indices
        self.phase_temperature = phase_temperature

        # Learnable phase estimator: takes foot-signal features → continuous
        # phase.  We concatenate left/right foot x-displacement, velocity, and
        # acceleration → 6-D input per frame.
        self.phase_net = nn.Sequential(
            nn.Linear(6, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 2),   # output (sin θ, cos θ)
        )

        # Canonical bin centres: uniformly spaced on the unit circle.
        angles = torch.linspace(0, 2 * math.pi * (1 - 1 / num_bins),
                                num_bins)
        self.register_buffer('bin_sin', angles.sin())  # [K]
        self.register_buffer('bin_cos', angles.cos())  # [K]

        # Per-bin learnable scale (temperature sharpening)
        self.bin_logit_scale = nn.Parameter(torch.ones(1) * 2.0)

        # Small projection applied *after* cycle-pooling for refinement
        self.cycle_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def _foot_signal(self, mu: torch.Tensor) -> torch.Tensor:
        """Extract foot kinematics from part centroids.

        Args:
            mu: [N, T, P, 2] — part centroid coordinates.

        Returns:
            [N, T, 6] — (left_x, right_x, left_vx, right_vx,
                          left_ax, right_ax)
        """
        n, t, p, _ = mu.shape
        li = min(self.foot_indices[0], p - 1)
        ri = min(self.foot_indices[1], p - 1)
        lx = mu[:, :, li, 0]   # [N, T]
        rx = mu[:, :, ri, 0]
        # Velocity
        lv = torch.zeros_like(lx)
        rv = torch.zeros_like(rx)
        lv[:, 1:] = lx[:, 1:] - lx[:, :-1]
        rv[:, 1:] = rx[:, 1:] - rx[:, :-1]
        # Acceleration
        la = torch.zeros_like(lx)
        ra = torch.zeros_like(rx)
        la[:, 1:] = lv[:, 1:] - lv[:, :-1]
        ra[:, 1:] = rv[:, 1:] - rv[:, :-1]
        return torch.stack([lx, rx, lv, rv, la, ra], dim=-1)  # [N,T,6]

    def forward(self, mu: torch.Tensor,
                kine_seq: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mu:        [N, T, P, 2] part centroids from GauPE.
            kine_seq:  [N, T, P, C] part-kinematic features from KineTF.

        Returns dict:
            cycle_tokens:  [N, K, P, C]  phase-canonical representation.
            phase_code:    [N, T, 2]     (sin θ, cos θ) continuous phase.
            phase_weight:  [N, T, K]     soft assignment weights.
            foot_signal:   [N, T, 6]     raw foot kinematics.
        """
        n, t, p, c = kine_seq.shape

        # --- Step 1: continuous phase estimation ---
        foot_feat = self._foot_signal(mu)               # [N, T, 6]
        phase_sc = self.phase_net(foot_feat)             # [N, T, 2]
        # Normalise to unit circle
        phase_norm = phase_sc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_sc = phase_sc / phase_norm                 # (sinθ, cosθ)

        # --- Step 2: soft phase binning ---
        # Cosine similarity between frame phase and bin centres
        # phase_sc: [N, T, 2],  bin centres: [K, 2]
        bin_vecs = torch.stack([self.bin_sin, self.bin_cos], dim=-1)  # [K,2]
        # cos(θ_frame - θ_bin) = sin·sin + cos·cos
        sim = torch.einsum('ntd,kd->ntk', phase_sc, bin_vecs)  # [N,T,K]
        # Scale and softmax
        scale = self.bin_logit_scale.clamp_min(0.1)
        phase_weight = torch.softmax(sim * scale, dim=-1)       # [N,T,K]

        # --- Step 3: weighted aggregation into canonical cycle tokens ---
        # kine_seq: [N,T,P,C],  phase_weight: [N,T,K]
        # cycle_tokens[n,k,p,c] = Σ_t  w[n,t,k] * kine_seq[n,t,p,c]
        cycle_tokens = torch.einsum(
            'ntk,ntpc->nkpc', phase_weight, kine_seq)           # [N,K,P,C]

        # Normalise by total weight per bin to avoid magnitude bias
        weight_sum = phase_weight.sum(dim=1, keepdim=False)      # [N,K]
        weight_sum = weight_sum.clamp_min(1e-6).unsqueeze(-1).unsqueeze(-1)
        cycle_tokens = cycle_tokens / weight_sum                 # [N,K,P,C]

        # Refinement projection
        cycle_tokens = cycle_tokens + self.cycle_proj(cycle_tokens)

        return {
            'cycle_tokens': cycle_tokens,   # [N, K, P, C]
            'phase_code': phase_sc,         # [N, T, 2]
            'phase_weight': phase_weight,   # [N, T, K]
            'foot_signal': foot_feat,       # [N, T, 6]
        }


# ===================================================================
# INNOVATION 4 : Silhouette Integration
# ===================================================================

class SilhouetteAwareModule(nn.Module):
    """Three uses of silhouette:
    1. sil-guided RGB masking (foreground / background streams)
    2. attention-inside-foreground penalty
    3. background adversarial head (background cannot predict identity)
    """

    def __init__(self, adapter_dim: int, feature_hw: Tuple[int, int]):
        super().__init__()
        self.feature_hw = feature_hw
        # Background adversarial head — a small identity classifier that
        # we train to *fail* (gradient reversal via negative loss weight).
        self.bg_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(adapter_dim, adapter_dim // 2),
            nn.GELU(),
            nn.Linear(adapter_dim // 2, adapter_dim),
        )

    def _resize_sil(self, sil: torch.Tensor,
                    n: int, s: int) -> torch.Tensor:
        """Resize silhouette to feature_hw.
        Input sil: [N, S, H, W] or [N, S, 1, H, W].
        Output: [N, 1, S, Hf, Wf] broadcast-friendly with feat5d.
        """
        if sil.dim() == 5:
            sil = sil[:, :, 0]                        # [N, S, H, W]
        sil_flat = rearrange(sil, 'n s h w -> (n s) 1 h w').float()
        sil_flat = F.interpolate(
            sil_flat, self.feature_hw, mode='bilinear',
            align_corners=False)
        sil_5d = rearrange(sil_flat, '(n s) 1 h w -> n 1 s h w',
                           n=n, s=s)
        # Soft binarise: make sure values are in [0, 1]
        sil_5d = sil_5d.clamp(0, 1)
        return sil_5d

    def forward(self, feat5d: torch.Tensor,
                sil: Optional[torch.Tensor],
                n: int, s: int
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat5d: [N, C, S, H, W]
            sil:    raw silhouette tensor or None.
            n, s:   batch and sequence lengths.

        Returns dict with:
            fg_feat5d:         [N, C, S, H, W]  foreground-masked features.
            bg_feat5d:         [N, C, S, H, W]  background-masked features.
            sil_mask:          [N, 1, S, H, W]  resized silhouette mask.
            bg_embed:          [N, D]  background embedding for adversarial loss.
            attn_fg_penalty:   scalar — attention-outside-sil penalty (placeholder;
                               actual penalty computed in loss with attn_map).
        """
        out: Dict[str, torch.Tensor] = {}

        if sil is None:
            # No silhouette → pass through; all penalties are zero.
            out['fg_feat5d'] = feat5d
            out['bg_feat5d'] = torch.zeros_like(feat5d)
            out['sil_mask'] = torch.ones(
                n, 1, s, *self.feature_hw,
                device=feat5d.device, dtype=feat5d.dtype)
            out['bg_embed'] = torch.zeros(
                n, feat5d.size(1), device=feat5d.device, dtype=feat5d.dtype)
            return out

        sil_mask = self._resize_sil(sil, n, s)          # [N,1,S,H,W]
        # Foreground / background split
        fg_feat5d = feat5d * sil_mask
        bg_feat5d = feat5d * (1.0 - sil_mask)

        # Background adversarial embedding (time-averaged)
        bg_map = bg_feat5d.mean(2)                       # [N, C, H, W]
        bg_embed = self.bg_head(bg_map)                  # [N, D]

        out['fg_feat5d'] = fg_feat5d
        out['bg_feat5d'] = bg_feat5d
        out['sil_mask'] = sil_mask
        out['bg_embed'] = bg_embed
        return out


# ===================================================================
# INNOVATION 3 : Uncertainty-aware Privileged Teachers
# ===================================================================

class UncertaintyHead(nn.Module):
    """Predicts a scalar log-uncertainty from a teacher's raw feature."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns log σ² (unbounded).  [N] or [N, 1] → [N]."""
        return self.net(x).squeeze(-1)


class UncertaintySMPLTeacher(nn.Module):
    """SMPL teacher with uncertainty estimation.

    Uncertainty proxies (when raw SMPL predictions are available):
      - reconstruction confidence (norm of residual)
      - pose prior likelihood (deviation from mean pose)
    Here we learn uncertainty jointly from the projected embedding.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
            nn.GELU(), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() > 2:
            x = x.mean(1)
        z = self.proj(x)
        log_var = self.unc(z)
        return z, log_var


class UncertaintyPCTeacher(nn.Module):
    """Point-cloud teacher with uncertainty estimation.

    Uncertainty correlates with: low point density, heavy occlusion,
    far range, missing body parts.
    """

    def __init__(self, out_dim: int, in_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, out_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)

        # Density-aware: estimate uncertainty from point count statistics
        self.density_unc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, pc: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pc.dim() == 5:
            pc = pc.flatten(2, 3)
        if pc.size(-1) < 4:
            pad = torch.zeros(
                *pc.shape[:-1], 4 - pc.size(-1),
                device=pc.device, dtype=pc.dtype)
            pc = torch.cat([pc, pad], -1)
        pc = pc[..., :4]

        # Point-level features → per-frame max pool → sequence mean
        x = self.mlp(pc).max(dim=2)[0]   # [N, T, C]
        z = self.proj(x.mean(1))          # [N, C]

        # Density statistics for explicit uncertainty input
        # pc: [N, T, Pts, 4]
        n_pts = (pc[..., :3].abs().sum(-1) > 1e-6).float()  # [N,T,Pts]
        pt_count = n_pts.sum(-1)                              # [N,T]
        density_stats = torch.stack([
            pt_count.mean(1),
            pt_count.std(1).clamp_min(1e-6),
            pt_count.min(1)[0],
        ], dim=-1)  # [N, 3]

        log_var_feat = self.unc(z)                            # [N]
        log_var_density = self.density_unc(density_stats).squeeze(-1)
        log_var = log_var_feat + log_var_density

        return z, log_var


class UncertaintyEventTeacher(nn.Module):
    """Event-stream teacher with uncertainty estimation.

    Uncertainty correlates with: low event rate, static scenes,
    illumination instability, motion blur.
    """

    def __init__(self, adapter_dim: int, in_channels: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, adapter_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(adapter_dim), nn.GELU())
        self.unc = UncertaintyHead(adapter_dim)

        # Event-rate based uncertainty
        self.rate_unc = nn.Sequential(
            nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, ev: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ev: [N, C, S, H, W]"""
        z_map = self.encoder(ev)              # [N, D, S, H, W]
        z = z_map.mean((2, 3, 4))             # [N, D]
        log_var_feat = self.unc(z)

        # Event rate statistics
        rate = ev.abs().mean((1, 3, 4))       # [N, S]
        rate_stats = torch.stack([
            rate.mean(1),
            rate.std(1).clamp_min(1e-6),
            rate.min(1)[0],
        ], dim=-1)                            # [N, 3]
        log_var_rate = self.rate_unc(rate_stats).squeeze(-1)
        log_var = log_var_feat + log_var_rate

        return z, log_var


# ---------------------------------------------------------------------------
# Pose Kinematic Teacher  (NEW — uses pose directly in forward)
# ---------------------------------------------------------------------------

class PoseKinematicTeacher(nn.Module):
    """Directly encode 2D/3D pose sequences into a kinematic embedding.

    Pose format: [N, T, J, 3]  (x, y, confidence).
    The teacher provides both a feature and a per-joint confidence-weighted
    uncertainty.
    """

    def __init__(self, out_dim: int, num_joints: int = 17):
        super().__init__()
        self.joint_embed = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU())
        # Temporal conv for joint-level dynamics
        self.temporal = nn.Sequential(
            nn.Conv1d(128 * num_joints, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim), nn.GELU())
        self.proj = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim))
        self.unc = UncertaintyHead(out_dim)
        self.num_joints = num_joints

    def forward(self, pose: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pose: [N, T, J, 3] or [N, T, J*3]
        Returns:
            z: [N, D]
            log_var: [N]
        """
        if pose.dim() == 3:
            # [N, T, J*3] → [N, T, J, 3]
            pose = pose.reshape(pose.size(0), pose.size(1), -1, 3)
        n, t, j, _ = pose.shape
        # Pad or truncate joints
        if j < self.num_joints:
            pad = torch.zeros(n, t, self.num_joints - j, 3,
                              device=pose.device, dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=2)
        pose = pose[:, :, :self.num_joints]

        x = self.joint_embed(pose)           # [N, T, J, 128]
        x = rearrange(x, 'n t j c -> n (j c) t')
        x = self.temporal(x)                 # [N, D, T]
        z = self.proj(x.mean(-1))            # [N, D]

        # Confidence-based uncertainty
        conf = pose[:, :, :, 2].clamp(0, 1)  # [N, T, J]
        mean_conf = conf.mean((1, 2))         # [N]
        log_var = self.unc(z) - mean_conf     # Lower confidence → higher unc

        return z, log_var


# ---------------------------------------------------------------------------
# Sequence Feature Projector (unchanged)
# ---------------------------------------------------------------------------

class SequenceFeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
            nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.mean(1)
        return self.net(x)


# ===================================================================
# MAIN MODEL : RealKineGait++
# ===================================================================

class RealKineGaitPP(BaseModel):
    """Phase-Canonical Causal Kinematic Identity Recognition.

    At inference only RGB is required.  All other modalities (silhouette,
    pose, text, SMPL, point cloud, event) serve as privileged training-time
    teachers, causal intervention signals, or auxiliary supervisors.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        self.input_indices = model_cfg.get('input_indices', {'rgb': 0})
        self.rgb_index = self.input_indices.get('rgb', 0)
        self.sil_index = self.input_indices.get('sil', None)
        self.pose_index = self.input_indices.get('pose', None)
        self.motion_text_index = self.input_indices.get('motion_text', None)
        self.nuisance_text_index = self.input_indices.get(
            'nuisance_text', None)
        self.smpl_index = self.input_indices.get('smpl', None)
        self.pc_index = self.input_indices.get('pointcloud', None)
        self.event_index = self.input_indices.get('event', None)

        self.image_hw = tuple(model_cfg.get('image_hw', [224, 112]))
        self.feature_hw = tuple(model_cfg.get('feature_hw', [32, 16]))
        self.adapter_dim = int(model_cfg.get('adapter_dim', 256))
        self.parts_num = int(model_cfg.get('parts_num', 10))
        self.hpp_bins = model_cfg.get('bin_num', [16, 8, 4, 2, 1])
        self.hpp_parts = sum(self.hpp_bins)
        self.total_parts = self.hpp_parts + self.parts_num
        self.num_phase_bins = int(model_cfg.get('num_phase_bins', 16))
        self.use_sil = bool(model_cfg.get('use_sil', True))
        self.use_pose_teacher = bool(model_cfg.get('use_pose_teacher', True))

        # --- Backbone ---
        selected_layers = model_cfg.get('lvm_layers', [3, 7, 11])
        lvm_hint_dim = int(model_cfg.get('lvm_dim', 384))
        self.LVM = FrozenDINOBackbone(
            name_or_path=model_cfg.get('lvm_name_or_path', None),
            out_dim=lvm_hint_dim,
            selected_layers=selected_layers,
            freeze=bool(model_cfg.get('freeze_lvm', True)))
        self.Adapter = LayerWiseGaitAdapter(
            self.LVM.out_dim, self.adapter_dim,
            self.feature_hw, len(selected_layers))

        # --- Part Query ---
        self.PartQuery = PartQueryGauPE(
            self.adapter_dim, self.parts_num,
            model_cfg.get('part_temperature', 0.07))

        # --- Kinematic Transformer ---
        self.KineTF = CrossPartKinematicTransformer(
            self.adapter_dim, self.parts_num,
            depth=int(model_cfg.get('kinematic_depth', 2)),
            heads=int(model_cfg.get('kinematic_heads', 4)),
            graph_bias_strength=float(
                model_cfg.get('graph_bias_strength', 0.25)))

        # --- [NEW] Phase-Canonical Module ---
        self.Phase = PhaseCanonicalModule(
            self.parts_num, self.adapter_dim,
            num_bins=self.num_phase_bins,
            foot_indices=tuple(model_cfg.get('foot_indices', [7, 8])),
            phase_temperature=float(
                model_cfg.get('phase_temperature', 1.0)))

        # --- [NEW] Silhouette Module ---
        if self.use_sil:
            self.SilModule = SilhouetteAwareModule(
                self.adapter_dim, self.feature_hw)

        # --- Semantic + Cycle Head ---
        self.HPP = HorizontalPoolingPyramid(self.hpp_bins)
        self.FCs = SeparateFCs(
            parts_num=self.total_parts, **model_cfg['SeparateFCs'])
        bn_cfg = dict(model_cfg['SeparateBNNecks'])
        bn_cfg['parts_num'] = self.total_parts
        self.BNNecks = SeparateBNNecks(**bn_cfg)

        # Phase-canonical pooling head: [N, K, P, C] → [N, C, P]
        self.cycle_pool_fc = nn.Sequential(
            nn.LayerNorm(self.adapter_dim),
            nn.Linear(self.adapter_dim, self.adapter_dim))

        # --- Text projectors ---
        text_dim = int(model_cfg.get('text_dim', 512))
        self.MotionTextProj = SequenceFeatureProjector(
            text_dim, self.adapter_dim)
        self.NuisanceTextProj = SequenceFeatureProjector(
            text_dim, self.adapter_dim)

        # --- [NEW] Uncertainty-aware privileged teachers ---
        smpl_dim = int(model_cfg.get('smpl_dim', 256))
        self.SMPLTeacher = UncertaintySMPLTeacher(
            smpl_dim, self.adapter_dim)
        self.PCTeacher = UncertaintyPCTeacher(
            self.adapter_dim,
            in_dim=int(model_cfg.get('pc_in_dim', 4)))
        self.EventTeacher = UncertaintyEventTeacher(
            self.adapter_dim,
            in_channels=int(model_cfg.get('event_in_channels', 2)))

        # --- [NEW] Pose Kinematic Teacher ---
        if self.use_pose_teacher:
            num_joints = int(model_cfg.get('num_pose_joints', 17))
            self.PoseTeacher = PoseKinematicTeacher(
                self.adapter_dim, num_joints)

        # Quality gate: now 4-way (semantic / kinematic / privileged / pose)
        n_gate = 4 if self.use_pose_teacher else 3
        gate_in = self.adapter_dim * n_gate
        self.QualityGate = nn.Sequential(
            nn.Linear(gate_in, self.adapter_dim), nn.GELU(),
            nn.Linear(self.adapter_dim, n_gate))
        self._n_gate = n_gate

        self.id_pool = nn.AdaptiveAvgPool1d(1)

    def init_parameters(self):
        for name, m in self.named_modules():
            if name.startswith('LVM.model'):
                continue
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d,
                                nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        n_params = sum(p.numel() for p in self.parameters()
                       if p.requires_grad)
        self.msg_mgr.log_info(
            f'RealKineGait++ trainable params: {n_params / 1e6:.3f}M')

    # ----- helpers -----

    def _get(self, ipts, idx):
        if idx is None or idx >= len(ipts):
            return None
        return ipts[idx]

    def _prep_rgb(self, rgb: torch.Tensor
                  ) -> Tuple[torch.Tensor, int, int]:
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        if rgb.size(2) == 1:
            rgb = rgb.repeat(1, 1, 3, 1, 1)
        n, s, c, h, w = rgb.shape
        x = rearrange(rgb, 'n s c h w -> (n s) c h w').contiguous()
        x = F.interpolate(x, self.image_hw, mode='bilinear',
                          align_corners=False)
        return x, n, s

    def _prep_event(self, event, n, s):
        if event is None or event.dim() != 5:
            return None
        ec = int(self.model_cfg.get('event_in_channels', 2))
        if event.size(2) != ec:
            event = event[:, :, :ec]
        ev = rearrange(event, 'n s c h w -> (n s) c h w').contiguous()
        ev = F.interpolate(ev, self.feature_hw, mode='bilinear',
                           align_corners=False)
        return rearrange(ev, '(n s) c h w -> n c s h w',
                         n=n, s=s).contiguous()

    # ----- forward -----

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        rgb = ipts[self.rgb_index]
        x, n, s = self._prep_rgb(rgb)

        # === LVM backbone ===
        with (torch.no_grad() if self.model_cfg.get('freeze_lvm', True)
              else torch.enable_grad()):
            hidden = self.LVM(x)
        feat5d = self.Adapter(hidden, n, s)      # [N, C, S, H, W]

        # === Silhouette-guided foreground / background split ===
        sil = self._get(ipts, self.sil_index)
        if self.use_sil and hasattr(self, 'SilModule'):
            sil_out = self.SilModule(feat5d, sil, n, s)
            fg_feat5d = sil_out['fg_feat5d']
            bg_embed = sil_out['bg_embed']
            sil_mask = sil_out['sil_mask']
        else:
            fg_feat5d = feat5d
            bg_embed = torch.zeros(
                n, self.adapter_dim,
                device=feat5d.device, dtype=feat5d.dtype)
            sil_mask = None

        # === Part Query on foreground features ===
        pq = self.PartQuery(fg_feat5d)
        kine_seq = self.KineTF(pq['part_feat'])  # [N, T, P, C]

        # === [INNOVATION 1] Phase-Canonical Tokens ===
        phase_out = self.Phase(pq['mu'], kine_seq)
        cycle_tokens = phase_out['cycle_tokens']  # [N, K, P, C]

        # Cycle-canonical identity features: pool over K and project
        cycle_pooled = self.cycle_pool_fc(
            cycle_tokens.mean(1))                 # [N, P, C]
        kine_parts = cycle_pooled.permute(0, 2, 1).contiguous()  # [N,C,P]
        z_kin = cycle_tokens.mean((1, 2))         # [N, C]

        # Semantic HPP branch (on full features, not just FG, for
        # comparison — the identity should come from FG anyway)
        sem_map = feat5d.mean(2)                  # [N, C, H, W]
        sem_parts = self.HPP(sem_map)             # [N, C, P_hpp]

        # === Pose kinematic teacher ===
        pose = self._get(ipts, self.pose_index)
        if (self.use_pose_teacher and hasattr(self, 'PoseTeacher')
                and pose is not None and pose.dim() >= 3):
            z_pose, pose_log_var = self.PoseTeacher(pose)
        else:
            z_pose = torch.zeros_like(z_kin)
            pose_log_var = torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype)

        # === Event teacher ===
        event = self._get(ipts, self.event_index)
        ev_vol = self._prep_event(event, n, s)
        if ev_vol is not None:
            z_event, event_log_var = self.EventTeacher(ev_vol)
        else:
            z_event = torch.zeros_like(z_kin)
            event_log_var = torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype)

        # === Quality-aware gating ===
        z_sem = sem_map.mean((2, 3))              # [N, C]
        if self._n_gate == 4:
            gate_inp = torch.cat(
                [z_sem, z_kin, z_event, z_pose], -1)
        else:
            gate_inp = torch.cat([z_sem, z_kin, z_event], -1)
        gate = torch.softmax(self.QualityGate(gate_inp), -1)  # [N, n_gate]

        sem_parts = sem_parts * gate[:, 0].view(n, 1, 1)
        kin_gate = gate[:, 1].view(n, 1, 1)
        priv_gate = gate[:, 2].view(n, 1, 1)
        if self._n_gate == 4:
            pose_gate = gate[:, 3].view(n, 1, 1)
            kine_parts = kine_parts * (kin_gate + priv_gate + pose_gate)
        else:
            kine_parts = kine_parts * (kin_gate + priv_gate)

        # Fuse semantic and kinematic
        feat_parts = torch.cat([sem_parts, kine_parts], -1)   # [N,C,P_total]
        embed_1 = self.FCs(feat_parts)
        embed_2, logits = self.BNNecks(embed_1)
        z_id = embed_1.mean(-1)                               # [N, C]

        # === Text projections ===
        motion_text = self._get(ipts, self.motion_text_index)
        nuisance_text = self._get(ipts, self.nuisance_text_index)
        z_motion_text = (self.MotionTextProj(motion_text)
                         if motion_text is not None
                         else torch.zeros_like(z_kin))
        z_nuisance_text = (self.NuisanceTextProj(nuisance_text)
                           if nuisance_text is not None
                           else torch.zeros_like(z_kin))

        # === [INNOVATION 3] Uncertainty-aware privileged distillation ===
        smpl = self._get(ipts, self.smpl_index)
        pc = self._get(ipts, self.pc_index)

        # Collect teacher outputs and uncertainties
        teacher_feats = []
        teacher_log_vars = []
        teacher_valid_flags = []

        # SMPL
        if smpl is not None:
            z_smpl, smpl_log_var = self.SMPLTeacher(smpl)
            teacher_feats.append(z_smpl)
            teacher_log_vars.append(smpl_log_var)
            teacher_valid_flags.append(torch.ones(
                n, device=z_kin.device, dtype=z_kin.dtype))
        else:
            teacher_feats.append(torch.zeros_like(z_kin))
            teacher_log_vars.append(torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype))
            teacher_valid_flags.append(torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype))

        # Point cloud
        if pc is not None and pc.dim() >= 4:
            z_pc, pc_log_var = self.PCTeacher(pc)
            teacher_feats.append(z_pc)
            teacher_log_vars.append(pc_log_var)
            teacher_valid_flags.append(torch.ones(
                n, device=z_kin.device, dtype=z_kin.dtype))
        else:
            teacher_feats.append(torch.zeros_like(z_kin))
            teacher_log_vars.append(torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype))
            teacher_valid_flags.append(torch.zeros(
                n, device=z_kin.device, dtype=z_kin.dtype))

        # Event
        teacher_feats.append(z_event)
        teacher_log_vars.append(event_log_var)
        teacher_valid_flags.append(
            torch.ones(n, device=z_kin.device, dtype=z_kin.dtype)
            if ev_vol is not None
            else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype))

        # Pose
        if self.use_pose_teacher:
            teacher_feats.append(z_pose)
            teacher_log_vars.append(pose_log_var)
            teacher_valid_flags.append(
                torch.ones(n, device=z_kin.device, dtype=z_kin.dtype)
                if pose is not None
                else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype))

        teacher_stack = torch.stack(teacher_feats, 1)      # [N, M, C]
        log_var_stack = torch.stack(teacher_log_vars, 1)    # [N, M]
        valid_stack = torch.stack(teacher_valid_flags, 1)   # [N, M]

        # Uncertainty weight: w_m ∝ valid_m · exp(-log_var_m)
        unc_weight = valid_stack * torch.exp(-log_var_stack.clamp(-10, 10))
        unc_weight = unc_weight / unc_weight.sum(1, keepdim=True).clamp_min(
            1e-6)
        z_teacher = (teacher_stack * unc_weight.unsqueeze(-1)).sum(1).detach()

        # =================== BUILD RETURN DICT ===================
        retval = {
            'training_feat': {
                'triplet': {
                    'embeddings': embed_1,
                    'labels': labs,
                },
                'softmax': {
                    'logits': logits,
                    'labels': labs,
                },
                # --- [INNOVATION 2] Causal nuisance intervention ---
                'cd': {
                    'embeddings': z_id,
                    'nuisance_embeddings': z_nuisance_text,
                    'labels': labs,
                },
                'motion_align': {
                    'kin_embeddings': z_kin,
                    'motion_embeddings': z_motion_text,
                },
                'counterfactual': {
                    'embeddings': z_id,
                    'labels': labs,
                    'nuisance_embeddings': z_nuisance_text,
                },
                # --- [INNOVATION 1] Phase-canonical loss ---
                'phase': {
                    'phase_code': phase_out['phase_code'],
                    'part_mu': pq['mu'],
                    'cycle_tokens': cycle_tokens,
                },
                # --- [INNOVATION 3] Uncertainty distillation ---
                'distill_unc': {
                    'student_embeddings': z_kin,
                    'teacher_stack': teacher_stack.detach(),
                    'log_var_stack': log_var_stack.detach(),
                    'valid_stack': valid_stack.detach(),
                },
                # --- [INNOVATION 4] Silhouette losses ---
                'sil_fg': {
                    'attn_map': pq['attn_map'],
                    'sil_mask': sil_mask,
                },
                'sil_bg_adv': {
                    'bg_embed': bg_embed,
                    'labels': labs,
                },
                # Pose alignment (pose teacher ↔ kinematic)
                'pose_align': {
                    'kin_embeddings': z_kin,
                    'pose_embeddings': z_pose,
                },
            },
            'visual_summary': {
                'image/rgb': (
                    rgb[:, :min(4, rgb.size(1))].flatten(0, 1)
                    if rgb.dim() == 5
                    else rgb[:, :min(4, rgb.size(1))].unsqueeze(2)
                         .flatten(0, 1)),
                'scalar/gate_semantic': gate[:, 0].mean().detach(),
                'scalar/gate_kinematic': gate[:, 1].mean().detach(),
                'scalar/gate_privileged': gate[:, 2].mean().detach(),
            },
            'inference_feat': {
                'embeddings': embed_1,
                'kin_embeddings': z_kin,
                'phase_code': phase_out['phase_code'].mean(1),
                'cycle_tokens': cycle_tokens.mean(1).flatten(1),
            },
        }

        if self._n_gate == 4:
            retval['visual_summary']['scalar/gate_pose'] = (
                gate[:, 3].mean().detach())

        return retval
