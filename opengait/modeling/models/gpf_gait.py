# -*- coding: utf-8 -*-
"""
Primitive-aware GPFGait for OpenGait.

Drop this file into:
    OpenGait/opengait/modeling/models/gpf_gait.py

Then add to OpenGait/opengait/modeling/models/__init__.py:
    from .gpf_gait import GPFGait

Expected sequence files produced by pretreatment_gpf.py:
    000-gpf.pkl   : [T, 12, H, W]
    001-sils.pkl  : [T, 1,  H, W] or [T, H, W]
    002-phase.pkl : [T, 2]
    003-parts.pkl : optional; not consumed by this wrapper yet

Default GPF channel order:
    00 silhouette mask
    01 body edge
    02 contour normal x
    03 contour normal y
    04 signed distance field
    05 motion amplitude
    06 phase sin
    07 phase cos
    08 optical-flow dx
    09 optical-flow dy
    10 motion reliability
    11 mid-frequency residual

Important version notes:
    1) The stable default variant is "gpf_structured": geometry + motion are
       encoded separately, reliability is used as a gate, residual and phase are
       disabled by default.
    2) Phase is only used when use_phase_film=True or temporal_pool is a phase
       pooling mode. This avoids injecting sequence-local absolute phase twice.
    3) TemporalPrimitivePool now correctly handles OpenGait seqL, [N,T,2] phase,
       and [N,T] / [N,1,T] reliability signals.
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


def _as_list(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _to_ncthw(x: torch.Tensor, expected_channels: Optional[int] = None) -> torch.Tensor:
    """Convert common OpenGait tensor layouts to [N, C, T, H, W]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.dim() == 4:
        # [N, T, H, W] -> [N, 1, T, H, W]
        return x.unsqueeze(1).float().contiguous()

    if x.dim() != 5:
        raise ValueError(f"Expected a 4D or 5D input tensor, got shape {tuple(x.shape)}")

    # Already [N, C, T, H, W].
    if expected_channels is not None and x.size(1) == expected_channels:
        return x.float().contiguous()

    # Usually OpenGait collates [T, C, H, W] as [N, T, C, H, W].
    if expected_channels is not None and x.size(2) == expected_channels:
        return x.transpose(1, 2).float().contiguous()

    # Heuristic fallback.
    if x.size(1) <= 16 and x.size(2) > 16:
        return x.float().contiguous()
    return x.transpose(1, 2).float().contiguous()


def _to_nt2(x: torch.Tensor) -> torch.Tensor:
    """Convert phase tensor to [N, T, 2]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected phase tensor with 3 dims, got shape {tuple(x.shape)}")
    if x.size(-1) == 2:
        return x.float().contiguous()
    if x.size(1) == 2:
        return x.transpose(1, 2).float().contiguous()
    raise ValueError(f"Cannot infer phase layout from shape {tuple(x.shape)}")


def _select_channels(x: torch.Tensor, indices: Sequence[int], name: str = "channels") -> torch.Tensor:
    if indices is None:
        return x
    if len(indices) == 0:
        raise ValueError(f"{name} is empty. Please check model_cfg channel settings.")
    max_idx = max(int(i) for i in indices)
    min_idx = min(int(i) for i in indices)
    if min_idx < 0 or max_idx >= x.size(1):
        raise ValueError(
            f"{name} contains invalid index. Got {indices}, but input has {x.size(1)} channels."
        )
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(1, idx)


def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a 2D module to a [N, C, T, H, W] tensor frame by frame."""
    n, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
    y = module(y)
    _, c2, h2, w2 = y.shape
    return y.reshape(n, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


def _match_nt_time(x: Optional[torch.Tensor], target_t: int, pad_value: str = "replicate"):
    """Match [N,T,...] temporal signal length to target_t by crop/pad."""
    if x is None:
        return None
    if x.size(1) == target_t:
        return x
    if x.size(1) > target_t:
        return x[:, :target_t].contiguous()
    pad_len = target_t - x.size(1)
    if pad_value == "zero":
        pad = torch.zeros(x.size(0), pad_len, *x.shape[2:], device=x.device, dtype=x.dtype)
    else:
        pad = x[:, -1:].expand(x.size(0), pad_len, *x.shape[2:]).contiguous()
    return torch.cat([x, pad], dim=1).contiguous()


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BasicFrameBlock(nn.Module):
    """Small residual 2D block used frame-wise."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act=False)
        if in_channels != out_channels or stride != 1:
            self.down = ConvBNAct(in_channels, out_channels, 1, stride, padding=0, act=False)
        else:
            self.down = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.down(x))


class FrameBackbone(nn.Module):
    """GaitBase-style frame encoder; input/output are [N, C, T, H, W]."""
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int] = (64, 128, 256, 512),
        strides: Sequence[int] = (1, 2, 1, 1),
        blocks_per_stage: Sequence[int] = (1, 1, 1, 1),
    ):
        super().__init__()
        channels = list(channels)
        strides = list(strides)
        blocks_per_stage = list(blocks_per_stage)
        assert len(channels) == len(strides) == len(blocks_per_stage)

        layers = []
        c_in = in_channels
        for c_out, stride, num_blocks in zip(channels, strides, blocks_per_stage):
            stage = [BasicFrameBlock(c_in, c_out, stride=stride)]
            for _ in range(max(0, int(num_blocks) - 1)):
                stage.append(BasicFrameBlock(c_out, c_out, stride=1))
            layers.append(nn.Sequential(*stage))
            c_in = c_out
        self.layers = nn.ModuleList(layers)
        self.out_channels = c_in

    def forward(self, x):
        for layer in self.layers:
            x = _apply_framewise(layer, x)
        return x


class GPFChannelNorm(nn.Module):
    """Conservative per-channel sanitization for the 12-channel GPF."""
    def __init__(self, enabled=True, sdf_clip=1.0, flow_clip=1.0, residual_clip=1.0):
        super().__init__()
        self.enabled = enabled
        self.sdf_clip = float(sdf_clip)
        self.flow_clip = float(flow_clip)
        self.residual_clip = float(residual_clip)

    def forward(self, x):
        x = x.float()
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if not self.enabled:
            return x

        y = x.clone()
        c = y.size(1)
        if c > 0:
            y[:, 0] = y[:, 0].clamp(0.0, 1.0)
        if c > 1:
            y[:, 1] = y[:, 1].clamp(0.0, 1.0)
        if c > 2:
            y[:, 2] = y[:, 2].clamp(-1.0, 1.0)
        if c > 3:
            y[:, 3] = y[:, 3].clamp(-1.0, 1.0)
        if c > 4:
            y[:, 4] = y[:, 4].clamp(-self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
        if c > 5:
            y[:, 5] = y[:, 5].clamp(0.0, 1.0)
        if c > 6:
            y[:, 6] = y[:, 6].clamp(-1.0, 1.0)
        if c > 7:
            y[:, 7] = y[:, 7].clamp(-1.0, 1.0)
        if c > 8:
            y[:, 8] = y[:, 8].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
        if c > 9:
            y[:, 9] = y[:, 9].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
        if c > 10:
            y[:, 10] = y[:, 10].clamp(0.0, 1.0)
        if c > 11:
            y[:, 11] = y[:, 11].clamp(-self.residual_clip, self.residual_clip) / max(self.residual_clip, 1e-6)
        return y


class PhaseFiLM(nn.Module):
    """Frame-wise FiLM conditioning from phase [sin(phi), cos(phi)]."""
    def __init__(self, channels: int, hidden_channels: int = 32, strength: float = 0.5):
        super().__init__()
        self.strength = float(strength)
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels * 2),
        )
        # Identity initialization: enabling the module should not destroy early training.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]):
        if phase is None:
            return x
        phase = _match_nt_time(phase, x.size(2))
        gb = self.mlp(phase)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.strength * gamma) + self.strength * beta


class DirectGPFStem(nn.Module):
    """Adapter for scalar/direct ablations."""
    def __init__(self, in_channels: int, out_channels: int, scalar: bool = False):
        super().__init__()
        if scalar:
            self.net = nn.Sequential(
                ConvBNAct(in_channels, 16, 3),
                nn.Conv2d(16, 1, 1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                ConvBNAct(1, out_channels, 3),
            )
        else:
            self.net = nn.Sequential(
                ConvBNAct(in_channels, out_channels, 3),
                ConvBNAct(out_channels, out_channels, 3),
            )

    def forward(self, x):
        return _apply_framewise(self.net, x), {}


class TypedGPFStem(nn.Module):
    """
    Primitive-aware adapter.

    geometry:    mask / edge / normals / SDF
    motion:      motion amplitude + flow dx/dy
    reliability: explicit soft gate, not treated as a normal feature
    residual:    optional branch; disabled by default for the stable run
    """
    def __init__(
        self,
        geo_channels: int,
        motion_channels: int,
        residual_channels: int,
        out_channels: int,
        branch_channels: int = 24,
        residual_branch_channels: int = 12,
        use_reliability_gate: bool = True,
        gate_strength: float = 0.75,
        primitive_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_reliability_gate = bool(use_reliability_gate)
        self.gate_strength = float(gate_strength)
        self.primitive_dropout = float(primitive_dropout)

        if geo_channels <= 0 or motion_channels <= 0:
            raise ValueError(f"geo_channels and motion_channels must be > 0, got {geo_channels}, {motion_channels}")

        self.geo_stem = nn.Sequential(
            ConvBNAct(geo_channels, branch_channels, 3),
            ConvBNAct(branch_channels, branch_channels, 3),
        )
        self.motion_stem = nn.Sequential(
            ConvBNAct(motion_channels, branch_channels, 3),
            ConvBNAct(branch_channels, branch_channels, 3),
        )

        self.has_residual = residual_channels > 0
        if self.has_residual:
            self.residual_stem = nn.Sequential(
                ConvBNAct(residual_channels, residual_branch_channels, 3),
                ConvBNAct(residual_branch_channels, residual_branch_channels, 3),
            )
        else:
            self.residual_stem = None

        self.motion_gate = nn.Sequential(
            ConvBNAct(1, branch_channels, 3),
            nn.Conv2d(branch_channels, branch_channels, 1),
        )
        if self.has_residual:
            self.residual_gate = nn.Sequential(
                ConvBNAct(1, residual_branch_channels, 3),
                nn.Conv2d(residual_branch_channels, residual_branch_channels, 1),
            )
        else:
            self.residual_gate = None

        fuse_in = branch_channels * 2 + (residual_branch_channels if self.has_residual else 0)
        self.fuse = nn.Sequential(
            ConvBNAct(fuse_in, out_channels, 1, padding=0),
            ConvBNAct(out_channels, out_channels, 3),
        )

    def _soft_gate(self, logits):
        gate = torch.sigmoid(logits)
        return (1.0 - self.gate_strength) + self.gate_strength * gate

    def _maybe_drop(self, x):
        if self.training and self.primitive_dropout > 0:
            if torch.rand((), device=x.device) < self.primitive_dropout:
                return x * 0.0
        return x

    def forward(
        self,
        geo: torch.Tensor,
        motion: torch.Tensor,
        reliability: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        foreground: Optional[torch.Tensor] = None,
    ):
        aux = {}
        geo_feat = _apply_framewise(self.geo_stem, geo)
        motion_feat = _apply_framewise(self.motion_stem, motion)

        if self.use_reliability_gate and reliability is not None:
            motion_gate = _apply_framewise(self.motion_gate, reliability)
            motion_gate = self._soft_gate(motion_gate)
            if foreground is not None:
                fg = foreground.clamp(0, 1)
                motion_gate = motion_gate * (0.5 + 0.5 * fg)
            motion_feat = motion_feat * motion_gate
            aux["motion_gate"] = motion_gate

        feats = [geo_feat, self._maybe_drop(motion_feat)]

        if self.has_residual and residual is not None:
            res_feat = _apply_framewise(self.residual_stem, residual)
            if self.use_reliability_gate and reliability is not None:
                res_gate = _apply_framewise(self.residual_gate, reliability)
                res_gate = self._soft_gate(res_gate)
                if foreground is not None:
                    fg = foreground.clamp(0, 1)
                    res_gate = res_gate * (0.5 + 0.5 * fg)
                res_feat = res_feat * res_gate
                aux["residual_gate"] = res_gate
            feats.append(self._maybe_drop(res_feat))

        fused = torch.cat(feats, dim=1)
        fused = _apply_framewise(self.fuse, fused)
        return fused, aux


class TemporalPrimitivePool(nn.Module):
    """
    Temporal pooling with optional reliability and phase-bin support.

    Important OpenGait detail:
        With unfixed_ordered / all_ordered sampling, OpenGait may pack a local
        batch of variable-length sequences as x = [1, C, sum(seqL), H, W]
        while labels still has local batch size B. In that case we must split
        along the temporal dimension according to seqL, pool each sequence, and
        concatenate back to [B, C, H, W].
    """
    def __init__(
        self,
        pool_type: str = "max",
        phase_bins: int = 8,
        phase_kappa: float = 4.0,
        phase_reduce: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.pool_type = str(pool_type)
        self.phase_bins = int(phase_bins)
        self.phase_kappa = float(phase_kappa)
        self.phase_reduce = phase_reduce
        self.eps = eps

    def _flatten_seqL(self, seqL):
        """Flatten OpenGait seqL containers into a plain Python list."""
        if seqL is None:
            return []
        if torch.is_tensor(seqL):
            return seqL.detach().cpu().reshape(-1).tolist()
        if isinstance(seqL, np.ndarray):
            return seqL.reshape(-1).tolist()
        if isinstance(seqL, (list, tuple)):
            out = []
            for v in seqL:
                out.extend(self._flatten_seqL(v))
            return out
        return [seqL]

    def _seqL_to_int_list(self, seqL, batch_size, max_len):
        vals = self._flatten_seqL(seqL)

        if len(vals) == 0:
            return [max_len for _ in range(batch_size)]

        lens = []
        for v in vals:
            if torch.is_tensor(v):
                v = v.detach().cpu().reshape(-1)[0].item()
            elif isinstance(v, np.ndarray):
                v = v.reshape(-1)[0].item()
            elif isinstance(v, (list, tuple)):
                if len(v) == 0:
                    continue
                v = v[0]
            lens.append(int(v))

        if len(lens) == 0:
            return [max_len for _ in range(batch_size)]

        if len(lens) == 1 and batch_size == 1:
            return [max(1, min(lens[0], max_len))]

        # OpenGait may provide seqL for multiple input modalities. The first B
        # entries correspond to the first input stream used here.
        if len(lens) >= batch_size:
            lens = lens[:batch_size]
            return [max(1, min(int(v), max_len)) for v in lens]

        # Defensive fallback: preserve batch size instead of shrinking embeddings.
        base = max(1, max_len // batch_size)
        fallback = [base for _ in range(batch_size)]
        fallback[-1] = max(1, max_len - base * (batch_size - 1))
        return fallback

    @staticmethod
    def _phase_to_nt2(phase: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize phase to [N, T, 2]."""
        if phase is None:
            return None
        if not torch.is_tensor(phase):
            phase = torch.as_tensor(phase)
        if phase.dim() == 2:
            if phase.size(-1) == 2:
                return phase.unsqueeze(0).float().contiguous()
            raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")
        if phase.dim() != 3:
            raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")
        if phase.size(-1) == 2:
            return phase.float().contiguous()
        if phase.size(1) == 2:
            return phase.transpose(1, 2).float().contiguous()
        raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")

    @staticmethod
    def _reliability_to_nt(reliability: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize reliability to [N, T]."""
        if reliability is None:
            return None
        if not torch.is_tensor(reliability):
            reliability = torch.as_tensor(reliability)
        if reliability.dim() == 2:
            return reliability.float().contiguous()
        if reliability.dim() == 3:
            if reliability.size(1) == 1:
                return reliability[:, 0].float().contiguous()
            if reliability.size(-1) == 1:
                return reliability[..., 0].float().contiguous()
        if reliability.dim() == 5:
            return reliability.float().mean(dim=(1, 3, 4)).contiguous()
        raise ValueError(f"Unsupported reliability shape: {tuple(reliability.shape)}")

    def _match_aux_for_time(self, phase, reliability, target_t):
        phase = self._phase_to_nt2(phase)
        if phase is not None:
            phase = _match_nt_time(phase, target_t)

        reliability = self._reliability_to_nt(reliability)
        if reliability is not None:
            reliability = _match_nt_time(reliability.unsqueeze(-1), target_t).squeeze(-1)

        return phase, reliability

    def _split_by_seql(self, x, phase=None, reliability=None, seqL=None, batch_size=None):
        """
        Return per-sequence chunks.

        Supports:
            x = [B, C, T, H, W]
            x = [1, C, sum(seqL), H, W]
        """
        n, c, max_len, h, w = x.shape
        if batch_size is None:
            batch_size = n
        batch_size = int(batch_size)

        phase, reliability = self._match_aux_for_time(phase, reliability, max_len)

        # Normal fixed-length batch.
        if n == batch_size:
            lens = self._seqL_to_int_list(seqL, batch_size=batch_size, max_len=max_len)
            chunks = []
            for i, l in enumerate(lens):
                xi = x[i:i + 1, :, :l]
                if phase is not None:
                    pi = phase[i:i + 1, :l] if phase.size(0) == n else phase[:1, :l]
                else:
                    pi = None
                if reliability is not None:
                    ri = reliability[i:i + 1, :l] if reliability.size(0) == n else reliability[:1, :l]
                else:
                    ri = None
                chunks.append((xi, pi, ri))
            return chunks

        # OpenGait packed variable-length batch.
        if n == 1 and batch_size > 1:
            lens = self._seqL_to_int_list(seqL, batch_size=batch_size, max_len=max_len)
            chunks = []
            start = 0

            for i, l in enumerate(lens):
                end = min(start + int(l), max_len)
                if end <= start:
                    end = min(start + 1, max_len)

                xi = x[:, :, start:end]

                if phase is None:
                    pi = None
                elif phase.size(0) == 1:
                    pi = phase[:, start:end]
                else:
                    pi = phase[i:i + 1, :end - start]

                if reliability is None:
                    ri = None
                elif reliability.size(0) == 1:
                    ri = reliability[:, start:end]
                else:
                    ri = reliability[i:i + 1, :end - start]

                chunks.append((xi, pi, ri))
                start = end

            return chunks

        raise RuntimeError(
            f"Cannot align temporal batch: x.shape={tuple(x.shape)}, "
            f"labels batch_size={batch_size}, seqL={seqL}"
        )

    def _pool_single(self, x, phase=None, reliability=None):
        if self.pool_type == "avg":
            return x.mean(dim=2)

        if self.pool_type == "max" or x.size(2) == 1:
            return x.max(dim=2)[0]

        if self.pool_type == "reliability":
            if reliability is None:
                return x.max(dim=2)[0]
            w = reliability.clamp(0, 1).view(reliability.size(0), 1, reliability.size(1), 1, 1)
            return (x * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

        if self.pool_type in ("phase_bin", "reliability_phase_bin"):
            if phase is None:
                return x.max(dim=2)[0]

            angle = torch.atan2(phase[..., 0], phase[..., 1])
            centers = torch.linspace(
                -math.pi,
                math.pi,
                steps=self.phase_bins + 1,
                device=x.device,
                dtype=x.dtype,
            )[:-1]
            weights = torch.exp(self.phase_kappa * torch.cos(angle.unsqueeze(-1) - centers))

            if self.pool_type == "reliability_phase_bin" and reliability is not None:
                weights = weights * reliability.clamp(0, 1).unsqueeze(-1)

            w = weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            xb = x.unsqueeze(3)
            z = (xb * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

            if self.phase_reduce == "max":
                return z.max(dim=2)[0]
            if self.phase_reduce == "sum":
                return z.sum(dim=2)
            return z.mean(dim=2)

        return x.max(dim=2)[0]

    def forward(self, x, seqL=None, phase=None, reliability=None, batch_size=None):
        if x.dim() != 5:
            raise RuntimeError(f"TemporalPool expects [N,C,T,H,W], but got {tuple(x.shape)}")

        if batch_size is None:
            batch_size = x.size(0)
        batch_size = int(batch_size)

        # Fast path: fixed-length normal batch.
        if x.size(0) == batch_size and self.pool_type == "max":
            return x.max(dim=2)[0]

        # Packed or advanced temporal pooling path.
        chunks = self._split_by_seql(
            x,
            phase=phase,
            reliability=reliability,
            seqL=seqL,
            batch_size=batch_size,
        )
        outs = [self._pool_single(xi, pi, ri) for xi, pi, ri in chunks]
        out = torch.cat(outs, dim=0)

        if out.size(0) != batch_size:
            raise RuntimeError(
                f"TemporalPool output batch mismatch: out.shape={tuple(out.shape)}, "
                f"expected batch_size={batch_size}, input x.shape={tuple(x.shape)}, seqL={seqL}"
            )

        return out



def _visual_time_indices(t: int, max_frames: int, device=None):
    """Evenly sample temporal indices for lightweight visualization."""
    if max_frames is None or int(max_frames) <= 0 or t <= int(max_frames):
        return torch.arange(t, device=device)
    return torch.linspace(0, t - 1, steps=int(max_frames), device=device).round().long()


def _visual_flatten_nt(x: torch.Tensor, max_frames: int = 8, normalize: bool = True) -> torch.Tensor:
    """Convert [N,C,T,H,W] to [N*T,C,H,W] for OpenGait visual_summary."""
    if x is None:
        return None
    if x.dim() != 5:
        raise ValueError(f"Expected [N,C,T,H,W] visual tensor, got {tuple(x.shape)}")
    x = x.detach().float()
    idx = _visual_time_indices(x.size(2), max_frames, device=x.device)
    x = x.index_select(2, idx)
    n, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
    if normalize:
        x = _visual_minmax(x)
    return x


def _visual_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min-max normalization for heatmaps or primitive maps."""
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    dims = tuple(range(2, x.dim()))
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _visual_energy(x: torch.Tensor, out_hw=None) -> torch.Tensor:
    """Channel-average activation energy as a weak attention proxy."""
    e = x.detach().float().abs().mean(dim=1, keepdim=True)
    if out_hw is not None and e.shape[-2:] != tuple(out_hw):
        n, c, t, h, w = e.shape
        e2 = e.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
        e2 = F.interpolate(e2, size=tuple(out_hw), mode="bilinear", align_corners=False)
        e = e2.reshape(n, t, c, *out_hw).permute(0, 2, 1, 3, 4).contiguous()
    return e


def _limb_prior_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Build a coarse limb prior from canonical gait silhouettes.

    It is intentionally heuristic and only for diagnostics:
      - lower-body foreground is treated as leg-dominant;
      - upper/mid side foreground is treated as arm-dominant;
      - central upper torso is down-weighted.
    """
    if mask is None:
        return None
    fg = mask.detach().float().clamp(0, 1)
    n, c, t, h, w = fg.shape
    yy = torch.linspace(0, 1, h, device=fg.device, dtype=fg.dtype).view(1, 1, 1, h, 1)
    xx = torch.linspace(-1, 1, w, device=fg.device, dtype=fg.dtype).view(1, 1, 1, 1, w)

    leg = torch.sigmoid((yy - 0.55) * 24.0)
    arm_vertical = torch.sigmoid((yy - 0.25) * 24.0) * torch.sigmoid((0.72 - yy) * 24.0)
    arm_side = torch.sigmoid((xx.abs() - 0.42) * 16.0)
    arm = arm_vertical * arm_side
    prior = torch.maximum(leg, arm).expand_as(fg) * fg
    return prior.clamp(0, 1)



class ViewPromptPool2D(nn.Module):
    """
    Lightweight supervised view prompt pool for GPFGait.

    It is intentionally conservative:
      - no view token is concatenated into the identity embedding;
      - prompt query uses x.detach() by default, so view CE / prompt regularizers
        do not push the CNN identity backbone to encode view labels;
      - modulation is a small residual FiLM correction;
      - modulation head is zero-initialized, so the block starts as identity.
    """
    def __init__(
        self,
        channels: int,
        num_prompts: int = 8,
        prompt_dim: int = 64,
        hidden_channels: int = 128,
        strength: float = 0.10,
        temperature: float = 0.07,
        detach_query: bool = True,
        dropout: float = 0.0,
        num_view_classes: int = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_prompts = int(num_prompts)
        self.prompt_dim = int(prompt_dim)
        self.strength = float(strength)
        self.temperature = max(float(temperature), 1e-4)
        self.detach_query = bool(detach_query)
        self.eps = float(eps)

        if self.num_prompts <= 1:
            raise ValueError(f"num_prompts must be > 1, got {self.num_prompts}")

        self.query = nn.Sequential(
            nn.Linear(self.channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_channels, self.prompt_dim),
        )

        self.prompts = nn.Parameter(torch.empty(self.num_prompts, self.prompt_dim))
        nn.init.normal_(self.prompts, std=0.02)

        self.modulator = nn.Sequential(
            nn.Linear(self.prompt_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.channels * 2),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

        if int(num_view_classes) > 0:
            self.view_classifier = nn.Linear(self.prompt_dim, int(num_view_classes))
        else:
            self.view_classifier = None

    def _balance_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        KL(mean_alpha || uniform). Keeps prompt usage from collapsing to one token.
        alpha: [B, K]
        """
        mean_alpha = alpha.mean(dim=0).clamp_min(self.eps)
        uniform = torch.full_like(mean_alpha, 1.0 / float(self.num_prompts))
        return torch.sum(mean_alpha * (torch.log(mean_alpha) - torch.log(uniform)))

    def _orthogonality_loss(self) -> torch.Tensor:
        """
        Mean squared off-diagonal cosine similarity of prompt vectors.
        Keeps different prompts from becoming identical.
        """
        p = F.normalize(self.prompts, dim=-1)
        gram = torch.matmul(p, p.t())
        eye = torch.eye(self.num_prompts, device=gram.device, dtype=gram.dtype)
        off_diag = gram - eye
        denom = max(self.num_prompts * (self.num_prompts - 1), 1)
        return off_diag.pow(2).sum() / float(denom)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W], usually output of TemporalPrimitivePool.

        Returns:
            y: [B, C, H, W], view-prompt residual-modulated feature.
            aux: dict with prompt logits/weights, optional view logits, and regularizers.
        """
        if x.dim() != 4:
            raise RuntimeError(f"ViewPromptPool2D expects [B,C,H,W], got {tuple(x.shape)}")

        # View CE / balance gradients stop here by default. This keeps view
        # supervision from shaping the CNN identity backbone directly.
        q_src = x.detach() if self.detach_query else x
        pooled = q_src.mean(dim=(-1, -2))  # [B, C]
        q = self.query(pooled)             # [B, D]

        qn = F.normalize(q, dim=-1)
        pn = F.normalize(self.prompts, dim=-1)
        prompt_logits = torch.matmul(qn, pn.t()) / self.temperature
        alpha = F.softmax(prompt_logits, dim=-1)
        p_view = torch.matmul(alpha, self.prompts)

        gamma_beta = self.modulator(p_view)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        # Small residual correction only. No token is appended to identity emb.
        y = x + self.strength * (x * gamma + beta)

        aux = {
            "view_prompt_alpha": alpha,
            "view_prompt_logits": prompt_logits,
            "view_prompt_balance_loss": self._balance_loss(alpha),
            "view_prompt_orthogonality_loss": self._orthogonality_loss(),
        }

        if self.view_classifier is not None:
            aux["view_logits"] = self.view_classifier(p_view)

        return y, aux


class GPFGait(BaseModel):
    """OpenGait model wrapper for GPF ablations and the stable structured model."""

    def build_network(self, model_cfg: Dict):
        self.model_cfg = model_cfg
        self.variant = model_cfg.get("variant", "gpf_structured")
        self.input_index = int(model_cfg.get("input_index", 0))
        self.phase_input_index = model_cfg.get("phase_input_index", None)
        self.source_channels = int(model_cfg.get("source_channels", 12))
        self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))

        # Visualization is off by default to keep training/evaluation fast and logs small.
        # Enable with model_cfg.visualize: True. Returned tensors are placed in
        # output["visual_summary"] and are compatible with OpenGait image logging.
        self.visualize = bool(model_cfg.get("visualize", model_cfg.get("enable_visualization", False)))
        self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))
        self.visualize_inputs = bool(model_cfg.get("visualize_inputs", True))
        self.visualize_gates = bool(model_cfg.get("visualize_gates", True))
        self.visualize_features = bool(model_cfg.get("visualize_features", True))
        self.visualize_limb_focus = bool(model_cfg.get("visualize_limb_focus", True))

        self.norm = GPFChannelNorm(
            enabled=bool(model_cfg.get("enable_channel_norm", True)),
            sdf_clip=float(model_cfg.get("sdf_clip", 1.0)),
            flow_clip=float(model_cfg.get("flow_clip", 1.0)),
            residual_clip=float(model_cfg.get("residual_clip", 1.0)),
        )

        stem_out_channels = int(model_cfg.get("stem_out_channels", model_cfg.get("grouped_fuse_channels", 32)))

        # Optional global channel filter for grouped/structured variants.
        allowed_channels = model_cfg.get("channel_indices", None)
        self.allowed_channels = set(map(int, allowed_channels)) if allowed_channels is not None else None

        def _filter_allowed(indices):
            indices = list(map(int, indices))
            if self.allowed_channels is None:
                return indices
            return [i for i in indices if i in self.allowed_channels]

        self.geo_indices = _filter_allowed(_as_list(model_cfg.get("geo_indices", [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4]))
        self.motion_indices = _filter_allowed(_as_list(model_cfg.get("motion_indices", [5, 8, 9]), [5, 8, 9]))

        reliability_index = model_cfg.get("reliability_index", 10)
        if reliability_index is not None:
            reliability_index = int(reliability_index)
            if self.allowed_channels is not None and reliability_index not in self.allowed_channels:
                reliability_index = None
        self.reliability_index = reliability_index

        # Stable default: no residual branch. Enable explicitly with residual_indices: [11].
        self.residual_indices = _filter_allowed(_as_list(model_cfg.get("residual_indices", []), []))
        self.phase_indices = _filter_allowed(_as_list(model_cfg.get("phase_indices", [6, 7]), [6, 7]))

        if self.variant == "sil":
            self.adapter_type = "sil"
            self.adapter = DirectGPFStem(1, stem_out_channels, scalar=False)

        elif self.variant in ("gpf_geometry_scalar", "gpf_motion_scalar", "gpf_all_scalar"):
            if self.variant == "gpf_geometry_scalar":
                self.channel_indices = self.geo_indices if len(self.geo_indices) > 0 else [0, 1, 2, 3, 4]
            elif self.variant == "gpf_motion_scalar":
                if "channel_indices" in model_cfg:
                    self.channel_indices = _as_list(model_cfg.get("channel_indices"), None)
                else:
                    # Keep the original ablation definition: motion + reliability + residual.
                    self.channel_indices = [5, 8, 9, 10, 11]
            else:
                self.channel_indices = _as_list(
                    model_cfg.get("channel_indices", list(range(self.source_channels))),
                    list(range(self.source_channels)),
                )
            self.channel_indices = list(map(int, self.channel_indices))
            self.adapter_type = "direct"
            self.adapter = DirectGPFStem(len(self.channel_indices), stem_out_channels, scalar=True)

        elif self.variant in ("gpf_all_direct", "gpf_direct"):
            self.channel_indices = _as_list(
                model_cfg.get("channel_indices", list(range(self.source_channels))),
                list(range(self.source_channels)),
            )
            self.channel_indices = list(map(int, self.channel_indices))
            self.adapter_type = "direct"
            self.adapter = DirectGPFStem(len(self.channel_indices), stem_out_channels, scalar=False)

        elif self.variant in ("gpf_grouped", "gpf_structured"):
            self.adapter_type = "typed"
            self.adapter = TypedGPFStem(
                geo_channels=len(self.geo_indices),
                motion_channels=len(self.motion_indices),
                residual_channels=len(self.residual_indices),
                out_channels=stem_out_channels,
                branch_channels=int(model_cfg.get("grouped_branch_channels", 24)),
                residual_branch_channels=int(model_cfg.get("residual_branch_channels", 12)),
                use_reliability_gate=bool(model_cfg.get("use_reliability_gate", True)),
                gate_strength=float(model_cfg.get("reliability_gate_strength", 0.75)),
                primitive_dropout=float(model_cfg.get("primitive_dropout", 0.0)),
            )
        else:
            raise ValueError(f"Unknown GPFGait variant: {self.variant}")

        self.use_phase_film = bool(model_cfg.get("use_phase_film", False))
        if self.use_phase_film:
            self.phase_film = PhaseFiLM(
                stem_out_channels,
                hidden_channels=int(model_cfg.get("phase_hidden_channels", 32)),
                strength=float(model_cfg.get("phase_film_strength", 0.5)),
            )
        else:
            self.phase_film = nn.Identity()

        self.backbone = FrameBackbone(
            stem_out_channels,
            channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
            strides=model_cfg.get("backbone_strides", [1, 2, 1, 1]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
        )

        bin_num = model_cfg.get("bin_num", [32])
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        parts_num = int(sum(bin_num))

        self.TP = TemporalPrimitivePool(
            pool_type=model_cfg.get("temporal_pool", "max"),
            phase_bins=int(model_cfg.get("phase_bins", 8)),
            phase_kappa=float(model_cfg.get("phase_kappa", 4.0)),
            phase_reduce=model_cfg.get("phase_reduce", "mean"),
        )

        self.use_view_prompt = bool(model_cfg.get("use_view_prompt", False))
        self.view_values = [str(v) for v in model_cfg.get("view_values", [])]
        self.view_to_label = {v: i for i, v in enumerate(self.view_values)}

        if self.use_view_prompt:
            configured_num_views = int(model_cfg.get("view_prompt_num_views", len(self.view_values)))
            if len(self.view_values) > 0 and configured_num_views != len(self.view_values):
                raise ValueError(
                    f"view_prompt_num_views={configured_num_views} but len(view_values)={len(self.view_values)}. "
                    "Please use the count_ccpg_views.py script to generate a consistent config snippet."
                )

            self.view_prompt = ViewPromptPool2D(
                channels=self.backbone.out_channels,
                num_prompts=int(model_cfg.get("view_prompt_num_prompts", 8)),
                prompt_dim=int(model_cfg.get("view_prompt_dim", 64)),
                hidden_channels=int(model_cfg.get("view_prompt_hidden_channels", 128)),
                strength=float(model_cfg.get("view_prompt_strength", 0.10)),
                temperature=float(model_cfg.get("view_prompt_temperature", 0.07)),
                detach_query=bool(model_cfg.get("view_prompt_detach_query", True)),
                dropout=float(model_cfg.get("view_prompt_dropout", 0.0)),
                num_view_classes=configured_num_views if len(self.view_values) > 0 else 0,
                eps=float(model_cfg.get("view_prompt_eps", 1e-6)),
            )
        else:
            self.view_prompt = None

        emb_channels = int(model_cfg.get("emb_channels", 256))
        class_num = int(model_cfg["class_num"])
        fc_cfg = model_cfg.get("SeparateFCs", {})
        bn_cfg = model_cfg.get("SeparateBNNecks", {})

        self.FCs = SeparateFCs(
            parts_num=fc_cfg.get("parts_num", parts_num),
            in_channels=fc_cfg.get("in_channels", self.backbone.out_channels),
            out_channels=fc_cfg.get("out_channels", emb_channels),
            norm=fc_cfg.get("norm", False),
        )
        self.BNNecks = SeparateBNNecks(
            parts_num=bn_cfg.get("parts_num", parts_num),
            in_channels=bn_cfg.get("in_channels", emb_channels),
            class_num=bn_cfg.get("class_num", class_num),
            norm=bn_cfg.get("norm", True),
        )

    def _get_phase(self, ipts, gpf: Optional[torch.Tensor]):
        # Prefer explicit 002-phase.pkl when configured.
        if self.phase_input_index is not None:
            idx = int(self.phase_input_index)
            if 0 <= idx < len(ipts):
                try:
                    phase = _to_nt2(ipts[idx])
                    if gpf is not None:
                        phase = phase.to(gpf.device)
                    return phase
                except Exception:
                    # Fallback to GPF phase channels below.
                    pass

        if gpf is None or len(self.phase_indices) < 2:
            return None

        i0, i1 = self.phase_indices[:2]
        if max(i0, i1) >= gpf.size(1):
            return None
        # GPF channels 6/7 are spatially constant phase maps. Average to [N,2,T].
        phase_map = gpf[:, [i0, i1]].mean(dim=(-1, -2))
        return phase_map.transpose(1, 2).contiguous()  # [N,T,2]

    def _get_reliability_scalar(self, gpf: Optional[torch.Tensor]):
        if gpf is None or self.reliability_index is None:
            return None
        idx = int(self.reliability_index)
        if idx >= gpf.size(1):
            return None
        return gpf[:, idx].mean(dim=(-1, -2)).clamp(0, 1)  # [N,T]

    def _build_visual_summary(
        self,
        visual_src: Optional[torch.Tensor],
        gpf: Optional[torch.Tensor],
        stem_feat: Optional[torch.Tensor],
        backbone_feat: Optional[torch.Tensor],
        aux: Optional[Dict[str, torch.Tensor]],
    ):
        """
        Build lightweight visual diagnostics.

        The maps are not a proof of causality by themselves. They are intended to answer:
        1) what primitive fields the network sees;
        2) whether the reliability gate suppresses unreliable motion;
        3) where the learned feature energy is spatially concentrated;
        4) whether that energy overlaps a coarse limb prior.
        """
        visual_summary = {}
        max_frames = self.visualize_max_frames

        if visual_src is not None:
            visual_summary["image/gpf_or_sil_mask"] = _visual_flatten_nt(
                visual_src.clamp(0, 1), max_frames=max_frames, normalize=False
            )

        if not self.visualize:
            return visual_summary

        base_hw = None
        if visual_src is not None:
            base_hw = visual_src.shape[-2:]
        elif gpf is not None:
            base_hw = gpf.shape[-2:]

        if self.visualize_inputs and gpf is not None:
            # Show physically meaningful GPF channels. Names follow the default channel order.
            channel_names = {
                0: "mask",
                1: "edge",
                4: "sdf",
                5: "motion_amp",
                10: "motion_reliability",
                11: "midfreq_residual_abs",
            }
            for idx, name in channel_names.items():
                if idx < gpf.size(1):
                    x = gpf[:, idx:idx + 1]
                    if idx == 11:
                        x = x.abs()
                    visual_summary[f"image/gpf_{idx:02d}_{name}"] = _visual_flatten_nt(
                        x, max_frames=max_frames, normalize=True
                    )

            if gpf.size(1) > 9:
                flow_mag = torch.sqrt(gpf[:, 8:9].pow(2) + gpf[:, 9:10].pow(2) + 1e-6)
                visual_summary["image/gpf_flow_magnitude"] = _visual_flatten_nt(
                    flow_mag, max_frames=max_frames, normalize=True
                )

        if self.visualize_gates and aux is not None:
            if "motion_gate" in aux:
                motion_gate = aux["motion_gate"].detach().float().mean(dim=1, keepdim=True)
                visual_summary["image/learned_motion_gate"] = _visual_flatten_nt(
                    motion_gate, max_frames=max_frames, normalize=True
                )
            if "residual_gate" in aux:
                residual_gate = aux["residual_gate"].detach().float().mean(dim=1, keepdim=True)
                visual_summary["image/learned_residual_gate"] = _visual_flatten_nt(
                    residual_gate, max_frames=max_frames, normalize=True
                )

        if self.visualize_features and stem_feat is not None:
            stem_energy = _visual_energy(stem_feat, out_hw=base_hw)
            if visual_src is not None and stem_energy.shape[-2:] == visual_src.shape[-2:]:
                stem_energy = stem_energy * (0.25 + 0.75 * visual_src.clamp(0, 1))
            visual_summary["image/stem_feature_energy"] = _visual_flatten_nt(
                stem_energy, max_frames=max_frames, normalize=True
            )

        if self.visualize_features and backbone_feat is not None:
            backbone_energy = _visual_energy(backbone_feat, out_hw=base_hw)
            if visual_src is not None and backbone_energy.shape[-2:] == visual_src.shape[-2:]:
                backbone_energy = backbone_energy * (0.25 + 0.75 * visual_src.clamp(0, 1))
            visual_summary["image/backbone_feature_energy"] = _visual_flatten_nt(
                backbone_energy, max_frames=max_frames, normalize=True
            )

            if self.visualize_limb_focus and visual_src is not None:
                limb_prior = _limb_prior_from_mask(visual_src)
                visual_summary["image/coarse_limb_prior"] = _visual_flatten_nt(
                    limb_prior, max_frames=max_frames, normalize=False
                )
                visual_summary["image/backbone_energy_on_limb_prior"] = _visual_flatten_nt(
                    backbone_energy * limb_prior, max_frames=max_frames, normalize=True
                )
                torso_prior = (visual_src.clamp(0, 1) - limb_prior).clamp(0, 1)
                visual_summary["image/backbone_energy_on_torso_prior"] = _visual_flatten_nt(
                    backbone_energy * torso_prior, max_frames=max_frames, normalize=True
                )

        return visual_summary


    def _get_view_labels(self, vies, device, expected_bs: Optional[int] = None):
        """Map OpenGait view names to contiguous class labels using model_cfg.view_values."""
        if not getattr(self, "view_to_label", None):
            return None

        def _flatten(v):
            if torch.is_tensor(v):
                return v.detach().cpu().reshape(-1).tolist()
            if isinstance(v, np.ndarray):
                return v.reshape(-1).tolist()
            if isinstance(v, (list, tuple)):
                out = []
                for item in v:
                    out.extend(_flatten(item))
                return out
            return [v]

        vals = _flatten(vies)
        if expected_bs is not None and len(vals) >= int(expected_bs):
            vals = vals[:int(expected_bs)]

        labels = []
        for v in vals:
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            key = str(v)

            # Support common numeric-string fallbacks, e.g. 3 -> "003";
            # CCPG-style strings such as "03_0" remain unchanged.
            if key not in self.view_to_label:
                try:
                    key3 = f"{int(float(key)):03d}"
                    key = key3 if key3 in self.view_to_label else key
                except Exception:
                    pass

            if key not in self.view_to_label:
                return None
            labels.append(self.view_to_label[key])

        if expected_bs is not None and len(labels) != int(expected_bs):
            return None

        return torch.as_tensor(labels, dtype=torch.long, device=device)

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        gpf = None
        phase = None
        reliability_scalar = None

        if self.adapter_type == "sil":
            x_in = _to_ncthw(ipts[self.input_index], expected_channels=1)
            x, aux = self.adapter(x_in)
            visual_src = x_in
        else:
            gpf = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
            gpf = self.norm(gpf)
            phase = self._get_phase(ipts, gpf)
            reliability_scalar = self._get_reliability_scalar(gpf)
            visual_src = gpf[:, 0:1] if gpf.size(1) > 0 else None

            if self.adapter_type == "direct":
                x_in = _select_channels(gpf, self.channel_indices, "channel_indices")
                x, aux = self.adapter(x_in)
            else:
                geo = _select_channels(gpf, self.geo_indices, "geo_indices")
                motion = _select_channels(gpf, self.motion_indices, "motion_indices")

                reliability = None
                if self.reliability_index is not None and int(self.reliability_index) < gpf.size(1):
                    reliability = gpf[:, int(self.reliability_index):int(self.reliability_index) + 1]

                residual = None
                if len(self.residual_indices) > 0:
                    residual = _select_channels(gpf, self.residual_indices, "residual_indices")

                foreground = gpf[:, 0:1] if gpf.size(1) > 0 else None
                x, aux = self.adapter(
                    geo,
                    motion,
                    reliability=reliability,
                    residual=residual,
                    foreground=foreground,
                )

        if self.use_phase_film:
            x = self.phase_film(x, phase)

        stem_feat = x
        x = self.backbone(x)
        backbone_feat = x
        # outs = self.TP(x, seqL=seqL, phase=phase, reliability=reliability_scalar)
        local_bs = labs.size(0) if torch.is_tensor(labs) else len(labs)

        outs = self.TP(
            x,
            seqL=seqL,
            phase=phase,
            reliability=reliability_scalar,
            batch_size=local_bs,
        )

        view_aux = {}
        if self.use_view_prompt and self.view_prompt is not None:
            outs, view_aux = self.view_prompt(outs)

        feat = self.HPP(outs)

        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_2 if self.inference_use_emb2 else embed_1

        visual_summary = self._build_visual_summary(
            visual_src=visual_src,
            gpf=gpf,
            stem_feat=stem_feat,
            backbone_feat=backbone_feat,
            aux=aux,
        )

        if embed_1.size(0) != local_bs:
            raise RuntimeError(
                f"Embedding batch mismatch: embed_1.shape={tuple(embed_1.shape)}, "
                f"labels.shape={tuple(labs.shape)}. "
                f"Expected embed_1.size(0) == labels.size(0)."
            )

        if logits.size(0) != local_bs:
            raise RuntimeError(
                f"Logits batch mismatch: logits.shape={tuple(logits.shape)}, "
                f"labels.shape={tuple(labs.shape)}."
            )


        training_feat = {
            "triplet": {"embeddings": embed_1, "labels": labs},
            "softmax": {"logits": logits, "labels": labs},
        }

        if self.use_view_prompt and self.view_prompt is not None and "view_logits" in view_aux:
            view_labels = self._get_view_labels(vies, embed_1.device, expected_bs=local_bs)
            if view_labels is not None and view_labels.numel() == local_bs:
                # Use [B, num_views, 1] to match OpenGait's part-wise CE convention.
                training_feat["view_softmax"] = {
                    "logits": view_aux["view_logits"].unsqueeze(-1),
                    "labels": view_labels,
                }
                training_feat["view_prompt_reg"] = {
                    "balance_loss": view_aux["view_prompt_balance_loss"],
                    "orthogonality_loss": view_aux["view_prompt_orthogonality_loss"],
                }

        return {
            "training_feat": training_feat,
            "visual_summary": visual_summary,
            "inference_feat": {
                "embeddings": embed,
            },
        }





# # -*- coding: utf-8 -*-
# """
# Primitive-aware GPFGait for OpenGait.

# Drop this file into:
#     OpenGait/opengait/modeling/models/gpf_gait.py

# Then add to OpenGait/opengait/modeling/models/__init__.py:
#     from .gpf_gait import GPFGait

# Expected sequence files produced by pretreatment_gpf.py:
#     000-gpf.pkl   : [T, 12, H, W]
#     001-sils.pkl  : [T, 1,  H, W] or [T, H, W]
#     002-phase.pkl : [T, 2]
#     003-parts.pkl : optional; not consumed by this wrapper yet

# Default GPF channel order:
#     00 silhouette mask
#     01 body edge
#     02 contour normal x
#     03 contour normal y
#     04 signed distance field
#     05 motion amplitude
#     06 phase sin
#     07 phase cos
#     08 optical-flow dx
#     09 optical-flow dy
#     10 motion reliability
#     11 mid-frequency residual

# Important version notes:
#     1) The stable default variant is "gpf_structured": geometry + motion are
#        encoded separately, reliability is used as a gate, residual and phase are
#        disabled by default.
#     2) Phase is only used when use_phase_film=True or temporal_pool is a phase
#        pooling mode. This avoids injecting sequence-local absolute phase twice.
#     3) TemporalPrimitivePool now correctly handles OpenGait seqL, [N,T,2] phase,
#        and [N,T] / [N,1,T] reliability signals.
# """

# import math
# from typing import Dict, Optional, Sequence

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# def _as_list(x, default=None):
#     if x is None:
#         return default
#     if isinstance(x, (list, tuple)):
#         return list(x)
#     return [x]


# def _to_ncthw(x: torch.Tensor, expected_channels: Optional[int] = None) -> torch.Tensor:
#     """Convert common OpenGait tensor layouts to [N, C, T, H, W]."""
#     if not torch.is_tensor(x):
#         x = torch.as_tensor(x)

#     if x.dim() == 4:
#         # [N, T, H, W] -> [N, 1, T, H, W]
#         return x.unsqueeze(1).float().contiguous()

#     if x.dim() != 5:
#         raise ValueError(f"Expected a 4D or 5D input tensor, got shape {tuple(x.shape)}")

#     # Already [N, C, T, H, W].
#     if expected_channels is not None and x.size(1) == expected_channels:
#         return x.float().contiguous()

#     # Usually OpenGait collates [T, C, H, W] as [N, T, C, H, W].
#     if expected_channels is not None and x.size(2) == expected_channels:
#         return x.transpose(1, 2).float().contiguous()

#     # Heuristic fallback.
#     if x.size(1) <= 16 and x.size(2) > 16:
#         return x.float().contiguous()
#     return x.transpose(1, 2).float().contiguous()


# def _to_nt2(x: torch.Tensor) -> torch.Tensor:
#     """Convert phase tensor to [N, T, 2]."""
#     if not torch.is_tensor(x):
#         x = torch.as_tensor(x)
#     if x.dim() == 2:
#         x = x.unsqueeze(0)
#     if x.dim() != 3:
#         raise ValueError(f"Expected phase tensor with 3 dims, got shape {tuple(x.shape)}")
#     if x.size(-1) == 2:
#         return x.float().contiguous()
#     if x.size(1) == 2:
#         return x.transpose(1, 2).float().contiguous()
#     raise ValueError(f"Cannot infer phase layout from shape {tuple(x.shape)}")


# def _select_channels(x: torch.Tensor, indices: Sequence[int], name: str = "channels") -> torch.Tensor:
#     if indices is None:
#         return x
#     if len(indices) == 0:
#         raise ValueError(f"{name} is empty. Please check model_cfg channel settings.")
#     max_idx = max(int(i) for i in indices)
#     min_idx = min(int(i) for i in indices)
#     if min_idx < 0 or max_idx >= x.size(1):
#         raise ValueError(
#             f"{name} contains invalid index. Got {indices}, but input has {x.size(1)} channels."
#         )
#     idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
#     return x.index_select(1, idx)


# def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
#     """Apply a 2D module to a [N, C, T, H, W] tensor frame by frame."""
#     n, c, t, h, w = x.shape
#     y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
#     y = module(y)
#     _, c2, h2, w2 = y.shape
#     return y.reshape(n, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


# def _match_nt_time(x: Optional[torch.Tensor], target_t: int, pad_value: str = "replicate"):
#     """Match [N,T,...] temporal signal length to target_t by crop/pad."""
#     if x is None:
#         return None
#     if x.size(1) == target_t:
#         return x
#     if x.size(1) > target_t:
#         return x[:, :target_t].contiguous()
#     pad_len = target_t - x.size(1)
#     if pad_value == "zero":
#         pad = torch.zeros(x.size(0), pad_len, *x.shape[2:], device=x.device, dtype=x.dtype)
#     else:
#         pad = x[:, -1:].expand(x.size(0), pad_len, *x.shape[2:]).contiguous()
#     return torch.cat([x, pad], dim=1).contiguous()


# class ConvBNAct(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, act=True):
#         super().__init__()
#         if padding is None:
#             padding = kernel_size // 2
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True) if act else nn.Identity()

#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))


# class BasicFrameBlock(nn.Module):
#     """Small residual 2D block used frame-wise."""
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
#         self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act=False)
#         if in_channels != out_channels or stride != 1:
#             self.down = ConvBNAct(in_channels, out_channels, 1, stride, padding=0, act=False)
#         else:
#             self.down = nn.Identity()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv2(self.conv1(x)) + self.down(x))


# class FrameBackbone(nn.Module):
#     """GaitBase-style frame encoder; input/output are [N, C, T, H, W]."""
#     def __init__(
#         self,
#         in_channels: int,
#         channels: Sequence[int] = (64, 128, 256, 512),
#         strides: Sequence[int] = (1, 2, 1, 1),
#         blocks_per_stage: Sequence[int] = (1, 1, 1, 1),
#     ):
#         super().__init__()
#         channels = list(channels)
#         strides = list(strides)
#         blocks_per_stage = list(blocks_per_stage)
#         assert len(channels) == len(strides) == len(blocks_per_stage)

#         layers = []
#         c_in = in_channels
#         for c_out, stride, num_blocks in zip(channels, strides, blocks_per_stage):
#             stage = [BasicFrameBlock(c_in, c_out, stride=stride)]
#             for _ in range(max(0, int(num_blocks) - 1)):
#                 stage.append(BasicFrameBlock(c_out, c_out, stride=1))
#             layers.append(nn.Sequential(*stage))
#             c_in = c_out
#         self.layers = nn.ModuleList(layers)
#         self.out_channels = c_in

#     def forward(self, x):
#         for layer in self.layers:
#             x = _apply_framewise(layer, x)
#         return x


# class GPFChannelNorm(nn.Module):
#     """Conservative per-channel sanitization for the 12-channel GPF."""
#     def __init__(self, enabled=True, sdf_clip=1.0, flow_clip=1.0, residual_clip=1.0):
#         super().__init__()
#         self.enabled = enabled
#         self.sdf_clip = float(sdf_clip)
#         self.flow_clip = float(flow_clip)
#         self.residual_clip = float(residual_clip)

#     def forward(self, x):
#         x = x.float()
#         x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
#         if not self.enabled:
#             return x

#         y = x.clone()
#         c = y.size(1)
#         if c > 0:
#             y[:, 0] = y[:, 0].clamp(0.0, 1.0)
#         if c > 1:
#             y[:, 1] = y[:, 1].clamp(0.0, 1.0)
#         if c > 2:
#             y[:, 2] = y[:, 2].clamp(-1.0, 1.0)
#         if c > 3:
#             y[:, 3] = y[:, 3].clamp(-1.0, 1.0)
#         if c > 4:
#             y[:, 4] = y[:, 4].clamp(-self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
#         if c > 5:
#             y[:, 5] = y[:, 5].clamp(0.0, 1.0)
#         if c > 6:
#             y[:, 6] = y[:, 6].clamp(-1.0, 1.0)
#         if c > 7:
#             y[:, 7] = y[:, 7].clamp(-1.0, 1.0)
#         if c > 8:
#             y[:, 8] = y[:, 8].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
#         if c > 9:
#             y[:, 9] = y[:, 9].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
#         if c > 10:
#             y[:, 10] = y[:, 10].clamp(0.0, 1.0)
#         if c > 11:
#             y[:, 11] = y[:, 11].clamp(-self.residual_clip, self.residual_clip) / max(self.residual_clip, 1e-6)
#         return y


# class PhaseFiLM(nn.Module):
#     """Frame-wise FiLM conditioning from phase [sin(phi), cos(phi)]."""
#     def __init__(self, channels: int, hidden_channels: int = 32, strength: float = 0.5):
#         super().__init__()
#         self.strength = float(strength)
#         self.mlp = nn.Sequential(
#             nn.Linear(2, hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_channels, channels * 2),
#         )
#         # Identity initialization: enabling the module should not destroy early training.
#         nn.init.zeros_(self.mlp[-1].weight)
#         nn.init.zeros_(self.mlp[-1].bias)

#     def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]):
#         if phase is None:
#             return x
#         phase = _match_nt_time(phase, x.size(2))
#         gb = self.mlp(phase)
#         gamma, beta = gb.chunk(2, dim=-1)
#         gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
#         beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
#         return x * (1.0 + self.strength * gamma) + self.strength * beta


# class DirectGPFStem(nn.Module):
#     """Adapter for scalar/direct ablations."""
#     def __init__(self, in_channels: int, out_channels: int, scalar: bool = False):
#         super().__init__()
#         if scalar:
#             self.net = nn.Sequential(
#                 ConvBNAct(in_channels, 16, 3),
#                 nn.Conv2d(16, 1, 1, bias=False),
#                 nn.BatchNorm2d(1),
#                 nn.ReLU(inplace=True),
#                 ConvBNAct(1, out_channels, 3),
#             )
#         else:
#             self.net = nn.Sequential(
#                 ConvBNAct(in_channels, out_channels, 3),
#                 ConvBNAct(out_channels, out_channels, 3),
#             )

#     def forward(self, x):
#         return _apply_framewise(self.net, x), {}


# class TypedGPFStem(nn.Module):
#     """
#     Primitive-aware adapter.

#     geometry:    mask / edge / normals / SDF
#     motion:      motion amplitude + flow dx/dy
#     reliability: explicit soft gate, not treated as a normal feature
#     residual:    optional branch; disabled by default for the stable run
#     """
#     def __init__(
#         self,
#         geo_channels: int,
#         motion_channels: int,
#         residual_channels: int,
#         out_channels: int,
#         branch_channels: int = 24,
#         residual_branch_channels: int = 12,
#         use_reliability_gate: bool = True,
#         gate_strength: float = 0.75,
#         primitive_dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.use_reliability_gate = bool(use_reliability_gate)
#         self.gate_strength = float(gate_strength)
#         self.primitive_dropout = float(primitive_dropout)

#         if geo_channels <= 0 or motion_channels <= 0:
#             raise ValueError(f"geo_channels and motion_channels must be > 0, got {geo_channels}, {motion_channels}")

#         self.geo_stem = nn.Sequential(
#             ConvBNAct(geo_channels, branch_channels, 3),
#             ConvBNAct(branch_channels, branch_channels, 3),
#         )
#         self.motion_stem = nn.Sequential(
#             ConvBNAct(motion_channels, branch_channels, 3),
#             ConvBNAct(branch_channels, branch_channels, 3),
#         )

#         self.has_residual = residual_channels > 0
#         if self.has_residual:
#             self.residual_stem = nn.Sequential(
#                 ConvBNAct(residual_channels, residual_branch_channels, 3),
#                 ConvBNAct(residual_branch_channels, residual_branch_channels, 3),
#             )
#         else:
#             self.residual_stem = None

#         self.motion_gate = nn.Sequential(
#             ConvBNAct(1, branch_channels, 3),
#             nn.Conv2d(branch_channels, branch_channels, 1),
#         )
#         if self.has_residual:
#             self.residual_gate = nn.Sequential(
#                 ConvBNAct(1, residual_branch_channels, 3),
#                 nn.Conv2d(residual_branch_channels, residual_branch_channels, 1),
#             )
#         else:
#             self.residual_gate = None

#         fuse_in = branch_channels * 2 + (residual_branch_channels if self.has_residual else 0)
#         self.fuse = nn.Sequential(
#             ConvBNAct(fuse_in, out_channels, 1, padding=0),
#             ConvBNAct(out_channels, out_channels, 3),
#         )

#     def _soft_gate(self, logits):
#         gate = torch.sigmoid(logits)
#         return (1.0 - self.gate_strength) + self.gate_strength * gate

#     def _maybe_drop(self, x):
#         if self.training and self.primitive_dropout > 0:
#             if torch.rand((), device=x.device) < self.primitive_dropout:
#                 return x * 0.0
#         return x

#     def forward(
#         self,
#         geo: torch.Tensor,
#         motion: torch.Tensor,
#         reliability: Optional[torch.Tensor] = None,
#         residual: Optional[torch.Tensor] = None,
#         foreground: Optional[torch.Tensor] = None,
#     ):
#         aux = {}
#         geo_feat = _apply_framewise(self.geo_stem, geo)
#         motion_feat = _apply_framewise(self.motion_stem, motion)

#         if self.use_reliability_gate and reliability is not None:
#             motion_gate = _apply_framewise(self.motion_gate, reliability)
#             motion_gate = self._soft_gate(motion_gate)
#             if foreground is not None:
#                 fg = foreground.clamp(0, 1)
#                 motion_gate = motion_gate * (0.5 + 0.5 * fg)
#             motion_feat = motion_feat * motion_gate
#             aux["motion_gate"] = motion_gate

#         feats = [geo_feat, self._maybe_drop(motion_feat)]

#         if self.has_residual and residual is not None:
#             res_feat = _apply_framewise(self.residual_stem, residual)
#             if self.use_reliability_gate and reliability is not None:
#                 res_gate = _apply_framewise(self.residual_gate, reliability)
#                 res_gate = self._soft_gate(res_gate)
#                 if foreground is not None:
#                     fg = foreground.clamp(0, 1)
#                     res_gate = res_gate * (0.5 + 0.5 * fg)
#                 res_feat = res_feat * res_gate
#                 aux["residual_gate"] = res_gate
#             feats.append(self._maybe_drop(res_feat))

#         fused = torch.cat(feats, dim=1)
#         fused = _apply_framewise(self.fuse, fused)
#         return fused, aux


# class TemporalPrimitivePool(nn.Module):
#     """
#     Temporal pooling with optional reliability and phase-bin support.

#     Important OpenGait detail:
#         With unfixed_ordered / all_ordered sampling, OpenGait may pack a local
#         batch of variable-length sequences as x = [1, C, sum(seqL), H, W]
#         while labels still has local batch size B. In that case we must split
#         along the temporal dimension according to seqL, pool each sequence, and
#         concatenate back to [B, C, H, W].
#     """
#     def __init__(
#         self,
#         pool_type: str = "max",
#         phase_bins: int = 8,
#         phase_kappa: float = 4.0,
#         phase_reduce: str = "mean",
#         eps: float = 1e-6,
#     ):
#         super().__init__()
#         self.pool_type = str(pool_type)
#         self.phase_bins = int(phase_bins)
#         self.phase_kappa = float(phase_kappa)
#         self.phase_reduce = phase_reduce
#         self.eps = eps

#     def _flatten_seqL(self, seqL):
#         """Flatten OpenGait seqL containers into a plain Python list."""
#         if seqL is None:
#             return []
#         if torch.is_tensor(seqL):
#             return seqL.detach().cpu().reshape(-1).tolist()
#         if isinstance(seqL, np.ndarray):
#             return seqL.reshape(-1).tolist()
#         if isinstance(seqL, (list, tuple)):
#             out = []
#             for v in seqL:
#                 out.extend(self._flatten_seqL(v))
#             return out
#         return [seqL]

#     def _seqL_to_int_list(self, seqL, batch_size, max_len):
#         vals = self._flatten_seqL(seqL)

#         if len(vals) == 0:
#             return [max_len for _ in range(batch_size)]

#         lens = []
#         for v in vals:
#             if torch.is_tensor(v):
#                 v = v.detach().cpu().reshape(-1)[0].item()
#             elif isinstance(v, np.ndarray):
#                 v = v.reshape(-1)[0].item()
#             elif isinstance(v, (list, tuple)):
#                 if len(v) == 0:
#                     continue
#                 v = v[0]
#             lens.append(int(v))

#         if len(lens) == 0:
#             return [max_len for _ in range(batch_size)]

#         if len(lens) == 1 and batch_size == 1:
#             return [max(1, min(lens[0], max_len))]

#         # OpenGait may provide seqL for multiple input modalities. The first B
#         # entries correspond to the first input stream used here.
#         if len(lens) >= batch_size:
#             lens = lens[:batch_size]
#             return [max(1, min(int(v), max_len)) for v in lens]

#         # Defensive fallback: preserve batch size instead of shrinking embeddings.
#         base = max(1, max_len // batch_size)
#         fallback = [base for _ in range(batch_size)]
#         fallback[-1] = max(1, max_len - base * (batch_size - 1))
#         return fallback

#     @staticmethod
#     def _phase_to_nt2(phase: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         """Normalize phase to [N, T, 2]."""
#         if phase is None:
#             return None
#         if not torch.is_tensor(phase):
#             phase = torch.as_tensor(phase)
#         if phase.dim() == 2:
#             if phase.size(-1) == 2:
#                 return phase.unsqueeze(0).float().contiguous()
#             raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")
#         if phase.dim() != 3:
#             raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")
#         if phase.size(-1) == 2:
#             return phase.float().contiguous()
#         if phase.size(1) == 2:
#             return phase.transpose(1, 2).float().contiguous()
#         raise ValueError(f"Unsupported phase shape: {tuple(phase.shape)}")

#     @staticmethod
#     def _reliability_to_nt(reliability: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         """Normalize reliability to [N, T]."""
#         if reliability is None:
#             return None
#         if not torch.is_tensor(reliability):
#             reliability = torch.as_tensor(reliability)
#         if reliability.dim() == 2:
#             return reliability.float().contiguous()
#         if reliability.dim() == 3:
#             if reliability.size(1) == 1:
#                 return reliability[:, 0].float().contiguous()
#             if reliability.size(-1) == 1:
#                 return reliability[..., 0].float().contiguous()
#         if reliability.dim() == 5:
#             return reliability.float().mean(dim=(1, 3, 4)).contiguous()
#         raise ValueError(f"Unsupported reliability shape: {tuple(reliability.shape)}")

#     def _match_aux_for_time(self, phase, reliability, target_t):
#         phase = self._phase_to_nt2(phase)
#         if phase is not None:
#             phase = _match_nt_time(phase, target_t)

#         reliability = self._reliability_to_nt(reliability)
#         if reliability is not None:
#             reliability = _match_nt_time(reliability.unsqueeze(-1), target_t).squeeze(-1)

#         return phase, reliability

#     def _split_by_seql(self, x, phase=None, reliability=None, seqL=None, batch_size=None):
#         """
#         Return per-sequence chunks.

#         Supports:
#             x = [B, C, T, H, W]
#             x = [1, C, sum(seqL), H, W]
#         """
#         n, c, max_len, h, w = x.shape
#         if batch_size is None:
#             batch_size = n
#         batch_size = int(batch_size)

#         phase, reliability = self._match_aux_for_time(phase, reliability, max_len)

#         # Normal fixed-length batch.
#         if n == batch_size:
#             lens = self._seqL_to_int_list(seqL, batch_size=batch_size, max_len=max_len)
#             chunks = []
#             for i, l in enumerate(lens):
#                 xi = x[i:i + 1, :, :l]
#                 if phase is not None:
#                     pi = phase[i:i + 1, :l] if phase.size(0) == n else phase[:1, :l]
#                 else:
#                     pi = None
#                 if reliability is not None:
#                     ri = reliability[i:i + 1, :l] if reliability.size(0) == n else reliability[:1, :l]
#                 else:
#                     ri = None
#                 chunks.append((xi, pi, ri))
#             return chunks

#         # OpenGait packed variable-length batch.
#         if n == 1 and batch_size > 1:
#             lens = self._seqL_to_int_list(seqL, batch_size=batch_size, max_len=max_len)
#             chunks = []
#             start = 0

#             for i, l in enumerate(lens):
#                 end = min(start + int(l), max_len)
#                 if end <= start:
#                     end = min(start + 1, max_len)

#                 xi = x[:, :, start:end]

#                 if phase is None:
#                     pi = None
#                 elif phase.size(0) == 1:
#                     pi = phase[:, start:end]
#                 else:
#                     pi = phase[i:i + 1, :end - start]

#                 if reliability is None:
#                     ri = None
#                 elif reliability.size(0) == 1:
#                     ri = reliability[:, start:end]
#                 else:
#                     ri = reliability[i:i + 1, :end - start]

#                 chunks.append((xi, pi, ri))
#                 start = end

#             return chunks

#         raise RuntimeError(
#             f"Cannot align temporal batch: x.shape={tuple(x.shape)}, "
#             f"labels batch_size={batch_size}, seqL={seqL}"
#         )

#     def _pool_single(self, x, phase=None, reliability=None):
#         if self.pool_type == "avg":
#             return x.mean(dim=2)

#         if self.pool_type == "max" or x.size(2) == 1:
#             return x.max(dim=2)[0]

#         if self.pool_type == "reliability":
#             if reliability is None:
#                 return x.max(dim=2)[0]
#             w = reliability.clamp(0, 1).view(reliability.size(0), 1, reliability.size(1), 1, 1)
#             return (x * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

#         if self.pool_type in ("phase_bin", "reliability_phase_bin"):
#             if phase is None:
#                 return x.max(dim=2)[0]

#             angle = torch.atan2(phase[..., 0], phase[..., 1])
#             centers = torch.linspace(
#                 -math.pi,
#                 math.pi,
#                 steps=self.phase_bins + 1,
#                 device=x.device,
#                 dtype=x.dtype,
#             )[:-1]
#             weights = torch.exp(self.phase_kappa * torch.cos(angle.unsqueeze(-1) - centers))

#             if self.pool_type == "reliability_phase_bin" and reliability is not None:
#                 weights = weights * reliability.clamp(0, 1).unsqueeze(-1)

#             w = weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#             xb = x.unsqueeze(3)
#             z = (xb * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

#             if self.phase_reduce == "max":
#                 return z.max(dim=2)[0]
#             if self.phase_reduce == "sum":
#                 return z.sum(dim=2)
#             return z.mean(dim=2)

#         return x.max(dim=2)[0]

#     def forward(self, x, seqL=None, phase=None, reliability=None, batch_size=None):
#         if x.dim() != 5:
#             raise RuntimeError(f"TemporalPool expects [N,C,T,H,W], but got {tuple(x.shape)}")

#         if batch_size is None:
#             batch_size = x.size(0)
#         batch_size = int(batch_size)

#         # Fast path: fixed-length normal batch.
#         if x.size(0) == batch_size and self.pool_type == "max":
#             return x.max(dim=2)[0]

#         # Packed or advanced temporal pooling path.
#         chunks = self._split_by_seql(
#             x,
#             phase=phase,
#             reliability=reliability,
#             seqL=seqL,
#             batch_size=batch_size,
#         )
#         outs = [self._pool_single(xi, pi, ri) for xi, pi, ri in chunks]
#         out = torch.cat(outs, dim=0)

#         if out.size(0) != batch_size:
#             raise RuntimeError(
#                 f"TemporalPool output batch mismatch: out.shape={tuple(out.shape)}, "
#                 f"expected batch_size={batch_size}, input x.shape={tuple(x.shape)}, seqL={seqL}"
#             )

#         return out



# def _visual_time_indices(t: int, max_frames: int, device=None):
#     """Evenly sample temporal indices for lightweight visualization."""
#     if max_frames is None or int(max_frames) <= 0 or t <= int(max_frames):
#         return torch.arange(t, device=device)
#     return torch.linspace(0, t - 1, steps=int(max_frames), device=device).round().long()


# def _visual_flatten_nt(x: torch.Tensor, max_frames: int = 8, normalize: bool = True) -> torch.Tensor:
#     """Convert [N,C,T,H,W] to [N*T,C,H,W] for OpenGait visual_summary."""
#     if x is None:
#         return None
#     if x.dim() != 5:
#         raise ValueError(f"Expected [N,C,T,H,W] visual tensor, got {tuple(x.shape)}")
#     x = x.detach().float()
#     idx = _visual_time_indices(x.size(2), max_frames, device=x.device)
#     x = x.index_select(2, idx)
#     n, c, t, h, w = x.shape
#     x = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
#     if normalize:
#         x = _visual_minmax(x)
#     return x


# def _visual_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """Per-image min-max normalization for heatmaps or primitive maps."""
#     x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
#     dims = tuple(range(2, x.dim()))
#     x_min = x.amin(dim=dims, keepdim=True)
#     x_max = x.amax(dim=dims, keepdim=True)
#     return (x - x_min) / (x_max - x_min + eps)


# def _visual_energy(x: torch.Tensor, out_hw=None) -> torch.Tensor:
#     """Channel-average activation energy as a weak attention proxy."""
#     e = x.detach().float().abs().mean(dim=1, keepdim=True)
#     if out_hw is not None and e.shape[-2:] != tuple(out_hw):
#         n, c, t, h, w = e.shape
#         e2 = e.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
#         e2 = F.interpolate(e2, size=tuple(out_hw), mode="bilinear", align_corners=False)
#         e = e2.reshape(n, t, c, *out_hw).permute(0, 2, 1, 3, 4).contiguous()
#     return e


# def _limb_prior_from_mask(mask: torch.Tensor) -> torch.Tensor:
#     """
#     Build a coarse limb prior from canonical gait silhouettes.

#     It is intentionally heuristic and only for diagnostics:
#       - lower-body foreground is treated as leg-dominant;
#       - upper/mid side foreground is treated as arm-dominant;
#       - central upper torso is down-weighted.
#     """
#     if mask is None:
#         return None
#     fg = mask.detach().float().clamp(0, 1)
#     n, c, t, h, w = fg.shape
#     yy = torch.linspace(0, 1, h, device=fg.device, dtype=fg.dtype).view(1, 1, 1, h, 1)
#     xx = torch.linspace(-1, 1, w, device=fg.device, dtype=fg.dtype).view(1, 1, 1, 1, w)

#     leg = torch.sigmoid((yy - 0.55) * 24.0)
#     arm_vertical = torch.sigmoid((yy - 0.25) * 24.0) * torch.sigmoid((0.72 - yy) * 24.0)
#     arm_side = torch.sigmoid((xx.abs() - 0.42) * 16.0)
#     arm = arm_vertical * arm_side
#     prior = torch.maximum(leg, arm).expand_as(fg) * fg
#     return prior.clamp(0, 1)


# class GPFGait(BaseModel):
#     """OpenGait model wrapper for GPF ablations and the stable structured model."""

#     def build_network(self, model_cfg: Dict):
#         self.model_cfg = model_cfg
#         self.variant = model_cfg.get("variant", "gpf_structured")
#         self.input_index = int(model_cfg.get("input_index", 0))
#         self.phase_input_index = model_cfg.get("phase_input_index", None)
#         self.source_channels = int(model_cfg.get("source_channels", 12))
#         self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))

#         # Visualization is off by default to keep training/evaluation fast and logs small.
#         # Enable with model_cfg.visualize: True. Returned tensors are placed in
#         # output["visual_summary"] and are compatible with OpenGait image logging.
#         self.visualize = bool(model_cfg.get("visualize", model_cfg.get("enable_visualization", False)))
#         self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))
#         self.visualize_inputs = bool(model_cfg.get("visualize_inputs", True))
#         self.visualize_gates = bool(model_cfg.get("visualize_gates", True))
#         self.visualize_features = bool(model_cfg.get("visualize_features", True))
#         self.visualize_limb_focus = bool(model_cfg.get("visualize_limb_focus", True))

#         self.norm = GPFChannelNorm(
#             enabled=bool(model_cfg.get("enable_channel_norm", True)),
#             sdf_clip=float(model_cfg.get("sdf_clip", 1.0)),
#             flow_clip=float(model_cfg.get("flow_clip", 1.0)),
#             residual_clip=float(model_cfg.get("residual_clip", 1.0)),
#         )

#         stem_out_channels = int(model_cfg.get("stem_out_channels", model_cfg.get("grouped_fuse_channels", 32)))

#         # Optional global channel filter for grouped/structured variants.
#         allowed_channels = model_cfg.get("channel_indices", None)
#         self.allowed_channels = set(map(int, allowed_channels)) if allowed_channels is not None else None

#         def _filter_allowed(indices):
#             indices = list(map(int, indices))
#             if self.allowed_channels is None:
#                 return indices
#             return [i for i in indices if i in self.allowed_channels]

#         self.geo_indices = _filter_allowed(_as_list(model_cfg.get("geo_indices", [0, 1, 2, 3, 4]), [0, 1, 2, 3, 4]))
#         self.motion_indices = _filter_allowed(_as_list(model_cfg.get("motion_indices", [5, 8, 9]), [5, 8, 9]))

#         reliability_index = model_cfg.get("reliability_index", 10)
#         if reliability_index is not None:
#             reliability_index = int(reliability_index)
#             if self.allowed_channels is not None and reliability_index not in self.allowed_channels:
#                 reliability_index = None
#         self.reliability_index = reliability_index

#         # Stable default: no residual branch. Enable explicitly with residual_indices: [11].
#         self.residual_indices = _filter_allowed(_as_list(model_cfg.get("residual_indices", []), []))
#         self.phase_indices = _filter_allowed(_as_list(model_cfg.get("phase_indices", [6, 7]), [6, 7]))

#         if self.variant == "sil":
#             self.adapter_type = "sil"
#             self.adapter = DirectGPFStem(1, stem_out_channels, scalar=False)

#         elif self.variant in ("gpf_geometry_scalar", "gpf_motion_scalar", "gpf_all_scalar"):
#             if self.variant == "gpf_geometry_scalar":
#                 self.channel_indices = self.geo_indices if len(self.geo_indices) > 0 else [0, 1, 2, 3, 4]
#             elif self.variant == "gpf_motion_scalar":
#                 if "channel_indices" in model_cfg:
#                     self.channel_indices = _as_list(model_cfg.get("channel_indices"), None)
#                 else:
#                     # Keep the original ablation definition: motion + reliability + residual.
#                     self.channel_indices = [5, 8, 9, 10, 11]
#             else:
#                 self.channel_indices = _as_list(
#                     model_cfg.get("channel_indices", list(range(self.source_channels))),
#                     list(range(self.source_channels)),
#                 )
#             self.channel_indices = list(map(int, self.channel_indices))
#             self.adapter_type = "direct"
#             self.adapter = DirectGPFStem(len(self.channel_indices), stem_out_channels, scalar=True)

#         elif self.variant in ("gpf_all_direct", "gpf_direct"):
#             self.channel_indices = _as_list(
#                 model_cfg.get("channel_indices", list(range(self.source_channels))),
#                 list(range(self.source_channels)),
#             )
#             self.channel_indices = list(map(int, self.channel_indices))
#             self.adapter_type = "direct"
#             self.adapter = DirectGPFStem(len(self.channel_indices), stem_out_channels, scalar=False)

#         elif self.variant in ("gpf_grouped", "gpf_structured"):
#             self.adapter_type = "typed"
#             self.adapter = TypedGPFStem(
#                 geo_channels=len(self.geo_indices),
#                 motion_channels=len(self.motion_indices),
#                 residual_channels=len(self.residual_indices),
#                 out_channels=stem_out_channels,
#                 branch_channels=int(model_cfg.get("grouped_branch_channels", 24)),
#                 residual_branch_channels=int(model_cfg.get("residual_branch_channels", 12)),
#                 use_reliability_gate=bool(model_cfg.get("use_reliability_gate", True)),
#                 gate_strength=float(model_cfg.get("reliability_gate_strength", 0.75)),
#                 primitive_dropout=float(model_cfg.get("primitive_dropout", 0.0)),
#             )
#         else:
#             raise ValueError(f"Unknown GPFGait variant: {self.variant}")

#         self.use_phase_film = bool(model_cfg.get("use_phase_film", False))
#         if self.use_phase_film:
#             self.phase_film = PhaseFiLM(
#                 stem_out_channels,
#                 hidden_channels=int(model_cfg.get("phase_hidden_channels", 32)),
#                 strength=float(model_cfg.get("phase_film_strength", 0.5)),
#             )
#         else:
#             self.phase_film = nn.Identity()

#         self.backbone = FrameBackbone(
#             stem_out_channels,
#             channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
#             strides=model_cfg.get("backbone_strides", [1, 2, 1, 1]),
#             blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
#         )

#         bin_num = model_cfg.get("bin_num", [32])
#         self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
#         parts_num = int(sum(bin_num))

#         self.TP = TemporalPrimitivePool(
#             pool_type=model_cfg.get("temporal_pool", "max"),
#             phase_bins=int(model_cfg.get("phase_bins", 8)),
#             phase_kappa=float(model_cfg.get("phase_kappa", 4.0)),
#             phase_reduce=model_cfg.get("phase_reduce", "mean"),
#         )

#         emb_channels = int(model_cfg.get("emb_channels", 256))
#         class_num = int(model_cfg["class_num"])
#         fc_cfg = model_cfg.get("SeparateFCs", {})
#         bn_cfg = model_cfg.get("SeparateBNNecks", {})

#         self.FCs = SeparateFCs(
#             parts_num=fc_cfg.get("parts_num", parts_num),
#             in_channels=fc_cfg.get("in_channels", self.backbone.out_channels),
#             out_channels=fc_cfg.get("out_channels", emb_channels),
#             norm=fc_cfg.get("norm", False),
#         )
#         self.BNNecks = SeparateBNNecks(
#             parts_num=bn_cfg.get("parts_num", parts_num),
#             in_channels=bn_cfg.get("in_channels", emb_channels),
#             class_num=bn_cfg.get("class_num", class_num),
#             norm=bn_cfg.get("norm", True),
#         )

#     def _get_phase(self, ipts, gpf: Optional[torch.Tensor]):
#         # Prefer explicit 002-phase.pkl when configured.
#         if self.phase_input_index is not None:
#             idx = int(self.phase_input_index)
#             if 0 <= idx < len(ipts):
#                 try:
#                     phase = _to_nt2(ipts[idx])
#                     if gpf is not None:
#                         phase = phase.to(gpf.device)
#                     return phase
#                 except Exception:
#                     # Fallback to GPF phase channels below.
#                     pass

#         if gpf is None or len(self.phase_indices) < 2:
#             return None

#         i0, i1 = self.phase_indices[:2]
#         if max(i0, i1) >= gpf.size(1):
#             return None
#         # GPF channels 6/7 are spatially constant phase maps. Average to [N,2,T].
#         phase_map = gpf[:, [i0, i1]].mean(dim=(-1, -2))
#         return phase_map.transpose(1, 2).contiguous()  # [N,T,2]

#     def _get_reliability_scalar(self, gpf: Optional[torch.Tensor]):
#         if gpf is None or self.reliability_index is None:
#             return None
#         idx = int(self.reliability_index)
#         if idx >= gpf.size(1):
#             return None
#         return gpf[:, idx].mean(dim=(-1, -2)).clamp(0, 1)  # [N,T]

#     def _build_visual_summary(
#         self,
#         visual_src: Optional[torch.Tensor],
#         gpf: Optional[torch.Tensor],
#         stem_feat: Optional[torch.Tensor],
#         backbone_feat: Optional[torch.Tensor],
#         aux: Optional[Dict[str, torch.Tensor]],
#     ):
#         """
#         Build lightweight visual diagnostics.

#         The maps are not a proof of causality by themselves. They are intended to answer:
#         1) what primitive fields the network sees;
#         2) whether the reliability gate suppresses unreliable motion;
#         3) where the learned feature energy is spatially concentrated;
#         4) whether that energy overlaps a coarse limb prior.
#         """
#         visual_summary = {}
#         max_frames = self.visualize_max_frames

#         if visual_src is not None:
#             visual_summary["image/gpf_or_sil_mask"] = _visual_flatten_nt(
#                 visual_src.clamp(0, 1), max_frames=max_frames, normalize=False
#             )

#         if not self.visualize:
#             return visual_summary

#         base_hw = None
#         if visual_src is not None:
#             base_hw = visual_src.shape[-2:]
#         elif gpf is not None:
#             base_hw = gpf.shape[-2:]

#         if self.visualize_inputs and gpf is not None:
#             # Show physically meaningful GPF channels. Names follow the default channel order.
#             channel_names = {
#                 0: "mask",
#                 1: "edge",
#                 4: "sdf",
#                 5: "motion_amp",
#                 10: "motion_reliability",
#                 11: "midfreq_residual_abs",
#             }
#             for idx, name in channel_names.items():
#                 if idx < gpf.size(1):
#                     x = gpf[:, idx:idx + 1]
#                     if idx == 11:
#                         x = x.abs()
#                     visual_summary[f"image/gpf_{idx:02d}_{name}"] = _visual_flatten_nt(
#                         x, max_frames=max_frames, normalize=True
#                     )

#             if gpf.size(1) > 9:
#                 flow_mag = torch.sqrt(gpf[:, 8:9].pow(2) + gpf[:, 9:10].pow(2) + 1e-6)
#                 visual_summary["image/gpf_flow_magnitude"] = _visual_flatten_nt(
#                     flow_mag, max_frames=max_frames, normalize=True
#                 )

#         if self.visualize_gates and aux is not None:
#             if "motion_gate" in aux:
#                 motion_gate = aux["motion_gate"].detach().float().mean(dim=1, keepdim=True)
#                 visual_summary["image/learned_motion_gate"] = _visual_flatten_nt(
#                     motion_gate, max_frames=max_frames, normalize=True
#                 )
#             if "residual_gate" in aux:
#                 residual_gate = aux["residual_gate"].detach().float().mean(dim=1, keepdim=True)
#                 visual_summary["image/learned_residual_gate"] = _visual_flatten_nt(
#                     residual_gate, max_frames=max_frames, normalize=True
#                 )

#         if self.visualize_features and stem_feat is not None:
#             stem_energy = _visual_energy(stem_feat, out_hw=base_hw)
#             if visual_src is not None and stem_energy.shape[-2:] == visual_src.shape[-2:]:
#                 stem_energy = stem_energy * (0.25 + 0.75 * visual_src.clamp(0, 1))
#             visual_summary["image/stem_feature_energy"] = _visual_flatten_nt(
#                 stem_energy, max_frames=max_frames, normalize=True
#             )

#         if self.visualize_features and backbone_feat is not None:
#             backbone_energy = _visual_energy(backbone_feat, out_hw=base_hw)
#             if visual_src is not None and backbone_energy.shape[-2:] == visual_src.shape[-2:]:
#                 backbone_energy = backbone_energy * (0.25 + 0.75 * visual_src.clamp(0, 1))
#             visual_summary["image/backbone_feature_energy"] = _visual_flatten_nt(
#                 backbone_energy, max_frames=max_frames, normalize=True
#             )

#             if self.visualize_limb_focus and visual_src is not None:
#                 limb_prior = _limb_prior_from_mask(visual_src)
#                 visual_summary["image/coarse_limb_prior"] = _visual_flatten_nt(
#                     limb_prior, max_frames=max_frames, normalize=False
#                 )
#                 visual_summary["image/backbone_energy_on_limb_prior"] = _visual_flatten_nt(
#                     backbone_energy * limb_prior, max_frames=max_frames, normalize=True
#                 )
#                 torso_prior = (visual_src.clamp(0, 1) - limb_prior).clamp(0, 1)
#                 visual_summary["image/backbone_energy_on_torso_prior"] = _visual_flatten_nt(
#                     backbone_energy * torso_prior, max_frames=max_frames, normalize=True
#                 )

#         return visual_summary

#     def forward(self, inputs):
#         ipts, labs, typs, vies, seqL = inputs

#         gpf = None
#         phase = None
#         reliability_scalar = None

#         if self.adapter_type == "sil":
#             x_in = _to_ncthw(ipts[self.input_index], expected_channels=1)
#             x, aux = self.adapter(x_in)
#             visual_src = x_in
#         else:
#             gpf = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
#             gpf = self.norm(gpf)
#             phase = self._get_phase(ipts, gpf)
#             reliability_scalar = self._get_reliability_scalar(gpf)
#             visual_src = gpf[:, 0:1] if gpf.size(1) > 0 else None

#             if self.adapter_type == "direct":
#                 x_in = _select_channels(gpf, self.channel_indices, "channel_indices")
#                 x, aux = self.adapter(x_in)
#             else:
#                 geo = _select_channels(gpf, self.geo_indices, "geo_indices")
#                 motion = _select_channels(gpf, self.motion_indices, "motion_indices")

#                 reliability = None
#                 if self.reliability_index is not None and int(self.reliability_index) < gpf.size(1):
#                     reliability = gpf[:, int(self.reliability_index):int(self.reliability_index) + 1]

#                 residual = None
#                 if len(self.residual_indices) > 0:
#                     residual = _select_channels(gpf, self.residual_indices, "residual_indices")

#                 foreground = gpf[:, 0:1] if gpf.size(1) > 0 else None
#                 x, aux = self.adapter(
#                     geo,
#                     motion,
#                     reliability=reliability,
#                     residual=residual,
#                     foreground=foreground,
#                 )

#         if self.use_phase_film:
#             x = self.phase_film(x, phase)

#         stem_feat = x
#         x = self.backbone(x)
#         backbone_feat = x
#         # outs = self.TP(x, seqL=seqL, phase=phase, reliability=reliability_scalar)
#         local_bs = labs.size(0) if torch.is_tensor(labs) else len(labs)

#         outs = self.TP(
#             x,
#             seqL=seqL,
#             phase=phase,
#             reliability=reliability_scalar,
#             batch_size=local_bs,
#         )

#         feat = self.HPP(outs)

#         embed_1 = self.FCs(feat)
#         embed_2, logits = self.BNNecks(embed_1)
#         embed = embed_2 if self.inference_use_emb2 else embed_1

#         visual_summary = self._build_visual_summary(
#             visual_src=visual_src,
#             gpf=gpf,
#             stem_feat=stem_feat,
#             backbone_feat=backbone_feat,
#             aux=aux,
#         )

#         if embed_1.size(0) != local_bs:
#             raise RuntimeError(
#                 f"Embedding batch mismatch: embed_1.shape={tuple(embed_1.shape)}, "
#                 f"labels.shape={tuple(labs.shape)}. "
#                 f"Expected embed_1.size(0) == labels.size(0)."
#             )

#         if logits.size(0) != local_bs:
#             raise RuntimeError(
#                 f"Logits batch mismatch: logits.shape={tuple(logits.shape)}, "
#                 f"labels.shape={tuple(labs.shape)}."
#             )


#         return {
#             "training_feat": {
#                 "triplet": {"embeddings": embed_1, "labels": labs},
#                 "softmax": {"logits": logits, "labels": labs},
#             },
#             "visual_summary": visual_summary,
#             "inference_feat": {
#                 "embeddings": embed,
#             },
#         }





# # # -*- coding: utf-8 -*-
# # """
# # GPFGait ablation models for OpenGait.

# # Put this file at:
# #     opengait/modeling/models/gpf_gait.py

# # Then add the following line to opengait/modeling/models/__init__.py:
# #     from .gpf_gait import GPFGait

# # Supported variants via model_cfg.variant:
# #     sil                 : 001-sils.pkl -> GaitBase-style recognizer
# #     gpf_geometry_scalar : GPF channels [0,1,2,3,4] -> scalar map -> recognizer
# #     gpf_motion_scalar   : GPF channels [5,8,9,10,11] -> scalar map -> recognizer
# #     gpf_all_scalar      : GPF channels [0..11] -> scalar map -> recognizer
# #     gpf_all_direct      : GPF channels [0..11] -> recognizer directly
# #     gpf_grouped         : geometry branch + motion branch + phase FiLM -> recognizer

# # Input conventions supported:
# #     [N, T, H, W]
# #     [N, T, C, H, W]
# #     [N, C, T, H, W]
# # """

# # from typing import Optional, Sequence

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # try:
# #     from ..base_model import BaseModel
# #     from ..modules import HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks, SeparateFCs
# # except Exception:  # pragma: no cover - helpful when importing this file outside OpenGait for syntax checks
# #     BaseModel = nn.Module
# #     HorizontalPoolingPyramid = None
# #     PackSequenceWrapper = None
# #     SeparateBNNecks = None
# #     SeparateFCs = None


# # # -----------------------------
# # # Small utilities
# # # -----------------------------

# # def _as_list(x, default=None):
# #     if x is None:
# #         return default
# #     if isinstance(x, (list, tuple)):
# #         return list(x)
# #     return [x]


# # def _sum_bins(bin_num: Sequence[int]) -> int:
# #     return int(sum(int(b) for b in bin_num))


# # def _flatten_ncthw_to_nchw(x: torch.Tensor) -> torch.Tensor:
# #     """[N, C, T, H, W] -> [N*T, C, H, W]."""
# #     n, c, t, h, w = x.shape
# #     return x.permute(0, 2, 1, 3, 4).contiguous().view(n * t, c, h, w)


# # def _unflatten_nchw_to_ncthw(x: torch.Tensor, n: int, t: int) -> torch.Tensor:
# #     """[N*T, C, H, W] -> [N, C, T, H, W]."""
# #     _, c, h, w = x.shape
# #     return x.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


# # def _to_ncthw(x: torch.Tensor, expected_channels: Optional[int] = None) -> torch.Tensor:
# #     """
# #     Convert common OpenGait tensor layouts to [N, C, T, H, W].

# #     The GPF pkl produced by pretreatment_gpf.py is [T, C, H, W], so after
# #     OpenGait collation it is normally [N, T, C, H, W]. Some existing
# #     silhouette transforms may produce [N, T, H, W].
# #     """
# #     if x.dim() == 4:
# #         # [N, T, H, W]
# #         return x.unsqueeze(2).permute(0, 2, 1, 3, 4).contiguous()

# #     if x.dim() != 5:
# #         raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(x.shape)}")

# #     # Already [N, C, T, H, W].
# #     if expected_channels is not None and x.size(1) == expected_channels:
# #         return x.contiguous()

# #     # Common GPF/sil layout after collation: [N, T, C, H, W].
# #     if expected_channels is not None and x.size(2) == expected_channels:
# #         return x.permute(0, 2, 1, 3, 4).contiguous()

# #     # Heuristic fallback.
# #     # If dimension 1 looks like channels and dimension 2 looks like time, keep it.
# #     if x.size(1) <= 16 and x.size(2) > 16:
# #         return x.contiguous()
# #     # Otherwise assume [N, T, C, H, W].
# #     return x.permute(0, 2, 1, 3, 4).contiguous()


# # def _select_channels(x: torch.Tensor, channel_indices: Optional[Sequence[int]]) -> torch.Tensor:
# #     if channel_indices is None:
# #         return x
# #     idx = torch.as_tensor(channel_indices, dtype=torch.long, device=x.device)
# #     return x.index_select(1, idx)


# # # -----------------------------
# # # Fallback OpenGait-like modules
# # # -----------------------------

# # class _FallbackHPP(object):
# #     def __init__(self, bin_num=None):
# #         self.bin_num = [16] if bin_num is None else list(bin_num)

# #     def __call__(self, x: torch.Tensor) -> torch.Tensor:
# #         # x: [N, C, H, W] -> [N, C, P]
# #         n, c = x.size()[:2]
# #         feats = []
# #         for b in self.bin_num:
# #             z = x.view(n, c, int(b), -1)
# #             z = z.mean(-1) + z.max(-1)[0]
# #             feats.append(z)
# #         return torch.cat(feats, dim=-1)


# # class _FallbackSeparateFCs(nn.Module):
# #     def __init__(self, parts_num, in_channels, out_channels, norm=False):
# #         super().__init__()
# #         self.p = parts_num
# #         self.norm = norm
# #         self.fc_bin = nn.Parameter(torch.empty(parts_num, in_channels, out_channels))
# #         nn.init.xavier_uniform_(self.fc_bin)

# #     def forward(self, x):
# #         # [N, C_in, P] -> [N, C_out, P]
# #         x = x.permute(2, 0, 1).contiguous()
# #         weight = F.normalize(self.fc_bin, dim=1) if self.norm else self.fc_bin
# #         out = x.matmul(weight)
# #         return out.permute(1, 2, 0).contiguous()


# # class _FallbackSeparateBNNecks(nn.Module):
# #     def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
# #         super().__init__()
# #         self.p = parts_num
# #         self.class_num = class_num
# #         self.norm = norm
# #         self.parallel_BN1d = parallel_BN1d
# #         self.fc_bin = nn.Parameter(torch.empty(parts_num, in_channels, class_num))
# #         nn.init.xavier_uniform_(self.fc_bin)
# #         self.bn1d = nn.BatchNorm1d(in_channels * parts_num)

# #     def forward(self, x):
# #         n, c, p = x.size()
# #         y = self.bn1d(x.view(n, c * p)).view(n, c, p)
# #         feat = y.permute(2, 0, 1).contiguous()
# #         if self.norm:
# #             logits = F.normalize(feat, dim=-1).matmul(F.normalize(self.fc_bin, dim=1))
# #         else:
# #             logits = feat.matmul(self.fc_bin)
# #         return feat.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


# # # -----------------------------
# # # Network blocks
# # # -----------------------------

# # class TimeDistributed2D(nn.Module):
# #     """Apply a 2D module frame-by-frame on [N, C, T, H, W]."""

# #     def __init__(self, module: nn.Module):
# #         super().__init__()
# #         self.module = module

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         n, _, t, _, _ = x.shape
# #         y = self.module(_flatten_ncthw_to_nchw(x))
# #         return _unflatten_nchw_to_ncthw(y, n, t)


# # class ConvBNReLU(nn.Sequential):
# #     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
# #         super().__init__(
# #             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.ReLU(inplace=True),
# #         )


# # class BasicBlock2D(nn.Module):
# #     def __init__(self, in_channels, out_channels, stride=1):
# #         super().__init__()
# #         self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
# #         self.conv2 = nn.Sequential(
# #             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #         )
# #         if stride != 1 or in_channels != out_channels:
# #             self.downsample = nn.Sequential(
# #                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
# #                 nn.BatchNorm2d(out_channels),
# #             )
# #         else:
# #             self.downsample = nn.Identity()
# #         self.relu = nn.ReLU(inplace=True)

# #     def forward(self, x):
# #         identity = self.downsample(x)
# #         out = self.conv1(x)
# #         out = self.conv2(out)
# #         return self.relu(out + identity)


# # class FrameResNet9(nn.Module):
# #     """
# #     A compact GaitBase-style 2D frame encoder.

# #     Input : [N, C, T, H, W]
# #     Output: [N, C_out, T, H/4, W/4] when strides are [1, 2, 2].
# #     """

# #     def __init__(self, in_channels=1, channels=(64, 128, 256), strides=(1, 2, 2), blocks_per_stage=(1, 1, 1)):
# #         super().__init__()
# #         channels = list(channels)
# #         strides = list(strides)
# #         blocks_per_stage = list(blocks_per_stage)
# #         assert len(channels) == len(strides) == len(blocks_per_stage)

# #         layers = []
# #         curr_c = int(in_channels)
# #         for out_c, stride, num_blocks in zip(channels, strides, blocks_per_stage):
# #             stage = [BasicBlock2D(curr_c, int(out_c), stride=int(stride))]
# #             for _ in range(int(num_blocks) - 1):
# #                 stage.append(BasicBlock2D(int(out_c), int(out_c), stride=1))
# #             layers.append(TimeDistributed2D(nn.Sequential(*stage)))
# #             curr_c = int(out_c)
# #         self.layers = nn.Sequential(*layers)
# #         self.out_channels = curr_c

# #     def forward(self, x):
# #         return self.layers(x)


# # class ScalarGPFStem(nn.Module):
# #     """Compress selected GPF primitive channels into one gait-native scalar map."""

# #     def __init__(self, in_channels, hidden_channels=16, out_activation="none"):
# #         super().__init__()
# #         self.net = TimeDistributed2D(nn.Sequential(
# #             ConvBNReLU(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
# #             nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
# #         ))
# #         self.out_activation = out_activation

# #     def forward(self, x):
# #         x = self.net(x)
# #         if self.out_activation == "sigmoid":
# #             return torch.sigmoid(x)
# #         if self.out_activation == "tanh":
# #             return torch.tanh(x)
# #         return x


# # class GroupedGPFStem(nn.Module):
# #     """Encode geometry and motion primitives separately, then fuse them."""

# #     def __init__(
# #         self,
# #         geo_channels=5,
# #         motion_channels=5,
# #         branch_channels=8,
# #         fuse_channels=16,
# #         geo_indices=(0, 1, 2, 3, 4),
# #         motion_indices=(5, 8, 9, 10, 11),
# #     ):
# #         super().__init__()
# #         self.geo_indices = list(geo_indices)
# #         self.motion_indices = list(motion_indices)
# #         self.geo = TimeDistributed2D(nn.Sequential(
# #             ConvBNReLU(geo_channels, branch_channels, kernel_size=3, stride=1, padding=1),
# #             ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1),
# #         ))
# #         self.motion = TimeDistributed2D(nn.Sequential(
# #             ConvBNReLU(motion_channels, branch_channels, kernel_size=3, stride=1, padding=1),
# #             ConvBNReLU(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1),
# #         ))
# #         self.fuse = TimeDistributed2D(nn.Sequential(
# #             ConvBNReLU(branch_channels * 2, fuse_channels, kernel_size=1, stride=1, padding=0),
# #             ConvBNReLU(fuse_channels, fuse_channels, kernel_size=3, stride=1, padding=1),
# #         ))
# #         self.out_channels = int(fuse_channels)

# #     def forward(self, gpf):
# #         geo = gpf.index_select(1, torch.as_tensor(self.geo_indices, dtype=torch.long, device=gpf.device))
# #         mot = gpf.index_select(1, torch.as_tensor(self.motion_indices, dtype=torch.long, device=gpf.device))
# #         return self.fuse(torch.cat([self.geo(geo), self.motion(mot)], dim=1))


# # class PhaseFiLM(nn.Module):
# #     """Use [sin(phase), cos(phase)] to modulate frame features."""

# #     def __init__(self, channels, hidden_channels=32, zero_init=True):
# #         super().__init__()
# #         self.channels = int(channels)
# #         self.mlp = nn.Sequential(
# #             nn.Linear(2, hidden_channels),
# #             nn.ReLU(inplace=True),
# #             nn.Linear(hidden_channels, self.channels * 2),
# #         )
# #         if zero_init:
# #             nn.init.zeros_(self.mlp[-1].weight)
# #             nn.init.zeros_(self.mlp[-1].bias)

# #     def forward(self, x, phase):
# #         # x: [N, C, T, H, W], phase: [N, T, 2]
# #         if phase is None:
# #             return x
# #         gamma_beta = self.mlp(phase.float())  # [N, T, 2C]
# #         gamma, beta = gamma_beta.chunk(2, dim=-1)
# #         gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
# #         beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
# #         return x * (1.0 + gamma) + beta


# # # -----------------------------
# # # Main OpenGait model
# # # -----------------------------

# # class GPFGait(BaseModel):
# #     """One configurable model class for all first-round GPF ablations."""

# #     DEFAULT_CHANNELS = {
# #         "sil": None,
# #         "gpf_geometry_scalar": [0, 1, 2, 3, 4],
# #         "gpf_motion_scalar": [5, 8, 9, 10, 11],
# #         "gpf_all_scalar": list(range(12)),
# #         "gpf_all_direct": list(range(12)),
# #         "gpf_grouped": list(range(12)),
# #     }

# #     def build_network(self, model_cfg):
# #         self.variant = model_cfg.get("variant", "gpf_all_scalar")
# #         self.input_index = int(model_cfg.get("input_index", 0))
# #         self.phase_input_index = model_cfg.get("phase_input_index", None)
# #         if self.phase_input_index is not None:
# #             self.phase_input_index = int(self.phase_input_index)
# #         self.source_channels = int(model_cfg.get("source_channels", 1 if self.variant == "sil" else 12))
# #         self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", True))

# #         self.channel_indices = model_cfg.get("channel_indices", self.DEFAULT_CHANNELS.get(self.variant, None))
# #         self.channel_indices = _as_list(self.channel_indices, None)

# #         scalar_hidden = int(model_cfg.get("scalar_hidden_channels", 16))
# #         scalar_activation = model_cfg.get("scalar_activation", "none")
# #         grouped_branch_channels = int(model_cfg.get("grouped_branch_channels", 8))
# #         grouped_fuse_channels = int(model_cfg.get("grouped_fuse_channels", 16))
# #         phase_hidden_channels = int(model_cfg.get("phase_hidden_channels", 32))
# #         self.use_phase_film = bool(model_cfg.get("use_phase_film", self.variant == "gpf_grouped"))

# #         if self.variant == "sil":
# #             self.input_adapter = nn.Identity()
# #             backbone_in_channels = 1
# #         elif self.variant in ["gpf_geometry_scalar", "gpf_motion_scalar", "gpf_all_scalar"]:
# #             selected_c = len(self.channel_indices)
# #             self.input_adapter = ScalarGPFStem(selected_c, hidden_channels=scalar_hidden, out_activation=scalar_activation)
# #             backbone_in_channels = 1
# #         elif self.variant == "gpf_all_direct":
# #             self.input_adapter = nn.Identity()
# #             backbone_in_channels = len(self.channel_indices) if self.channel_indices is not None else self.source_channels
# #         elif self.variant == "gpf_grouped":
# #             geo_indices = model_cfg.get("geo_indices", [0, 1, 2, 3, 4])
# #             motion_indices = model_cfg.get("motion_indices", [5, 8, 9, 10, 11])
# #             self.input_adapter = GroupedGPFStem(
# #                 geo_channels=len(geo_indices),
# #                 motion_channels=len(motion_indices),
# #                 branch_channels=grouped_branch_channels,
# #                 fuse_channels=grouped_fuse_channels,
# #                 geo_indices=geo_indices,
# #                 motion_indices=motion_indices,
# #             )
# #             backbone_in_channels = grouped_fuse_channels
# #         else:
# #             raise ValueError(f"Unknown GPFGait variant: {self.variant}")

# #         if self.use_phase_film:
# #             self.phase_film = PhaseFiLM(backbone_in_channels, hidden_channels=phase_hidden_channels)
# #         else:
# #             self.phase_film = None

# #         backbone_channels = model_cfg.get("backbone_channels", [64, 128, 256])
# #         backbone_strides = model_cfg.get("backbone_strides", [1, 2, 2])
# #         blocks_per_stage = model_cfg.get("blocks_per_stage", [1, 1, 1])
# #         self.Backbone = FrameResNet9(
# #             in_channels=backbone_in_channels,
# #             channels=backbone_channels,
# #             strides=backbone_strides,
# #             blocks_per_stage=blocks_per_stage,
# #         )
# #         backbone_out_channels = self.Backbone.out_channels

# #         # GaitBase-style set/temporal pooling and HPP.
# #         if PackSequenceWrapper is not None:
# #             self.TP = PackSequenceWrapper(torch.max)
# #         else:
# #             self.TP = None
# #         bin_num = model_cfg.get("bin_num", [16])
# #         self.bin_num = list(bin_num)
# #         HPP = HorizontalPoolingPyramid if HorizontalPoolingPyramid is not None else _FallbackHPP
# #         self.HPP = HPP(bin_num=self.bin_num)
# #         parts_num = _sum_bins(self.bin_num)

# #         FCs = SeparateFCs if SeparateFCs is not None else _FallbackSeparateFCs
# #         BNNecks = SeparateBNNecks if SeparateBNNecks is not None else _FallbackSeparateBNNecks

# #         fcs_cfg = dict(model_cfg.get("SeparateFCs", {}))
# #         fcs_cfg.setdefault("parts_num", parts_num)
# #         fcs_cfg.setdefault("in_channels", backbone_out_channels)
# #         fcs_cfg.setdefault("out_channels", int(model_cfg.get("emb_channels", 256)))
# #         self.FCs = FCs(**fcs_cfg)

# #         bn_cfg = dict(model_cfg.get("SeparateBNNecks", {}))
# #         bn_cfg.setdefault("parts_num", parts_num)
# #         bn_cfg.setdefault("in_channels", fcs_cfg["out_channels"])
# #         bn_cfg.setdefault("class_num", int(model_cfg.get("class_num", 200)))
# #         self.BNNecks = BNNecks(**bn_cfg)

# #     def _prepare_main_input(self, ipts: Sequence[torch.Tensor]) -> torch.Tensor:
# #         x = ipts[self.input_index]
# #         x = _to_ncthw(x.float(), expected_channels=self.source_channels)
# #         if self.variant not in ["sil", "gpf_grouped"]:
# #             x = _select_channels(x, self.channel_indices)
# #         return x

# #     def _prepare_phase(self, ipts: Sequence[torch.Tensor], gpf_ncthw: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
# #         phase = None
# #         if self.phase_input_index is not None and self.phase_input_index < len(ipts):
# #             phase = ipts[self.phase_input_index].float()
# #             if phase.dim() == 3:
# #                 # [N, T, 2] or [N, 2, T]
# #                 if phase.size(-1) == 2:
# #                     pass
# #                 elif phase.size(1) == 2:
# #                     phase = phase.transpose(1, 2).contiguous()
# #                 else:
# #                     phase = None
# #             elif phase.dim() == 5:
# #                 # [N, T, 2, H, W] or [N, 2, T, H, W]
# #                 ph = _to_ncthw(phase, expected_channels=2)
# #                 phase = ph.mean(dim=(-1, -2)).transpose(1, 2).contiguous()  # [N, T, 2]
# #             else:
# #                 phase = None

# #         # Fallback: channels 6/7 in GPF are broadcast phase maps.
# #         if phase is None and gpf_ncthw is not None and gpf_ncthw.size(1) > 7:
# #             phase = gpf_ncthw[:, [6, 7]].mean(dim=(-1, -2)).transpose(1, 2).contiguous()
# #         return phase

# #     def _temporal_pool(self, x: torch.Tensor, seqL):
# #         if self.TP is not None:
# #             return self.TP(x, seqL, options={"dim": 2})[0]
# #         # Fallback for standalone testing only.
# #         return torch.max(x, dim=2)[0]

# #     def _visualize_input(self, x: torch.Tensor) -> torch.Tensor:
# #         # Return at most one channel for TensorBoard image logging: [N*T, 1, H, W]
# #         vis = x[:, :1].detach()
# #         return _flatten_ncthw_to_nchw(vis)

# #     def forward(self, inputs):
# #         ipts, labs, typs, vies, seqL = inputs
# #         main_x = self._prepare_main_input(ipts)  # [N, C, T, H, W]

# #         if self.variant == "gpf_grouped":
# #             # Keep original full GPF for phase fallback before grouped stem.
# #             gpf_full = main_x
# #             x = self.input_adapter(gpf_full)
# #             phase = self._prepare_phase(ipts, gpf_ncthw=gpf_full)
# #             if self.phase_film is not None:
# #                 x = self.phase_film(x, phase)
# #         else:
# #             x = self.input_adapter(main_x)

# #         out = self.Backbone(x)               # [N, C, T, H', W']
# #         out = self._temporal_pool(out, seqL) # [N, C, H', W']
# #         feat = self.HPP(out)                 # [N, C, P]
# #         embed_1 = self.FCs(feat)             # [N, C_emb, P]
# #         embed_2, logits = self.BNNecks(embed_1)
# #         embed = embed_2 if self.inference_use_emb2 else embed_1

# #         retval = {
# #             "training_feat": {
# #                 "triplet": {"embeddings": embed_1, "labels": labs},
# #                 "softmax": {"logits": logits, "labels": labs},
# #             },
# #             "visual_summary": {
# #                 "image/input_first_channel": self._visualize_input(main_x),
# #                 "image/adapter_first_channel": self._visualize_input(x),
# #             },
# #             "inference_feat": {
# #                 "embeddings": embed,
# #             },
# #         }
# #         return retval
