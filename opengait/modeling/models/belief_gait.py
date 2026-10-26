# -*- coding: utf-8 -*-
"""
BeliefGait: Pose-Free Dynamic View Belief Tracking for GPF Gait Recognition.

Drop into:
    OpenGait/opengait/modeling/models/belief_gait.py
Register in:
    OpenGait/opengait/modeling/models/__init__.py
with:
    from .belief_gait import BeliefGait

The on-disk 000-gpf.pkl remains unchanged. Existing 12-channel GPF is compacted
lazily at the model boundary using:
    source_channels: 12
    input_channel_indices: [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]

Compact internal layout:
    00 silhouette mask
    01 body edge
    02 contour normal x
    03 contour normal y
    04 signed distance field
    05 motion amplitude
    06 optical-flow dx
    07 optical-flow dy
    08 motion reliability
    09 mid-frequency residual

Design principles:
    * no explicit phase or pre-defined body-part input;
    * framewise circular view belief tracking, not fabricated absolute angles;
    * turning and interaction frames are soft-gated, not discarded;
    * direction-key primitive memory reads aligned and complementary evidence;
    * primitive transport and camera adapters are residual and zero-initialized;
    * every new module has a YAML switch and deliberately small default capacity.
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateBNNecks, SeparateFCs


# -----------------------------------------------------------------------------
# Tensor utilities
# -----------------------------------------------------------------------------


def _to_ncthw(x: torch.Tensor, expected_channels: Optional[int] = None) -> torch.Tensor:
    """Convert common OpenGait layouts to [N,C,T,H,W]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 4:  # [N,T,H,W]
        return x.unsqueeze(1).float().contiguous()
    if x.dim() != 5:
        raise ValueError(f"Expected a 4D/5D tensor, got {tuple(x.shape)}")
    if expected_channels is not None and x.size(1) == expected_channels:
        return x.float().contiguous()
    if expected_channels is not None and x.size(2) == expected_channels:
        return x.transpose(1, 2).float().contiguous()
    if x.size(1) <= 32 and x.size(2) > 32:
        return x.float().contiguous()
    return x.transpose(1, 2).float().contiguous()


def _select_channels(x: torch.Tensor, indices: Sequence[int], name: str) -> torch.Tensor:
    indices = list(map(int, indices))
    if not indices:
        raise ValueError(f"{name} is empty")
    if min(indices) < 0 or max(indices) >= x.size(1):
        raise ValueError(f"{name}={indices} is invalid for x.shape={tuple(x.shape)}")
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(1, idx)


def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a 2D module to [N,C,T,H,W] frame-by-frame."""
    n, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
    y = module(y)
    _, c2, h2, w2 = y.shape
    return y.reshape(n, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


def _resize_ncthw(x: Optional[torch.Tensor], size: Tuple[int, int]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    n, c, t, h, w = x.shape
    if (h, w) == tuple(size):
        return x
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
    return y.reshape(n, t, c, *size).permute(0, 2, 1, 3, 4).contiguous()


def _flatten_seqL(seqL) -> List[int]:
    if seqL is None:
        return []
    if torch.is_tensor(seqL):
        return [int(v) for v in seqL.detach().cpu().reshape(-1).tolist()]
    if isinstance(seqL, np.ndarray):
        return [int(v) for v in seqL.reshape(-1).tolist()]
    if isinstance(seqL, (list, tuple)):
        out: List[int] = []
        for v in seqL:
            out.extend(_flatten_seqL(v))
        return out
    return [int(seqL)]


def _unpack_temporal_batch(x: torch.Tensor, seqL, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Convert OpenGait normal or packed temporal layouts to padded [B,C,T,H,W].

    Training commonly provides [B,C,T,H,W]. Evaluation may provide one packed
    temporal stream [1,C,sum(T_i),H,W] together with seqL. The belief tracker
    requires explicit sequence boundaries, so unpack before any recurrent state.
    """
    n, c, total_t, h, w = x.shape
    lengths = _flatten_seqL(seqL)
    if n == batch_size:
        if not lengths or len(lengths) != batch_size:
            lengths = [total_t] * batch_size
        lengths = [max(1, min(int(v), total_t)) for v in lengths[:batch_size]]
        mask = torch.arange(total_t, device=x.device).view(1, -1) < torch.tensor(lengths, device=x.device).view(-1, 1)
        return x, mask, lengths
    if n == 1 and batch_size > 1:
        if len(lengths) < batch_size:
            raise ValueError(f"Packed input requires {batch_size} lengths, got seqL={lengths}")
        lengths = [max(1, int(v)) for v in lengths[:batch_size]]
        if sum(lengths) > total_t:
            raise ValueError(f"sum(seqL)={sum(lengths)} exceeds packed temporal size={total_t}")
        max_t = max(lengths)
        out = x.new_zeros(batch_size, c, max_t, h, w)
        mask = torch.zeros(batch_size, max_t, dtype=torch.bool, device=x.device)
        start = 0
        for i, length in enumerate(lengths):
            out[i, :, :length] = x[0, :, start:start + length]
            mask[i, :length] = True
            start += length
        return out, mask, lengths
    if batch_size == 1:
        mask = torch.ones(1, total_t, dtype=torch.bool, device=x.device)
        return x, mask, [total_t]
    raise ValueError(f"Cannot infer temporal packing: x.shape={tuple(x.shape)}, batch_size={batch_size}, seqL={lengths}")


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    shape = [1] * x.dim()
    shape[0], shape[dim] = mask.size(0), mask.size(1)
    w = mask.to(x.dtype).view(*shape)
    return (x * w).sum(dim=dim) / w.sum(dim=dim).clamp_min(1e-6)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    masked = logits.masked_fill(~mask, -1e4)
    probs = F.softmax(masked, dim=dim)
    probs = probs * mask.to(probs.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


def _masked_topk_softmax(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """scores/valid: [B,S]. Return zero weights when a row has no valid slot."""
    b, s = scores.shape
    k = max(1, min(int(k), s))
    masked = scores.masked_fill(~valid, -1e4)
    top_idx = masked.topk(k, dim=-1).indices
    keep = torch.zeros_like(valid)
    keep.scatter_(1, top_idx, True)
    keep = keep & valid
    weights = _masked_softmax(scores, keep, dim=-1)
    return torch.where(valid.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))


def circular_convolution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Circular convolution on the last dimension. Bin 0 is the zero shift."""
    if p.shape[-1] != q.shape[-1]:
        raise ValueError(f"Circular convolution shape mismatch: {tuple(p.shape)} vs {tuple(q.shape)}")
    out = torch.zeros_like(p)
    for shift in range(p.size(-1)):
        out = out + q[..., shift:shift + 1] * torch.roll(p, shifts=shift, dims=-1)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def mirror_circular_distribution(p: torch.Tensor) -> torch.Tensor:
    """Map delta-bin i to (-i) mod K; bin 0 remains fixed."""
    k = p.size(-1)
    index = torch.tensor([(-i) % k for i in range(k)], device=p.device, dtype=torch.long)
    return p.index_select(-1, index)


def normalized_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    k = max(2, p.size(-1))
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1) / math.log(k)


def _safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=1e-6)


def _visual_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + eps)


def _visual_flatten_nt(x: torch.Tensor, max_frames: int = 8, normalize: bool = True):
    x = x.detach().float()
    if max_frames > 0 and x.size(2) > max_frames:
        idx = torch.linspace(0, x.size(2) - 1, steps=max_frames, device=x.device).round().long()
        x = x.index_select(2, idx)
    n, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
    return _visual_minmax(x) if normalize else x


# -----------------------------------------------------------------------------
# Camera metadata utilities
# -----------------------------------------------------------------------------


def _camera_targets_from_metadata(
    views,
    device: torch.device,
    num_classes: int,
    label_map: Optional[Dict] = None,
    parse_mode: str = "ccpg_prefix",
    index_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parse camera/category tags such as CCPG '06_0' without treating them as angles."""
    if torch.is_tensor(views):
        raw = views.detach().cpu().reshape(-1).tolist()
    elif isinstance(views, np.ndarray):
        raw = views.reshape(-1).tolist()
    elif isinstance(views, (list, tuple)):
        raw = list(views)
    else:
        raw = [views]
    label_map = {str(k): int(v) for k, v in (label_map or {}).items()}
    parse_mode = str(parse_mode).lower().strip()
    labels, valid = [], []
    for item in raw:
        key = str(item)
        value = None
        if key in label_map:
            value = label_map[key]
        elif parse_mode == "ccpg_prefix":
            match = re.fullmatch(r"\s*(\d+)(?:[_-].*)?\s*", key)
            if match is not None:
                value = int(match.group(1)) - int(index_offset)
        elif parse_mode == "strict_numeric":
            match = re.fullmatch(r"\s*(\d+)\s*", key)
            if match is not None:
                value = int(match.group(1)) - int(index_offset)
        elif parse_mode != "map_only":
            raise ValueError(f"Unknown camera_label_parse_mode={parse_mode!r}")
        ok = value is not None and 0 <= int(value) < int(num_classes)
        labels.append(int(value) if ok else 0)
        valid.append(bool(ok))
    return torch.tensor(labels, dtype=torch.long, device=device), torch.tensor(valid, dtype=torch.bool, device=device)


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = float(strength)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def grad_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, strength)


# -----------------------------------------------------------------------------
# GPF encoder
# -----------------------------------------------------------------------------


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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act=False)
        self.down = ConvBNAct(in_channels, out_channels, 1, stride, padding=0, act=False) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.down(x))


class FrameBackbone(nn.Module):
    def __init__(self, in_channels, channels=(64, 128, 256, 512), strides=(1, 2, 2, 1), blocks_per_stage=(1, 1, 1, 1)):
        super().__init__()
        if not (len(channels) == len(strides) == len(blocks_per_stage)):
            raise ValueError("channels, strides and blocks_per_stage must have the same length")
        layers = []
        c_in = int(in_channels)
        for c_out, stride, count in zip(channels, strides, blocks_per_stage):
            stage = [BasicFrameBlock(c_in, int(c_out), int(stride))]
            for _ in range(max(0, int(count) - 1)):
                stage.append(BasicFrameBlock(int(c_out), int(c_out), 1))
            layers.append(nn.Sequential(*stage))
            c_in = int(c_out)
        self.layers = nn.ModuleList(layers)
        self.out_channels = c_in

    def forward(self, x):
        for layer in self.layers:
            x = _apply_framewise(layer, x)
        return x


class GPFChannelNorm10(nn.Module):
    def __init__(self, enabled=True, sdf_clip=1.0, flow_clip=1.0, residual_clip=1.0):
        super().__init__()
        self.enabled = bool(enabled)
        self.sdf_clip = float(sdf_clip)
        self.flow_clip = float(flow_clip)
        self.residual_clip = float(residual_clip)

    def forward(self, x):
        x = torch.where(torch.isfinite(x.float()), x.float(), torch.zeros_like(x.float()))
        if not self.enabled:
            return x
        y = x.clone()
        y[:, 0] = y[:, 0].clamp(0.0, 1.0)
        y[:, 1] = y[:, 1].clamp(0.0, 1.0)
        y[:, 2:4] = y[:, 2:4].clamp(-1.0, 1.0)
        y[:, 4] = y[:, 4].clamp(-self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
        y[:, 5] = y[:, 5].clamp(0.0, 1.0)
        y[:, 6:8] = y[:, 6:8].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
        y[:, 8] = y[:, 8].clamp(0.0, 1.0)
        y[:, 9] = y[:, 9].clamp(-self.residual_clip, self.residual_clip) / max(self.residual_clip, 1e-6)
        return y


class TypedGPFStem10(nn.Module):
    """Typed GPF stem: geometry, motion and residual are encoded separately."""
    def __init__(
        self,
        geo_channels: int,
        motion_channels: int,
        residual_channels: int,
        out_channels: int,
        branch_channels: int = 20,
        residual_branch_channels: int = 8,
        use_reliability_gate: bool = True,
        gate_strength: float = 0.75,
        residual_scale: float = 0.5,
        primitive_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_reliability_gate = bool(use_reliability_gate)
        self.gate_strength = float(gate_strength)
        self.residual_scale = float(residual_scale)
        self.primitive_dropout = float(primitive_dropout)
        self.geo_stem = nn.Sequential(ConvBNAct(geo_channels, branch_channels), ConvBNAct(branch_channels, branch_channels))
        self.motion_stem = nn.Sequential(ConvBNAct(motion_channels, branch_channels), ConvBNAct(branch_channels, branch_channels))
        self.has_residual = residual_channels > 0
        self.residual_stem = nn.Sequential(ConvBNAct(residual_channels, residual_branch_channels), ConvBNAct(residual_branch_channels, residual_branch_channels)) if self.has_residual else None
        self.motion_gate = nn.Sequential(ConvBNAct(1, branch_channels), nn.Conv2d(branch_channels, branch_channels, 1))
        self.residual_gate = nn.Sequential(ConvBNAct(1, residual_branch_channels), nn.Conv2d(residual_branch_channels, residual_branch_channels, 1)) if self.has_residual else None
        fuse_in = branch_channels * 2 + (residual_branch_channels if self.has_residual else 0)
        self.fuse = nn.Sequential(ConvBNAct(fuse_in, out_channels, 1, padding=0), ConvBNAct(out_channels, out_channels))

    def _gate(self, logits):
        return (1.0 - self.gate_strength) + self.gate_strength * torch.sigmoid(logits)

    def _drop_branch(self, x):
        if self.training and self.primitive_dropout > 0.0 and torch.rand((), device=x.device) < self.primitive_dropout:
            return x * 0.0
        return x

    def forward(self, geo, motion, reliability=None, residual=None, foreground=None):
        aux = {}
        geo_feat = _apply_framewise(self.geo_stem, geo)
        motion_feat = _apply_framewise(self.motion_stem, motion)
        if self.use_reliability_gate and reliability is not None:
            gate = self._gate(_apply_framewise(self.motion_gate, reliability))
            if foreground is not None:
                gate = gate * (0.5 + 0.5 * foreground.clamp(0, 1))
            motion_feat = motion_feat * gate
            aux["stem_motion_gate"] = gate
        feats = [geo_feat, self._drop_branch(motion_feat)]
        if self.has_residual and residual is not None:
            residual_feat = _apply_framewise(self.residual_stem, residual)
            if self.use_reliability_gate and reliability is not None:
                gate = self._gate(_apply_framewise(self.residual_gate, reliability))
                if foreground is not None:
                    gate = gate * (0.5 + 0.5 * foreground.clamp(0, 1))
                residual_feat = residual_feat * gate
                aux["stem_residual_gate"] = gate
            feats.append(self._drop_branch(residual_feat) * self.residual_scale)
        return _apply_framewise(self.fuse, torch.cat(feats, dim=1)), aux


class ZeroInitMapAdapter(nn.Module):
    """A conservative residual adapter whose initial output is exactly zero."""
    def __init__(self, in_channels: int, out_channels: int, depthwise: bool = False):
        super().__init__()
        if depthwise and in_channels == out_channels:
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
            )
        else:
            hidden = max(16, min(out_channels, in_channels))
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, 1, bias=True),
            )
        self.out = self.body[-1]

    def reset_identity(self):
        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.body(x)


# -----------------------------------------------------------------------------
# Frame descriptor, belief tracking and action-aware gates
# -----------------------------------------------------------------------------


class FrameDescriptorExtractor(nn.Module):
    """Spatially aware frame descriptor with signed x/y moments for view tracking."""
    def __init__(self, channels: int, hidden_channels: int = 48, descriptor_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(channels, hidden_channels, 1, bias=False), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4 + 1, descriptor_dim),
            nn.ReLU(inplace=True),
            nn.Linear(descriptor_dim, descriptor_dim),
        )
        self.descriptor_dim = int(descriptor_dim)

    def forward(self, feat: torch.Tensor, reliability: Optional[torch.Tensor], temporal_mask: torch.Tensor):
        b, c, t, h, w = feat.shape
        x = _apply_framewise(self.proj, feat).permute(0, 2, 1, 3, 4)  # [B,T,D,H,W]
        d = x.size(2)
        avg = x.mean(dim=(-1, -2))
        mx = x.amax(dim=(-1, -2))
        energy = x.abs() + 1e-6
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, 1, w)
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, 1, h, 1)
        denom = energy.sum(dim=(-1, -2)).clamp_min(1e-6)
        xmom = (energy * xx).sum(dim=(-1, -2)) / denom
        ymom = (energy * yy).sum(dim=(-1, -2)) / denom
        if reliability is None:
            rel = torch.ones(b, t, 1, device=x.device, dtype=x.dtype)
        else:
            rel = reliability.mean(dim=(1, 3, 4)).unsqueeze(-1).clamp(0, 1)
        desc = self.mlp(torch.cat([avg, mx, xmom, ymom, rel], dim=-1))
        return desc * temporal_mask.unsqueeze(-1).to(desc.dtype)


class CircularViewBeliefFilter(nn.Module):
    """Predict frame observations and relative shifts, then perform a Bayesian circular filter."""
    def __init__(self, descriptor_dim: int, num_bins: int = 8, hidden_channels: int = 96, temperature: float = 1.0):
        super().__init__()
        self.num_bins = int(num_bins)
        self.temperature = float(temperature)
        self.observation_head = nn.Sequential(nn.Linear(descriptor_dim, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, self.num_bins))
        self.transition_head = nn.Sequential(nn.Linear(descriptor_dim * 4, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, self.num_bins))

    def _pair_logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.transition_head(torch.cat([left, right, right - left, right * left], dim=-1))

    def predict(self, desc: torch.Tensor, temporal_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _ = desc.shape
        obs_logits = self.observation_head(desc)
        obs = F.softmax(obs_logits / max(self.temperature, 1e-6), dim=-1)
        transition_logits = desc.new_zeros(b, t, self.num_bins)
        transition_logits[:, 0, 0] = 6.0  # identity shift for initial step
        if t > 1:
            transition_logits[:, 1:] = self._pair_logits(desc[:, :-1], desc[:, 1:])
        transition = F.softmax(transition_logits / max(self.temperature, 1e-6), dim=-1)
        direct2 = desc.new_zeros(b, max(0, t - 2), self.num_bins)
        if t > 2:
            direct2 = F.softmax(self._pair_logits(desc[:, :-2], desc[:, 2:]) / max(self.temperature, 1e-6), dim=-1)
        obs = obs * temporal_mask.unsqueeze(-1).to(obs.dtype)
        transition = transition * temporal_mask.unsqueeze(-1).to(transition.dtype)
        return {
            "observation_logits": obs_logits,
            "observation_posterior": obs,
            "transition_logits": transition_logits,
            "transition_posterior": transition,
            "direct2_transition_posterior": direct2,
        }

    def filter(
        self,
        observation: torch.Tensor,
        transition: torch.Tensor,
        temporal_mask: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
        observation_scale: Optional[torch.Tensor] = None,
        transition_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, k = observation.shape
        if prior is None:
            prior = observation.new_full((b, k), 1.0 / k)
        if observation_scale is None:
            observation_scale = observation.new_ones(b, t)
        if transition_scale is None:
            transition_scale = observation.new_ones(b, t)
        beliefs = []
        previous = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        for i in range(t):
            obs_i = observation[:, i].clamp_min(1e-8)
            if i == 0:
                predicted = previous
            else:
                propagated = circular_convolution(previous, transition[:, i].clamp_min(1e-8))
                blend = transition_scale[:, i:i + 1].clamp(0, 1)
                predicted = (1.0 - blend) * previous + blend * propagated
            log_post = torch.log(predicted.clamp_min(1e-8)) + observation_scale[:, i:i + 1].clamp(0.05, 2.0) * torch.log(obs_i)
            post = F.softmax(log_post, dim=-1)
            is_valid = temporal_mask[:, i:i + 1]
            previous = torch.where(is_valid, post, previous)
            beliefs.append(torch.where(is_valid, previous, torch.zeros_like(previous)))
        return torch.stack(beliefs, dim=1)


class ActionAwareTransitionGate(nn.Module):
    """Predict soft walk/turn/interaction/uncertain states using GPF-derived cues."""
    NUM_ACTIONS = 4

    def __init__(self, descriptor_dim: int, hidden_channels: int = 48):
        super().__init__()
        self.cue_dim = 12
        self.head = nn.Sequential(
            nn.Linear(descriptor_dim + self.cue_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, self.NUM_ACTIONS),
        )
        self.register_buffer("identity_coeff", torch.tensor([1.0, 0.45, 0.25, 0.05]), persistent=True)
        self.register_buffer("tracking_coeff", torch.tensor([0.75, 1.0, 0.65, 0.20]), persistent=True)
        self.register_buffer("observation_coeff", torch.tensor([1.0, 0.80, 0.55, 0.20]), persistent=True)
        self.register_buffer("transport_coeff", torch.tensor([1.0, 0.65, 0.35, 0.10]), persistent=True)
        self.register_buffer("memory_coeff", torch.tensor([1.0, 0.85, 0.30, 0.05]), persistent=True)

    @staticmethod
    def _temporal_delta(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        if x.size(1) > 1:
            out[:, 1:] = (x[:, 1:] - x[:, :-1]).abs()
        return out

    @staticmethod
    def _per_sequence_normalize(x: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        valid = temporal_mask.to(x.dtype)
        mean = (x * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        var = ((x - mean).pow(2) * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return ((x - mean) / torch.sqrt(var + 1e-5)).clamp(-3.0, 3.0)

    def _gpf_cues(self, gpf: torch.Tensor, desc: torch.Tensor, obs_entropy: torch.Tensor, trans_entropy: torch.Tensor, temporal_mask: torch.Tensor):
        b, _, t, h, w = gpf.shape
        mask = gpf[:, 0].clamp(0, 1)
        rel = gpf[:, 8].mean(dim=(-1, -2)).clamp(0, 1)
        flow_x = gpf[:, 6].mean(dim=(-1, -2))
        flow_y = gpf[:, 7].mean(dim=(-1, -2))
        flow_mag = torch.sqrt(flow_x.pow(2) + flow_y.pow(2) + 1e-6)
        normal_x = gpf[:, 2].mean(dim=(-1, -2))
        normal_y = gpf[:, 3].mean(dim=(-1, -2))
        yy = torch.linspace(-1.0, 1.0, h, device=gpf.device, dtype=gpf.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=gpf.device, dtype=gpf.dtype).view(1, 1, 1, w)
        mass = mask.sum(dim=(-1, -2)).clamp_min(1e-6)
        cx = (mask * xx).sum(dim=(-1, -2)) / mass
        cy = (mask * yy).sum(dim=(-1, -2)) / mass
        area = mask.mean(dim=(-1, -2))
        desc_delta = torch.zeros(b, t, device=gpf.device, dtype=gpf.dtype)
        if t > 1:
            desc_delta[:, 1:] = (desc[:, 1:] - desc[:, :-1]).pow(2).mean(dim=-1).sqrt()
        flow_turn = self._temporal_delta(flow_x) + self._temporal_delta(flow_y)
        normal_turn = self._temporal_delta(normal_x) + self._temporal_delta(normal_y)
        centroid_delta = self._temporal_delta(cx) + self._temporal_delta(cy)
        area_delta = self._temporal_delta(area)
        cues = torch.stack([
            rel,
            1.0 - rel,
            flow_mag,
            flow_turn,
            normal_turn,
            centroid_delta,
            area_delta,
            desc_delta,
            obs_entropy,
            trans_entropy,
            self._temporal_delta(obs_entropy),
            self._temporal_delta(trans_entropy),
        ], dim=-1)
        cues = cues * temporal_mask.unsqueeze(-1).to(cues.dtype)
        return cues

    def _pseudo_targets(self, cues: torch.Tensor, temporal_mask: torch.Tensor):
        z = torch.stack([self._per_sequence_normalize(cues[..., i], temporal_mask) for i in range(cues.size(-1))], dim=-1)
        rel = cues[..., 0]
        lowrel = cues[..., 1]
        turn = 0.9 * z[..., 3] + 0.7 * z[..., 4] + 0.6 * z[..., 5] + 0.3 * z[..., 9]
        interact = 0.7 * z[..., 6] + 0.7 * z[..., 7] + 0.25 * z[..., 2] - 0.15 * turn
        uncertain = 1.2 * lowrel + 0.7 * cues[..., 8] + 0.5 * cues[..., 9]
        walk = 1.2 * rel - 0.55 * F.relu(turn) - 0.45 * F.relu(interact) - 0.55 * uncertain
        pseudo = F.softmax(torch.stack([walk, turn, interact, uncertain], dim=-1), dim=-1)
        return pseudo * temporal_mask.unsqueeze(-1).to(pseudo.dtype)

    def forward(self, gpf, desc, observation, transition, temporal_mask):
        obs_entropy = normalized_entropy(observation)
        trans_entropy = normalized_entropy(transition)
        cues = self._gpf_cues(gpf, desc, obs_entropy, trans_entropy, temporal_mask)
        logits = self.head(torch.cat([desc, cues], dim=-1))
        probs = F.softmax(logits, dim=-1) * temporal_mask.unsqueeze(-1).to(logits.dtype)
        pseudo = self._pseudo_targets(cues.detach(), temporal_mask)
        return {
            "action_logits": logits,
            "action_probs": probs,
            "action_pseudo": pseudo,
            "action_cues": cues,
            "identity_weight": probs @ self.identity_coeff,
            "tracking_scale": probs @ self.tracking_coeff,
            "observation_scale": probs @ self.observation_coeff,
            "transport_gate": probs @ self.transport_coeff,
            "memory_write_gate": probs @ self.memory_coeff,
        }


class CameraWeakPrior(nn.Module):
    def __init__(self, num_cameras: int, num_bins: int, prior_floor: float = 0.40, temperature: float = 1.5):
        super().__init__()
        self.num_bins = int(num_bins)
        self.prior_floor = float(prior_floor)
        self.temperature = float(temperature)
        self.logits = nn.Embedding(int(num_cameras), self.num_bins)
        nn.init.zeros_(self.logits.weight)

    def forward(self, labels: torch.Tensor, valid: torch.Tensor):
        logits = self.logits(labels)
        learned = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        uniform = torch.full_like(learned, 1.0 / self.num_bins)
        prior = self.prior_floor * uniform + (1.0 - self.prior_floor) * learned
        return torch.where(valid.unsqueeze(-1), prior, uniform)


# -----------------------------------------------------------------------------
# Gaussian primitives, residual transport and direction-key memory
# -----------------------------------------------------------------------------


class LatentGaussianPrimitiveEncoder(nn.Module):
    """Data-driven local evidence primitives. Slots are not semantic body parts."""
    def __init__(self, channels: int, num_primitives: int = 8, min_variance: float = 0.01, evidence_floor: float = 0.05):
        super().__init__()
        self.channels = int(channels)
        self.num_primitives = int(num_primitives)
        self.min_variance = float(min_variance)
        self.evidence_floor = float(evidence_floor)
        hidden = max(16, channels // 4)
        self.slot_head = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(hidden, self.num_primitives, 1))
        self.quality_head = nn.Sequential(nn.Linear(channels, max(16, channels // 8)), nn.ReLU(inplace=True), nn.Linear(max(16, channels // 8), 1))

    @staticmethod
    def _grid(h, w, device, dtype):
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def forward(self, feat: torch.Tensor, reliability: Optional[torch.Tensor], foreground: Optional[torch.Tensor], temporal_mask: torch.Tensor):
        b, c, t, h, w = feat.shape
        bt, q = b * t, h * w
        frame = feat.permute(0, 2, 1, 3, 4).reshape(bt, c, h, w)
        rel = _resize_ncthw(reliability, (h, w))
        fg = _resize_ncthw(foreground, (h, w))
        evidence = torch.ones(bt, 1, h, w, device=feat.device, dtype=feat.dtype)
        if rel is not None:
            evidence = evidence * rel.permute(0, 2, 1, 3, 4).reshape(bt, 1, h, w).clamp(0, 1)
        if fg is not None:
            evidence = evidence * (0.25 + 0.75 * fg.permute(0, 2, 1, 3, 4).reshape(bt, 1, h, w).clamp(0, 1))
        evidence = evidence.clamp_min(self.evidence_floor)
        logits = self.slot_head(frame).reshape(bt, self.num_primitives, q)
        slot = F.softmax(logits, dim=-1) * evidence.reshape(bt, 1, q)
        mass = slot.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        assign = slot / mass
        grid = self._grid(h, w, feat.device, feat.dtype)
        mu = torch.einsum("bmq,qd->bmd", assign, grid)
        centered = grid.view(1, 1, q, 2) - mu.unsqueeze(2)
        cov = torch.einsum("bmq,bmqi,bmqj->bmij", assign, centered, centered)
        eye = torch.eye(2, device=feat.device, dtype=feat.dtype).view(1, 1, 2, 2)
        cov = cov + self.min_variance * eye
        content = torch.einsum("bmq,bcq->bmc", assign, frame.reshape(bt, c, q))
        quality = torch.sigmoid(self.quality_head(content)).squeeze(-1)
        valid = temporal_mask.reshape(bt, 1).to(feat.dtype)
        return {
            "primitive_assign": assign.reshape(b, t, self.num_primitives, h, w) * valid.view(b, t, 1, 1, 1),
            "primitive_mu": mu.reshape(b, t, self.num_primitives, 2),
            "primitive_cov": cov.reshape(b, t, self.num_primitives, 2, 2),
            "primitive_content": content.reshape(b, t, self.num_primitives, c),
            "primitive_quality": quality.reshape(b, t, self.num_primitives) * valid.reshape(b, t, 1),
        }


class ResidualPrimitiveTransport(nn.Module):
    """Belief-weighted, low-freedom Gaussian splat transport with zero-init residual injection."""
    def __init__(
        self,
        channels: int,
        descriptor_dim: int,
        num_bins: int = 8,
        num_primitives: int = 8,
        transport_dim: int = 96,
        max_affine_delta: float = 0.15,
        max_offset: float = 0.10,
        residual_strength: float = 0.50,
        uncertainty_floor: float = 0.15,
        enable_context_delta: bool = True,
        context_rank: int = 16,
        min_variance: float = 0.01,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_bins = int(num_bins)
        self.num_primitives = int(num_primitives)
        self.transport_dim = int(transport_dim)
        self.max_affine_delta = float(max_affine_delta)
        self.max_offset = float(max_offset)
        self.residual_strength = float(residual_strength)
        self.uncertainty_floor = float(uncertainty_floor)
        self.enable_context_delta = bool(enable_context_delta)
        self.min_variance = float(min_variance)
        self.content_proj = nn.Linear(channels, self.transport_dim)
        self.global_affine = nn.Parameter(torch.zeros(self.num_bins, 2, 3))
        self.global_offsets = nn.Parameter(torch.zeros(self.num_bins, self.num_primitives, 2))
        param_num = self.num_bins * (6 + self.num_primitives * 2)
        self.context_head = nn.Sequential(nn.Linear(descriptor_dim, int(context_rank)), nn.ReLU(inplace=True), nn.Linear(int(context_rank), param_num)) if self.enable_context_delta else None
        self.out_adapter = ZeroInitMapAdapter(self.transport_dim, channels)

    def reset_identity(self):
        nn.init.zeros_(self.global_affine)
        nn.init.zeros_(self.global_offsets)
        if self.context_head is not None:
            nn.init.zeros_(self.context_head[-1].weight)
            nn.init.zeros_(self.context_head[-1].bias)
        self.out_adapter.reset_identity()

    @staticmethod
    def _grid(h, w, device, dtype):
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(-1, 2)

    def _render(self, mu, cov, tokens, quality, affine, offsets, h, w):
        bt, m, _ = mu.shape
        linear = affine[:, :, :2]
        bias = affine[:, :, 2]
        mu_t = torch.einsum("bij,bmj->bmi", linear, mu) + bias.unsqueeze(1) + offsets
        cov_t = torch.einsum("bij,bmjk,blk->bmil", linear, cov, linear)
        eye = torch.eye(2, device=mu.device, dtype=mu.dtype).view(1, 1, 2, 2)
        cov_t = cov_t + self.min_variance * eye
        # Covariance still receives regularization gradients. Detaching only the
        # inverse-render path avoids an expensive and unstable inverse backward.
        inv = torch.linalg.inv(cov_t.float().detach()).to(mu.dtype)
        grid = self._grid(h, w, mu.device, mu.dtype)
        diff = grid.view(1, 1, h * w, 2) - mu_t.unsqueeze(2)
        maha = torch.einsum("bmqi,bmij,bmqj->bmq", diff, inv, diff)
        gaussian = torch.exp(-0.5 * maha.clamp(max=30.0)) * quality.unsqueeze(-1)
        denom = gaussian.sum(dim=1).unsqueeze(1).clamp_min(1e-6)
        rendered = torch.einsum("bmq,bmv->bvq", gaussian, tokens) / denom
        return rendered.reshape(bt, self.transport_dim, h, w), mu_t, cov_t

    def forward(self, feat, desc, belief, action_transport_gate, primitives, temporal_mask):
        b, c, t, h, w = feat.shape
        bt = b * t
        mu = primitives["primitive_mu"].reshape(bt, self.num_primitives, 2)
        cov = primitives["primitive_cov"].reshape(bt, self.num_primitives, 2, 2)
        quality = primitives["primitive_quality"].reshape(bt, self.num_primitives)
        content = primitives["primitive_content"].reshape(bt, self.num_primitives, c)
        tokens = self.content_proj(content)
        global_affine = self.global_affine.view(1, 1, self.num_bins, 2, 3).expand(b, t, -1, -1, -1)
        global_offsets = self.global_offsets.view(1, 1, self.num_bins, self.num_primitives, 2).expand(b, t, -1, -1, -1)
        affine_delta = global_affine
        offsets = global_offsets
        if self.context_head is not None:
            raw = self.context_head(desc).reshape(b, t, self.num_bins, 6 + self.num_primitives * 2)
            affine_delta = affine_delta + raw[..., :6].reshape(b, t, self.num_bins, 2, 3)
            offsets = offsets + raw[..., 6:].reshape(b, t, self.num_bins, self.num_primitives, 2)
        affine_delta = torch.tanh(affine_delta) * self.max_affine_delta
        offsets = torch.tanh(offsets) * self.max_offset
        eye = feat.new_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).view(1, 1, 1, 2, 3)
        affine = eye + affine_delta
        canonical = feat.new_zeros(bt, self.transport_dim, h, w)
        transported_mu = []
        for k in range(self.num_bins):
            ak = affine[:, :, k].reshape(bt, 2, 3)
            ok = offsets[:, :, k].reshape(bt, self.num_primitives, 2)
            rendered, mu_k, _ = self._render(mu, cov, tokens, quality, ak, ok, h, w)
            weight = belief[:, :, k].reshape(bt, 1, 1, 1)
            canonical = canonical + weight * rendered
            transported_mu.append(mu_k)
        delta = self.out_adapter(canonical).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        uncertainty = normalized_entropy(belief)
        confidence = self.uncertainty_floor + (1.0 - self.uncertainty_floor) * (1.0 - uncertainty)
        gate = (confidence * action_transport_gate * temporal_mask.to(feat.dtype)).view(b, 1, t, 1, 1)
        refined = feat + self.residual_strength * gate * delta
        return refined, {
            "transport_delta": delta,
            "transport_gate": gate,
            "transport_affine_delta": affine_delta,
            "transport_offsets": offsets,
            "transported_mu": torch.stack(transported_mu, dim=2).reshape(b, t, self.num_bins, self.num_primitives, 2),
        }


class DirectionKeyPrimitiveMemory(nn.Module):
    """Small online sequence memory keyed by latent content and circular direction belief."""
    def __init__(
        self,
        channels: int,
        descriptor_dim: int,
        num_bins: int = 8,
        num_primitives: int = 8,
        num_slots: int = 8,
        latent_key_dim: int = 64,
        value_dim: int = 96,
        direction_weight: float = 0.75,
        aligned_topk: int = 3,
        complementary_topk: int = 2,
        residual_strength: float = 0.40,
        usage_decay: float = 0.90,
        write_threshold: float = 0.10,
        novelty_threshold: float = 0.18,
        bptt_steps: int = 4,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_bins = int(num_bins)
        self.num_primitives = int(num_primitives)
        self.num_slots = int(num_slots)
        self.direction_weight = float(direction_weight)
        self.aligned_topk = int(aligned_topk)
        self.complementary_topk = int(complementary_topk)
        self.residual_strength = float(residual_strength)
        self.usage_decay = float(usage_decay)
        self.write_threshold = float(write_threshold)
        self.novelty_threshold = float(novelty_threshold)
        self.bptt_steps = max(0, int(bptt_steps))
        self.latent_key_head = nn.Linear(descriptor_dim, latent_key_dim)
        self.value_head = nn.Linear(channels, value_dim)
        self.read_adapter = ZeroInitMapAdapter(value_dim * 2, channels)

    def reset_identity(self):
        self.read_adapter.reset_identity()

    def _redundancy(self, directions: torch.Tensor, valid: torch.Tensor):
        sim = torch.einsum("bsd,btd->bst", directions, directions)
        eye = torch.eye(self.num_slots, device=directions.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
        sim = sim.masked_fill(~pair_valid, -1.0)
        return sim.max(dim=-1).values.clamp_min(0.0)

    def forward(self, feat, desc, belief, primitives, action, reliability, temporal_mask):
        b, c, t, h, w = feat.shape
        m, s = self.num_primitives, self.num_slots
        latent = _safe_normalize(self.latent_key_head(desc), dim=-1)
        direction = _safe_normalize(belief, dim=-1)
        values = self.value_head(primitives["primitive_content"])  # [B,T,M,V]
        assign = primitives["primitive_assign"]
        primitive_quality = primitives["primitive_quality"].mean(dim=-1)
        if reliability is None:
            rel = feat.new_ones(b, t)
        else:
            rel = reliability.mean(dim=(1, 3, 4)).clamp(0, 1)
        uncertainty = normalized_entropy(belief)
        write_gate = action["memory_write_gate"] * rel * temporal_mask.to(feat.dtype)
        alpha = action["identity_weight"] * rel * (1.0 - 0.5 * uncertainty) * temporal_mask.to(feat.dtype)
        beta = action["memory_write_gate"] * rel * (0.35 + 0.65 * uncertainty) * temporal_mask.to(feat.dtype)

        mem_latent = feat.new_zeros(b, s, latent.size(-1))
        mem_direction = feat.new_zeros(b, s, self.num_bins)
        mem_values = feat.new_zeros(b, s, m, values.size(-1))
        mem_quality = feat.new_zeros(b, s)
        mem_usage = feat.new_zeros(b, s)
        mem_valid = torch.zeros(b, s, dtype=torch.bool, device=feat.device)

        output_frames, align_weights_all, comp_weights_all, slot_valid_count = [], [], [], []
        for i in range(t):
            q_latent = latent[:, i]
            q_direction = direction[:, i]
            score_latent = torch.einsum("bd,bsd->bs", q_latent, mem_latent)
            score_direction = torch.einsum("bd,bsd->bs", q_direction, mem_direction)
            align_scores = score_latent + self.direction_weight * score_direction
            comp_scores = score_latent - self.direction_weight * score_direction
            align_w = _masked_topk_softmax(align_scores, mem_valid, self.aligned_topk)
            comp_w = _masked_topk_softmax(comp_scores, mem_valid, self.complementary_topk)
            align_tokens = torch.einsum("bs,bsmv->bmv", align_w, mem_values)
            comp_tokens = torch.einsum("bs,bsmv->bmv", comp_w, mem_values)
            assign_i = assign[:, i]
            align_map = torch.einsum("bmv,bmhw->bvhw", align_tokens, assign_i)
            comp_map = torch.einsum("bmv,bmhw->bvhw", comp_tokens, assign_i)
            delta = self.read_adapter(torch.cat([alpha[:, i:i + 1, None, None] * align_map, beta[:, i:i + 1, None, None] * comp_map], dim=1))
            frame_i = feat[:, :, i] + self.residual_strength * delta
            valid_i = temporal_mask[:, i].view(b, 1, 1, 1)
            output_frames.append(torch.where(valid_i, frame_i, feat[:, :, i]))
            align_weights_all.append(align_w)
            comp_weights_all.append(comp_w)
            slot_valid_count.append(mem_valid.float().sum(dim=-1))

            # Functional replacement update: no in-place autograd modification.
            usage_read = align_w + comp_w
            mem_usage = self.usage_decay * mem_usage + usage_read.detach()
            redundancy = self._redundancy(mem_direction.detach(), mem_valid)
            keep_priority = 0.50 * mem_usage + 0.50 * mem_quality - 0.35 * redundancy
            full = mem_valid.all(dim=-1)
            empty_idx = (~mem_valid).float().argmax(dim=-1)
            low_idx = keep_priority.argmin(dim=-1)
            replace_idx = torch.where(full, low_idx, empty_idx)
            existing_sim = torch.einsum("bd,bsd->bs", q_direction, mem_direction.detach()).masked_fill(~mem_valid, -1.0)
            novelty = 1.0 - existing_sim.max(dim=-1).values.clamp_min(0.0)
            candidate_quality = primitive_quality[:, i] * write_gate[:, i]
            weakest = keep_priority.gather(1, replace_idx.unsqueeze(1)).squeeze(1)
            do_write = temporal_mask[:, i] & (write_gate[:, i] > self.write_threshold) & ((~full) | (candidate_quality > weakest) | (novelty > self.novelty_threshold))
            hot = F.one_hot(replace_idx, num_classes=s).to(feat.dtype) * do_write.unsqueeze(-1).to(feat.dtype)
            hot_key = hot.unsqueeze(-1)
            hot_value = hot.unsqueeze(-1).unsqueeze(-1)
            mem_latent = mem_latent * (1.0 - hot_key) + q_latent.unsqueeze(1) * hot_key
            mem_direction = mem_direction * (1.0 - hot_key) + q_direction.unsqueeze(1) * hot_key
            mem_values = mem_values * (1.0 - hot_value) + values[:, i].unsqueeze(1) * hot_value
            mem_quality = mem_quality * (1.0 - hot) + candidate_quality.unsqueeze(1) * hot
            mem_usage = mem_usage * (1.0 - hot) + candidate_quality.detach().unsqueeze(1) * hot
            mem_valid = mem_valid | hot.bool()
            # Truncated BPTT keeps memory training practical for long CCPG clips.
            # Values still receive gradients within a local temporal window.
            if self.bptt_steps > 0 and (i + 1) % self.bptt_steps == 0:
                mem_latent = mem_latent.detach()
                mem_direction = mem_direction.detach()
                mem_values = mem_values.detach()

        fused = torch.stack(output_frames, dim=2)
        return fused, {
            "memory_align_weights": torch.stack(align_weights_all, dim=1),
            "memory_comp_weights": torch.stack(comp_weights_all, dim=1),
            "memory_write_gate": write_gate,
            "memory_slot_valid_count": torch.stack(slot_valid_count, dim=1),
            "memory_final_direction": mem_direction,
            "memory_final_quality": mem_quality,
            "memory_final_usage": mem_usage,
            "memory_final_valid": mem_valid,
        }


class CameraResidualAdapter(nn.Module):
    """Optional weak camera-conditioned residual; keep disabled in the tag-free main experiment."""
    def __init__(self, num_cameras: int, channels: int, embedding_dim: int = 16, strength: float = 0.15, dropout: float = 0.50):
        super().__init__()
        self.strength = float(strength)
        self.dropout = float(dropout)
        self.embedding = nn.Embedding(int(num_cameras), int(embedding_dim))
        self.head = nn.Linear(int(embedding_dim), channels * 2)

    def reset_identity(self):
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feat, labels, valid):
        emb = self.embedding(labels)
        if self.training and self.dropout > 0:
            keep = (torch.rand(labels.size(0), 1, device=feat.device) >= self.dropout).to(feat.dtype)
        else:
            keep = torch.ones(labels.size(0), 1, device=feat.device, dtype=feat.dtype)
        params = self.head(emb) * keep * valid.unsqueeze(-1).to(feat.dtype)
        scale, bias = params.chunk(2, dim=-1)
        scale = torch.tanh(scale).view(feat.size(0), feat.size(1), 1, 1, 1)
        bias = torch.tanh(bias).view(feat.size(0), feat.size(1), 1, 1, 1)
        return feat + self.strength * (scale * feat + bias)


class ActionAwareTemporalPool(nn.Module):
    """Preserve robust max pooling and add a small action-aware mean residual."""
    def __init__(self, mean_residual_strength: float = 0.20):
        super().__init__()
        self.mean_residual_strength = float(mean_residual_strength)

    def forward(self, x, temporal_mask, identity_weight=None, reliability=None):
        b, c, t, h, w = x.shape
        valid = temporal_mask.view(b, 1, t, 1, 1)
        max_feat = x.masked_fill(~valid, -1e4).max(dim=2).values
        if identity_weight is None:
            weight = temporal_mask.to(x.dtype)
        else:
            weight = identity_weight * temporal_mask.to(x.dtype)
        if reliability is not None:
            weight = weight * (0.25 + 0.75 * reliability.mean(dim=(1, 3, 4)).clamp(0, 1))
        weight = weight.view(b, 1, t, 1, 1)
        mean_feat = (x * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1e-6)
        return max_feat + self.mean_residual_strength * mean_feat


# -----------------------------------------------------------------------------
# BeliefGait wrapper
# -----------------------------------------------------------------------------


class BeliefGait(BaseModel):
    """Full configurable OpenGait model for dynamic latent-view tracking on CCPG."""

    def build_network(self, model_cfg: Dict):
        self.model_cfg = model_cfg
        self.source_channels = int(model_cfg.get("source_channels", 12))
        self.input_channel_indices = list(map(int, model_cfg.get("input_channel_indices", [0, 1, 2, 3, 4, 5, 8, 9, 10, 11])))
        if len(self.input_channel_indices) != 10:
            raise ValueError("BeliefGait requires exactly 10 effective channels after input selection")
        if min(self.input_channel_indices) < 0 or max(self.input_channel_indices) >= self.source_channels:
            raise ValueError(f"Invalid input_channel_indices={self.input_channel_indices} for source_channels={self.source_channels}")
        self.input_index = int(model_cfg.get("input_index", 0))
        self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))
        self.visualize = bool(model_cfg.get("visualize", False))
        self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))
        self.export_belief_state = bool(model_cfg.get("export_belief_state", False))

        self.norm = GPFChannelNorm10(
            enabled=bool(model_cfg.get("enable_channel_norm", True)),
            sdf_clip=float(model_cfg.get("sdf_clip", 1.0)),
            flow_clip=float(model_cfg.get("flow_clip", 1.0)),
            residual_clip=float(model_cfg.get("residual_clip", 1.0)),
        )
        self.geo_indices = list(map(int, model_cfg.get("geo_indices", [0, 1, 2, 3, 4])))
        self.motion_indices = list(map(int, model_cfg.get("motion_indices", [5, 6, 7])))
        self.reliability_index = int(model_cfg.get("reliability_index", 8))
        self.residual_indices = list(map(int, model_cfg.get("residual_indices", [9]))) if bool(model_cfg.get("use_residual_branch", True)) else []

        stem_channels = int(model_cfg.get("stem_out_channels", 32))
        self.stem = TypedGPFStem10(
            geo_channels=len(self.geo_indices),
            motion_channels=len(self.motion_indices),
            residual_channels=len(self.residual_indices),
            out_channels=stem_channels,
            branch_channels=int(model_cfg.get("grouped_branch_channels", 20)),
            residual_branch_channels=int(model_cfg.get("residual_branch_channels", 8)),
            use_reliability_gate=bool(model_cfg.get("use_reliability_gate", True)),
            gate_strength=float(model_cfg.get("reliability_gate_strength", 0.75)),
            residual_scale=float(model_cfg.get("input_residual_scale", 0.5)),
            primitive_dropout=float(model_cfg.get("primitive_dropout", 0.03)),
        )
        self.backbone = FrameBackbone(
            stem_channels,
            channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
            strides=model_cfg.get("backbone_strides", [1, 2, 2, 1]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
        )
        channels = self.backbone.out_channels
        self.num_bins = int(model_cfg.get("num_direction_bins", 8))
        self.num_primitives = int(model_cfg.get("num_primitives", 8))
        descriptor_dim = int(model_cfg.get("descriptor_dim", 96))

        # Independent switches for controlled ablation.
        self.enable_framewise_view_belief = bool(model_cfg.get("enable_framewise_view_belief", True))
        self.enable_relative_view_transition = bool(model_cfg.get("enable_relative_view_transition", True))
        self.enable_circular_belief_filter = bool(model_cfg.get("enable_circular_belief_filter", True))
        self.enable_action_gate = bool(model_cfg.get("enable_action_gate", True))
        self.enable_direction_key_memory = bool(model_cfg.get("enable_direction_key_memory", True))
        self.enable_residual_primitive_transport = bool(model_cfg.get("enable_residual_primitive_transport", True))
        self.enable_internal_flip_consistency = bool(model_cfg.get("enable_internal_flip_consistency", True))
        self.enable_camera_weak_prior = bool(model_cfg.get("enable_camera_weak_prior", False))
        self.enable_camera_adapter = bool(model_cfg.get("enable_camera_adapter", False))
        self.enable_camera_adversary = bool(model_cfg.get("enable_camera_adversary", True))

        self.descriptor = FrameDescriptorExtractor(channels, hidden_channels=int(model_cfg.get("descriptor_hidden_channels", 48)), descriptor_dim=descriptor_dim)
        self.belief_filter = CircularViewBeliefFilter(descriptor_dim, self.num_bins, hidden_channels=int(model_cfg.get("belief_hidden_channels", 96)), temperature=float(model_cfg.get("belief_temperature", 1.0)))
        self.action_gate = ActionAwareTransitionGate(descriptor_dim, hidden_channels=int(model_cfg.get("action_hidden_channels", 48)))
        self.primitive_encoder = LatentGaussianPrimitiveEncoder(channels, self.num_primitives, min_variance=float(model_cfg.get("primitive_min_variance", 0.01)), evidence_floor=float(model_cfg.get("primitive_evidence_floor", 0.05)))
        self.transport = ResidualPrimitiveTransport(
            channels,
            descriptor_dim,
            self.num_bins,
            self.num_primitives,
            transport_dim=int(model_cfg.get("transport_dim", 96)),
            max_affine_delta=float(model_cfg.get("transport_max_affine_delta", 0.15)),
            max_offset=float(model_cfg.get("transport_max_offset", 0.10)),
            residual_strength=float(model_cfg.get("transport_residual_strength", 0.50)),
            uncertainty_floor=float(model_cfg.get("transport_uncertainty_floor", 0.15)),
            enable_context_delta=bool(model_cfg.get("enable_context_transport_delta", True)),
            context_rank=int(model_cfg.get("transport_context_rank", 16)),
            min_variance=float(model_cfg.get("primitive_min_variance", 0.01)),
        )
        self.memory = DirectionKeyPrimitiveMemory(
            channels,
            descriptor_dim,
            self.num_bins,
            self.num_primitives,
            num_slots=int(model_cfg.get("memory_num_slots", 8)),
            latent_key_dim=int(model_cfg.get("memory_latent_key_dim", 64)),
            value_dim=int(model_cfg.get("memory_value_dim", 96)),
            direction_weight=float(model_cfg.get("memory_direction_weight", 0.75)),
            aligned_topk=int(model_cfg.get("memory_aligned_topk", 3)),
            complementary_topk=int(model_cfg.get("memory_complementary_topk", 2)),
            residual_strength=float(model_cfg.get("memory_residual_strength", 0.40)),
            usage_decay=float(model_cfg.get("memory_usage_decay", 0.90)),
            write_threshold=float(model_cfg.get("memory_write_threshold", 0.10)),
            novelty_threshold=float(model_cfg.get("memory_novelty_threshold", 0.18)),
            bptt_steps=int(model_cfg.get("memory_bptt_steps", 4)),
        )

        self.camera_num_classes = int(model_cfg.get("camera_num_classes", 16))
        self.camera_label_parse_mode = str(model_cfg.get("camera_label_parse_mode", "ccpg_prefix"))
        self.camera_label_map = model_cfg.get("camera_label_map", {}) or {}
        self.camera_index_offset = int(model_cfg.get("camera_index_offset", 0))
        self.camera_prior = CameraWeakPrior(self.camera_num_classes, self.num_bins, prior_floor=float(model_cfg.get("camera_prior_floor", 0.40)), temperature=float(model_cfg.get("camera_prior_temperature", 1.5))) if self.enable_camera_weak_prior else None
        self.camera_adapter = CameraResidualAdapter(self.camera_num_classes, channels, embedding_dim=int(model_cfg.get("camera_embedding_dim", 16)), strength=float(model_cfg.get("camera_adapter_strength", 0.15)), dropout=float(model_cfg.get("camera_dropout", 0.50))) if self.enable_camera_adapter else None

        self.temporal_pool = ActionAwareTemporalPool(mean_residual_strength=float(model_cfg.get("identity_mean_residual_strength", 0.20)))
        bin_num = list(model_cfg.get("bin_num", [32]))
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        parts_num = int(sum(bin_num))
        emb_channels = int(model_cfg.get("emb_channels", 256))
        class_num = int(model_cfg["class_num"])
        fc_cfg = model_cfg.get("SeparateFCs", {})
        bn_cfg = model_cfg.get("SeparateBNNecks", {})
        self.FCs = SeparateFCs(parts_num=int(fc_cfg.get("parts_num", parts_num)), in_channels=int(fc_cfg.get("in_channels", channels)), out_channels=int(fc_cfg.get("out_channels", emb_channels)), norm=bool(fc_cfg.get("norm", False)))
        self.BNNecks = SeparateBNNecks(parts_num=int(bn_cfg.get("parts_num", parts_num)), in_channels=int(bn_cfg.get("in_channels", emb_channels)), class_num=int(bn_cfg.get("class_num", class_num)), norm=bool(bn_cfg.get("norm", True)))

        self.camera_adversary_grl_strength = float(model_cfg.get("camera_adversary_grl_strength", 0.20))
        if self.enable_camera_adversary:
            hidden = int(model_cfg.get("camera_adversary_hidden", 128))
            self.camera_adversary = nn.Sequential(nn.Linear(emb_channels, hidden), nn.ReLU(inplace=True), nn.Dropout(float(model_cfg.get("camera_adversary_dropout", 0.10))), nn.Linear(hidden, self.camera_num_classes))
        else:
            self.camera_adversary = None

        # Loss switches stay separate from module switches.
        self.enable_belief_cycle_loss = bool(model_cfg.get("enable_belief_cycle_loss", True))
        self.enable_belief_regularization = bool(model_cfg.get("enable_belief_regularization", True))
        self.enable_action_selfsup_loss = bool(model_cfg.get("enable_action_selfsup_loss", True))
        self.enable_memory_regularization = bool(model_cfg.get("enable_memory_regularization", True))
        self.enable_transport_regularization = bool(model_cfg.get("enable_transport_regularization", True))
        self.enable_canonical_consistency = bool(model_cfg.get("enable_canonical_consistency", True))

    def init_parameters(self):
        super().init_parameters()
        self.transport.reset_identity()
        self.memory.reset_identity()
        if self.camera_adapter is not None:
            self.camera_adapter.reset_identity()

    def _prepare_gpf(self, ipts, seqL, batch_size):
        raw = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
        compact = _select_channels(raw, self.input_channel_indices, "input_channel_indices")
        compact, temporal_mask, lengths = _unpack_temporal_batch(compact, seqL, batch_size)
        return self.norm(compact), temporal_mask, lengths

    def _camera_targets(self, views, device):
        return _camera_targets_from_metadata(views, device, self.camera_num_classes, self.camera_label_map, self.camera_label_parse_mode, self.camera_index_offset)

    def _uniform_belief(self, b, t, device, dtype):
        return torch.full((b, t, self.num_bins), 1.0 / self.num_bins, device=device, dtype=dtype)

    def _default_action(self, b, t, device, dtype, temporal_mask):
        probs = torch.zeros(b, t, 4, device=device, dtype=dtype)
        probs[..., 0] = 1.0
        probs = probs * temporal_mask.unsqueeze(-1).to(dtype)
        return {
            "action_logits": probs.clamp_min(1e-6).log(),
            "action_probs": probs,
            "action_pseudo": probs.detach(),
            "action_cues": torch.zeros(b, t, 12, device=device, dtype=dtype),
            "identity_weight": probs[..., 0],
            "tracking_scale": 0.75 * probs[..., 0],
            "observation_scale": probs[..., 0],
            "transport_gate": probs[..., 0],
            "memory_write_gate": probs[..., 0],
        }

    def _visual_summary(self, gpf, native, refined, primitives, belief, action, memory_aux, transport_aux):
        result = {"image/gpf_mask": _visual_flatten_nt(gpf[:, 0:1], self.visualize_max_frames, normalize=False)}
        if not self.visualize:
            return result
        result["image/gpf_motion_reliability"] = _visual_flatten_nt(gpf[:, 8:9], self.visualize_max_frames)
        result["image/gpf_residual_abs"] = _visual_flatten_nt(gpf[:, 9:10].abs(), self.visualize_max_frames)
        result["image/native_energy"] = _visual_flatten_nt(native.abs().mean(dim=1, keepdim=True), self.visualize_max_frames)
        result["image/refined_energy"] = _visual_flatten_nt(refined.abs().mean(dim=1, keepdim=True), self.visualize_max_frames)
        pmap = primitives["primitive_assign"].max(dim=2).values.unsqueeze(1)
        result["image/primitive_max_assignment"] = _visual_flatten_nt(pmap, self.visualize_max_frames)
        result["scalar/belief_entropy_mean"] = normalized_entropy(belief).mean().detach()
        result["scalar/action_walk_mean"] = action["action_probs"][..., 0].mean().detach()
        result["scalar/action_turn_mean"] = action["action_probs"][..., 1].mean().detach()
        result["scalar/action_interaction_mean"] = action["action_probs"][..., 2].mean().detach()
        result["scalar/action_uncertain_mean"] = action["action_probs"][..., 3].mean().detach()
        if memory_aux:
            result["scalar/memory_valid_slots_mean"] = memory_aux["memory_final_valid"].float().sum(dim=-1).mean().detach()
        if transport_aux:
            result["scalar/transport_gate_mean"] = transport_aux["transport_gate"].mean().detach()
        return result

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        batch_size = labs.size(0) if torch.is_tensor(labs) else len(labs)
        gpf, temporal_mask, lengths = self._prepare_gpf(ipts, seqL, batch_size)
        geo = _select_channels(gpf, self.geo_indices, "geo_indices")
        motion = _select_channels(gpf, self.motion_indices, "motion_indices")
        reliability = gpf[:, self.reliability_index:self.reliability_index + 1]
        residual = _select_channels(gpf, self.residual_indices, "residual_indices") if self.residual_indices else None
        foreground = gpf[:, 0:1]
        stem, stem_aux = self.stem(geo, motion, reliability, residual, foreground)
        native = self.backbone(stem)
        desc = self.descriptor(native, reliability, temporal_mask)

        prediction = self.belief_filter.predict(desc, temporal_mask)
        if not self.enable_relative_view_transition:
            identity = torch.zeros_like(prediction["transition_posterior"])
            identity[..., 0] = temporal_mask.to(identity.dtype)
            prediction["transition_posterior"] = identity
            if prediction["direct2_transition_posterior"].numel() > 0:
                direct2 = torch.zeros_like(prediction["direct2_transition_posterior"])
                direct2[..., 0] = temporal_mask[:, 2:].to(direct2.dtype)
                prediction["direct2_transition_posterior"] = direct2
        action = self.action_gate(gpf, desc, prediction["observation_posterior"], prediction["transition_posterior"], temporal_mask) if self.enable_action_gate else self._default_action(batch_size, gpf.size(2), gpf.device, gpf.dtype, temporal_mask)
        camera_labels, camera_valid = self._camera_targets(vies, gpf.device)
        prior = self.camera_prior(camera_labels, camera_valid) if self.camera_prior is not None else None
        if self.enable_circular_belief_filter:
            belief = self.belief_filter.filter(prediction["observation_posterior"], prediction["transition_posterior"], temporal_mask, prior=prior, observation_scale=action["observation_scale"], transition_scale=action["tracking_scale"])
        else:
            belief = prediction["observation_posterior"] if self.enable_framewise_view_belief else self._uniform_belief(batch_size, gpf.size(2), gpf.device, gpf.dtype)
        if not self.enable_framewise_view_belief:
            belief = self._uniform_belief(batch_size, gpf.size(2), gpf.device, gpf.dtype) * temporal_mask.unsqueeze(-1).to(gpf.dtype)

        primitives = self.primitive_encoder(native, reliability, foreground, temporal_mask)
        if self.enable_residual_primitive_transport:
            canonical, transport_aux = self.transport(native, desc, belief, action["transport_gate"], primitives, temporal_mask)
        else:
            canonical, transport_aux = native, {}
        if self.enable_direction_key_memory:
            refined, memory_aux = self.memory(canonical, desc, belief, primitives, action, reliability, temporal_mask)
        else:
            refined, memory_aux = canonical, {}
        if self.camera_adapter is not None:
            refined = self.camera_adapter(refined, camera_labels, camera_valid)

        pooled = self.temporal_pool(refined, temporal_mask, action["identity_weight"], reliability)
        native_pooled = self.temporal_pool(native, temporal_mask, action["identity_weight"], reliability)
        canonical_pooled = self.temporal_pool(canonical, temporal_mask, action["identity_weight"], reliability)
        feat = self.HPP(pooled)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_2 if self.inference_use_emb2 else embed_1

        training_feat = {
            "triplet": {"embeddings": embed_1, "labels": labs},
            "softmax": {"logits": logits, "labels": labs},
        }
        if self.enable_belief_cycle_loss:
            training_feat["belief_cycle"] = {
                "transition_posterior": prediction["transition_posterior"],
                "direct2_transition_posterior": prediction["direct2_transition_posterior"],
                "temporal_mask": temporal_mask,
            }
        if self.enable_belief_regularization:
            training_feat["belief_reg"] = {
                "observation_posterior": prediction["observation_posterior"],
                "transition_posterior": prediction["transition_posterior"],
                "belief": belief,
                "action_probs": action["action_probs"],
                "temporal_mask": temporal_mask,
            }
        if self.enable_action_selfsup_loss and self.enable_action_gate:
            training_feat["action_selfsup"] = {"action_probs": action["action_probs"], "action_pseudo": action["action_pseudo"], "temporal_mask": temporal_mask}
        if self.enable_internal_flip_consistency:
            flip_desc = self.descriptor(torch.flip(native, dims=[-1]), reliability, temporal_mask)
            flip_prediction = self.belief_filter.predict(flip_desc, temporal_mask)
            training_feat["belief_flip"] = {"transition_posterior": prediction["transition_posterior"], "flip_transition_posterior": flip_prediction["transition_posterior"], "temporal_mask": temporal_mask}
        if self.enable_memory_regularization and memory_aux:
            training_feat["memory_reg"] = {
                "slot_direction": memory_aux["memory_final_direction"],
                "slot_quality": memory_aux["memory_final_quality"],
                "slot_usage": memory_aux["memory_final_usage"],
                "slot_valid": memory_aux["memory_final_valid"],
                "write_gate": memory_aux["memory_write_gate"],
                "temporal_mask": temporal_mask,
            }
        if self.enable_transport_regularization and transport_aux:
            training_feat["transport_reg"] = {
                "primitive_assign": primitives["primitive_assign"],
                "primitive_cov": primitives["primitive_cov"],
                "transport_offsets": transport_aux["transport_offsets"],
                "transport_affine_delta": transport_aux["transport_affine_delta"],
                "transport_gate": transport_aux["transport_gate"],
                "temporal_mask": temporal_mask,
            }
        if self.enable_canonical_consistency:
            training_feat["canonical_consistency"] = {
                "native": native_pooled.mean(dim=(-1, -2)),
                "canonical": canonical_pooled.mean(dim=(-1, -2)),
                "confidence": (1.0 - normalized_entropy(belief)).mean(dim=1),
            }
        if self.camera_adversary is not None:
            identity_global = embed_1.mean(dim=2)
            camera_logits = self.camera_adversary(grad_reverse(identity_global, self.camera_adversary_grl_strength))
            training_feat["camera_adv"] = {"logits": camera_logits, "labels": camera_labels, "valid_mask": camera_valid}

        visual_summary = self._visual_summary(gpf, native, refined, primitives, belief, action, memory_aux, transport_aux)
        inference_feat = {"embeddings": embed}
        if self.export_belief_state:
            inference_feat.update({
                "view_belief": belief,
                "view_uncertainty": normalized_entropy(belief),
                "action_probs": action["action_probs"],
            })
        return {"training_feat": training_feat, "visual_summary": visual_summary, "inference_feat": inference_feat}
