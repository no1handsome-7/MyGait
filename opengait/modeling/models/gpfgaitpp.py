# -*- coding: utf-8 -*-
"""
GPF++ Gait model for OpenGait.

Drop this file into:
    OpenGait/opengait/modeling/models/gpf_gait.py
Then make sure opengait/modeling/models/__init__.py includes:
    from .gpf_gait import GPFGait

Expected files from pretreatment_gpf.py:
    000-gpf.pkl     : [T, 28, H, W]
    001-sils.pkl    : [T, 1, H, W] or [T, H, W]
    002-phase.pkl   : [T, 5] = [sin_phi, cos_phi, cycle_id, phase_quality, period]
    003-parts.pkl   : [T, K, H, W]
    004-quality.pkl : diagnostic metadata, not consumed by default

Design choices:
    1) Phase is NOT used as strong FiLM by default. It can guide temporal pooling
       only when the config asks for phase-aware pooling.
    2) The stem is grouped by gait primitive type: geometry / motion / frequency /
       residual-affinity. Texture-risk inverse and motion reliability act as gates.
    3) Temporal pooling defaults to evidence pooling: max + mean + reliability +
       top-k reliable aggregation, fused with a light 1x1 conv.
    4) Part maps modulate the pooled feature map before HPP, preserving the normal
       GaitBase/HPP/SeparateFCs/BNNeck output contract.
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
    """Convert common OpenGait layouts to [N, C, T, H, W]."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 4:
        return x.unsqueeze(1).float().contiguous()
    if x.dim() != 5:
        raise ValueError(f"Expected 4D/5D tensor, got {tuple(x.shape)}")
    if expected_channels is not None and x.size(1) == expected_channels:
        return x.float().contiguous()
    if expected_channels is not None and x.size(2) == expected_channels:
        return x.transpose(1, 2).float().contiguous()
    # Heuristic fallback: [N,C,T,H,W] usually has small C, while [N,T,C,H,W]
    # usually has larger T at dim=1.
    if x.size(1) <= 64 and x.size(2) > 8:
        return x.float().contiguous()
    return x.transpose(1, 2).float().contiguous()


def _to_nt_phase(x: torch.Tensor) -> torch.Tensor:
    """Convert phase metadata to [N, T, D]. D can be 2 or 5; first two dims are sin/cos."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected phase tensor with 3 dims, got {tuple(x.shape)}")
    if x.size(-1) in (2, 5):
        return x.float().contiguous()
    if x.size(1) in (2, 5):
        return x.transpose(1, 2).float().contiguous()
    raise ValueError(f"Cannot infer phase layout from {tuple(x.shape)}")


def _select_channels(x: torch.Tensor, indices: Sequence[int], name: str) -> torch.Tensor:
    indices = list(map(int, indices))
    if len(indices) == 0:
        raise ValueError(f"{name} is empty; check model_cfg.")
    if min(indices) < 0 or max(indices) >= x.size(1):
        raise ValueError(f"{name}={indices} is invalid for input with {x.size(1)} channels.")
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(1, idx)


def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    n, c, t, h, w = x.shape
    y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
    y = module(y)
    _, c2, h2, w2 = y.shape
    return y.reshape(n, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


def _match_nt_time(x: Optional[torch.Tensor], target_t: int, pad_value: str = "replicate"):
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
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act=False)
        self.down = ConvBNAct(in_channels, out_channels, 1, stride, padding=0, act=False) if (in_channels != out_channels or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.down(x))


class FrameBackbone(nn.Module):
    """GaitBase-style 2D frame encoder; input/output are [N, C, T, H, W]."""
    def __init__(self, in_channels: int, channels=(64, 128, 256, 512), strides=(1, 2, 1, 1), blocks_per_stage=(1, 1, 1, 1)):
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


class GPFPPChannelNorm(nn.Module):
    """Conservative per-channel sanitization for the 28-channel GPF++."""
    POSITIVE = {0, 1, 6, 7, 9, 12, 15, 17, 18, 19, 22, 23, 25}
    SIGNED = {2, 3, 4, 5, 8, 10, 11, 13, 14, 16, 20, 21, 24, 26, 27}

    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = bool(enabled)

    def forward(self, x):
        x = x.float()
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if not self.enabled:
            return x
        y = x.clone()
        for idx in range(y.size(1)):
            if idx in self.POSITIVE:
                y[:, idx] = y[:, idx].clamp(0.0, 1.0)
            elif idx in self.SIGNED:
                y[:, idx] = y[:, idx].clamp(-1.0, 1.0)
            else:
                y[:, idx] = y[:, idx].clamp(-1.0, 1.0)
        return y


class PhaseFiLM(nn.Module):
    """Optional diagnostic FiLM. Keep disabled for the main GPF++ baseline."""
    def __init__(self, channels: int, hidden_channels: int = 32, strength: float = 0.25):
        super().__init__()
        self.strength = float(strength)
        self.mlp = nn.Sequential(nn.Linear(2, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, channels * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]):
        if phase is None:
            return x
        phase = phase[..., :2]
        phase = _match_nt_time(phase, x.size(2))
        gb = self.mlp(phase)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.strength * gamma) + self.strength * beta


class FrameSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class GPFPPStem(nn.Module):
    """Grouped primitive stem: geometry / motion / frequency / residual-affinity."""
    def __init__(
        self,
        group_channels: Dict[str, int],
        out_channels: int = 48,
        branch_channels: int = 16,
        affinity_branch_channels: int = 12,
        use_gates: bool = True,
        gate_strength: float = 0.75,
        primitive_dropout: float = 0.0,
    ):
        super().__init__()
        self.group_names = [k for k, v in group_channels.items() if int(v) > 0]
        if len(self.group_names) == 0:
            raise ValueError("At least one GPF++ primitive group must be non-empty.")
        self.use_gates = bool(use_gates)
        self.gate_strength = float(gate_strength)
        self.primitive_dropout = float(primitive_dropout)

        self.branches = nn.ModuleDict()
        self.branch_dims = {}
        for name in self.group_names:
            c_in = int(group_channels[name])
            c_out = affinity_branch_channels if name == "affinity" else branch_channels
            self.branch_dims[name] = c_out
            self.branches[name] = nn.Sequential(ConvBNAct(c_in, c_out, 3), ConvBNAct(c_out, c_out, 3))

        self.gate_nets = nn.ModuleDict()
        for name in ["motion", "frequency", "affinity"]:
            if name in self.group_names:
                c_out = self.branch_dims[name]
                self.gate_nets[name] = nn.Sequential(ConvBNAct(1, c_out, 3), nn.Conv2d(c_out, c_out, 1))

        fuse_in = sum(self.branch_dims.values())
        self.fuse = nn.Sequential(ConvBNAct(fuse_in, out_channels, 1, padding=0), ConvBNAct(out_channels, out_channels, 3), FrameSE(out_channels))

    def _drop_branch(self, x):
        if self.training and self.primitive_dropout > 0 and torch.rand((), device=x.device) < self.primitive_dropout:
            return x * 0.0
        return x

    def _soft_gate(self, logits):
        g = torch.sigmoid(logits)
        return (1.0 - self.gate_strength) + self.gate_strength * g

    def forward(self, groups: Dict[str, torch.Tensor], gates: Dict[str, Optional[torch.Tensor]], foreground: Optional[torch.Tensor] = None):
        feats = []
        aux = {}
        for name in self.group_names:
            z = _apply_framewise(self.branches[name], groups[name])
            if self.use_gates and name in self.gate_nets and gates.get(name, None) is not None:
                gate = _apply_framewise(self.gate_nets[name], gates[name])
                gate = self._soft_gate(gate)
                if foreground is not None:
                    gate = gate * (0.5 + 0.5 * foreground.clamp(0, 1))
                z = z * gate
                aux[f"{name}_gate"] = gate
            if name != "geometry":
                z = self._drop_branch(z)
            feats.append(z)
        fused = torch.cat(feats, dim=1)
        fused = _apply_framewise(self.fuse, fused)
        return fused, aux


class TemporalEvidencePool(nn.Module):
    """Max/mean/reliability/top-k evidence pooling with packed seqL support."""
    def __init__(self, channels: int, pool_type: str = "evidence", topk_ratio: float = 0.5, phase_bins: int = 8, phase_kappa: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.pool_type = str(pool_type)
        self.topk_ratio = float(topk_ratio)
        self.phase_bins = int(phase_bins)
        self.phase_kappa = float(phase_kappa)
        self.eps = float(eps)
        self.evidence_fuse = nn.Sequential(nn.Conv2d(channels * 4, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def _flatten_seqL(self, seqL):
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
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                v = v[0]
            lens.append(int(v))
        if len(lens) >= batch_size:
            return [max(1, min(int(v), max_len)) for v in lens[:batch_size]]
        base = max(1, max_len // max(batch_size, 1))
        fallback = [base for _ in range(batch_size)]
        fallback[-1] = max(1, max_len - base * (batch_size - 1))
        return fallback

    @staticmethod
    def _phase_to_nt(phase: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if phase is None:
            return None
        phase = _to_nt_phase(phase)
        return phase

    @staticmethod
    def _scalar_to_nt(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 2:
            return x.float().contiguous()
        if x.dim() == 3:
            if x.size(1) == 1:
                return x[:, 0].float().contiguous()
            if x.size(-1) == 1:
                return x[..., 0].float().contiguous()
        if x.dim() == 5:
            return x.float().mean(dim=(1, 3, 4)).contiguous()
        raise ValueError(f"Unsupported temporal scalar shape: {tuple(x.shape)}")

    def _match_aux(self, phase, reliability, target_t):
        phase = self._phase_to_nt(phase)
        reliability = self._scalar_to_nt(reliability)
        if phase is not None:
            phase = _match_nt_time(phase, target_t)
        if reliability is not None:
            reliability = _match_nt_time(reliability.unsqueeze(-1), target_t).squeeze(-1)
        return phase, reliability

    def _split_by_seqL(self, x, phase=None, reliability=None, seqL=None, batch_size=None):
        n, _, max_len, _, _ = x.shape
        batch_size = int(batch_size if batch_size is not None else n)
        phase, reliability = self._match_aux(phase, reliability, max_len)
        if n == batch_size:
            lens = self._seqL_to_int_list(seqL, batch_size, max_len)
            chunks = []
            for i, l in enumerate(lens):
                pi = phase[i:i + 1, :l] if phase is not None and phase.size(0) == n else (phase[:1, :l] if phase is not None else None)
                ri = reliability[i:i + 1, :l] if reliability is not None and reliability.size(0) == n else (reliability[:1, :l] if reliability is not None else None)
                chunks.append((x[i:i + 1, :, :l], pi, ri))
            return chunks
        if n == 1 and batch_size > 1:
            lens = self._seqL_to_int_list(seqL, batch_size, max_len)
            chunks, start = [], 0
            for i, l in enumerate(lens):
                end = min(start + int(l), max_len)
                if end <= start:
                    end = min(start + 1, max_len)
                pi = None if phase is None else (phase[:, start:end] if phase.size(0) == 1 else phase[i:i + 1, :end - start])
                ri = None if reliability is None else (reliability[:, start:end] if reliability.size(0) == 1 else reliability[i:i + 1, :end - start])
                chunks.append((x[:, :, start:end], pi, ri))
                start = end
            return chunks
        raise RuntimeError(f"Cannot align temporal batch: x={tuple(x.shape)}, batch_size={batch_size}, seqL={seqL}")

    def _weighted_mean(self, x, reliability):
        if reliability is None:
            return x.mean(dim=2)
        w = reliability.clamp(0, 1).view(reliability.size(0), 1, reliability.size(1), 1, 1)
        return (x * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

    def _topk_reliable(self, x, reliability):
        if reliability is None or x.size(2) <= 1:
            return x.max(dim=2)[0]
        t = x.size(2)
        k = max(1, int(round(t * self.topk_ratio)))
        idx = torch.topk(reliability.clamp(0, 1), k=k, dim=1, largest=True)[1]
        gather_idx = idx.view(idx.size(0), 1, k, 1, 1).expand(-1, x.size(1), -1, x.size(3), x.size(4))
        return x.gather(dim=2, index=gather_idx).mean(dim=2)

    def _phase_bin_pool(self, x, phase, reliability):
        if phase is None:
            return self._weighted_mean(x, reliability)
        phase2 = phase[..., :2]
        angle = torch.atan2(phase2[..., 0], phase2[..., 1])
        centers = torch.linspace(-math.pi, math.pi, steps=self.phase_bins + 1, device=x.device, dtype=x.dtype)[:-1]
        weights = torch.exp(self.phase_kappa * torch.cos(angle.unsqueeze(-1) - centers))
        if reliability is not None:
            weights = weights * reliability.clamp(0, 1).unsqueeze(-1)
        w = weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        z = (x.unsqueeze(3) * w).sum(dim=2) / (w.sum(dim=2) + self.eps)
        return z.mean(dim=2)

    def _pool_single(self, x, phase=None, reliability=None):
        if self.pool_type == "avg":
            return x.mean(dim=2)
        if self.pool_type == "max" or x.size(2) == 1:
            return x.max(dim=2)[0]
        if self.pool_type == "reliability":
            return self._weighted_mean(x, reliability)
        if self.pool_type == "topk_reliability":
            return self._topk_reliable(x, reliability)
        if self.pool_type == "phase_bin":
            return self._phase_bin_pool(x, phase, reliability=None)
        if self.pool_type == "reliability_phase_bin":
            return self._phase_bin_pool(x, phase, reliability=reliability)
        # evidence: robust default for noisy GPF++ primitives.
        z_max = x.max(dim=2)[0]
        z_avg = x.mean(dim=2)
        z_rel = self._weighted_mean(x, reliability)
        z_top = self._topk_reliable(x, reliability)
        return self.evidence_fuse(torch.cat([z_max, z_avg, z_rel, z_top], dim=1))

    def forward(self, x, seqL=None, phase=None, reliability=None, batch_size=None):
        if x.dim() != 5:
            raise RuntimeError(f"TemporalEvidencePool expects [N,C,T,H,W], got {tuple(x.shape)}")
        batch_size = int(batch_size if batch_size is not None else x.size(0))
        if x.size(0) == batch_size and self.pool_type == "max":
            return x.max(dim=2)[0]
        chunks = self._split_by_seqL(x, phase=phase, reliability=reliability, seqL=seqL, batch_size=batch_size)
        out = torch.cat([self._pool_single(xi, pi, ri) for xi, pi, ri in chunks], dim=0)
        if out.size(0) != batch_size:
            raise RuntimeError(f"Temporal pool batch mismatch: got {out.size(0)}, expected {batch_size}.")
        return out


class PartSpatialGate(nn.Module):
    """Use body-coordinate part maps as a soft spatial reliability gate before HPP."""
    def __init__(self, num_parts: int, channels: int, strength: float = 0.35):
        super().__init__()
        hidden = min(64, max(16, channels // 4))
        self.num_parts = int(num_parts)
        self.strength = float(strength)
        self.net = nn.Sequential(ConvBNAct(num_parts, hidden, 3), nn.Conv2d(hidden, channels, 1))

    def forward(self, feat: torch.Tensor, parts: Optional[torch.Tensor]):
        if parts is None:
            return feat, {}
        parts = _to_ncthw(parts, expected_channels=self.num_parts).to(feat.device).float()
        p = parts.mean(dim=2)  # [N,K,H,W]
        if p.shape[-2:] != feat.shape[-2:]:
            p = F.interpolate(p, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.net(p))
        gate = (1.0 - self.strength) + self.strength * gate
        return feat * gate, {"part_gate": gate.mean(dim=1, keepdim=True), "parts_pooled": p.mean(dim=1, keepdim=True)}


class DirectGPFStem(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(in_channels, out_channels, 3), ConvBNAct(out_channels, out_channels, 3))

    def forward(self, x):
        return _apply_framewise(self.net, x), {}


def _visual_flatten_nt(x: Optional[torch.Tensor], max_frames: int = 8, normalize: bool = True) -> Optional[torch.Tensor]:
    if x is None:
        return None
    x = x.detach().float()
    if x.dim() == 4:
        if normalize:
            x = _visual_minmax(x)
        return x
    if x.dim() != 5:
        return None
    t = x.size(2)
    idx = torch.linspace(0, t - 1, steps=min(t, int(max_frames)), device=x.device).round().long() if t > max_frames else torch.arange(t, device=x.device)
    x = x.index_select(2, idx)
    n, c, tt, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(n * tt, c, h, w).contiguous()
    if normalize:
        x = _visual_minmax(x)
    return x


def _visual_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    dims = tuple(range(2, x.dim()))
    mn = x.amin(dim=dims, keepdim=True)
    mx = x.amax(dim=dims, keepdim=True)
    return (x - mn) / (mx - mn + eps)


class GPFGaitPP(BaseModel):
    """GPF++ model wrapper compatible with OpenGait's BaseModel API."""

    def build_network(self, model_cfg: Dict):
        self.model_cfg = model_cfg
        self.variant = model_cfg.get("variant", "gpfpp_structured")
        self.input_index = int(model_cfg.get("input_index", 0))
        self.phase_input_index = model_cfg.get("phase_input_index", None)
        self.parts_input_index = model_cfg.get("parts_input_index", None)
        # For ablations: do not silently recover phase from GPF++ channels unless explicitly enabled.
        self.use_internal_phase = bool(model_cfg.get("use_internal_phase", True))
        self.source_channels = int(model_cfg.get("source_channels", 28))
        self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))
        self.visualize = bool(model_cfg.get("visualize", False))
        self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))

        self.norm = GPFPPChannelNorm(enabled=bool(model_cfg.get("enable_channel_norm", True)))
        stem_out = int(model_cfg.get("stem_out_channels", 48))

        self.geo_indices = list(map(int, model_cfg.get("geo_indices", list(range(0, 10)))))
        self.motion_indices = list(map(int, model_cfg.get("motion_indices", list(range(10, 20)))))
        self.freq_indices = list(map(int, model_cfg.get("frequency_indices", list(range(20, 24)))))
        self.affinity_indices = list(map(int, model_cfg.get("affinity_indices", list(range(24, 28)))))
        self.phase_indices = list(map(int, model_cfg.get("phase_indices", [20, 21])))

        # Empty primitive groups are allowed so configs can do clean channel ablations
        # (e.g. core geometry+motion+residual without phase/frequency).
        self.group_indices = {
            "geometry": self.geo_indices,
            "motion": self.motion_indices,
            "frequency": self.freq_indices,
            "affinity": self.affinity_indices,
        }

        self.reliability_index = model_cfg.get("reliability_index", 19)
        self.periodic_index = model_cfg.get("periodic_index", 18)
        self.texture_inverse_index = model_cfg.get("texture_inverse_index", 25)

        if self.variant == "sil":
            self.adapter_type = "sil"
            self.adapter = DirectGPFStem(1, stem_out)
        elif self.variant in ("gpfpp_direct", "gpf_direct"):
            self.adapter_type = "direct"
            self.channel_indices = list(map(int, model_cfg.get("channel_indices", list(range(self.source_channels)))))
            self.adapter = DirectGPFStem(len(self.channel_indices), stem_out)
        elif self.variant in ("gpfpp_structured", "gpf_structured", "gpf_grouped"):
            self.adapter_type = "gpfpp"
            group_channels = {
                "geometry": len(self.geo_indices),
                "motion": len(self.motion_indices),
                "frequency": len(self.freq_indices),
                "affinity": len(self.affinity_indices),
            }
            self.adapter = GPFPPStem(
                group_channels=group_channels,
                out_channels=stem_out,
                branch_channels=int(model_cfg.get("grouped_branch_channels", 16)),
                affinity_branch_channels=int(model_cfg.get("affinity_branch_channels", 12)),
                use_gates=bool(model_cfg.get("use_primitive_gates", True)),
                gate_strength=float(model_cfg.get("primitive_gate_strength", 0.75)),
                primitive_dropout=float(model_cfg.get("primitive_dropout", 0.05)),
            )
        else:
            raise ValueError(f"Unknown GPFGait variant: {self.variant}")

        self.use_phase_film = bool(model_cfg.get("use_phase_film", False))
        self.phase_film = PhaseFiLM(stem_out, hidden_channels=int(model_cfg.get("phase_hidden_channels", 32)), strength=float(model_cfg.get("phase_film_strength", 0.25))) if self.use_phase_film else nn.Identity()

        self.backbone = FrameBackbone(
            stem_out,
            channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
            strides=model_cfg.get("backbone_strides", [1, 2, 1, 1]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
        )
        backbone_out = self.backbone.out_channels

        self.TP = TemporalEvidencePool(
            backbone_out,
            pool_type=model_cfg.get("temporal_pool", "evidence"),
            topk_ratio=float(model_cfg.get("temporal_topk_ratio", 0.5)),
            phase_bins=int(model_cfg.get("phase_bins", 8)),
            phase_kappa=float(model_cfg.get("phase_kappa", 4.0)),
        )

        self.use_part_gate = bool(model_cfg.get("use_part_gate", True))
        self.num_parts = int(model_cfg.get("num_parts", 8))
        self.part_gate = PartSpatialGate(self.num_parts, backbone_out, strength=float(model_cfg.get("part_gate_strength", 0.35))) if self.use_part_gate else None

        bin_num = model_cfg.get("bin_num", [32])
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        parts_num = int(sum(bin_num))
        emb_channels = int(model_cfg.get("emb_channels", 256))
        class_num = int(model_cfg["class_num"])
        fc_cfg = model_cfg.get("SeparateFCs", {})
        bn_cfg = model_cfg.get("SeparateBNNecks", {})
        self.FCs = SeparateFCs(
            parts_num=fc_cfg.get("parts_num", parts_num),
            in_channels=fc_cfg.get("in_channels", backbone_out),
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
        if self.phase_input_index is not None:
            idx = int(self.phase_input_index)
            if 0 <= idx < len(ipts):
                try:
                    return _to_nt_phase(ipts[idx]).to(gpf.device if gpf is not None else ipts[idx].device)
                except Exception:
                    pass
        # Optional fallback: use phase-like channels inside 000-gpf.pkl only when the
        # ablation explicitly enables it. This prevents hidden phase leakage in no-phase runs.
        if self.use_internal_phase and gpf is not None and len(self.phase_indices) >= 2:
            idx = self.phase_indices[:2]
            if min(idx) >= 0 and max(idx) < gpf.size(1):
                phase = gpf[:, idx].mean(dim=(-1, -2)).transpose(1, 2).contiguous()
                return phase
        return None

    def _get_parts(self, ipts):
        if self.parts_input_index is None:
            return None
        idx = int(self.parts_input_index)
        if not (0 <= idx < len(ipts)):
            return None
        return ipts[idx]

    def _scalar_from_gpf(self, gpf: Optional[torch.Tensor], idx, default: Optional[torch.Tensor] = None):
        if gpf is None or idx is None:
            return default
        idx = int(idx)
        if idx < 0 or idx >= gpf.size(1):
            return default
        return gpf[:, idx:idx + 1]

    def _frame_scalar_from_gpf(self, gpf: Optional[torch.Tensor], idx):
        x = self._scalar_from_gpf(gpf, idx)
        if x is None:
            return None
        return x.mean(dim=(-1, -2)).clamp(0, 1)

    def _build_visual_summary(self, gpf, stem_feat, backbone_feat, temporal_feat, aux):
        visual = {}
        if gpf is not None:
            visual["image/gpfpp_mask"] = _visual_flatten_nt(gpf[:, 0:1], self.visualize_max_frames, normalize=False)
        if not self.visualize:
            return visual
        if gpf is not None:
            for idx, name in [(1, "edge"), (12, "flow_mag"), (18, "periodic"), (19, "reliability"), (24, "stable_residual_abs"), (25, "texture_inverse")]:
                if idx < gpf.size(1):
                    x = gpf[:, idx:idx + 1].abs() if idx == 24 else gpf[:, idx:idx + 1]
                    visual[f"image/gpfpp_{idx:02d}_{name}"] = _visual_flatten_nt(x, self.visualize_max_frames, normalize=True)
        if aux:
            for name, value in aux.items():
                if value is not None and torch.is_tensor(value):
                    visual[f"image/{name}"] = _visual_flatten_nt(value, self.visualize_max_frames, normalize=True)
        if temporal_feat is not None:
            energy = temporal_feat.detach().abs().mean(dim=1, keepdim=True)
            visual["image/pooled_feature_energy"] = _visual_flatten_nt(energy, self.visualize_max_frames, normalize=True)
        return visual

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        local_bs = labs.size(0) if torch.is_tensor(labs) else len(labs)
        gpf = None
        phase = None
        parts = None
        aux = {}

        if self.adapter_type == "sil":
            x_in = _to_ncthw(ipts[self.input_index], expected_channels=1)
            x, stem_aux = self.adapter(x_in)
            gpf_for_vis = x_in
        else:
            gpf = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
            gpf = self.norm(gpf)
            gpf_for_vis = gpf
            phase = self._get_phase(ipts, gpf)
            parts = self._get_parts(ipts)

            if self.adapter_type == "direct":
                x_in = _select_channels(gpf, self.channel_indices, "channel_indices")
                x, stem_aux = self.adapter(x_in)
            else:
                active_names = getattr(self.adapter, "group_names", list(self.group_indices.keys()))
                groups = {
                    name: _select_channels(gpf, self.group_indices[name], f"{name}_indices")
                    for name in active_names
                }
                reliability = self._scalar_from_gpf(gpf, self.reliability_index)
                periodic = self._scalar_from_gpf(gpf, self.periodic_index, reliability)
                texture_inv = self._scalar_from_gpf(gpf, self.texture_inverse_index, reliability)
                gates = {"motion": reliability, "frequency": periodic, "affinity": texture_inv}
                foreground = gpf[:, 0:1] if gpf.size(1) > 0 else None
                x, stem_aux = self.adapter(groups, gates=gates, foreground=foreground)
        aux.update(stem_aux)

        if self.use_phase_film:
            x = self.phase_film(x, phase)

        stem_feat = x
        x = self.backbone(x)
        backbone_feat = x
        reliability_scalar = self._frame_scalar_from_gpf(gpf, self.reliability_index)
        x = self.TP(x, seqL=seqL, phase=phase, reliability=reliability_scalar, batch_size=local_bs)
        temporal_feat = x

        if self.part_gate is not None:
            x, part_aux = self.part_gate(x, parts)
            aux.update(part_aux)

        feat = self.HPP(x)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_2 if self.inference_use_emb2 else embed_1

        if embed_1.size(0) != local_bs:
            raise RuntimeError(f"Embedding batch mismatch: {tuple(embed_1.shape)} vs labels batch {local_bs}")
        if logits.size(0) != local_bs:
            raise RuntimeError(f"Logits batch mismatch: {tuple(logits.shape)} vs labels batch {local_bs}")

        return {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": self._build_visual_summary(gpf_for_vis, stem_feat, backbone_feat, temporal_feat, aux),
            "inference_feat": {"embeddings": embed},
        }




# # -*- coding: utf-8 -*-
# """
# GPF++ Gait model for OpenGait.

# Drop this file into:
#     OpenGait/opengait/modeling/models/gpf_gait.py
# Then make sure opengait/modeling/models/__init__.py includes:
#     from .gpf_gait import GPFGait

# Expected files from pretreatment_gpf.py:
#     000-gpf.pkl     : [T, 28, H, W]
#     001-sils.pkl    : [T, 1, H, W] or [T, H, W]
#     002-phase.pkl   : [T, 5] = [sin_phi, cos_phi, cycle_id, phase_quality, period]
#     003-parts.pkl   : [T, K, H, W]
#     004-quality.pkl : diagnostic metadata, not consumed by default

# Design choices:
#     1) Phase is NOT used as strong FiLM by default. It can guide temporal pooling
#        only when the config asks for phase-aware pooling.
#     2) The stem is grouped by gait primitive type: geometry / motion / frequency /
#        residual-affinity. Texture-risk inverse and motion reliability act as gates.
#     3) Temporal pooling defaults to evidence pooling: max + mean + reliability +
#        top-k reliable aggregation, fused with a light 1x1 conv.
#     4) Part maps modulate the pooled feature map before HPP, preserving the normal
#        GaitBase/HPP/SeparateFCs/BNNeck output contract.
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
#     """Convert common OpenGait layouts to [N, C, T, H, W]."""
#     if not torch.is_tensor(x):
#         x = torch.as_tensor(x)
#     if x.dim() == 4:
#         return x.unsqueeze(1).float().contiguous()
#     if x.dim() != 5:
#         raise ValueError(f"Expected 4D/5D tensor, got {tuple(x.shape)}")
#     if expected_channels is not None and x.size(1) == expected_channels:
#         return x.float().contiguous()
#     if expected_channels is not None and x.size(2) == expected_channels:
#         return x.transpose(1, 2).float().contiguous()
#     # Heuristic fallback: [N,C,T,H,W] usually has small C, while [N,T,C,H,W]
#     # usually has larger T at dim=1.
#     if x.size(1) <= 64 and x.size(2) > 8:
#         return x.float().contiguous()
#     return x.transpose(1, 2).float().contiguous()


# def _to_nt_phase(x: torch.Tensor) -> torch.Tensor:
#     """Convert phase metadata to [N, T, D]. D can be 2 or 5; first two dims are sin/cos."""
#     if not torch.is_tensor(x):
#         x = torch.as_tensor(x)
#     if x.dim() == 2:
#         x = x.unsqueeze(0)
#     if x.dim() != 3:
#         raise ValueError(f"Expected phase tensor with 3 dims, got {tuple(x.shape)}")
#     if x.size(-1) in (2, 5):
#         return x.float().contiguous()
#     if x.size(1) in (2, 5):
#         return x.transpose(1, 2).float().contiguous()
#     raise ValueError(f"Cannot infer phase layout from {tuple(x.shape)}")


# def _select_channels(x: torch.Tensor, indices: Sequence[int], name: str) -> torch.Tensor:
#     indices = list(map(int, indices))
#     if len(indices) == 0:
#         raise ValueError(f"{name} is empty; check model_cfg.")
#     if min(indices) < 0 or max(indices) >= x.size(1):
#         raise ValueError(f"{name}={indices} is invalid for input with {x.size(1)} channels.")
#     idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
#     return x.index_select(1, idx)


# def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
#     n, c, t, h, w = x.shape
#     y = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).contiguous()
#     y = module(y)
#     _, c2, h2, w2 = y.shape
#     return y.reshape(n, t, c2, h2, w2).permute(0, 2, 1, 3, 4).contiguous()


# def _match_nt_time(x: Optional[torch.Tensor], target_t: int, pad_value: str = "replicate"):
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
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride)
#         self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act=False)
#         self.down = ConvBNAct(in_channels, out_channels, 1, stride, padding=0, act=False) if (in_channels != out_channels or stride != 1) else nn.Identity()
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.conv2(self.conv1(x)) + self.down(x))


# class FrameBackbone(nn.Module):
#     """GaitBase-style 2D frame encoder; input/output are [N, C, T, H, W]."""
#     def __init__(self, in_channels: int, channels=(64, 128, 256, 512), strides=(1, 2, 1, 1), blocks_per_stage=(1, 1, 1, 1)):
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


# class GPFPPChannelNorm(nn.Module):
#     """Conservative per-channel sanitization for the 28-channel GPF++."""
#     POSITIVE = {0, 1, 6, 7, 9, 12, 15, 17, 18, 19, 22, 23, 25}
#     SIGNED = {2, 3, 4, 5, 8, 10, 11, 13, 14, 16, 20, 21, 24, 26, 27}

#     def __init__(self, enabled=True):
#         super().__init__()
#         self.enabled = bool(enabled)

#     def forward(self, x):
#         x = x.float()
#         x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
#         if not self.enabled:
#             return x
#         y = x.clone()
#         for idx in range(y.size(1)):
#             if idx in self.POSITIVE:
#                 y[:, idx] = y[:, idx].clamp(0.0, 1.0)
#             elif idx in self.SIGNED:
#                 y[:, idx] = y[:, idx].clamp(-1.0, 1.0)
#             else:
#                 y[:, idx] = y[:, idx].clamp(-1.0, 1.0)
#         return y


# class PhaseFiLM(nn.Module):
#     """Optional diagnostic FiLM. Keep disabled for the main GPF++ baseline."""
#     def __init__(self, channels: int, hidden_channels: int = 32, strength: float = 0.25):
#         super().__init__()
#         self.strength = float(strength)
#         self.mlp = nn.Sequential(nn.Linear(2, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, channels * 2))
#         nn.init.zeros_(self.mlp[-1].weight)
#         nn.init.zeros_(self.mlp[-1].bias)

#     def forward(self, x: torch.Tensor, phase: Optional[torch.Tensor]):
#         if phase is None:
#             return x
#         phase = phase[..., :2]
#         phase = _match_nt_time(phase, x.size(2))
#         gb = self.mlp(phase)
#         gamma, beta = gb.chunk(2, dim=-1)
#         gamma = gamma.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
#         beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
#         return x * (1.0 + self.strength * gamma) + self.strength * beta


# class FrameSE(nn.Module):
#     def __init__(self, channels: int, reduction: int = 8):
#         super().__init__()
#         hidden = max(8, channels // reduction)
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden, channels, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         return x * self.net(x)


# class GPFPPStem(nn.Module):
#     """Grouped primitive stem: geometry / motion / frequency / residual-affinity."""
#     def __init__(
#         self,
#         group_channels: Dict[str, int],
#         out_channels: int = 48,
#         branch_channels: int = 16,
#         affinity_branch_channels: int = 12,
#         use_gates: bool = True,
#         gate_strength: float = 0.75,
#         primitive_dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.group_names = [k for k, v in group_channels.items() if int(v) > 0]
#         if len(self.group_names) == 0:
#             raise ValueError("At least one GPF++ primitive group must be non-empty.")
#         self.use_gates = bool(use_gates)
#         self.gate_strength = float(gate_strength)
#         self.primitive_dropout = float(primitive_dropout)

#         self.branches = nn.ModuleDict()
#         self.branch_dims = {}
#         for name in self.group_names:
#             c_in = int(group_channels[name])
#             c_out = affinity_branch_channels if name == "affinity" else branch_channels
#             self.branch_dims[name] = c_out
#             self.branches[name] = nn.Sequential(ConvBNAct(c_in, c_out, 3), ConvBNAct(c_out, c_out, 3))

#         self.gate_nets = nn.ModuleDict()
#         for name in ["motion", "frequency", "affinity"]:
#             if name in self.group_names:
#                 c_out = self.branch_dims[name]
#                 self.gate_nets[name] = nn.Sequential(ConvBNAct(1, c_out, 3), nn.Conv2d(c_out, c_out, 1))

#         fuse_in = sum(self.branch_dims.values())
#         self.fuse = nn.Sequential(ConvBNAct(fuse_in, out_channels, 1, padding=0), ConvBNAct(out_channels, out_channels, 3), FrameSE(out_channels))

#     def _drop_branch(self, x):
#         if self.training and self.primitive_dropout > 0 and torch.rand((), device=x.device) < self.primitive_dropout:
#             return x * 0.0
#         return x

#     def _soft_gate(self, logits):
#         g = torch.sigmoid(logits)
#         return (1.0 - self.gate_strength) + self.gate_strength * g

#     def forward(self, groups: Dict[str, torch.Tensor], gates: Dict[str, Optional[torch.Tensor]], foreground: Optional[torch.Tensor] = None):
#         feats = []
#         aux = {}
#         for name in self.group_names:
#             z = _apply_framewise(self.branches[name], groups[name])
#             if self.use_gates and name in self.gate_nets and gates.get(name, None) is not None:
#                 gate = _apply_framewise(self.gate_nets[name], gates[name])
#                 gate = self._soft_gate(gate)
#                 if foreground is not None:
#                     gate = gate * (0.5 + 0.5 * foreground.clamp(0, 1))
#                 z = z * gate
#                 aux[f"{name}_gate"] = gate
#             if name != "geometry":
#                 z = self._drop_branch(z)
#             feats.append(z)
#         fused = torch.cat(feats, dim=1)
#         fused = _apply_framewise(self.fuse, fused)
#         return fused, aux


# class TemporalEvidencePool(nn.Module):
#     """Max/mean/reliability/top-k evidence pooling with packed seqL support."""
#     def __init__(self, channels: int, pool_type: str = "evidence", topk_ratio: float = 0.5, phase_bins: int = 8, phase_kappa: float = 4.0, eps: float = 1e-6):
#         super().__init__()
#         self.pool_type = str(pool_type)
#         self.topk_ratio = float(topk_ratio)
#         self.phase_bins = int(phase_bins)
#         self.phase_kappa = float(phase_kappa)
#         self.eps = float(eps)
#         self.evidence_fuse = nn.Sequential(nn.Conv2d(channels * 4, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

#     def _flatten_seqL(self, seqL):
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
#             elif isinstance(v, (list, tuple)) and len(v) > 0:
#                 v = v[0]
#             lens.append(int(v))
#         if len(lens) >= batch_size:
#             return [max(1, min(int(v), max_len)) for v in lens[:batch_size]]
#         base = max(1, max_len // max(batch_size, 1))
#         fallback = [base for _ in range(batch_size)]
#         fallback[-1] = max(1, max_len - base * (batch_size - 1))
#         return fallback

#     @staticmethod
#     def _phase_to_nt(phase: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         if phase is None:
#             return None
#         phase = _to_nt_phase(phase)
#         return phase

#     @staticmethod
#     def _scalar_to_nt(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
#         if x is None:
#             return None
#         if not torch.is_tensor(x):
#             x = torch.as_tensor(x)
#         if x.dim() == 2:
#             return x.float().contiguous()
#         if x.dim() == 3:
#             if x.size(1) == 1:
#                 return x[:, 0].float().contiguous()
#             if x.size(-1) == 1:
#                 return x[..., 0].float().contiguous()
#         if x.dim() == 5:
#             return x.float().mean(dim=(1, 3, 4)).contiguous()
#         raise ValueError(f"Unsupported temporal scalar shape: {tuple(x.shape)}")

#     def _match_aux(self, phase, reliability, target_t):
#         phase = self._phase_to_nt(phase)
#         reliability = self._scalar_to_nt(reliability)
#         if phase is not None:
#             phase = _match_nt_time(phase, target_t)
#         if reliability is not None:
#             reliability = _match_nt_time(reliability.unsqueeze(-1), target_t).squeeze(-1)
#         return phase, reliability

#     def _split_by_seqL(self, x, phase=None, reliability=None, seqL=None, batch_size=None):
#         n, _, max_len, _, _ = x.shape
#         batch_size = int(batch_size if batch_size is not None else n)
#         phase, reliability = self._match_aux(phase, reliability, max_len)
#         if n == batch_size:
#             lens = self._seqL_to_int_list(seqL, batch_size, max_len)
#             chunks = []
#             for i, l in enumerate(lens):
#                 pi = phase[i:i + 1, :l] if phase is not None and phase.size(0) == n else (phase[:1, :l] if phase is not None else None)
#                 ri = reliability[i:i + 1, :l] if reliability is not None and reliability.size(0) == n else (reliability[:1, :l] if reliability is not None else None)
#                 chunks.append((x[i:i + 1, :, :l], pi, ri))
#             return chunks
#         if n == 1 and batch_size > 1:
#             lens = self._seqL_to_int_list(seqL, batch_size, max_len)
#             chunks, start = [], 0
#             for i, l in enumerate(lens):
#                 end = min(start + int(l), max_len)
#                 if end <= start:
#                     end = min(start + 1, max_len)
#                 pi = None if phase is None else (phase[:, start:end] if phase.size(0) == 1 else phase[i:i + 1, :end - start])
#                 ri = None if reliability is None else (reliability[:, start:end] if reliability.size(0) == 1 else reliability[i:i + 1, :end - start])
#                 chunks.append((x[:, :, start:end], pi, ri))
#                 start = end
#             return chunks
#         raise RuntimeError(f"Cannot align temporal batch: x={tuple(x.shape)}, batch_size={batch_size}, seqL={seqL}")

#     def _weighted_mean(self, x, reliability):
#         if reliability is None:
#             return x.mean(dim=2)
#         w = reliability.clamp(0, 1).view(reliability.size(0), 1, reliability.size(1), 1, 1)
#         return (x * w).sum(dim=2) / (w.sum(dim=2) + self.eps)

#     def _topk_reliable(self, x, reliability):
#         if reliability is None or x.size(2) <= 1:
#             return x.max(dim=2)[0]
#         t = x.size(2)
#         k = max(1, int(round(t * self.topk_ratio)))
#         idx = torch.topk(reliability.clamp(0, 1), k=k, dim=1, largest=True)[1]
#         gather_idx = idx.view(idx.size(0), 1, k, 1, 1).expand(-1, x.size(1), -1, x.size(3), x.size(4))
#         return x.gather(dim=2, index=gather_idx).mean(dim=2)

#     def _phase_bin_pool(self, x, phase, reliability):
#         if phase is None:
#             return self._weighted_mean(x, reliability)
#         phase2 = phase[..., :2]
#         angle = torch.atan2(phase2[..., 0], phase2[..., 1])
#         centers = torch.linspace(-math.pi, math.pi, steps=self.phase_bins + 1, device=x.device, dtype=x.dtype)[:-1]
#         weights = torch.exp(self.phase_kappa * torch.cos(angle.unsqueeze(-1) - centers))
#         if reliability is not None:
#             weights = weights * reliability.clamp(0, 1).unsqueeze(-1)
#         w = weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
#         z = (x.unsqueeze(3) * w).sum(dim=2) / (w.sum(dim=2) + self.eps)
#         return z.mean(dim=2)

#     def _pool_single(self, x, phase=None, reliability=None):
#         if self.pool_type == "avg":
#             return x.mean(dim=2)
#         if self.pool_type == "max" or x.size(2) == 1:
#             return x.max(dim=2)[0]
#         if self.pool_type == "reliability":
#             return self._weighted_mean(x, reliability)
#         if self.pool_type == "topk_reliability":
#             return self._topk_reliable(x, reliability)
#         if self.pool_type == "phase_bin":
#             return self._phase_bin_pool(x, phase, reliability=None)
#         if self.pool_type == "reliability_phase_bin":
#             return self._phase_bin_pool(x, phase, reliability=reliability)
#         # evidence: robust default for noisy GPF++ primitives.
#         z_max = x.max(dim=2)[0]
#         z_avg = x.mean(dim=2)
#         z_rel = self._weighted_mean(x, reliability)
#         z_top = self._topk_reliable(x, reliability)
#         return self.evidence_fuse(torch.cat([z_max, z_avg, z_rel, z_top], dim=1))

#     def forward(self, x, seqL=None, phase=None, reliability=None, batch_size=None):
#         if x.dim() != 5:
#             raise RuntimeError(f"TemporalEvidencePool expects [N,C,T,H,W], got {tuple(x.shape)}")
#         batch_size = int(batch_size if batch_size is not None else x.size(0))
#         if x.size(0) == batch_size and self.pool_type == "max":
#             return x.max(dim=2)[0]
#         chunks = self._split_by_seqL(x, phase=phase, reliability=reliability, seqL=seqL, batch_size=batch_size)
#         out = torch.cat([self._pool_single(xi, pi, ri) for xi, pi, ri in chunks], dim=0)
#         if out.size(0) != batch_size:
#             raise RuntimeError(f"Temporal pool batch mismatch: got {out.size(0)}, expected {batch_size}.")
#         return out


# class PartSpatialGate(nn.Module):
#     """Use body-coordinate part maps as a soft spatial reliability gate before HPP."""
#     def __init__(self, num_parts: int, channels: int, strength: float = 0.35):
#         super().__init__()
#         hidden = min(64, max(16, channels // 4))
#         self.num_parts = int(num_parts)
#         self.strength = float(strength)
#         self.net = nn.Sequential(ConvBNAct(num_parts, hidden, 3), nn.Conv2d(hidden, channels, 1))

#     def forward(self, feat: torch.Tensor, parts: Optional[torch.Tensor]):
#         if parts is None:
#             return feat, {}
#         parts = _to_ncthw(parts, expected_channels=self.num_parts).to(feat.device).float()
#         p = parts.mean(dim=2)  # [N,K,H,W]
#         if p.shape[-2:] != feat.shape[-2:]:
#             p = F.interpolate(p, size=feat.shape[-2:], mode="bilinear", align_corners=False)
#         gate = torch.sigmoid(self.net(p))
#         gate = (1.0 - self.strength) + self.strength * gate
#         return feat * gate, {"part_gate": gate.mean(dim=1, keepdim=True), "parts_pooled": p.mean(dim=1, keepdim=True)}


# class DirectGPFStem(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int):
#         super().__init__()
#         self.net = nn.Sequential(ConvBNAct(in_channels, out_channels, 3), ConvBNAct(out_channels, out_channels, 3))

#     def forward(self, x):
#         return _apply_framewise(self.net, x), {}


# def _visual_flatten_nt(x: Optional[torch.Tensor], max_frames: int = 8, normalize: bool = True) -> Optional[torch.Tensor]:
#     if x is None:
#         return None
#     x = x.detach().float()
#     if x.dim() == 4:
#         if normalize:
#             x = _visual_minmax(x)
#         return x
#     if x.dim() != 5:
#         return None
#     t = x.size(2)
#     idx = torch.linspace(0, t - 1, steps=min(t, int(max_frames)), device=x.device).round().long() if t > max_frames else torch.arange(t, device=x.device)
#     x = x.index_select(2, idx)
#     n, c, tt, h, w = x.shape
#     x = x.permute(0, 2, 1, 3, 4).reshape(n * tt, c, h, w).contiguous()
#     if normalize:
#         x = _visual_minmax(x)
#     return x


# def _visual_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
#     dims = tuple(range(2, x.dim()))
#     mn = x.amin(dim=dims, keepdim=True)
#     mx = x.amax(dim=dims, keepdim=True)
#     return (x - mn) / (mx - mn + eps)


# class GPFGaitPP(BaseModel):
#     """GPF++ model wrapper compatible with OpenGait's BaseModel API."""

#     def build_network(self, model_cfg: Dict):
#         self.model_cfg = model_cfg
#         self.variant = model_cfg.get("variant", "gpfpp_structured")
#         self.input_index = int(model_cfg.get("input_index", 0))
#         self.phase_input_index = model_cfg.get("phase_input_index", None)
#         self.parts_input_index = model_cfg.get("parts_input_index", None)
#         self.source_channels = int(model_cfg.get("source_channels", 28))
#         self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))
#         self.visualize = bool(model_cfg.get("visualize", False))
#         self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))

#         self.norm = GPFPPChannelNorm(enabled=bool(model_cfg.get("enable_channel_norm", True)))
#         stem_out = int(model_cfg.get("stem_out_channels", 48))

#         self.geo_indices = list(map(int, model_cfg.get("geo_indices", list(range(0, 10)))))
#         self.motion_indices = list(map(int, model_cfg.get("motion_indices", list(range(10, 20)))))
#         self.freq_indices = list(map(int, model_cfg.get("frequency_indices", list(range(20, 24)))))
#         self.affinity_indices = list(map(int, model_cfg.get("affinity_indices", list(range(24, 28)))))

#         self.reliability_index = model_cfg.get("reliability_index", 19)
#         self.periodic_index = model_cfg.get("periodic_index", 18)
#         self.texture_inverse_index = model_cfg.get("texture_inverse_index", 25)

#         if self.variant == "sil":
#             self.adapter_type = "sil"
#             self.adapter = DirectGPFStem(1, stem_out)
#         elif self.variant in ("gpfpp_direct", "gpf_direct"):
#             self.adapter_type = "direct"
#             self.channel_indices = list(map(int, model_cfg.get("channel_indices", list(range(self.source_channels)))))
#             self.adapter = DirectGPFStem(len(self.channel_indices), stem_out)
#         elif self.variant in ("gpfpp_structured", "gpf_structured", "gpf_grouped"):
#             self.adapter_type = "gpfpp"
#             group_channels = {
#                 "geometry": len(self.geo_indices),
#                 "motion": len(self.motion_indices),
#                 "frequency": len(self.freq_indices),
#                 "affinity": len(self.affinity_indices),
#             }
#             self.adapter = GPFPPStem(
#                 group_channels=group_channels,
#                 out_channels=stem_out,
#                 branch_channels=int(model_cfg.get("grouped_branch_channels", 16)),
#                 affinity_branch_channels=int(model_cfg.get("affinity_branch_channels", 12)),
#                 use_gates=bool(model_cfg.get("use_primitive_gates", True)),
#                 gate_strength=float(model_cfg.get("primitive_gate_strength", 0.75)),
#                 primitive_dropout=float(model_cfg.get("primitive_dropout", 0.05)),
#             )
#         else:
#             raise ValueError(f"Unknown GPFGait variant: {self.variant}")

#         self.use_phase_film = bool(model_cfg.get("use_phase_film", False))
#         self.phase_film = PhaseFiLM(stem_out, hidden_channels=int(model_cfg.get("phase_hidden_channels", 32)), strength=float(model_cfg.get("phase_film_strength", 0.25))) if self.use_phase_film else nn.Identity()

#         self.backbone = FrameBackbone(
#             stem_out,
#             channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
#             strides=model_cfg.get("backbone_strides", [1, 2, 1, 1]),
#             blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
#         )
#         backbone_out = self.backbone.out_channels

#         self.TP = TemporalEvidencePool(
#             backbone_out,
#             pool_type=model_cfg.get("temporal_pool", "evidence"),
#             topk_ratio=float(model_cfg.get("temporal_topk_ratio", 0.5)),
#             phase_bins=int(model_cfg.get("phase_bins", 8)),
#             phase_kappa=float(model_cfg.get("phase_kappa", 4.0)),
#         )

#         self.use_part_gate = bool(model_cfg.get("use_part_gate", True))
#         self.num_parts = int(model_cfg.get("num_parts", 8))
#         self.part_gate = PartSpatialGate(self.num_parts, backbone_out, strength=float(model_cfg.get("part_gate_strength", 0.35))) if self.use_part_gate else None

#         bin_num = model_cfg.get("bin_num", [32])
#         self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
#         parts_num = int(sum(bin_num))
#         emb_channels = int(model_cfg.get("emb_channels", 256))
#         class_num = int(model_cfg["class_num"])
#         fc_cfg = model_cfg.get("SeparateFCs", {})
#         bn_cfg = model_cfg.get("SeparateBNNecks", {})
#         self.FCs = SeparateFCs(
#             parts_num=fc_cfg.get("parts_num", parts_num),
#             in_channels=fc_cfg.get("in_channels", backbone_out),
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
#         if self.phase_input_index is not None:
#             idx = int(self.phase_input_index)
#             if 0 <= idx < len(ipts):
#                 try:
#                     return _to_nt_phase(ipts[idx]).to(gpf.device if gpf is not None else ipts[idx].device)
#                 except Exception:
#                     pass
#         if gpf is not None and gpf.size(1) > 21:
#             phase = gpf[:, [20, 21]].mean(dim=(-1, -2)).transpose(1, 2).contiguous()
#             return phase
#         return None

#     def _get_parts(self, ipts):
#         if self.parts_input_index is None:
#             return None
#         idx = int(self.parts_input_index)
#         if not (0 <= idx < len(ipts)):
#             return None
#         return ipts[idx]

#     def _scalar_from_gpf(self, gpf: Optional[torch.Tensor], idx, default: Optional[torch.Tensor] = None):
#         if gpf is None or idx is None:
#             return default
#         idx = int(idx)
#         if idx < 0 or idx >= gpf.size(1):
#             return default
#         return gpf[:, idx:idx + 1]

#     def _frame_scalar_from_gpf(self, gpf: Optional[torch.Tensor], idx):
#         x = self._scalar_from_gpf(gpf, idx)
#         if x is None:
#             return None
#         return x.mean(dim=(-1, -2)).clamp(0, 1)

#     def _build_visual_summary(self, gpf, stem_feat, backbone_feat, temporal_feat, aux):
#         visual = {}
#         if gpf is not None:
#             visual["image/gpfpp_mask"] = _visual_flatten_nt(gpf[:, 0:1], self.visualize_max_frames, normalize=False)
#         if not self.visualize:
#             return visual
#         if gpf is not None:
#             for idx, name in [(1, "edge"), (12, "flow_mag"), (18, "periodic"), (19, "reliability"), (24, "stable_residual_abs"), (25, "texture_inverse")]:
#                 if idx < gpf.size(1):
#                     x = gpf[:, idx:idx + 1].abs() if idx == 24 else gpf[:, idx:idx + 1]
#                     visual[f"image/gpfpp_{idx:02d}_{name}"] = _visual_flatten_nt(x, self.visualize_max_frames, normalize=True)
#         if aux:
#             for name, value in aux.items():
#                 if value is not None and torch.is_tensor(value):
#                     visual[f"image/{name}"] = _visual_flatten_nt(value, self.visualize_max_frames, normalize=True)
#         if temporal_feat is not None:
#             energy = temporal_feat.detach().abs().mean(dim=1, keepdim=True)
#             visual["image/pooled_feature_energy"] = _visual_flatten_nt(energy, self.visualize_max_frames, normalize=True)
#         return visual

#     def forward(self, inputs):
#         ipts, labs, typs, vies, seqL = inputs
#         local_bs = labs.size(0) if torch.is_tensor(labs) else len(labs)
#         gpf = None
#         phase = None
#         parts = None
#         aux = {}

#         if self.adapter_type == "sil":
#             x_in = _to_ncthw(ipts[self.input_index], expected_channels=1)
#             x, stem_aux = self.adapter(x_in)
#             gpf_for_vis = x_in
#         else:
#             gpf = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
#             gpf = self.norm(gpf)
#             gpf_for_vis = gpf
#             phase = self._get_phase(ipts, gpf)
#             parts = self._get_parts(ipts)

#             if self.adapter_type == "direct":
#                 x_in = _select_channels(gpf, self.channel_indices, "channel_indices")
#                 x, stem_aux = self.adapter(x_in)
#             else:
#                 groups = {
#                     "geometry": _select_channels(gpf, self.geo_indices, "geo_indices"),
#                     "motion": _select_channels(gpf, self.motion_indices, "motion_indices"),
#                     "frequency": _select_channels(gpf, self.freq_indices, "frequency_indices"),
#                     "affinity": _select_channels(gpf, self.affinity_indices, "affinity_indices"),
#                 }
#                 reliability = self._scalar_from_gpf(gpf, self.reliability_index)
#                 periodic = self._scalar_from_gpf(gpf, self.periodic_index, reliability)
#                 texture_inv = self._scalar_from_gpf(gpf, self.texture_inverse_index, reliability)
#                 gates = {"motion": reliability, "frequency": periodic, "affinity": texture_inv}
#                 foreground = gpf[:, 0:1] if gpf.size(1) > 0 else None
#                 x, stem_aux = self.adapter(groups, gates=gates, foreground=foreground)
#         aux.update(stem_aux)

#         if self.use_phase_film:
#             x = self.phase_film(x, phase)

#         stem_feat = x
#         x = self.backbone(x)
#         backbone_feat = x
#         reliability_scalar = self._frame_scalar_from_gpf(gpf, self.reliability_index)
#         x = self.TP(x, seqL=seqL, phase=phase, reliability=reliability_scalar, batch_size=local_bs)
#         temporal_feat = x

#         if self.part_gate is not None:
#             x, part_aux = self.part_gate(x, parts)
#             aux.update(part_aux)

#         feat = self.HPP(x)
#         embed_1 = self.FCs(feat)
#         embed_2, logits = self.BNNecks(embed_1)
#         embed = embed_2 if self.inference_use_emb2 else embed_1

#         if embed_1.size(0) != local_bs:
#             raise RuntimeError(f"Embedding batch mismatch: {tuple(embed_1.shape)} vs labels batch {local_bs}")
#         if logits.size(0) != local_bs:
#             raise RuntimeError(f"Logits batch mismatch: {tuple(logits.shape)} vs labels batch {local_bs}")

#         return {
#             "training_feat": {
#                 "triplet": {"embeddings": embed_1, "labels": labs},
#                 "softmax": {"logits": logits, "labels": labs},
#             },
#             "visual_summary": self._build_visual_summary(gpf_for_vis, stem_feat, backbone_feat, temporal_feat, aux),
#             "inference_feat": {"embeddings": embed},
#         }
