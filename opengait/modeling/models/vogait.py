# -*- coding: utf-8 -*-
"""
VOGait: View-Observed Gait Strategy Network for multi-channel GPF inputs.

Drop this file into:
    OpenGait/opengait/modeling/models/vogait.py

Then add to OpenGait/opengait/modeling/models/__init__.py:
    from .vogait import VOGait

Expected input:
    ipts[0] = 000-gpf.pkl, shape [T, C, 128, 64] or collated OpenGait variants.

Supported GPF layouts (configured via model_cfg channel indices):

  10-channel (no phase):
    00 silhouette mask         05 motion amplitude
    01 body edge               06 optical-flow dx
    02 contour normal x        07 optical-flow dy
    03 contour normal y        08 motion reliability
    04 signed distance field   09 mid-frequency residual

  12-channel (with phase):
    00 silhouette mask         06 phase sin
    01 body edge               07 phase cos
    02 contour normal x        08 optical-flow dx
    03 contour normal y        09 optical-flow dy
    04 signed distance field   10 motion reliability
    05 motion amplitude        11 mid-frequency residual

Main modules:
    Stage 1: View Observer
    Stage 2: Body-Motion Slot Encoder
    Stage 3: View-Conditioned Strategy Bank
    Stage 4: Koopman-style Part-Phase Dynamical Reasoner
    Stage 5: HPP + SeparateFCs + SeparateBNNecks identity head
"""

import math
import re
from typing import Dict, Optional, Sequence, Any, List

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
        return x.unsqueeze(1).float().contiguous()  # [N,T,H,W] -> [N,1,T,H,W]
    if x.dim() != 5:
        raise ValueError(f"Expected a 4D or 5D input tensor, got shape {tuple(x.shape)}")
    if expected_channels is not None and x.size(1) == expected_channels:
        return x.float().contiguous()
    if expected_channels is not None and x.size(2) == expected_channels:
        return x.transpose(1, 2).float().contiguous()
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
        raise ValueError(f"{name} contains invalid index {indices}, but input has {x.size(1)} channels.")
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    return x.index_select(1, idx)


def _apply_framewise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a 2D module to [N, C, T, H, W] frame by frame."""
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


def _flatten_meta(meta: Any) -> List[Any]:
    """Flatten common OpenGait metadata containers such as list/tuple/np arrays."""
    if meta is None:
        return []
    if torch.is_tensor(meta):
        return meta.detach().cpu().reshape(-1).tolist()
    if isinstance(meta, np.ndarray):
        return meta.reshape(-1).tolist()
    if isinstance(meta, (list, tuple)):
        out = []
        for v in meta:
            out.extend(_flatten_meta(v))
        return out
    return [meta]


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
        strides: Sequence[int] = (1, 2, 2, 1),
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
    """Per-channel sanitization for GPF, parameterized by channel semantics.

    When ``channel_semantics`` is provided (a dict mapping int index to a
    semantic type string), normalization is applied according to each
    channel's meaning rather than its positional index.  This makes the
    module agnostic to the number/order of input channels.

    Supported types:
        'binary'      – clamp [0, 1]  (mask, edge, motion amplitude, reliability)
        'signed_unit' – clamp [-1, 1] (contour normals, phase sin/cos)
        'sdf'         – clamp & scale by sdf_clip
        'flow'        – clamp & scale by flow_clip
        'residual'    – clamp & scale by residual_clip

    If ``channel_semantics`` is *None*, the module falls back to the
    hardcoded 12-channel layout for backward compatibility.
    """
    def __init__(self, enabled=True, sdf_clip=1.0, flow_clip=1.0, residual_clip=1.0,
                 channel_semantics=None):
        super().__init__()
        self.enabled = enabled
        self.sdf_clip = float(sdf_clip)
        self.flow_clip = float(flow_clip)
        self.residual_clip = float(residual_clip)
        self.channel_semantics = channel_semantics  # dict[int, str] or None

    def forward(self, x):
        x = x.float()
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        if not self.enabled:
            return x
        y = x.clone()
        c = y.size(1)

        if self.channel_semantics is not None:
            for idx_str, sem in self.channel_semantics.items():
                idx = int(idx_str)
                if idx >= c:
                    continue
                if sem == 'binary':
                    y[:, idx] = y[:, idx].clamp(0.0, 1.0)
                elif sem == 'signed_unit':
                    y[:, idx] = y[:, idx].clamp(-1.0, 1.0)
                elif sem == 'sdf':
                    y[:, idx] = y[:, idx].clamp(-self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
                elif sem == 'flow':
                    y[:, idx] = y[:, idx].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
                elif sem == 'residual':
                    y[:, idx] = y[:, idx].clamp(-self.residual_clip, self.residual_clip) / max(self.residual_clip, 1e-6)
            return y

        # Fallback: hardcoded 12-channel layout for backward compatibility.
        if c > 0: y[:, 0] = y[:, 0].clamp(0.0, 1.0)
        if c > 1: y[:, 1] = y[:, 1].clamp(0.0, 1.0)
        if c > 2: y[:, 2] = y[:, 2].clamp(-1.0, 1.0)
        if c > 3: y[:, 3] = y[:, 3].clamp(-1.0, 1.0)
        if c > 4: y[:, 4] = y[:, 4].clamp(-self.sdf_clip, self.sdf_clip) / max(self.sdf_clip, 1e-6)
        if c > 5: y[:, 5] = y[:, 5].clamp(0.0, 1.0)
        if c > 6: y[:, 6] = y[:, 6].clamp(-1.0, 1.0)
        if c > 7: y[:, 7] = y[:, 7].clamp(-1.0, 1.0)
        if c > 8: y[:, 8] = y[:, 8].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
        if c > 9: y[:, 9] = y[:, 9].clamp(-self.flow_clip, self.flow_clip) / max(self.flow_clip, 1e-6)
        if c > 10: y[:, 10] = y[:, 10].clamp(0.0, 1.0)
        if c > 11: y[:, 11] = y[:, 11].clamp(-self.residual_clip, self.residual_clip) / max(self.residual_clip, 1e-6)
        return y


class GPFPrimitiveStem(nn.Module):
    """Typed GPF adapter: geometry/motion/reliability/residual are not treated as identical channels."""
    def __init__(
        self,
        geo_channels: int,
        motion_channels: int,
        residual_channels: int,
        out_channels: int,
        branch_channels: int = 24,
        residual_branch_channels: int = 12,
        gate_strength: float = 0.75,
        primitive_dropout: float = 0.0,
    ):
        super().__init__()
        self.gate_strength = float(gate_strength)
        self.primitive_dropout = float(primitive_dropout)
        self.geo_stem = nn.Sequential(ConvBNAct(geo_channels, branch_channels, 3), ConvBNAct(branch_channels, branch_channels, 3))
        self.motion_stem = nn.Sequential(ConvBNAct(motion_channels, branch_channels, 3), ConvBNAct(branch_channels, branch_channels, 3))
        self.has_residual = residual_channels > 0
        if self.has_residual:
            self.residual_stem = nn.Sequential(
                ConvBNAct(residual_channels, residual_branch_channels, 3),
                ConvBNAct(residual_branch_channels, residual_branch_channels, 3),
            )
            self.residual_gate = nn.Sequential(ConvBNAct(1, residual_branch_channels, 3), nn.Conv2d(residual_branch_channels, residual_branch_channels, 1))
        else:
            self.residual_stem = None
            self.residual_gate = None
        self.motion_gate = nn.Sequential(ConvBNAct(1, branch_channels, 3), nn.Conv2d(branch_channels, branch_channels, 1))
        fuse_in = branch_channels * 2 + (residual_branch_channels if self.has_residual else 0)
        self.fuse = nn.Sequential(ConvBNAct(fuse_in, out_channels, 1, padding=0), ConvBNAct(out_channels, out_channels, 3))

    def _soft_gate(self, logits):
        gate = torch.sigmoid(logits)
        return (1.0 - self.gate_strength) + self.gate_strength * gate

    def _maybe_drop(self, x):
        if self.training and self.primitive_dropout > 0:
            if torch.rand((), device=x.device) < self.primitive_dropout:
                return x * 0.0
        return x

    def forward(self, geo, motion, reliability=None, residual=None, foreground=None):
        aux = {}
        geo_feat = _apply_framewise(self.geo_stem, geo)
        motion_feat = _apply_framewise(self.motion_stem, motion)
        if reliability is not None:
            motion_gate = self._soft_gate(_apply_framewise(self.motion_gate, reliability))
            if foreground is not None:
                motion_gate = motion_gate * (0.5 + 0.5 * foreground.clamp(0, 1))
            motion_feat = motion_feat * motion_gate
            aux["stem_motion_gate"] = motion_gate
        feats = [geo_feat, self._maybe_drop(motion_feat)]
        if self.has_residual and residual is not None:
            res_feat = _apply_framewise(self.residual_stem, residual)
            if reliability is not None:
                res_gate = self._soft_gate(_apply_framewise(self.residual_gate, reliability))
                if foreground is not None:
                    res_gate = res_gate * (0.5 + 0.5 * foreground.clamp(0, 1))
                res_feat = res_feat * res_gate
                aux["stem_residual_gate"] = res_gate
            feats.append(self._maybe_drop(res_feat))
        return _apply_framewise(self.fuse, torch.cat(feats, dim=1)), aux


class ViewObserver(nn.Module):
    """Explicitly estimates view-state tokens before identity reasoning."""
    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        token_dim: int = 128,
        num_parts: int = 6,
        num_azimuth_bins: int = 8,
        num_elevation_bins: int = 3,
    ):
        super().__init__()
        self.num_parts = int(num_parts)
        self.num_azimuth_bins = int(num_azimuth_bins)
        self.num_elevation_bins = int(num_elevation_bins)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 3),
            BasicFrameBlock(hidden_channels, hidden_channels, 1),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.token_proj = nn.Sequential(nn.Linear(hidden_channels, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.azimuth_head = nn.Linear(token_dim, num_azimuth_bins)
        self.elevation_head = nn.Linear(token_dim, num_elevation_bins)
        self.visibility_head = nn.Linear(token_dim, num_parts)
        self.occlusion_head = nn.Linear(token_dim, num_parts)
        self.part_conf_head = nn.Linear(token_dim, num_parts)
        self.foreshortening_head = nn.Linear(token_dim, 1)
        self.phase_head = nn.Linear(token_dim, 2)

    def forward(self, gpf: torch.Tensor):
        feat = _apply_framewise(self.stem, gpf)
        desc = feat.mean(dim=(-1, -2)).transpose(1, 2).contiguous()  # [N,T,C]
        tmp = self.temporal(desc.transpose(1, 2)).transpose(1, 2).contiguous()
        token = self.token_proj(tmp)
        phase = F.normalize(self.phase_head(token), dim=-1, eps=1e-6)
        return {
            "observer_feat": feat,
            "view_token": token,
            "azimuth_logits": self.azimuth_head(token),
            "elevation_logits": self.elevation_head(token),
            "visibility_logits": self.visibility_head(token),
            "occlusion_logits": self.occlusion_head(token),
            "part_confidence_logits": self.part_conf_head(token),
            "foreshortening": torch.sigmoid(self.foreshortening_head(token)),
            "phase_pred": phase,
        }


class BodyMotionSlotEncoder(nn.Module):
    """Lightweight body-motion slots, using foreground-aware spatial attention."""
    def __init__(self, in_channels: int, slot_dim: int = 128, num_slots: int = 7, attn_temperature: float = 1.0):
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.attn_temperature = float(attn_temperature)
        self.value_proj = nn.Sequential(ConvBNAct(in_channels, slot_dim, 1, padding=0), ConvBNAct(slot_dim, slot_dim, 3))
        self.attn = nn.Conv2d(in_channels, num_slots, 1)
        self.slot_norm = nn.LayerNorm(slot_dim)

    def forward(self, x: torch.Tensor, foreground: Optional[torch.Tensor] = None):
        n, c, t, h, w = x.shape
        logits = _apply_framewise(self.attn, x)  # [N,K,T,H,W]
        if foreground is not None:
            fg = foreground.clamp(0, 1)
            if fg.shape[-2:] != (h, w):
                fg2 = fg.permute(0, 2, 1, 3, 4).reshape(n * t, 1, fg.size(-2), fg.size(-1))
                fg2 = F.interpolate(fg2, size=(h, w), mode="bilinear", align_corners=False)
                fg = fg2.reshape(n, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            logits = logits + torch.log(fg + 1e-3)
        logits = logits / max(self.attn_temperature, 1e-6)
        attn = F.softmax(logits.reshape(n, self.num_slots, t, h * w), dim=-1)
        values = _apply_framewise(self.value_proj, x).permute(0, 2, 3, 4, 1).reshape(n, t, h * w, self.slot_dim)
        attn_ntkhw = attn.permute(0, 2, 1, 3).contiguous()  # [N,T,K,HW]
        slots = torch.einsum("ntks,ntsd->ntkd", attn_ntkhw, values)
        slots = self.slot_norm(slots)
        slot_maps = attn.reshape(n, self.num_slots, t, h, w)
        return slots, slot_maps


class StrategyExpert(nn.Module):
    def __init__(self, channels: int, expansion: float = 1.0):
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.net = nn.Sequential(
            BasicFrameBlock(channels, hidden, 1),
            BasicFrameBlock(hidden, channels, 1),
        )

    def forward(self, x):
        return _apply_framewise(self.net, x)


class ViewConditionedStrategyBank(nn.Module):
    """Soft routing over view strategy experts."""
    def __init__(
        self,
        channels: int,
        token_dim: int,
        num_experts: int = 6,
        router_hidden: int = 128,
        film_strength: float = 0.5,
        expert_expansion: float = 1.0,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.film_strength = float(film_strength)
        self.router = nn.Sequential(
            nn.Linear(token_dim, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, num_experts),
        )
        self.film = nn.Sequential(
            nn.Linear(token_dim, router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden, channels * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.experts = nn.ModuleList([StrategyExpert(channels, expansion=expert_expansion) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, view_token: torch.Tensor):
        view_token = _match_nt_time(view_token, x.size(2))
        gb = self.film(view_token)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        x_mod = x * (1.0 + self.film_strength * gamma) + self.film_strength * beta

        alpha = F.softmax(self.router(view_token), dim=-1)  # [N,T,E]
        expert_feats = torch.stack([expert(x_mod) for expert in self.experts], dim=0)  # [E,N,C,T,H,W]
        weight = alpha.permute(2, 0, 1).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        y = (expert_feats * weight).sum(dim=0)
        return y, alpha


class PartPhaseDynamicalReasoner(nn.Module):
    """Koopman-style latent dynamics plus view-conditioned observation gate."""
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        view_token_dim: int,
        latent_dim: int = 128,
        context_channels: int = 512,
        context_strength: float = 0.25,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_channels = int(context_channels)
        self.context_strength = float(context_strength)
        self.slot_to_z = nn.Sequential(
            nn.Linear(num_slots * slot_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        eye = torch.eye(latent_dim)
        self.koopman = nn.Parameter(eye + 0.01 * torch.randn_like(eye))
        self.obs_gate = nn.Sequential(
            nn.Linear(view_token_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(latent_dim, context_channels * 2),
        )
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, slots: torch.Tensor, view_token: torch.Tensor):
        # slots: [N,T,K,D]
        n, t, k, d = slots.shape
        z = self.slot_to_z(slots.reshape(n, t, k * d))
        view_token = _match_nt_time(view_token, t)
        obs_gate = torch.sigmoid(self.obs_gate(view_token))
        z_obs = z * obs_gate
        if t > 1:
            z_next_pred = torch.matmul(z[:, :-1], self.koopman.t())
            z_next_target = z[:, 1:]
            z_obs_next_pred = z_next_pred * obs_gate[:, 1:]
            z_obs_next_target = z_obs[:, 1:]
        else:
            z_next_pred = z[:, :0]
            z_next_target = z[:, :0]
            z_obs_next_pred = z[:, :0]
            z_obs_next_target = z[:, :0]
        gb = self.context(z)
        return {
            "z": z,
            "z_obs": z_obs,
            "z_next_pred": z_next_pred,
            "z_next_target": z_next_target,
            "z_obs_next_pred": z_obs_next_pred,
            "z_obs_next_target": z_obs_next_target,
            "dynamics_context": gb,
        }

    def apply_context(self, x: torch.Tensor, dyn_out: Dict[str, torch.Tensor]):
        gb = _match_nt_time(dyn_out["dynamics_context"], x.size(2))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.context_strength * gamma) + self.context_strength * beta


class TemporalPrimitivePool(nn.Module):
    """Temporal pooling that supports OpenGait packed variable-length inference."""
    def __init__(self, pool_type: str = "max", phase_bins: int = 8, phase_kappa: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.pool_type = str(pool_type)
        self.phase_bins = int(phase_bins)
        self.phase_kappa = float(phase_kappa)
        self.eps = eps

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
            try:
                lens.append(max(1, int(v)))
            except Exception:
                pass
        if len(lens) == 1 and batch_size > 1:
            return [min(lens[0], max_len) for _ in range(batch_size)]
        if len(lens) < batch_size:
            lens = lens + [lens[-1] if lens else max_len] * (batch_size - len(lens))
        return lens[:batch_size]

    def _pool_single(self, x, phase=None, reliability=None):
        if self.pool_type == "avg":
            return x.mean(dim=2)
        if self.pool_type == "reliability" and reliability is not None:
            w = reliability.clamp(0, 1).view(reliability.size(0), 1, reliability.size(1), 1, 1)
            return (x * w).sum(dim=2) / (w.sum(dim=2) + self.eps)
        if self.pool_type == "phase_bin" and phase is not None:
            phase = _match_nt_time(phase, x.size(2))
            angle = torch.atan2(phase[..., 0], phase[..., 1])
            centers = torch.linspace(-math.pi, math.pi, steps=self.phase_bins + 1, device=x.device, dtype=x.dtype)[:-1]
            weights = torch.exp(self.phase_kappa * torch.cos(angle.unsqueeze(-1) - centers))
            w = weights.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            xb = x.unsqueeze(3)
            z = (xb * w).sum(dim=2) / (w.sum(dim=2) + self.eps)
            return z.mean(dim=2)
        return x.max(dim=2)[0]

    def forward(self, x, seqL=None, phase=None, reliability=None, batch_size=None):
        if batch_size is None:
            batch_size = x.size(0)
        batch_size = int(batch_size)
        if x.size(0) == batch_size:
            outs = []
            lens = self._seqL_to_int_list(seqL, batch_size, x.size(2)) if seqL is not None else [x.size(2)] * batch_size
            for i, l in enumerate(lens):
                xi = x[i:i + 1, :, :min(l, x.size(2))]
                pi = None if phase is None else phase[i:i + 1, :min(l, phase.size(1))]
                ri = None if reliability is None else reliability[i:i + 1, :min(l, reliability.size(1))]
                outs.append(self._pool_single(xi, pi, ri))
            return torch.cat(outs, dim=0)
        if x.size(0) == 1 and batch_size > 1:
            lens = self._seqL_to_int_list(seqL, batch_size, x.size(2))
            outs, start = [], 0
            for i, l in enumerate(lens):
                end = min(start + int(l), x.size(2))
                if end <= start:
                    end = min(start + 1, x.size(2))
                xi = x[:, :, start:end]
                pi = None if phase is None else phase[:, start:end]
                ri = None if reliability is None else reliability[:, start:end]
                outs.append(self._pool_single(xi, pi, ri))
                start = end
            return torch.cat(outs, dim=0)
        raise RuntimeError(f"Cannot align temporal batch: x.shape={tuple(x.shape)}, batch_size={batch_size}, seqL={seqL}")


def _part_adaptive_pool(x: torch.Tensor, num_parts: int) -> torch.Tensor:
    """Pool [N,1,T,H,W] into [N,T,K] vertical body parts."""
    n, c, t, h, w = x.shape
    x2 = x.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
    pooled = F.adaptive_avg_pool2d(x2, (num_parts, 1)).reshape(n, t, c, num_parts)
    return pooled.mean(dim=2)


def _build_visual_summary(gpf, slot_maps=None, alpha=None, max_frames=8, reliability_index=None):
    out = {}
    if gpf is None:
        return out
    t = gpf.size(2)
    if t <= max_frames:
        idx = torch.arange(t, device=gpf.device)
    else:
        idx = torch.linspace(0, t - 1, steps=max_frames, device=gpf.device).round().long()
    def flat(x, normalize=True):
        x = x.detach().float().index_select(2, idx)
        n, c, tt, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(n * tt, c, h, w).contiguous()
        if normalize:
            dims = tuple(range(2, x.dim()))
            x = (x - x.amin(dim=dims, keepdim=True)) / (x.amax(dim=dims, keepdim=True) - x.amin(dim=dims, keepdim=True) + 1e-6)
        return x
    if gpf.size(1) > 0:
        out["image/gpf_mask"] = flat(gpf[:, 0:1], normalize=False)
    if gpf.size(1) > 5:
        out["image/gpf_motion_amp"] = flat(gpf[:, 5:6])
    if reliability_index is not None and gpf.size(1) > reliability_index:
        out["image/gpf_reliability"] = flat(gpf[:, reliability_index:reliability_index + 1])
    if slot_maps is not None:
        out["image/body_motion_slots"] = flat(slot_maps.mean(dim=1, keepdim=True))
    if alpha is not None:
        # [N,T,E] -> simple temporal expert-probability strips as image-like diagnostics.
        a = alpha.detach().float().permute(0, 2, 1).unsqueeze(-1)  # [N,E,T,1]
        a = a.unsqueeze(-1)  # [N,E,T,1,1]
        out["image/strategy_alpha"] = flat(a, normalize=False)
    return out


class VOGait(BaseModel):
    """View-Observed Gait Strategy Network for multi-channel GPF (10 or 12 channels)."""

    def build_network(self, model_cfg: Dict):
        self.model_cfg = model_cfg
        self.input_index = int(model_cfg.get("input_index", 0))
        self.source_channels = int(model_cfg.get("source_channels", 12))
        self.phase_input_index = model_cfg.get("phase_input_index", None)
        self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))
        self.num_parts = int(model_cfg.get("num_parts", 6))
        self.num_slots = int(model_cfg.get("num_slots", 7))
        self.num_experts = int(model_cfg.get("num_experts", 6))
        self.azimuth_bins = list(map(int, model_cfg.get("azimuth_bins", [0, 45, 90, 135, 180, 225, 270, 315])))
        self.num_azimuth_bins = int(model_cfg.get("num_azimuth_bins", len(self.azimuth_bins)))
        self.num_elevation_bins = int(model_cfg.get("num_elevation_bins", 3))
        self.view_label_mapping = {str(k): int(v) for k, v in model_cfg.get("view_label_mapping", {}).items()}
        self.elevation_label_mapping = {str(k): int(v) for k, v in model_cfg.get("elevation_label_mapping", {}).items()}
        self.view_supervision_mode = str(model_cfg.get("view_supervision_mode", "auto"))
        self.visualize = bool(model_cfg.get("visualize", False))
        self.visualize_max_frames = int(model_cfg.get("visualize_max_frames", 8))

        self.geo_indices = list(map(int, model_cfg.get("geo_indices", [0, 1, 2, 3, 4])))
        self.motion_indices = list(map(int, model_cfg.get("motion_indices", [5, 8, 9])))
        self.residual_indices = list(map(int, model_cfg.get("residual_indices", [11])))
        self.phase_indices = list(map(int, model_cfg.get("phase_indices", [6, 7])))
        self.reliability_index = model_cfg.get("reliability_index", 10)
        self.reliability_index = None if self.reliability_index is None else int(self.reliability_index)

        # Build per-channel semantic map for GPFChannelNorm.
        # geo default order: mask, edge, normal_x, normal_y, sdf
        _geo_types = ['binary', 'binary', 'signed_unit', 'signed_unit', 'sdf']
        channel_semantics = {}
        for j, idx in enumerate(self.geo_indices):
            channel_semantics[idx] = _geo_types[j] if j < len(_geo_types) else 'binary'
        # motion: first element is motion amplitude (binary), rest are optical flow
        for j, idx in enumerate(self.motion_indices):
            channel_semantics[idx] = 'binary' if j == 0 else 'flow'
        for idx in self.phase_indices:
            channel_semantics[idx] = 'signed_unit'
        if self.reliability_index is not None:
            channel_semantics[self.reliability_index] = 'binary'
        for idx in self.residual_indices:
            channel_semantics[idx] = 'residual'

        self.norm = GPFChannelNorm(
            enabled=bool(model_cfg.get("enable_channel_norm", True)),
            sdf_clip=float(model_cfg.get("sdf_clip", 1.0)),
            flow_clip=float(model_cfg.get("flow_clip", 1.0)),
            residual_clip=float(model_cfg.get("residual_clip", 1.0)),
            channel_semantics=channel_semantics,
        )

        stem_out = int(model_cfg.get("stem_out_channels", 48))
        self.primitive_stem = GPFPrimitiveStem(
            geo_channels=len(self.geo_indices),
            motion_channels=len(self.motion_indices),
            residual_channels=len(self.residual_indices),
            out_channels=stem_out,
            branch_channels=int(model_cfg.get("grouped_branch_channels", 32)),
            residual_branch_channels=int(model_cfg.get("residual_branch_channels", 16)),
            gate_strength=float(model_cfg.get("reliability_gate_strength", 0.75)),
            primitive_dropout=float(model_cfg.get("primitive_dropout", 0.05)),
        )

        token_dim = int(model_cfg.get("view_token_dim", 128))
        self.view_observer = ViewObserver(
            in_channels=self.source_channels,
            hidden_channels=int(model_cfg.get("observer_hidden_channels", 64)),
            token_dim=token_dim,
            num_parts=self.num_parts,
            num_azimuth_bins=self.num_azimuth_bins,
            num_elevation_bins=self.num_elevation_bins,
        )
        slot_dim = int(model_cfg.get("slot_dim", 128))
        self.slot_encoder = BodyMotionSlotEncoder(stem_out, slot_dim=slot_dim, num_slots=self.num_slots)
        self.strategy_bank = ViewConditionedStrategyBank(
            channels=stem_out,
            token_dim=token_dim,
            num_experts=self.num_experts,
            router_hidden=int(model_cfg.get("router_hidden_channels", 128)),
            film_strength=float(model_cfg.get("strategy_film_strength", 0.5)),
            expert_expansion=float(model_cfg.get("expert_expansion", 1.0)),
        )

        self.backbone = FrameBackbone(
            stem_out,
            channels=model_cfg.get("backbone_channels", [64, 128, 256, 512]),
            strides=model_cfg.get("backbone_strides", [1, 2, 2, 1]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
        )

        self.dynamics = PartPhaseDynamicalReasoner(
            num_slots=self.num_slots,
            slot_dim=slot_dim,
            view_token_dim=token_dim,
            latent_dim=int(model_cfg.get("dynamics_latent_dim", 128)),
            context_channels=self.backbone.out_channels,
            context_strength=float(model_cfg.get("dynamics_context_strength", 0.25)),
        )

        bin_num = model_cfg.get("bin_num", [16])
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        parts_num = int(sum(bin_num))
        emb_channels = int(model_cfg.get("emb_channels", 256))
        class_num = int(model_cfg["class_num"])
        # Embedding dropout for regularization (anti-overfitting).
        self.embed_dropout_rate = float(model_cfg.get("embed_dropout", 0.0))
        self.embed_dropout = nn.Dropout(p=self.embed_dropout_rate) if self.embed_dropout_rate > 0 else nn.Identity()
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
        self.TP = TemporalPrimitivePool(
            pool_type=model_cfg.get("temporal_pool", "max"),
            phase_bins=int(model_cfg.get("phase_bins", 8)),
            phase_kappa=float(model_cfg.get("phase_kappa", 4.0)),
        )

    def _get_phase(self, ipts, gpf: Optional[torch.Tensor]):
        if self.phase_input_index is not None:
            idx = int(self.phase_input_index)
            if 0 <= idx < len(ipts):
                try:
                    phase = _to_nt2(ipts[idx])
                    return phase.to(gpf.device) if gpf is not None else phase
                except Exception:
                    pass
        if gpf is not None and len(self.phase_indices) >= 2 and max(self.phase_indices) < gpf.size(1):
            phase_map = _select_channels(gpf, self.phase_indices[:2], "phase_indices")
            phase = phase_map.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
            return F.normalize(phase, dim=-1, eps=1e-6)
        return None

    def _get_reliability_scalar(self, gpf: Optional[torch.Tensor]):
        if gpf is None or self.reliability_index is None or self.reliability_index >= gpf.size(1):
            return None
        return gpf[:, self.reliability_index].mean(dim=(-1, -2)).clamp(0, 1)

    def _nearest_azimuth_bin(self, angle: int) -> int:
        if self.num_azimuth_bins <= 0:
            return -1
        if len(self.azimuth_bins) == self.num_azimuth_bins:
            diffs = [min(abs(angle - b) % 360, abs(b - angle) % 360) for b in self.azimuth_bins]
            return int(np.argmin(diffs))
        return int(angle) % self.num_azimuth_bins

    def _parse_one_view(self, v: Any) -> int:
        if v is None:
            return -1
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        raw = str(v)
        if raw in self.view_label_mapping:
            return self.view_label_mapping[raw]
        # Numeric angle or camera id.
        m = re.search(r"-?\d+", raw)
        if m is None:
            return -1
        num = int(m.group(0))
        if str(num) in self.view_label_mapping:
            return self.view_label_mapping[str(num)]
        if num in self.view_label_mapping:
            return self.view_label_mapping[num]
        # For explicit angle datasets: 0/45/90/.../270.
        if abs(num) > self.num_azimuth_bins or num in self.azimuth_bins:
            return self._nearest_azimuth_bin(num % 360)
        # For implicit camera-id datasets: cam0/cam1/... if no mapping is provided.
        if 0 <= num < self.num_azimuth_bins:
            return int(num)
        return -1

    def _parse_view_labels(self, vies, batch_size: int, device) -> torch.Tensor:
        vals = _flatten_meta(vies)
        labels = [self._parse_one_view(v) for v in vals]
        if len(labels) < batch_size:
            labels += [-1] * (batch_size - len(labels))
        labels = labels[:batch_size]
        return torch.as_tensor(labels, dtype=torch.long, device=device)

    def _parse_one_elevation(self, v: Any) -> int:
        """Parse a single view label into an elevation bin index via elevation_label_mapping.

        Lookup order:
          1. Full raw string   e.g. "01_0"
          2. str(int(first_number))  e.g. "1"
        Returns -1 if no mapping matches (elevation loss will ignore this sample).
        """
        if not self.elevation_label_mapping:
            return -1
        if v is None:
            return -1
        if isinstance(v, bytes):
            v = v.decode("utf-8", errors="ignore")
        raw = str(v)
        # 1. Try full label string (e.g. "01_0" or "04_1").
        if raw in self.elevation_label_mapping:
            return self.elevation_label_mapping[raw]
        # 2. Extract first integer and look up.
        m = re.search(r"-?\d+", raw)
        if m is None:
            return -1
        num = int(m.group(0))
        if str(num) in self.elevation_label_mapping:
            return self.elevation_label_mapping[str(num)]
        return -1

    def _parse_elevation_labels(self, vies, batch_size: int, device) -> torch.Tensor:
        vals = _flatten_meta(vies)
        labels = [self._parse_one_elevation(v) for v in vals]
        if len(labels) < batch_size:
            labels += [-1] * (batch_size - len(labels))
        labels = labels[:batch_size]
        return torch.as_tensor(labels, dtype=torch.long, device=device)

    def _build_pseudo_targets(self, gpf: torch.Tensor, phase: Optional[torch.Tensor]):
        n, c, t, h, w = gpf.shape
        mask = gpf[:, 0:1].clamp(0, 1) if c > 0 else torch.ones(n, 1, t, h, w, device=gpf.device, dtype=gpf.dtype)
        if self.reliability_index is not None and self.reliability_index < c:
            rel = gpf[:, self.reliability_index:self.reliability_index + 1].clamp(0, 1)
        else:
            rel = mask
        part_occ = _part_adaptive_pool(mask, self.num_parts).clamp(0, 1)
        part_rel_num = _part_adaptive_pool(mask * rel, self.num_parts)
        part_rel = part_rel_num / (_part_adaptive_pool(mask, self.num_parts) + 1e-6)
        visibility = (part_occ / (part_occ.amax(dim=-1, keepdim=True) + 1e-6)).clamp(0, 1)
        part_conf = (visibility * part_rel.clamp(0, 1)).clamp(0, 1)
        occlusion = (visibility * (1.0 - part_rel.clamp(0, 1)) + (1.0 - visibility) * 0.5).clamp(0, 1)

        col = (mask > 0.2).float().amax(dim=-2).sum(dim=-1) / float(w)  # [N,1,T]
        row = (mask > 0.2).float().amax(dim=-1).sum(dim=-1) / float(h)  # [N,1,T]
        foreshortening = (col / (row + 1e-6)).transpose(1, 2).contiguous().clamp(0, 1)  # [N,T,1]
        if phase is None:
            phase_target = None
        else:
            phase_target = _match_nt_time(phase, t)
        return {
            "visibility_targets": visibility,
            "occlusion_targets": occlusion,
            "part_confidence_targets": part_conf,
            "foreshortening_targets": foreshortening,
            "phase_targets": phase_target,
        }

    def _strategy_part_usage(self, alpha: torch.Tensor, part_conf: Optional[torch.Tensor]):
        # alpha [N,T,E], part_conf [N,T,K] -> [E,K]
        if part_conf is None:
            return None
        pc = _match_nt_time(part_conf, alpha.size(1))
        usage = torch.einsum("nte,ntk->ek", alpha, pc)
        usage = usage / (usage.sum(dim=-1, keepdim=True) + 1e-6)
        return usage

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        gpf = _to_ncthw(ipts[self.input_index], expected_channels=self.source_channels)
        gpf = self.norm(gpf)
        local_bs = labs.size(0) if torch.is_tensor(labs) else len(labs)

        phase = self._get_phase(ipts, gpf)
        reliability_scalar = self._get_reliability_scalar(gpf)
        foreground = gpf[:, 0:1] if gpf.size(1) > 0 else None
        reliability = None
        if self.reliability_index is not None and self.reliability_index < gpf.size(1):
            reliability = gpf[:, self.reliability_index:self.reliability_index + 1]

        geo = _select_channels(gpf, self.geo_indices, "geo_indices")
        motion = _select_channels(gpf, self.motion_indices, "motion_indices")
        residual = _select_channels(gpf, self.residual_indices, "residual_indices") if len(self.residual_indices) > 0 else None
        stem_feat, stem_aux = self.primitive_stem(geo, motion, reliability=reliability, residual=residual, foreground=foreground)

        view_out = self.view_observer(gpf)
        view_token = view_out["view_token"]

        slots, slot_maps = self.slot_encoder(stem_feat, foreground=foreground)
        strategy_feat, alpha = self.strategy_bank(stem_feat, view_token)

        dyn_out = self.dynamics(slots, view_token)
        x = self.backbone(strategy_feat)
        x = self.dynamics.apply_context(x, dyn_out)

        outs = self.TP(x, seqL=seqL, phase=phase, reliability=reliability_scalar, batch_size=local_bs)
        feat = self.HPP(outs)
        feat = self.embed_dropout(feat)  # Regularization dropout
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        embed = embed_2 if self.inference_use_emb2 else embed_1

        # Aux targets. They are intentionally weak/pseudo unless dataset view labels are parseable.
        azimuth_labels = self._parse_view_labels(vies, gpf.size(0), gpf.device)
        elevation_labels = self._parse_elevation_labels(vies, gpf.size(0), gpf.device)
        pseudo = self._build_pseudo_targets(gpf, phase)
        expert_part_usage = self._strategy_part_usage(alpha, pseudo["part_confidence_targets"])
        canonical_embedding = F.normalize(dyn_out["z"].mean(dim=1), dim=-1, eps=1e-6)
        projected_embedding = F.normalize(dyn_out["z_obs"].mean(dim=1), dim=-1, eps=1e-6)

        visual_summary = _build_visual_summary(
            gpf, slot_maps=slot_maps, alpha=alpha, max_frames=self.visualize_max_frames,
            reliability_index=self.reliability_index
        ) if self.visualize else {}

        # Merge stem auxiliary maps only for visualization; training losses use compact states.
        if self.visualize:
            for k, v in stem_aux.items():
                if torch.is_tensor(v):
                    visual_summary[f"image/{k}"] = v.detach().float().mean(dim=1, keepdim=True).permute(0, 2, 1, 3, 4).reshape(-1, 1, v.size(-2), v.size(-1))

        return {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
                "view_observer": {
                    "azimuth_logits": view_out["azimuth_logits"],
                    "azimuth_labels": azimuth_labels,
                    "elevation_logits": view_out["elevation_logits"],
                    "elevation_labels": elevation_labels,
                    "visibility_logits": view_out["visibility_logits"],
                    "visibility_targets": pseudo["visibility_targets"],
                    "occlusion_logits": view_out["occlusion_logits"],
                    "occlusion_targets": pseudo["occlusion_targets"],
                    "part_confidence_logits": view_out["part_confidence_logits"],
                    "part_confidence_targets": pseudo["part_confidence_targets"],
                    "foreshortening": view_out["foreshortening"],
                    "foreshortening_targets": pseudo["foreshortening_targets"],
                    "phase_pred": view_out["phase_pred"],
                    "phase_targets": pseudo["phase_targets"],
                },
                "dynamics": {
                    "z_next_pred": dyn_out["z_next_pred"],
                    "z_next_target": dyn_out["z_next_target"],
                    "z_obs_next_pred": dyn_out["z_obs_next_pred"],
                    "z_obs_next_target": dyn_out["z_obs_next_target"],
                },
                "strategy": {
                    "alpha": alpha,
                    "expert_part_usage": expert_part_usage,
                },
                "view_equiv": {
                    "canonical_embedding": canonical_embedding,
                    "projected_embedding": projected_embedding,
                    "labels": labs,
                    "azimuth_labels": azimuth_labels,
                },
            },
            "visual_summary": visual_summary,
            "inference_feat": {"embeddings": embed},
        }
