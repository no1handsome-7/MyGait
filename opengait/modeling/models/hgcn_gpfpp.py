
# -*- coding: utf-8 -*-
"""
Human-like Gait Cognition Network for GPF++ Core-28.

Drop-in path:
    OpenGait/opengait/modeling/models/hgcn_gpfpp.py

Register:
    from .hgcn_gpfpp import HGCNGPFPP

Expected input layout when CCPG GPF++ Core-28 is used:
    data_in_use: [True, False, True, True, False]
    ipts[0] = 000-gpf.pkl   [B, T, 28, H, W]
    ipts[1] = 002-phase.pkl [B, T, 5]
    ipts[2] = 003-parts.pkl [B, T, K, H, W]

The model follows:
    GPF++
    -> primitive-specific grouped stem
    -> view-observable encoder
    -> part-action tokenizer
    -> factorized heads: z_gait / z_view / z_nui / z_act
    -> cluster prototype head
    -> standard OpenGait triplet + CE plus optional custom losses.

This implementation keeps factor disentanglement clean during training.
For OpenGait evaluation, inference_feat must be a 3D HPP-style tensor [B, C, P],
because evaluator.metric.cuda_dist indexes x.size(2).
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..base_model import BaseModel
except Exception:
    from opengait.modeling.base_model import BaseModel

try:
    from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks
except Exception:
    from opengait.modeling.modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# -----------------------------
# Small utilities
# -----------------------------

class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReverseFn.apply(x, lambd)


def _to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if torch.is_tensor(x):
        return x.detach().cpu().reshape(-1).tolist()
    return [x]


def _safe_int_from_any(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    if torch.is_tensor(x):
        if x.numel() == 0:
            return default
        return int(x.detach().cpu().reshape(-1)[0].item())
    s = str(x)
    m = re.search(r"-?\d+", s)
    if m:
        return int(m.group(0))
    return default


def encode_ccpg_condition(typs: Any, device: torch.device, num_classes: int = 32) -> torch.Tensor:
    """
    Coarse condition labels from CCPG sequence names.

    Typical CCPG strings seen in logs:
        U0_D0, U0_D0_BG, U0_D3, U3_D3, U1_D1, U1_D0, ...

    Encoding:
        base = U * 4 + D, clipped to [0, 15]
        +16 if BG appears.
    If strings are unavailable, returns zeros.
    """
    vals = _to_list(typs)
    out = []
    for v in vals:
        s = str(v)
        m = re.search(r"U(\d+)_D(\d+)", s)
        if m:
            u, d = int(m.group(1)), int(m.group(2))
            lab = min(u, 3) * 4 + min(d, 3)
        else:
            lab = 0
        if "BG" in s.upper():
            lab += 16
        out.append(lab % max(1, int(num_classes)))
    if len(out) == 0:
        out = [0]
    return torch.as_tensor(out, dtype=torch.long, device=device)


def encode_view_label(vies: Any, device: torch.device, num_classes: int = 12) -> torch.Tensor:
    """
    Coarse view labels.
    If view is an angle string such as 000/030/090, use angle / 30.
    Otherwise hash-like numeric fallback from the first integer.
    """
    vals = _to_list(vies)
    out = []
    for v in vals:
        angle = _safe_int_from_any(v, 0)
        if angle > 180 and angle % 10 == 0:
            # Still fine; this line just preserves common angle names.
            pass
        if num_classes <= 1:
            lab = 0
        else:
            # Use 30-degree bins by default. For datasets with 11 views, modulo is safer.
            lab = int(round(angle / 30.0)) % int(num_classes)
        out.append(lab)
    if len(out) == 0:
        out = [0]
    return torch.as_tensor(out, dtype=torch.long, device=device)


def ensure_bcthw(x: torch.Tensor, source_channels: int = 28) -> torch.Tensor:
    """
    Converts common OpenGait tensor layouts to [B, C, T, H, W].
    GPF pkl normally arrives as [B, T, C, H, W].
    """
    if x.dim() == 4:
        # [B, T, H, W] or [B, C, H, W]. Assume sequence of one-channel maps.
        x = x.unsqueeze(2)
    if x.dim() != 5:
        raise ValueError(f"Expected a 5D GPF tensor, got shape={tuple(x.shape)}")

    # [B, T, C, H, W] -> [B, C, T, H, W]
    if x.size(2) == source_channels:
        return x.permute(0, 2, 1, 3, 4).contiguous().float()

    # Already [B, C, T, H, W]
    if x.size(1) == source_channels:
        return x.contiguous().float()

    # Fallback: choose the smaller non-spatial dim as channel if plausible.
    if x.size(1) < x.size(2):
        return x.contiguous().float()
    return x.permute(0, 2, 1, 3, 4).contiguous().float()


def ensure_phase_btf(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        return None
    # OpenGait normally gives [B, T, 5].
    return x.float().contiguous()


def ensure_parts_btkhw(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dim() == 4:
        x = x.unsqueeze(2)
    if x.dim() != 5:
        return None
    # [B,T,K,H,W] is expected. If [B,K,T,H,W], convert.
    if x.size(1) <= 16 and x.size(2) > 16:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.float().contiguous()


# -----------------------------
# Primitive stem
# -----------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class PrimitiveSpecificStem(nn.Module):
    """
    Primitive-specific stem for GPF++ Core-28.

    Groups:
        Geometry: 0-9
        Motion: 10-19
        Frequency/Cycle: 20-23
        Residual/Affinity: 24-27

    It does not treat all channels equally: motion/affinity/residual features are
    gated by reliability / periodicity / texture-risk-inverse primitives.
    """
    def __init__(
        self,
        source_channels: int = 28,
        geo_indices: Sequence[int] = tuple(range(0, 10)),
        motion_indices: Sequence[int] = tuple(range(10, 20)),
        frequency_indices: Sequence[int] = tuple(range(20, 24)),
        affinity_indices: Sequence[int] = tuple(range(24, 28)),
        geo_branch_channels: int = 16,
        motion_branch_channels: int = 16,
        frequency_branch_channels: int = 8,
        affinity_branch_channels: int = 12,
        out_channels: int = 48,
        reliability_index: int = 19,
        periodic_index: int = 18,
        texture_inverse_index: int = 25,
        primitive_gate_strength: float = 0.75,
        primitive_dropout: float = 0.05,
    ):
        super().__init__()
        self.source_channels = source_channels
        self.geo_indices = list(geo_indices)
        self.motion_indices = list(motion_indices)
        self.frequency_indices = list(frequency_indices)
        self.affinity_indices = list(affinity_indices)
        self.reliability_index = reliability_index
        self.periodic_index = periodic_index
        self.texture_inverse_index = texture_inverse_index
        self.primitive_gate_strength = float(primitive_gate_strength)
        self.primitive_dropout = float(primitive_dropout)

        self.geo = nn.Sequential(
            ConvBNAct(len(self.geo_indices), geo_branch_channels, 3),
            ConvBNAct(geo_branch_channels, geo_branch_channels, 3),
        )
        self.motion = nn.Sequential(
            ConvBNAct(len(self.motion_indices), motion_branch_channels, 3),
            ConvBNAct(motion_branch_channels, motion_branch_channels, 3),
        )
        self.freq = nn.Sequential(
            ConvBNAct(len(self.frequency_indices), frequency_branch_channels, 3),
            ConvBNAct(frequency_branch_channels, frequency_branch_channels, 3),
        )
        self.aff = nn.Sequential(
            ConvBNAct(len(self.affinity_indices), affinity_branch_channels, 3),
            ConvBNAct(affinity_branch_channels, affinity_branch_channels, 3),
        )

        fuse_in = geo_branch_channels + motion_branch_channels + frequency_branch_channels + affinity_branch_channels
        self.fuse = nn.Sequential(
            ConvBNAct(fuse_in, out_channels, 1, p=0),
            ConvBNAct(out_channels, out_channels, 3),
        )

        self.group_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(fuse_in, max(8, fuse_in // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, fuse_in // 4), 4, 1),
        )

    def _pick(self, x_btchw: torch.Tensor, idxs: Sequence[int]) -> torch.Tensor:
        return x_btchw[:, idxs]

    def _primitive_map(self, x_btchw: torch.Tensor, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= x_btchw.size(1):
            return torch.ones((x_btchw.size(0), 1, x_btchw.size(2), x_btchw.size(3)),
                              device=x_btchw.device, dtype=x_btchw.dtype)
        return x_btchw[:, idx:idx + 1].clamp(0, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: [B, C, T, H, W]
        returns:
            y: [B, C_out, T, H, W]
        """
        b, c, t, h, w = x.shape
        x2 = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        geo = self.geo(self._pick(x2, self.geo_indices))
        mot = self.motion(self._pick(x2, self.motion_indices))
        frq = self.freq(self._pick(x2, self.frequency_indices))
        aff = self.aff(self._pick(x2, self.affinity_indices))

        rel = self._primitive_map(x2, self.reliability_index)
        per = self._primitive_map(x2, self.periodic_index)
        tex_inv = self._primitive_map(x2, self.texture_inverse_index)

        # Reliability makes motion/affinity stronger only where evidence is trustworthy.
        mot = mot * (1.0 + self.primitive_gate_strength * (0.5 * rel + 0.5 * per))
        aff = aff * (1.0 + self.primitive_gate_strength * (0.7 * tex_inv + 0.3 * rel))
        # Frequency branch is softly aligned to periodic evidence.
        frq = frq * (1.0 + 0.5 * self.primitive_gate_strength * per)

        if self.training and self.primitive_dropout > 0:
            # Drop primitive groups, not random pixels. This discourages shortcut reliance
            # on residual/texture or any single cue.
            keep = torch.empty((b * t, 4, 1, 1), device=x.device).bernoulli_(1.0 - self.primitive_dropout)
            scale = 1.0 / max(1e-6, (1.0 - self.primitive_dropout))
            geo = geo * keep[:, 0:1] * scale
            mot = mot * keep[:, 1:2] * scale
            frq = frq * keep[:, 2:3] * scale
            aff = aff * keep[:, 3:4] * scale

        cat = torch.cat([geo, mot, frq, aff], dim=1)
        raw_group_gate = torch.sigmoid(self.group_gate(cat)).flatten(1)  # [BT,4]

        # Apply sample-level primitive gate.
        g = raw_group_gate.view(b * t, 4, 1, 1)
        geo = geo * (0.5 + g[:, 0:1])
        mot = mot * (0.5 + g[:, 1:2])
        frq = frq * (0.5 + g[:, 2:3])
        aff = aff * (0.5 + g[:, 3:4])
        cat = torch.cat([geo, mot, frq, aff], dim=1)

        y2 = self.fuse(cat)
        y = y2.view(b, t, y2.size(1), h, w).permute(0, 2, 1, 3, 4).contiguous()

        aux = {
            "primitive_group_gate": raw_group_gate.view(b, t, 4).mean(dim=1),  # [B,4]
            "primitive_rel_mean": rel.view(b, t, 1, h, w).mean(dim=(1, 3, 4)).squeeze(1),
            "primitive_periodic_mean": per.view(b, t, 1, h, w).mean(dim=(1, 3, 4)).squeeze(1),
            "primitive_texture_inverse_mean": tex_inv.view(b, t, 1, h, w).mean(dim=(1, 3, 4)).squeeze(1),
        }
        return y, aux


# -----------------------------
# Backbone and temporal pooling
# -----------------------------

class BasicBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, 3, s=stride)
        self.conv2 = ConvBNAct(out_ch, out_ch, 3, act=False)
        self.down = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.down = ConvBNAct(in_ch, out_ch, 1, s=stride, p=0, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.down(x))


class FrameBackbone(nn.Module):
    """
    2D CNN applied frame-wise.
    Input/Output:
        [B, C, T, H, W] -> [B, C_out, T, H', W']
    """
    def __init__(self, in_channels: int, channels=(64, 128, 256, 512), strides=(1, 2, 1, 1), blocks_per_stage=(1, 1, 1, 1)):
        super().__init__()
        layers = []
        cur = in_channels
        for ch, st, nblk in zip(channels, strides, blocks_per_stage):
            layers.append(BasicBlock2D(cur, ch, stride=st))
            cur = ch
            for _ in range(max(0, int(nblk) - 1)):
                layers.append(BasicBlock2D(cur, ch, stride=1))
        self.net = nn.Sequential(*layers)
        self.out_channels = cur

    def forward(self, x):
        b, c, t, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = self.net(y)
        _, co, ho, wo = y.shape
        return y.view(b, t, co, ho, wo).permute(0, 2, 1, 3, 4).contiguous()


class TemporalEvidencePooling(nn.Module):
    """
    Multi-evidence temporal pooling:
        max + mean + reliability-weighted + top-k reliable pooling.
    It deliberately does not use absolute phase as strong FiLM.
    """
    def __init__(self, mode: str = "evidence", topk_ratio: float = 0.5, out_channels: Optional[int] = None):
        super().__init__()
        self.mode = mode
        self.topk_ratio = float(topk_ratio)
        self.out_channels = out_channels
        self.fuse = None

    def _make_fuse(self, c: int, device: torch.device):
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 4, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        ).to(device)

    def forward(self, x: torch.Tensor, reliability: Optional[torch.Tensor] = None, periodic: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,C,T,H,W]
        reliability/periodic: [B,1,T,H,W] in input resolution or feature resolution.
        """
        if self.mode == "max":
            return torch.max(x, dim=2)[0]
        if self.mode == "mean":
            return torch.mean(x, dim=2)

        b, c, t, h, w = x.shape
        f_max = torch.max(x, dim=2)[0]
        f_mean = torch.mean(x, dim=2)

        if reliability is None:
            w_e = torch.ones((b, 1, t, h, w), dtype=x.dtype, device=x.device)
        else:
            w_e = reliability.float()
            if w_e.shape[-2:] != (h, w):
                w_e = F.interpolate(
                    w_e.permute(0, 2, 1, 3, 4).reshape(b * t, 1, w_e.size(-2), w_e.size(-1)),
                    size=(h, w), mode="bilinear", align_corners=False
                ).view(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            if w_e.size(2) != t:
                w_e = F.interpolate(w_e.squeeze(1).unsqueeze(1), size=(t, h, w), mode="trilinear", align_corners=False)
            w_e = w_e.clamp(0, 1)

        if periodic is not None:
            w_p = periodic.float()
            if w_p.shape[-2:] != (h, w):
                w_p = F.interpolate(
                    w_p.permute(0, 2, 1, 3, 4).reshape(b * t, 1, w_p.size(-2), w_p.size(-1)),
                    size=(h, w), mode="bilinear", align_corners=False
                ).view(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
            if w_p.size(2) != t:
                w_p = F.interpolate(w_p.squeeze(1).unsqueeze(1), size=(t, h, w), mode="trilinear", align_corners=False)
            w_e = (0.7 * w_e + 0.3 * w_p.clamp(0, 1)).clamp(0, 1)

        f_rel = (x * (w_e + 1e-4)).sum(dim=2) / ((w_e + 1e-4).sum(dim=2).clamp_min(1e-4))

        # Top-k reliable frames per sample, globally by reliability score.
        score = w_e.mean(dim=(1, 3, 4))  # [B,T]
        k = max(1, int(math.ceil(t * self.topk_ratio)))
        top_idx = score.topk(k=k, dim=1).indices  # [B,k]
        gather_idx = top_idx[:, None, :, None, None].expand(b, c, k, h, w)
        f_topk = x.gather(dim=2, index=gather_idx).mean(dim=2)

        if self.fuse is None:
            self._make_fuse(c, x.device)
        return self.fuse(torch.cat([f_max, f_mean, f_rel, f_topk], dim=1))


# -----------------------------
# View and part-action modules
# -----------------------------

class ViewObservableEncoder(nn.Module):
    """
    Estimate view token and use it to modulate intermediate feature maps.
    z_view is supervised to predict view; z_gait later gets view adversarial.
    """
    def __init__(self, in_channels: int, token_dim: int = 128, view_class_num: int = 12, film: bool = True):
        super().__init__()
        self.film = bool(film)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.token = nn.Sequential(
            nn.Linear(in_channels, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.view_cls = nn.Linear(token_dim, view_class_num)
        if self.film:
            self.to_film = nn.Linear(token_dim, in_channels * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B,C,T,H,W]
        b, c, _, _, _ = x.shape
        pooled = self.pool(x).view(b, c)
        z_view = self.token(pooled)
        view_logits = self.view_cls(z_view)
        if self.film:
            gamma_beta = self.to_film(z_view).view(b, 2, c, 1, 1, 1)
            gamma = torch.tanh(gamma_beta[:, 0]) * 0.5
            beta = gamma_beta[:, 1] * 0.1
            x = x * (1.0 + gamma) + beta
        return x, z_view, view_logits


class PartActionTokenizer(nn.Module):
    """
    Produces body part tokens and action/snippet tokens.
    It uses GPF++ part maps as weak body priors, not as hard semantic parsing.
    """
    def __init__(self, in_channels: int, token_dim: int = 256, num_parts: int = 8, num_action_tokens: int = 4, hpp_parts: int = 32):
        super().__init__()
        self.num_parts = int(num_parts)
        self.num_action_tokens = int(num_action_tokens)
        self.hpp_parts = int(hpp_parts)
        self.part_proj = nn.Sequential(nn.Linear(in_channels, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.action_proj = nn.Sequential(nn.Linear(in_channels, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.token_fuse = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.ReLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.to_hpp_gate = nn.Linear(token_dim, hpp_parts)

    def pool_by_parts(self, feat2d: torch.Tensor, parts: Optional[torch.Tensor]) -> torch.Tensor:
        """
        feat2d: [B,C,H,W]
        parts: [B,T,K,H0,W0], soft maps.
        returns part tokens [B,C,K]
        """
        b, c, h, w = feat2d.shape
        if parts is None:
            # Soft stripes as fallback.
            yy = torch.linspace(0, 1, h, device=feat2d.device, dtype=feat2d.dtype).view(1, 1, h, 1)
            centers = torch.linspace(0.05, 0.95, self.num_parts, device=feat2d.device, dtype=feat2d.dtype).view(1, self.num_parts, 1, 1)
            maps = torch.exp(-0.5 * ((yy - centers) / (0.7 / max(1, self.num_parts))) ** 2)
            maps = maps.expand(b, self.num_parts, h, w)
        else:
            maps = parts.mean(dim=1)  # [B,K,H0,W0]
            if maps.size(1) != self.num_parts:
                if maps.size(1) > self.num_parts:
                    maps = maps[:, :self.num_parts]
                else:
                    pad = torch.zeros((b, self.num_parts - maps.size(1), maps.size(2), maps.size(3)), device=maps.device, dtype=maps.dtype)
                    maps = torch.cat([maps, pad], dim=1)
            if maps.shape[-2:] != (h, w):
                maps = F.interpolate(maps, size=(h, w), mode="bilinear", align_corners=False)
            maps = maps.clamp_min(0)

        denom = maps.sum(dim=(2, 3)).clamp_min(1e-4)  # [B,K]
        part_feat = torch.einsum("bchw,bkhw->bck", feat2d, maps) / denom[:, None, :]
        return part_feat

    def action_tokens(self, feat_seq: torch.Tensor, reliability: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        feat_seq: [B,C,T,H,W]
        returns action tokens [B,C,M]
        """
        b, c, t, h, w = feat_seq.shape
        # 1) static/mean action
        mean_tok = feat_seq.mean(dim=(2, 3, 4))  # [B,C]
        # 2) max evidence action
        max_tok = feat_seq.amax(dim=(2, 3, 4))
        # 3) short temporal change
        if t > 1:
            diff_tok = (feat_seq[:, :, 1:] - feat_seq[:, :, :-1]).abs().mean(dim=(2, 3, 4))
        else:
            diff_tok = mean_tok
        # 4) reliability-weighted action
        if reliability is not None:
            rel = reliability.float()
            if rel.shape[-2:] != (h, w):
                rel = F.interpolate(
                    rel.permute(0, 2, 1, 3, 4).reshape(b * t, 1, rel.size(-2), rel.size(-1)),
                    size=(h, w), mode="bilinear", align_corners=False
                ).view(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
            rel = rel.clamp(0, 1)
            rel_tok = (feat_seq * (rel + 1e-4)).sum(dim=(2, 3, 4)) / (rel + 1e-4).sum(dim=(2, 3, 4)).clamp_min(1e-4)
        else:
            rel_tok = mean_tok

        toks = torch.stack([mean_tok, max_tok, diff_tok, rel_tok], dim=-1)  # [B,C,4]
        if self.num_action_tokens <= 4:
            toks = toks[:, :, :self.num_action_tokens]
        else:
            pad = toks[:, :, -1:].expand(b, c, self.num_action_tokens - 4)
            toks = torch.cat([toks, pad], dim=-1)
        return toks

    def forward(self, feat_seq: torch.Tensor, feat2d: torch.Tensor, parts: Optional[torch.Tensor], reliability: Optional[torch.Tensor] = None):
        """
        returns:
            token_vec: [B, token_dim]
            hpp_gate: [B, 1, P]
            part_tokens: [B, token_dim, K]
            action_tokens: [B, token_dim, M]
        """
        part_feat = self.pool_by_parts(feat2d, parts)  # [B,C,K]
        action_feat = self.action_tokens(feat_seq, reliability=reliability)  # [B,C,M]

        part_tokens = self.part_proj(part_feat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        action_tokens = self.action_proj(action_feat.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

        part_global = part_tokens.mean(dim=-1)
        action_global = action_tokens.mean(dim=-1)
        token_vec = self.token_fuse(torch.cat([part_global, action_global], dim=1))
        hpp_gate = torch.sigmoid(self.to_hpp_gate(token_vec)).unsqueeze(1)
        return token_vec, hpp_gate, part_tokens, action_tokens


class FactorizedHeads(nn.Module):
    def __init__(
        self,
        emb_dim: int = 256,
        z_dim: int = 256,
        class_num: int = 100,
        condition_class_num: int = 32,
        view_class_num: int = 12,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl_lambda = float(grl_lambda)
        self.gait_proj = nn.Sequential(nn.Linear(emb_dim, z_dim), nn.BatchNorm1d(z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, z_dim))
        self.nui_proj = nn.Sequential(nn.Linear(emb_dim, z_dim), nn.BatchNorm1d(z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, z_dim))
        self.act_proj = nn.Sequential(nn.Linear(z_dim, z_dim), nn.BatchNorm1d(z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, z_dim))

        self.nui_condition_cls = nn.Linear(z_dim, condition_class_num)
        self.gait_condition_adv = nn.Linear(z_dim, condition_class_num)

        self.gait_view_adv = nn.Linear(z_dim, view_class_num)
        self.nui_id_adv = nn.Linear(z_dim, class_num)

    def forward(self, gait_base: torch.Tensor, nui_base: torch.Tensor, act_base: torch.Tensor):
        z_gait = self.gait_proj(gait_base)
        z_nui = self.nui_proj(nui_base)
        z_act = self.act_proj(act_base)

        condition_logits = self.nui_condition_cls(z_nui)
        adv_condition_logits = self.gait_condition_adv(grad_reverse(z_gait, self.grl_lambda))
        adv_view_logits = self.gait_view_adv(grad_reverse(z_gait, self.grl_lambda))
        nui_id_logits = self.nui_id_adv(grad_reverse(z_nui, self.grl_lambda))

        return {
            "z_gait": z_gait,
            "z_nui": z_nui,
            "z_act": z_act,
            "condition_logits": condition_logits,
            "adv_condition_logits": adv_condition_logits,
            "adv_view_logits": adv_view_logits,
            "nui_id_logits": nui_id_logits,
        }


class ClusterPrototypeHead(nn.Module):
    """
    Trainable identity and condition prototypes.
    It makes the retrieval/clustering goal explicit while keeping inference on z_gait.
    """
    def __init__(self, class_num: int, condition_class_num: int, emb_dim: int, tau: float = 0.07):
        super().__init__()
        self.class_num = int(class_num)
        self.condition_class_num = int(condition_class_num)
        self.tau = float(tau)
        self.global_proto = nn.Parameter(torch.randn(class_num, emb_dim) * 0.02)
        self.condition_proto = nn.Parameter(torch.randn(class_num, condition_class_num, emb_dim) * 0.02)

    def forward(self, z: torch.Tensor, labels: torch.Tensor, condition_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_n = F.normalize(z, dim=1)
        gp = F.normalize(self.global_proto, dim=1)
        logits = z_n @ gp.t() / self.tau

        labels_c = labels.clamp(0, self.class_num - 1).long()
        cond_c = condition_labels.clamp(0, self.condition_class_num - 1).long()
        cp = F.normalize(self.condition_proto, dim=2)
        selected_cp = cp[labels_c, cond_c]
        selected_gp = gp[labels_c]

        sim_z_global = (z_n * selected_gp).sum(dim=1)
        sim_z_condition = (z_n * selected_cp).sum(dim=1)
        sim_condition_global = (selected_cp * selected_gp).sum(dim=1)

        return {
            "prototype_logits": logits,
            "proto_sim_z_global": sim_z_global,
            "proto_sim_z_condition": sim_z_condition,
            "proto_sim_condition_global": sim_condition_global,
        }


# -----------------------------
# Model
# -----------------------------

class HGCNGPFPP(BaseModel):
    """
    Human-like Gait Cognition Network from GPF++ Core-28.
    """

    def build_network(self, model_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.source_channels = int(model_cfg.get("source_channels", 28))
        self.input_index = int(model_cfg.get("input_index", 0))
        # self.phase_input_index = int(model_cfg.get("phase_input_index", 1))
        # self.parts_input_index = int(model_cfg.get("parts_input_index", 2))
        self.class_num = int(model_cfg.get("class_num", 100))
        self.condition_class_num = int(model_cfg.get("condition_class_num", 32))
        self.view_class_num = int(model_cfg.get("view_class_num", 12))
        self.num_parts = int(model_cfg.get("num_parts", 8))
        self.hpp_parts = int(sum(model_cfg.get("bin_num", [32])))

        self.reliability_index = int(model_cfg.get("reliability_index", 19))
        self.periodic_index = int(model_cfg.get("periodic_index", 18))
        self.texture_inverse_index = int(model_cfg.get("texture_inverse_index", 25))

        self.channel_mean = None
        self.channel_std = None
        if model_cfg.get("enable_channel_norm", True):
            # GPF++ channels are already bounded; this learn-free normalization only centers signed maps less aggressively.
            mean = torch.zeros(self.source_channels)
            std = torch.ones(self.source_channels)
            self.register_buffer("gpf_channel_mean", mean.view(1, self.source_channels, 1, 1, 1), persistent=False)
            self.register_buffer("gpf_channel_std", std.view(1, self.source_channels, 1, 1, 1), persistent=False)
        else:
            self.gpf_channel_mean = None
            self.gpf_channel_std = None

        stem_out = int(model_cfg.get("stem_out_channels", 48))
        self.Stem = PrimitiveSpecificStem(
            source_channels=self.source_channels,
            geo_indices=model_cfg.get("geo_indices", list(range(0, 10))),
            motion_indices=model_cfg.get("motion_indices", list(range(10, 20))),
            frequency_indices=model_cfg.get("frequency_indices", list(range(20, 24))),
            affinity_indices=model_cfg.get("affinity_indices", list(range(24, 28))),
            geo_branch_channels=int(model_cfg.get("geo_branch_channels", 16)),
            motion_branch_channels=int(model_cfg.get("motion_branch_channels", 16)),
            frequency_branch_channels=int(model_cfg.get("frequency_branch_channels", 8)),
            affinity_branch_channels=int(model_cfg.get("affinity_branch_channels", 12)),
            out_channels=stem_out,
            reliability_index=self.reliability_index,
            periodic_index=self.periodic_index,
            texture_inverse_index=self.texture_inverse_index,
            primitive_gate_strength=float(model_cfg.get("primitive_gate_strength", 0.75)),
            primitive_dropout=float(model_cfg.get("primitive_dropout", 0.05)),
        )

        self.ViewEncoder = ViewObservableEncoder(
            in_channels=stem_out,
            token_dim=int(model_cfg.get("view_token_dim", 128)),
            view_class_num=self.view_class_num,
            film=bool(model_cfg.get("use_view_film", True)),
        )

        backbone_channels = model_cfg.get("backbone_channels", [64, 128, 256, 512])
        self.Backbone = FrameBackbone(
            in_channels=stem_out,
            channels=backbone_channels,
            strides=model_cfg.get("backbone_strides", [1, 2, 1, 1]),
            blocks_per_stage=model_cfg.get("blocks_per_stage", [1, 1, 1, 1]),
        )
        feat_ch = self.Backbone.out_channels

        self.TP = TemporalEvidencePooling(
            mode=model_cfg.get("temporal_pool", "evidence"),
            topk_ratio=float(model_cfg.get("temporal_topk_ratio", 0.5)),
        )
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg.get("bin_num", [32]))

        self.Tokenizer = PartActionTokenizer(
            in_channels=feat_ch,
            token_dim=int(model_cfg.get("token_dim", 256)),
            num_parts=self.num_parts,
            num_action_tokens=int(model_cfg.get("num_action_tokens", 4)),
            hpp_parts=self.hpp_parts,
        )
        self.use_part_action_gate = bool(model_cfg.get("use_part_action_gate", True))
        self.part_action_gate_strength = float(model_cfg.get("part_action_gate_strength", 0.35))

        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

        emb_channels = int(model_cfg["SeparateFCs"].get("out_channels", model_cfg.get("emb_channels", 256)))
        self.factor_heads = FactorizedHeads(
            emb_dim=emb_channels,
            z_dim=int(model_cfg.get("factor_dim", emb_channels)),
            class_num=self.class_num,
            condition_class_num=self.condition_class_num,
            view_class_num=self.view_class_num,
            grl_lambda=float(model_cfg.get("grl_lambda", 1.0)),
        )
        self.view_token_to_factor = nn.Linear(int(model_cfg.get("view_token_dim", 128)), int(model_cfg.get("factor_dim", emb_channels)))

        self.prototype_head = ClusterPrototypeHead(
            class_num=self.class_num,
            condition_class_num=self.condition_class_num,
            emb_dim=int(model_cfg.get("factor_dim", emb_channels)),
            tau=float(model_cfg.get("prototype_tau", 0.07)),
        )

        self.inference_use_emb2 = bool(model_cfg.get("inference_use_emb2", False))

    def _primitive_5d(self, x: torch.Tensor, idx: int) -> Optional[torch.Tensor]:
        if idx < 0 or idx >= x.size(1):
            return None
        return x[:, idx:idx + 1].contiguous()

    def _visual_summary(self, gpf: torch.Tensor, pooled_feat: torch.Tensor, aux: Dict[str, torch.Tensor]):
        # Keep it light. Large summaries slow training.
        if not self.model_cfg.get("visualize", False):
            return {}
        with torch.no_grad():
            b, c, t, h, w = gpf.shape
            mid = min(t - 1, t // 2)
            vis = {
                "image/gpf_mask": gpf[:, 0:1, mid].detach(),
                "image/gpf_motion_reliability": gpf[:, self.reliability_index:self.reliability_index + 1, mid].detach(),
                "image/gpf_periodic_energy": gpf[:, self.periodic_index:self.periodic_index + 1, mid].detach(),
                "image/gpf_texture_inverse": gpf[:, self.texture_inverse_index:self.texture_inverse_index + 1, mid].detach(),
                "image/pooled_feature_energy": pooled_feat.pow(2).mean(dim=1, keepdim=True).detach(),
            }
            return vis

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs

        gpf = ensure_bcthw(ipts[self.input_index], source_channels=self.source_channels)
        if self.gpf_channel_mean is not None:
            gpf = (gpf - self.gpf_channel_mean) / self.gpf_channel_std.clamp_min(1e-6)

        phase = None
        parts = None
        # if len(ipts) > self.phase_input_index:
        #     phase = ensure_phase_btf(ipts[self.phase_input_index])
        # if len(ipts) > self.parts_input_index:
        #     parts = ensure_parts_btkhw(ipts[self.parts_input_index])

        device = gpf.device
        labs = labs.long().to(device)
        condition_labels = encode_ccpg_condition(typs, device, self.condition_class_num)
        view_labels = encode_view_label(vies, device, self.view_class_num)
        # Dataloader sometimes returns scalar metadata; expand defensively.
        if condition_labels.numel() != labs.numel():
            condition_labels = condition_labels[:1].expand(labs.numel()).contiguous()
        if view_labels.numel() != labs.numel():
            view_labels = view_labels[:1].expand(labs.numel()).contiguous()

        reliability = self._primitive_5d(gpf, self.reliability_index)
        periodic = self._primitive_5d(gpf, self.periodic_index)

        x, stem_aux = self.Stem(gpf)
        x, z_view_raw, view_logits = self.ViewEncoder(x)
        feat_seq = self.Backbone(x)
        pooled = self.TP(feat_seq, reliability=reliability, periodic=periodic)

        token_vec, hpp_gate, part_tokens, action_tokens = self.Tokenizer(feat_seq, pooled, parts, reliability=reliability)
        hpp_feat = self.HPP(pooled)  # [B,C,P]

        if self.use_part_action_gate:
            hpp_feat = hpp_feat * (1.0 + self.part_action_gate_strength * (hpp_gate - 0.5))

        embed_1 = self.FCs(hpp_feat)      # [B,emb,P]
        embed_2, logits = self.BNNecks(embed_1)

        # Global vectors for factor losses/prototype losses.
        gait_base = embed_1.mean(dim=2)
        # Nuisance branch receives more of the texture/residual/action information via token vector.
        nui_base = gait_base + 0.25 * token_vec[:, :gait_base.size(1)]
        factors = self.factor_heads(gait_base, nui_base, token_vec)
        z_gait = factors["z_gait"]
        z_nui = factors["z_nui"]
        z_act = factors["z_act"]
        z_view = self.view_token_to_factor(z_view_raw)

        proto_out = self.prototype_head(z_gait, labs, condition_labels)

        if embed_1.size(0) != labs.size(0):
            raise RuntimeError(
                f"Batch mismatch before loss: embed_1 N={embed_1.size(0)}, labels N={labs.size(0)}. "
                "Check temporal pooling or input layout."
            )

        # IMPORTANT:
        # OpenGait's evaluator.metric.cuda_dist expects inference embeddings to be
        # a 3D tensor [N, C, P], where P is the HPP/bin dimension.  Returning z_gait
        # directly gives [N, D] and causes:
        #   IndexError: Dimension out of range ... num_bin = x.size(2)
        #
        # Keep z_gait for custom disentanglement/prototype losses, but use the
        # HPP-style embedding for evaluation.
        inference_emb = embed_2 if self.inference_use_emb2 else embed_1
        if inference_emb.dim() != 3:
            raise RuntimeError(
                f"OpenGait evaluator requires 3D inference_feat embeddings [B,C,P], "
                f"but got shape={tuple(inference_emb.shape)}"
            )

        retval = {
            "training_feat": {
                # Standard OpenGait losses.
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},

                # Optional custom losses in hgcn_losses.py.
                "cross_condition_supcon": {
                    "embeddings": z_gait,
                    "labels": labs,
                    "condition_labels": condition_labels,
                    "view_labels": view_labels,
                },
                "prototype_cluster": {
                    "prototype_logits": proto_out["prototype_logits"],
                    "labels": labs,
                    "proto_sim_z_global": proto_out["proto_sim_z_global"],
                    "proto_sim_z_condition": proto_out["proto_sim_z_condition"],
                    "proto_sim_condition_global": proto_out["proto_sim_condition_global"],
                },
                "factor_disentangle": {
                    "z_gait": z_gait,
                    "z_nui": z_nui,
                    "z_view": z_view,
                    "z_act": z_act,
                    "condition_logits": factors["condition_logits"],
                    "adv_condition_logits": factors["adv_condition_logits"],
                    "view_logits": view_logits,
                    "adv_view_logits": factors["adv_view_logits"],
                    "nui_id_logits": factors["nui_id_logits"],
                    "labels": labs,
                    "condition_labels": condition_labels,
                    "view_labels": view_labels,
                },
                "view_observable": {
                    "primitive_group_gate": stem_aux["primitive_group_gate"],
                    "reliability_mean": stem_aux["primitive_rel_mean"],
                    "periodic_mean": stem_aux["primitive_periodic_mean"],
                    "texture_inverse_mean": stem_aux["primitive_texture_inverse_mean"],
                },
                # "part_action": {
                #     "part_tokens": part_tokens,
                #     "action_tokens": action_tokens,
                # },
            },
            "visual_summary": self._visual_summary(gpf, pooled, stem_aux),
            "inference_feat": {"embeddings": inference_emb},
        }
        return retval
