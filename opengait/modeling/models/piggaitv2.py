"""PIGGait: Phase-aware Intrinsic Gait Field for RGB gait recognition.

This file is intentionally self-contained so it can be dropped into
`opengait/modeling/models/` and selected by `model_cfg.model: PIGGait`.
The implementation follows three stages:
  A) RGB-to-PIG preprocessing: DINOv2/fallback structure, foreground mask,
     BiggerGait-style human-space FPN, adaptive YCbCr-DCT canonicalization, phase alignment and cycle residuals.
  B) Cycle-aware encoder: static/dynamic streams, slow-fast temporal fusion,
     phase bottleneck, reliability-aware HPP.
  C) Causal nuisance disentanglement: gait/style split, GRL nuisance adversary,
     style prediction, MixStyle/AdaIN counterfactual consistency, REx penalty.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, PackSequenceWrapper, SeparateFCs, SeparateBNNecks


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _as_pair(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2
        return int(x[0]), int(x[1])
    return int(x), int(x)


class GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return GradientReverseFn.apply(x, lambd)


def _dct_matrix(n: int, device=None, dtype=None):
    """Orthonormal DCT-II matrix."""
    mat = torch.empty(n, n, device=device, dtype=dtype)
    scale0 = math.sqrt(1.0 / n)
    scale = math.sqrt(2.0 / n)
    for k in range(n):
        for i in range(n):
            alpha = scale0 if k == 0 else scale
            mat[k, i] = alpha * math.cos(math.pi / n * (i + 0.5) * k)
    return mat


def _logit(p: torch.Tensor, eps: float = 1e-4):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Module A: RGB-to-PIG preprocessing
# ---------------------------------------------------------------------------


class AdaptiveDCTCanonicalizer(nn.Module):
    """YCbCr + part-aware block-DCT photometric shortcut suppression.

    The DC and chroma coefficients are learnably gated per vertical body part.
    This is a differentiable upgrade of fixed JPEG-like filtering: the initial
    gate suppresses chroma/DC shortcuts, while training can recover frequencies
    that are useful for gait.
    """

    def __init__(
        self,
        block_size: int = 8,
        num_parts: int = 16,
        init_y_dc: float = 0.25,
        init_chroma: float = 0.25,
        init_chroma_dc: float = 0.05,
        init_high: float = 0.85,
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.num_parts = int(num_parts)

        init = torch.full((num_parts, 3, self.block_size, self.block_size), float(init_high))
        init[:, 0, 0, 0] = float(init_y_dc)          # luminance DC
        init[:, 1:, :, :] = float(init_chroma)       # chroma coefficients
        init[:, 1:, 0, 0] = float(init_chroma_dc)    # chroma DC, strongest shortcut
        self.freq_gate_logits = nn.Parameter(_logit(init))

    @staticmethod
    def rgb_to_ycbcr(rgb):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.564 * (b - y) + 0.5
        cr = 0.713 * (r - y) + 0.5
        return torch.cat([y, cb, cr], dim=1).clamp(0.0, 1.0)

    @staticmethod
    def ycbcr_to_rgb(ycbcr):
        y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2] - 0.5, ycbcr[:, 2:3] - 0.5
        r = y + 1.403 * cr
        g = y - 0.344 * cb - 0.714 * cr
        b = y + 1.773 * cb
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    def forward(self, rgb01):
        """Args: rgb01 [B, 3, H, W] in [0,1]. Returns canonical RGB [B,3,H,W]."""
        b, c, h, w = rgb01.shape
        bs = self.block_size
        ycbcr = self.rgb_to_ycbcr(rgb01)

        pad_h = (bs - h % bs) % bs
        pad_w = (bs - w % bs) % bs
        if pad_h or pad_w:
            ycbcr = F.pad(ycbcr, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = ycbcr.shape[-2:]
        nh, nw = hp // bs, wp // bs

        patches = F.unfold(ycbcr, kernel_size=bs, stride=bs)  # [B, 3*bs*bs, L]
        patches = patches.transpose(1, 2).reshape(b, nh * nw, c, bs, bs)
        dct = _dct_matrix(bs, device=rgb01.device, dtype=rgb01.dtype)
        coeff = torch.einsum("uv,blcvw,wx->blcux", dct, patches, dct.t())

        gate = torch.sigmoid(self.freq_gate_logits).to(dtype=rgb01.dtype, device=rgb01.device)
        rows = torch.arange(nh, device=rgb01.device).repeat_interleave(nw)
        part = ((rows.float() + 0.5) / max(float(nh), 1.0) * self.num_parts).long().clamp(0, self.num_parts - 1)
        coeff = coeff * gate[part].unsqueeze(0)  # [B,L,3,bs,bs]

        recon = torch.einsum("uv,blcvw,wx->blcux", dct.t(), coeff, dct)
        recon = recon.reshape(b, nh * nw, c * bs * bs).transpose(1, 2)
        recon = F.fold(recon, output_size=(hp, wp), kernel_size=bs, stride=bs)
        recon = recon[..., :h, :w]
        return self.ycbcr_to_rgb(recon)


class ResizeToHW(nn.Module):
    """Bilinear resize wrapper used inside HumanSpaceFPN."""

    def __init__(self, target_size):
        super().__init__()
        self.target_size = tuple(target_size)

    def forward(self, x):
        return F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)


class HumanSpaceFPN(nn.Module):
    """BiggerGait-style all-layer DINOv2 human-space projection.

    Instead of directly using sparse DINOv2 layers as gait features, this module
    consumes all transformer layers, groups consecutive layers, projects each
    group into a compact human-space, and returns a soft human mask.
    """

    def __init__(self, dino: nn.Module, cfg: Dict, dino_dim: int, freeze_dino: bool = True):
        super().__init__()
        self.dino = dino
        self.freeze_dino = bool(freeze_dino)
        self.dino_size = _as_pair(cfg.get("dino_size", [448, 224]))
        self.field_size = _as_pair(cfg.get("field_size", [64, 32]))
        self.hidden_dim = int(cfg.get("dino_dim", dino_dim))
        self.total_layer_num = int(cfg.get("total_layer_num", 12))
        self.group_layer_num = int(cfg.get("group_layer_num", 2))
        self.num_unknown = int(cfg.get("num_unknown", 8))
        self.micro_batch = int(cfg.get("dino_micro_batch", 64))

        assert self.total_layer_num % self.group_layer_num == 0, (
            "total_layer_num must be divisible by group_layer_num"
        )
        self.num_groups = self.total_layer_num // self.group_layer_num
        self.out_channels = self.num_groups * self.num_unknown

        self.human_convs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(self.hidden_dim * self.group_layer_num, affine=False),
                nn.Conv2d(self.hidden_dim * self.group_layer_num, self.hidden_dim // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.hidden_dim // 2, affine=False),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim // 2, self.num_unknown, kernel_size=1, bias=True),
                ResizeToHW(self.field_size),
                nn.BatchNorm2d(self.num_unknown, affine=False),
                nn.Sigmoid(),
            )
            for _ in range(self.num_groups)
        ])

        self.mask_head = nn.Sequential(
            nn.BatchNorm2d(self.hidden_dim, affine=False),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim // 2, 1, kernel_size=1, bias=True),
            ResizeToHW(self.field_size),
            nn.Sigmoid(),
        )

    def forward(self, rgb_norm_flat):
        """Args: rgb_norm_flat [B,3,H,W]. Returns human_space and human_mask."""
        x = F.interpolate(rgb_norm_flat, size=self.dino_size, mode="bilinear", align_corners=False)
        patch_h = self.dino_size[0] // 14
        patch_w = self.dino_size[1] // 14

        all_feats, all_masks = [], []
        for x_chunk in x.split(self.micro_batch, dim=0):
            with torch.set_grad_enabled((not self.freeze_dino) and self.training):
                hidden_states = self.dino(x_chunk, output_hidden_states=True).hidden_states
            hidden_states = hidden_states[1:1 + self.total_layer_num]
            if len(hidden_states) < self.total_layer_num:
                raise RuntimeError(
                    f"DINOv2 returned only {len(hidden_states)} layers, "
                    f"but total_layer_num={self.total_layer_num}."
                )

            patch_tokens = [h[:, 1:].contiguous() for h in hidden_states]
            concat_tokens = torch.cat(patch_tokens, dim=-1)
            concat_tokens = F.layer_norm(
                concat_tokens,
                normalized_shape=(self.hidden_dim * self.total_layer_num,),
            )

            b = x_chunk.size(0)
            concat_map = concat_tokens.view(
                b, patch_h, patch_w, self.hidden_dim * self.total_layer_num
            ).permute(0, 3, 1, 2).contiguous()

            layer_maps = list(torch.chunk(concat_map, self.total_layer_num, dim=1))
            group_feats = []
            for i in range(0, self.total_layer_num, self.group_layer_num):
                group_id = i // self.group_layer_num
                group_map = torch.cat(layer_maps[i:i + self.group_layer_num], dim=1).contiguous()
                group_feats.append(self.human_convs[group_id](group_map))
            human_space = torch.cat(group_feats, dim=1)

            last_tokens = F.layer_norm(patch_tokens[-1], normalized_shape=(self.hidden_dim,))
            last_map = last_tokens.view(b, patch_h, patch_w, self.hidden_dim).permute(0, 3, 1, 2).contiguous()
            human_mask = self.mask_head(last_map)

            all_feats.append(human_space)
            all_masks.append(human_mask)

        return torch.cat(all_feats, dim=0), torch.cat(all_masks, dim=0)


class RGBToPIGField(nn.Module):
    """Builds a phase-aware intrinsic gait field from normalized RGB clips."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.field_size = _as_pair(cfg.get("field_size", [64, 32]))
        self.dino_size = _as_pair(cfg.get("dino_size", [448, 224]))
        self.field_channels = int(cfg.get("field_channels", 32))
        self.use_dino = bool(cfg.get("use_dino", True))
        self.freeze_dino = bool(cfg.get("freeze_dino", True))
        self.feature_mode = str(cfg.get("feature_mode", "human_space_fpn" if self.use_dino else "rgb_fallback"))
        self.selected_layers = cfg.get("selected_layers", [-1])
        self.cycle_shift = int(cfg.get("cycle_shift", 8))
        self.phase_align = bool(cfg.get("phase_align", True))
        self.mask_temperature = float(cfg.get("mask_temperature", 1.0))
        self.mask_floor = float(cfg.get("mask_floor", 0.5))
        self.mask_blend = float(cfg.get("mask_blend", 0.5))

        # PIGGaitPlusPlus / LRCE experiment:
        # 1) reduce RGB photometric shortcut from the DCT photo branch;
        # 2) lightly regularize the fused human-space field.
        self.photo_weight = float(cfg.get("photo_weight", 1.0))
        self.field_dropout = nn.Dropout2d(p=float(cfg.get("field_dropout", 0.0)))

        mean = torch.tensor(cfg.get("rgb_mean", [0.485, 0.456, 0.406])).view(1, 3, 1, 1)
        std = torch.tensor(cfg.get("rgb_std", [0.229, 0.224, 0.225])).view(1, 3, 1, 1)
        self.register_buffer("rgb_mean", mean, persistent=False)
        self.register_buffer("rgb_std", std, persistent=False)

        dct_cfg = cfg.get("dct_cfg", {})
        self.dct = AdaptiveDCTCanonicalizer(**dct_cfg)
        self.photo_proj = nn.Sequential(
            nn.Conv2d(3, self.field_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.field_channels),
            nn.GELU(),
        )

        if self.use_dino:
            pretrained = cfg.get("pretrained_lvm", None)
            if pretrained is None:
                raise ValueError("RGBToPIGField.use_dino=True requires `pretrained_lvm` in rgb_pig_cfg.")
            from transformers import Dinov2Config, Dinov2Model
            dino_config = Dinov2Config.from_pretrained(pretrained)
            self.dino = Dinov2Model.from_pretrained(
                pretrained,
                config=dino_config,
                use_safetensors=bool(cfg.get("use_safetensors", False)),
            )
            dino_dim = int(dino_config.hidden_size)
            if self.freeze_dino:
                self.dino.eval()
                self.dino.requires_grad_(False)

            if self.feature_mode == "human_space_fpn":
                self.human_space_fpn = HumanSpaceFPN(self.dino, cfg, dino_dim=dino_dim, freeze_dino=self.freeze_dino)
                self.human_proj = nn.Sequential(
                    nn.Conv2d(self.human_space_fpn.out_channels, self.field_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.field_channels),
                    nn.GELU(),
                )
            else:
                num_layers = len(self.selected_layers)
                self.dino_proj = nn.Sequential(
                    nn.Conv2d(dino_dim * num_layers, self.field_channels, 1, bias=False),
                    nn.BatchNorm2d(self.field_channels),
                    nn.GELU(),
                )
        else:
            self.dino = None
            self.rgb_fallback = nn.Sequential(
                nn.Conv2d(3, self.field_channels, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.field_channels),
                nn.GELU(),
                nn.Conv2d(self.field_channels, self.field_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.field_channels),
                nn.GELU(),
            )

        self.static_fuse = nn.Sequential(
            nn.Conv2d(self.field_channels * 2, self.field_channels, 1, bias=False),
            nn.BatchNorm2d(self.field_channels),
            nn.GELU(),
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.field_channels, self.field_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.field_channels // 2),
            nn.GELU(),
            nn.Conv2d(self.field_channels // 2, 1, 1),
        )
        self.dynamic_fuse = nn.Sequential(
            nn.Conv3d(self.field_channels * 2, self.field_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(self.field_channels),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_dino and self.freeze_dino:
            self.dino.eval()
        return self

    def denormalize_rgb(self, x):
        return (x * self.rgb_std.to(x) + self.rgb_mean.to(x)).clamp(0.0, 1.0)

    def _extract_structure(self, rgb_norm):
        """Args: rgb_norm [N,S,3,H,W]. Returns struct [N,S,C,Hf,Wf], mask [N,1,S,Hf,Wf]."""
        n, s = rgb_norm.shape[:2]
        flat = rearrange(rgb_norm, "n s c h w -> (n s) c h w")

        if self.use_dino and self.feature_mode == "human_space_fpn":
            human_space, human_mask = self.human_space_fpn(flat)
            feat = self.human_proj(human_space)
            feat = F.interpolate(feat, size=self.field_size, mode="bilinear", align_corners=False)
            human_mask = F.interpolate(human_mask, size=self.field_size, mode="bilinear", align_corners=False)
            return (
                rearrange(feat, "(n s) c h w -> n s c h w", n=n, s=s),
                rearrange(human_mask, "(n s) c h w -> n c s h w", n=n, s=s),
            )

        if self.use_dino:
            din = F.interpolate(flat, size=self.dino_size, mode="bilinear", align_corners=False)
            with torch.set_grad_enabled((not self.freeze_dino) and self.training):
                outs = self.dino(din, output_hidden_states=True).hidden_states
            feats = []
            for lid in self.selected_layers:
                z = outs[int(lid)][:, 1:]
                hp = self.dino_size[0] // 14
                wp = self.dino_size[1] // 14
                z = rearrange(z, "b (h w) c -> b c h w", h=hp, w=wp)
                feats.append(z)
            feat = self.dino_proj(torch.cat(feats, dim=1))
        else:
            rgb01 = self.denormalize_rgb(flat)
            rgb01 = F.interpolate(rgb01, size=self.dino_size, mode="bilinear", align_corners=False)
            feat = self.rgb_fallback(rgb01)

        feat = F.interpolate(feat, size=self.field_size, mode="bilinear", align_corners=False)
        mask = torch.ones(feat.size(0), 1, self.field_size[0], self.field_size[1], dtype=feat.dtype, device=feat.device)
        return rearrange(feat, "(n s) c h w -> n s c h w", n=n, s=s), rearrange(mask, "(n s) c h w -> n c s h w", n=n, s=s)

    def _phase_roll(self, x):
        """Align the clip start to the strongest motion event. x: [N,C,S,H,W]."""
        if (not self.phase_align) or x.size(2) < 3:
            return x, torch.zeros(x.size(0), device=x.device, dtype=torch.long)
        energy = (x[:, :, 1:] - x[:, :, :-1]).pow(2).mean(dim=(1, 3, 4))
        starts = energy.argmax(dim=1) + 1
        aligned = []
        for i, st in enumerate(starts.tolist()):
            aligned.append(torch.roll(x[i:i + 1], shifts=-st, dims=2))
        return torch.cat(aligned, dim=0), starts

    def _cycle_residuals(self, x):
        """Adjacent and cycle-shift residuals, x [N,C,S,H,W]."""
        n, c, s, h, w = x.shape
        adj = torch.zeros_like(x)
        adj[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        p = min(max(1, self.cycle_shift), max(1, s - 1))
        cyc = torch.zeros_like(x)
        if s > 1:
            cyc[:, :, p:] = x[:, :, p:] - x[:, :, :-p]
        return adj, cyc

    def _phase_bottleneck_loss(self, x):
        """A light self-similarity loss: same-cycle lag should be more similar."""
        n, c, s, h, w = x.shape
        if s <= self.cycle_shift + 2:
            return x.new_tensor(0.0)
        z = F.normalize(x.mean(dim=(3, 4)).transpose(1, 2), dim=-1)
        p = min(self.cycle_shift, s - 1)
        pos = (z[:, p:] * z[:, :-p]).sum(-1).mean()
        neg_shift = max(1, p // 2)
        neg = (z[:, neg_shift:] * z[:, :-neg_shift]).sum(-1).mean()
        return F.relu(neg - pos + 0.1)

    def _pseudo_nuisance_label(self, rgb_norm, mask, nuisance_classes=8):
        """Quantize Y/Cb/Cr means into pseudo nuisance classes."""
        n, s = rgb_norm.shape[:2]
        rgb01 = self.denormalize_rgb(rearrange(rgb_norm, "n s c h w -> (n s) c h w"))
        rgb01 = F.interpolate(rgb01, size=self.field_size, mode="bilinear", align_corners=False)
        ycbcr = self.dct.rgb_to_ycbcr(rgb01).view(n, s, 3, *self.field_size)
        m = mask.detach().transpose(1, 2).clamp(0.0, 1.0)
        denom = m.sum(dim=(1, 3, 4)).clamp_min(1.0)
        stat = (ycbcr * m).sum(dim=(1, 3, 4)) / denom
        bits = torch.isfinite(stat).logical_and(stat > 0.5).long()
        label = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]
        if nuisance_classes != 8:
            label = label % int(nuisance_classes)
        return label.clamp(0, int(nuisance_classes) - 1).long()

    def forward(self, rgb_norm, nuisance_classes: int = 8):
        """Returns static field, dynamic field and auxiliary maps/losses.

        Args:
            rgb_norm: [N,S,3,H,W], ImageNet-normalized RGB.
        """
        n, s = rgb_norm.shape[:2]
        flat = rearrange(rgb_norm, "n s c h w -> (n s) c h w")
        rgb01 = self.denormalize_rgb(flat)
        rgb_small = F.interpolate(rgb01, size=self.field_size, mode="bilinear", align_corners=False)

        photo = self.dct(rgb_small)
        photo_feat = self.photo_proj(photo)

        # Reduce the contribution of photometric RGB branch so the model relies
        # more on human-space DINO features and less on clothing/color cues.
        photo_feat = photo_feat * self.photo_weight

        struct, human_mask = self._extract_structure(rgb_norm)
        struct_2d = rearrange(struct, "n s c h w -> (n s) c h w")

        static_2d = self.static_fuse(torch.cat([struct_2d, photo_feat], dim=1))

        # Light field-level dropout to reduce overfitting to training IDs.
        static_2d = self.field_dropout(static_2d)
        learned_mask = torch.sigmoid(self.mask_head(static_2d) / self.mask_temperature)
        learned_mask = self.mask_floor + (1.0 - self.mask_floor) * learned_mask
        human_mask_2d = rearrange(human_mask, "n c s h w -> (n s) c h w")
        human_mask_2d = self.mask_floor + (1.0 - self.mask_floor) * human_mask_2d
        mask = (1.0 - self.mask_blend) * learned_mask + self.mask_blend * human_mask_2d
        mask = mask.clamp(self.mask_floor, 1.0)

        static_2d = static_2d * mask
        static = rearrange(static_2d, "(n s) c h w -> n c s h w", n=n, s=s)
        mask = rearrange(mask, "(n s) c h w -> n c s h w", n=n, s=s)

        static, starts = self._phase_roll(static)
        if self.phase_align and s >= 3:
            aligned_mask = []
            for i, st in enumerate(starts.tolist()):
                aligned_mask.append(torch.roll(mask[i:i + 1], shifts=-st, dims=2))
            mask = torch.cat(aligned_mask, dim=0)

        adj, cyc = self._cycle_residuals(static)
        dynamic = self.dynamic_fuse(torch.cat([adj, cyc], dim=1)) * mask

        motion_energy = (adj.abs().mean(dim=1, keepdim=True) + cyc.abs().mean(dim=1, keepdim=True)).detach()
        reliability = (mask.detach() * (1.0 + motion_energy)).clamp(0.0, 2.0)
        phase_loss = self._phase_bottleneck_loss(static)
        pseudo_label = self._pseudo_nuisance_label(rgb_norm, mask, nuisance_classes)

        aux = {
            "mask": mask,
            "human_mask": human_mask,
            "reliability": reliability,
            "phase_loss": phase_loss,
            "pseudo_nuisance_label": pseudo_label,
            "rgb01_small": rgb_small,
            "photo": photo,
        }
        return static, dynamic, aux


# ---------------------------------------------------------------------------
# Module B: cycle-aware gait encoder
# ---------------------------------------------------------------------------


class CycleTemporalFusion(nn.Module):
    """Slow-fast temporal fusion with a cycle-shift residual branch."""

    def __init__(self, channels: int, cycle_shift: int = 8):
        super().__init__()
        self.cycle_shift = int(cycle_shift)
        self.fast = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        self.slow = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=channels, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.out = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        fast = self.fast(x)
        p = min(max(1, self.cycle_shift), max(1, x.size(2) - 1))
        cyc = torch.zeros_like(x)
        if x.size(2) > 1:
            cyc[:, :, p:] = x[:, :, p:] - x[:, :, :-p]
        slow = self.slow(x + cyc)
        return self.out(torch.cat([fast, slow], dim=1))


class PartReliabilityHPP(nn.Module):
    """Horizontal pyramid pooling modulated by foreground/motion reliability."""

    def __init__(self, bin_num: Optional[List[int]] = None, gamma: float = 0.5):
        super().__init__()
        self.bin_num = bin_num if bin_num is not None else [16]
        self.gamma = float(gamma)

    @property
    def parts_num(self):
        return int(sum(self.bin_num))

    def forward(self, x, reliability=None):
        # x: [N,C,H,W], reliability: [N,1,H0,W0]
        n, c, h, w = x.shape
        if reliability is not None:
            reliability = F.interpolate(reliability, size=(h, w), mode="bilinear", align_corners=False).clamp(0.0, 2.0)
        feats = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            feat = z.mean(-1) + z.max(-1)[0]
            if reliability is not None:
                r = reliability.view(n, 1, b, -1).mean(-1)
                feat = feat * (1.0 + self.gamma * (r - r.mean(dim=2, keepdim=True)))
            feats.append(feat)
        return torch.cat(feats, dim=-1)


class MixStyle3D(nn.Module):
    """AdaIN/MixStyle-style counterfactual perturbation for PIG fields."""

    def __init__(self, p: float = 0.5, alpha: float = 0.3):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)

    def forward(self, x):
        if (not self.training) or x.size(0) < 2 or torch.rand(1, device=x.device) > self.p:
            return x
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sig = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False).add(1e-6).sqrt()
        x_norm = (x - mu) / sig
        perm = torch.randperm(x.size(0), device=x.device)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((x.size(0), 1, 1, 1, 1)).to(x.device, x.dtype)
        mu_mix = lmda * mu + (1.0 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1.0 - lmda) * sig[perm]
        return x_norm * sig_mix + mu_mix


class CycleAwareEncoder(nn.Module):
    def __init__(self, model_cfg: Dict, field_channels: int):
        super().__init__()
        backbone_cfg = copy.deepcopy(model_cfg["backbone_cfg"])
        static_cfg = copy.deepcopy(backbone_cfg)
        dynamic_cfg = copy.deepcopy(backbone_cfg)
        static_cfg["in_channel"] = int(field_channels)
        dynamic_cfg["in_channel"] = int(field_channels)

        # Reuse OpenGait backbone builder through a tiny local factory.
        from .. import backbones
        from utils import get_attr_from, get_valid_args

        def build_backbone(cfg):
            Backbone = get_attr_from([backbones], cfg["type"])
            valid_args = get_valid_args(Backbone, cfg, ["type"])
            return Backbone(**valid_args)

        self.static_backbone = SetBlockWrapper(build_backbone(static_cfg))
        self.dynamic_backbone = SetBlockWrapper(build_backbone(dynamic_cfg))
        out_channels = int(backbone_cfg.get("channels", [64, 128, 256, 512])[-1])
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )
        self.temporal = CycleTemporalFusion(out_channels, cycle_shift=model_cfg.get("cycle_shift", 8))
        self.tp = PackSequenceWrapper(torch.max)
        self.hpp = PartReliabilityHPP(bin_num=model_cfg.get("bin_num", [16]), gamma=model_cfg.get("reliability_gamma", 0.5))

    def forward(self, static, dynamic, reliability, seqL=None):
        s_feat = self.static_backbone(static)
        d_feat = self.dynamic_backbone(dynamic)
        x = self.fuse(torch.cat([s_feat, d_feat], dim=1))
        x = self.temporal(x)
        x = self.tp(x, seqL, options={"dim": 2})[0]
        if reliability is not None:
            rel = reliability.mean(dim=2)  # [N,1,H,W]
        else:
            rel = None
        feat = self.hpp(x, rel)
        return feat


# ---------------------------------------------------------------------------
# Module C + OpenGait model wrapper
# ---------------------------------------------------------------------------


class PIGGait(BaseModel):
    """OpenGait-compatible model."""

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        pig_cfg = model_cfg.get("rgb_pig_cfg", {})
        self.nuisance_classes = int(model_cfg.get("nuisance_classes", 8))
        self.loss_weights = model_cfg.get("loss_weights", {})
        self.ce_scale = float(model_cfg.get("rex_ce_scale", 16.0))
        self.grl_lambda = float(model_cfg.get("grl_lambda", 1.0))
        self.input_normalization = str(model_cfg.get("input_normalization", "imagenet"))
        self.enable_nuisance_losses = bool(model_cfg.get("enable_nuisance_losses", False))

        self.PIG = RGBToPIGField(pig_cfg)
        field_channels = int(pig_cfg.get("field_channels", 32))
        encoder_cfg = copy.deepcopy(model_cfg)
        encoder_cfg["cycle_shift"] = pig_cfg.get("cycle_shift", model_cfg.get("cycle_shift", 8))
        self.Encoder = CycleAwareEncoder(encoder_cfg, field_channels=field_channels)

        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        emb_dim = int(model_cfg["SeparateFCs"]["out_channels"])

        # Stage-1 should not register nuisance/adversarial parameters when the
        # corresponding losses are disabled. Otherwise DDP with
        # find_unused_parameters=False will fail because these parameters do not
        # participate in the loss graph.
        if self.enable_nuisance_losses:
            self.style_projector = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.BatchNorm1d(emb_dim),
                nn.GELU(),
                nn.Dropout(model_cfg.get("style_dropout", 0.1)),
            )
            self.style_classifier = nn.Linear(emb_dim, self.nuisance_classes)
            self.adv_classifier = nn.Linear(emb_dim, self.nuisance_classes)
        else:
            self.style_projector = None
            self.style_classifier = None
            self.adv_classifier = None

        self.mixstyle = MixStyle3D(**model_cfg.get("mixstyle_cfg", {"p": 0.5, "alpha": 0.3}))
        self.enable_counterfactual = bool(model_cfg.get("enable_counterfactual", True))

    def init_parameters(self):
        # Same policy as BaseModel, but avoid reinitializing pretrained DINO weights.
        for name, m in self.named_modules():
            if ".dino" in name:
                continue
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        if hasattr(self, "PIG") and self.PIG.use_dino and self.PIG.freeze_dino:
            self.PIG.dino.eval()
            self.PIG.dino.requires_grad_(False)

    def _encode_to_embedding(self, rgb, seqL=None, return_aux=False, counterfactual=False):
        static, dynamic, aux = self.PIG(rgb, nuisance_classes=self.nuisance_classes)
        if counterfactual:
            static = self.mixstyle(static)
            dynamic = self.mixstyle(dynamic)
        feat = self.Encoder(static, dynamic, aux["reliability"], seqL=seqL)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)
        if return_aux:
            return embed_1, embed_2, logits, aux
        return embed_1, embed_2, logits

    def _split_forward_for_all_ordered(self, rgb, seqL):
        lengths = seqL[0].detach().cpu().numpy().tolist()
        chunks = torch.split(rgb[:, :sum(lengths)], lengths, dim=1)
        embed_list, logit_list = [], []
        for chunk in chunks:
            emb, _, logit = self._encode_to_embedding(chunk, seqL=None, return_aux=False)
            embed_list.append(emb)
            logit_list.append(logit)
        return torch.cat(embed_list, dim=0), torch.cat(logit_list, dim=0)

    def _nuisance_losses(self, embed, aux):
        if not self.enable_nuisance_losses:
            raise RuntimeError("_nuisance_losses() was called while enable_nuisance_losses=False.")
        pooled = embed.mean(dim=-1)  # [N,C]
        style_feat = self.style_projector(pooled)
        style_logits = self.style_classifier(style_feat).unsqueeze(-1)
        adv_logits = self.adv_classifier(grad_reverse(pooled, self.grl_lambda)).unsqueeze(-1)
        orth = (F.normalize(pooled, dim=1) * F.normalize(style_feat, dim=1)).sum(dim=1).pow(2).mean()
        return style_logits, adv_logits, style_feat, orth

    def _rex_loss(self, logits, labels, envs):
        if envs is None or len(envs) != logits.size(0):
            return logits.new_tensor(0.0)
        losses = []
        logits_mean = logits.mean(dim=-1) * self.ce_scale
        per = F.cross_entropy(logits_mean.float(), labels, reduction="none")
        env_to_idx = {}
        for i, e in enumerate(envs):
            env_to_idx.setdefault(str(e), []).append(i)
        if len(env_to_idx) <= 1:
            return logits.new_tensor(0.0)
        for idxs in env_to_idx.values():
            losses.append(per[torch.as_tensor(idxs, device=per.device)].mean())
        losses = torch.stack(losses)
        return losses.var(unbiased=False)

    def _normalize_input_rgb(self, rgb):
        """Convert configured RGB input to ImageNet-normalized CHW tensors."""
        mode = self.input_normalization.lower()
        if mode in ["imagenet", "imagenet_norm", "normalized"]:
            return rgb.float()
        mean = self.PIG.rgb_mean.to(rgb).view(1, 1, 3, 1, 1)
        std = self.PIG.rgb_std.to(rgb).view(1, 1, 3, 1, 1)
        if mode in ["uint8", "raw_uint8"]:
            rgb01 = rgb.float() / 255.0
        elif mode in ["zero_one", "01", "float01"]:
            rgb01 = rgb.float().clamp(0.0, 1.0)
        else:
            raise ValueError(f"Unknown input_normalization={self.input_normalization}")
        return (rgb01 - mean) / std

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        rgb = ipts[0]
        del ipts

        # RGB pkl may be [N,S,H,W,3] in SUSTech1K raw; convert to [N,S,3,H,W].
        if rgb.dim() == 5 and rgb.size(-1) == 3:
            rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()
        if rgb.dim() != 5 or rgb.size(2) != 3:
            raise ValueError(f"PIGGait expects RGB input [N,S,3,H,W] or [N,S,H,W,3], got {tuple(rgb.shape)}")
        rgb = self._normalize_input_rgb(rgb)

        # OpenGait all_ordered evaluation packs all sequences as [1,total_T,...] with seqL.
        if seqL is not None and rgb.size(0) == 1 and (not self.training):
            embed, logits = self._split_forward_for_all_ordered(rgb, seqL)
            return {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {"embeddings": embed},
            }

        embed_1, embed_2, logits, aux = self._encode_to_embedding(rgb, seqL=seqL, return_aux=True)
        embed = embed_1

        if not self.training:
            return {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {"embeddings": embed},
            }

        training_feat = {
            "triplet": {"embeddings": embed_1, "labels": labs},
            "softmax": {"logits": logits, "labels": labs},
        }
        orth = embed_1.new_tensor(0.0)
        if self.enable_nuisance_losses:
            style_logits, adv_logits, style_feat, orth = self._nuisance_losses(embed_1, aux)
            training_feat["style_cls"] = {"logits": style_logits, "labels": aux["pseudo_nuisance_label"]}
            training_feat["nuisance_adv"] = {"logits": adv_logits, "labels": aux["pseudo_nuisance_label"]}

        # Direct scalar losses are supported by OpenGait LossAggregator.
        w_phase = float(self.loss_weights.get("phase", 0.0))
        w_orth = float(self.loss_weights.get("orth", 0.0))
        w_rex_type = float(self.loss_weights.get("rex_type", 0.0))
        w_rex_view = float(self.loss_weights.get("rex_view", 0.0))
        w_cf = float(self.loss_weights.get("counterfactual", 0.0))
        if w_phase > 0:
            training_feat["pig_phase"] = aux["phase_loss"] * w_phase
        if w_orth > 0:
            training_feat["pig_orth"] = orth * w_orth
        if w_rex_type > 0:
            training_feat["pig_rex_type"] = self._rex_loss(logits, labs, typs) * w_rex_type
        if w_rex_view > 0:
            training_feat["pig_rex_view"] = self._rex_loss(logits, labs, vies) * w_rex_view
        if self.enable_counterfactual and w_cf > 0:
            embed_cf, _, _ = self._encode_to_embedding(rgb, seqL=seqL, return_aux=False, counterfactual=True)
            cf_loss = F.mse_loss(F.normalize(embed_cf, dim=1), F.normalize(embed_1.detach(), dim=1))
            training_feat["pig_counterfactual"] = cf_loss * w_cf

        rgb_vis = self.PIG.denormalize_rgb(rearrange(rgb, "n s c h w -> (n s) c h w"))[:5]
        visual_summary = {
            "image/rgb": rgb_vis.float(),
            "image/pig_mask": rearrange(aux["mask"], "n c s h w -> (n s) c h w")[:5].float(),
            "image/human_mask": rearrange(aux.get("human_mask", aux["mask"]), "n c s h w -> (n s) c h w")[:5].float(),
            "image/pig_photo": aux["photo"][:5].float(),
        }
        return {
            "training_feat": training_feat,
            "visual_summary": visual_summary,
            "inference_feat": {"embeddings": embed},
        }



class PIGGaitPlusPlus(PIGGait):
    """LRCE variant of PIGGait.

    This class intentionally reuses PIGGait implementation while enabling
    config-controlled field_dropout and photo_weight in RGBToPIGField.
    Use it with:

        model_cfg:
          model: PIGGaitPlusPlus
          rgb_pig_cfg:
            field_dropout: 0.1
            photo_weight: 0.5
    """
    pass
