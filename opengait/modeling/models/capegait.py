# CAPEGait: Cycle-Aware Phase-locked Frequency-Field Gait for OpenGait
# Drop this file into: OpenGait/opengait/modeling/models/cape_gait.py
#
# Design goal:
#   RGB is the main modality. Silhouette is only used as an auxiliary support/phase cue.
#   The model converts RGB/LVM features into phase-locked self-similarity and motion fields,
#   so the recognition head receives a gait-oriented representation rather than raw color/texture.
#
# This code follows OpenGait's BaseModel output protocol:
#   retval["training_feat"] contains "triplet", "softmax", and optional scalar auxiliary losses.
#   retval["inference_feat"]["embeddings"] is used for evaluation.

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, PackSequenceWrapper, HorizontalPoolingPyramid
from ..modules import SeparateFCs, SeparateBNNecks


def _conv3x3(in_ch, out_ch, stride=1, groups=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False, groups=groups)


def _conv1x1(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, 1, bias=False)


class ResizeToHW(nn.Module):
    def __init__(self, size: Tuple[int, int], mode: str = "bilinear"):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        if self.mode == "nearest":
            return F.interpolate(x, size=self.size, mode=self.mode)
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=False)


class SpatialFrequencyResidualizer(nn.Module):
    """Stable spatial-frequency filtering on LVM feature maps.

    It is not a DCT copy of Gait-X. It operates on frozen LVM feature maps and learns a small
    residual mixture of low/mid/high spatial bands. Initial gates favor mid-frequency dynamics:
    - very low frequency often carries illumination/background/domain statistics;
    - very high frequency often carries clothing texture/noisy details;
    - mid band keeps body structure and coarse limb motion.
    """

    def __init__(self, channels: int, init=(0.10, 0.80, 0.10), k_low: int = 7, k_mid: int = 3):
        super().__init__()
        self.k_low = k_low
        self.k_mid = k_mid
        init_t = torch.tensor(init, dtype=torch.float32)
        init_t = init_t / init_t.sum()
        self.logits = nn.Parameter(torch.log(init_t + 1e-6).view(1, 3, 1, 1, 1))
        self.mix = nn.Sequential(
            nn.BatchNorm2d(channels, affine=False),
            _conv1x1(channels, channels),
            nn.GELU(),
            _conv3x3(channels, channels, groups=1),
        )

    def _blur(self, x, k):
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        # x: [B, C, H, W]
        low = self._blur(x, self.k_low)
        smooth = self._blur(x, self.k_mid)
        mid = smooth - low
        high = x - smooth
        bands = torch.stack([low, mid, high], dim=1)  # [B, 3, C, H, W]
        gate = torch.softmax(self.logits, dim=1)
        y = (bands * gate).sum(dim=1)
        return self.mix(y)


class LocalFeatureField(nn.Module):
    """Convert multi-channel features into low-dimensional direction fields.

    Static field:
        local self-similarity orientation within the same frame.
    Dynamic field:
        local feature-displacement orientation between consecutive frames.

    The operation uses normalized dot products, so absolute RGB color magnitude is largely removed.
    """

    def __init__(self, in_channels: int, out_channels: int = 8, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.proj = nn.Sequential(
            nn.BatchNorm2d(4, affine=False),
            _conv1x1(4, out_channels),
            nn.GELU(),
            _conv3x3(out_channels, out_channels),
        )

    def _shift(self, x, dy: int, dx: int):
        # zero-padded spatial shift
        b, c, h, w = x.shape
        pad_l, pad_r = max(dx, 0), max(-dx, 0)
        pad_t, pad_b = max(dy, 0), max(-dy, 0)
        y = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        ys = pad_b
        xs = pad_r
        return y[:, :, ys:ys + h, xs:xs + w]

    def _sim(self, a, b):
        a = F.normalize(a, dim=1, eps=self.eps)
        b = F.normalize(b, dim=1, eps=self.eps)
        return (a * b).sum(dim=1, keepdim=True)

    def forward(self, feat_seq):
        # feat_seq: [N, S, C, H, W]
        n, s, c, h, w = feat_seq.shape
        x = feat_seq.reshape(n * s, c, h, w)

        sim_l = self._sim(x, self._shift(x, 0, -1))
        sim_r = self._sim(x, self._shift(x, 0, 1))
        sim_u = self._sim(x, self._shift(x, -1, 0))
        sim_d = self._sim(x, self._shift(x, 1, 0))
        static_dx = sim_r - sim_l
        static_dy = sim_d - sim_u

        # next-frame dynamic field; last frame repeats itself.
        x1 = feat_seq
        x2 = torch.cat([feat_seq[:, 1:], feat_seq[:, -1:].detach()], dim=1)
        a = x1.reshape(n * s, c, h, w)
        b = x2.reshape(n * s, c, h, w)
        dyn_l = self._sim(a, self._shift(b, 0, -1))
        dyn_r = self._sim(a, self._shift(b, 0, 1))
        dyn_u = self._sim(a, self._shift(b, -1, 0))
        dyn_d = self._sim(a, self._shift(b, 1, 0))
        dyn_dx = dyn_r - dyn_l
        dyn_dy = dyn_d - dyn_u

        field = torch.cat([static_dx, static_dy, dyn_dx, dyn_dy], dim=1)
        field = self.proj(field)
        return field.reshape(n, s, -1, h, w)


class SilhouettePhaseCue(nn.Module):
    """Auxiliary phase/motion cue from silhouettes.

    This module does not turn the method into a silhouette method:
    it supplies support masks and a low-dimensional periodic motion cue only.
    """

    def __init__(self, out_channels: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv3x3(3, 16),
            nn.BatchNorm2d(16),
            nn.GELU(),
            _conv3x3(16, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, sil_seq):
        # sil_seq: [N, S, 1, H, W], values in [0,1]
        n, s, _, h, w = sil_seq.shape
        sil = sil_seq.clamp(0, 1)
        diff = torch.zeros_like(sil)
        diff[:, 1:] = (sil[:, 1:] - sil[:, :-1]).abs()

        # a crude left/right anti-phase cue, useful for walking periodicity.
        yy = torch.linspace(-1, 1, h, device=sil.device, dtype=sil.dtype).view(1, 1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=sil.device, dtype=sil.dtype).view(1, 1, 1, 1, w)
        weighted_x = sil * xx
        weighted_y = sil * yy
        phase_map = torch.cat([diff, weighted_x, weighted_y], dim=2)  # [N,S,3,H,W]
        y = self.encoder(phase_map.reshape(n * s, 3, h, w))
        return y.reshape(n, s, -1, h, w)


class CycleAwareTemporalMixer(nn.Module):
    """Temporal module aligned with gait periodicity and snippets.

    It combines:
    - local temporal convolution for short actions/snippets;
    - dilated temporal convolution for roughly half/one-cycle interactions;
    - squeeze gate from silhouette-derived phase cue.
    """

    def __init__(self, channels: int, phase_channels: int = 8, cycle_dilation: int = 4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        self.cycle = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1),
                      padding=(cycle_dilation, 0, 0), dilation=(cycle_dilation, 1, 1),
                      groups=channels, bias=False),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        self.phase_gate = nn.Sequential(
            nn.Conv3d(phase_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, phase):
        # x: [N,C,S,H,W], phase: [N,Cp,S,H,W]
        gate = self.phase_gate(phase)
        return x + self.local(x) + gate * self.cycle(x)


class CAPEBackbone(nn.Module):
    """Compact gait head compatible with OpenGait losses."""

    def __init__(self, model_cfg):
        super().__init__()
        in_channels = model_cfg["cape_repr_dim"]
        channels = model_cfg.get("channels", [64, 128, 256, 512])
        self.pre = SetBlockWrapper(nn.Sequential(
            _conv3x3(in_channels, channels[0], stride=1),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            _conv3x3(channels[0], channels[1], stride=2),
            nn.BatchNorm2d(channels[1]),
            nn.GELU(),
        ))
        self.temporal = CycleAwareTemporalMixer(
            channels[1],
            phase_channels=model_cfg.get("phase_channels", 8),
            cycle_dilation=model_cfg.get("cycle_dilation", 4),
        )
        self.post = SetBlockWrapper(nn.Sequential(
            _conv3x3(channels[1], channels[2], stride=2),
            nn.BatchNorm2d(channels[2]),
            nn.GELU(),
            _conv3x3(channels[2], channels[3], stride=1),
            nn.BatchNorm2d(channels[3]),
            nn.GELU(),
        ))
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

    def forward(self, x, phase, seqL):
        # x: [N,C,S,H,W]
        y = self.pre(x)
        # resize phase to y spatial size
        n, cp, s, h, w = phase.shape
        phase_rs = F.interpolate(
            phase.transpose(1, 2).reshape(n * s, cp, h, w),
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).reshape(n, s, cp, y.shape[-2], y.shape[-1]).transpose(1, 2).contiguous()
        y = self.temporal(y, phase_rs)
        y = self.post(y)
        y = self.TP(y, seqL, options={"dim": 2})[0]
        y = self.HPP(y)
        embed = self.FCs(y)
        _, logits = self.BNNecks(embed)
        return embed, logits


class CAPEGait(BaseModel):
    """Cycle-Aware Phase-locked Frequency-Field Gait.

    Expected inputs:
      ipts[0]: RGB sequence, [N,S,3,H,W].
      ipts[1]: silhouette sequence [N,S,H,W] or [N,S,1,H,W] if available.
      ipts[2]: optional real image ratios, currently not required.

    If silhouettes are absent, the model falls back to all-one masks.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        # image_size/sils_size are kept only for backward compatibility.
        # For square 128x128 RGB/SIL data, set explicit lvm_hw/repr_hw in yaml.
        self.image_size = model_cfg.get("image_size", 224)
        self.sils_size = model_cfg.get("sils_size", 32)
        self.lvm_hw = tuple(model_cfg.get("lvm_hw", [self.image_size * 2, self.image_size]))
        # self.repr_hw = tuple(model_cfg.get("repr_hw", [self.repr_hw[0], self.repr_hw[1]]))
        self.repr_hw = tuple(model_cfg.get("repr_hw", [self.sils_size * 2, self.sils_size]))
        self.source_dim = model_cfg.get("source_dim", 384)
        self.selected_layers = model_cfg.get("selected_layers", [2, 5, 8, 11])
        self.freeze_lvm = model_cfg.get("freeze_lvm", True)

        lvm_concat_dim = self.source_dim * len(self.selected_layers)
        self.freq = SpatialFrequencyResidualizer(lvm_concat_dim)
        self.field = LocalFeatureField(lvm_concat_dim, out_channels=model_cfg.get("field_channels", 8))
        self.phase = SilhouettePhaseCue(out_channels=model_cfg.get("phase_channels", 8))

        repr_mid = model_cfg.get("repr_mid_channels", 64)
        cape_repr_dim = model_cfg["cape_repr_dim"]
        self.repr_project = nn.Sequential(
            nn.BatchNorm2d(lvm_concat_dim + model_cfg.get("field_channels", 8) + model_cfg.get("phase_channels", 8) + 1,
                           affine=False),
            _conv1x1(lvm_concat_dim + model_cfg.get("field_channels", 8) + model_cfg.get("phase_channels", 8) + 1,
                     repr_mid),
            nn.GELU(),
            _conv3x3(repr_mid, cape_repr_dim),
            nn.BatchNorm2d(cape_repr_dim),
            nn.GELU(),
        )
        self.nuisance_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(lvm_concat_dim, model_cfg.get("nuisance_dim", 128)),
            nn.LayerNorm(model_cfg.get("nuisance_dim", 128)),
        )
        self.Backbone = CAPEBackbone(model_cfg)

    def init_DINOv2(self):
        from transformers import Dinov2Config, Dinov2Model

        config = Dinov2Config.from_pretrained(
            self.pretrained_lvm,
            local_files_only=True
        )

        self.LVM = Dinov2Model.from_pretrained(
            self.pretrained_lvm,
            config=config,
            use_safetensors=False,
            local_files_only=True
        )

        if self.freeze_lvm:
            self.LVM.eval()
            self.LVM.requires_grad_(False)

        self.msg_mgr.log_info(f"[CAPEGait] loaded DINOv2 from {self.pretrained_lvm}")


    def init_parameters(self):
        for m in self.modules():
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
        self.init_DINOv2()
        n_params = sum(p.numel() for p in self.parameters())
        self.msg_mgr.log_info(f"[CAPEGait] parameter count: {n_params / 1e6:.3f}M")

    def preprocess_rgb(self, rgb):
        return F.interpolate(rgb, self.lvm_hw, mode="bilinear", align_corners=False)

    def preprocess_sil(self, sil, n, s):
        if sil is None:
            return None
        if sil.dim() == 4:
            sil = sil.unsqueeze(2)
        sil = rearrange(sil, "n s c h w -> (n s) c h w").float()
        sil = F.interpolate(sil, self.repr_hw, mode="nearest")
        return rearrange(sil, "(n s) c h w -> n s c h w", n=n, s=s).clamp(0, 1)

    def _parse_inputs(self, ipts):
        rgb = ipts[0]
        sil = None
        if len(ipts) > 1 and torch.is_tensor(ipts[1]):
            # silhouette usually has 4/5 dims; ratios usually has <=3 dims
            if ipts[1].dim() >= 4:
                sil = ipts[1]
        return rgb, sil

    def _extract_lvm_layers(self, rgb):
        # rgb: [N,S,3,H,W]
        n, s, c, h, w = rgb.shape
        x = rearrange(rgb, "n s c h w -> (n s) c h w").contiguous()
        x = self.preprocess_rgb(x)

        with torch.set_grad_enabled(not self.freeze_lvm):
            outs = self.LVM(x, output_hidden_states=True).hidden_states[1:]
        feats = []
        patch = getattr(getattr(self.LVM, "config", None), "patch_size", 14)
        hp, wp = self.lvm_hw[0] // patch, self.lvm_hw[1] // patch
        if self.lvm_hw[0] % patch != 0 or self.lvm_hw[1] % patch != 0:
            raise ValueError(f"lvm_hw={self.lvm_hw} must be divisible by DINOv2 patch size {patch}.")
        for li in self.selected_layers:
            f = outs[li]
            f = F.layer_norm(f, f.shape[-1:])
            tokens = f[:, 1:, :]
            if tokens.shape[1] != hp * wp:
                raise RuntimeError(
                    f"Unexpected DINO token grid: got {tokens.shape[1]} tokens, "
                    f"expected {hp}x{wp} from lvm_hw={self.lvm_hw}."
                )
            f = tokens.reshape(n * s, hp, wp, self.source_dim).permute(0, 3, 1, 2).contiguous()
            f = F.interpolate(f, self.repr_hw, mode="bilinear", align_corners=False)
            feats.append(f)
        feat = torch.cat(feats, dim=1)
        return feat, (n, s, h, w)

    @staticmethod
    def _decorrelation_loss(gait_embed, nuis):
        # gait_embed: [N,C,P]; nuis: [N,D]
        z = gait_embed.mean(dim=-1)
        z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-5)
        n = (nuis - nuis.mean(0, keepdim=True)) / (nuis.std(0, keepdim=True) + 1e-5)
        c = z.T @ n / max(1, z.shape[0] - 1)
        return (c ** 2).mean()

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        rgb, sil = self._parse_inputs(ipts)
        del ipts

        lvm_feat, shape_info = self._extract_lvm_layers(rgb)
        n, s, _, _ = shape_info

        sil = self.preprocess_sil(sil, n, s)
        if sil is None:
            mask = torch.ones(n, s, 1, self.repr_hw[0], self.repr_hw[1],
                              dtype=lvm_feat.dtype, device=lvm_feat.device)
        else:
            mask = (sil > 0.5).to(lvm_feat.dtype)

        lvm_freq = self.freq(lvm_feat)
        lvm_seq = lvm_freq.reshape(n, s, -1, self.repr_hw[0], self.repr_hw[1])
        field = self.field(lvm_seq)
        phase = self.phase(mask)

        ns = n * s
        mask_2d = mask.reshape(ns, 1, self.repr_hw[0], self.repr_hw[1])
        phase_2d = phase.reshape(ns, -1, self.repr_hw[0], self.repr_hw[1])
        field_2d = field.reshape(ns, -1, self.repr_hw[0], self.repr_hw[1])
        repr_2d = torch.cat([lvm_freq, field_2d, phase_2d, mask_2d], dim=1)
        repr_2d = self.repr_project(repr_2d) * mask_2d

        repr_seq = repr_2d.reshape(n, s, -1, self.repr_hw[0], self.repr_hw[1])
        repr_seq = repr_seq.permute(0, 2, 1, 3, 4).contiguous()
        phase_seq = phase.permute(0, 2, 1, 3, 4).contiguous()

        embed, logits = self.Backbone(repr_seq, phase_seq, seqL)

        # nuisance proxy from suppressed low/high residual channels.
        # It is used only for decorrelation, not for identity prediction.
        nuis = self.nuisance_head(lvm_feat).reshape(n, s, -1).mean(dim=1)
        decov = self._decorrelation_loss(embed, nuis) * self.model_cfg.get("decov_weight", 0.02)

        # keep RGB representation supported by the auxiliary silhouette.
        outside = (repr_2d.abs() * (1.0 - mask_2d)).mean() * self.model_cfg.get("outside_weight", 0.01)

        # discourage trivial zero motion fields; very small weight.
        dyn_energy = field_2d[:, -2:].abs().mean()
        motion_floor = F.relu(self.model_cfg.get("motion_floor", 0.015) - dyn_energy) * self.model_cfg.get("motion_weight", 0.005)

        if self.training:
            retval = {
                "training_feat": {
                    "triplet": {"embeddings": embed, "labels": labs},
                    "softmax": {"logits": logits, "labels": labs},
                    "aux_decov": decov,
                    "aux_outside": outside,
                    "aux_motion_floor": motion_floor,
                },
                "visual_summary": {
                    "image/rgb": rearrange(rgb, "n s c h w -> (n s) c h w")[:5].float(),
                    "image/mask": mask_2d[:5].float(),
                    "image/cape_repr": repr_2d[:, :1][:5].detach().float(),
                },
                "inference_feat": {"embeddings": embed},
            }
        else:
            retval = {
                "training_feat": {},
                "visual_summary": {},
                "inference_feat": {"embeddings": embed},
            }
        return retval
