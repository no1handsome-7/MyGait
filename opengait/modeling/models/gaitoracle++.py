
import os
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# ============================================================
# Utilities
# ============================================================

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverseFn.apply(x, lambd)


def l2n(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


def _scalar_like(x, value=0.0):
    return x.new_tensor(float(value))


def symmetric_infonce(z1, z2, temperature=0.07):
    if z1 is None or z2 is None or z1.size(0) <= 1 or z2.size(0) <= 1:
        base = z1 if z1 is not None else z2
        return _scalar_like(base, 0.0)
    z1 = l2n(z1.float(), dim=-1)
    z2 = l2n(z2.float(), dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
    if z_img is None or z_txt is None:
        base = z_img if z_img is not None else z_txt
        return _scalar_like(base, 0.0)

    if mask is not None:
        mask = mask.float().view(mask.size(0), -1).mean(-1)
        valid = mask > 0
        if valid.sum() <= 1:
            return z_img.new_tensor(0.0)
        z_img = z_img[valid]
        z_txt = z_txt[valid]
        weights = mask[valid].clamp(0, 1)
    else:
        if z_img.size(0) <= 1:
            return z_img.new_tensor(0.0)
        weights = None

    z_img = l2n(z_img.float(), dim=-1)
    z_txt = l2n(z_txt.float(), dim=-1)
    logits = z_img @ z_txt.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    if weights is None:
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    loss_i = F.cross_entropy(logits, labels, reduction='none')
    loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
    weights = weights / weights.mean().clamp_min(1e-6)
    return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


def ramp_value(step: int, start: int, length: int, max_value: float = 1.0):
    if length <= 0:
        return max_value if step >= start else 0.0
    if step < start:
        return 0.0
    return float(max_value) * min(1.0, max(0.0, (step - start) / float(length)))


# ============================================================
# Text compression: keep old Qwen 3584-dim features compatible
# ============================================================

class TextEmbeddingAdapter(nn.Module):
    """Compress high-dimensional MLLM text embeddings to a gait-sized code.

    Existing Qwen2.5-VL features such as 3584-dim prompt/nuisance embeddings can be
    used directly.  The adapter crops/pads unexpected dimensions, then learns a
    low-rank bottleneck.  No offline feature regeneration is required.
    """

    def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024, dropout=0.1, l2_norm=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.code_dim = int(code_dim)
        self.l2_norm = bool(l2_norm)
        hidden_dim = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
        self.in_norm = nn.LayerNorm(self.input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.code_dim, bias=False),
            nn.LayerNorm(self.code_dim),
        )

    def _fit_dim(self, x):
        d = x.size(-1)
        if d == self.input_dim:
            return x
        if d > self.input_dim:
            return x[..., :self.input_dim]
        pad = x.new_zeros(*x.shape[:-1], self.input_dim - d)
        return torch.cat([x, pad], dim=-1)

    def forward(self, x):
        if x is None:
            return None
        if x.dim() == 3:
            x = x.mean(1)
        elif x.dim() > 3:
            x = x.view(x.size(0), -1)
        x = self._fit_dim(x.float())
        z = self.net(self.in_norm(x))
        return l2n(z, dim=-1) if self.l2_norm else z


# ============================================================
# Module A: RGB encoder
# ============================================================

class TinyDenseRGBEncoder(nn.Module):
    """Dependency-free dense RGB encoder; use timm backbone if available/needed."""

    def __init__(self, out_channels=512, in_channels=3):
        super().__init__()
        c1 = max(out_channels // 8, 32)
        c2 = max(out_channels // 4, 64)
        c3 = max(out_channels // 2, 128)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        y = self.net(x.reshape(B * T, C, H, W))
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


class TimmDenseRGBEncoder(nn.Module):
    """Optional dense foundation encoder via timm features_only models."""

    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', out_channels=512, pretrained=True, freeze=True):
        super().__init__()
        try:
            import timm
        except Exception as exc:
            raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

        self.model_name = model_name
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        chs = self.backbone.feature_info.channels()
        in_c = chs[-1]
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.out_channels = out_channels

    def forward(self, x):
        B, T, C, H, W = x.shape
        y = self.backbone(x.reshape(B * T, C, H, W))[-1]
        y = self.proj(y)
        _, C2, H2, W2 = y.shape
        return y.reshape(B, T, C2, H2, W2)


# ============================================================
# Module B1: positive gait prompt gate
# ============================================================

class PositiveTextGate(nn.Module):
    """Fixed gait-positive prompt bank -> channel/spatial gait gate.

    This preserves motion-biometric cues before early RGB-anchor fusion.  The
    prompt bank can still be the old Qwen .npy file; the high-dimensional text
    vectors are compressed by TextEmbeddingAdapter inside the module.
    """

    def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
                 trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.channels = channels

        if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
            arr = np.load(prompt_feat_path).astype('float32')
            if arr.ndim == 1:
                arr = arr[None, :]
            prompt = torch.from_numpy(arr)
            text_dim = int(prompt.shape[-1])
        else:
            prompt = torch.randn(num_prompts, text_dim) * 0.02

        self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
        self.text_adapter = TextEmbeddingAdapter(
            input_dim=text_dim,
            code_dim=text_code_dim,
            hidden_dim=text_hidden_dim,
            dropout=dropout,
            l2_norm=False,
        )
        self.text_to_channel = nn.Sequential(
            nn.Linear(text_code_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        mid = max(channels // 4, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_map):
        # feat_map: [B, T, C, H, W]
        B, T, C, H, W = feat_map.shape
        prompt = self.prompt_feat.to(device=feat_map.device)
        pos_code = self.text_adapter(prompt).mean(0)  # [Dtxt]
        pos = self.text_to_channel(pos_code)          # [C]
        ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

        x = feat_map * (1.0 + ch_gate)
        sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
        guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
        return guided, sp_gate


# ============================================================
# Module B2: staged nuisance removal
# ============================================================


class StagedNuisanceRemoval(nn.Module):
    """Immediate GRL nuisance detachment for RGB-only gait representation.

    This keeps the original pre_temporal/post_temporal API but restores the
    empirically useful behavior: nuisance adversarial learning starts from the
    beginning of training. Qwen nuisance text/masks are used only as training
    supervision. Inference remains RGB-only.

    Unlike the staged version, this module does not suppress parts or rewrite
    the forward feature path. It only applies a GRL nuisance objective on the
    identity feature, which is the cleanest ablation for:
        RGB + positive text guidance + nuisance feature disentanglement.
    """

    def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256, text_hidden_dim=1024,
                 sub_dim=256, factors=None, grl_lambda=0.05, temperature=0.07,
                 warmup_iters=0, probe_iters=0, adv_ramp_iters=0,
                 reliability_start_iter=1000000000, reliability_ramp_iters=1,
                 probe_loss_weight=0.0, adv_loss_weight=0.05, mask_loss_weight=0.0,
                 max_suppression=0.0, detach_reliability_gate=True, factor_weights=None,
                 dropout=0.1, loss_weight=None, immediate_adv=True, **unused):
        super().__init__()
        self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
        self.parts_num = int(parts_num)
        self.grl_lambda = float(grl_lambda)
        self.temperature = float(temperature)
        if loss_weight is not None:
            adv_loss_weight = float(loss_weight)
        self.adv_loss_weight = float(adv_loss_weight)

        self.factor_weights = {k: 1.0 for k in self.factors}
        if factor_weights:
            self.factor_weights.update({str(k): float(v) for k, v in factor_weights.items()})

        self.text_adapter = TextEmbeddingAdapter(
            text_dim, text_code_dim, text_hidden_dim, dropout=dropout, l2_norm=True
        )
        self.txt_heads = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim))
            for k in self.factors
        })
        self.adv_heads = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(in_dim, sub_dim),
                nn.LayerNorm(sub_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(sub_dim, sub_dim),
                nn.LayerNorm(sub_dim),
            ) for k in self.factors
        })

    def schedule(self, step: int):
        return {'probe': 1.0, 'adv': 1.0, 'rel': 0.0}

    def _target_mask(self, masks, factor):
        if masks is None or factor not in masks or masks[factor] is None:
            return None
        m = masks[factor].float()
        if m.dim() > 1:
            m = m.view(m.size(0), -1).mean(-1)
        return m.clamp(0, 1)

    def _text_z(self, text_feats, factor):
        if text_feats is None or factor not in text_feats or text_feats[factor] is None:
            return None
        code = self.text_adapter(text_feats[factor])
        return l2n(self.txt_heads[factor](code), dim=-1)

    def pre_temporal(self, visual_global, text_feats=None, masks=None, step: int = 0):
        # No part suppression in this RGB-only ablation.
        B = visual_global.size(0)
        reliability = visual_global.new_ones(B, 1, 1, self.parts_num)
        diag = {
            'sched_probe': visual_global.new_tensor(1.0),
            'sched_adv': visual_global.new_tensor(1.0),
            'sched_rel': visual_global.new_tensor(0.0),
            'mean_reliability': reliability.mean().detach(),
        }
        return reliability, {}, diag

    def post_temporal(self, id_global, text_feats=None, masks=None, step: int = 0):
        losses = {}
        if self.adv_loss_weight <= 0:
            return losses

        id_rev = grad_reverse(id_global.float(), self.grl_lambda)

        for f in self.factors:
            z_txt = self._text_z(text_feats, f)
            if z_txt is None:
                continue
            target = self._target_mask(masks, f)
            z_id = l2n(self.adv_heads[f](id_rev), dim=-1)
            fac_w = float(self.factor_weights.get(f, 1.0))
            losses[f'adv_{f}'] = masked_symmetric_infonce(
                z_id, z_txt, mask=target, temperature=self.temperature
            ) * (self.adv_loss_weight * fac_w)

        return losses


# Backward-compatible alias for old imports/checkpoints that refer to this name.
NuisanceDetachHead = StagedNuisanceRemoval


# ============================================================
# Module C: geometry / silhouette / point-cloud anchors
# ============================================================

class SilhouetteAnchorEncoder(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

    def forward(self, sil):
        # sil: [B,T,H,W] or [B,T,1,H,W]
        if sil.dim() == 4:
            sil = sil.unsqueeze(2)
        elif sil.dim() == 5 and sil.shape[2] != 1:
            sil = sil[:, :, :1]
        return self.net(sil.float())


class PointCloudAnchorEncoder(nn.Module):
    def __init__(self, out_channels=512, parts_num=16, in_dim=3):
        super().__init__()
        self.parts_num = int(parts_num)
        self.out_channels = int(out_channels)
        self.in_dim = int(in_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_channels),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(out_channels)

    def _fit_dim(self, pc):
        d = pc.size(-1)
        if d == self.in_dim:
            return pc
        if d > self.in_dim:
            return pc[..., :self.in_dim]
        pad = pc.new_zeros(*pc.shape[:-1], self.in_dim - d)
        return torch.cat([pc, pad], dim=-1)

    def forward(self, pc):
        # pc: [B,T,N,C]
        B, T, N, _ = pc.shape
        pc = self._fit_dim(pc.float())
        xyz = pc[..., :min(3, self.in_dim)]
        z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
        z_min = z.min(dim=-1, keepdim=True).values
        z_max = z.max(dim=-1, keepdim=True).values
        z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

        feat = self.point_mlp(pc.reshape(B * T * N, self.in_dim)).reshape(B, T, N, self.out_channels)
        feat = self.norm(feat)

        part_feats = []
        for p in range(self.parts_num):
            lo = p / self.parts_num
            hi = (p + 1) / self.parts_num
            if p == self.parts_num - 1:
                mask = ((z_norm >= lo) & (z_norm <= hi)).float().unsqueeze(-1)
            else:
                mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
            denom = mask.sum(dim=2).clamp_min(1.0)
            pooled = (feat * mask).sum(dim=2) / denom
            part_feats.append(pooled)
        return torch.stack(part_feats, dim=-1)  # [B,T,C,P]


# ============================================================
# Module D: early part-token fusion + unified temporal encoder
# ============================================================

class EarlyPartTokenFusion(nn.Module):
    """Fuse RGB and anchor part tokens before temporal modeling.

    This fixes the previous late-fusion design where each modality was temporally
    encoded independently before interaction.
    """

    def __init__(self, channels=512, dropout=0.05):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, 1, 1),
        )
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, rgb_part_seq, anchor_part_seq=None):
        # [B,T,C,P]
        B, T, C, P = rgb_part_seq.shape
        if anchor_part_seq is None:
            x = rgb_part_seq.reshape(B * T, C, P)
            return self.out(x).reshape(B, T, C, P), None
        if anchor_part_seq.size(-1) != P:
            raise ValueError(f'Anchor parts {anchor_part_seq.size(-1)} != RGB parts {P}')
        pair = torch.cat([rgb_part_seq, anchor_part_seq], dim=2).reshape(B * T, C * 2, P)
        gate = torch.sigmoid(self.gate(pair)).reshape(B, T, 1, P)
        fused = rgb_part_seq * (1.0 - gate) + anchor_part_seq * gate
        fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)
        return fused, gate


class CycleAwareTemporalEncoder(nn.Module):
    """Temporal encoder with frame, difference and low-frequency gait cycle branches."""

    def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1, future_pred=True):
        super().__init__()
        self.channels = int(channels)
        self.parts_num = int(parts_num)
        self.future_pred = bool(future_pred)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.diff_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.freq_proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.LayerNorm(channels))
        self.attn = nn.Sequential(nn.Linear(channels, max(channels // 4, 16)), nn.Tanh(), nn.Linear(max(channels // 4, 16), 1))
        if self.future_pred:
            self.future_head = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, part_seq):
        # part_seq: [B,T,C,P]
        B, T, C, P = part_seq.shape
        x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)
        x_tem = self.temporal(x)
        if T > 1:
            diff = x[:, 1:] - x[:, :-1]
            diff = F.pad(diff, (0, 0, 1, 0))
        else:
            diff = torch.zeros_like(x)
        diff = self.diff_proj(diff)
        if T > 1:
            freq = torch.fft.rfft(x.float(), dim=1).abs()
            freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
        else:
            freq = x.float().mean(1)
        freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)
        y = x_tem + diff + freq
        att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)
        pooled = (y * att.unsqueeze(-1)).sum(1)
        pooled = pooled.reshape(B, P, C).permute(0, 2, 1).contiguous()
        future_loss = pooled.new_tensor(0.0)
        if self.future_pred and T > 2:
            pred = self.future_head(y[:, :-1])
            target = x[:, 1:].detach()
            future_loss = F.mse_loss(pred, target)
        att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()
        return pooled, future_loss, att


# ============================================================
# Module E: lightweight verifier regularization
# ============================================================

class RuleBasedVerifier(nn.Module):
    def __init__(self, attention_weight=0.0):
        super().__init__()
        self.attention_weight = float(attention_weight)

    def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
        if self.attention_weight <= 0 or gait_attn is None:
            return gait_attn.new_tensor(0.0)
        B, T, _, Hf, Wf = gait_attn.shape
        target = None
        if sil_seq is not None:
            sil = sil_seq.unsqueeze(2) if sil_seq.dim() == 4 else sil_seq[:, :, :1]
            target = F.interpolate(
                sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
                size=(Hf, Wf), mode='bilinear', align_corners=False,
            ).reshape(B, T, 1, Hf, Wf)
            target = (target > 0.05).float()
        elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
            gray = rgb_seq.float().mean(2, keepdim=True)
            diff = torch.zeros_like(gray)
            diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
            target = F.interpolate(
                diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
                size=(Hf, Wf), mode='bilinear', align_corners=False,
            ).reshape(B, T, 1, Hf, Wf)
            target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        if target is None:
            return gait_attn.new_tensor(0.0)
        # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
        if self.attention_weight <= 0:
            return gait_attn.new_tensor(0.0)

        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(
                gait_attn.float().clamp(1e-6, 1 - 1e-6),
                target.float()
            )

        return loss * self.attention_weight


# ============================================================
# Main OpenGait model
# ============================================================

class GaitOraclePlusPlus(BaseModel):
    """GaitOracle strong version.

    Key changes against the previous version:
      1) RGB/geometry anchor are fused at part-token level before temporal encoding.
      2) Qwen 3584-d text embeddings are compressed to gait-sized codes on the fly.
      3) Nuisance removal is staged: warmup -> probe -> adversarial -> reliability refinement.
      4) Existing positive prompt .npy and qwen25vl_*_feat.pkl / mask.pkl files remain compatible.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        C = model_cfg.get('channels', 512)
        P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
        self.channels = int(C)
        self.parts_num = int(P)
        self.rgb_index = model_cfg.get('rgb_index', 0)
        self.sil_index = model_cfg.get('sil_index', None)
        self.pc_index = model_cfg.get('pc_index', None)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

        text_cfg = model_cfg.get('text_adapter', {})
        text_dim = int(text_cfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        text_code_dim = int(text_cfg.get('code_dim', 256))
        text_hidden_dim = int(text_cfg.get('hidden_dim', 1024))
        text_dropout = float(text_cfg.get('dropout', 0.1))

        rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
        if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
            self.rgb_encoder = TimmDenseRGBEncoder(
                model_name=rgb_backbone_cfg.get('model_name', 'vit_base_patch14_dinov2.lvd142m'),
                out_channels=C,
                pretrained=rgb_backbone_cfg.get('pretrained', True),
                freeze=rgb_backbone_cfg.get('freeze', True),
            )
        else:
            self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

        pos_cfg = model_cfg.get('positive_text', {})
        self.positive_gate = PositiveTextGate(
            text_dim=pos_cfg.get('text_dim', text_dim),
            channels=C,
            prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
            num_prompts=pos_cfg.get('num_prompts', 4),
            trainable_prompts=pos_cfg.get('trainable', False),
            text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
            text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
            dropout=pos_cfg.get('dropout', text_dropout),
        )

        self.use_sil_anchor = self.sil_index is not None
        self.use_pc_anchor = self.pc_index is not None
        self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
        self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        early_cfg = model_cfg.get('early_fusion', {})
        self.early_fusion = EarlyPartTokenFusion(channels=C, dropout=early_cfg.get('dropout', 0.05))

        temporal_cfg = model_cfg.get('cycle_temporal', {})
        self.temporal = CycleAwareTemporalEncoder(
            channels=C,
            parts_num=P,
            heads=temporal_cfg.get('heads', 4),
            layers=temporal_cfg.get('layers', 1),
            dropout=temporal_cfg.get('dropout', 0.1),
            future_pred=temporal_cfg.get('future_pred', True),
        )

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
        self.use_nuisance = bool(nuis_cfg.get('enabled', False))
        legacy_loss_w = float(nuis_cfg.get('loss_weight', 0.03))
        if self.use_nuisance:
            self.nuisance_removal = StagedNuisanceRemoval(
                in_dim=C,
                parts_num=P,
                text_dim=nuis_cfg.get('text_dim', text_dim),
                text_code_dim=nuis_cfg.get('text_code_dim', text_code_dim),
                text_hidden_dim=nuis_cfg.get('text_hidden_dim', text_hidden_dim),
                sub_dim=nuis_cfg.get('sub_dim', 256),
                factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
                grl_lambda=nuis_cfg.get('grl_lambda', 0.08),
                temperature=nuis_cfg.get('temperature', 0.07),
                warmup_iters=nuis_cfg.get('warmup_iters', 5000),
                probe_iters=nuis_cfg.get('probe_iters', 10000),
                adv_ramp_iters=nuis_cfg.get('adv_ramp_iters', 10000),
                reliability_start_iter=nuis_cfg.get('reliability_start_iter', 15000),
                reliability_ramp_iters=nuis_cfg.get('reliability_ramp_iters', 10000),
                probe_loss_weight=nuis_cfg.get('probe_loss_weight', legacy_loss_w),
                adv_loss_weight=nuis_cfg.get('adv_loss_weight', legacy_loss_w),
                mask_loss_weight=nuis_cfg.get('mask_loss_weight', legacy_loss_w * 0.5),
                max_suppression=nuis_cfg.get('max_suppression', 0.35),
                detach_reliability_gate=nuis_cfg.get('detach_reliability_gate', True),
                factor_weights=nuis_cfg.get('factor_weights', None),
                dropout=nuis_cfg.get('dropout', text_dropout),
            )
        else:
            self.nuisance_removal = None

        ver_cfg = model_cfg.get('verifier', {})
        self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))
        self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.future_weight = float(model_cfg.get('future_weight', 0.01))
        self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))

    def _current_step(self):
        if self.training:
            self._forward_iter += 1
        for attr in ('iteration', 'iter', 'current_iter', 'global_step'):
            if hasattr(self, attr):
                try:
                    val = getattr(self, attr)
                    if torch.is_tensor(val):
                        return int(val.detach().item())
                    return int(val)
                except Exception:
                    pass
        return int(self._forward_iter.detach().item())

    def _to_rgb(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif x.dim() == 5:
            if x.shape[2] == 1:
                x = x.repeat(1, 1, 3, 1, 1)
            elif x.shape[-1] == 3 and x.shape[2] != 3:
                x = x.permute(0, 1, 4, 2, 3)
        return x.float()

    def _to_text_feat(self, x):
        if x is None:
            return None
        x = x.float()
        if x.dim() == 3:
            return x.mean(1)
        if x.dim() == 2:
            return x
        return x.view(x.size(0), -1)

    def _get_optional(self, ipts, idx):
        if idx is None:
            return None
        if isinstance(idx, str):
            if idx.lower() in ('none', 'null'):
                return None
            idx = int(idx)
        if idx < 0 or idx >= len(ipts):
            return None
        return ipts[idx]

    def _dense_to_part_seq(self, feat_map):
        B, T, C, H, W = feat_map.shape
        part = self.HPP(feat_map.reshape(B * T, C, H, W))
        return part.reshape(B, T, C, -1)

    def _collect_nuisance_inputs(self, ipts):
        text_feats, masks = {}, {}
        for k, idx in self.nuisance_indices.items():
            text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
        for k, idx in self.nuisance_mask_indices.items():
            masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
        return text_feats, masks

    def _sum_losses(self, losses: Dict[str, torch.Tensor], base: torch.Tensor):
        total = base.new_tensor(0.0)
        for v in losses.values():
            if v is not None:
                total = total + v
        return total

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._current_step()
        rgb = self._to_rgb(ipts[self.rgb_index])
        sil = self._get_optional(ipts, self.sil_index)
        pc = self._get_optional(ipts, self.pc_index)

        rgb_map = self.rgb_encoder(rgb)
        rgb_guided, gait_attn = self.positive_gate(rgb_map)
        rgb_part_seq = self._dense_to_part_seq(rgb_guided)

        anchor_part_seq = None
        sil_for_verifier = None
        if pc is not None and self.pc_encoder is not None:
            anchor_part_seq = self.pc_encoder(pc.float())
        elif sil is not None and self.sil_encoder is not None:
            sil_for_verifier = sil
            sil_map = self.sil_encoder(sil.float())
            anchor_part_seq = self._dense_to_part_seq(sil_map)

        nuisance_pre_losses = {}
        nuisance_post_losses = {}
        nuisance_diag = {}
        text_feats, nuisance_masks = ({}, {})
        if self.use_nuisance:
            text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
            pre_global = rgb_part_seq.mean(dim=(1, 3))
            reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
                pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            rgb_part_seq = rgb_part_seq * reliability

        # Early fusion happens before temporal encoding.
        fused_seq, modality_gate = self.early_fusion(rgb_part_seq, anchor_part_seq)
        fused_feat, future_loss, temporal_attn = self.temporal(fused_seq)

        embed_1 = self.FCs(fused_feat)
        embed_2, logits = self.BNNecks(embed_1)

        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }

        aux_losses = {}
        if anchor_part_seq is not None and self.rgb_anchor_weight > 0:
            rgb_g = rgb_part_seq.mean(dim=(1, 3))
            anc_g = anchor_part_seq.mean(dim=(1, 3))
            aux_losses['rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

        if self.future_weight > 0:
            aux_losses['future_pred'] = future_loss * self.future_weight

        if self.use_nuisance:
            id_global = fused_feat.mean(-1)
            nuisance_post_losses = self.nuisance_removal.post_temporal(
                id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
            )
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_pre_losses.items()})
            aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_post_losses.items()})

        attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
        if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
            aux_losses['attention_verifier'] = attn_loss

        if self.use_scalar_aux_loss:
            aux_total = self._sum_losses(aux_losses, embed_1)
            training_feat['gaitoracle_aux'] = {'loss': aux_total}
        else:
            for k, v in aux_losses.items():
                training_feat[f'loss_{k}'] = v

        visual_summary = {
            'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
            'image/gait_attention': rearrange(gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'),
        }

        if self.training and self.use_nuisance:
            # Scalars are intentionally kept detached for loggers that inspect visual_summary.
            for k, v in nuisance_diag.items():
                visual_summary[f'scalar/{k}'] = v.detach()

        return {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embed_1},
        }






# import os
# import math
# from typing import Dict, Optional, List, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# # ============================================================
# # Utilities
# # ============================================================

# class GradReverseFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd):
#         ctx.lambd = float(lambd)
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.lambd * grad_output, None


# def grad_reverse(x, lambd=1.0):
#     return GradReverseFn.apply(x, lambd)


# def l2n(x, dim=-1, eps=1e-12):
#     return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# def symmetric_infonce(z1, z2, temperature=0.07):
#     z1 = l2n(z1.float(), dim=-1)
#     z2 = l2n(z2.float(), dim=-1)
#     logits = z1 @ z2.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)
#     return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
#     if mask is not None:
#         mask = mask.float().view(-1)
#         valid = mask > 0
#         if valid.sum() <= 1:
#             return z_img.new_tensor(0.0)
#         z_img = z_img[valid]
#         z_txt = z_txt[valid]
#         weights = mask[valid].clamp(0, 1)
#     else:
#         if z_img.size(0) <= 1:
#             return z_img.new_tensor(0.0)
#         weights = None

#     z_img = l2n(z_img.float(), dim=-1)
#     z_txt = l2n(z_txt.float(), dim=-1)
#     logits = z_img @ z_txt.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)

#     if weights is None:
#         return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

#     loss_i = F.cross_entropy(logits, labels, reduction='none')
#     loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
#     weights = weights / weights.mean().clamp_min(1e-6)
#     return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


# def safe_get_cfg(cfg, key, default=None):
#     return cfg[key] if key in cfg else default


# # ============================================================
# # Module A: Foundation RGB Encoder
# # ============================================================

# class TinyDenseRGBEncoder(nn.Module):
#     """Fallback dense RGB encoder.

#     This is intentionally lightweight and dependency-free. If timm is available,
#     GaitOracle can switch to a foundation backbone via TimmDenseRGBEncoder.
#     """

#     def __init__(self, out_channels=512, in_channels=3):
#         super().__init__()
#         c1 = max(out_channels // 8, 32)
#         c2 = max(out_channels // 4, 64)
#         c3 = max(out_channels // 2, 128)
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.GELU(),
#             nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.GELU(),
#             nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c3),
#             nn.GELU(),
#             nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.out_channels = out_channels

#     def forward(self, x):
#         # x: [B, T, 3, H, W]
#         B, T, C, H, W = x.shape
#         y = self.net(x.reshape(B * T, C, H, W))
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# class TimmDenseRGBEncoder(nn.Module):
#     """Optional dense foundation encoder via timm ViT/DINOv2 models.

#     Offline checkpoint loading is supported through checkpoint_path.

#     This version avoids two common offline DINOv2 issues:
#       1) features_only=True wraps the ViT as FeatureGetterNet, whose state_dict
#          keys are prefixed with "model.". We adapt the checkpoint keys manually.
#       2) DINOv2-B/14 timm defaults to 518x518 inputs. Gait images are usually
#          much smaller, e.g. 128x88. We create the ViT with a configurable smaller
#          img_size and resize positional embeddings accordingly.
#     """

#     def __init__(self, model_name='vit_small_patch14_dinov2.lvd142m', out_channels=512,
#                  pretrained=True, freeze=True, checkpoint_path=None,
#                  img_size=(126, 84), chunk_size=64):
#         super().__init__()
#         try:
#             import timm
#         except Exception as exc:
#             raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

#         self.model_name = model_name
#         self.checkpoint_path = checkpoint_path
#         self.chunk_size = int(chunk_size) if chunk_size is not None else 0

#         if img_size is None:
#             self.img_size = None
#         elif isinstance(img_size, int):
#             self.img_size = (int(img_size), int(img_size))
#         elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
#             self.img_size = (int(img_size[0]), int(img_size[1]))
#         else:
#             raise ValueError(f"img_size should be int or [H, W], got {img_size}")

#         has_local_ckpt = checkpoint_path is not None and str(checkpoint_path).lower() not in ('', 'none', 'null')

#         create_kwargs = dict(
#             pretrained=(bool(pretrained) and not has_local_ckpt),
#             features_only=True,
#         )
#         if self.img_size is not None:
#             create_kwargs['img_size'] = self.img_size

#         # Do not pass checkpoint_path into timm.create_model when features_only=True.
#         # We load it manually below to handle FeatureGetterNet's "model." prefix and
#         # position embedding resizing.
#         self.backbone = timm.create_model(model_name, **create_kwargs)

#         if has_local_ckpt:
#             if not os.path.exists(checkpoint_path):
#                 raise FileNotFoundError(
#                     f"DINOv2 checkpoint_path does not exist: {checkpoint_path}. "
#                     "Please download model.safetensors manually and update rgb_backbone.checkpoint_path."
#                 )
#             self._load_local_checkpoint(checkpoint_path)

#         chs = self.backbone.feature_info.channels()
#         in_c = chs[-1]
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_c, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         if freeze:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         self.out_channels = out_channels

#     def _read_checkpoint(self, checkpoint_path):
#         if str(checkpoint_path).endswith('.safetensors'):
#             try:
#                 from safetensors.torch import load_file
#             except Exception as exc:
#                 raise ImportError(
#                     "Loading .safetensors requires safetensors. "
#                     "Install it with: pip install safetensors"
#                 ) from exc
#             state = load_file(checkpoint_path, device='cpu')
#         else:
#             state = torch.load(checkpoint_path, map_location='cpu')

#         if isinstance(state, dict):
#             for key in ('state_dict', 'model', 'model_state_dict', 'teacher', 'student'):
#                 if key in state and isinstance(state[key], dict):
#                     maybe = state[key]
#                     if any(torch.is_tensor(v) for v in maybe.values()):
#                         state = maybe
#                         break

#         if not isinstance(state, dict):
#             raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

#         clean = {}
#         for k, v in state.items():
#             if not torch.is_tensor(v):
#                 continue
#             name = str(k)
#             for pfx in ('module.', 'backbone.', 'encoder.'):
#                 if name.startswith(pfx):
#                     name = name[len(pfx):]
#             clean[name] = v
#         return clean

#     def _get_patch_grid_size(self):
#         # FeatureGetterNet exposes the original ViT as self.backbone.model.
#         m = getattr(self.backbone, 'model', self.backbone)
#         pe = getattr(m, 'patch_embed', None)
#         grid = getattr(pe, 'grid_size', None)
#         if grid is not None:
#             return (int(grid[0]), int(grid[1]))
#         if self.img_size is not None:
#             patch_size = getattr(pe, 'patch_size', (14, 14))
#             if isinstance(patch_size, int):
#                 patch_size = (patch_size, patch_size)
#             return (int(self.img_size[0] // patch_size[0]), int(self.img_size[1] // patch_size[1]))
#         return None

#     def _resize_pos_embed(self, pos_embed, target):
#         if tuple(pos_embed.shape) == tuple(target.shape):
#             return pos_embed
#         if pos_embed.dim() != 3 or target.dim() != 3 or pos_embed.shape[0] != 1 or target.shape[0] != 1:
#             return None
#         if pos_embed.shape[-1] != target.shape[-1]:
#             return None

#         old_tokens = pos_embed.shape[1]
#         new_tokens = target.shape[1]
#         if old_tokens <= 1 or new_tokens <= 1:
#             return None

#         cls_pos = pos_embed[:, :1]
#         patch_pos = pos_embed[:, 1:]
#         old_grid = int(round(math.sqrt(patch_pos.shape[1])))
#         if old_grid * old_grid != patch_pos.shape[1]:
#             return None

#         new_grid = self._get_patch_grid_size()
#         if new_grid is None:
#             new_patch_tokens = new_tokens - 1
#             new_side = int(round(math.sqrt(new_patch_tokens)))
#             if new_side * new_side != new_patch_tokens:
#                 return None
#             new_grid = (new_side, new_side)

#         new_h, new_w = int(new_grid[0]), int(new_grid[1])
#         if new_h * new_w != new_tokens - 1:
#             return None

#         patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
#         patch_pos = F.interpolate(patch_pos.float(), size=(new_h, new_w), mode='bicubic', align_corners=False)
#         patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1).to(dtype=pos_embed.dtype)
#         return torch.cat([cls_pos, patch_pos], dim=1)

#     def _load_local_checkpoint(self, checkpoint_path):
#         raw_state = self._read_checkpoint(checkpoint_path)
#         target_state = self.backbone.state_dict()

#         # Candidate 1: as-is. Candidate 2: add "model." prefix for FeatureGetterNet.
#         prefixed_state = {}
#         for k, v in raw_state.items():
#             prefixed_state[k if k.startswith('model.') else 'model.' + k] = v

#         def adapt_and_score(sd):
#             adapted = {}
#             matched = 0
#             resized_pos = 0
#             for k, v in sd.items():
#                 if k not in target_state:
#                     adapted[k] = v
#                     continue
#                 tgt = target_state[k]
#                 if hasattr(v, 'shape') and tuple(v.shape) == tuple(tgt.shape):
#                     adapted[k] = v
#                     matched += 1
#                 elif k.endswith('pos_embed'):
#                     resized = self._resize_pos_embed(v, tgt)
#                     if resized is not None and tuple(resized.shape) == tuple(tgt.shape):
#                         adapted[k] = resized
#                         matched += 1
#                         resized_pos += 1
#                 # Other shape-mismatched tensors are dropped.
#             return adapted, matched, resized_pos

#         candidates = [raw_state, prefixed_state]
#         adapted_candidates = [adapt_and_score(c) for c in candidates]
#         best_state, matched, resized_pos = max(adapted_candidates, key=lambda x: x[1])

#         if matched == 0:
#             sample_ckpt = list(raw_state.keys())[:10]
#             sample_model = list(target_state.keys())[:10]
#             raise RuntimeError(
#                 "No compatible keys found when loading DINOv2 checkpoint.\n"
#                 f"checkpoint_path={checkpoint_path}\n"
#                 f"sample checkpoint keys={sample_ckpt}\n"
#                 f"sample model keys={sample_model}"
#             )

#         incompatible = self.backbone.load_state_dict(best_state, strict=False)
#         missing = getattr(incompatible, 'missing_keys', [])
#         unexpected = getattr(incompatible, 'unexpected_keys', [])

#         print(
#             f"[TimmDenseRGBEncoder] Loaded local checkpoint: {checkpoint_path}; "
#             f"matched_keys={matched}; resized_pos_embed={resized_pos}; "
#             f"missing_keys={len(missing)}; unexpected_keys={len(unexpected)}; "
#             f"img_size={self.img_size}; chunk_size={self.chunk_size}"
#         )

#     def _forward_flat(self, x_flat):
#         if self.img_size is not None and tuple(x_flat.shape[-2:]) != tuple(self.img_size):
#             x_flat = F.interpolate(x_flat, size=self.img_size, mode='bilinear', align_corners=False)
#         return self.backbone(x_flat)[-1]

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x_flat = x.reshape(B * T, C, H, W)

#         if self.chunk_size and self.chunk_size > 0 and x_flat.size(0) > self.chunk_size:
#             ys = []
#             for s in range(0, x_flat.size(0), self.chunk_size):
#                 ys.append(self._forward_flat(x_flat[s:s + self.chunk_size]))
#             y = torch.cat(ys, dim=0)
#         else:
#             y = self._forward_flat(x_flat)

#         y = self.proj(y)
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# class PositiveTextGate(nn.Module):
#     """Fixed gait-positive prompt bank -> channel and spatial motion gate.

#     Positive text tells the model what to preserve:
#       arm/leg swing, body boundary motion, stride rhythm, torso stability.
#     """

#     def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4, trainable_prompts=False):
#         super().__init__()
#         self.channels = channels
#         self.text_dim = text_dim

#         if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
#             arr = np.load(prompt_feat_path).astype('float32')
#             if arr.ndim == 1:
#                 arr = arr[None, :]
#             prompt = torch.from_numpy(arr)
#             text_dim = prompt.shape[-1]
#             self.text_dim = text_dim
#         else:
#             prompt = torch.randn(num_prompts, text_dim) * 0.02

#         self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
#         self.text_proj = nn.Sequential(
#             nn.Linear(text_dim, channels),
#             nn.LayerNorm(channels),
#             nn.GELU(),
#         )
#         self.channel_gate = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.Sigmoid(),
#         )
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 1, bias=False),
#             nn.BatchNorm2d(channels // 4),
#             nn.GELU(),
#             nn.Conv2d(channels // 4, 1, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, feat_map):
#         # feat_map: [B, T, C, H, W]
#         B, T, C, H, W = feat_map.shape
#         pos = self.text_proj(self.prompt_feat.to(feat_map.device))  # [K, C]
#         pos = pos.mean(0)  # [C]
#         ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

#         x = feat_map * (1.0 + ch_gate)
#         gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)

#         # Residual gate avoids destroying useful RGB features at early training.
#         guided = feat_map * (1.0 + gate) * (1.0 + ch_gate)
#         return guided, gate


# class NuisanceDetachHead(nn.Module):
#     """Qwen nuisance captions -> GRL/NBDetach.

#     Negative text tells the model what to suppress:
#       clothing, carrying/bag, background, occlusion.
#     """

#     def __init__(self, in_dim, text_dim, sub_dim=256, factors=None, grl_lambda=0.05, temperature=0.07):
#         super().__init__()
#         self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
#         self.grl_lambda = grl_lambda
#         self.temperature = temperature

#         self.img_heads = nn.ModuleDict({
#             k: nn.Sequential(
#                 nn.Linear(in_dim, sub_dim),
#                 nn.BatchNorm1d(sub_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(sub_dim, sub_dim),
#             )
#             for k in self.factors
#         })
#         self.txt_heads = nn.ModuleDict({
#             k: nn.Sequential(
#                 nn.Linear(text_dim, sub_dim),
#                 nn.LayerNorm(sub_dim),
#             )
#             for k in self.factors
#         })

#     def forward(self, rgb_global, text_feats: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None):
#         out = {}
#         losses = {}
#         rgb_rev = grad_reverse(rgb_global, self.grl_lambda)

#         for k in self.factors:
#             if k not in text_feats or text_feats[k] is None:
#                 continue
#             txt = text_feats[k]
#             if txt.dim() == 3:
#                 txt = txt.mean(1)
#             txt = txt.float()

#             z_img = self.img_heads[k](rgb_rev.float())
#             z_txt = self.txt_heads[k](txt)
#             z_img = l2n(z_img, dim=-1)
#             z_txt = l2n(z_txt, dim=-1)

#             mask = None
#             if masks is not None and k in masks and masks[k] is not None:
#                 mask = masks[k]
#                 if mask.dim() > 1:
#                     mask = mask.float().view(mask.size(0), -1).mean(-1)

#             losses[k] = masked_symmetric_infonce(z_img, z_txt, mask=mask, temperature=self.temperature)
#             out[k] = {'img': z_img, 'txt': z_txt, 'mask': mask}

#         return out, losses


# # ============================================================
# # Module C: Geometry / Silhouette Anchor
# # ============================================================

# class SilhouetteAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512):
#         super().__init__()
#         self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

#     def forward(self, sil):
#         # sil: [B, T, H, W] or [B,T,1,H,W]
#         if sil.dim() == 4:
#             sil = sil.unsqueeze(2)
#         elif sil.dim() == 5 and sil.shape[2] != 1:
#             sil = sil[:, :, :1]
#         return self.net(sil.float())


# class PointCloudAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512, parts_num=16, in_dim=3):
#         super().__init__()
#         self.parts_num = parts_num
#         self.out_channels = out_channels
#         self.point_mlp = nn.Sequential(
#             nn.Linear(in_dim, 64),
#             nn.GELU(),
#             nn.Linear(64, 128),
#             nn.GELU(),
#             nn.Linear(128, out_channels),
#             nn.GELU(),
#         )
#         self.norm = nn.LayerNorm(out_channels)

#     def forward(self, pc):
#         # pc: [B, T, N, C]
#         B, T, N, C = pc.shape
#         pc = pc[..., :C].float()
#         xyz = pc[..., :3]
#         z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
#         z_min = z.min(dim=-1, keepdim=True).values
#         z_max = z.max(dim=-1, keepdim=True).values
#         z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

#         feat = self.point_mlp(pc.reshape(B * T * N, C)).reshape(B, T, N, self.out_channels)
#         feat = self.norm(feat)

#         part_feats = []
#         for p in range(self.parts_num):
#             lo = p / self.parts_num
#             hi = (p + 1) / self.parts_num
#             mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
#             denom = mask.sum(dim=2).clamp_min(1.0)
#             pooled = (feat * mask).sum(dim=2) / denom  # [B,T,C]
#             part_feats.append(pooled)
#         part_feats = torch.stack(part_feats, dim=-1)  # [B,T,C,P]
#         return part_feats


# # ============================================================
# # Module D: Cycle-Aware Temporal Encoder
# # ============================================================

# class CycleAwareTemporalEncoder(nn.Module):
#     """Temporal encoder with frame, diff, and frequency branches.

#     It is not a pure pooling module: it learns dynamic biometric cues through
#     temporal attention, short-term difference tokens, and FFT magnitude tokens.
#     """

#     def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1, future_pred=True):
#         super().__init__()
#         self.channels = channels
#         self.parts_num = parts_num
#         self.future_pred = future_pred

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=channels,
#             nhead=heads,
#             dim_feedforward=channels * 4,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu',
#             norm_first=True,
#         )
#         self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
#         self.diff_proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.LayerNorm(channels),
#         )
#         self.freq_proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.LayerNorm(channels),
#         )
#         self.attn = nn.Sequential(
#             nn.Linear(channels, channels // 4),
#             nn.Tanh(),
#             nn.Linear(channels // 4, 1),
#         )
#         if future_pred:
#             self.future_head = nn.Sequential(
#                 nn.Linear(channels, channels),
#                 nn.GELU(),
#                 nn.Linear(channels, channels),
#             )

#     def forward(self, part_seq):
#         # part_seq: [B,T,C,P]
#         B, T, C, P = part_seq.shape
#         x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)  # [B*P,T,C]

#         x_tem = self.temporal(x)

#         if T > 1:
#             diff = x[:, 1:] - x[:, :-1]
#             diff = F.pad(diff, (0, 0, 1, 0))
#         else:
#             diff = torch.zeros_like(x)
#         diff = self.diff_proj(diff)

#         # FFT branch: magnitude of low-frequency temporal spectrum, broadcast to time tokens.
#         if T > 1:
#             freq = torch.fft.rfft(x.float(), dim=1).abs()
#             freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
#         else:
#             freq = x.float().mean(1)
#         freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)

#         y = x_tem + diff + freq
#         att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)  # [B*P,T]
#         pooled = (y * att.unsqueeze(-1)).sum(1)  # [B*P,C]
#         pooled = pooled.reshape(B, P, C).permute(0, 2, 1).contiguous()  # [B,C,P]

#         future_loss = pooled.new_tensor(0.0)
#         if self.future_pred and T > 2:
#             pred = self.future_head(y[:, :-1])
#             target = x[:, 1:].detach()
#             future_loss = F.mse_loss(pred, target)

#         att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()  # [B,T,P]
#         return pooled, future_loss, att


# # ============================================================
# # Module E: Verifier-Guided auxiliary training
# # ============================================================

# class RuleBasedVerifier(nn.Module):
#     """Lightweight verifier losses.

#     This does not call MLLM online. It uses available weak signals:
#       - silhouette / RGB motion magnitude for attention sanity;
#       - nuisance masks for reliability weighting;
#       - same batch hard severity can be handled by triplet sampler/loss.
#     """

#     def __init__(self, attention_weight=0.0):
#         super().__init__()
#         self.attention_weight = attention_weight

#     def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
#         if self.attention_weight <= 0 or gait_attn is None:
#             return gait_attn.new_tensor(0.0)

#         # gait_attn: [B,T,1,Hf,Wf]
#         B, T, _, Hf, Wf = gait_attn.shape
#         target = None

#         if sil_seq is not None:
#             if sil_seq.dim() == 4:
#                 sil = sil_seq.unsqueeze(2)
#             else:
#                 sil = sil_seq[:, :, :1]
#             target = F.interpolate(
#                 sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
#                 size=(Hf, Wf),
#                 mode='bilinear',
#                 align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = (target > 0.05).float()

#         elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
#             gray = rgb_seq.float().mean(2, keepdim=True)
#             diff = torch.zeros_like(gray)
#             diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
#             target = F.interpolate(
#                 diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
#                 size=(Hf, Wf),
#                 mode='bilinear',
#                 align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)

#         if target is None:
#             return gait_attn.new_tensor(0.0)

#         # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
#         with torch.cuda.amp.autocast(enabled=False):
#             loss = F.binary_cross_entropy(
#             gait_attn.float().clamp(1e-6, 1 - 1e-6),
#             target.float()
#             )
#         return loss * self.attention_weight



# # ============================================================
# # Fusion
# # ============================================================

# class ModalityFusion(nn.Module):
#     def __init__(self, channels=512, parts_num=16):
#         super().__init__()
#         self.gate = nn.Sequential(
#             nn.Linear(channels * 2, channels),
#             nn.GELU(),
#             nn.Linear(channels, 2),
#         )
#         self.out = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#         )

#     def forward(self, rgb_feat, anchor_feat=None):
#         # rgb_feat/anchor_feat: [B,C,P]
#         if anchor_feat is None:
#             return self.out(rgb_feat), None
#         g_rgb = rgb_feat.mean(-1)
#         g_anc = anchor_feat.mean(-1)
#         w = torch.softmax(self.gate(torch.cat([g_rgb, g_anc], dim=-1)), dim=-1)  # [B,2]
#         fused = rgb_feat * w[:, 0].view(-1, 1, 1) + anchor_feat * w[:, 1].view(-1, 1, 1)
#         return self.out(fused), w


# # ============================================================
# # Main OpenGait model
# # ============================================================

# class GaitOracle(BaseModel):
#     """GaitOracle: MLLM-supervised, geometry-anchored dynamic biometric gait framework.

#     Five modules:
#       A. Foundation RGB Encoder
#       B. Positive-Negative Text Supervisor
#       C. Geometry / Silhouette Anchor
#       D. Cycle-Aware Temporal Encoder
#       E. Verifier-Guided Training Loop

#     Inference uses RGB + optional Sil/PC only. Qwen nuisance captions and text
#     features are training-only supervision signals.
#     """

#     def build_network(self, model_cfg):
#         self.model_cfg = model_cfg
#         C = model_cfg.get('channels', 512)
#         P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
#         self.channels = C
#         self.parts_num = P

#         # Input indices after data_in_use.
#         self.rgb_index = model_cfg.get('rgb_index', 0)
#         self.sil_index = model_cfg.get('sil_index', None)
#         self.pc_index = model_cfg.get('pc_index', None)
#         self.nuisance_indices = model_cfg.get('nuisance_indices', {})
#         self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})

#         # Module A
#         rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
#         if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
#             self.rgb_encoder = TimmDenseRGBEncoder(
#                 model_name=rgb_backbone_cfg.get('model_name', 'vit_small_patch14_dinov2.lvd142m'),
#                 out_channels=C,
#                 pretrained=rgb_backbone_cfg.get('pretrained', True),
#                 freeze=rgb_backbone_cfg.get('freeze', True),
#                 checkpoint_path=rgb_backbone_cfg.get('checkpoint_path', None),
#                 img_size=rgb_backbone_cfg.get('img_size', (224, 112)),
#                 chunk_size=rgb_backbone_cfg.get('chunk_size', 16),
#             )
#         else:
#             self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

#         # Module B positive branch
#         pos_cfg = model_cfg.get('positive_text', {})
#         self.positive_gate = PositiveTextGate(
#             text_dim=pos_cfg.get('text_dim', model_cfg.get('text_dim', 3584)),
#             channels=C,
#             prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
#             num_prompts=pos_cfg.get('num_prompts', 4),
#             trainable_prompts=pos_cfg.get('trainable', False),
#         )

#         # Module C
#         self.use_sil_anchor = self.sil_index is not None
#         self.use_pc_anchor = self.pc_index is not None
#         self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
#         self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None

#         self.HPP = HorizontalPoolingPyramid(bin_num=[P])

#         # Module D
#         temporal_cfg = model_cfg.get('cycle_temporal', {})
#         self.rgb_temporal = CycleAwareTemporalEncoder(
#             channels=C,
#             parts_num=P,
#             heads=temporal_cfg.get('heads', 4),
#             layers=temporal_cfg.get('layers', 1),
#             dropout=temporal_cfg.get('dropout', 0.1),
#             future_pred=temporal_cfg.get('future_pred', True),
#         )
#         self.anchor_temporal = CycleAwareTemporalEncoder(
#             channels=C,
#             parts_num=P,
#             heads=temporal_cfg.get('heads', 4),
#             layers=temporal_cfg.get('layers', 1),
#             dropout=temporal_cfg.get('dropout', 0.1),
#             future_pred=False,
#         )

#         # Fusion
#         self.fusion = ModalityFusion(channels=C, parts_num=P)

#         # Identity heads
#         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
#         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

#         # Module B negative branch
#         nuis_cfg = model_cfg.get('nuisance_detach', {})
#         self.use_nuisance = nuis_cfg.get('enabled', False)
#         if self.use_nuisance:
#             self.nuisance_detach = NuisanceDetachHead(
#                 in_dim=C,
#                 text_dim=nuis_cfg.get('text_dim', model_cfg.get('text_dim', 3584)),
#                 sub_dim=nuis_cfg.get('sub_dim', 256),
#                 factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
#                 grl_lambda=nuis_cfg.get('grl_lambda', 0.05),
#                 temperature=nuis_cfg.get('temperature', 0.07),
#             )
#             self.nuisance_loss_weight = nuis_cfg.get('loss_weight', 0.05)
#         else:
#             self.nuisance_detach = None
#             self.nuisance_loss_weight = 0.0

#         # Module E
#         ver_cfg = model_cfg.get('verifier', {})
#         self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))

#         # Auxiliary weights
#         self.rgb_anchor_weight = model_cfg.get('rgb_anchor_weight', 0.2)
#         self.future_weight = model_cfg.get('future_weight', 0.02)

#     def _to_rgb(self, x):
#         if x.dim() == 4:
#             # [B,T,H,W] -> [B,T,3,H,W]
#             x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
#         elif x.dim() == 5:
#             if x.shape[2] == 1:
#                 x = x.repeat(1, 1, 3, 1, 1)
#             elif x.shape[-1] == 3 and x.shape[2] != 3:
#                 x = x.permute(0, 1, 4, 2, 3)
#         return x.float()

#     def _to_text_feat(self, x):
#         # OpenGait fixed sampler usually gives [B,T,D]; reduce over sampled frames.
#         if x is None:
#             return None
#         x = x.float()
#         if x.dim() == 3:
#             return x.mean(1)
#         if x.dim() == 2:
#             return x
#         return x.view(x.size(0), -1)

#     def _get_optional(self, ipts, idx):
#         if idx is None:
#             return None
#         if isinstance(idx, str):
#             idx = int(idx)
#         if idx < 0 or idx >= len(ipts):
#             return None
#         return ipts[idx]

#     def _dense_to_part_seq(self, feat_map):
#         # feat_map [B,T,C,H,W] -> part_seq [B,T,C,P]
#         B, T, C, H, W = feat_map.shape
#         feat = feat_map.reshape(B * T, C, H, W)
#         part = self.HPP(feat)  # [B*T,C,P]
#         return part.reshape(B, T, C, -1)

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs

#         rgb = self._to_rgb(ipts[self.rgb_index])
#         sil = self._get_optional(ipts, self.sil_index)
#         pc = self._get_optional(ipts, self.pc_index)

#         # Module A: RGB dense feature.
#         rgb_map = self.rgb_encoder(rgb)

#         # Module B positive: gait-positive text gate.
#         rgb_guided, gait_attn = self.positive_gate(rgb_map)

#         # Module D: RGB cycle-aware temporal encoding.
#         rgb_part_seq = self._dense_to_part_seq(rgb_guided)
#         rgb_feat, future_loss, temporal_attn = self.rgb_temporal(rgb_part_seq)

#         # Module C: anchor branch.
#         anchor_feat = None
#         sil_for_verifier = None
#         if pc is not None and self.pc_encoder is not None:
#             pc_part_seq = self.pc_encoder(pc.float())  # [B,T,C,P]
#             anchor_feat, _, _ = self.anchor_temporal(pc_part_seq)
#         elif sil is not None and self.sil_encoder is not None:
#             sil_for_verifier = sil
#             sil_map = self.sil_encoder(sil.float())
#             sil_part_seq = self._dense_to_part_seq(sil_map)
#             anchor_feat, _, _ = self.anchor_temporal(sil_part_seq)

#         # Fusion and ID heads.
#         fused_feat, modality_gate = self.fusion(rgb_feat, anchor_feat)
#         embed_1 = self.FCs(fused_feat)
#         embed_2, logits = self.BNNecks(embed_1)

#         # Training losses.
#         training_feat = {
#             'triplet': {'embeddings': embed_1, 'labels': labs},
#             'softmax': {'logits': logits, 'labels': labs},
#         }

#         # RGB-anchor consistency.
#         if anchor_feat is not None and self.rgb_anchor_weight > 0:
#             rgb_g = rgb_feat.mean(-1)
#             anc_g = anchor_feat.mean(-1)
#             training_feat['loss_rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

#         # Future latent prediction.
#         if self.future_weight > 0:
#             training_feat['loss_future_pred'] = future_loss * self.future_weight

#         # Nuisance detach.
#         nuisance_out = {}
#         if self.use_nuisance:
#             text_feats = {}
#             masks = {}
#             for k, idx in self.nuisance_indices.items():
#                 text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
#             for k, idx in self.nuisance_mask_indices.items():
#                 masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
#             nuisance_out, nuisance_losses = self.nuisance_detach(rgb_feat.mean(-1), text_feats, masks)
#             total_nuis = None
#             for _, v in nuisance_losses.items():
#                 total_nuis = v if total_nuis is None else total_nuis + v
#             if total_nuis is not None:
#                 training_feat['loss_nuisance_detach'] = total_nuis * self.nuisance_loss_weight

#         # Rule-based verifier loss.
#         attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
#         if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
#             training_feat['loss_attention_verifier'] = attn_loss

#         visual_summary = {
#             'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
#             'image/gait_attention': rearrange(gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'),
#         }

#         retval = {
#             'training_feat': training_feat,
#             'visual_summary': visual_summary,
#             'inference_feat': {
#                 'embeddings': embed_1
#             }
#         }
#         return retval
    






# import os
# import math
# from typing import Dict, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# # ============================================================
# # Utilities
# # ============================================================

# class GradReverseFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd):
#         ctx.lambd = float(lambd)
#         return x.view_as(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return -ctx.lambd * grad_output, None


# def grad_reverse(x, lambd=1.0):
#     return GradReverseFn.apply(x, lambd)


# def l2n(x, dim=-1, eps=1e-12):
#     return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# def _scalar_like(x, value=0.0):
#     return x.new_tensor(float(value))


# def symmetric_infonce(z1, z2, temperature=0.07):
#     if z1 is None or z2 is None or z1.size(0) <= 1 or z2.size(0) <= 1:
#         base = z1 if z1 is not None else z2
#         return _scalar_like(base, 0.0)
#     z1 = l2n(z1.float(), dim=-1)
#     z2 = l2n(z2.float(), dim=-1)
#     logits = z1 @ z2.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)
#     return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
#     if z_img is None or z_txt is None:
#         base = z_img if z_img is not None else z_txt
#         return _scalar_like(base, 0.0)

#     if mask is not None:
#         mask = mask.float().view(mask.size(0), -1).mean(-1)
#         valid = mask > 0
#         if valid.sum() <= 1:
#             return z_img.new_tensor(0.0)
#         z_img = z_img[valid]
#         z_txt = z_txt[valid]
#         weights = mask[valid].clamp(0, 1)
#     else:
#         if z_img.size(0) <= 1:
#             return z_img.new_tensor(0.0)
#         weights = None

#     z_img = l2n(z_img.float(), dim=-1)
#     z_txt = l2n(z_txt.float(), dim=-1)
#     logits = z_img @ z_txt.t() / temperature
#     labels = torch.arange(logits.size(0), device=logits.device)

#     if weights is None:
#         return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

#     loss_i = F.cross_entropy(logits, labels, reduction='none')
#     loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
#     weights = weights / weights.mean().clamp_min(1e-6)
#     return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


# def ramp_value(step: int, start: int, length: int, max_value: float = 1.0):
#     if length <= 0:
#         return max_value if step >= start else 0.0
#     if step < start:
#         return 0.0
#     return float(max_value) * min(1.0, max(0.0, (step - start) / float(length)))


# # ============================================================
# # Text compression: keep old Qwen 3584-dim features compatible
# # ============================================================

# class TextEmbeddingAdapter(nn.Module):
#     """Compress high-dimensional MLLM text embeddings to a gait-sized code.

#     Existing Qwen2.5-VL features such as 3584-dim prompt/nuisance embeddings can be
#     used directly.  The adapter crops/pads unexpected dimensions, then learns a
#     low-rank bottleneck.  No offline feature regeneration is required.
#     """

#     def __init__(self, input_dim=3584, code_dim=256, hidden_dim=1024, dropout=0.1, l2_norm=True):
#         super().__init__()
#         self.input_dim = int(input_dim)
#         self.code_dim = int(code_dim)
#         self.l2_norm = bool(l2_norm)
#         hidden_dim = int(hidden_dim or max(code_dim * 2, min(input_dim, 1024)))
#         self.in_norm = nn.LayerNorm(self.input_dim)
#         self.net = nn.Sequential(
#             nn.Linear(self.input_dim, hidden_dim, bias=False),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, self.code_dim, bias=False),
#             nn.LayerNorm(self.code_dim),
#         )

#     def _fit_dim(self, x):
#         d = x.size(-1)
#         if d == self.input_dim:
#             return x
#         if d > self.input_dim:
#             return x[..., :self.input_dim]
#         pad = x.new_zeros(*x.shape[:-1], self.input_dim - d)
#         return torch.cat([x, pad], dim=-1)

#     def forward(self, x):
#         if x is None:
#             return None
#         if x.dim() == 3:
#             x = x.mean(1)
#         elif x.dim() > 3:
#             x = x.view(x.size(0), -1)
#         x = self._fit_dim(x.float())
#         z = self.net(self.in_norm(x))
#         return l2n(z, dim=-1) if self.l2_norm else z


# # ============================================================
# # Module A: RGB encoder
# # ============================================================

# class TinyDenseRGBEncoder(nn.Module):
#     """Dependency-free dense RGB encoder; use timm backbone if available/needed."""

#     def __init__(self, out_channels=512, in_channels=3):
#         super().__init__()
#         c1 = max(out_channels // 8, 32)
#         c2 = max(out_channels // 4, 64)
#         c3 = max(out_channels // 2, 128)
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(c1),
#             nn.GELU(),
#             nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.GELU(),
#             nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(c3),
#             nn.GELU(),
#             nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         self.out_channels = out_channels

#     def forward(self, x):
#         # x: [B, T, C, H, W]
#         B, T, C, H, W = x.shape
#         y = self.net(x.reshape(B * T, C, H, W))
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# class TimmDenseRGBEncoder(nn.Module):
#     """Optional dense foundation encoder via timm features_only models."""

#     def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', out_channels=512, pretrained=True, freeze=True):
#         super().__init__()
#         try:
#             import timm
#         except Exception as exc:
#             raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

#         self.model_name = model_name
#         self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
#         chs = self.backbone.feature_info.channels()
#         in_c = chs[-1]
#         self.proj = nn.Sequential(
#             nn.Conv2d(in_c, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU(),
#         )
#         if freeze:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#         self.out_channels = out_channels

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         y = self.backbone(x.reshape(B * T, C, H, W))[-1]
#         y = self.proj(y)
#         _, C2, H2, W2 = y.shape
#         return y.reshape(B, T, C2, H2, W2)


# # ============================================================
# # Module B1: positive gait prompt gate
# # ============================================================

# class PositiveTextGate(nn.Module):
#     """Fixed gait-positive prompt bank -> channel/spatial gait gate.

#     This preserves motion-biometric cues before early RGB-anchor fusion.  The
#     prompt bank can still be the old Qwen .npy file; the high-dimensional text
#     vectors are compressed by TextEmbeddingAdapter inside the module.
#     """

#     def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4,
#                  trainable_prompts=False, text_code_dim=256, text_hidden_dim=1024, dropout=0.1):
#         super().__init__()
#         self.channels = channels

#         if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
#             arr = np.load(prompt_feat_path).astype('float32')
#             if arr.ndim == 1:
#                 arr = arr[None, :]
#             prompt = torch.from_numpy(arr)
#             text_dim = int(prompt.shape[-1])
#         else:
#             prompt = torch.randn(num_prompts, text_dim) * 0.02

#         self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
#         self.text_adapter = TextEmbeddingAdapter(
#             input_dim=text_dim,
#             code_dim=text_code_dim,
#             hidden_dim=text_hidden_dim,
#             dropout=dropout,
#             l2_norm=False,
#         )
#         self.text_to_channel = nn.Sequential(
#             nn.Linear(text_code_dim, channels),
#             nn.LayerNorm(channels),
#             nn.GELU(),
#         )
#         self.channel_gate = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.Sigmoid(),
#         )
#         mid = max(channels // 4, 16)
#         self.spatial_gate = nn.Sequential(
#             nn.Conv2d(channels, mid, 1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.GELU(),
#             nn.Conv2d(mid, 1, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, feat_map):
#         # feat_map: [B, T, C, H, W]
#         B, T, C, H, W = feat_map.shape
#         prompt = self.prompt_feat.to(device=feat_map.device)
#         pos_code = self.text_adapter(prompt).mean(0)  # [Dtxt]
#         pos = self.text_to_channel(pos_code)          # [C]
#         ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

#         x = feat_map * (1.0 + ch_gate)
#         sp_gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)
#         guided = feat_map * (1.0 + ch_gate) * (1.0 + sp_gate)
#         return guided, sp_gate


# # ============================================================
# # Module B2: staged nuisance removal
# # ============================================================


# class StagedNuisanceRemoval(nn.Module):
#     """Immediate GRL nuisance detachment for gait identity features.

#     This class intentionally keeps the original StagedNuisanceRemoval API
#     (pre_temporal / post_temporal) but changes the behavior back to the
#     empirically strong initial design:

#       - nuisance adversarial learning starts from iteration 0;
#       - no reconstruction, SAE, future prediction, or order-bound objective;
#       - Qwen nuisance text/mask is training-only supervision;
#       - the main forward identity path is not structurally split or rewritten.

#     pre_temporal returns an identity reliability gate by default, because we do
#     not want weak Qwen masks to hard-suppress gait parts at the beginning.
#     post_temporal applies GRL to the identity embedding and optimizes nuisance
#     heads to align with Qwen nuisance factors; the backbone receives the reversed
#     gradient and learns to remove nuisance information.
#     """

#     def __init__(self, in_dim, parts_num, text_dim=3584, text_code_dim=256, text_hidden_dim=1024,
#                  sub_dim=256, factors=None, grl_lambda=0.05, temperature=0.07,
#                  warmup_iters=0, probe_iters=0, adv_ramp_iters=0,
#                  reliability_start_iter=1000000000, reliability_ramp_iters=1,
#                  probe_loss_weight=0.0, adv_loss_weight=0.05, mask_loss_weight=0.0,
#                  max_suppression=0.0, detach_reliability_gate=True, factor_weights=None,
#                  dropout=0.1, loss_weight=None, immediate_adv=True, **unused):
#         super().__init__()
#         self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
#         self.parts_num = int(parts_num)
#         self.grl_lambda = float(grl_lambda)
#         self.temperature = float(temperature)
#         self.immediate_adv = bool(immediate_adv)

#         # Use loss_weight as backward-compatible shorthand if provided.
#         if loss_weight is not None:
#             adv_loss_weight = float(loss_weight)
#         self.adv_loss_weight = float(adv_loss_weight)
#         self.probe_loss_weight = float(probe_loss_weight)
#         self.mask_loss_weight = float(mask_loss_weight)
#         self.max_suppression = float(max_suppression)
#         self.detach_reliability_gate = bool(detach_reliability_gate)

#         self.factor_weights = {k: 1.0 for k in self.factors}
#         if factor_weights:
#             self.factor_weights.update({str(k): float(v) for k, v in factor_weights.items()})

#         self.text_adapter = TextEmbeddingAdapter(
#             text_dim, text_code_dim, text_hidden_dim, dropout=dropout, l2_norm=True
#         )
#         self.txt_heads = nn.ModuleDict({
#             k: nn.Sequential(nn.Linear(text_code_dim, sub_dim), nn.LayerNorm(sub_dim))
#             for k in self.factors
#         })
#         self.adv_heads = nn.ModuleDict({
#             k: nn.Sequential(
#                 nn.Linear(in_dim, sub_dim),
#                 nn.LayerNorm(sub_dim),
#                 nn.GELU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(sub_dim, sub_dim),
#                 nn.LayerNorm(sub_dim),
#             ) for k in self.factors
#         })

#     def schedule(self, step: int):
#         # Kept for logger compatibility. The adversarial branch is active from iter=0.
#         one = 1.0 if self.immediate_adv else ramp_value(step, 0, 1, 1.0)
#         return {'probe': one, 'adv': one, 'rel': 0.0}

#     def _target_mask(self, masks, factor):
#         if masks is None or factor not in masks or masks[factor] is None:
#             return None
#         m = masks[factor].float()
#         if m.dim() > 1:
#             m = m.view(m.size(0), -1).mean(-1)
#         return m.clamp(0, 1)

#     def _text_z(self, text_feats, factor):
#         if text_feats is None or factor not in text_feats or text_feats[factor] is None:
#             return None
#         code = self.text_adapter(text_feats[factor])
#         return l2n(self.txt_heads[factor](code), dim=-1)

#     def pre_temporal(self, visual_global, text_feats=None, masks=None, step: int = 0):
#         B = visual_global.size(0)
#         reliability = visual_global.new_ones(B, 1, 1, self.parts_num)
#         diag = {
#             'sched_probe': visual_global.new_tensor(1.0),
#             'sched_adv': visual_global.new_tensor(1.0),
#             'sched_rel': visual_global.new_tensor(0.0),
#             'mean_reliability': reliability.mean().detach(),
#         }
#         return reliability, {}, diag

#     def post_temporal(self, id_global, text_feats=None, masks=None, step: int = 0):
#         losses = {}
#         if self.adv_loss_weight <= 0:
#             return losses

#         # id_global: [B,C]. GRL makes the identity feature nuisance-invariant.
#         id_rev = grad_reverse(id_global.float(), self.grl_lambda)

#         for f in self.factors:
#             z_txt = self._text_z(text_feats, f)
#             if z_txt is None:
#                 continue
#             target = self._target_mask(masks, f)
#             z_id = l2n(self.adv_heads[f](id_rev), dim=-1)
#             fac_w = float(self.factor_weights.get(f, 1.0))
#             losses[f'adv_{f}'] = masked_symmetric_infonce(
#                 z_id, z_txt, mask=target, temperature=self.temperature
#             ) * (self.adv_loss_weight * fac_w)

#         return losses


# # Backward-compatible alias for old imports/checkpoints that refer to this name.
# NuisanceDetachHead = StagedNuisanceRemoval


# # ============================================================
# # Module C: geometry / silhouette / point-cloud anchors
# # ============================================================

# class SilhouetteAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512):
#         super().__init__()
#         self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

#     def forward(self, sil):
#         # sil: [B,T,H,W] or [B,T,1,H,W]
#         if sil.dim() == 4:
#             sil = sil.unsqueeze(2)
#         elif sil.dim() == 5 and sil.shape[2] != 1:
#             sil = sil[:, :, :1]
#         return self.net(sil.float())


# class PointCloudAnchorEncoder(nn.Module):
#     def __init__(self, out_channels=512, parts_num=16, in_dim=3):
#         super().__init__()
#         self.parts_num = int(parts_num)
#         self.out_channels = int(out_channels)
#         self.in_dim = int(in_dim)
#         self.point_mlp = nn.Sequential(
#             nn.Linear(self.in_dim, 64),
#             nn.GELU(),
#             nn.Linear(64, 128),
#             nn.GELU(),
#             nn.Linear(128, out_channels),
#             nn.GELU(),
#         )
#         self.norm = nn.LayerNorm(out_channels)

#     def _fit_dim(self, pc):
#         d = pc.size(-1)
#         if d == self.in_dim:
#             return pc
#         if d > self.in_dim:
#             return pc[..., :self.in_dim]
#         pad = pc.new_zeros(*pc.shape[:-1], self.in_dim - d)
#         return torch.cat([pc, pad], dim=-1)

#     def forward(self, pc):
#         # pc: [B,T,N,C]
#         B, T, N, _ = pc.shape
#         pc = self._fit_dim(pc.float())
#         xyz = pc[..., :min(3, self.in_dim)]
#         z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
#         z_min = z.min(dim=-1, keepdim=True).values
#         z_max = z.max(dim=-1, keepdim=True).values
#         z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

#         feat = self.point_mlp(pc.reshape(B * T * N, self.in_dim)).reshape(B, T, N, self.out_channels)
#         feat = self.norm(feat)

#         part_feats = []
#         for p in range(self.parts_num):
#             lo = p / self.parts_num
#             hi = (p + 1) / self.parts_num
#             if p == self.parts_num - 1:
#                 mask = ((z_norm >= lo) & (z_norm <= hi)).float().unsqueeze(-1)
#             else:
#                 mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
#             denom = mask.sum(dim=2).clamp_min(1.0)
#             pooled = (feat * mask).sum(dim=2) / denom
#             part_feats.append(pooled)
#         return torch.stack(part_feats, dim=-1)  # [B,T,C,P]


# # ============================================================
# # Module D: early part-token fusion + unified temporal encoder
# # ============================================================


# class EarlyPartTokenFusion(nn.Module):
#     """Structure-guided RGB/Sil residual part-token fusion.

#     RGB keeps discriminative local appearance and fine details.
#     Silhouette provides body structure, part alignment, and background suppression.
#     A lightweight motion-energy branch lets the gate pay more attention to parts
#     with meaningful gait variation instead of blindly averaging modalities.
#     """

#     def __init__(self, channels=512, dropout=0.05,
#                  sil_scale_init=0.45, cross_scale_init=0.15, motion_scale_init=0.10):
#         super().__init__()
#         self.channels = int(channels)

#         self.sil_delta = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#         )
#         self.cross_delta = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#         )
#         self.motion_delta = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#         )

#         # Input: rgb, sil, |rgb-sil|, rgb*sil, rgb_motion, sil_motion.
#         self.gate = nn.Sequential(
#             nn.Conv1d(channels * 6, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv1d(channels, 3, 1),
#             nn.Sigmoid(),
#         )

#         self.out = nn.Sequential(
#             nn.Conv1d(channels, channels, 1, bias=False),
#             nn.BatchNorm1d(channels),
#             nn.GELU(),
#         )

#         self.sil_scale = nn.Parameter(torch.tensor(float(sil_scale_init)))
#         self.cross_scale = nn.Parameter(torch.tensor(float(cross_scale_init)))
#         self.motion_scale = nn.Parameter(torch.tensor(float(motion_scale_init)))

#     def _motion_energy(self, x):
#         # x: [B,T,C,P]. Works for fixed_unordered as sequence-level variation,
#         # not as strict adjacent velocity.
#         if x.size(1) <= 1:
#             return torch.zeros_like(x)
#         center = x - x.mean(dim=1, keepdim=True)
#         roll = x - torch.roll(x, shifts=1, dims=1)
#         return 0.5 * (center.abs() + roll.abs())

#     def forward(self, rgb_part_seq, anchor_part_seq=None):
#         # [B,T,C,P]
#         B, T, C, P = rgb_part_seq.shape
#         if anchor_part_seq is None:
#             x = rgb_part_seq.reshape(B * T, C, P)
#             fused = self.out(x).reshape(B, T, C, P)
#             return fused, {
#                 'fusion_gate_sil': rgb_part_seq.new_tensor(0.0),
#                 'fusion_gate_cross': rgb_part_seq.new_tensor(0.0),
#                 'fusion_gate_motion': rgb_part_seq.new_tensor(0.0),
#             }

#         if anchor_part_seq.size(-1) != P:
#             raise ValueError(f'Anchor parts {anchor_part_seq.size(-1)} != RGB parts {P}')

#         rgb_m = self._motion_energy(rgb_part_seq)
#         anc_m = self._motion_energy(anchor_part_seq)

#         gate_in = torch.cat([
#             rgb_part_seq,
#             anchor_part_seq,
#             (rgb_part_seq - anchor_part_seq).abs(),
#             rgb_part_seq * anchor_part_seq,
#             rgb_m,
#             anc_m,
#         ], dim=2).reshape(B * T, C * 6, P)

#         gates = self.gate(gate_in).reshape(B, T, 3, P)
#         g_sil = gates[:, :, 0:1]
#         g_cross = gates[:, :, 1:2]
#         g_motion = gates[:, :, 2:3]

#         sil_delta = self.sil_delta(anchor_part_seq.reshape(B * T, C, P)).reshape(B, T, C, P)
#         cross_delta = self.cross_delta((0.5 * (rgb_part_seq + anchor_part_seq)).reshape(B * T, C, P)).reshape(B, T, C, P)
#         motion_delta = self.motion_delta((0.5 * (rgb_m + anc_m)).reshape(B * T, C, P)).reshape(B, T, C, P)

#         fused = (
#             rgb_part_seq
#             + self.sil_scale * g_sil * sil_delta
#             + self.cross_scale * g_cross * cross_delta
#             + self.motion_scale * g_motion * motion_delta
#         )
#         fused = self.out(fused.reshape(B * T, C, P)).reshape(B, T, C, P)

#         diag = {
#             'fusion_gate_sil': g_sil.detach().mean(),
#             'fusion_gate_cross': g_cross.detach().mean(),
#             'fusion_gate_motion': g_motion.detach().mean(),
#             'fusion_sil_scale': self.sil_scale.detach(),
#             'fusion_cross_scale': self.cross_scale.detach(),
#             'fusion_motion_scale': self.motion_scale.detach(),
#         }
#         return fused, diag





# class CycleAwareTemporalEncoder(nn.Module):
#     """Order-weak gait motion/frequency temporal mixer.

#     fixed_unordered sampling should not use future prediction or reconstruction,
#     but gait still benefits from frame variation, difference statistics, temporal
#     convolution and frequency magnitude. This module keeps diff/FFT information
#     as motion statistics rather than strict causal adjacent-frame dynamics.
#     """

#     def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1,
#                  future_pred=False, use_center_diff=True, use_roll_diff=True,
#                  use_adjacent_diff=True, use_abs_motion=True, use_fft=True,
#                  fft_bands=4, use_dwconv=True, use_part_mixer=True):
#         super().__init__()
#         self.channels = int(channels)
#         self.parts_num = int(parts_num)
#         self.future_pred = bool(future_pred)
#         self.use_center_diff = bool(use_center_diff)
#         self.use_roll_diff = bool(use_roll_diff)
#         self.use_adjacent_diff = bool(use_adjacent_diff)
#         self.use_abs_motion = bool(use_abs_motion)
#         self.use_fft = bool(use_fft)
#         self.fft_bands = int(fft_bands)
#         self.use_dwconv = bool(use_dwconv)
#         self.use_part_mixer = bool(use_part_mixer)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=channels,
#             nhead=heads,
#             dim_feedforward=channels * 4,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu',
#             norm_first=True,
#         )
#         self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)

#         self.motion_proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.LayerNorm(channels),
#         )
#         self.freq_proj = nn.Sequential(
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.LayerNorm(channels),
#         )
#         self.dw_conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
#         self.dw_norm = nn.LayerNorm(channels)

#         self.branch_gate = nn.Sequential(
#             nn.LayerNorm(channels * 4),
#             nn.Linear(channels * 4, channels),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(channels, 3),
#             nn.Sigmoid(),
#         )

#         part_layer = nn.TransformerEncoderLayer(
#             d_model=channels,
#             nhead=heads,
#             dim_feedforward=channels * 2,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu',
#             norm_first=True,
#         )
#         self.part_mixer = nn.TransformerEncoder(part_layer, num_layers=1)
#         self.attn = nn.Sequential(
#             nn.Linear(channels, max(channels // 4, 16)),
#             nn.Tanh(),
#             nn.Linear(max(channels // 4, 16), 1),
#         )

#         self.future_head = (
#             nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
#             if self.future_pred else None
#         )

#     def _motion_branch(self, x):
#         # x: [B*P,T,C]
#         terms = []
#         if x.size(1) <= 1:
#             return torch.zeros_like(x)

#         if self.use_center_diff:
#             terms.append(x - x.mean(dim=1, keepdim=True))

#         if self.use_roll_diff:
#             terms.append(x - torch.roll(x, shifts=1, dims=1))

#         if self.use_adjacent_diff:
#             adj = x[:, 1:] - x[:, :-1]
#             adj = F.pad(adj, (0, 0, 1, 0))
#             terms.append(adj)

#         if not terms:
#             return torch.zeros_like(x)

#         motion = sum(terms) / float(len(terms))
#         if self.use_abs_motion:
#             motion = motion + motion.abs()
#         return self.motion_proj(motion)

#     def _freq_branch(self, x):
#         if (not self.use_fft) or x.size(1) <= 1:
#             return torch.zeros_like(x)
#         freq = torch.fft.rfft(x.float(), dim=1).abs()
#         # Remove DC; use low-band magnitude as a gait-cycle-like statistic.
#         if freq.size(1) > 1:
#             hi = min(freq.size(1), max(2, self.fft_bands + 1))
#             low = freq[:, 1:hi].mean(1)
#             if hi < freq.size(1):
#                 rest = freq[:, hi:].mean(1)
#                 stat = 0.5 * (low + rest)
#             else:
#                 stat = low
#         else:
#             stat = freq.mean(1)
#         stat = self.freq_proj(stat.to(dtype=x.dtype))
#         return stat.unsqueeze(1).expand(-1, x.size(1), -1)

#     def _conv_branch(self, x):
#         if not self.use_dwconv:
#             return torch.zeros_like(x)
#         return self.dw_norm(self.dw_conv(x.transpose(1, 2)).transpose(1, 2))

#     def forward(self, part_seq):
#         # part_seq: [B,T,C,P]
#         B, T, C, P = part_seq.shape
#         x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)

#         frame = self.temporal(x)
#         motion = self._motion_branch(x)
#         freq = self._freq_branch(x)
#         conv = self._conv_branch(x)

#         gate_in = torch.cat([frame, motion, freq.to(dtype=frame.dtype), conv], dim=-1)
#         gates = self.branch_gate(gate_in)
#         g_motion = gates[..., 0:1]
#         g_freq = gates[..., 1:2]
#         g_conv = gates[..., 2:3]

#         y = frame + g_motion * motion + g_freq * freq.to(dtype=frame.dtype) + g_conv * conv

#         att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)
#         pooled = (y * att.unsqueeze(-1)).sum(1).reshape(B, P, C)
#         if self.use_part_mixer:
#             pooled = self.part_mixer(pooled)
#         pooled = pooled.permute(0, 2, 1).contiguous()

#         future_loss = pooled.new_tensor(0.0)
#         if self.future_head is not None and T > 2:
#             # Kept only for backward compatibility; disabled in the provided config.
#             pred = self.future_head(y[:, :-1])
#             target = x[:, 1:].detach()
#             future_loss = F.mse_loss(pred, target)

#         att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()
#         return pooled, future_loss, att




# # ============================================================
# # Module E: lightweight verifier regularization
# # ============================================================

# class RuleBasedVerifier(nn.Module):
#     def __init__(self, attention_weight=0.0):
#         super().__init__()
#         self.attention_weight = float(attention_weight)

#     def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
#         if self.attention_weight <= 0 or gait_attn is None:
#             return gait_attn.new_tensor(0.0)
#         B, T, _, Hf, Wf = gait_attn.shape
#         target = None
#         if sil_seq is not None:
#             sil = sil_seq.unsqueeze(2) if sil_seq.dim() == 4 else sil_seq[:, :, :1]
#             target = F.interpolate(
#                 sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
#                 size=(Hf, Wf), mode='bilinear', align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = (target > 0.05).float()
#         elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
#             gray = rgb_seq.float().mean(2, keepdim=True)
#             diff = torch.zeros_like(gray)
#             diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
#             target = F.interpolate(
#                 diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
#                 size=(Hf, Wf), mode='bilinear', align_corners=False,
#             ).reshape(B, T, 1, Hf, Wf)
#             target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
#         if target is None:
#             return gait_attn.new_tensor(0.0)
#         # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
#         if self.attention_weight <= 0:
#             return gait_attn.new_tensor(0.0)

#         with torch.cuda.amp.autocast(enabled=False):
#             loss = F.binary_cross_entropy(
#                 gait_attn.float().clamp(1e-6, 1 - 1e-6),
#                 target.float()
#             )

#         return loss * self.attention_weight


# # ============================================================
# # Main OpenGait model
# # ============================================================
# class GaitOraclePlusPlus(BaseModel):
#     """GaitOracle strong version.

#     Key changes against the previous version:
#       1) RGB/geometry anchor are fused at part-token level before temporal encoding.
#       2) Qwen 3584-d text embeddings are compressed to gait-sized codes on the fly.
#       3) Nuisance removal is staged: warmup -> probe -> adversarial -> reliability refinement.
#       4) Existing positive prompt .npy and qwen25vl_*_feat.pkl / mask.pkl files remain compatible.
#     """

#     def build_network(self, model_cfg):
#         self.model_cfg = model_cfg
#         C = model_cfg.get('channels', 512)
#         P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
#         self.channels = int(C)
#         self.parts_num = int(P)
#         self.rgb_index = model_cfg.get('rgb_index', 0)
#         self.sil_index = model_cfg.get('sil_index', None)
#         self.pc_index = model_cfg.get('pc_index', None)
#         self.nuisance_indices = model_cfg.get('nuisance_indices', {})
#         self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
#         self.register_buffer('_forward_iter', torch.zeros((), dtype=torch.long), persistent=False)

#         text_cfg = model_cfg.get('text_adapter', {})
#         text_dim = int(text_cfg.get('input_dim', model_cfg.get('text_dim', 3584)))
#         text_code_dim = int(text_cfg.get('code_dim', 256))
#         text_hidden_dim = int(text_cfg.get('hidden_dim', 1024))
#         text_dropout = float(text_cfg.get('dropout', 0.1))

#         rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
#         if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
#             self.rgb_encoder = TimmDenseRGBEncoder(
#                 model_name=rgb_backbone_cfg.get('model_name', 'vit_base_patch14_dinov2.lvd142m'),
#                 out_channels=C,
#                 pretrained=rgb_backbone_cfg.get('pretrained', True),
#                 freeze=rgb_backbone_cfg.get('freeze', True),
#             )
#         else:
#             self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

#         pos_cfg = model_cfg.get('positive_text', {})
#         self.positive_gate = PositiveTextGate(
#             text_dim=pos_cfg.get('text_dim', text_dim),
#             channels=C,
#             prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
#             num_prompts=pos_cfg.get('num_prompts', 4),
#             trainable_prompts=pos_cfg.get('trainable', False),
#             text_code_dim=pos_cfg.get('text_code_dim', text_code_dim),
#             text_hidden_dim=pos_cfg.get('text_hidden_dim', text_hidden_dim),
#             dropout=pos_cfg.get('dropout', text_dropout),
#         )

#         self.use_sil_anchor = self.sil_index is not None
#         self.use_pc_anchor = self.pc_index is not None
#         self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
#         self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None
#         self.HPP = HorizontalPoolingPyramid(bin_num=[P])

#         early_cfg = model_cfg.get('early_fusion', {})
#         self.early_fusion = EarlyPartTokenFusion(
#             channels=C,
#             dropout=early_cfg.get('dropout', 0.05),
#             sil_scale_init=early_cfg.get('sil_scale_init', 0.45),
#             cross_scale_init=early_cfg.get('cross_scale_init', 0.15),
#             motion_scale_init=early_cfg.get('motion_scale_init', 0.10),
#         )

#         temporal_cfg = model_cfg.get('cycle_temporal', {})
#         self.temporal = CycleAwareTemporalEncoder(
#             channels=C,
#             parts_num=P,
#             heads=temporal_cfg.get('heads', 4),
#             layers=temporal_cfg.get('layers', 1),
#             dropout=temporal_cfg.get('dropout', 0.1),
#             future_pred=temporal_cfg.get('future_pred', False),
#             use_center_diff=temporal_cfg.get('use_center_diff', True),
#             use_roll_diff=temporal_cfg.get('use_roll_diff', True),
#             use_adjacent_diff=temporal_cfg.get('use_adjacent_diff', True),
#             use_abs_motion=temporal_cfg.get('use_abs_motion', True),
#             use_fft=temporal_cfg.get('use_fft', True),
#             fft_bands=temporal_cfg.get('fft_bands', 4),
#             use_dwconv=temporal_cfg.get('use_dwconv', True),
#             use_part_mixer=temporal_cfg.get('use_part_mixer', True),
#         )

#         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
#         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

#         nuis_cfg = dict(model_cfg.get('nuisance_detach', {}))
#         self.use_nuisance = bool(nuis_cfg.get('enabled', False))
#         legacy_loss_w = float(nuis_cfg.get('loss_weight', 0.03))
#         if self.use_nuisance:
#             self.nuisance_removal = StagedNuisanceRemoval(
#                 in_dim=C,
#                 parts_num=P,
#                 text_dim=nuis_cfg.get('text_dim', text_dim),
#                 text_code_dim=nuis_cfg.get('text_code_dim', text_code_dim),
#                 text_hidden_dim=nuis_cfg.get('text_hidden_dim', text_hidden_dim),
#                 sub_dim=nuis_cfg.get('sub_dim', 256),
#                 factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
#                 grl_lambda=nuis_cfg.get('grl_lambda', 0.08),
#                 temperature=nuis_cfg.get('temperature', 0.07),
#                 warmup_iters=nuis_cfg.get('warmup_iters', 5000),
#                 probe_iters=nuis_cfg.get('probe_iters', 10000),
#                 adv_ramp_iters=nuis_cfg.get('adv_ramp_iters', 10000),
#                 reliability_start_iter=nuis_cfg.get('reliability_start_iter', 15000),
#                 reliability_ramp_iters=nuis_cfg.get('reliability_ramp_iters', 10000),
#                 probe_loss_weight=nuis_cfg.get('probe_loss_weight', legacy_loss_w),
#                 adv_loss_weight=nuis_cfg.get('adv_loss_weight', legacy_loss_w),
#                 mask_loss_weight=nuis_cfg.get('mask_loss_weight', legacy_loss_w * 0.5),
#                 max_suppression=nuis_cfg.get('max_suppression', 0.35),
#                 detach_reliability_gate=nuis_cfg.get('detach_reliability_gate', True),
#                 factor_weights=nuis_cfg.get('factor_weights', None),
#                 dropout=nuis_cfg.get('dropout', text_dropout),
#             )
#         else:
#             self.nuisance_removal = None

#         ver_cfg = model_cfg.get('verifier', {})
#         self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))
#         self.rgb_anchor_weight = float(model_cfg.get('rgb_anchor_weight', 0.1))
#         self.future_weight = float(model_cfg.get('future_weight', 0.01))
#         self.use_scalar_aux_loss = bool(model_cfg.get('use_scalar_aux_loss', True))

#     def _current_step(self):
#         if self.training:
#             self._forward_iter += 1
#         for attr in ('iteration', 'iter', 'current_iter', 'global_step'):
#             if hasattr(self, attr):
#                 try:
#                     val = getattr(self, attr)
#                     if torch.is_tensor(val):
#                         return int(val.detach().item())
#                     return int(val)
#                 except Exception:
#                     pass
#         return int(self._forward_iter.detach().item())

#     def _to_rgb(self, x):
#         if x.dim() == 4:
#             x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
#         elif x.dim() == 5:
#             if x.shape[2] == 1:
#                 x = x.repeat(1, 1, 3, 1, 1)
#             elif x.shape[-1] == 3 and x.shape[2] != 3:
#                 x = x.permute(0, 1, 4, 2, 3)
#         return x.float()

#     def _to_text_feat(self, x):
#         if x is None:
#             return None
#         x = x.float()
#         if x.dim() == 3:
#             return x.mean(1)
#         if x.dim() == 2:
#             return x
#         return x.view(x.size(0), -1)

#     def _get_optional(self, ipts, idx):
#         if idx is None:
#             return None
#         if isinstance(idx, str):
#             if idx.lower() in ('none', 'null'):
#                 return None
#             idx = int(idx)
#         if idx < 0 or idx >= len(ipts):
#             return None
#         return ipts[idx]

#     def _dense_to_part_seq(self, feat_map):
#         B, T, C, H, W = feat_map.shape
#         part = self.HPP(feat_map.reshape(B * T, C, H, W))
#         return part.reshape(B, T, C, -1)

#     def _collect_nuisance_inputs(self, ipts):
#         text_feats, masks = {}, {}
#         for k, idx in self.nuisance_indices.items():
#             text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
#         for k, idx in self.nuisance_mask_indices.items():
#             masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
#         return text_feats, masks

#     def _sum_losses(self, losses: Dict[str, torch.Tensor], base: torch.Tensor):
#         total = base.new_tensor(0.0)
#         for v in losses.values():
#             if v is not None:
#                 total = total + v
#         return total

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs
#         step = self._current_step()
#         rgb = self._to_rgb(ipts[self.rgb_index])
#         sil = self._get_optional(ipts, self.sil_index)
#         pc = self._get_optional(ipts, self.pc_index)

#         rgb_map = self.rgb_encoder(rgb)
#         rgb_guided, gait_attn = self.positive_gate(rgb_map)
#         rgb_part_seq = self._dense_to_part_seq(rgb_guided)

#         anchor_part_seq = None
#         sil_for_verifier = None
#         if pc is not None and self.pc_encoder is not None:
#             anchor_part_seq = self.pc_encoder(pc.float())
#         elif sil is not None and self.sil_encoder is not None:
#             sil_for_verifier = sil
#             sil_map = self.sil_encoder(sil.float())
#             anchor_part_seq = self._dense_to_part_seq(sil_map)

#         nuisance_pre_losses = {}
#         nuisance_post_losses = {}
#         nuisance_diag = {}
#         text_feats, nuisance_masks = ({}, {})
#         if self.use_nuisance:
#             text_feats, nuisance_masks = self._collect_nuisance_inputs(ipts)
#             pre_global = rgb_part_seq.mean(dim=(1, 3))
#             reliability, nuisance_pre_losses, nuisance_diag = self.nuisance_removal.pre_temporal(
#                 pre_global, text_feats=text_feats, masks=nuisance_masks, step=step,
#             )
#             rgb_part_seq = rgb_part_seq * reliability

#         # Early fusion happens before temporal encoding.
#         fused_seq, modality_gate = self.early_fusion(rgb_part_seq, anchor_part_seq)
#         fused_feat, future_loss, temporal_attn = self.temporal(fused_seq)

#         embed_1 = self.FCs(fused_feat)
#         embed_2, logits = self.BNNecks(embed_1)

#         training_feat = {
#             'triplet': {'embeddings': embed_1, 'labels': labs},
#             'softmax': {'logits': logits, 'labels': labs},
#         }

#         aux_losses = {}
#         if anchor_part_seq is not None and self.rgb_anchor_weight > 0:
#             rgb_g = rgb_part_seq.mean(dim=(1, 3))
#             anc_g = anchor_part_seq.mean(dim=(1, 3))
#             aux_losses['rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

#         if self.future_weight > 0:
#             aux_losses['future_pred'] = future_loss * self.future_weight

#         if self.use_nuisance:
#             id_global = fused_feat.mean(-1)
#             nuisance_post_losses = self.nuisance_removal.post_temporal(
#                 id_global, text_feats=text_feats, masks=nuisance_masks, step=step,
#             )
#             aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_pre_losses.items()})
#             aux_losses.update({f'nuisance_{k}': v for k, v in nuisance_post_losses.items()})

#         attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
#         if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
#             aux_losses['attention_verifier'] = attn_loss

#         if self.use_scalar_aux_loss:
#             aux_total = self._sum_losses(aux_losses, embed_1)
#             training_feat['gaitoracle_aux'] = {'loss': aux_total}
#         else:
#             for k, v in aux_losses.items():
#                 training_feat[f'loss_{k}'] = v

#         visual_summary = {
#             'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
#             'image/gait_attention': rearrange(gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'),
#         }

#         if self.training:
#             if isinstance(modality_gate, dict):
#                 for k, v in modality_gate.items():
#                     if torch.is_tensor(v):
#                         visual_summary[f'scalar/{k}'] = v.detach()
#             if self.use_nuisance:
#                 # Scalars are intentionally kept detached for loggers that inspect visual_summary.
#                 for k, v in nuisance_diag.items():
#                     visual_summary[f'scalar/{k}'] = v.detach()

#         return {
#             'training_feat': training_feat,
#             'visual_summary': visual_summary,
#             'inference_feat': {'embeddings': embed_1},
#         }






# # import os
# # import math
# # from typing import Dict, Optional, List, Tuple

# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from einops import rearrange

# # from ..base_model import BaseModel
# # from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


# # # ============================================================
# # # Utilities
# # # ============================================================

# # class GradReverseFn(torch.autograd.Function):
# #     @staticmethod
# #     def forward(ctx, x, lambd):
# #         ctx.lambd = float(lambd)
# #         return x.view_as(x)

# #     @staticmethod
# #     def backward(ctx, grad_output):
# #         return -ctx.lambd * grad_output, None


# # def grad_reverse(x, lambd=1.0):
# #     return GradReverseFn.apply(x, lambd)


# # def l2n(x, dim=-1, eps=1e-12):
# #     return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


# # def symmetric_infonce(z1, z2, temperature=0.07):
# #     z1 = l2n(z1.float(), dim=-1)
# #     z2 = l2n(z2.float(), dim=-1)
# #     logits = z1 @ z2.t() / temperature
# #     labels = torch.arange(logits.size(0), device=logits.device)
# #     return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# # def masked_symmetric_infonce(z_img, z_txt, mask=None, temperature=0.07):
# #     if mask is not None:
# #         mask = mask.float().view(-1)
# #         valid = mask > 0
# #         if valid.sum() <= 1:
# #             return z_img.new_tensor(0.0)
# #         z_img = z_img[valid]
# #         z_txt = z_txt[valid]
# #         weights = mask[valid].clamp(0, 1)
# #     else:
# #         if z_img.size(0) <= 1:
# #             return z_img.new_tensor(0.0)
# #         weights = None

# #     z_img = l2n(z_img.float(), dim=-1)
# #     z_txt = l2n(z_txt.float(), dim=-1)
# #     logits = z_img @ z_txt.t() / temperature
# #     labels = torch.arange(logits.size(0), device=logits.device)

# #     if weights is None:
# #         return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

# #     loss_i = F.cross_entropy(logits, labels, reduction='none')
# #     loss_t = F.cross_entropy(logits.t(), labels, reduction='none')
# #     weights = weights / weights.mean().clamp_min(1e-6)
# #     return 0.5 * ((loss_i * weights).mean() + (loss_t * weights).mean())


# # def safe_get_cfg(cfg, key, default=None):
# #     return cfg[key] if key in cfg else default


# # # ============================================================
# # # Module A: Foundation RGB Encoder
# # # ============================================================

# # class TinyDenseRGBEncoder(nn.Module):
# #     """Fallback dense RGB encoder.

# #     This is intentionally lightweight and dependency-free. If timm is available,
# #     GaitOracle can switch to a foundation backbone via TimmDenseRGBEncoder.
# #     """

# #     def __init__(self, out_channels=512, in_channels=3):
# #         super().__init__()
# #         c1 = max(out_channels // 8, 32)
# #         c2 = max(out_channels // 4, 64)
# #         c3 = max(out_channels // 2, 128)
# #         self.net = nn.Sequential(
# #             nn.Conv2d(in_channels, c1, 3, stride=1, padding=1, bias=False),
# #             nn.BatchNorm2d(c1),
# #             nn.GELU(),
# #             nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.GELU(),
# #             nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False),
# #             nn.BatchNorm2d(c3),
# #             nn.GELU(),
# #             nn.Conv2d(c3, out_channels, 3, stride=1, padding=1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.GELU(),
# #         )
# #         self.out_channels = out_channels

# #     def forward(self, x):
# #         # x: [B, T, 3, H, W]
# #         B, T, C, H, W = x.shape
# #         y = self.net(x.reshape(B * T, C, H, W))
# #         _, C2, H2, W2 = y.shape
# #         return y.reshape(B, T, C2, H2, W2)


# # class TimmDenseRGBEncoder(nn.Module):
# #     """Optional dense foundation encoder via timm ViT/DINOv2 models.

# #     Offline checkpoint loading is supported through checkpoint_path.

# #     This version avoids two common offline DINOv2 issues:
# #       1) features_only=True wraps the ViT as FeatureGetterNet, whose state_dict
# #          keys are prefixed with "model.". We adapt the checkpoint keys manually.
# #       2) DINOv2-B/14 timm defaults to 518x518 inputs. Gait images are usually
# #          much smaller, e.g. 128x88. We create the ViT with a configurable smaller
# #          img_size and resize positional embeddings accordingly.
# #     """

# #     def __init__(self, model_name='vit_small_patch14_dinov2.lvd142m', out_channels=512,
# #                  pretrained=True, freeze=True, checkpoint_path=None,
# #                  img_size=(126, 84), chunk_size=64):
# #         super().__init__()
# #         try:
# #             import timm
# #         except Exception as exc:
# #             raise ImportError("timm is required for TimmDenseRGBEncoder. Install timm or use rgb_backbone.type='tiny'.") from exc

# #         self.model_name = model_name
# #         self.checkpoint_path = checkpoint_path
# #         self.chunk_size = int(chunk_size) if chunk_size is not None else 0

# #         if img_size is None:
# #             self.img_size = None
# #         elif isinstance(img_size, int):
# #             self.img_size = (int(img_size), int(img_size))
# #         elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
# #             self.img_size = (int(img_size[0]), int(img_size[1]))
# #         else:
# #             raise ValueError(f"img_size should be int or [H, W], got {img_size}")

# #         has_local_ckpt = checkpoint_path is not None and str(checkpoint_path).lower() not in ('', 'none', 'null')

# #         create_kwargs = dict(
# #             pretrained=(bool(pretrained) and not has_local_ckpt),
# #             features_only=True,
# #         )
# #         if self.img_size is not None:
# #             create_kwargs['img_size'] = self.img_size

# #         # Do not pass checkpoint_path into timm.create_model when features_only=True.
# #         # We load it manually below to handle FeatureGetterNet's "model." prefix and
# #         # position embedding resizing.
# #         self.backbone = timm.create_model(model_name, **create_kwargs)

# #         if has_local_ckpt:
# #             if not os.path.exists(checkpoint_path):
# #                 raise FileNotFoundError(
# #                     f"DINOv2 checkpoint_path does not exist: {checkpoint_path}. "
# #                     "Please download model.safetensors manually and update rgb_backbone.checkpoint_path."
# #                 )
# #             self._load_local_checkpoint(checkpoint_path)

# #         chs = self.backbone.feature_info.channels()
# #         in_c = chs[-1]
# #         self.proj = nn.Sequential(
# #             nn.Conv2d(in_c, out_channels, 1, bias=False),
# #             nn.BatchNorm2d(out_channels),
# #             nn.GELU(),
# #         )
# #         if freeze:
# #             for p in self.backbone.parameters():
# #                 p.requires_grad = False
# #         self.out_channels = out_channels

# #     def _read_checkpoint(self, checkpoint_path):
# #         if str(checkpoint_path).endswith('.safetensors'):
# #             try:
# #                 from safetensors.torch import load_file
# #             except Exception as exc:
# #                 raise ImportError(
# #                     "Loading .safetensors requires safetensors. "
# #                     "Install it with: pip install safetensors"
# #                 ) from exc
# #             state = load_file(checkpoint_path, device='cpu')
# #         else:
# #             state = torch.load(checkpoint_path, map_location='cpu')

# #         if isinstance(state, dict):
# #             for key in ('state_dict', 'model', 'model_state_dict', 'teacher', 'student'):
# #                 if key in state and isinstance(state[key], dict):
# #                     maybe = state[key]
# #                     if any(torch.is_tensor(v) for v in maybe.values()):
# #                         state = maybe
# #                         break

# #         if not isinstance(state, dict):
# #             raise RuntimeError(f"Unsupported checkpoint format: {type(state)}")

# #         clean = {}
# #         for k, v in state.items():
# #             if not torch.is_tensor(v):
# #                 continue
# #             name = str(k)
# #             for pfx in ('module.', 'backbone.', 'encoder.'):
# #                 if name.startswith(pfx):
# #                     name = name[len(pfx):]
# #             clean[name] = v
# #         return clean

# #     def _get_patch_grid_size(self):
# #         # FeatureGetterNet exposes the original ViT as self.backbone.model.
# #         m = getattr(self.backbone, 'model', self.backbone)
# #         pe = getattr(m, 'patch_embed', None)
# #         grid = getattr(pe, 'grid_size', None)
# #         if grid is not None:
# #             return (int(grid[0]), int(grid[1]))
# #         if self.img_size is not None:
# #             patch_size = getattr(pe, 'patch_size', (14, 14))
# #             if isinstance(patch_size, int):
# #                 patch_size = (patch_size, patch_size)
# #             return (int(self.img_size[0] // patch_size[0]), int(self.img_size[1] // patch_size[1]))
# #         return None

# #     def _resize_pos_embed(self, pos_embed, target):
# #         if tuple(pos_embed.shape) == tuple(target.shape):
# #             return pos_embed
# #         if pos_embed.dim() != 3 or target.dim() != 3 or pos_embed.shape[0] != 1 or target.shape[0] != 1:
# #             return None
# #         if pos_embed.shape[-1] != target.shape[-1]:
# #             return None

# #         old_tokens = pos_embed.shape[1]
# #         new_tokens = target.shape[1]
# #         if old_tokens <= 1 or new_tokens <= 1:
# #             return None

# #         cls_pos = pos_embed[:, :1]
# #         patch_pos = pos_embed[:, 1:]
# #         old_grid = int(round(math.sqrt(patch_pos.shape[1])))
# #         if old_grid * old_grid != patch_pos.shape[1]:
# #             return None

# #         new_grid = self._get_patch_grid_size()
# #         if new_grid is None:
# #             new_patch_tokens = new_tokens - 1
# #             new_side = int(round(math.sqrt(new_patch_tokens)))
# #             if new_side * new_side != new_patch_tokens:
# #                 return None
# #             new_grid = (new_side, new_side)

# #         new_h, new_w = int(new_grid[0]), int(new_grid[1])
# #         if new_h * new_w != new_tokens - 1:
# #             return None

# #         patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
# #         patch_pos = F.interpolate(patch_pos.float(), size=(new_h, new_w), mode='bicubic', align_corners=False)
# #         patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1).to(dtype=pos_embed.dtype)
# #         return torch.cat([cls_pos, patch_pos], dim=1)

# #     def _load_local_checkpoint(self, checkpoint_path):
# #         raw_state = self._read_checkpoint(checkpoint_path)
# #         target_state = self.backbone.state_dict()

# #         # Candidate 1: as-is. Candidate 2: add "model." prefix for FeatureGetterNet.
# #         prefixed_state = {}
# #         for k, v in raw_state.items():
# #             prefixed_state[k if k.startswith('model.') else 'model.' + k] = v

# #         def adapt_and_score(sd):
# #             adapted = {}
# #             matched = 0
# #             resized_pos = 0
# #             for k, v in sd.items():
# #                 if k not in target_state:
# #                     adapted[k] = v
# #                     continue
# #                 tgt = target_state[k]
# #                 if hasattr(v, 'shape') and tuple(v.shape) == tuple(tgt.shape):
# #                     adapted[k] = v
# #                     matched += 1
# #                 elif k.endswith('pos_embed'):
# #                     resized = self._resize_pos_embed(v, tgt)
# #                     if resized is not None and tuple(resized.shape) == tuple(tgt.shape):
# #                         adapted[k] = resized
# #                         matched += 1
# #                         resized_pos += 1
# #                 # Other shape-mismatched tensors are dropped.
# #             return adapted, matched, resized_pos

# #         candidates = [raw_state, prefixed_state]
# #         adapted_candidates = [adapt_and_score(c) for c in candidates]
# #         best_state, matched, resized_pos = max(adapted_candidates, key=lambda x: x[1])

# #         if matched == 0:
# #             sample_ckpt = list(raw_state.keys())[:10]
# #             sample_model = list(target_state.keys())[:10]
# #             raise RuntimeError(
# #                 "No compatible keys found when loading DINOv2 checkpoint.\n"
# #                 f"checkpoint_path={checkpoint_path}\n"
# #                 f"sample checkpoint keys={sample_ckpt}\n"
# #                 f"sample model keys={sample_model}"
# #             )

# #         incompatible = self.backbone.load_state_dict(best_state, strict=False)
# #         missing = getattr(incompatible, 'missing_keys', [])
# #         unexpected = getattr(incompatible, 'unexpected_keys', [])

# #         print(
# #             f"[TimmDenseRGBEncoder] Loaded local checkpoint: {checkpoint_path}; "
# #             f"matched_keys={matched}; resized_pos_embed={resized_pos}; "
# #             f"missing_keys={len(missing)}; unexpected_keys={len(unexpected)}; "
# #             f"img_size={self.img_size}; chunk_size={self.chunk_size}"
# #         )

# #     def _forward_flat(self, x_flat):
# #         if self.img_size is not None and tuple(x_flat.shape[-2:]) != tuple(self.img_size):
# #             x_flat = F.interpolate(x_flat, size=self.img_size, mode='bilinear', align_corners=False)
# #         return self.backbone(x_flat)[-1]

# #     def forward(self, x):
# #         B, T, C, H, W = x.shape
# #         x_flat = x.reshape(B * T, C, H, W)

# #         if self.chunk_size and self.chunk_size > 0 and x_flat.size(0) > self.chunk_size:
# #             ys = []
# #             for s in range(0, x_flat.size(0), self.chunk_size):
# #                 ys.append(self._forward_flat(x_flat[s:s + self.chunk_size]))
# #             y = torch.cat(ys, dim=0)
# #         else:
# #             y = self._forward_flat(x_flat)

# #         y = self.proj(y)
# #         _, C2, H2, W2 = y.shape
# #         return y.reshape(B, T, C2, H2, W2)


# # class PositiveTextGate(nn.Module):
# #     """Fixed gait-positive prompt bank -> channel and spatial motion gate.

# #     Positive text tells the model what to preserve:
# #       arm/leg swing, body boundary motion, stride rhythm, torso stability.
# #     """

# #     def __init__(self, text_dim, channels, prompt_feat_path=None, num_prompts=4, trainable_prompts=False):
# #         super().__init__()
# #         self.channels = channels
# #         self.text_dim = text_dim

# #         if prompt_feat_path is not None and os.path.exists(prompt_feat_path):
# #             arr = np.load(prompt_feat_path).astype('float32')
# #             if arr.ndim == 1:
# #                 arr = arr[None, :]
# #             prompt = torch.from_numpy(arr)
# #             text_dim = prompt.shape[-1]
# #             self.text_dim = text_dim
# #         else:
# #             prompt = torch.randn(num_prompts, text_dim) * 0.02

# #         self.prompt_feat = nn.Parameter(prompt, requires_grad=trainable_prompts)
# #         self.text_proj = nn.Sequential(
# #             nn.Linear(text_dim, channels),
# #             nn.LayerNorm(channels),
# #             nn.GELU(),
# #         )
# #         self.channel_gate = nn.Sequential(
# #             nn.Linear(channels, channels),
# #             nn.Sigmoid(),
# #         )
# #         self.spatial_gate = nn.Sequential(
# #             nn.Conv2d(channels, channels // 4, 1, bias=False),
# #             nn.BatchNorm2d(channels // 4),
# #             nn.GELU(),
# #             nn.Conv2d(channels // 4, 1, 1),
# #             nn.Sigmoid(),
# #         )

# #     def forward(self, feat_map):
# #         # feat_map: [B, T, C, H, W]
# #         B, T, C, H, W = feat_map.shape
# #         pos = self.text_proj(self.prompt_feat.to(feat_map.device))  # [K, C]
# #         pos = pos.mean(0)  # [C]
# #         ch_gate = self.channel_gate(pos).view(1, 1, C, 1, 1)

# #         x = feat_map * (1.0 + ch_gate)
# #         gate = self.spatial_gate(x.reshape(B * T, C, H, W)).reshape(B, T, 1, H, W)

# #         # Residual gate avoids destroying useful RGB features at early training.
# #         guided = feat_map * (1.0 + gate) * (1.0 + ch_gate)
# #         return guided, gate


# # class NuisanceDetachHead(nn.Module):
# #     """Qwen nuisance captions -> GRL/NBDetach.

# #     Negative text tells the model what to suppress:
# #       clothing, carrying/bag, background, occlusion.
# #     """

# #     def __init__(self, in_dim, text_dim, sub_dim=256, factors=None, grl_lambda=0.05, temperature=0.07):
# #         super().__init__()
# #         self.factors = factors or ['clothing', 'carrying', 'background', 'occlusion']
# #         self.grl_lambda = grl_lambda
# #         self.temperature = temperature

# #         self.img_heads = nn.ModuleDict({
# #             k: nn.Sequential(
# #                 nn.Linear(in_dim, sub_dim),
# #                 nn.BatchNorm1d(sub_dim),
# #                 nn.ReLU(inplace=True),
# #                 nn.Linear(sub_dim, sub_dim),
# #             )
# #             for k in self.factors
# #         })
# #         self.txt_heads = nn.ModuleDict({
# #             k: nn.Sequential(
# #                 nn.Linear(text_dim, sub_dim),
# #                 nn.LayerNorm(sub_dim),
# #             )
# #             for k in self.factors
# #         })

# #     def forward(self, rgb_global, text_feats: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None):
# #         out = {}
# #         losses = {}
# #         rgb_rev = grad_reverse(rgb_global, self.grl_lambda)

# #         for k in self.factors:
# #             if k not in text_feats or text_feats[k] is None:
# #                 continue
# #             txt = text_feats[k]
# #             if txt.dim() == 3:
# #                 txt = txt.mean(1)
# #             txt = txt.float()

# #             z_img = self.img_heads[k](rgb_rev.float())
# #             z_txt = self.txt_heads[k](txt)
# #             z_img = l2n(z_img, dim=-1)
# #             z_txt = l2n(z_txt, dim=-1)

# #             mask = None
# #             if masks is not None and k in masks and masks[k] is not None:
# #                 mask = masks[k]
# #                 if mask.dim() > 1:
# #                     mask = mask.float().view(mask.size(0), -1).mean(-1)

# #             losses[k] = masked_symmetric_infonce(z_img, z_txt, mask=mask, temperature=self.temperature)
# #             out[k] = {'img': z_img, 'txt': z_txt, 'mask': mask}

# #         return out, losses


# # # ============================================================
# # # Module C: Geometry / Silhouette Anchor
# # # ============================================================

# # class SilhouetteAnchorEncoder(nn.Module):
# #     def __init__(self, out_channels=512):
# #         super().__init__()
# #         self.net = TinyDenseRGBEncoder(out_channels=out_channels, in_channels=1)

# #     def forward(self, sil):
# #         # sil: [B, T, H, W] or [B,T,1,H,W]
# #         if sil.dim() == 4:
# #             sil = sil.unsqueeze(2)
# #         elif sil.dim() == 5 and sil.shape[2] != 1:
# #             sil = sil[:, :, :1]
# #         return self.net(sil.float())


# # class PointCloudAnchorEncoder(nn.Module):
# #     def __init__(self, out_channels=512, parts_num=16, in_dim=3):
# #         super().__init__()
# #         self.parts_num = parts_num
# #         self.out_channels = out_channels
# #         self.point_mlp = nn.Sequential(
# #             nn.Linear(in_dim, 64),
# #             nn.GELU(),
# #             nn.Linear(64, 128),
# #             nn.GELU(),
# #             nn.Linear(128, out_channels),
# #             nn.GELU(),
# #         )
# #         self.norm = nn.LayerNorm(out_channels)

# #     def forward(self, pc):
# #         # pc: [B, T, N, C]
# #         B, T, N, C = pc.shape
# #         pc = pc[..., :C].float()
# #         xyz = pc[..., :3]
# #         z = xyz[..., 2] if xyz.shape[-1] >= 3 else xyz[..., -1]
# #         z_min = z.min(dim=-1, keepdim=True).values
# #         z_max = z.max(dim=-1, keepdim=True).values
# #         z_norm = (z - z_min) / (z_max - z_min).clamp_min(1e-6)

# #         feat = self.point_mlp(pc.reshape(B * T * N, C)).reshape(B, T, N, self.out_channels)
# #         feat = self.norm(feat)

# #         part_feats = []
# #         for p in range(self.parts_num):
# #             lo = p / self.parts_num
# #             hi = (p + 1) / self.parts_num
# #             mask = ((z_norm >= lo) & (z_norm < hi)).float().unsqueeze(-1)
# #             denom = mask.sum(dim=2).clamp_min(1.0)
# #             pooled = (feat * mask).sum(dim=2) / denom  # [B,T,C]
# #             part_feats.append(pooled)
# #         part_feats = torch.stack(part_feats, dim=-1)  # [B,T,C,P]
# #         return part_feats


# # # ============================================================
# # # Module D: Cycle-Aware Temporal Encoder
# # # ============================================================

# # class CycleAwareTemporalEncoder(nn.Module):
# #     """Temporal encoder with frame, diff, and frequency branches.

# #     It is not a pure pooling module: it learns dynamic biometric cues through
# #     temporal attention, short-term difference tokens, and FFT magnitude tokens.
# #     """

# #     def __init__(self, channels=512, parts_num=16, heads=4, layers=1, dropout=0.1, future_pred=True):
# #         super().__init__()
# #         self.channels = channels
# #         self.parts_num = parts_num
# #         self.future_pred = future_pred

# #         enc_layer = nn.TransformerEncoderLayer(
# #             d_model=channels,
# #             nhead=heads,
# #             dim_feedforward=channels * 4,
# #             dropout=dropout,
# #             batch_first=True,
# #             activation='gelu',
# #             norm_first=True,
# #         )
# #         self.temporal = nn.TransformerEncoder(enc_layer, num_layers=layers)
# #         self.diff_proj = nn.Sequential(
# #             nn.Linear(channels, channels),
# #             nn.GELU(),
# #             nn.LayerNorm(channels),
# #         )
# #         self.freq_proj = nn.Sequential(
# #             nn.Linear(channels, channels),
# #             nn.GELU(),
# #             nn.LayerNorm(channels),
# #         )
# #         self.attn = nn.Sequential(
# #             nn.Linear(channels, channels // 4),
# #             nn.Tanh(),
# #             nn.Linear(channels // 4, 1),
# #         )
# #         if future_pred:
# #             self.future_head = nn.Sequential(
# #                 nn.Linear(channels, channels),
# #                 nn.GELU(),
# #                 nn.Linear(channels, channels),
# #             )

# #     def forward(self, part_seq):
# #         # part_seq: [B,T,C,P]
# #         B, T, C, P = part_seq.shape
# #         x = part_seq.permute(0, 3, 1, 2).reshape(B * P, T, C)  # [B*P,T,C]

# #         x_tem = self.temporal(x)

# #         if T > 1:
# #             diff = x[:, 1:] - x[:, :-1]
# #             diff = F.pad(diff, (0, 0, 1, 0))
# #         else:
# #             diff = torch.zeros_like(x)
# #         diff = self.diff_proj(diff)

# #         # FFT branch: magnitude of low-frequency temporal spectrum, broadcast to time tokens.
# #         if T > 1:
# #             freq = torch.fft.rfft(x.float(), dim=1).abs()
# #             freq = freq[:, 1:min(freq.size(1), 4)].mean(1) if freq.size(1) > 1 else freq.mean(1)
# #         else:
# #             freq = x.float().mean(1)
# #         freq = self.freq_proj(freq).unsqueeze(1).expand(-1, T, -1)

# #         y = x_tem + diff + freq
# #         att = torch.softmax(self.attn(y).squeeze(-1), dim=-1)  # [B*P,T]
# #         pooled = (y * att.unsqueeze(-1)).sum(1)  # [B*P,C]
# #         pooled = pooled.reshape(B, P, C).permute(0, 2, 1).contiguous()  # [B,C,P]

# #         future_loss = pooled.new_tensor(0.0)
# #         if self.future_pred and T > 2:
# #             pred = self.future_head(y[:, :-1])
# #             target = x[:, 1:].detach()
# #             future_loss = F.mse_loss(pred, target)

# #         att = att.reshape(B, P, T).permute(0, 2, 1).contiguous()  # [B,T,P]
# #         return pooled, future_loss, att


# # # ============================================================
# # # Module E: Verifier-Guided auxiliary training
# # # ============================================================

# # class RuleBasedVerifier(nn.Module):
# #     """Lightweight verifier losses.

# #     This does not call MLLM online. It uses available weak signals:
# #       - silhouette / RGB motion magnitude for attention sanity;
# #       - nuisance masks for reliability weighting;
# #       - same batch hard severity can be handled by triplet sampler/loss.
# #     """

# #     def __init__(self, attention_weight=0.0):
# #         super().__init__()
# #         self.attention_weight = attention_weight

# #     def forward(self, gait_attn, rgb_seq=None, sil_seq=None):
# #         if self.attention_weight <= 0 or gait_attn is None:
# #             return gait_attn.new_tensor(0.0)

# #         # gait_attn: [B,T,1,Hf,Wf]
# #         B, T, _, Hf, Wf = gait_attn.shape
# #         target = None

# #         if sil_seq is not None:
# #             if sil_seq.dim() == 4:
# #                 sil = sil_seq.unsqueeze(2)
# #             else:
# #                 sil = sil_seq[:, :, :1]
# #             target = F.interpolate(
# #                 sil.reshape(B * T, 1, sil.shape[-2], sil.shape[-1]).float(),
# #                 size=(Hf, Wf),
# #                 mode='bilinear',
# #                 align_corners=False,
# #             ).reshape(B, T, 1, Hf, Wf)
# #             target = (target > 0.05).float()

# #         elif rgb_seq is not None and rgb_seq.dim() == 5 and T > 1:
# #             gray = rgb_seq.float().mean(2, keepdim=True)
# #             diff = torch.zeros_like(gray)
# #             diff[:, 1:] = (gray[:, 1:] - gray[:, :-1]).abs()
# #             target = F.interpolate(
# #                 diff.reshape(B * T, 1, diff.shape[-2], diff.shape[-1]),
# #                 size=(Hf, Wf),
# #                 mode='bilinear',
# #                 align_corners=False,
# #             ).reshape(B, T, 1, Hf, Wf)
# #             target = target / target.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6)

# #         if target is None:
# #             return gait_attn.new_tensor(0.0)

# #         # return F.binary_cross_entropy(gait_attn.clamp(1e-6, 1 - 1e-6), target.float()) * self.attention_weight
# #         with torch.cuda.amp.autocast(enabled=False):
# #             loss = F.binary_cross_entropy(
# #             gait_attn.float().clamp(1e-6, 1 - 1e-6),
# #             target.float()
# #             )
# #         return loss * self.attention_weight



# # # ============================================================
# # # Fusion
# # # ============================================================

# # class ModalityFusion(nn.Module):
# #     def __init__(self, channels=512, parts_num=16):
# #         super().__init__()
# #         self.gate = nn.Sequential(
# #             nn.Linear(channels * 2, channels),
# #             nn.GELU(),
# #             nn.Linear(channels, 2),
# #         )
# #         self.out = nn.Sequential(
# #             nn.Conv1d(channels, channels, 1, bias=False),
# #             nn.BatchNorm1d(channels),
# #             nn.GELU(),
# #         )

# #     def forward(self, rgb_feat, anchor_feat=None):
# #         # rgb_feat/anchor_feat: [B,C,P]
# #         if anchor_feat is None:
# #             return self.out(rgb_feat), None
# #         g_rgb = rgb_feat.mean(-1)
# #         g_anc = anchor_feat.mean(-1)
# #         w = torch.softmax(self.gate(torch.cat([g_rgb, g_anc], dim=-1)), dim=-1)  # [B,2]
# #         fused = rgb_feat * w[:, 0].view(-1, 1, 1) + anchor_feat * w[:, 1].view(-1, 1, 1)
# #         return self.out(fused), w


# # # ============================================================
# # # Main OpenGait model
# # # ============================================================

# # class GaitOracle(BaseModel):
# #     """GaitOracle: MLLM-supervised, geometry-anchored dynamic biometric gait framework.

# #     Five modules:
# #       A. Foundation RGB Encoder
# #       B. Positive-Negative Text Supervisor
# #       C. Geometry / Silhouette Anchor
# #       D. Cycle-Aware Temporal Encoder
# #       E. Verifier-Guided Training Loop

# #     Inference uses RGB + optional Sil/PC only. Qwen nuisance captions and text
# #     features are training-only supervision signals.
# #     """

# #     def build_network(self, model_cfg):
# #         self.model_cfg = model_cfg
# #         C = model_cfg.get('channels', 512)
# #         P = model_cfg.get('parts_num', model_cfg.get('bin_num', [16])[0] if isinstance(model_cfg.get('bin_num', [16]), list) else model_cfg.get('bin_num', 16))
# #         self.channels = C
# #         self.parts_num = P

# #         # Input indices after data_in_use.
# #         self.rgb_index = model_cfg.get('rgb_index', 0)
# #         self.sil_index = model_cfg.get('sil_index', None)
# #         self.pc_index = model_cfg.get('pc_index', None)
# #         self.nuisance_indices = model_cfg.get('nuisance_indices', {})
# #         self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})

# #         # Module A
# #         rgb_backbone_cfg = model_cfg.get('rgb_backbone', {'type': 'tiny'})
# #         if rgb_backbone_cfg.get('type', 'tiny').lower() == 'timm':
# #             self.rgb_encoder = TimmDenseRGBEncoder(
# #                 model_name=rgb_backbone_cfg.get('model_name', 'vit_small_patch14_dinov2.lvd142m'),
# #                 out_channels=C,
# #                 pretrained=rgb_backbone_cfg.get('pretrained', True),
# #                 freeze=rgb_backbone_cfg.get('freeze', True),
# #                 checkpoint_path=rgb_backbone_cfg.get('checkpoint_path', None),
# #                 img_size=rgb_backbone_cfg.get('img_size', (224, 112)),
# #                 chunk_size=rgb_backbone_cfg.get('chunk_size', 16),
# #             )
# #         else:
# #             self.rgb_encoder = TinyDenseRGBEncoder(out_channels=C, in_channels=3)

# #         # Module B positive branch
# #         pos_cfg = model_cfg.get('positive_text', {})
# #         self.positive_gate = PositiveTextGate(
# #             text_dim=pos_cfg.get('text_dim', model_cfg.get('text_dim', 3584)),
# #             channels=C,
# #             prompt_feat_path=pos_cfg.get('prompt_feat_path', None),
# #             num_prompts=pos_cfg.get('num_prompts', 4),
# #             trainable_prompts=pos_cfg.get('trainable', False),
# #         )

# #         # Module C
# #         self.use_sil_anchor = self.sil_index is not None
# #         self.use_pc_anchor = self.pc_index is not None
# #         self.sil_encoder = SilhouetteAnchorEncoder(out_channels=C) if self.use_sil_anchor else None
# #         self.pc_encoder = PointCloudAnchorEncoder(out_channels=C, parts_num=P, in_dim=model_cfg.get('pc_in_dim', 3)) if self.use_pc_anchor else None

# #         self.HPP = HorizontalPoolingPyramid(bin_num=[P])

# #         # Module D
# #         temporal_cfg = model_cfg.get('cycle_temporal', {})
# #         self.rgb_temporal = CycleAwareTemporalEncoder(
# #             channels=C,
# #             parts_num=P,
# #             heads=temporal_cfg.get('heads', 4),
# #             layers=temporal_cfg.get('layers', 1),
# #             dropout=temporal_cfg.get('dropout', 0.1),
# #             future_pred=temporal_cfg.get('future_pred', True),
# #         )
# #         self.anchor_temporal = CycleAwareTemporalEncoder(
# #             channels=C,
# #             parts_num=P,
# #             heads=temporal_cfg.get('heads', 4),
# #             layers=temporal_cfg.get('layers', 1),
# #             dropout=temporal_cfg.get('dropout', 0.1),
# #             future_pred=False,
# #         )

# #         # Fusion
# #         self.fusion = ModalityFusion(channels=C, parts_num=P)

# #         # Identity heads
# #         self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
# #         self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

# #         # Module B negative branch
# #         nuis_cfg = model_cfg.get('nuisance_detach', {})
# #         self.use_nuisance = nuis_cfg.get('enabled', False)
# #         if self.use_nuisance:
# #             self.nuisance_detach = NuisanceDetachHead(
# #                 in_dim=C,
# #                 text_dim=nuis_cfg.get('text_dim', model_cfg.get('text_dim', 3584)),
# #                 sub_dim=nuis_cfg.get('sub_dim', 256),
# #                 factors=nuis_cfg.get('factors', ['clothing', 'carrying', 'background', 'occlusion']),
# #                 grl_lambda=nuis_cfg.get('grl_lambda', 0.05),
# #                 temperature=nuis_cfg.get('temperature', 0.07),
# #             )
# #             self.nuisance_loss_weight = nuis_cfg.get('loss_weight', 0.05)
# #         else:
# #             self.nuisance_detach = None
# #             self.nuisance_loss_weight = 0.0

# #         # Module E
# #         ver_cfg = model_cfg.get('verifier', {})
# #         self.verifier = RuleBasedVerifier(attention_weight=ver_cfg.get('attention_weight', 0.0))

# #         # Auxiliary weights
# #         self.rgb_anchor_weight = model_cfg.get('rgb_anchor_weight', 0.2)
# #         self.future_weight = model_cfg.get('future_weight', 0.02)

# #     def _to_rgb(self, x):
# #         if x.dim() == 4:
# #             # [B,T,H,W] -> [B,T,3,H,W]
# #             x = x.unsqueeze(2).repeat(1, 1, 3, 1, 1)
# #         elif x.dim() == 5:
# #             if x.shape[2] == 1:
# #                 x = x.repeat(1, 1, 3, 1, 1)
# #             elif x.shape[-1] == 3 and x.shape[2] != 3:
# #                 x = x.permute(0, 1, 4, 2, 3)
# #         return x.float()

# #     def _to_text_feat(self, x):
# #         # OpenGait fixed sampler usually gives [B,T,D]; reduce over sampled frames.
# #         if x is None:
# #             return None
# #         x = x.float()
# #         if x.dim() == 3:
# #             return x.mean(1)
# #         if x.dim() == 2:
# #             return x
# #         return x.view(x.size(0), -1)

# #     def _get_optional(self, ipts, idx):
# #         if idx is None:
# #             return None
# #         if isinstance(idx, str):
# #             idx = int(idx)
# #         if idx < 0 or idx >= len(ipts):
# #             return None
# #         return ipts[idx]

# #     def _dense_to_part_seq(self, feat_map):
# #         # feat_map [B,T,C,H,W] -> part_seq [B,T,C,P]
# #         B, T, C, H, W = feat_map.shape
# #         feat = feat_map.reshape(B * T, C, H, W)
# #         part = self.HPP(feat)  # [B*T,C,P]
# #         return part.reshape(B, T, C, -1)

# #     def forward(self, inputs):
# #         ipts, labs, typs, views, seqL = inputs

# #         rgb = self._to_rgb(ipts[self.rgb_index])
# #         sil = self._get_optional(ipts, self.sil_index)
# #         pc = self._get_optional(ipts, self.pc_index)

# #         # Module A: RGB dense feature.
# #         rgb_map = self.rgb_encoder(rgb)

# #         # Module B positive: gait-positive text gate.
# #         rgb_guided, gait_attn = self.positive_gate(rgb_map)

# #         # Module D: RGB cycle-aware temporal encoding.
# #         rgb_part_seq = self._dense_to_part_seq(rgb_guided)
# #         rgb_feat, future_loss, temporal_attn = self.rgb_temporal(rgb_part_seq)

# #         # Module C: anchor branch.
# #         anchor_feat = None
# #         sil_for_verifier = None
# #         if pc is not None and self.pc_encoder is not None:
# #             pc_part_seq = self.pc_encoder(pc.float())  # [B,T,C,P]
# #             anchor_feat, _, _ = self.anchor_temporal(pc_part_seq)
# #         elif sil is not None and self.sil_encoder is not None:
# #             sil_for_verifier = sil
# #             sil_map = self.sil_encoder(sil.float())
# #             sil_part_seq = self._dense_to_part_seq(sil_map)
# #             anchor_feat, _, _ = self.anchor_temporal(sil_part_seq)

# #         # Fusion and ID heads.
# #         fused_feat, modality_gate = self.fusion(rgb_feat, anchor_feat)
# #         embed_1 = self.FCs(fused_feat)
# #         embed_2, logits = self.BNNecks(embed_1)

# #         # Training losses.
# #         training_feat = {
# #             'triplet': {'embeddings': embed_1, 'labels': labs},
# #             'softmax': {'logits': logits, 'labels': labs},
# #         }

# #         # RGB-anchor consistency.
# #         if anchor_feat is not None and self.rgb_anchor_weight > 0:
# #             rgb_g = rgb_feat.mean(-1)
# #             anc_g = anchor_feat.mean(-1)
# #             training_feat['loss_rgb_anchor'] = symmetric_infonce(rgb_g, anc_g) * self.rgb_anchor_weight

# #         # Future latent prediction.
# #         if self.future_weight > 0:
# #             training_feat['loss_future_pred'] = future_loss * self.future_weight

# #         # Nuisance detach.
# #         nuisance_out = {}
# #         if self.use_nuisance:
# #             text_feats = {}
# #             masks = {}
# #             for k, idx in self.nuisance_indices.items():
# #                 text_feats[k] = self._to_text_feat(self._get_optional(ipts, idx))
# #             for k, idx in self.nuisance_mask_indices.items():
# #                 masks[k] = self._to_text_feat(self._get_optional(ipts, idx))
# #             nuisance_out, nuisance_losses = self.nuisance_detach(rgb_feat.mean(-1), text_feats, masks)
# #             total_nuis = None
# #             for _, v in nuisance_losses.items():
# #                 total_nuis = v if total_nuis is None else total_nuis + v
# #             if total_nuis is not None:
# #                 training_feat['loss_nuisance_detach'] = total_nuis * self.nuisance_loss_weight

# #         # Rule-based verifier loss.
# #         attn_loss = self.verifier(gait_attn, rgb_seq=rgb, sil_seq=sil_for_verifier)
# #         if attn_loss.requires_grad or float(attn_loss.detach().cpu()) != 0.0:
# #             training_feat['loss_attention_verifier'] = attn_loss

# #         visual_summary = {
# #             'image/rgb': rearrange(rgb[:, :min(rgb.size(1), 4)], 'b t c h w -> (b t) c h w'),
# #             'image/gait_attention': rearrange(gait_attn[:, :min(gait_attn.size(1), 4)], 'b t c h w -> (b t) c h w'),
# #         }

# #         retval = {
# #             'training_feat': training_feat,
# #             'visual_summary': visual_summary,
# #             'inference_feat': {
# #                 'embeddings': embed_1
# #             }
# #         }
# #         return retval
    


