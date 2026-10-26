import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..backbones.resnet import ResNet9
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks


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


class SequenceTextPool(nn.Module):
    def __init__(self, text_dim, out_dim, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def _pool_variable(self, x, seqL):
        lens = seqL[0].data.cpu().numpy().tolist()
        if x.dim() == 4:
            x = x.squeeze(2)
        if x.dim() != 3:
            raise ValueError('Text feature should be [B,T,D] or [B,T,1,D].')
        if x.size(0) == len(lens):
            pooled = []
            for i, curr_len in enumerate(lens):
                pooled.append(x[i, :curr_len].mean(0, keepdim=True))
            return torch.cat(pooled, dim=0)
        if x.size(0) == 1:
            start = 0
            pooled = []
            for curr_len in lens:
                pooled.append(x[:, start:start + curr_len].mean(1))
                start += curr_len
            return torch.cat(pooled, dim=0)
        return x.mean(1)

    def forward(self, x, seqL=None):
        if seqL is None:
            if x.dim() == 4:
                x = x.squeeze(2)
            x = x.mean(1)
        else:
            x = self._pool_variable(x, seqL)
        return self.proj(x.float())


class LearnableMotionPrompt(nn.Module):
    def __init__(self, out_dim, num_prompts=4):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(num_prompts, out_dim) * 0.02)

    def forward(self, batch_size, device):
        # Use the mean of task-level motion prompts when no Qwen text feature is loaded.
        return self.prompt.mean(0, keepdim=True).expand(batch_size, -1).to(device)


class TemporalPartAttention(nn.Module):
    """Transformer + attention pooling over the temporal dimension for each body part."""

    def __init__(self, channels, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.score = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
        )
        self.norm = nn.LayerNorm(channels)

    def _pool_fixed(self, x):
        # x: [B, C, T, P]
        b, c, t, p = x.shape
        xp = x.permute(0, 3, 2, 1).contiguous().view(b * p, t, c)
        xp = self.encoder(xp)
        attn = torch.softmax(self.score(xp).squeeze(-1), dim=-1)
        pooled = (xp * attn.unsqueeze(-1)).sum(1)
        pooled = self.norm(pooled).view(b, p, c).permute(0, 2, 1).contiguous()
        return pooled

    def forward(self, x, seqL=None):
        # x: [B, C, T, P]
        if seqL is None:
            return self._pool_fixed(x)
        lens = seqL[0].data.cpu().numpy().tolist()
        if x.size(0) == len(lens):
            return self._pool_fixed(x)
        # OpenGait all_ordered evaluation often concatenates sequences on the temporal axis.
        if x.size(0) != 1:
            return self._pool_fixed(x)
        start = 0
        outs = []
        for curr_len in lens:
            outs.append(self._pool_fixed(x[:, :, start:start + curr_len, :]))
            start += curr_len
        return torch.cat(outs, dim=0)


class MapToTemporalParts(nn.Module):
    def __init__(self, bin_num, channels, temporal_heads=4, temporal_layers=1, dropout=0.1):
        super().__init__()
        self.hpp = HorizontalPoolingPyramid(bin_num=bin_num)
        self.temporal = TemporalPartAttention(channels, temporal_heads, temporal_layers, dropout)

    @property
    def parts_num(self):
        return sum(self.hpp.bin_num)

    def forward(self, fmap, seqL=None):
        # fmap: [B, C, T, H, W]
        b, c, t, h, w = fmap.shape
        part = self.hpp(fmap.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        part = part.view(b, t, c, -1).permute(0, 2, 1, 3).contiguous()  # [B,C,T,P]
        return self.temporal(part, seqL=seqL)


class TextGuidedRGBDenoiser(nn.Module):
    """
    RGB branch that suppresses clothing appearance by combining:
      1. RGB spatial feature;
      2. frame-difference motion feature;
      3. optional silhouette structural feature;
      4. text-conditioned FiLM gate from Qwen2.5-VL motion descriptions.
    """

    def __init__(self, channels, text_dim, dropout=0.1):
        super().__init__()
        self.rgb_backbone = SetBlockWrapper(ResNet9('BasicBlock', in_channel=3, channels=[64, 128, 256, channels], layers=[1, 1, 1, 1], strides=[1, 2, 2, 1], maxpool=False))
        self.motion_backbone = SetBlockWrapper(ResNet9('BasicBlock', in_channel=3, channels=[64, 128, 256, channels], layers=[1, 1, 1, 1], strides=[1, 2, 2, 1], maxpool=False))
        self.sil_adapter = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, channels * 2),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.GELU(),
            nn.Dropout3d(dropout),
        )
        self.rgb_residual_gate = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, channels),
            nn.Sigmoid(),
        )

    @staticmethod
    def _motion_delta(rgb):
        # rgb: [B,3,T,H,W]
        delta = rgb[:, :, 1:] - rgb[:, :, :-1]
        first = torch.zeros_like(rgb[:, :, :1])
        return torch.cat([first, delta.abs()], dim=2)

    def forward(self, rgb, text_feat, sil_map=None):
        # rgb: [B,3,T,H,W]
        rgb_map = self.rgb_backbone(rgb)
        mot_map = self.motion_backbone(self._motion_delta(rgb))

        gamma_beta = self.film(text_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        rgb_text = rgb_map * (1.0 + torch.tanh(gamma)) + beta
        rgb_keep = self.rgb_residual_gate(text_feat).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        rgb_text = rgb_text * rgb_keep

        if sil_map is None:
            sil_map = torch.zeros_like(rgb_map)
        else:
            if sil_map.shape[-2:] != rgb_map.shape[-2:]:
                sil_map = F.interpolate(
                    sil_map.permute(0, 2, 1, 3, 4).reshape(-1, sil_map.size(1), sil_map.size(-2), sil_map.size(-1)),
                    size=rgb_map.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).view(sil_map.size(0), sil_map.size(2), sil_map.size(1), rgb_map.size(-2), rgb_map.size(-1)).permute(0, 2, 1, 3, 4).contiguous()
            sil_map = self.sil_adapter(sil_map)

        return self.fuse(torch.cat([rgb_text, mot_map, sil_map], dim=1))


class SilhouetteEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = SetBlockWrapper(ResNet9('BasicBlock', in_channel=1, channels=[64, 128, 256, channels], layers=[1, 1, 1, 1], strides=[1, 2, 2, 1], maxpool=False))

    def forward(self, sil):
        return self.backbone(sil)


class PointPartEncoder(nn.Module):
    """Simple point-cloud branch that pools point features into height-based body parts."""

    def __init__(self, in_dim, channels, parts_num, temporal_heads=4, temporal_layers=1, dropout=0.1):
        super().__init__()
        self.parts_num = parts_num
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, channels), nn.LayerNorm(channels),
        )
        self.temporal = TemporalPartAttention(channels, temporal_heads, temporal_layers, dropout)

    def forward(self, pc, seqL=None):
        # pc: [B,T,N,C], C is 3 or 4.
        b, t, n, c = pc.shape
        feat = self.point_mlp(pc.float())  # [B,T,N,Cout]
        z = pc[..., 2].float()
        z_min = z.amin(dim=2, keepdim=True)
        z_max = z.amax(dim=2, keepdim=True)
        z_norm = (z - z_min) / (z_max - z_min + 1e-6)
        part_feats = []
        for p in range(self.parts_num):
            lo = float(p) / self.parts_num
            hi = float(p + 1) / self.parts_num
            mask = (z_norm >= lo) & (z_norm < hi if p < self.parts_num - 1 else z_norm <= hi)
            masked = feat.masked_fill(~mask.unsqueeze(-1), -1e4)
            pooled = masked.max(dim=2).values
            empty = ~mask.any(dim=2)
            pooled = pooled.masked_fill(empty.unsqueeze(-1), 0.0)
            part_feats.append(pooled)
        # [B,T,P,C] -> [B,C,T,P]
        part = torch.stack(part_feats, dim=2).permute(0, 3, 1, 2).contiguous()
        return self.temporal(part, seqL=seqL)


class PromptConditionedFusion(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, 1),
        )
        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, feats, text_feat):
        # feats: list of [B,C,P]
        if len(feats) == 1:
            stacked = feats[0].unsqueeze(1)
            gate = torch.ones(stacked.size(0), 1, stacked.size(-1), device=stacked.device, dtype=stacked.dtype)
            return feats[0], gate, stacked
        stacked = torch.stack(feats, dim=1)  # [B,M,C,P]
        b, m, c, p = stacked.shape
        text = text_feat.unsqueeze(1).unsqueeze(-1).expand(b, m, c, p)
        logits = self.gate(torch.cat([stacked, text], dim=2).permute(0, 1, 3, 2)).squeeze(-1)  # [B,M,P]
        gate = torch.softmax(logits, dim=1)
        fused = (stacked * gate.unsqueeze(2)).sum(dim=1)
        fused = self.out(fused)
        return fused, gate, stacked


class ClothAdversary(nn.Module):
    def __init__(self, in_channels, parts_num, num_clothes, lambd=0.1):
        super().__init__()
        self.lambd = lambd
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * parts_num, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, num_clothes),
        )

    def forward(self, embed, clothes_labels):
        x = grad_reverse(embed.flatten(1), self.lambd)
        logits = self.classifier(x)
        return F.cross_entropy(logits, clothes_labels), logits


class MLLMGait(BaseModel):
    """
    OpenGait-compatible implementation of the proposed idea:
      - Qwen2.5-VL motion text guidance for RGB appearance suppression;
      - RGB + PointCloud default fusion, with optional Silhouette branch;
      - enhanced temporal part attention;
      - identity metric learning with optional clothes-adversarial regularization.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        channels = model_cfg.get('channels', 512)
        bin_num = model_cfg.get('bin_num', [16])
        parts_num = sum(bin_num)
        self.channels = channels
        self.parts_num = parts_num

        self.rgb_index = model_cfg.get('rgb_index', None)
        self.sil_index = model_cfg.get('sil_index', None)
        self.pc_index = model_cfg.get('pc_index', None)
        self.text_index = model_cfg.get('text_index', None)
        self.use_rgb = self.rgb_index is not None
        self.use_sil = self.sil_index is not None
        self.use_pc = self.pc_index is not None
        self.use_text_cache = self.text_index is not None

        text_dim = model_cfg.get('text_dim', channels)
        self.text_from_cache = SequenceTextPool(text_dim, channels, model_cfg.get('dropout', 0.1))
        self.prompt = LearnableMotionPrompt(channels)

        self.sil_encoder = SilhouetteEncoder(channels) if self.use_sil else None
        self.rgb_denoiser = TextGuidedRGBDenoiser(channels, channels, model_cfg.get('dropout', 0.1)) if self.use_rgb else None
        self.map_to_parts = MapToTemporalParts(bin_num, channels, model_cfg.get('temporal_heads', 4), model_cfg.get('temporal_layers', 1), model_cfg.get('dropout', 0.1))
        self.pc_encoder = PointPartEncoder(model_cfg.get('pc_in_dim', 3), channels, parts_num, model_cfg.get('temporal_heads', 4), model_cfg.get('temporal_layers', 1), model_cfg.get('dropout', 0.1)) if self.use_pc else None
        self.fusion = PromptConditionedFusion(channels, model_cfg.get('dropout', 0.1))

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.cloth_adv = None
        if model_cfg.get('cloth_adversary', {}).get('enabled', False):
            cfg = model_cfg['cloth_adversary']
            self.cloth_patterns = cfg.get('patterns', {})
            self.cloth_adv = ClothAdversary(model_cfg['SeparateFCs']['out_channels'], parts_num, cfg['num_clothes'], cfg.get('lambda', 0.1))
            self.cloth_loss_weight = cfg.get('loss_weight', 0.1)
        else:
            self.cloth_patterns = {}
            self.cloth_loss_weight = 0.0

    @staticmethod
    def _as_sil(x):
        if x.dim() == 4:
            return x.unsqueeze(1)
        if x.dim() == 5:
            # [B,T,C,H,W] -> [B,C,T,H,W]
            if x.size(2) == 1:
                return rearrange(x, 'b t c h w -> b c t h w')
            return rearrange(x[:, :, :1], 'b t c h w -> b c t h w')
        raise ValueError('Silhouette input should be [B,T,H,W] or [B,T,1,H,W].')

    @staticmethod
    def _as_rgb(x):
        if x.dim() != 5:
            raise ValueError('RGB input should be [B,T,3,H,W] or [B,T,H,W,3].')
        if x.size(2) == 3:
            return rearrange(x, 'b t c h w -> b c t h w')
        if x.size(-1) == 3:
            return rearrange(x, 'b t h w c -> b c t h w')
        if x.size(2) == 1:
            return rearrange(x.repeat(1, 1, 3, 1, 1), 'b t c h w -> b c t h w')
        raise ValueError('Cannot infer RGB channel dimension.')

    def _make_text_feat(self, ipts, seqL, batch_size, device):
        if self.use_text_cache and self.text_index < len(ipts):
            return self.text_from_cache(ipts[self.text_index], seqL=seqL)
        return self.prompt(batch_size, device)

    def _clothes_to_labels(self, typs, device):
        if self.cloth_adv is None:
            return None
        labels = []
        for typ in typs:
            s = str(typ)
            cid = None
            for k, v in self.cloth_patterns.items():
                if str(k) in s:
                    cid = int(v)
                    break
            if cid is None:
                # Stable fallback: common clothing-change datasets encode condition in typ.
                cid = int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % self.cloth_adv.classifier[-1].out_features
            labels.append(cid)
        return torch.tensor(labels, dtype=torch.long, device=device)

    def forward(self, inputs):
        ipts, labs, typs, _, seqL = inputs
        branch_feats = []
        visual_summary = {}

        # infer batch size/device from first selected modality
        first = ipts[0]
        batch_size = first.size(0)
        device = first.device
        text_feat = self._make_text_feat(ipts, seqL, batch_size, device)

        sil_map = None
        if self.use_sil and self.sil_index < len(ipts):
            sil = self._as_sil(ipts[self.sil_index])
            sil_map = self.sil_encoder(sil)
            sil_part = self.map_to_parts(sil_map, seqL=seqL)
            branch_feats.append(sil_part)
            visual_summary['image/sil'] = rearrange(sil[:, :, :min(8, sil.size(2))], 'b c t h w -> (b t) c h w')

        if self.use_rgb and self.rgb_index < len(ipts):
            rgb = self._as_rgb(ipts[self.rgb_index])
            rgb_map = self.rgb_denoiser(rgb, text_feat, sil_map=sil_map)
            rgb_part = self.map_to_parts(rgb_map, seqL=seqL)
            branch_feats.append(rgb_part)
            # show normalized RGB roughly; tensorboard can still render it as diagnostic.
            visual_summary['image/rgb_first'] = rearrange(rgb[:, :, :min(4, rgb.size(2))], 'b c t h w -> (b t) c h w')

        if self.use_pc and self.pc_index < len(ipts):
            pc = ipts[self.pc_index]
            if pc.size(-1) > self.model_cfg.get('pc_in_dim', 3):
                pc = pc[..., :self.model_cfg.get('pc_in_dim', 3)]
            pc_part = self.pc_encoder(pc, seqL=seqL)
            branch_feats.append(pc_part)

        if not branch_feats:
            raise ValueError('No valid modality branch is enabled. Check rgb_index/sil_index/pc_index and data_in_use.')

        fused, modality_gate, branch_stack = self.fusion(branch_feats, text_feat)
        embed_1 = self.FCs(fused)
        embed_2, logits = self.BNNecks(embed_1)

        training_feat = {
            'triplet': {'embeddings': embed_1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
            'xmodal': {'modality_embeddings': branch_stack},
        }
        # Use contrastive text alignment only when sequence-level Qwen features are loaded.
        # If no cache is provided, text_feat is a shared learnable task prompt and should
        # guide FiLM/gating but not act as instance-level positive text.
        if self.use_text_cache:
            training_feat['text_align'] = {'embeddings': embed_1, 'text_embeddings': text_feat}

        if self.cloth_adv is not None and self.training:
            clothes_labels = self._clothes_to_labels(typs, embed_1.device)
            if clothes_labels is not None and clothes_labels.numel() == embed_1.size(0):
                adv_loss, _ = self.cloth_adv(embed_1.detach() + (embed_1 - embed_1.detach()), clothes_labels)
                training_feat['cloth_adv_loss'] = adv_loss * self.cloth_loss_weight

        visual_summary['scalar/modality_gate_mean'] = modality_gate.mean().detach()
        retval = {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {
                'embeddings': embed_1,
            }
        }
        return retval
