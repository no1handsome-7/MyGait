import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateBNNecks, SeparateFCs


class TinyFrameBackbone(nn.Module):
    """A light fallback used when transformers/DINO weights are unavailable.

    It keeps the RealKineGait code importable and debuggable on a CPU box. For real
    experiments set model_cfg.lvm_name_or_path to a local DINOv2/DINOv3 checkpoint.
    """

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
    def __init__(self, name_or_path: Optional[str], out_dim: int, selected_layers: List[int], freeze: bool = True):
        super().__init__()
        self.name_or_path = name_or_path
        self.selected_layers = selected_layers
        self.out_dim = out_dim
        self.use_hf = False
        if name_or_path:
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
                self.use_hf = True
                hidden = getattr(self.model.config, 'hidden_size', out_dim)
                self.out_dim = int(hidden)
                if freeze:
                    self.model.eval()
                    self.model.requires_grad_(False)
            except Exception as exc:
                warnings.warn(
                    f"Failed to load LVM from {name_or_path}. Falling back to TinyFrameBackbone. Error: {exc}")
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
            outs = self.model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        except TypeError:
            outs = self.model(x, output_hidden_states=True)
        hiddens = list(outs.hidden_states[1:])
        max_idx = len(hiddens) - 1
        selected = []
        for idx in self.selected_layers:
            real_idx = max(0, min(max_idx, idx if idx >= 0 else len(hiddens) + idx))
            selected.append(hiddens[real_idx])
        return selected


class LayerWiseGaitAdapter(nn.Module):
    def __init__(self, lvm_dim: int, adapter_dim: int, out_hw: Tuple[int, int], num_layers: int):
        super().__init__()
        self.out_hw = out_hw
        self.projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(lvm_dim), nn.Linear(lvm_dim, adapter_dim), nn.GELU())
            for _ in range(num_layers)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(adapter_dim * num_layers, adapter_dim, 1, bias=False),
            nn.BatchNorm2d(adapter_dim),
            nn.GELU(),
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(adapter_dim),
            nn.GELU(),
        )

    @staticmethod
    def tokens_to_map(tokens: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        # tokens: [B, 1+L, C] or [B, L, C]
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
        return F.interpolate(feat, out_hw, mode='bilinear', align_corners=False)

    def forward(self, hidden_states: List[torch.Tensor], batch: int, frames: int) -> torch.Tensor:
        maps = []
        for h, proj in zip(hidden_states, self.projs):
            z = proj(h)
            maps.append(self.tokens_to_map(z, self.out_hw))
        x = self.fuse(torch.cat(maps, 1))
        return rearrange(x, '(n s) c h w -> n c s h w', n=batch, s=frames).contiguous()


class PartQueryGauPE(nn.Module):
    def __init__(self, dim: int, parts_num: int, temperature: float = 0.07, eps: float = 1e-6):
        super().__init__()
        self.parts_num = parts_num
        self.temperature = temperature
        self.eps = eps
        self.queries = nn.Parameter(torch.randn(parts_num, dim) * 0.02)
        self.part_proj = nn.Linear(dim, dim)
        self.gaupe_proj = nn.Sequential(
            nn.Linear(12, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        # feat: [N, C, T, H, W]
        n, c, t, h, w = feat.shape
        x = rearrange(feat, 'n c t h w -> (n t) (h w) c')
        q = F.normalize(self.queries, dim=-1)
        k = F.normalize(x, dim=-1)
        attn = torch.einsum('pc,blc->bpl', q, k) / max(self.temperature, self.eps)
        attn = torch.softmax(attn, dim=-1)
        part_feat = torch.einsum('bpl,blc->bpc', attn, x)
        attn_map = attn.reshape(n, t, self.parts_num, h, w)

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing='ij')
        grid = torch.stack([xx, yy], dim=-1).reshape(1, 1, 1, h * w, 2)
        prob = attn.reshape(n, t, self.parts_num, h * w, 1)
        mu = (prob * grid).sum(3)  # [N,T,P,2]
        delta = grid - mu.unsqueeze(3)
        var = (prob * delta ** 2).sum(3)
        cov = (prob[..., 0] * delta[..., 0] * delta[..., 1]).sum(3, keepdim=True)
        theta = 0.5 * torch.atan2(2 * cov, (var[..., 0:1] - var[..., 1:2]).clamp_min(self.eps))
        geom = torch.cat([mu, var, cov, theta], -1)  # 6 dims

        # velocity and acceleration of part centroids.
        vel = torch.zeros_like(mu)
        vel[:, 1:] = mu[:, 1:] - mu[:, :-1]
        acc = torch.zeros_like(mu)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        geom12 = torch.cat([geom, vel, acc], -1)  # [N,T,P,10]
        pad = torch.zeros(*geom12.shape[:-1], 2, device=feat.device, dtype=feat.dtype)
        geom12 = torch.cat([geom12, pad], -1)

        part_feat = self.part_proj(part_feat).reshape(n, t, self.parts_num, c)
        part_feat = part_feat + self.gaupe_proj(geom12)
        return {'part_feat': part_feat, 'attn_map': attn_map, 'mu': mu, 'vel': vel, 'acc': acc, 'geom': geom12}


class GraphTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class CrossPartKinematicTransformer(nn.Module):
    def __init__(self, dim: int, parts_num: int, depth: int = 2, heads: int = 4, graph_bias_strength: float = 0.25):
        super().__init__()
        self.parts_num = parts_num
        self.graph_bias_strength = graph_bias_strength
        self.blocks = nn.ModuleList([GraphTemporalBlock(dim, heads) for _ in range(depth)])
        self.part_type = nn.Parameter(torch.randn(1, parts_num, dim) * 0.02)

    def _mask(self, frames: int, device, dtype) -> torch.Tensor:
        p = self.parts_num
        # A default 10-part order: head, torso, pelvis, l_arm, r_arm, l_leg, r_leg, l_foot, r_foot, misc.
        edges = {(i, i) for i in range(p)}
        def add(a, b):
            if a < p and b < p:
                edges.add((a, b)); edges.add((b, a))
        add(1, 2); add(3, 6); add(4, 5); add(5, 6); add(5, 7); add(6, 8); add(2, 5); add(2, 6)
        base = torch.full((p, p), -float(self.graph_bias_strength), device=device, dtype=dtype)
        for a, b in edges:
            base[a, b] = 0.0
        # Keep temporal connections comparatively free; anatomical bias is strongest within a frame.
        mask = torch.zeros(frames * p, frames * p, device=device, dtype=dtype)
        for tt in range(frames):
            mask[tt*p:(tt+1)*p, tt*p:(tt+1)*p] = base
        return mask

    def forward(self, part_feat: torch.Tensor) -> torch.Tensor:
        # [N,T,P,C]
        n, t, p, c = part_feat.shape
        x = part_feat + self.part_type[:, :p].unsqueeze(1)
        x = rearrange(x, 'n t p c -> n (t p) c')
        mask = self._mask(t, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)
        return rearrange(x, 'n (t p) c -> n t p c', t=t, p=p).contiguous()


class SelfSupervisedPhase(nn.Module):
    def __init__(self, parts_num: int, foot_indices: Tuple[int, int] = (7, 8)):
        super().__init__()
        self.parts_num = parts_num
        self.foot_indices = foot_indices

    def forward(self, mu: torch.Tensor) -> Dict[str, torch.Tensor]:
        # mu: [N,T,P,2]
        n, t, p, _ = mu.shape
        li, ri = self.foot_indices
        li = min(li, p - 1); ri = min(ri, p - 1)
        left = mu[:, :, li, 0]
        right = mu[:, :, ri, 0]
        diff = left - right
        phase = torch.atan2(diff, torch.roll(diff, shifts=1, dims=1) - diff)
        # [sin, cos] is continuous and useful for phase-normalized losses.
        phase_code = torch.stack([torch.sin(phase), torch.cos(phase)], -1)
        return {'phase_code': phase_code, 'left_right_signal': diff}


class PointCloudPrivilegedEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, out_dim)
        )
        self.proj = nn.Sequential(nn.LayerNorm(out_dim), nn.Linear(out_dim, out_dim))

    def forward(self, pc: torch.Tensor) -> torch.Tensor:
        # pc: [N,T,P,C] or [N,T,H,W,C]
        if pc.dim() == 5:
            pc = pc.flatten(2, 3)
        if pc.size(-1) < 4:
            pad = torch.zeros(*pc.shape[:-1], 4 - pc.size(-1), device=pc.device, dtype=pc.dtype)
            pc = torch.cat([pc, pad], -1)
        pc = pc[..., :4]
        x = self.mlp(pc).max(dim=2)[0]  # [N,T,C]
        return self.proj(x.mean(1))


class SequenceFeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N,T,D] or [N,D]
        if x.dim() > 2:
            x = x.mean(1)
        return self.net(x)


class RealKineGait(BaseModel):
    """Training-heavy / inference-light causal kinematic gait recognizer.

    The model is deliberately modular. At inference it only needs RGB by default;
    text, pose, SMPL, point cloud, and event inputs are treated as training-time
    supervisors/privileged modalities when present in the prepared OpenGait pkl tree.
    """

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        self.input_indices = model_cfg.get('input_indices', {'rgb': 0})
        self.rgb_index = self.input_indices.get('rgb', 0)
        self.sil_index = self.input_indices.get('sil', None)
        self.motion_text_index = self.input_indices.get('motion_text', None)
        self.nuisance_text_index = self.input_indices.get('nuisance_text', None)
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

        selected_layers = model_cfg.get('lvm_layers', [3, 7, 11])
        lvm_hint_dim = int(model_cfg.get('lvm_dim', 384))
        self.LVM = FrozenDINOBackbone(
            name_or_path=model_cfg.get('lvm_name_or_path', None),
            out_dim=lvm_hint_dim,
            selected_layers=selected_layers,
            freeze=bool(model_cfg.get('freeze_lvm', True)))
        self.Adapter = LayerWiseGaitAdapter(self.LVM.out_dim, self.adapter_dim, self.feature_hw, len(selected_layers))
        self.PartQuery = PartQueryGauPE(self.adapter_dim, self.parts_num, model_cfg.get('part_temperature', 0.07))
        self.KineTF = CrossPartKinematicTransformer(
            self.adapter_dim, self.parts_num,
            depth=int(model_cfg.get('kinematic_depth', 2)),
            heads=int(model_cfg.get('kinematic_heads', 4)),
            graph_bias_strength=float(model_cfg.get('graph_bias_strength', 0.25)))
        self.Phase = SelfSupervisedPhase(self.parts_num, tuple(model_cfg.get('foot_indices', [7, 8])))

        self.HPP = HorizontalPoolingPyramid(self.hpp_bins)
        self.FCs = SeparateFCs(parts_num=self.total_parts, **model_cfg['SeparateFCs'])
        bn_cfg = dict(model_cfg['SeparateBNNecks'])
        bn_cfg['parts_num'] = self.total_parts
        self.BNNecks = SeparateBNNecks(**bn_cfg)

        text_dim = int(model_cfg.get('text_dim', 512))
        smpl_dim = int(model_cfg.get('smpl_dim', 256))
        self.MotionTextProj = SequenceFeatureProjector(text_dim, self.adapter_dim)
        self.NuisanceTextProj = SequenceFeatureProjector(text_dim, self.adapter_dim)
        self.SMPLTeacher = SequenceFeatureProjector(smpl_dim, self.adapter_dim)
        self.PCTeacher = PointCloudPrivilegedEncoder(self.adapter_dim, in_dim=int(model_cfg.get('pc_in_dim', 4)))
        self.EventTeacher = nn.Sequential(
            nn.Conv3d(int(model_cfg.get('event_in_channels', 2)), 32, 3, padding=1, bias=False), nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, self.adapter_dim, 3, padding=1, bias=False), nn.BatchNorm3d(self.adapter_dim), nn.GELU())
        self.QualityGate = nn.Sequential(
            nn.Linear(self.adapter_dim * 3, self.adapter_dim), nn.GELU(), nn.Linear(self.adapter_dim, 3))
        self.id_pool = nn.AdaptiveAvgPool1d(1)

    def init_parameters(self):
        # Keep DINO untouched. Init only trainable modules.
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
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.msg_mgr.log_info(f'RealKineGait trainable params: {n_parameters / 1e6:.3f}M')

    def _get_optional(self, ipts: List[torch.Tensor], index: Optional[int]) -> Optional[torch.Tensor]:
        if index is None or index >= len(ipts):
            return None
        return ipts[index]

    def _prep_rgb(self, rgb: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if rgb.dim() == 4:
            # Silhouette passed as primary input: [N,S,H,W] -> pseudo RGB.
            rgb = rgb.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        if rgb.size(2) == 1:
            rgb = rgb.repeat(1, 1, 3, 1, 1)
        n, s, c, h, w = rgb.shape
        x = rearrange(rgb, 'n s c h w -> (n s) c h w').contiguous()
        x = F.interpolate(x, self.image_hw, mode='bilinear', align_corners=False)
        return x, n, s

    def _pool_seq(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x.mean(1)
        if x.dim() == 4:
            return x.mean((1, 2))
        return x

    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        rgb = ipts[self.rgb_index]
        x, n, s = self._prep_rgb(rgb)
        with torch.no_grad() if self.model_cfg.get('freeze_lvm', True) else torch.enable_grad():
            hidden = self.LVM(x)
        feat5d = self.Adapter(hidden, n, s)  # [N,C,T,H,W]

        pq = self.PartQuery(feat5d)
        kine_seq = self.KineTF(pq['part_feat'])  # [N,T,P,C]
        phase = self.Phase(pq['mu'])
        kine_parts = kine_seq.mean(1).permute(0, 2, 1).contiguous()  # [N,C,P]
        z_kin = kine_seq.mean((1, 2))

        sem_map = feat5d.mean(2)
        sem_parts = self.HPP(sem_map)

        # Quality-aware fusion: semantic vs kinematic vs privileged dynamic hints.
        event = self._get_optional(ipts, self.event_index)
        if event is not None and event.dim() == 5:
            ev = event
            if ev.size(2) != int(self.model_cfg.get('event_in_channels', 2)):
                ev = ev[:, :, :int(self.model_cfg.get('event_in_channels', 2))]
            ev2 = rearrange(ev, 'n s c h w -> (n s) c h w').contiguous()
            ev2 = F.interpolate(ev2, self.feature_hw, mode='bilinear', align_corners=False)
            ev = rearrange(ev2, '(n s) c h w -> n c s h w', n=n, s=s).contiguous()
            z_event = self.EventTeacher(ev).mean((2, 3, 4))
        else:
            z_event = torch.zeros_like(z_kin)
        z_sem = sem_map.mean((2, 3))
        gate_logits = self.QualityGate(torch.cat([z_sem, z_kin, z_event], -1))
        gate = torch.softmax(gate_logits, -1)
        sem_parts = sem_parts * gate[:, 0].view(n, 1, 1)
        kine_parts = kine_parts * (gate[:, 1].view(n, 1, 1) + gate[:, 2].view(n, 1, 1))

        feat_parts = torch.cat([sem_parts, kine_parts], -1)
        embed_1 = self.FCs(feat_parts)
        embed_2, logits = self.BNNecks(embed_1)
        z_id = embed_1.mean(-1)

        # Text and privileged teachers. Missing modalities become detached zeros so losses are safe.
        motion_text = self._get_optional(ipts, self.motion_text_index)
        nuisance_text = self._get_optional(ipts, self.nuisance_text_index)
        smpl = self._get_optional(ipts, self.smpl_index)
        pc = self._get_optional(ipts, self.pc_index)
        z_motion_text = self.MotionTextProj(motion_text) if motion_text is not None else torch.zeros_like(z_kin)
        z_nuisance_text = self.NuisanceTextProj(nuisance_text) if nuisance_text is not None else torch.zeros_like(z_kin)
        z_smpl = self.SMPLTeacher(smpl) if smpl is not None else torch.zeros_like(z_kin)
        z_pc = self.PCTeacher(pc) if pc is not None and pc.dim() >= 4 else torch.zeros_like(z_kin)
        teacher = torch.stack([z_smpl, z_pc, z_event], 1)
        teacher_valid = torch.stack([
            torch.ones(n, device=z_kin.device, dtype=z_kin.dtype) if smpl is not None else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype),
            torch.ones(n, device=z_kin.device, dtype=z_kin.dtype) if pc is not None else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype),
            torch.ones(n, device=z_kin.device, dtype=z_kin.dtype) if event is not None else torch.zeros(n, device=z_kin.device, dtype=z_kin.dtype),
        ], 1)
        teacher_weight = teacher_valid / teacher_valid.sum(1, keepdim=True).clamp_min(1.0)
        z_teacher = (teacher * teacher_weight.unsqueeze(-1)).sum(1).detach()

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                'cd': {'embeddings': z_id, 'nuisance_embeddings': z_nuisance_text, 'labels': labs},
                'motion_align': {'kin_embeddings': z_kin, 'motion_embeddings': z_motion_text},
                'counterfactual': {'embeddings': z_id, 'labels': labs, 'nuisance_embeddings': z_nuisance_text},
                'phase': {'phase_code': phase['phase_code'], 'part_mu': pq['mu']},
                'distill3d': {'student_embeddings': z_kin, 'teacher_embeddings': z_teacher, 'teacher_valid': teacher_valid.sum(1)},
            },
            'visual_summary': {
                'image/rgb': rgb[:, :min(4, rgb.size(1))].flatten(0, 1) if rgb.dim() == 5 else rgb[:, :min(4, rgb.size(1))].unsqueeze(2).flatten(0, 1),
                'scalar/gate_semantic': gate[:, 0].mean().detach(),
                'scalar/gate_kinematic': gate[:, 1].mean().detach(),
                'scalar/gate_privileged': gate[:, 2].mean().detach(),
            },
            'inference_feat': {
                'embeddings': embed_1,
                'kin_embeddings': z_kin,
                'phase_code': phase['phase_code'].mean(1),
            }
        }
        return retval
