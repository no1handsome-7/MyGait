"""GaitMoEOracleV3 — Temporal BEFORE fusion, sequence-level fusion.

V3.1 update: replaced hand-crafted ConditionEncoder with learned
ModalityReliabilityProbe for end-to-end modality quality assessment.

Data flow:
  RGB -> Encoder -> [TextGate] -> HPP -> [Nuisance] -> reshape [B*T,C,P]
    -> AppearanceMoE -> reshape [B,T,C,P] -> RGB_TemporalEncoder -> [B,C,P]
    -> Projector -> [B,C',P]
  PC  -> Encoder -> HPP -> reshape [B*T,C,P]
    -> StructureMoE -> reshape [B,T,C,P] -> PC_TemporalEncoder -> [B,C,P]
    -> Projector -> [B,C',P]
  [ModalityReliabilityProbe] -> cond_emb, mod_weights
  Weighted-Stack [B,2P,C'] -> FusionMoE (condition-aware) -> [B,C',P]
  -> SeparateFCs -> SeparateBNNecks -> [Nuisance post-temporal]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks
from .gait_moe_blocks import (
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    TopKMoELayer, FusionMoELayer, CrossAttentionFusion, MLPFusion,
    ModalityReliabilityProbe,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    symmetric_infonce, probe_balance_loss,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
)


class GaitMoEOracleV3(BaseModel):

    def build_network(self, model_cfg):
        C = int(model_cfg.get('channels', 256))
        P = int(model_cfg.get('parts_num', 16))
        proj_dim = int(model_cfg.get('proj_dim', C))
        self.channels, self.parts_num = C, P
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_fwd_iter', torch.zeros((), dtype=torch.long), persistent=False)

        abl = model_cfg.get('ablation', {})
        self.app_moe_on = bool(abl.get('appearance_moe_enabled', True))
        self.str_moe_on = bool(abl.get('structure_moe_enabled', True))
        self.fus_moe_on = bool(abl.get('fusion_moe_enabled', True))
        self.fus_fallback = str(abl.get('fusion_fallback', 'cross_attention'))
        self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
        self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
        er = int(abl.get('expert_rank', 16))
        ed = float(abl.get('expert_dropout', 0.0))
        self.bal_w = float(abl.get('balance_loss_weight', 0.01))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))

        self.proj_dim = proj_dim

        tcfg = model_cfg.get('text_adapter', {})
        td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        tcd, thd = int(tcfg.get('code_dim', 256)), int(tcfg.get('hidden_dim', 1024))
        tdp = float(tcfg.get('dropout', 0.1))

        # --- Encoders ---
        self.rgb_enc = TinyDenseRGBEncoder(C, 3)
        if self.txt_gate_on:
            pcfg = model_cfg.get('positive_text', {})
            self.pos_gate = PositiveTextGate(
                pcfg.get('text_dim', td), C, pcfg.get('prompt_feat_path'),
                pcfg.get('num_prompts', 4), pcfg.get('trainable', False),
                pcfg.get('text_code_dim', tcd), pcfg.get('text_hidden_dim', thd),
                pcfg.get('dropout', tdp))
        else:
            self.pos_gate = None

        ptype = str(model_cfg.get('pc_encoder_type', 'simple'))
        if ptype == 'lidargaitpp':
            pc_cfg = model_cfg.get('pc_model_cfg', {})
            self.pc_enc = LidarGaitPPEncoder(
                pc_cfg.get('channel', 16), pc_cfg.get('npoints', [256, 128, 64]),
                pc_cfg.get('nsample', 8), pc_cfg.get('sampling', 'knn'),
                pc_cfg.get('scale_aware', True), pc_cfg.get('normalize_dp', True), C, P)
        else:
            self.pc_enc = PointCloudSimpleEncoder(C, P, int(model_cfg.get('pc_in_dim', 3)))
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        # --- Nuisance ---
        if self.nuis_on:
            nc = dict(model_cfg.get('nuisance_detach', {}))
            self.nuis = StagedNuisanceRemoval(
                C, P, nc.get('text_dim', td), nc.get('text_code_dim', tcd),
                nc.get('text_hidden_dim', thd), nc.get('sub_dim', 256),
                nc.get('factors'), nc.get('grl_lambda', 0.08), nc.get('temperature', 0.07),
                nc.get('warmup_iters', 5000), nc.get('probe_iters', 10000),
                nc.get('adv_ramp_iters', 10000), nc.get('reliability_start_iter', 15000),
                nc.get('reliability_ramp_iters', 10000), nc.get('probe_loss_weight', 0.03),
                nc.get('adv_loss_weight', 0.03), nc.get('mask_loss_weight', 0.015),
                nc.get('max_suppression', 0.35), nc.get('detach_reliability_gate', True),
                nc.get('factor_weights'), nc.get('dropout', tdp))
        else:
            self.nuis = None

        # --- MoE layers (appearance / structure) ---
        self.app_moe = TopKMoELayer(C, P, int(abl.get('appearance_num_experts', 8)),
                                    int(abl.get('appearance_topk', 2)), er, ed) if self.app_moe_on else None
        self.str_moe = TopKMoELayer(C, P, int(abl.get('structure_num_experts', 8)),
                                    int(abl.get('structure_topk', 2)), er, ed) if self.str_moe_on else None

        # --- TWO independent temporal encoders (key V3 difference) ---
        tc = model_cfg.get('cycle_temporal', {})
        self.rgb_temporal = CycleAwareTemporalEncoder(
            C, P, tc.get('heads', 4), tc.get('layers', 1),
            tc.get('dropout', 0.1), tc.get('future_pred', True))
        self.pc_temporal = CycleAwareTemporalEncoder(
            C, P, tc.get('heads', 4), tc.get('layers', 1),
            tc.get('dropout', 0.1), tc.get('future_pred', False))   # PC: no future pred

        # --- Projectors ---
        self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = (nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity())

        # --- Modality Reliability Probe (replaces ConditionEncoder) ---
        probe_cfg = model_cfg.get('modality_probe', {})
        self.probe_on = bool(probe_cfg.get('enabled', True))
        probe_cond_dim = int(probe_cfg.get('cond_dim', 32))
        if self.probe_on:
            self.mod_probe = ModalityReliabilityProbe(
                channels=proj_dim,
                cond_dim=probe_cond_dim,
                hidden_dim=int(probe_cfg.get('hidden_dim', 64)),
            )
            self.probe_temp_init = float(probe_cfg.get('temperature_init', 2.0))
            self.probe_temp_min = float(probe_cfg.get('temperature_min', 0.5))
            self.probe_temp_anneal = int(probe_cfg.get('temp_anneal_iters', 30000))
            self.probe_bal_w = float(probe_cfg.get('balance_weight', 0.01))
            self.probe_drop_prob = float(probe_cfg.get('modality_drop_prob', 0.1))
        else:
            self.mod_probe = None
            self.probe_bal_w = 0.0
            self.probe_drop_prob = 0.0

        # --- Fusion MoE (operates at sample level, no seq_len) ---
        nit = 2 * P
        fus_rank = int(abl.get('fusion_expert_rank', 32))
        fus_cond_dim = probe_cond_dim if self.probe_on else 0
        if self.fus_moe_on:
            self.fus_moe = FusionMoELayer(
                proj_dim, P, nit, int(abl.get('fusion_num_experts', 8)),
                int(abl.get('fusion_topk', 2)), fus_cond_dim, fus_rank, ed)
        else:
            self.fus_moe = None
            if self.fus_fallback == 'cross_attention':
                self.fus_fb = CrossAttentionFusion(proj_dim, P)
            else:
                self.fus_fb = MLPFusion(proj_dim, P, nit)

        # --- ID heads ---
        fc, bn = model_cfg.get('SeparateFCs', {}), model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim), fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256), bn.get('class_num', 250))

        self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))

    # --- helpers ---

    def _step(self):
        if self.training: self._fwd_iter += 1
        for a in ('iteration', 'global_step'):
            if hasattr(self, a):
                v = getattr(self, a)
                try: return int(v.item()) if torch.is_tensor(v) else int(v)
                except: pass
        return int(self._fwd_iter.item())

    def _probe_temperature(self, step):
        """Exponential annealing: start soft (high tau) -> end sharp (low tau)."""
        if not self.probe_on:
            return 1.0
        t_max = self.probe_temp_init
        t_min = self.probe_temp_min
        anneal = self.probe_temp_anneal
        if anneal <= 0 or step >= anneal:
            return t_min
        ratio = float(step) / float(anneal)
        return t_max * (t_min / max(t_max, 1e-6)) ** ratio

    def _rgb(self, x):
        if x.dim() == 4: x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        elif x.dim() == 5:
            if x.shape[2] == 1: x = x.expand(-1, -1, 3, -1, -1)
            elif x.shape[-1] == 3 and x.shape[2] != 3: x = x.permute(0, 1, 4, 2, 3)
        return x.float()

    def _tf(self, x):
        if x is None: return None
        x = x.float()
        return x.mean(1) if x.dim() == 3 else (x if x.dim() == 2 else x.view(x.size(0), -1))

    def _opt(self, ipts, idx):
        if idx is None or (isinstance(idx, str) and idx.lower() in ('none', 'null')): return None
        idx = int(idx)
        return ipts[idx] if 0 <= idx < len(ipts) else None

    def _nuis_io(self, ipts):
        tf, mk = {}, {}
        for k, i in self.nuisance_indices.items(): tf[k] = self._tf(self._opt(ipts, i))
        for k, i in self.nuisance_mask_indices.items(): mk[k] = self._tf(self._opt(ipts, i))
        return tf, mk

    def _suml(self, ls, ref):
        t = ref.new_tensor(0.0)
        for v in ls.values():
            if v is not None: t = t + v
        return t

    # --- forward ---

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._step()

        # ═══ 1. Encode ═══
        rgb = self._rgb(ipts[self.rgb_index])
        pc = ipts[self.pc_index].float()
        rgb_map = self.rgb_enc(rgb)
        ga = None
        if self.pos_gate is not None:
            rgb_map, ga = self.pos_gate(rgb_map)
        rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(*rgb_map.shape[:2], self.channels, self.parts_num)
        pc_ps = self.pc_enc(pc)
        B, T, C, P = rgb_ps.shape

        # ═══ 2. Nuisance pre-temporal ═══
        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(rgb_ps.mean((1, 3)), tfs, mks, step)
            rgb_ps = rgb_ps * rel

        # ═══ 3. Per-frame MoE ═══
        rf = rgb_ps.reshape(B * T, C, P)
        pf = pc_ps.reshape(B * T, C, P)

        al, aw, sl, sw = None, None, None, None
        if self.app_moe is not None:
            rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None:
            pf, sl, sw = self.str_moe(pf, seq_len=T)

        # ═══ 4. Per-modality temporal encoding (V3 key) ═══
        rgb_expert_seq = rf.reshape(B, T, C, P)
        pc_expert_seq = pf.reshape(B, T, C, P)

        rgb_feat, rgb_fl, _ = self.rgb_temporal(rgb_expert_seq)   # [B, C, P]
        pc_feat, pc_fl, _ = self.pc_temporal(pc_expert_seq)       # [B, C, P]

        # ═══ 5. Project to shared dim ═══
        rgb_p = self.rgb_proj(rgb_feat)                            # [B, C', P]
        pc_p = self.pc_proj(pc_feat)                               # [B, C', P]

        # ═══ 5.5 Modality Reliability Probe ═══
        probe_diag = {}
        if self.mod_probe is not None:
            temp = self._probe_temperature(step)
            cond_emb, mod_weights, probe_diag = self.mod_probe(
                rgb_p, pc_p, temperature=temp,
            )

            # -- Modality dropout: training-time regularization --
            if self.training and self.probe_drop_prob > 0:
                if torch.rand(1).item() < self.probe_drop_prob:
                    drop_idx = torch.randint(0, 2, (1,)).item()
                    mod_weights = mod_weights.clone()
                    mod_weights[:, drop_idx] = 0.0
                    mod_weights = mod_weights / mod_weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)

            # -- Apply modality reliability weights to fusion tokens --
            rgb_w = mod_weights[:, 0:1].unsqueeze(-1)              # [B, 1, 1]
            pc_w = mod_weights[:, 1:2].unsqueeze(-1)               # [B, 1, 1]
            tokens = torch.cat([
                rgb_p.permute(0, 2, 1) * rgb_w,                    # [B, P, C']
                pc_p.permute(0, 2, 1) * pc_w,                      # [B, P, C']
            ], dim=1)                                               # [B, 2P, C']
        else:
            cond_emb = None
            mod_weights = None
            tokens = torch.cat([
                rgb_p.permute(0, 2, 1),
                pc_p.permute(0, 2, 1),
            ], dim=1)                                               # [B, 2P, C']

        # ═══ 6. Fusion MoE (sequence-level, naturally) ═══
        fl, fw = None, None

        if self.fus_moe is not None:
            fused, fl, fw = self.fus_moe(tokens, cond_emb, seq_len=1)
        else:
            fused, _, _ = self.fus_fb(tokens, seq_len=1)

        # ═══ 7. ID heads ═══
        e1 = self.FCs(fused)
        e2, logits = self.BNNecks(e1)

        # ═══ 8. Losses ═══
        tf = {'triplet': {'embeddings': e1, 'labels': labs},
              'softmax': {'logits': logits, 'labels': labs}}
        aux = {}

        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(rgb_feat.mean(-1), pc_feat.mean(-1)) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = (rgb_fl + pc_fl) * self.fut_w

        if self.training:
            # MoE balance & diversity losses
            if self.bal_w > 0:
                bl = sum(moe_balance_loss(l) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt: aux['moe_bal'] = bl / cnt * self.bal_w
            if self.div_w > 0:
                dv = e1.new_tensor(0.0); cnt = 0
                if self.app_moe is not None: dv = dv + moe_diversity_loss(self.app_moe.experts, thr=self.div_thr); cnt += 1
                if self.str_moe is not None: dv = dv + moe_diversity_loss(self.str_moe.experts, thr=self.div_thr); cnt += 1
                if self.fus_moe is not None: dv = dv + moe_token_diversity_loss(self.fus_moe.experts, self.div_thr); cnt += 1
                if cnt: aux['moe_div'] = dv / cnt * self.div_w

            # Probe balance loss (prevents modality weight collapse)
            if mod_weights is not None and self.probe_bal_w > 0:
                aux['probe_bal'] = probe_balance_loss(mod_weights) * self.probe_bal_w

        if ga is not None:
            vl = self.verifier(ga, rgb)
            if vl.requires_grad or vl.item() != 0: aux['verifier'] = vl

        if self.nuis is not None:
            id_global = fused.mean(-1)                                   # [B, proj_dim]
            id_global = self.post_nuisance_adapter(id_global.float())    # [B, C]
            npost = self.nuis.post_temporal(id_global, tfs, mks, step)
            aux.update({f'n_{k}': v for k, v in npre.items()})
            aux.update({f'n_{k}': v for k, v in npost.items()})

        if self.scalar_aux:
            tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
        else:
            for k, v in aux.items(): tf[f'loss_{k}'] = v

        # ═══ 9. Visual summary ═══
        vs = {}
        for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        if self.nuis is not None and self.training:
            for k, v in ndiag.items(): vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v
        # Probe diagnostics
        for k, v in probe_diag.items():
            vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v
        if hasattr(self, 'probe_temp_init') and self.probe_on:
            vs['scalar/probe_temperature'] = self._probe_temperature(step)

        return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': {'embeddings': e1}}






# """GaitMoEOracleV3 — Temporal BEFORE fusion, sequence-level fusion.

# Data flow:
#   RGB → Encoder → [TextGate] → HPP → [Nuisance] → reshape [B*T,C,P]
#     → AppearanceMoE → reshape [B,T,C,P] → RGB_TemporalEncoder → [B,C,P]
#     → Projector → [B,C',P]
#   PC  → Encoder → HPP → reshape [B*T,C,P]
#     → StructureMoE → reshape [B,T,C,P] → PC_TemporalEncoder → [B,C,P]
#     → Projector → [B,C',P]
#   Stack [B,2P,C'] → FusionMoE (naturally sequence-level) → [B,C',P]
#   → SeparateFCs → SeparateBNNecks → [Nuisance post-temporal]

# Key difference from V2: temporal aggregation happens per-modality BEFORE fusion,
# so FusionMoE operates on sequence-level features without needing seq_len tricks.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks
# from .gait_moe_blocks import (
#     TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
#     PositiveTextGate, StagedNuisanceRemoval,
#     TopKMoELayer, FusionMoELayer, CrossAttentionFusion, MLPFusion,
#     ConditionEncoder, CycleAwareTemporalEncoder, RuleBasedVerifier,
#     symmetric_infonce, compute_condition_stats,
#     moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
# )


# class GaitMoEOracleV3(BaseModel):

#     def build_network(self, model_cfg):
#         C = int(model_cfg.get('channels', 256))
#         P = int(model_cfg.get('parts_num', 16))
#         proj_dim = int(model_cfg.get('proj_dim', C))
#         self.channels, self.parts_num = C, P
#         self.rgb_index = model_cfg.get('rgb_index', 1)
#         self.pc_index = model_cfg.get('pc_index', 0)
#         self.nuisance_indices = model_cfg.get('nuisance_indices', {})
#         self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
#         self.register_buffer('_fwd_iter', torch.zeros((), dtype=torch.long), persistent=False)

#         abl = model_cfg.get('ablation', {})
#         self.app_moe_on = bool(abl.get('appearance_moe_enabled', True))
#         self.str_moe_on = bool(abl.get('structure_moe_enabled', True))
#         self.fus_moe_on = bool(abl.get('fusion_moe_enabled', True))
#         self.fus_fallback = str(abl.get('fusion_fallback', 'cross_attention'))
#         self.cond_on = bool(abl.get('fusion_condition_aware', True))
#         self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
#         self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
#         er = int(abl.get('expert_rank', 16))
#         ed = float(abl.get('expert_dropout', 0.0))
#         self.bal_w = float(abl.get('balance_loss_weight', 0.01))
#         self.div_w = float(abl.get('diversity_loss_weight', 0.01))
#         self.div_thr = float(abl.get('diversity_threshold', 0.5))
#         self.cond_w = float(abl.get('condition_cls_weight', 0.05))
#         cond_dim = int(abl.get('cond_dim', 32))

#         # proj_dim = int(model_cfg.get('proj_dim', C))
#         self.proj_dim = proj_dim

#         tcfg = model_cfg.get('text_adapter', {})
#         td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
#         tcd, thd = int(tcfg.get('code_dim', 256)), int(tcfg.get('hidden_dim', 1024))
#         tdp = float(tcfg.get('dropout', 0.1))

#         # --- Encoders ---
#         self.rgb_enc = TinyDenseRGBEncoder(C, 3)
#         if self.txt_gate_on:
#             pcfg = model_cfg.get('positive_text', {})
#             self.pos_gate = PositiveTextGate(
#                 pcfg.get('text_dim', td), C, pcfg.get('prompt_feat_path'),
#                 pcfg.get('num_prompts', 4), pcfg.get('trainable', False),
#                 pcfg.get('text_code_dim', tcd), pcfg.get('text_hidden_dim', thd),
#                 pcfg.get('dropout', tdp))
#         else:
#             self.pos_gate = None

#         ptype = str(model_cfg.get('pc_encoder_type', 'simple'))
#         if ptype == 'lidargaitpp':
#             pc_cfg = model_cfg.get('pc_model_cfg', {})
#             self.pc_enc = LidarGaitPPEncoder(
#                 pc_cfg.get('channel', 16), pc_cfg.get('npoints', [256, 128, 64]),
#                 pc_cfg.get('nsample', 8), pc_cfg.get('sampling', 'knn'),
#                 pc_cfg.get('scale_aware', True), pc_cfg.get('normalize_dp', True), C, P)
#         else:
#             self.pc_enc = PointCloudSimpleEncoder(C, P, int(model_cfg.get('pc_in_dim', 3)))
#         self.HPP = HorizontalPoolingPyramid(bin_num=[P])

#         # --- Nuisance ---
#         if self.nuis_on:
#             nc = dict(model_cfg.get('nuisance_detach', {}))
#             self.nuis = StagedNuisanceRemoval(
#                 C, P, nc.get('text_dim', td), nc.get('text_code_dim', tcd),
#                 nc.get('text_hidden_dim', thd), nc.get('sub_dim', 256),
#                 nc.get('factors'), nc.get('grl_lambda', 0.08), nc.get('temperature', 0.07),
#                 nc.get('warmup_iters', 5000), nc.get('probe_iters', 10000),
#                 nc.get('adv_ramp_iters', 10000), nc.get('reliability_start_iter', 15000),
#                 nc.get('reliability_ramp_iters', 10000), nc.get('probe_loss_weight', 0.03),
#                 nc.get('adv_loss_weight', 0.03), nc.get('mask_loss_weight', 0.015),
#                 nc.get('max_suppression', 0.35), nc.get('detach_reliability_gate', True),
#                 nc.get('factor_weights'), nc.get('dropout', tdp))
#         else:
#             self.nuis = None

#         # --- MoE layers (appearance / structure) ---
#         self.app_moe = TopKMoELayer(C, P, int(abl.get('appearance_num_experts', 8)),
#                                     int(abl.get('appearance_topk', 2)), er, ed) if self.app_moe_on else None
#         self.str_moe = TopKMoELayer(C, P, int(abl.get('structure_num_experts', 8)),
#                                     int(abl.get('structure_topk', 2)), er, ed) if self.str_moe_on else None

#         # --- TWO independent temporal encoders (key V3 difference) ---
#         tc = model_cfg.get('cycle_temporal', {})
#         self.rgb_temporal = CycleAwareTemporalEncoder(
#             C, P, tc.get('heads', 4), tc.get('layers', 1),
#             tc.get('dropout', 0.1), tc.get('future_pred', True))
#         self.pc_temporal = CycleAwareTemporalEncoder(
#             C, P, tc.get('heads', 4), tc.get('layers', 1),
#             tc.get('dropout', 0.1), tc.get('future_pred', False))   # PC: no future pred

#         # --- Projectors ---
#         self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
#         self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
#         self.post_nuisance_adapter = (nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity())

#         # --- Condition ---
#         self.cond_enc = ConditionEncoder(6, cond_dim, int(abl.get('cond_hidden_dim', 64))) if self.cond_on else None

#         # --- Fusion MoE (operates at sample level, no seq_len) ---
#         nit = 2 * P
#         fus_rank = int(abl.get('fusion_expert_rank', 32))
#         if self.fus_moe_on:
#             self.fus_moe = FusionMoELayer(
#                 proj_dim, P, nit, int(abl.get('fusion_num_experts', 8)),
#                 int(abl.get('fusion_topk', 2)), cond_dim if self.cond_on else 0, fus_rank, ed)
#         else:
#             self.fus_moe = None
#             if self.fus_fallback == 'cross_attention':
#                 self.fus_fb = CrossAttentionFusion(proj_dim, P)
#             else:
#                 self.fus_fb = MLPFusion(proj_dim, P, nit)

#         # --- ID heads ---
#         fc, bn = model_cfg.get('SeparateFCs', {}), model_cfg.get('SeparateBNNecks', {})
#         self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim), fc.get('out_channels', 256))
#         self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256), bn.get('class_num', 250))

#         self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
#         self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
#         self.fut_w = float(model_cfg.get('future_weight', 0.01))
#         self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))

#     # --- helpers ---

#     def _step(self):
#         if self.training: self._fwd_iter += 1
#         for a in ('iteration', 'global_step'):
#             if hasattr(self, a):
#                 v = getattr(self, a)
#                 try: return int(v.item()) if torch.is_tensor(v) else int(v)
#                 except: pass
#         return int(self._fwd_iter.item())

#     def _rgb(self, x):
#         if x.dim() == 4: x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
#         elif x.dim() == 5:
#             if x.shape[2] == 1: x = x.expand(-1, -1, 3, -1, -1)
#             elif x.shape[-1] == 3 and x.shape[2] != 3: x = x.permute(0, 1, 4, 2, 3)
#         return x.float()

#     def _tf(self, x):
#         if x is None: return None
#         x = x.float()
#         return x.mean(1) if x.dim() == 3 else (x if x.dim() == 2 else x.view(x.size(0), -1))

#     def _opt(self, ipts, idx):
#         if idx is None or (isinstance(idx, str) and idx.lower() in ('none', 'null')): return None
#         idx = int(idx)
#         return ipts[idx] if 0 <= idx < len(ipts) else None

#     def _nuis_io(self, ipts):
#         tf, mk = {}, {}
#         for k, i in self.nuisance_indices.items(): tf[k] = self._tf(self._opt(ipts, i))
#         for k, i in self.nuisance_mask_indices.items(): mk[k] = self._tf(self._opt(ipts, i))
#         return tf, mk

#     def _suml(self, ls, ref):
#         t = ref.new_tensor(0.0)
#         for v in ls.values():
#             if v is not None: t = t + v
#         return t

#     # --- forward ---

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs
#         step = self._step()

#         # ═══ 1. Encode ═══
#         rgb = self._rgb(ipts[self.rgb_index])
#         pc = ipts[self.pc_index].float()
#         rgb_map = self.rgb_enc(rgb)
#         ga = None
#         if self.pos_gate is not None:
#             rgb_map, ga = self.pos_gate(rgb_map)
#         rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(*rgb_map.shape[:2], self.channels, self.parts_num)
#         pc_ps = self.pc_enc(pc)
#         B, T, C, P = rgb_ps.shape

#         # ═══ 2. Nuisance pre-temporal ═══
#         npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
#         if self.nuis is not None:
#             tfs, mks = self._nuis_io(ipts)
#             rel, npre, ndiag = self.nuis.pre_temporal(rgb_ps.mean((1, 3)), tfs, mks, step)
#             rgb_ps = rgb_ps * rel

#         # ═══ 3. Per-frame MoE ═══
#         rf = rgb_ps.reshape(B * T, C, P)
#         pf = pc_ps.reshape(B * T, C, P)

#         al, aw, sl, sw = None, None, None, None
#         if self.app_moe is not None:
#             rf, al, aw = self.app_moe(rf, seq_len=T)
#         if self.str_moe is not None:
#             pf, sl, sw = self.str_moe(pf, seq_len=T)

#         # ═══ 4. Per-modality temporal encoding (V3 key) ═══
#         rgb_expert_seq = rf.reshape(B, T, C, P)
#         pc_expert_seq = pf.reshape(B, T, C, P)

#         rgb_feat, rgb_fl, _ = self.rgb_temporal(rgb_expert_seq)   # [B, C, P]
#         pc_feat, pc_fl, _ = self.pc_temporal(pc_expert_seq)       # [B, C, P]

#         # ═══ 5. Project to shared dim ═══
#         rgb_p = self.rgb_proj(rgb_feat)                            # [B, C', P]
#         pc_p = self.pc_proj(pc_feat)                               # [B, C', P]

#         # ═══ 6. Fusion MoE (sequence-level, naturally) ═══
#         tokens = torch.cat([rgb_p.permute(0, 2, 1),
#                             pc_p.permute(0, 2, 1)], dim=1)         # [B, 2P, C']

#         fl, fw = None, None
#         cond_loss = tokens.new_tensor(0.0)

#         if self.fus_moe is not None:
#             if self.cond_enc is not None:
#                 cs = compute_condition_stats(rgb, pc)
#                 ce, il, dl = self.cond_enc(cs)                     # [B, cond]
#                 if self.training and self.cond_w > 0:
#                     ilbl, dlbl = ConditionEncoder.pseudo_labels(cs)
#                     cond_loss = (F.cross_entropy(il, ilbl) + F.cross_entropy(dl, dlbl)) * self.cond_w
#             else:
#                 ce = tokens.new_zeros(B, 32)
#             fused, fl, fw = self.fus_moe(tokens, ce, seq_len=1)    # seq_len=1: already pooled
#         else:
#             fused, _, _ = self.fus_fb(tokens, seq_len=1)

#         # ═══ 7. ID heads ═══
#         e1 = self.FCs(fused)
#         e2, logits = self.BNNecks(e1)

#         # ═══ 8. Losses ═══
#         tf = {'triplet': {'embeddings': e1, 'labels': labs},
#               'softmax': {'logits': logits, 'labels': labs}}
#         aux = {}

#         if self.anc_w > 0:
#             aux['anchor'] = symmetric_infonce(rgb_feat.mean(-1), pc_feat.mean(-1)) * self.anc_w
#         if self.fut_w > 0:
#             aux['future'] = (rgb_fl + pc_fl) * self.fut_w

#         if self.training:
#             if self.bal_w > 0:
#                 bl = sum(moe_balance_loss(l) for l in [al, sl, fl] if l is not None)
#                 cnt = sum(1 for l in [al, sl, fl] if l is not None)
#                 if cnt: aux['moe_bal'] = bl / cnt * self.bal_w
#             if self.div_w > 0:
#                 dv = e1.new_tensor(0.0); cnt = 0
#                 if self.app_moe is not None: dv = dv + moe_diversity_loss(self.app_moe.experts, thr=self.div_thr); cnt += 1
#                 if self.str_moe is not None: dv = dv + moe_diversity_loss(self.str_moe.experts, thr=self.div_thr); cnt += 1
#                 if self.fus_moe is not None: dv = dv + moe_token_diversity_loss(self.fus_moe.experts, self.div_thr); cnt += 1
#                 if cnt: aux['moe_div'] = dv / cnt * self.div_w
#             if cond_loss.abs() > 0: aux['cond'] = cond_loss

#         if ga is not None:
#             vl = self.verifier(ga, rgb)
#             if vl.requires_grad or vl.item() != 0: aux['verifier'] = vl

#         if self.nuis is not None:
#             id_global = fused.mean(-1)                                   # [B, proj_dim]
#             id_global = self.post_nuisance_adapter(id_global.float())    # [B, C]
#             npost = self.nuis.post_temporal(id_global, tfs, mks, step)
#             aux.update({f'n_{k}': v for k, v in npre.items()})
#             aux.update({f'n_{k}': v for k, v in npost.items()})

#         if self.scalar_aux:
#             tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
#         else:
#             for k, v in aux.items(): tf[f'loss_{k}'] = v

#         vs = {}
#         for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
#             if w is not None:
#                 for i in range(w.size(1)):
#                     vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
#         if self.nuis is not None and self.training:
#             for k, v in ndiag.items(): vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

#         return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': {'embeddings': e1}}
