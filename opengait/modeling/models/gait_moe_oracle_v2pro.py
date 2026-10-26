"""GaitMoEOracleV2Pro — Align Before Fuse + Quality-Aware MoE.

= V2Plus (Fix-5..10) + Fix-11 (Part-Level Align-Before-Fuse)

Key idea:  RGB and PC features live in separate manifolds (modality gap).
V2/V2Plus fuse them directly, forcing MoE experts to bridge the gap AND
discriminate gait simultaneously.  V2Pro first ALIGNS part-level features
in a shared space via cross-modal contrastive learning + cross-attention
bridge, THEN fuses the aligned representations.  This follows the
"Align Before Fuse" principle (ALBEF, NeurIPS 2021).

Data flow:
  RGB → Encoder → [TextGate] → HPP → rgb_ps [B,T,C,P]
  PC  → Encoder →              HPP → pc_ps  [B,T,C,P]
  ──── Fix-11: Part-Level Align-Before-Fuse ────
  rgb_aligned, pc_aligned, align_loss = PLABF(rgb_ps, pc_ps)
  ────────────────────────────────────────────────
  rgb_aligned → [Nuisance] → AppearanceMoE → Projector
  pc_aligned  →              StructureMoE  → Projector
  MQE(globals) → quality_scores
  [ModalityDropout] → Stack → QualityAwareFusionMoE → Temporal
  → SeparateFCs → SeparateBNNecks
"""

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

from .gait_moe_blocks import (
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    CrossAttentionFusion, MLPFusion,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    symmetric_infonce, compute_condition_stats,
    moe_diversity_loss,
)
from .gait_moe_blocks_v2plus import (
    ModalityQualityEstimator, EnhancedConditionEncoder,
    QualityAwareFusionMoELayer, TopKMoELayerV2Plus, ModalityDropout,
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss, expert_utilization_stats,
)
from .gait_moe_blocks_v2pro import (
    PartLevelAlignBeforeFuse,
    cross_modal_alignment_loss,
)


class GaitMoEOracleV2Pro(BaseModel):

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
        self.cond_on = bool(abl.get('fusion_condition_aware', True))
        self.temp_type = str(abl.get('temporal_type', 'cycle_aware'))
        self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
        self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
        er = int(abl.get('expert_rank', 16))
        ed = float(abl.get('expert_dropout', 0.0))
        self.bal_w = float(abl.get('balance_loss_weight', 0.02))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))
        self.cond_w = float(abl.get('condition_cls_weight', 0.1))
        self.z_w = float(abl.get('z_loss_weight', 0.01))
        self.mqe_w = float(abl.get('quality_loss_weight', 0.05))
        self.mod_drop = float(abl.get('modality_dropout', 0.1))
        cond_dim = int(abl.get('cond_dim', 64))
        self.proj_dim = proj_dim

        # ---- Fix-11: Alignment config ----
        acfg = model_cfg.get('cross_modal_align', {})
        self.align_w = float(acfg.get('align_weight', 0.15))
        self.align_plc_w = float(acfg.get('plc_weight', 1.0))
        self.align_ims_w = float(acfg.get('ims_weight', 0.3))
        self.align_temp = float(acfg.get('temperature', 0.07))
        align_dim = int(acfg.get('align_dim', 128))
        align_heads = int(acfg.get('num_heads', 4))
        align_warmup = int(acfg.get('warmup_iters', 1000))
        align_ramp = int(acfg.get('ramp_iters', 5000))
        self.align_warmup, self.align_ramp = align_warmup, align_ramp

        tcfg = model_cfg.get('text_adapter', {})
        td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        tcd = int(tcfg.get('code_dim', 256))
        thd = int(tcfg.get('hidden_dim', 1024))
        tdp = float(tcfg.get('dropout', 0.1))

        # ---- Encoders ----
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

        # ---- Fix-11: Part-Level Align-Before-Fuse ----
        self.plabf = PartLevelAlignBeforeFuse(
            C, align_dim, P, align_heads, dropout=tdp)

        # ---- Nuisance ----
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

        # ---- MoE (Fix-10) ----
        self.app_moe = TopKMoELayerV2Plus(
            C, P, int(abl.get('appearance_num_experts', 8)),
            int(abl.get('appearance_topk', 2)), er, ed) if self.app_moe_on else None
        self.str_moe = TopKMoELayerV2Plus(
            C, P, int(abl.get('structure_num_experts', 8)),
            int(abl.get('structure_topk', 2)), er, ed) if self.str_moe_on else None

        # ---- Projectors ----
        self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False),
                                      nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False),
                                     nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = (nn.Linear(proj_dim, C, bias=False)
                                      if proj_dim != C else nn.Identity())

        # ---- Fix-5: MQE ----
        mqe_cfg = model_cfg.get('modality_quality', {})
        self.mqe = ModalityQualityEstimator(C, hidden_dim=int(mqe_cfg.get('hidden_dim', 128)))

        # ---- Fix-6: Enhanced ConditionEncoder ----
        if self.cond_on:
            self.cond_enc = EnhancedConditionEncoder(
                C, cond_dim, hidden_dim=int(abl.get('cond_hidden_dim', 128)),
                num_illum_bins=int(abl.get('num_illum_bins', 5)),
                num_dist_bins=int(abl.get('num_dist_bins', 5)))
        else:
            self.cond_enc = None

        # ---- Fix-9: Modality Dropout ----
        self.modality_dropout = ModalityDropout(p=self.mod_drop)

        # ---- Fix-7: Quality-Aware FusionMoE ----
        nit = 2 * P
        fus_rank = int(abl.get('fusion_expert_rank', 32))
        if self.fus_moe_on:
            self.fus_moe = QualityAwareFusionMoELayer(
                proj_dim, P, nit, int(abl.get('fusion_num_experts', 8)),
                int(abl.get('fusion_topk', 2)), cond_dim if self.cond_on else 0,
                fus_rank, ed)
        else:
            self.fus_moe = None
            if self.fus_fallback == 'cross_attention':
                self.fus_fb = CrossAttentionFusion(proj_dim, P)
            else:
                self.fus_fb = MLPFusion(proj_dim, P, nit)

        # ---- Temporal ----
        if self.temp_type == 'cycle_aware':
            tc = model_cfg.get('cycle_temporal', {})
            self.temporal = CycleAwareTemporalEncoder(
                proj_dim, P, tc.get('heads', 4), tc.get('layers', 1),
                tc.get('dropout', 0.1), tc.get('future_pred', True))
        else:
            self.temporal = None

        # ---- ID heads ----
        fc = model_cfg.get('SeparateFCs', {})
        bn = model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim),
                               fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256),
                                       bn.get('class_num', 250))

        self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))
        self.collect_viz = bool(model_cfg.get('collect_visualization', False))
        self._viz_output_dir = str(model_cfg.get('viz_output_dir', './viz_output'))

    # --- helpers ---
    def _step(self):
        if self.training: self._fwd_iter += 1
        for a in ('iteration', 'global_step'):
            if hasattr(self, a):
                v = getattr(self, a)
                try: return int(v.item()) if torch.is_tensor(v) else int(v)
                except: pass
        return int(self._fwd_iter.item())

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

    def _align_ramp(self, step):
        if step < self.align_warmup: return 0.0
        if self.align_ramp <= 0: return 1.0
        return min(1.0, (step - self.align_warmup) / self.align_ramp)

    # --- forward ---
    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._step()

        rgb = self._rgb(ipts[self.rgb_index])
        pc = ipts[self.pc_index].float()

        rgb_map = self.rgb_enc(rgb)
        ga = None
        if self.pos_gate is not None:
            rgb_map, ga = self.pos_gate(rgb_map)
        rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(
            *rgb_map.shape[:2], self.channels, self.parts_num)
        pc_ps = self.pc_enc(pc)
        B, T, C, P = rgb_ps.shape

        # ================================================================
        #  Fix-11: Part-Level Align-Before-Fuse
        #  Cross-modal attention bridge + alignment projections
        # ================================================================
        rgb_aligned, pc_aligned, rgb_align_feat, pc_align_feat = \
            self.plabf(rgb_ps, pc_ps)
        # rgb_aligned/pc_aligned: [B,T,C,P] — enhanced features for downstream
        # rgb_align_feat/pc_align_feat: [B,P,D_align] — for contrastive loss

        # Fix-5: MQE (on aligned features)
        rgb_global = rgb_aligned.mean(dim=(1, 3))
        pc_global = pc_aligned.mean(dim=(1, 3))
        rgb_quality, pc_quality = self.mqe(rgb_global, pc_global)
        quality_scores = torch.cat([rgb_quality, pc_quality], dim=-1)

        # Nuisance pre (on aligned RGB)
        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(
                rgb_aligned.mean((1, 3)), tfs, mks, step)
            rgb_aligned = rgb_aligned * rel

        # Flatten for MoE
        rf = rgb_aligned.reshape(B * T, C, P)
        pf = pc_aligned.reshape(B * T, C, P)

        al, aw, sl, sw = None, None, None, None
        if self.app_moe is not None: rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None: pf, sl, sw = self.str_moe(pf, seq_len=T)

        rp = self.rgb_proj(rf)
        pp = self.pc_proj(pf)
        rp, pp = self.modality_dropout(rp, pp)
        tokens = torch.cat([rp.permute(0, 2, 1), pp.permute(0, 2, 1)], dim=1)

        # Condition + Quality-Aware Fusion
        fl, fw = None, None
        cond_loss = tokens.new_tensor(0.0)
        if self.fus_moe is not None:
            if self.cond_enc is not None:
                cs = compute_condition_stats(rgb, pc)
                ce, il_l, dl_l, ql_l = self.cond_enc(
                    cs, rgb_global.detach(), pc_global.detach(),
                    rgb_quality.detach(), pc_quality.detach())
                if self.training and self.cond_w > 0:
                    il_lbl, dl_lbl, ql_lbl = EnhancedConditionEncoder.pseudo_labels(
                        cs, rgb_quality.detach(), pc_quality.detach(),
                        n_illum=self.cond_enc.num_illum_bins,
                        n_dist=self.cond_enc.num_dist_bins)
                    cond_loss = (F.cross_entropy(il_l, il_lbl) +
                                 F.cross_entropy(dl_l, dl_lbl) +
                                 F.cross_entropy(ql_l, ql_lbl)) * self.cond_w
            else:
                ce = tokens.new_zeros(B, 0)
            fused, fl, fw = self.fus_moe(tokens, ce, quality_scores, seq_len=T)
        else:
            fused, _, _ = self.fus_fb(tokens, seq_len=T)

        Cp = fused.size(1)
        fused_seq = fused.reshape(B, T, Cp, P)
        fut_l = fused.new_tensor(0.0)
        if self.temporal is not None:
            feat, fut_l, _ = self.temporal(fused_seq)
        else:
            feat = fused_seq.max(1).values

        e1 = self.FCs(feat)
        e2, logits = self.BNNecks(e1)

        # ---- Losses ----
        tf = {'triplet': {'embeddings': e1, 'labels': labs},
              'softmax': {'logits': logits, 'labels': labs}}
        aux = {}

        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(
                rgb_aligned.mean((1, 3)), pc_aligned.mean((1, 3))) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = fut_l * self.fut_w

        if self.training:
            # Fix-11: cross-modal alignment loss (ramped)
            ar = self._align_ramp(step)
            if self.align_w > 0 and ar > 0:
                a_loss = cross_modal_alignment_loss(
                    rgb_align_feat, pc_align_feat, labs,
                    self.align_plc_w, self.align_ims_w, self.align_temp)
                aux['align'] = a_loss * self.align_w * ar

            # MoE losses (Fix-8)
            if self.bal_w > 0:
                bl = sum(moe_load_balance_loss_v2(l, 2) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt: aux['moe_bal'] = bl / cnt * self.bal_w
            if self.z_w > 0:
                zl = sum(moe_z_loss(l) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt: aux['moe_z'] = zl / cnt * self.z_w
            if self.div_w > 0:
                dv = e1.new_tensor(0.0); cnt = 0
                if self.app_moe is not None: dv = dv + moe_diversity_loss(self.app_moe.experts, thr=self.div_thr); cnt += 1
                if self.str_moe is not None: dv = dv + moe_diversity_loss(self.str_moe.experts, thr=self.div_thr); cnt += 1
                if self.fus_moe is not None: dv = dv + moe_diversity_loss(self.fus_moe.experts, key='token_logit', thr=self.div_thr); cnt += 1
                if cnt: aux['moe_div'] = dv / cnt * self.div_w
            if cond_loss.abs() > 0: aux['cond'] = cond_loss
            if self.mqe_w > 0:
                aux['mqe'] = modality_quality_contrastive_loss(
                    rgb_quality, pc_quality, compute_condition_stats(rgb, pc)) * self.mqe_w

        if ga is not None:
            vl = self.verifier(ga, rgb)
            if vl.requires_grad or vl.item() != 0: aux['verifier'] = vl
        if self.nuis is not None:
            id_g = self.post_nuisance_adapter(feat.mean(-1).float())
            npost = self.nuis.post_temporal(id_g, tfs, mks, step)
            aux.update({f'n_{k}': v for k, v in npre.items()})
            aux.update({f'n_{k}': v for k, v in npost.items()})

        if self.scalar_aux:
            tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
        else:
            for k, v in aux.items(): tf[f'loss_{k}'] = v

        # ---- Visual summary ----
        vs = {}
        for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        vs['scalar/rgb_quality_mean'] = rgb_quality.mean().detach()
        vs['scalar/pc_quality_mean'] = pc_quality.mean().detach()
        vs['scalar/align_ramp'] = e1.new_tensor(self._align_ramp(step))

        if self.training and step % 100 == 0:
            for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
                if logit is not None:
                    stats = expert_utilization_stats(logit, 2)
                    vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(stats['balance_ratio'])
                    for i, u in enumerate(stats['usage']):
                        vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())
            # if step % 400 == 0:
            #     try:
            #         import torch.distributed as dist
            #         is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            #     except Exception:
            #         is_rank0 = True
            #     if is_rank0:
            #         try:
            #             from .gait_moe_visualize import format_expert_usage_for_log
            #             msg = format_expert_usage_for_log(vs)
            #             if msg: print(f"\n[MoE Expert Usage] Iter {step:05d}\n{msg}", flush=True)
            #         except Exception:
            #             pass
        if self.nuis is not None and self.training:
            for k, v in ndiag.items(): vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

        # ---- Inference feat (only tensors) ----
        inference_data = {'embeddings': e1}
        if self.collect_viz and not self.training:
            if al is not None: inference_data['viz_app_logits'] = al.detach()
            if sl is not None: inference_data['viz_str_logits'] = sl.detach()
            if fl is not None: inference_data['viz_fus_logits'] = fl.detach()
            inference_data['viz_rgb_quality'] = rgb_quality.detach()
            inference_data['viz_pc_quality'] = pc_quality.detach()

        return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': inference_data}

    # ---- Auto-save viz after eval ----
    def inference(self, rank):
        info_dict = super().inference(rank)
        if not self.collect_viz: return info_dict
        try:
            import torch.distributed as dist
            is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            is_rank0 = True
        if is_rank0: self._save_and_plot_viz(info_dict)
        return info_dict

    @torch.no_grad()
    def _save_and_plot_viz(self, info_dict):
        viz_dir = self._viz_output_dir
        os.makedirs(viz_dir, exist_ok=True)
        viz_data = {}
        for key in ['viz_app_logits', 'viz_str_logits', 'viz_fus_logits',
                     'viz_rgb_quality', 'viz_pc_quality']:
            if key in info_dict:
                viz_data[key] = torch.as_tensor(info_dict[key])
        if self.app_moe is not None:
            d = {}
            for i, e in enumerate(self.app_moe.experts):
                if hasattr(e, 'part_logit'):
                    d[f'expert_{i}'] = torch.sigmoid(e.part_logit).detach().cpu().numpy()
            viz_data['app_part_attention'] = d
        if self.fus_moe is not None:
            d = {}
            for i, e in enumerate(self.fus_moe.experts):
                if hasattr(e, 'token_logit'):
                    logit = e.token_logit.detach().cpu()
                    half = logit.size(0) // 2
                    attn = torch.sigmoid(logit)
                    d[f'expert_{i}'] = {
                        'rgb_mean': float(attn[:half].mean()), 'pc_mean': float(attn[half:].mean()),
                        'rgb_per_part': attn[:half].numpy(), 'pc_per_part': attn[half:].numpy(),
                        'preference': 'RGB' if attn[:half].mean() > attn[half:].mean() else 'PC',
                        'ratio': float(attn[:half].mean() / max(attn[half:].mean(), 1e-8)),
                    }
            viz_data['fus_token_preference'] = d
        save_path = os.path.join(viz_dir, 'viz_data.pth')
        torch.save(viz_data, save_path)
        print(f"\n[Visualization] Saved → {save_path}")
        try:
            from .gait_moe_visualize import (
                compute_expert_usage_from_logits, print_expert_usage_table,
                plot_expert_usage, plot_part_attention_heatmap, plot_fusion_token_preference)
            usage = {}
            for tag in ['app', 'str', 'fus']:
                k = f'viz_{tag}_logits'
                if k in viz_data and isinstance(viz_data[k], torch.Tensor):
                    usage[tag] = compute_expert_usage_from_logits(viz_data[k], top_k=2)
            if usage:
                print_expert_usage_table(usage, title="Test-Set Expert Usage")
                plot_expert_usage(usage, os.path.join(viz_dir, 'expert_usage.png'))
            if viz_data.get('app_part_attention'):
                plot_part_attention_heatmap(viz_data['app_part_attention'],
                    os.path.join(viz_dir, 'part_attention.png'), parts_num=self.parts_num)
            if viz_data.get('fus_token_preference'):
                plot_fusion_token_preference(viz_data['fus_token_preference'],
                    os.path.join(viz_dir, 'fusion_token_preference.png'))
            print(f"[Visualization] Plots saved in {viz_dir}/\n")
        except Exception as e:
            print(f"[Visualization] Plot failed: {e}")
