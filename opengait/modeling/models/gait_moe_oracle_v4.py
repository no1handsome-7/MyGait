"""GaitMoEOracleV4 — Token-level processing with learnable-token fusion.

Architecture changes from V2Plus:
  - HPP removed: spatial / point tokens preserved end-to-end.
  - AppearanceMoE & StructureMoE operate on spatial tokens [N, C, S].
  - Modality + positional encoding added before fusion.
  - Fusion replaced by LearnableTokenFusion:
        64 learnable queries → condition/quality-aware top-K selection
        → multi-layer cross-attention with RGB+PC tokens → K output tokens.
  - Visualization: spatial attention heatmaps from appearance experts,
    cross-attention maps from fusion module.

Data flow:
  RGB [B,T,3,H,W]
    → TinyDenseRGBEncoder → [B,T,C,H',W']
    → PositiveTextGate → gated feature map
    → Nuisance pre-temporal (mean-pooled input, broadcast scaling)
    → Flatten → [BT, C, H'*W']
    → AppearanceMoE (SpatialMoELayer) → [BT, C, H'*W']
    → ModalityPositionalEncoding (2D sincos + modality emb) → [BT, H'*W', C]

  PC [B,T,N,3]
    → PointCloudTokenEncoder → [B,T,C,N_pts], xyz
    → Flatten → [BT, C, N_pts]
    → StructureMoE (SpatialMoELayer) → [BT, C, N_pts]
    → ModalityPositionalEncoding (xyz proj + modality emb) → [BT, N_pts, C]

  MQE(rgb_global, pc_global) → quality [B, 2]
  EnhancedConditionEncoder(stats, globals, quality) → cond [B, cond_dim]

  ModalityDropout on projected tokens
  Concatenate → [BT, H'*W' + N_pts, C]
  LearnableTokenFusion(queries=64, top-K, cond, quality) → [BT, C, K]

  Reshape → [B, T, C, K]
  CycleAwareTemporalEncoder → [B, C, K]
  SeparateFCs(parts_num=K) → SeparateBNNecks → embeddings, logits
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks

from .gait_moe_blocks import (
    TinyDenseRGBEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    symmetric_infonce, compute_condition_stats,
    moe_diversity_loss,
)

from .gait_moe_blocks_v2plus import (
    ModalityQualityEstimator, EnhancedConditionEncoder,
    ModalityDropout,
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss,
    expert_utilization_stats,
)

from .gait_moe_blocks_v4 import (
    SpatialMoELayer,
    PointCloudTokenEncoder,
    ModalityPositionalEncoding,
    LearnableTokenFusion,
    query_balance_loss,
    query_diversity_loss,
)


class GaitMoEOracleV4(BaseModel):

    def build_network(self, model_cfg):
        C = int(model_cfg.get('channels', 256))
        self.channels = C
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_fwd_iter', torch.zeros((), dtype=torch.long),
                             persistent=False)

        abl = model_cfg.get('ablation', {})

        # ---- Feature switches ----
        self.app_moe_on = bool(abl.get('appearance_moe_enabled', True))
        self.str_moe_on = bool(abl.get('structure_moe_enabled', True))
        self.cond_on = bool(abl.get('fusion_condition_aware', True))
        self.temp_type = str(abl.get('temporal_type', 'cycle_aware'))
        self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
        self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
        er = int(abl.get('expert_rank', 16))
        ed = float(abl.get('expert_dropout', 0.0))
        cond_dim = int(abl.get('cond_dim', 64))

        # ---- Loss weights ----
        self.bal_w = float(abl.get('balance_loss_weight', 0.02))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))
        self.cond_w = float(abl.get('condition_cls_weight', 0.1))
        self.z_w = float(abl.get('z_loss_weight', 0.01))
        self.mqe_w = float(abl.get('quality_loss_weight', 0.05))
        self.mod_drop = float(abl.get('modality_dropout', 0.1))
        self.qbal_w = float(abl.get('query_balance_weight', 0.01))
        self.qdiv_w = float(abl.get('query_diversity_weight', 0.005))

        # ---- Fusion config ----
        fus_cfg = model_cfg.get('fusion', {})
        num_queries = int(fus_cfg.get('num_queries', 64))
        output_k = int(fus_cfg.get('output_k', 16))
        fus_heads = int(fus_cfg.get('num_heads', 8))
        fus_layers = int(fus_cfg.get('num_layers', 2))
        fus_dropout = float(fus_cfg.get('dropout', 0.1))
        self.output_k = output_k                # used as parts_num downstream

        # ---- Text config ----
        tcfg = model_cfg.get('text_adapter', {})
        td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        tcd = int(tcfg.get('code_dim', 256))
        thd = int(tcfg.get('hidden_dim', 1024))
        tdp = float(tcfg.get('dropout', 0.1))

        # ================================================================
        #  Build sub-modules
        # ================================================================

        # ---- RGB encoder ----
        self.rgb_enc = TinyDenseRGBEncoder(C, 3)

        # ---- Text gate ----
        if self.txt_gate_on:
            pcfg = model_cfg.get('positive_text', {})
            self.pos_gate = PositiveTextGate(
                pcfg.get('text_dim', td), C, pcfg.get('prompt_feat_path'),
                pcfg.get('num_prompts', 4), pcfg.get('trainable', False),
                pcfg.get('text_code_dim', tcd), pcfg.get('text_hidden_dim', thd),
                pcfg.get('dropout', tdp))
        else:
            self.pos_gate = None

        # ---- PC encoder (token-level, no HPP) ----
        self.pc_enc = PointCloudTokenEncoder(
            C, in_dim=int(model_cfg.get('pc_in_dim', 3)))

        # ---- Nuisance removal ----
        if self.nuis_on:
            nc = dict(model_cfg.get('nuisance_detach', {}))
            P_nuis = 16                        # internal parts_num for nuisance (unchanged)
            self.nuis = StagedNuisanceRemoval(
                C, P_nuis,
                nc.get('text_dim', td), nc.get('text_code_dim', tcd),
                nc.get('text_hidden_dim', thd), nc.get('sub_dim', 256),
                nc.get('factors'), nc.get('grl_lambda', 0.08),
                nc.get('temperature', 0.07),
                nc.get('warmup_iters', 5000), nc.get('probe_iters', 10000),
                nc.get('adv_ramp_iters', 10000),
                nc.get('reliability_start_iter', 15000),
                nc.get('reliability_ramp_iters', 10000),
                nc.get('probe_loss_weight', 0.03),
                nc.get('adv_loss_weight', 0.03),
                nc.get('mask_loss_weight', 0.015),
                nc.get('max_suppression', 0.35),
                nc.get('detach_reliability_gate', True),
                nc.get('factor_weights'), nc.get('dropout', tdp))
        else:
            self.nuis = None

        # ---- Spatial MoE layers ----
        self.app_moe = SpatialMoELayer(
            C, int(abl.get('appearance_num_experts', 8)),
            int(abl.get('appearance_topk', 2)), er, ed
        ) if self.app_moe_on else None

        self.str_moe = SpatialMoELayer(
            C, int(abl.get('structure_num_experts', 8)),
            int(abl.get('structure_topk', 2)), er, ed
        ) if self.str_moe_on else None

        # ---- Projectors (C → C, kept for optional dim change) ----
        proj_dim = int(model_cfg.get('proj_dim', C))
        self.proj_dim = proj_dim
        self.rgb_proj = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False),
            nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False),
            nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = (
            nn.Linear(proj_dim, C, bias=False)
            if proj_dim != C else nn.Identity())

        # ---- Modality Quality Estimator (Fix-5) ----
        mqe_cfg = model_cfg.get('modality_quality', {})
        self.mqe = ModalityQualityEstimator(
            C, hidden_dim=int(mqe_cfg.get('hidden_dim', 128)))

        # ---- Enhanced Condition Encoder (Fix-6) ----
        if self.cond_on:
            self.cond_enc = EnhancedConditionEncoder(
                C, cond_dim,
                hidden_dim=int(abl.get('cond_hidden_dim', 128)),
                stats_dim=6,
                num_illum_bins=int(abl.get('num_illum_bins', 5)),
                num_dist_bins=int(abl.get('num_dist_bins', 5)))
        else:
            self.cond_enc = None

        # ---- Modality Dropout (Fix-9) ----
        self.modality_dropout = ModalityDropout(p=self.mod_drop)

        # ---- Modality + Positional Encoding ----
        self.mod_pos_enc = ModalityPositionalEncoding(proj_dim)

        # ---- Learnable Token Fusion ----
        self.fusion = LearnableTokenFusion(
            proj_dim, num_queries, output_k,
            fus_heads, cond_dim if self.cond_on else 0,
            fus_layers, fus_dropout)

        # ---- Temporal ----
        if self.temp_type == 'cycle_aware':
            tc = model_cfg.get('cycle_temporal', {})
            self.temporal = CycleAwareTemporalEncoder(
                proj_dim, output_k,
                tc.get('heads', 4), tc.get('layers', 1),
                tc.get('dropout', 0.1), tc.get('future_pred', True))
        else:
            self.temporal = None

        # ---- ID heads (parts_num = output_k) ----
        fc = model_cfg.get('SeparateFCs', {})
        bn = model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(
            fc.get('parts_num', output_k),
            fc.get('in_channels', proj_dim),
            fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(
            bn.get('parts_num', output_k),
            bn.get('in_channels', 256),
            bn.get('class_num', 250))

        self.verifier = RuleBasedVerifier(
            model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))

        # Viz
        self.collect_viz = bool(model_cfg.get('collect_visualization', False))
        self._viz_output_dir = str(model_cfg.get('viz_output_dir', './viz_output_v4'))
        self.viz_num_sequences = int(model_cfg.get('viz_num_sequences', 6))
        self.viz_seed = int(model_cfg.get('viz_seed', 12345))
        self._viz_warned_missing_all_app_attn = False

    # ================================================================
    #  Helpers
    # ================================================================
    def _step(self):
        if self.training:
            self._fwd_iter += 1
        for a in ('iteration', 'global_step'):
            if hasattr(self, a):
                v = getattr(self, a)
                try:
                    return int(v.item()) if torch.is_tensor(v) else int(v)
                except Exception:
                    pass
        return int(self._fwd_iter.item())

    def _rgb(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2).expand(-1, -1, 3, -1, -1)
        elif x.dim() == 5:
            if x.shape[2] == 1:
                x = x.expand(-1, -1, 3, -1, -1)
            elif x.shape[-1] == 3 and x.shape[2] != 3:
                x = x.permute(0, 1, 4, 2, 3)
        return x.float()

    def _tf(self, x):
        if x is None:
            return None
        x = x.float()
        return x.mean(1) if x.dim() == 3 else (x if x.dim() == 2 else x.view(x.size(0), -1))

    def _opt(self, ipts, idx):
        if idx is None or (isinstance(idx, str) and idx.lower() in ('none', 'null')):
            return None
        idx = int(idx)
        return ipts[idx] if 0 <= idx < len(ipts) else None

    def _nuis_io(self, ipts):
        tf, mk = {}, {}
        for k, i in self.nuisance_indices.items():
            tf[k] = self._tf(self._opt(ipts, i))
        for k, i in self.nuisance_mask_indices.items():
            mk[k] = self._tf(self._opt(ipts, i))
        return tf, mk

    def _suml(self, ls, ref):
        t = ref.new_tensor(0.0)
        for v in ls.values():
            if v is not None:
                t = t + v
        return t


    @torch.no_grad()
    def _compute_all_app_expert_spatial_attn(self, x_first, Hf, Wf):
        """Compute all appearance experts' spatial attention on first-frame RGB tokens.

        Args:
            x_first: [B, C, S] input tokens to appearance MoE for first frame.
            Hf, Wf: spatial feature-map size, S = Hf * Wf.

        Returns:
            Tensor [B, E, S], or None if SpatialMoELayer does not expose the required API.

        Required SpatialMoELayer API:
            self.app_moe.get_all_expert_spatial_attn(x_first, Hf=Hf, Wf=Wf)
        """
        if self.app_moe is None or x_first is None:
            return None

        fn = None
        for name in (
            'get_all_expert_spatial_attn',
            'compute_all_expert_spatial_attn',
            'get_all_spatial_attn',
        ):
            if hasattr(self.app_moe, name):
                fn = getattr(self.app_moe, name)
                break

        if fn is None:
            if not self._viz_warned_missing_all_app_attn:
                print(
                    "[Visualization] WARNING: app_moe has no get_all_expert_spatial_attn(). "
                    "Only expert-usage/fusion/query/quality plots will be generated. "
                    "Add this method to SpatialMoELayer in gait_moe_blocks_v4.py "
                    "to enable all-expert RGB overlay.",
                    flush=True,
                )
                self._viz_warned_missing_all_app_attn = True
            return None

        # Be tolerant to slightly different helper signatures.
        try:
            attn = fn(x_first, Hf=Hf, Wf=Wf)
        except TypeError:
            try:
                attn = fn(x_first, Hf, Wf)
            except TypeError:
                attn = fn(x_first)

        if attn is None:
            return None

        # Accept either dict expert_i -> [B,S] or tensor [B,E,S]/[E,B,S]/[B,E,H,W].
        if isinstance(attn, dict):
            keys = sorted(attn.keys(), key=lambda k: int(str(k).split('_')[-1]) if str(k).split('_')[-1].isdigit() else str(k))
            vals = [torch.as_tensor(attn[k], device=x_first.device).float() for k in keys]
            vals = [v.reshape(v.size(0), -1) if v.dim() > 2 else v for v in vals]
            attn = torch.stack(vals, dim=1)  # [B,E,S]
        elif torch.is_tensor(attn):
            attn = attn.float()
            if attn.dim() == 4:             # [B,E,H,W]
                attn = attn.flatten(2)
            elif attn.dim() == 3:
                # Prefer [B,E,S]. If first dim equals num experts, convert [E,B,S] -> [B,E,S].
                E = len(getattr(self.app_moe, 'experts', []))
                if E > 0 and attn.size(0) == E and attn.size(1) == x_first.size(0):
                    attn = attn.permute(1, 0, 2).contiguous()
            else:
                return None
        else:
            return None

        if attn.dim() != 3:
            return None
        if attn.size(-1) != Hf * Wf:
            attn = attn.reshape(attn.size(0), attn.size(1), -1)
        return attn.detach()


    # ================================================================
    #  Forward
    # ================================================================
    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._step()

        rgb = self._rgb(ipts[self.rgb_index])               # [B, T, 3, H, W]
        pc = ipts[self.pc_index].float()                      # [B, T, N, D]

        # ---- RGB encode (keep spatial) ----
        rgb_map = self.rgb_enc(rgb)                           # [B, T, C, H', W']
        ga = None
        if self.pos_gate is not None:
            rgb_map, ga = self.pos_gate(rgb_map)
        B, T, C, Hf, Wf = rgb_map.shape
        S_rgb = Hf * Wf

        # ---- PC encode (keep per-point) ----
        pc_feat, pc_xyz = self.pc_enc(pc)                     # [B,T,C,N_pts], [B,T,N,3]
        N_pts = pc_feat.size(-1)

        # ---- Global features for MQE / Condition ----
        rgb_global = rgb_map.mean(dim=(1, 3, 4))              # [B, C]
        pc_global = pc_feat.mean(dim=(1, 3))                   # [B, C]

        # ---- Modality Quality (Fix-5) ----
        rgb_quality, pc_quality = self.mqe(rgb_global, pc_global)
        quality_scores = torch.cat([rgb_quality, pc_quality], dim=-1)   # [B, 2]

        # ---- Nuisance pre-temporal ----
        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(
                rgb_global, tfs, mks, step)
            # Broadcast reliability to spatial dims:  rel ~ [B, 1, C, 1]
            if rel.dim() == 2:
                rel = rel.unsqueeze(1).unsqueeze(-1)           # [B, 1, C, 1]
            rgb_map = rgb_map * rel.unsqueeze(-1)              # broadcast over T, H, W

        # ---- Flatten spatial ----
        rf = rgb_map.reshape(B * T, C, S_rgb)                 # [BT, C, S_rgb]
        pf = pc_feat.reshape(B * T, C, N_pts)                 # [BT, C, N_pts]
        pc_xyz_flat = pc_xyz.reshape(B * T, N_pts, 3)         # [BT, N_pts, 3]

        # For visualization: only the first RGB frame of each sequence.
        # Use the pre-MoE feature tokens so every appearance expert can be evaluated
        # on exactly the same image.
        viz_rgb_first = None
        viz_app_all_sa = None
        if self.collect_viz and not self.training:
            viz_rgb_first = rgb[:, 0].detach()                 # [B, 3, H, W]
            rf_first_for_viz = rf.reshape(B, T, C, S_rgb)[:, 0].detach()  # [B, C, S]
            viz_app_all_sa = self._compute_all_app_expert_spatial_attn(
                rf_first_for_viz, Hf, Wf)                     # [B, E, S] or None

        # ---- Appearance / Structure MoE ----
        al, aw, sl, sw = None, None, None, None
        viz_app_sa = None
        collect_sa = self.collect_viz and not self.training
        if self.app_moe is not None:
            if collect_sa:
                rf, al, aw, viz_app_sa = self.app_moe(
                    rf, seq_len=T, return_spatial_attn=True)
            else:
                rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None:
            pf, sl, sw = self.str_moe(pf, seq_len=T)

        # ---- Project ----
        rp = self.rgb_proj(rf)                                 # [BT, C', S_rgb]
        pp = self.pc_proj(pf)                                  # [BT, C', N_pts]

        # ---- Modality Dropout (Fix-9) ----
        rp, pp = self.modality_dropout(rp, pp)

        # ---- Modality + Positional Encoding ----
        rgb_tokens = self.mod_pos_enc.encode_rgb(rp, Hf, Wf)  # [BT, S_rgb, C']
        pc_tokens = self.mod_pos_enc.encode_pc(pp, pc_xyz_flat) # [BT, N_pts, C']

        # ---- Concatenate ----
        tokens = torch.cat([rgb_tokens, pc_tokens], dim=1)     # [BT, S_rgb+N_pts, C']

        # ---- Condition encoding (Fix-6) ----
        cond_loss = tokens.new_tensor(0.0)
        ce = tokens.new_zeros(B, 0)
        if self.cond_enc is not None:
            cs = compute_condition_stats(rgb, pc)
            ce, il_logits, dl_logits, ql_logits = self.cond_enc(
                cs, rgb_global.detach(), pc_global.detach(),
                rgb_quality.detach(), pc_quality.detach())
            if self.training and self.cond_w > 0:
                il_lbl, dl_lbl, ql_lbl = EnhancedConditionEncoder.pseudo_labels(
                    cs, rgb_quality.detach(), pc_quality.detach(),
                    n_illum=self.cond_enc.num_illum_bins,
                    n_dist=self.cond_enc.num_dist_bins)
                cond_loss = (
                    F.cross_entropy(il_logits, il_lbl) +
                    F.cross_entropy(dl_logits, dl_lbl) +
                    F.cross_entropy(ql_logits, ql_lbl)
                ) * self.cond_w

        # ---- Learnable Token Fusion ----
        return_ca = self.collect_viz and not self.training
        fused, q_scores, q_idx, ca_w = self.fusion(
            tokens, ce, quality_scores, seq_len=T,
            return_cross_attn=return_ca)                       # [BT, C', K]

        # ---- Temporal ----
        K = self.output_k
        Cp = fused.size(1)
        fused_seq = fused.reshape(B, T, Cp, K)
        fut_l = fused.new_tensor(0.0)
        if self.temporal is not None:
            feat, fut_l, _ = self.temporal(fused_seq)
        else:
            feat = fused_seq.max(1).values                     # [B, C', K]

        # ---- ID heads ----
        e1 = self.FCs(feat)
        e2, logits = self.BNNecks(e1)

        # ================================================================
        #  Loss computation
        # ================================================================
        tf = {
            'triplet': {'embeddings': e1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }
        aux = {}
        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(rgb_global, pc_global) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = fut_l * self.fut_w

        if self.training:
            # MoE balance / z-loss (appearance + structure only)
            if self.bal_w > 0:
                bl = sum(moe_load_balance_loss_v2(l, 2)
                         for l in [al, sl] if l is not None)
                cnt = sum(1 for l in [al, sl] if l is not None)
                if cnt:
                    aux['moe_bal'] = bl / cnt * self.bal_w

            if self.z_w > 0:
                zl = sum(moe_z_loss(l) for l in [al, sl] if l is not None)
                cnt = sum(1 for l in [al, sl] if l is not None)
                if cnt:
                    aux['moe_z'] = zl / cnt * self.z_w

            if self.div_w > 0:
                dv = e1.new_tensor(0.0)
                cnt = 0
                for moe in [self.app_moe, self.str_moe]:
                    if moe is not None:
                        # spatial experts don't have part_logit;
                        # use diversity on the se_fc weights instead
                        params = [e.down.weight.reshape(e.down.weight.size(0), -1)
                                  for e in moe.experts]
                        p = torch.stack(params, dim=0)                 # [E, rank, C]
                        p_flat = p.reshape(p.size(0), -1)
                        p_norm = F.normalize(p_flat, dim=-1)
                        sim = p_norm @ p_norm.T
                        mask = ~torch.eye(sim.size(0), device=sim.device,
                                          dtype=torch.bool)
                        excess = F.relu(sim[mask] - self.div_thr)
                        dv = dv + excess.mean()
                        cnt += 1
                if cnt:
                    aux['moe_div'] = dv / cnt * self.div_w

            # Query balance / diversity loss
            if self.qbal_w > 0 and q_scores is not None:
                aux['q_bal'] = query_balance_loss(q_scores) * self.qbal_w
            if self.qdiv_w > 0:
                aux['q_div'] = query_diversity_loss(
                    self.fusion.queries, self.div_thr) * self.qdiv_w

            if cond_loss.abs() > 0:
                aux['cond'] = cond_loss

            if self.mqe_w > 0:
                cs_for_mqe = compute_condition_stats(rgb, pc)
                aux['mqe'] = modality_quality_contrastive_loss(
                    rgb_quality, pc_quality, cs_for_mqe) * self.mqe_w

        # Verifier
        if ga is not None:
            vl = self.verifier(ga, rgb)
            if vl.requires_grad or vl.item() != 0:
                aux['verifier'] = vl

        # Nuisance post-temporal
        if self.nuis is not None:
            id_global = feat.mean(-1)
            id_global = self.post_nuisance_adapter(id_global.float())
            npost = self.nuis.post_temporal(id_global, tfs, mks, step)
            aux.update({f'n_{k}': v for k, v in npre.items()})
            aux.update({f'n_{k}': v for k, v in npost.items()})

        if self.scalar_aux:
            tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
        else:
            for k, v in aux.items():
                tf[f'loss_{k}'] = v

        # ================================================================
        #  Visual summary
        # ================================================================
        vs = {}
        for tag, w in [('app', aw), ('str', sw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        vs['scalar/rgb_quality_mean'] = rgb_quality.mean().detach()
        vs['scalar/pc_quality_mean'] = pc_quality.mean().detach()

        if self.training and step % 100 == 0:
            for tag, logit in [('app', al), ('str', sl)]:
                if logit is not None:
                    stats = expert_utilization_stats(logit, 2)
                    vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(
                        stats['balance_ratio'])
                    for i, u in enumerate(stats['usage']):
                        vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())
            # if step % 400 == 0:
            #     try:
            #         import torch.distributed as dist
            #         is_rank0 = ((not dist.is_available())
            #                     or (not dist.is_initialized())
            #                     or dist.get_rank() == 0)
            #     except Exception:
            #         is_rank0 = True
            #     if is_rank0:
            #         try:
            #             from .gait_moe_visualize import format_expert_usage_for_log
            #             msg = format_expert_usage_for_log(vs)
            #             if msg:
            #                 print(f"\n[MoE Expert Usage] Iter {step:05d}\n{msg}",
            #                       flush=True)
            #         except Exception:
            #             pass

        if self.nuis is not None and self.training:
            for k, v in ndiag.items():
                vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

        # ================================================================
        #  Inference data (tensors only for ddp_all_gather)
        # ================================================================
        inference_data = {'embeddings': e1}
        if self.collect_viz and not self.training:
            if al is not None:
                inference_data['viz_app_logits'] = al.detach()
            if sl is not None:
                inference_data['viz_str_logits'] = sl.detach()
            inference_data['viz_rgb_quality'] = rgb_quality.detach()
            inference_data['viz_pc_quality'] = pc_quality.detach()
            if q_scores is not None:
                inference_data['viz_query_scores'] = q_scores.detach()
            # Store spatial dims as small tensors for viz
            inference_data['viz_spatial_hw'] = torch.tensor(
                [Hf, Wf], device=e1.device)

            # DDP-safe tensors for the 6-random-sequence overlay.
            # These are gathered by OpenGait together with other inference tensors.
            if viz_rgb_first is not None:
                inference_data['viz_rgb_first'] = viz_rgb_first
            if viz_app_all_sa is not None:
                inference_data['viz_app_attn_all'] = viz_app_all_sa

            # Spatial attention from appearance experts → model buffer
            # (dicts, not safe for ddp_all_gather)
            if not hasattr(self, '_viz_buffer'):
                self._viz_buffer = []
            viz_dict = {}
            if viz_app_sa is not None:
                viz_dict['app_spatial_attn'] = viz_app_sa        # dict of tensors
            if ca_w is not None:
                viz_dict['fusion_cross_attn'] = ca_w.detach().cpu()
                viz_dict['S_rgb'] = S_rgb
                viz_dict['N_pts'] = N_pts
            self._viz_buffer.append(viz_dict)

        return {
            'training_feat': tf,
            'visual_summary': vs,
            'inference_feat': inference_data,
        }

    # ================================================================
    #  Auto-save viz data after evaluation
    # ================================================================
    def inference(self, rank):
        # Each evaluation should produce an independent viz_data file.
        if self.collect_viz:
            self._viz_buffer = []
        info_dict = super().inference(rank)
        if not self.collect_viz:
            return info_dict
        try:
            import torch.distributed as dist
            is_rank0 = ((not dist.is_available())
                        or (not dist.is_initialized())
                        or dist.get_rank() == 0)
        except Exception:
            is_rank0 = True
        step = self._step()
        if is_rank0:
            self._save_and_plot_viz(info_dict, step=step)
        # Avoid mixing visualization buffers across multiple validation runs.
        self._viz_buffer = []
        return info_dict

    @torch.no_grad()
    def _save_and_plot_viz(self, info_dict, step=None):
        viz_dir = self._viz_output_dir
        os.makedirs(viz_dir, exist_ok=True)

        viz_data = {}
        for key in ['viz_app_logits', 'viz_str_logits',
                     'viz_rgb_quality', 'viz_pc_quality',
                     'viz_query_scores', 'viz_spatial_hw']:
            if key in info_dict:
                v = info_dict[key]
                viz_data[key] = torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v

        # Randomly select 6 sequences from the whole gathered test set.
        # Only these 6 first-frame RGB images and their all-expert attention maps
        # are saved for overlay visualization.
        if 'viz_rgb_first' in info_dict and 'viz_app_attn_all' in info_dict:
            rgb_first = torch.as_tensor(info_dict['viz_rgb_first']).detach().cpu()
            app_attn_all = torch.as_tensor(info_dict['viz_app_attn_all']).detach().cpu()
            n = min(rgb_first.size(0), app_attn_all.size(0))
            k = min(int(self.viz_num_sequences), n)
            if k > 0:
                g = torch.Generator()
                seed = int(self.viz_seed) + (int(step) if step is not None else 0)
                g.manual_seed(seed)
                sample_idx = torch.randperm(n, generator=g)[:k]
                viz_data['viz_rgb_first'] = rgb_first[sample_idx]
                viz_data['viz_app_attn_all'] = app_attn_all[sample_idx]
                viz_data['viz_sample_indices'] = sample_idx
                print(
                    f"[Visualization] Randomly selected {k}/{n} test sequences for RGB-attention overlay: "
                    f"{sample_idx.tolist()}",
                    flush=True,
                )

        # Spatial attention from model buffer
        if hasattr(self, '_viz_buffer') and self._viz_buffer:
            all_app_sa = {}
            all_ca_w = []
            s_rgb, n_pts = None, None
            for vd in self._viz_buffer:
                if 'app_spatial_attn' in vd:
                    for ek, ev in vd['app_spatial_attn'].items():
                        all_app_sa.setdefault(ek, []).append(ev)
                if 'fusion_cross_attn' in vd:
                    all_ca_w.append(vd['fusion_cross_attn'])
                    s_rgb = vd.get('S_rgb', s_rgb)
                    n_pts = vd.get('N_pts', n_pts)
            if all_app_sa:
                merged = {}
                for ek, evs in all_app_sa.items():
                    merged[ek] = torch.cat(evs, dim=0)
                viz_data['app_spatial_attn'] = merged
            if all_ca_w:
                viz_data['fusion_cross_attn'] = torch.cat(all_ca_w, dim=0)
                if s_rgb is not None:
                    viz_data['S_rgb'] = s_rgb
                    viz_data['N_pts'] = n_pts

        step = int(step) if step is not None else int(self._step())
        prefix = f"{step}iter" if step > 0 else "eval"
        save_path = os.path.join(viz_dir, f"{prefix}_viz_data.pth")
        torch.save(viz_data, save_path)
        print(f"\n[Visualization] Saved viz data → {save_path}")

        # Generate plots in a step-specific subdirectory to avoid overwriting
        # results from previous validations.
        plot_dir = os.path.join(viz_dir, f"{prefix}_plots")
        os.makedirs(plot_dir, exist_ok=True)
        try:
            from .gait_moe_visualize_v4 import generate_v3_plots
            generate_v3_plots(viz_data, plot_dir, self)
            print(f"[Visualization] Plots saved in {plot_dir}/\n")
        except Exception as e:
            print(f"[Visualization] Plot generation failed: {e}")
