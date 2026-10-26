"""GaitMoEOracleV2Sol — CACT-driven condition-aware token routing.

= V2Pro + All 7 optimizations:
  (1) Shared Expert + Multi-scale Heterogeneous Experts (Method A+C)
  (2) Branch-level Auxiliary Supervision (Triplet + CE per branch)
  (3) Lightweight Pre-Fusion Temporal Aggregation
  (4) Modality-Aware Dynamic Router
  (5) Hard-Negative Cross-Modal Contrastive Loss
  (6) Identity-Nuisance Orthogonality Loss
  (7) Test-Time Multi-Branch Embedding Fusion

Every module has a config-level on/off switch for ablation.
"""

import os
import math
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
    moe_z_loss, moe_load_balance_loss_v2,
    modality_quality_contrastive_loss, expert_utilization_stats,
)
from .gait_moe_blocks_v2pro import (
    PartLevelAlignBeforeFuse,
)
from .gait_moe_blocks_v2max import (
    HeterogeneousMoELayer, HeterogeneousFusionMoELayer,
    LightTemporalAgg, BranchAuxHead,
    QualityWeightedModalityDropout,
    cross_modal_alignment_loss_v2,
    identity_nuisance_orthogonality_loss,
    heterogeneous_diversity_loss,
    default_appearance_expert_configs,
    default_structure_expert_configs,
    default_fusion_expert_ranks,
    QualityContextTokens,
    ContextAwareModalityDropout,
    context_cross_modal_consistency_loss,
    context_diversity_loss,
)
from .gait_moe_blocks_v2sol import (
    ConditionAwareContextTokens,
    ConditionAwareFusionMoELayer,
    ConditionContextModalityDropout,
    cact_cross_modal_consistency_loss,
    cact_token_diversity_loss,
    cact_token_expert_correlation,
)


class GaitMoEOracleV2Sol(BaseModel):

    def build_network(self, model_cfg):
        C = int(model_cfg.get('channels', 256))
        P = int(model_cfg.get('parts_num', 16))
        proj_dim = int(model_cfg.get('proj_dim', C))
        self.channels, self.parts_num = C, P
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_fwd_iter', torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.proj_dim = proj_dim

        # ---- Ablation switches ----
        abl = model_cfg.get('ablation', {})
        self.app_moe_on = bool(abl.get('appearance_moe_enabled', True))
        self.str_moe_on = bool(abl.get('structure_moe_enabled', True))
        self.fus_moe_on = bool(abl.get('fusion_moe_enabled', True))
        self.fus_fallback = str(abl.get('fusion_fallback', 'cross_attention'))
        self.cond_on = bool(abl.get('fusion_condition_aware', True))
        self.temp_type = str(abl.get('temporal_type', 'cycle_aware'))
        self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
        self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
        self.align_on = bool(abl.get('cross_modal_align_enabled', True))

        # V2Max switches
        self.shared_expert_on = bool(abl.get('shared_expert_enabled', True))
        self.hetero_expert_on = bool(abl.get('heterogeneous_experts_enabled', True))
        self.branch_aux_on = bool(abl.get('branch_aux_enabled', True))
        self.pre_fusion_temporal_on = bool(abl.get('pre_fusion_temporal_enabled', True))
        self.modality_aware_router_on = bool(abl.get('modality_aware_router_enabled', True))
        self.hard_negative_on = bool(abl.get('hard_negative_enabled', True))
        self.ortho_loss_on = bool(abl.get('orthogonality_loss_enabled', True))
        self.test_time_ensemble_on = bool(abl.get('test_time_ensemble_enabled', True))
        self.quality_dropout_on = bool(abl.get('quality_weighted_dropout_enabled', True))

        # V2Sol context controller. Default is CACT (no scalar quality).
        # context_mode: 'cact' | 'qact' | 'legacy' | 'none'
        self.context_mode = str(abl.get('context_mode',
                              'cact' if bool(abl.get('cact_enabled', True)) else 'legacy')).lower()
        if bool(abl.get('qact_enabled', False)):
            self.context_mode = 'qact'
        if bool(abl.get('no_context_enabled', False)):
            self.context_mode = 'none'
        self.cact_on = self.context_mode == 'cact'
        self.qact_on = self.context_mode == 'qact'
        self.legacy_context_on = self.context_mode == 'legacy'
        self.no_context_on = self.context_mode == 'none'
        self.qact_legacy_quality_fusion = bool(abl.get('qact_legacy_quality_fusion', True))
        # Eval-only CACT intervention for mechanism validation:
        # none | zero | mean | shuffle | cond_only
        self.cact_intervention = str(abl.get('cact_intervention', 'none')).lower()

        # Weights
        ed = float(abl.get('expert_dropout', 0.0))
        self.bal_w = float(abl.get('balance_loss_weight', 0.02))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))
        self.cond_w = float(abl.get('condition_cls_weight', 0.1))
        self.z_w = float(abl.get('z_loss_weight', 0.01))
        self.mqe_w = float(abl.get('quality_loss_weight', 0.05))
        self.mod_drop = float(abl.get('modality_dropout', 0.1))
        cond_dim = int(abl.get('cond_dim', 64))
        self.cond_dim = cond_dim
        self.branch_aux_w = float(abl.get('branch_aux_weight', 0.3))
        self.branch_aux_warmup = int(abl.get('branch_aux_warmup_iters', 2000))
        self.ortho_w = float(abl.get('orthogonality_loss_weight', 0.05))

        # ---- Alignment config ----
        acfg = model_cfg.get('cross_modal_align', {})
        self.align_w = float(acfg.get('align_weight', 0.15))
        self.align_plc_w = float(acfg.get('plc_weight', 1.0))
        self.align_ims_w = float(acfg.get('ims_weight', 0.3))
        self.align_temp = float(acfg.get('temperature', 0.07))
        align_dim = int(acfg.get('align_dim', 128))
        align_heads = int(acfg.get('num_heads', 4))
        self.align_warmup = int(acfg.get('warmup_iters', 1000))
        self.align_ramp = int(acfg.get('ramp_iters', 5000))
        self.hard_ratio = float(acfg.get('hard_ratio', 0.5))

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
                pc_cfg.get('scale_aware', True), pc_cfg.get('normalize_dp', True),
                C, P)
        else:
            self.pc_enc = PointCloudSimpleEncoder(
                C, P, int(model_cfg.get('pc_in_dim', 3)))
        self.HPP = HorizontalPoolingPyramid(bin_num=[P])

        # ---- PLABF ----
        if self.align_on:
            self.plabf = PartLevelAlignBeforeFuse(
                C, align_dim, P, align_heads, dropout=tdp)
        else:
            self.plabf = None

        # ---- Pre-fusion temporal (NEW) ----
        if self.pre_fusion_temporal_on:
            ptcfg = model_cfg.get('pre_fusion_temporal', {})
            self.rgb_pre_temporal = LightTemporalAgg(
                C, int(ptcfg.get('kernel_size', 3)))
            self.pc_pre_temporal = LightTemporalAgg(
                C, int(ptcfg.get('kernel_size', 3)))
        else:
            self.rgb_pre_temporal = None
            self.pc_pre_temporal = None

        # ---- Nuisance ----
        if self.nuis_on:
            nc = dict(model_cfg.get('nuisance_detach', {}))
            self.nuis = StagedNuisanceRemoval(
                C, P, nc.get('text_dim', td), nc.get('text_code_dim', tcd),
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

        # ---- MoE (Heterogeneous) ----
        expert_cfgs_cfg = model_cfg.get('expert_configs', {})
        app_cfgs = expert_cfgs_cfg.get('appearance',
                                       default_appearance_expert_configs(
                                           int(abl.get('appearance_num_experts', 8))))
        str_cfgs = expert_cfgs_cfg.get('structure',
                                       default_structure_expert_configs(
                                           int(abl.get('structure_num_experts', 8))))
        shared_rank = int(abl.get('shared_expert_rank', 16))

        if self.app_moe_on:
            if self.hetero_expert_on:
                self.app_moe = HeterogeneousMoELayer(
                    C, P, app_cfgs, int(abl.get('appearance_topk', 2)),
                    shared_rank, ed, self.shared_expert_on)
            else:
                # Fallback: homogeneous
                from .gait_moe_blocks_v2plus import TopKMoELayerV2Plus
                self.app_moe = TopKMoELayerV2Plus(
                    C, P, int(abl.get('appearance_num_experts', 8)),
                    int(abl.get('appearance_topk', 2)),
                    int(abl.get('expert_rank', 16)), ed)
        else:
            self.app_moe = None

        if self.str_moe_on:
            if self.hetero_expert_on:
                self.str_moe = HeterogeneousMoELayer(
                    C, P, str_cfgs, int(abl.get('structure_topk', 2)),
                    shared_rank, ed, self.shared_expert_on)
            else:
                from .gait_moe_blocks_v2plus import TopKMoELayerV2Plus
                self.str_moe = TopKMoELayerV2Plus(
                    C, P, int(abl.get('structure_num_experts', 8)),
                    int(abl.get('structure_topk', 2)),
                    int(abl.get('expert_rank', 16)), ed)
        else:
            self.str_moe = None

        # ---- Projectors ----
        self.rgb_proj = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False),
            nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(
            nn.Conv1d(C, proj_dim, 1, bias=False),
            nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = (
            nn.Linear(proj_dim, C, bias=False)
            if proj_dim != C else nn.Identity())

        # ---- Branch Auxiliary Heads (NEW) ----
        if self.branch_aux_on:
            aux_out = int(abl.get('branch_aux_out_channels', 256))
            aux_cls = int(model_cfg.get('SeparateBNNecks', {}).get('class_num', 250))
            self.rgb_aux_head = BranchAuxHead(C, aux_out, P, aux_cls)
            self.pc_aux_head = BranchAuxHead(C, aux_out, P, aux_cls)
        else:
            self.rgb_aux_head = None
            self.pc_aux_head = None

        # ---- CACT / QACT / Legacy Context Pipeline ----
        # CACT is the default V2Sol controller: no scalar quality score.
        if self.cact_on:
            cact_cfg = model_cfg.get('cact', model_cfg.get('quality_context', {}))
            self.cact = ConditionAwareContextTokens(
                channels=C,
                num_tokens=int(cact_cfg.get('num_tokens', 8)),
                num_heads=int(cact_cfg.get('num_heads', 4)),
                cond_dim=cond_dim,
                dropout=float(cact_cfg.get('dropout', 0.1)),
                cross_probe_enabled=bool(cact_cfg.get('cross_probe_enabled', True)),
                joint_refine_enabled=bool(cact_cfg.get('joint_refine_enabled', True)))
            self.cact_consistency_w = float(cact_cfg.get('consistency_weight', 0.05))
            self.cact_diversity_w = float(cact_cfg.get('diversity_weight', 0.01))
            self.cact_consistency_temp = float(cact_cfg.get('consistency_temperature', 0.07))
            self.qact = None
            self.qact_consistency_w = 0.0
            self.qact_diversity_w = 0.0
            self.mqe = None
            self.cond_enc = None
        elif self.qact_on:
            qact_cfg = model_cfg.get('quality_context', {})
            self.qact = QualityContextTokens(
                channels=C,
                num_tokens=int(qact_cfg.get('num_tokens', 8)),
                num_heads=int(qact_cfg.get('num_heads', 4)),
                cond_dim=cond_dim,
                dropout=float(qact_cfg.get('dropout', 0.1)),
                cross_probe_enabled=bool(qact_cfg.get('cross_probe_enabled', True)),
                joint_refine_enabled=bool(qact_cfg.get('joint_refine_enabled', True)))
            self.qact_consistency_w = float(qact_cfg.get('consistency_weight', 0.05))
            self.qact_diversity_w = float(qact_cfg.get('diversity_weight', 0.01))
            self.cact = None
            self.cact_consistency_w = 0.0
            self.cact_diversity_w = 0.0
            self.mqe = None
            self.cond_enc = None
        elif self.legacy_context_on:
            self.cact = None
            self.qact = None
            self.cact_consistency_w = 0.0
            self.cact_diversity_w = 0.0
            self.qact_consistency_w = 0.0
            self.qact_diversity_w = 0.0
            mqe_cfg = model_cfg.get('modality_quality', {})
            self.mqe = ModalityQualityEstimator(C, hidden_dim=int(mqe_cfg.get('hidden_dim', 128)))
            if self.cond_on:
                self.cond_enc = EnhancedConditionEncoder(
                    C, cond_dim,
                    hidden_dim=int(abl.get('cond_hidden_dim', 128)),
                    num_illum_bins=int(abl.get('num_illum_bins', 5)),
                    num_dist_bins=int(abl.get('num_dist_bins', 5)))
            else:
                self.cond_enc = None
        else:
            self.cact = None
            self.qact = None
            self.cact_consistency_w = 0.0
            self.cact_diversity_w = 0.0
            self.qact_consistency_w = 0.0
            self.qact_diversity_w = 0.0
            self.mqe = None
            self.cond_enc = None

        # ---- Modality Dropout ----
        if self.cact_on:
            self.modality_dropout = ConditionContextModalityDropout(p=self.mod_drop, cond_dim=cond_dim)
        elif self.qact_on:
            self.modality_dropout = ContextAwareModalityDropout(p=self.mod_drop)
        elif self.quality_dropout_on and self.legacy_context_on:
            self.modality_dropout = QualityWeightedModalityDropout(p=self.mod_drop, quality_weighted=True)
        else:
            from .gait_moe_blocks_v2plus import ModalityDropout
            self.modality_dropout = ModalityDropout(p=self.mod_drop)

        # ---- Fusion MoE ----
        nit = 2 * P
        fus_ranks_cfg = model_cfg.get('fusion_expert_ranks',
                                      default_fusion_expert_ranks(int(abl.get('fusion_num_experts', 8))))
        fus_rank_default = int(abl.get('fusion_expert_rank', 32))
        # CACT context consumed by router/expert: cond_emb + mean(rgb_ctx) + mean(pc_ctx).
        self.cact_fusion_context_dim = int(abl.get('fusion_context_dim', cond_dim + 2 * C))
        self.fusion_uses_legacy_quality = False

        if self.fus_moe_on:
            if self.qact_on and self.qact_legacy_quality_fusion:
                # For the legacy QACT ablation, keep the original quality-aware fusion path.
                self.fusion_uses_legacy_quality = True
                if self.hetero_expert_on:
                    self.fus_moe = HeterogeneousFusionMoELayer(
                        proj_dim, P, nit, fus_ranks_cfg,
                        int(abl.get('fusion_topk', 2)),
                        cond_dim if self.cond_on else 0, ed,
                        self.shared_expert_on, fus_rank_default,
                        self.modality_aware_router_on)
                else:
                    from .gait_moe_blocks_v2plus import QualityAwareFusionMoELayer
                    self.fus_moe = QualityAwareFusionMoELayer(
                        proj_dim, P, nit,
                        int(abl.get('fusion_num_experts', 8)),
                        int(abl.get('fusion_topk', 2)),
                        cond_dim if self.cond_on else 0,
                        fus_rank_default, ed)
            else:
                if self.hetero_expert_on:
                    ranks = fus_ranks_cfg
                else:
                    ranks = [fus_rank_default] * int(abl.get('fusion_num_experts', 8))
                self.fus_moe = ConditionAwareFusionMoELayer(
                    proj_dim, P, nit, ranks,
                    int(abl.get('fusion_topk', 2)),
                    cond_dim if self.cond_on else 0,
                    self.cact_fusion_context_dim if self.cond_on else 0,
                    ed, self.shared_expert_on, fus_rank_default,
                    self.modality_aware_router_on)
        else:
            self.fus_moe = None
            if self.fus_fallback == 'cross_attention':
                self.fus_fb = CrossAttentionFusion(proj_dim, P)
            else:
                self.fus_fb = MLPFusion(proj_dim, P, nit)

        # ---- Condition-gated test-time branch ensemble ----
        self.ensemble_gate = nn.Sequential(
            nn.Linear(cond_dim, max(cond_dim, 16)), nn.GELU(), nn.Linear(max(cond_dim, 16), 2))
        self.cond_gated_ensemble_on = bool(
            abl.get('cond_gated_ensemble_enabled', True))

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
        self.FCs = SeparateFCs(
            fc.get('parts_num', P), fc.get('in_channels', proj_dim),
            fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(
            bn.get('parts_num', P), bn.get('in_channels', 256),
            bn.get('class_num', 250))

        self.verifier = RuleBasedVerifier(
            model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))
        self.collect_viz = bool(model_cfg.get('collect_visualization', False))
        self.save_inference_info = bool(model_cfg.get('save_inference_info', True))
        self.inference_info_name = str(model_cfg.get('inference_info_name', 'inference_info.pth'))
        self._viz_output_dir = str(model_cfg.get('viz_output_dir', './viz_output_v2sol'))
        self._inference_cache = {}

    # --- helpers ---
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
        if x.dim() == 3:
            return x.mean(1)
        if x.dim() == 2:
            return x
        return x.view(x.size(0), -1)

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

    def _align_ramp(self, step):
        if step < self.align_warmup:
            return 0.0
        if self.align_ramp <= 0:
            return 1.0
        return min(1.0, (step - self.align_warmup) / self.align_ramp)

    def _aux_ramp(self, step):
        if self.branch_aux_warmup <= 0:
            return 1.0
        return min(1.0, step / max(1, self.branch_aux_warmup))

    @torch.no_grad()
    def _cache_inference_batch(self, inference_data):
        """Keep an internal eval cache for t-SNE/distance visualization.

        Some OpenGait inference implementations only return embeddings and drop
        labels/views/types. V2Sol therefore caches metadata from forward() and
        saves it after super().inference(rank).
        """
        if self.training or not (self.collect_viz or self.save_inference_info):
            return
        for key in ['embeddings', 'labels', 'views', 'types', 'seqL']:
            if key not in inference_data:
                continue
            val = inference_data[key]
            if val is None:
                continue
            if torch.is_tensor(val):
                val = val.detach().cpu()
            self._inference_cache.setdefault(key, []).append(val)


    @torch.no_grad()
    def _sanitize_inference_feat(self, inference_data):
        """Keep retval["inference_feat"] compatible with OpenGait.

        OpenGait's BaseModel.inference() applies ddp_all_gather() to every value
        in inference_feat. ddp_all_gather() requires Tensor inputs, so strings,
        lists, dicts, and other metadata must be removed here and saved through
        the side cache instead.
        """
        if not isinstance(inference_data, dict):
            return inference_data
        clean = {}
        skipped = []
        for k, v in inference_data.items():
            if torch.is_tensor(v):
                clean[k] = v
            else:
                skipped.append(k)
        if skipped:
            print("[V2Sol] Removed non-Tensor keys from inference_feat before "
                  f"OpenGait all_gather: {skipped}")
        return clean

    @staticmethod
    def _is_rank0():
        try:
            import torch.distributed as dist
            return ((not dist.is_available()) or
                    (not dist.is_initialized()) or
                    dist.get_rank() == 0)
        except Exception:
            return True

    @staticmethod
    def _all_gather_object_safe(obj):
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                gathered = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered, obj)
                return gathered
        except Exception as e:
            print(f"[V2Sol] all_gather_object skipped/failed: {e}")
        return [obj]

    @staticmethod
    def _flatten_meta_values(values):
        """Flatten metadata cached as a list of per-batch objects."""
        out = []
        for v in values:
            if v is None:
                continue
            if torch.is_tensor(v):
                out.extend(v.detach().cpu().reshape(-1).tolist())
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if torch.is_tensor(x):
                        out.extend(x.detach().cpu().reshape(-1).tolist())
                    elif isinstance(x, (list, tuple)):
                        out.extend(list(x))
                    else:
                        out.append(x)
            else:
                try:
                    out.extend(list(v))
                except Exception:
                    out.append(v)
        return out


    def _apply_cact_intervention(self, cact_out):
        """Eval-only CACT causal interventions.

        These are used by configs/intervention/*.yaml to verify whether
        sample-specific CACT context really controls routing/token transmission.
        """
        if self.training or cact_out is None or self.cact_intervention in ('none', '', 'false'):
            return cact_out
        mode = self.cact_intervention
        out = dict(cact_out)
        keys = ['cond_emb', 'rgb_ctx', 'pc_ctx', 'token_energy']
        if mode == 'zero':
            for k in keys:
                if k in out and torch.is_tensor(out[k]):
                    out[k] = torch.zeros_like(out[k])
        elif mode == 'mean':
            for k in keys:
                if k in out and torch.is_tensor(out[k]):
                    out[k] = out[k].mean(dim=0, keepdim=True).expand_as(out[k])
        elif mode == 'shuffle':
            B = out['cond_emb'].size(0) if torch.is_tensor(out.get('cond_emb')) else None
            if B is not None and B > 1:
                perm = torch.randperm(B, device=out['cond_emb'].device)
                for k in keys:
                    if k in out and torch.is_tensor(out[k]) and out[k].size(0) == B:
                        out[k] = out[k][perm]
        elif mode == 'cond_only':
            for k in ['rgb_ctx', 'pc_ctx', 'token_energy']:
                if k in out and torch.is_tensor(out[k]):
                    out[k] = torch.zeros_like(out[k])
        return out

    # --- forward ---
    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._step()

        rgb = self._rgb(ipts[self.rgb_index])
        pc = ipts[self.pc_index].float()

        # ---- Encode ----
        rgb_map = self.rgb_enc(rgb)
        ga = None
        if self.pos_gate is not None:
            rgb_map, ga = self.pos_gate(rgb_map)
        rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(
            *rgb_map.shape[:2], self.channels, self.parts_num)
        pc_ps = self.pc_enc(pc)
        B, T, C, P = rgb_ps.shape

        # ---- PLABF: Align Before Fuse ----
        if self.plabf is not None:
            rgb_aligned, pc_aligned, rgb_align_feat, pc_align_feat = \
                self.plabf(rgb_ps, pc_ps)
        else:
            rgb_aligned, pc_aligned = rgb_ps, pc_ps
            rgb_align_feat, pc_align_feat = None, None

        # ---- CACT / QACT / Legacy Context Estimation ----
        cact_out, qact_out = None, None
        context_info = None
        quality_scores, rgb_quality, pc_quality = None, None, None
        ce = None
        cond_loss = rgb_aligned.new_tensor(0.0)

        if self.cact_on:
            cact_out = self.cact(rgb_aligned, pc_aligned)
            cact_out = self._apply_cact_intervention(cact_out)
            ce = cact_out['cond_emb']
            context_info = cact_out
            # CACT intentionally has no scalar quality score. These placeholders
            # keep legacy logging robust but are not consumed by V2Sol fusion.
            rgb_quality = rgb_aligned.new_zeros(B, 1)
            pc_quality = pc_aligned.new_zeros(B, 1)
        elif self.qact_on:
            qact_out = self.qact(rgb_aligned, pc_aligned)
            ce = qact_out['cond_emb']
            quality_scores = qact_out.get('quality_scores', None)
            if quality_scores is not None:
                rgb_quality = quality_scores[:, :1]
                pc_quality = quality_scores[:, 1:]
            else:
                rgb_quality = rgb_aligned.new_zeros(B, 1)
                pc_quality = pc_aligned.new_zeros(B, 1)
            context_info = {
                'cond_emb': ce,
                'rgb_ctx': qact_out.get('rgb_ctx', None),
                'pc_ctx': qact_out.get('pc_ctx', None),
            }
        elif self.legacy_context_on:
            rgb_global = rgb_aligned.mean(dim=(1, 3))
            pc_global = pc_aligned.mean(dim=(1, 3))
            rgb_quality, pc_quality = self.mqe(rgb_global, pc_global)
            quality_scores = torch.cat([rgb_quality, pc_quality], dim=-1)
        else:
            ce = rgb_aligned.new_zeros(B, self.cond_dim)
            context_info = {'cond_emb': ce}
            rgb_quality = rgb_aligned.new_zeros(B, 1)
            pc_quality = pc_aligned.new_zeros(B, 1)

        # ---- Nuisance pre ----
        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        nuisance_probes = {}  # for orthogonality loss
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(
                rgb_aligned.mean((1, 3)), tfs, mks, step)
            rgb_aligned = rgb_aligned * rel
            # Collect visual probe outputs for orthogonality loss
            if self.ortho_loss_on and self.training:
                for f in self.nuis.factors:
                    zp = self.nuis.visual_probe[f](
                        rgb_aligned.mean((1, 3)).detach().float())
                    nuisance_probes[f] = zp

        # ---- Pre-fusion temporal (NEW) ----
        if self.rgb_pre_temporal is not None:
            rgb_aligned = self.rgb_pre_temporal(rgb_aligned)
        if self.pc_pre_temporal is not None:
            pc_aligned = self.pc_pre_temporal(pc_aligned)

        # ---- Flatten for MoE ----
        rf = rgb_aligned.reshape(B * T, C, P)
        pf = pc_aligned.reshape(B * T, C, P)

        al, aw, sl, sw = None, None, None, None
        if self.app_moe is not None:
            rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None:
            pf, sl, sw = self.str_moe(pf, seq_len=T)

        # ---- Branch Auxiliary Supervision (NEW) ----
        rgb_aux_e1, rgb_aux_logits = None, None
        pc_aux_e1, pc_aux_logits = None, None
        if self.branch_aux_on and self.rgb_aux_head is not None:
            rgb_aux_feat = rf.reshape(B, T, C, P).max(dim=1).values
            rgb_aux_e1, rgb_aux_logits = self.rgb_aux_head(rgb_aux_feat)
            pc_aux_feat = pf.reshape(B, T, C, P).max(dim=1).values
            pc_aux_e1, pc_aux_logits = self.pc_aux_head(pc_aux_feat)

        # ---- Project + Modality Dropout ----
        rp = self.rgb_proj(rf)
        pp = self.pc_proj(pf)

        if isinstance(self.modality_dropout, ConditionContextModalityDropout):
            rp, pp = self.modality_dropout(rp, pp, ce, seq_len=T)
        elif isinstance(self.modality_dropout, ContextAwareModalityDropout):
            rp, pp = self.modality_dropout(
                rp, pp,
                rgb_self_attn_w=qact_out.get('rgb_self_attn_w') if qact_out else None,
                pc_self_attn_w=qact_out.get('pc_self_attn_w') if qact_out else None)
        elif isinstance(self.modality_dropout, QualityWeightedModalityDropout):
            rp, pp = self.modality_dropout(rp, pp, rgb_quality, pc_quality)
        else:
            rp, pp = self.modality_dropout(rp, pp)

        tokens = torch.cat([rp.permute(0, 2, 1), pp.permute(0, 2, 1)], dim=1)

        # ---- Condition + Fusion MoE ----
        fl, fw = None, None
        if self.legacy_context_on and self.cond_enc is not None:
            cs = compute_condition_stats(rgb, pc)
            rgb_global = rgb_aligned.mean(dim=(1, 3))
            pc_global = pc_aligned.mean(dim=(1, 3))
            ce, il_l, dl_l, ql_l = self.cond_enc(
                cs, rgb_global.detach(), pc_global.detach(),
                rgb_quality.detach(), pc_quality.detach())
            context_info = {'cond_emb': ce}
            if self.training and self.cond_w > 0:
                il_lbl, dl_lbl, ql_lbl = EnhancedConditionEncoder.pseudo_labels(
                    cs, rgb_quality.detach(), pc_quality.detach(),
                    n_illum=self.cond_enc.num_illum_bins,
                    n_dist=self.cond_enc.num_dist_bins)
                cond_loss = (F.cross_entropy(il_l, il_lbl) +
                             F.cross_entropy(dl_l, dl_lbl) +
                             F.cross_entropy(ql_l, ql_lbl)) * self.cond_w
        elif ce is None:
            ce = tokens.new_zeros(B, self.cond_dim)
            context_info = {'cond_emb': ce}

        if self.fus_moe is not None:
            if self.fusion_uses_legacy_quality:
                fused, fl, fw = self.fus_moe(tokens, ce, quality_scores, seq_len=T)
            else:
                fused, fl, fw = self.fus_moe(tokens, context_info, seq_len=T)
        else:
            fused, _, _ = self.fus_fb(tokens, seq_len=T)

        Cp = fused.size(1)
        fused_seq = fused.reshape(B, T, Cp, P)

        # ---- Temporal ----
        fut_l = fused.new_tensor(0.0)
        if self.temporal is not None:
            feat, fut_l, _ = self.temporal(fused_seq)
        else:
            feat = fused_seq.max(1).values

        # ---- ID heads ----
        e1 = self.FCs(feat)
        e2, logits = self.BNNecks(e1)

        # ============================================================
        #  Losses
        # ============================================================
        tf = {
            'triplet': {'embeddings': e1, 'labels': labs},
            'softmax': {'logits': logits, 'labels': labs},
        }
        aux = {}

        # Branch aux losses (NEW)
        if self.branch_aux_on and self.training:
            ramp = self._aux_ramp(step)
            w = self.branch_aux_w * ramp
            if rgb_aux_e1 is not None and w > 0:
                tf['rgb_aux_triplet'] = {
                    'embeddings': rgb_aux_e1, 'labels': labs}
                tf['rgb_aux_softmax'] = {
                    'logits': rgb_aux_logits, 'labels': labs}
                tf['pc_aux_triplet'] = {
                    'embeddings': pc_aux_e1, 'labels': labs}
                tf['pc_aux_softmax'] = {
                    'logits': pc_aux_logits, 'labels': labs}

        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(
                rgb_aligned.mean((1, 3)), pc_aligned.mean((1, 3))) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = fut_l * self.fut_w

        if self.training:
            # Cross-modal alignment (with hard negative)
            ar = self._align_ramp(step)
            if self.align_w > 0 and ar > 0 and rgb_align_feat is not None:
                a_loss = cross_modal_alignment_loss_v2(
                    rgb_align_feat, pc_align_feat, labs,
                    self.align_plc_w, self.align_ims_w, self.align_temp,
                    self.hard_negative_on, self.hard_ratio)
                aux['align'] = a_loss * self.align_w * ar

            # MoE losses
            if self.bal_w > 0:
                bl = sum(moe_load_balance_loss_v2(l, 2)
                         for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt:
                    aux['moe_bal'] = bl / cnt * self.bal_w

            if self.z_w > 0:
                zl = sum(moe_z_loss(l) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt:
                    aux['moe_z'] = zl / cnt * self.z_w

            if self.div_w > 0:
                dv = e1.new_tensor(0.0)
                cnt = 0
                if self.app_moe is not None:
                    dv = dv + heterogeneous_diversity_loss(
                        self.app_moe.experts, thr=self.div_thr)
                    cnt += 1
                if self.str_moe is not None:
                    dv = dv + heterogeneous_diversity_loss(
                        self.str_moe.experts, thr=self.div_thr)
                    cnt += 1
                if self.fus_moe is not None:
                    dv = dv + moe_diversity_loss(
                        self.fus_moe.experts, key='token_logit', thr=self.div_thr)
                    cnt += 1
                if cnt:
                    aux['moe_div'] = dv / cnt * self.div_w

            if self.cact_on:
                if self.cact_consistency_w > 0 and cact_out is not None:
                    aux['cact_consist'] = cact_cross_modal_consistency_loss(
                        cact_out['rgb_ctx'], cact_out['pc_ctx'], labs,
                        temperature=self.cact_consistency_temp) * self.cact_consistency_w
                if self.cact_diversity_w > 0 and cact_out is not None:
                    aux['cact_div'] = cact_token_diversity_loss(
                        cact_out['rgb_ctx'], cact_out['pc_ctx']) * self.cact_diversity_w
            elif self.qact_on:
                if self.qact_consistency_w > 0 and qact_out is not None:
                    aux['qact_consist'] = context_cross_modal_consistency_loss(
                        qact_out['rgb_ctx'], qact_out['pc_ctx'], labs) * self.qact_consistency_w
                if self.qact_diversity_w > 0 and qact_out is not None:
                    aux['qact_div'] = context_diversity_loss(
                        qact_out['rgb_ctx'], qact_out['pc_ctx']) * self.qact_diversity_w
            else:
                if cond_loss.abs() > 0:
                    aux['cond'] = cond_loss
                if self.legacy_context_on and self.mqe_w > 0:
                    aux['mqe'] = modality_quality_contrastive_loss(
                        rgb_quality, pc_quality,
                        compute_condition_stats(rgb, pc)) * self.mqe_w

            # Orthogonality loss (NEW)
            if self.ortho_loss_on and self.ortho_w > 0 and nuisance_probes:
                id_g = self.post_nuisance_adapter(feat.mean(-1).float())
                aux['ortho'] = identity_nuisance_orthogonality_loss(
                    id_g, nuisance_probes) * self.ortho_w

        if ga is not None:
            vl = self.verifier(ga, rgb)
            if vl.requires_grad or vl.item() != 0:
                aux['verifier'] = vl

        if self.nuis is not None:
            id_g = self.post_nuisance_adapter(feat.mean(-1).float())
            npost = self.nuis.post_temporal(id_g, tfs, mks, step)
            aux.update({f'n_{k}': v for k, v in npre.items()})
            aux.update({f'n_{k}': v for k, v in npost.items()})

        if self.scalar_aux:
            tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
        else:
            for k, v in aux.items():
                tf[f'loss_{k}'] = v

        # ---- Visual summary ----
        vs = {}
        for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        if rgb_quality is not None:
            vs['scalar/rgb_quality_mean'] = rgb_quality.mean().detach()
        if pc_quality is not None:
            vs['scalar/pc_quality_mean'] = pc_quality.mean().detach()
        if self.cact_on and cact_out is not None:
            vs['scalar/cact_on'] = e1.new_tensor(1.0)
            vs['scalar/cact_token_energy'] = cact_out['token_energy'].mean().detach()
        if self.qact_on and qact_out is not None:
            vs['scalar/qact_on'] = e1.new_tensor(1.0)
        vs['scalar/align_ramp'] = e1.new_tensor(self._align_ramp(step))
        if self.branch_aux_on:
            vs['scalar/branch_aux_ramp'] = e1.new_tensor(self._aux_ramp(step))

        if self.training and step % 100 == 0:
            for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
                if logit is not None:
                    stats = expert_utilization_stats(logit, 2)
                    vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(
                        stats['balance_ratio'])
                    for i, u in enumerate(stats['usage']):
                        vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())

        if self.nuis is not None and self.training:
            for k, v in ndiag.items():
                vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

        # ---- Inference feat ----
        inference_data = {'embeddings': e1}
        rgb_aux_weight, pc_aux_weight = None, None

        # Condition-gated test-time multi-branch ensemble.
        if self.test_time_ensemble_on and not self.training:
            if self.branch_aux_on and rgb_aux_e1 is not None:
                gate_input = ce if ce is not None and ce.numel() > 0 else e1.new_zeros(B, self.cond_dim)
                gate_input = gate_input.to(device=e1.device, dtype=e1.dtype)
                gate_input = gate_input[:, :self.cond_dim] if gate_input.size(-1) >= self.cond_dim else torch.cat([
                    gate_input, gate_input.new_zeros(B, self.cond_dim - gate_input.size(-1))], dim=-1)
                aux_w = torch.sigmoid(self.ensemble_gate(gate_input))
                # rgb_aux_weight, pc_aux_weight = aux_w[:, :1], aux_w[:, 1:2]
                if self.cond_gated_ensemble_on:
                    rgb_aux_weight, pc_aux_weight = aux_w[:, :1], aux_w[:, 1:2]
                else:
                    rgb_aux_weight = rgb_aux_e1.new_full((rgb_aux_e1.size(0), 1, 1), 0.5)
                    pc_aux_weight = pc_aux_e1.new_full((pc_aux_e1.size(0), 1, 1), 0.5)
                inference_data = {
                    'embeddings': torch.cat([e1, rgb_aux_e1 * rgb_aux_weight, pc_aux_e1 * pc_aux_weight], dim=1)
                }

        # Cache non-Tensor metadata for t-SNE/distance visualizations.
        # IMPORTANT: do NOT put labels/views/types/seqL into inference_data.
        # OpenGait BaseModel.inference() calls ddp_all_gather() on every value
        # inside retval["inference_feat"], and ddp_all_gather only supports Tensor.
        if (not self.training) and (self.collect_viz or self.save_inference_info):
            self._cache_inference_batch({
                'labels': labs,
                'views': views,
                'types': typs,
                'seqL': seqL,
            })

        if self.collect_viz and not self.training:
            if al is not None:
                inference_data['viz_app_logits'] = al.detach()
            if sl is not None:
                inference_data['viz_str_logits'] = sl.detach()
            if fl is not None:
                inference_data['viz_fus_logits'] = fl.detach()
            if rgb_quality is not None:
                inference_data['viz_rgb_quality'] = rgb_quality.detach()
            if pc_quality is not None:
                inference_data['viz_pc_quality'] = pc_quality.detach()
            if cact_out is not None:
                inference_data['viz_cact_cond_emb'] = cact_out['cond_emb'].detach()
                inference_data['viz_cact_rgb_ctx'] = cact_out['rgb_ctx'].detach()
                inference_data['viz_cact_pc_ctx'] = cact_out['pc_ctx'].detach()
                inference_data['viz_cact_token_energy'] = cact_out['token_energy'].detach()
                inference_data['viz_cact_rgb_attn'] = cact_out['rgb_self_attn_w'].detach()
                inference_data['viz_cact_pc_attn'] = cact_out['pc_self_attn_w'].detach()
                if cact_out.get('rgb_cross_attn_w') is not None:
                    inference_data['viz_cact_rgb_cross_attn'] = cact_out['rgb_cross_attn_w'].detach()
                if cact_out.get('pc_cross_attn_w') is not None:
                    inference_data['viz_cact_pc_cross_attn'] = cact_out['pc_cross_attn_w'].detach()
                corr = cact_token_expert_correlation(cact_out.get('token_energy'), fl)
                if corr is not None:
                    inference_data['viz_cact_token_expert_corr'] = corr.detach()
            if qact_out is not None:
                inference_data['viz_qact_cond_emb'] = qact_out['cond_emb'].detach()
                inference_data['viz_qact_rgb_attn'] = qact_out['rgb_self_attn_w'].detach()
                inference_data['viz_qact_pc_attn'] = qact_out['pc_self_attn_w'].detach()
            if rgb_aux_weight is not None:
                inference_data['viz_rgb_aux_weight'] = rgb_aux_weight.detach()
                inference_data['viz_pc_aux_weight'] = pc_aux_weight.detach()

        # Final safety guard: OpenGait inference_feat must be Tensor-only.
        inference_data = self._sanitize_inference_feat(inference_data)

        return {
            'training_feat': tf,
            'visual_summary': vs,
            'inference_feat': inference_data,
        }

    # ---- Auto-save viz / inference metadata after eval ----
    def inference(self, rank):
        # Clear stale metadata before OpenGait iterates over the test dataloader.
        self._inference_cache = {}

        info_dict = super().inference(rank)

        # All ranks must enter _save_inference_and_viz because it may call
        # distributed all_gather_object() for labels/views/types/seqL. Only rank0
        # writes files and runs plotting.
        if self.collect_viz or self.save_inference_info:
            self._save_inference_and_viz(info_dict)
        return info_dict

    @torch.no_grad()
    def _value_from_info(self, info_dict, key):
        if info_dict is None:
            return None
        if isinstance(info_dict, dict):
            if key in info_dict:
                return info_dict[key]
            for sub in ('inference_feat', 'inference_data', 'results', 'output'):
                val = info_dict.get(sub, None)
                if isinstance(val, dict) and key in val:
                    return val[key]
        return None

    @torch.no_grad()
    def _to_cpu_tree(self, obj):
        import numpy as _np
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, _np.ndarray):
            return torch.from_numpy(obj)
        if isinstance(obj, dict):
            return {k: self._to_cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = [self._to_cpu_tree(v) for v in obj]
            if out and all(torch.is_tensor(v) for v in out):
                try:
                    return torch.cat(out, dim=0)
                except Exception:
                    pass
            return out
        return obj

    @torch.no_grad()
    def _save_inference_and_viz(self, info_dict):
        """Save inference_info.pth and viz_data.pth after OpenGait inference.

        The embeddings and visualization tensors are gathered by OpenGait through
        retval["inference_feat"]. Non-Tensor metadata (labels/views/types/seqL)
        is gathered here with all_gather_object() from the per-rank side cache.
        """
        viz_dir = self._viz_output_dir
        os.makedirs(viz_dir, exist_ok=True)

        # Gather cached metadata from all ranks. This function is called by every
        # rank from inference(); only rank0 writes files after the collective.
        local_cache = getattr(self, '_inference_cache', {})
        gathered_cache = self._all_gather_object_safe(local_cache)
        merged_cache = {'labels': [], 'views': [], 'types': [], 'seqL': []}
        for cache in gathered_cache:
            if not isinstance(cache, dict):
                continue
            for key in merged_cache:
                merged_cache[key].extend(
                    self._flatten_meta_values(cache.get(key, [])))

        # Build inference_info.pth. Prefer OpenGait-gathered embeddings from
        # info_dict, and metadata from our object-gathered cache.
        info_to_save = {}
        emb = self._value_from_info(info_dict, 'embeddings')
        if emb is not None:
            info_to_save['embeddings'] = self._to_cpu_tree(emb)
        for key in ['labels', 'views', 'types', 'seqL']:
            if merged_cache.get(key):
                info_to_save[key] = merged_cache[key]
            else:
                val = self._value_from_info(info_dict, key)
                if val is not None:
                    info_to_save[key] = self._to_cpu_tree(val)

        is_rank0 = self._is_rank0()
        if is_rank0 and self.save_inference_info and info_to_save:
            save_path = os.path.join(viz_dir, self.inference_info_name)
            torch.save(info_to_save, save_path)
            print(f"\n[V2Sol] Saved inference info → {save_path}")

        # Reset cache on every rank so repeated evals do not append stale samples.
        self._inference_cache = {}

        if not self.collect_viz:
            return
        if not is_rank0:
            return

        viz_data = {}
        keys = [
            'viz_app_logits', 'viz_str_logits', 'viz_fus_logits',
            'viz_rgb_quality', 'viz_pc_quality',
            'viz_cact_cond_emb', 'viz_cact_rgb_ctx', 'viz_cact_pc_ctx',
            'viz_cact_token_energy', 'viz_cact_rgb_attn', 'viz_cact_pc_attn',
            'viz_cact_rgb_cross_attn', 'viz_cact_pc_cross_attn',
            'viz_cact_token_expert_corr',
            'viz_qact_cond_emb', 'viz_qact_rgb_attn', 'viz_qact_pc_attn',
            'viz_rgb_aux_weight', 'viz_pc_aux_weight',
        ]
        for key in keys:
            val = self._value_from_info(info_dict, key)
            if val is not None:
                viz_data[key] = self._to_cpu_tree(val)

        def _branch_expert_info(moe):
            d = {}
            if moe is not None and hasattr(moe, 'experts'):
                for i, e in enumerate(moe.experts):
                    item = {'type': type(e).__name__}
                    if hasattr(e, 'part_logit'):
                        item['part_attention'] = torch.sigmoid(
                            e.part_logit.detach()).cpu().numpy()
                    d[f'expert_{i}'] = item
            return d

        def _fusion_expert_info(moe):
            d = {}
            if moe is not None and hasattr(moe, 'experts'):
                for i, e in enumerate(moe.experts):
                    item = {'type': type(e).__name__}
                    if hasattr(e, 'token_logit'):
                        ta = torch.sigmoid(e.token_logit.detach()).cpu()
                        P = max(ta.numel() // 2, 1)
                        rgb = ta[:P].mean().item()
                        pc = ta[P:2 * P].mean().item() if ta.numel() >= 2 * P else 0.5
                        s = max(rgb + pc, 1e-8)
                        item['rgb_preference'] = rgb / s
                        item['pc_preference'] = pc / s
                        item['token_attention'] = ta.numpy()
                    d[f'expert_{i}'] = item
            return d

        viz_data['app_expert_info'] = _branch_expert_info(self.app_moe)
        viz_data['str_expert_info'] = _branch_expert_info(self.str_moe)
        viz_data['fus_expert_info'] = _fusion_expert_info(self.fus_moe)

        save_path = os.path.join(viz_dir, 'viz_data.pth')
        torch.save(viz_data, save_path)
        print(f"[V2Sol Visualization] Saved → {save_path}")
        try:
            from .gait_moe_visualize_v2sol import run_all_visualizations
            run_all_visualizations(viz_data, info_to_save or info_dict,
                                   viz_dir, parts_num=self.parts_num)
        except Exception as e:
            print(f"[V2Sol Visualization] Plot failed: {e}")

