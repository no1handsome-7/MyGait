
"""PACE-Gait: Prompt-guided Adaptive Cross-modal Experts for long-range gait.

This is a cleaned, ablation-friendly replacement for ``GaitMoEOracleV2Max``.
It keeps the proven V2Max blocks but exposes one canonical set of paper-ready
switches. Legacy V2Max switch aliases are intentionally not accepted.

Ablation behavior:

- text_guided_rgb_attention_enabled: PositiveTextGate on RGB feature maps.
- text_guided_disentanglement_enabled: StagedNuisanceRemoval.
- cross_modal_alignment_enabled: PLABF + alignment loss; off = identity passthrough.
- appearance_expert_enabled / structure_expert_enabled: off = encoder feature passthrough.
- fusion_expert_enabled: off = CrossAttentionFusion fallback.
- learnable_quality_enabled: off = no quality in router/expert/dropout losses.
- condition_token_enabled: off = no condition token/embedding in router.
- temporal_module_enabled: off = max pooling over time.
"""

import os
import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

from .gait_moe_blocks import (
    TinyDenseRGBEncoder, PointCloudSimpleEncoder, LidarGaitPPEncoder,
    PositiveTextGate, StagedNuisanceRemoval,
    CrossAttentionFusion, MLPFusion,
    CycleAwareTemporalEncoder, RuleBasedVerifier,
    symmetric_infonce, compute_condition_stats, moe_diversity_loss,
)
from .gait_moe_blocks_v2plus import (
    moe_z_loss, moe_load_balance_loss_v2, expert_utilization_stats,
)
from .gait_moe_blocks_v2pro import PartLevelAlignBeforeFuse
from .gait_moe_blocks_v2max import (
    HeterogeneousMoELayer, LightTemporalAgg, BranchAuxHead,
    cross_modal_alignment_loss_v2,
    identity_nuisance_orthogonality_loss,
    heterogeneous_diversity_loss,
    default_appearance_expert_configs, default_structure_expert_configs,
    default_fusion_expert_ranks,
    QualityContextTokens, ContextAwareModalityDropout,
    context_cross_modal_consistency_loss, context_diversity_loss,
)
from .pace_gait_blocks import (
    ConfigurableHeterogeneousFusionMoELayer,
    collect_part_attention, collect_fusion_token_preference,
)


class PACEGait(BaseModel):
    """Prompt-guided Adaptive Cross-modal Experts for gait recognition."""

    def build_network(self, model_cfg):
        C = int(model_cfg.get('channels', 256))
        P = int(model_cfg.get('parts_num', 16))
        proj_dim = int(model_cfg.get('proj_dim', C))
        self.channels, self.parts_num, self.proj_dim = C, P, proj_dim
        self.rgb_index = model_cfg.get('rgb_index', 1)
        self.pc_index = model_cfg.get('pc_index', 0)
        self.nuisance_indices = model_cfg.get('nuisance_indices', {})
        self.nuisance_mask_indices = model_cfg.get('nuisance_mask_indices', {})
        self.register_buffer('_fwd_iter', torch.zeros((), dtype=torch.long), persistent=False)

        abl = model_cfg.get('ablation', {})
        allowed_ablation_keys = {
            # Module switches used for paper ablations.
            'text_guided_rgb_attention_enabled',
            'text_guided_disentanglement_enabled',
            'cross_modal_alignment_enabled',
            'appearance_expert_enabled',
            'structure_expert_enabled',
            'fusion_expert_enabled',
            'learnable_quality_enabled',
            'condition_token_enabled',
            'temporal_module_enabled',

            # Expert/fusion/temporal hyper-parameters.
            'appearance_num_experts', 'appearance_topk',
            'structure_num_experts', 'structure_topk',
            'fusion_num_experts', 'fusion_topk', 'fusion_expert_rank',
            'fusion_fallback', 'temporal_type',
            'expert_rank', 'expert_dropout', 'cond_dim',

            # MoE and training regularizers.
            'balance_loss_weight', 'diversity_loss_weight',
            'diversity_threshold', 'z_loss_weight', 'modality_dropout',
            'shared_expert_enabled', 'shared_expert_rank',
            'heterogeneous_experts_enabled', 'branch_aux_enabled',
            'branch_aux_weight', 'branch_aux_warmup_iters',
            'branch_aux_out_channels', 'pre_fusion_temporal_enabled',
            'modality_aware_router_enabled', 'hard_negative_enabled',
            'orthogonality_loss_enabled', 'orthogonality_loss_weight',
            'test_time_ensemble_enabled',
        }
        unknown_keys = sorted(set(abl.keys()) - allowed_ablation_keys)
        if unknown_keys:
            raise KeyError(
                'Unknown PACEGait ablation keys: '
                + ', '.join(unknown_keys)
                + '. Please use the canonical switch names only.'
            )

        # Canonical switch names only. No V2Max aliases are accepted here.
        self.text_rgb_on = bool(abl.get('text_guided_rgb_attention_enabled', True))
        self.text_dis_on = bool(abl.get('text_guided_disentanglement_enabled', True))
        self.cross_align_on = bool(abl.get('cross_modal_alignment_enabled', True))
        self.app_moe_on = bool(abl.get('appearance_expert_enabled', True))
        self.str_moe_on = bool(abl.get('structure_expert_enabled', True))
        self.fus_moe_on = bool(abl.get('fusion_expert_enabled', True))
        self.learnable_quality_on = bool(abl.get('learnable_quality_enabled', True))
        self.condition_token_on = bool(abl.get('condition_token_enabled', True))
        self.temporal_on = bool(abl.get('temporal_module_enabled', True))
        self.temp_type = str(abl.get('temporal_type', 'cycle_aware')) if self.temporal_on else 'max_pool'
        self.fus_fallback = str(abl.get('fusion_fallback', 'cross_attention'))

        self.shared_expert_on = bool(abl.get('shared_expert_enabled', True))
        self.hetero_expert_on = bool(abl.get('heterogeneous_experts_enabled', True))
        self.branch_aux_on = bool(abl.get('branch_aux_enabled', True))
        self.pre_fusion_temporal_on = self.temporal_on and bool(abl.get('pre_fusion_temporal_enabled', True))
        self.modality_aware_router_on = bool(abl.get('modality_aware_router_enabled', True))
        self.hard_negative_on = bool(abl.get('hard_negative_enabled', True))
        self.ortho_loss_on = bool(abl.get('orthogonality_loss_enabled', True))
        self.test_time_ensemble_on = bool(abl.get('test_time_ensemble_enabled', True))
        self.context_tokens_on = self.learnable_quality_on or self.condition_token_on

        ed = float(abl.get('expert_dropout', 0.0))
        self.bal_w = float(abl.get('balance_loss_weight', 0.02))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))
        self.z_w = float(abl.get('z_loss_weight', 0.01))
        self.mod_drop = float(abl.get('modality_dropout', 0.1))
        self.branch_aux_w = float(abl.get('branch_aux_weight', 0.3))
        self.branch_aux_warmup = int(abl.get('branch_aux_warmup_iters', 2000))
        self.ortho_w = float(abl.get('orthogonality_loss_weight', 0.05))
        cond_dim = int(abl.get('cond_dim', 64))
        self.cond_dim = cond_dim if self.condition_token_on else 0

        acfg = model_cfg.get('cross_modal_align', {})
        self.align_w = float(acfg.get('align_weight', 0.15))
        self.align_plc_w = float(acfg.get('plc_weight', 1.0))
        self.align_ims_w = float(acfg.get('ims_weight', 0.3))
        self.align_temp = float(acfg.get('temperature', 0.07))
        self.align_warmup = int(acfg.get('warmup_iters', 1000))
        self.align_ramp = int(acfg.get('ramp_iters', 5000))
        self.hard_ratio = float(acfg.get('hard_ratio', 0.5))
        align_dim = int(acfg.get('align_dim', 128))
        align_heads = int(acfg.get('num_heads', 4))

        tcfg = model_cfg.get('text_adapter', {})
        td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        tcd = int(tcfg.get('code_dim', 256))
        thd = int(tcfg.get('hidden_dim', 1024))
        tdp = float(tcfg.get('dropout', 0.1))

        self.rgb_enc = TinyDenseRGBEncoder(C, 3)
        if self.text_rgb_on:
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

        self.plabf = PartLevelAlignBeforeFuse(C, align_dim, P, align_heads, dropout=tdp) if self.cross_align_on else None

        if self.pre_fusion_temporal_on:
            ptcfg = model_cfg.get('pre_fusion_temporal', {})
            self.rgb_pre_temporal = LightTemporalAgg(C, int(ptcfg.get('kernel_size', 3)))
            self.pc_pre_temporal = LightTemporalAgg(C, int(ptcfg.get('kernel_size', 3)))
        else:
            self.rgb_pre_temporal = None
            self.pc_pre_temporal = None

        if self.text_dis_on:
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

        expert_cfgs_cfg = model_cfg.get('expert_configs', {})
        app_cfgs = expert_cfgs_cfg.get('appearance', default_appearance_expert_configs(int(abl.get('appearance_num_experts', 8))))
        str_cfgs = expert_cfgs_cfg.get('structure', default_structure_expert_configs(int(abl.get('structure_num_experts', 8))))
        shared_rank = int(abl.get('shared_expert_rank', 16))
        if self.app_moe_on:
            if self.hetero_expert_on:
                self.app_moe = HeterogeneousMoELayer(C, P, app_cfgs, int(abl.get('appearance_topk', 2)), shared_rank, ed, self.shared_expert_on)
            else:
                from .gait_moe_blocks_v2plus import TopKMoELayerV2Plus
                self.app_moe = TopKMoELayerV2Plus(C, P, int(abl.get('appearance_num_experts', 8)), int(abl.get('appearance_topk', 2)), int(abl.get('expert_rank', 16)), ed)
        else:
            self.app_moe = None
        if self.str_moe_on:
            if self.hetero_expert_on:
                self.str_moe = HeterogeneousMoELayer(C, P, str_cfgs, int(abl.get('structure_topk', 2)), shared_rank, ed, self.shared_expert_on)
            else:
                from .gait_moe_blocks_v2plus import TopKMoELayerV2Plus
                self.str_moe = TopKMoELayerV2Plus(C, P, int(abl.get('structure_num_experts', 8)), int(abl.get('structure_topk', 2)), int(abl.get('expert_rank', 16)), ed)
        else:
            self.str_moe = None

        self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity()

        if self.branch_aux_on:
            aux_out = int(abl.get('branch_aux_out_channels', 256))
            aux_cls = int(model_cfg.get('SeparateBNNecks', {}).get('class_num', 250))
            self.rgb_aux_head = BranchAuxHead(C, aux_out, P, aux_cls)
            self.pc_aux_head = BranchAuxHead(C, aux_out, P, aux_cls)
        else:
            self.rgb_aux_head = None
            self.pc_aux_head = None

        # Quality/condition pipeline. QACT is the single implementation:
        # - learnable_quality_enabled controls whether quality scores enter fusion.
        # - condition_token_enabled controls whether condition embeddings enter fusion.
        # If both switches are off, QACT is not instantiated.
        if self.context_tokens_on:
            qact_cfg = model_cfg.get('quality_context', {})
            self.qact = QualityContextTokens(
                channels=C, num_tokens=int(qact_cfg.get('num_tokens', 4)),
                num_heads=int(qact_cfg.get('num_heads', 4)), cond_dim=cond_dim,
                dropout=float(qact_cfg.get('dropout', 0.1)))
            self.qact_consistency_w = float(qact_cfg.get('consistency_weight', 0.05))
            self.qact_diversity_w = float(qact_cfg.get('diversity_weight', 0.01))
            self.modality_dropout = ContextAwareModalityDropout(p=self.mod_drop)
        else:
            self.qact = None
            self.qact_consistency_w = 0.0
            self.qact_diversity_w = 0.0
            from .gait_moe_blocks_v2plus import ModalityDropout
            self.modality_dropout = ModalityDropout(p=self.mod_drop)

        nit = 2 * P
        fus_ranks_cfg = model_cfg.get('fusion_expert_ranks', default_fusion_expert_ranks(int(abl.get('fusion_num_experts', 8))))
        fus_rank_default = int(abl.get('fusion_expert_rank', 32))
        if self.fus_moe_on:
            if self.hetero_expert_on:
                self.fus_moe = ConfigurableHeterogeneousFusionMoELayer(
                    proj_dim, P, nit, fus_ranks_cfg, int(abl.get('fusion_topk', 2)),
                    self.cond_dim, ed, self.shared_expert_on, fus_rank_default,
                    self.modality_aware_router_on,
                    use_quality_router=self.learnable_quality_on,
                    use_quality_expert=self.learnable_quality_on)
            else:
                # Homogeneous legacy fusion always has quality slots; pass None when disabled.
                from .gait_moe_blocks_v2plus import QualityAwareFusionMoELayer
                self.fus_moe = QualityAwareFusionMoELayer(
                    proj_dim, P, nit, int(abl.get('fusion_num_experts', 8)),
                    int(abl.get('fusion_topk', 2)), self.cond_dim, fus_rank_default, ed)
        else:
            self.fus_moe = None
            self.fus_fb = CrossAttentionFusion(proj_dim, P) if self.fus_fallback == 'cross_attention' else MLPFusion(proj_dim, P, nit)

        if self.temporal_on and self.temp_type == 'cycle_aware':
            tc = model_cfg.get('cycle_temporal', {})
            self.temporal = CycleAwareTemporalEncoder(
                proj_dim, P, tc.get('heads', 4), tc.get('layers', 1), tc.get('dropout', 0.1), tc.get('future_pred', True))
        else:
            self.temporal = None

        fc = model_cfg.get('SeparateFCs', {})
        bn = model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim), fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256), bn.get('class_num', 250))
        self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))
        self.collect_viz = bool(model_cfg.get('collect_visualization', False))
        self._viz_output_dir = str(model_cfg.get('viz_output_dir', './viz_output_pacegait'))

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

    def _suml(self, losses, ref):
        total = ref.new_tensor(0.0)
        for v in losses.values():
            if v is not None:
                total = total + v
        return total

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

    def _zero_quality(self, B, ref):
        z = ref.new_zeros(B, 1)
        return z, z

    def forward(self, inputs):
        ipts, labs, typs, views, seqL = inputs
        step = self._step()
        rgb = self._rgb(ipts[self.rgb_index])
        pc = ipts[self.pc_index].float()

        rgb_map = self.rgb_enc(rgb)
        ga = None
        if self.pos_gate is not None:
            rgb_map, ga = self.pos_gate(rgb_map)
        rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(*rgb_map.shape[:2], self.channels, self.parts_num)
        pc_ps = self.pc_enc(pc)
        B, T, C, P = rgb_ps.shape

        if self.plabf is not None:
            rgb_aligned, pc_aligned, rgb_align_feat, pc_align_feat = self.plabf(rgb_ps, pc_ps)
        else:
            rgb_aligned, pc_aligned = rgb_ps, pc_ps
            rgb_align_feat, pc_align_feat = None, None

        qact_out = None
        if self.qact is not None:
            qact_out = self.qact(rgb_aligned, pc_aligned)

        if self.learnable_quality_on and qact_out is not None:
            quality_scores = qact_out['quality_scores']
            rgb_quality, pc_quality = quality_scores[:, :1], quality_scores[:, 1:]
        else:
            rgb_quality, pc_quality = self._zero_quality(B, rgb_aligned)
            quality_scores = None

        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        nuisance_probes = {}
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(rgb_aligned.mean((1, 3)), tfs, mks, step)
            rgb_aligned = rgb_aligned * rel
            if self.ortho_loss_on and self.training:
                for f in self.nuis.factors:
                    zp = self.nuis.visual_probe[f](rgb_aligned.mean((1, 3)).detach().float())
                    nuisance_probes[f] = zp

        if self.rgb_pre_temporal is not None:
            rgb_aligned = self.rgb_pre_temporal(rgb_aligned)
        if self.pc_pre_temporal is not None:
            pc_aligned = self.pc_pre_temporal(pc_aligned)

        rf = rgb_aligned.reshape(B * T, C, P)
        pf = pc_aligned.reshape(B * T, C, P)
        al = aw = sl = sw = None
        if self.app_moe is not None:
            rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None:
            pf, sl, sw = self.str_moe(pf, seq_len=T)

        rgb_aux_e1 = rgb_aux_logits = pc_aux_e1 = pc_aux_logits = None
        if self.branch_aux_on and self.rgb_aux_head is not None:
            rgb_aux_feat = rf.reshape(B, T, C, P).max(dim=1).values
            pc_aux_feat = pf.reshape(B, T, C, P).max(dim=1).values
            rgb_aux_e1, rgb_aux_logits = self.rgb_aux_head(rgb_aux_feat)
            pc_aux_e1, pc_aux_logits = self.pc_aux_head(pc_aux_feat)

        rp = self.rgb_proj(rf)
        pp = self.pc_proj(pf)
        if isinstance(self.modality_dropout, ContextAwareModalityDropout):
            rp, pp = self.modality_dropout(
                rp, pp,
                rgb_self_attn_w=qact_out.get('rgb_self_attn_w') if qact_out else None,
                pc_self_attn_w=qact_out.get('pc_self_attn_w') if qact_out else None)
        else:
            rp, pp = self.modality_dropout(rp, pp)

        tokens = torch.cat([rp.permute(0, 2, 1), pp.permute(0, 2, 1)], dim=1)
        fl = fw = None
        if self.condition_token_on and qact_out is not None:
            ce = qact_out['cond_emb']
        elif self.condition_token_on:
            ce = tokens.new_zeros(B, self.cond_dim)
        else:
            ce = tokens.new_zeros(B, 0)

        if self.fus_moe is not None:
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
        tf = {'triplet': {'embeddings': e1, 'labels': labs}, 'softmax': {'logits': logits, 'labels': labs}}
        aux = {}

        if self.branch_aux_on and self.training:
            if rgb_aux_e1 is not None and self._aux_ramp(step) > 0:
                tf['rgb_aux_triplet'] = {'embeddings': rgb_aux_e1, 'labels': labs}
                tf['rgb_aux_softmax'] = {'logits': rgb_aux_logits, 'labels': labs}
                tf['pc_aux_triplet'] = {'embeddings': pc_aux_e1, 'labels': labs}
                tf['pc_aux_softmax'] = {'logits': pc_aux_logits, 'labels': labs}

        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(rgb_aligned.mean((1, 3)), pc_aligned.mean((1, 3))) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = fut_l * self.fut_w

        if self.training:
            ar = self._align_ramp(step)
            if self.cross_align_on and self.align_w > 0 and ar > 0 and rgb_align_feat is not None:
                aux['align'] = cross_modal_alignment_loss_v2(
                    rgb_align_feat, pc_align_feat, labs,
                    self.align_plc_w, self.align_ims_w, self.align_temp,
                    self.hard_negative_on, self.hard_ratio) * self.align_w * ar
            if self.bal_w > 0:
                bl = sum(moe_load_balance_loss_v2(l, 2) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt:
                    aux['moe_bal'] = bl / cnt * self.bal_w
            if self.z_w > 0:
                zl = sum(moe_z_loss(l) for l in [al, sl, fl] if l is not None)
                cnt = sum(1 for l in [al, sl, fl] if l is not None)
                if cnt:
                    aux['moe_z'] = zl / cnt * self.z_w
            if self.div_w > 0:
                dv = e1.new_tensor(0.0); cnt = 0
                if self.app_moe is not None:
                    dv = dv + heterogeneous_diversity_loss(self.app_moe.experts, thr=self.div_thr); cnt += 1
                if self.str_moe is not None:
                    dv = dv + heterogeneous_diversity_loss(self.str_moe.experts, thr=self.div_thr); cnt += 1
                if self.fus_moe is not None:
                    dv = dv + moe_diversity_loss(self.fus_moe.experts, key='token_logit', thr=self.div_thr); cnt += 1
                if cnt:
                    aux['moe_div'] = dv / cnt * self.div_w
            if qact_out is not None:
                if self.condition_token_on and self.qact_consistency_w > 0:
                    aux['qact_consist'] = context_cross_modal_consistency_loss(qact_out['rgb_ctx'], qact_out['pc_ctx'], labs) * self.qact_consistency_w
                if self.qact_diversity_w > 0:
                    aux['qact_div'] = context_diversity_loss(qact_out['rgb_ctx'], qact_out['pc_ctx']) * self.qact_diversity_w
            if self.ortho_loss_on and self.ortho_w > 0 and nuisance_probes:
                id_g = self.post_nuisance_adapter(feat.mean(-1).float())
                aux['ortho'] = identity_nuisance_orthogonality_loss(id_g, nuisance_probes) * self.ortho_w

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

        vs = {}
        for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        vs['scalar/rgb_quality_mean'] = rgb_quality.mean().detach()
        vs['scalar/pc_quality_mean'] = pc_quality.mean().detach()
        vs['scalar/align_on'] = e1.new_tensor(float(self.cross_align_on))
        vs['scalar/align_ramp'] = e1.new_tensor(self._align_ramp(step))
        vs['scalar/quality_on'] = e1.new_tensor(float(self.learnable_quality_on))
        vs['scalar/condition_token_on'] = e1.new_tensor(float(self.condition_token_on))

        if self.training and step % 100 == 0:
            for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
                if logit is not None:
                    stats = expert_utilization_stats(logit, 2)
                    vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(stats['balance_ratio'])
                    for i, u in enumerate(stats['usage']):
                        vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())
        if self.nuis is not None and self.training:
            for k, v in ndiag.items():
                vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

        inference_data = {'embeddings': e1}
        if self.test_time_ensemble_on and not self.training and self.branch_aux_on and rgb_aux_e1 is not None:
            inference_data = {'embeddings': torch.cat([e1, rgb_aux_e1 * 0.5, pc_aux_e1 * 0.5], dim=1)}

        if self.collect_viz and not self.training:
            for key, val in [('viz_app_logits', al), ('viz_str_logits', sl), ('viz_fus_logits', fl)]:
                if val is not None:
                    inference_data[key] = val.detach()
            inference_data['viz_rgb_quality'] = rgb_quality.detach()
            inference_data['viz_pc_quality'] = pc_quality.detach()
            inference_data['viz_condition_stats'] = compute_condition_stats(rgb, pc).detach()
            if qact_out is not None:
                for k in ['rgb_ctx', 'pc_ctx', 'cond_emb', 'rgb_self_attn_w', 'pc_self_attn_w']:
                    if k in qact_out:
                        inference_data[f'viz_qact_{k}'] = qact_out[k].detach()

        return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': inference_data}

    def inference(self, rank):
        info_dict = super().inference(rank)
        if self.collect_viz:
            try:
                import torch.distributed as dist
                is_rank0 = ((not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0)
            except Exception:
                is_rank0 = True
            if is_rank0:
                self._save_and_plot_viz(info_dict)
        return info_dict

    @torch.no_grad()
    def _save_and_plot_viz(self, info_dict):
        viz_dir = self._viz_output_dir
        os.makedirs(viz_dir, exist_ok=True)
        viz_data = {}
        keys = ['viz_app_logits', 'viz_str_logits', 'viz_fus_logits',
                'viz_rgb_quality', 'viz_pc_quality', 'viz_condition_stats',
                'viz_qact_rgb_ctx', 'viz_qact_pc_ctx', 'viz_qact_cond_emb',
                'viz_qact_rgb_self_attn_w', 'viz_qact_pc_self_attn_w']
        for key in keys:
            if key in info_dict:
                viz_data[key] = torch.as_tensor(info_dict[key])
        viz_data['app_expert_info'] = collect_part_attention(self.app_moe)
        viz_data['str_expert_info'] = collect_part_attention(self.str_moe)
        viz_data['fusion_token_preference'] = collect_fusion_token_preference(self.fus_moe)
        save_path = os.path.join(viz_dir, 'pacegait_viz_data.pth')
        torch.save(viz_data, save_path)
        print(f"\n[PACE-Gait Visualization] Saved -> {save_path}")
        try:
            from .pacegait_visualize import run_all_visualizations
        except Exception:
            try:
                from tools.pacegait.pacegait_visualize import run_all_visualizations
            except Exception as exc:
                print(f"[PACE-Gait Visualization] import failed: {exc}")
                return
        try:
            run_all_visualizations(viz_data, info_dict, viz_dir, parts_num=self.parts_num)
        except Exception as exc:
            print(f"[PACE-Gait Visualization] plotting failed: {exc}")
