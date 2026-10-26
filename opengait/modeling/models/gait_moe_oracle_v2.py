# """GaitMoEOracleV2 with visualization hooks — drop-in replacement.

# Identical to the original V2 except:
#   - Adds `collect_visualization` config flag
#   - In eval mode, returns expert routing, part attention, and token preference
#     data in inference_feat['visualization']
#   - Adds expert usage stats to visual_summary during training (every 100 iters)
#   - Adds format_expert_usage_for_log() for training log integration

# Usage:
#   # In config, set collect_visualization: true for eval runs
#   # Or programmatically: model.collect_viz = True

#   # During evaluation:
#   from gait_moe_visualize import VizDataCollector
#   collector = VizDataCollector()
#   for batch in eval_loader:
#       output = model(batch)
#       viz = output['inference_feat'].get('visualization', {})
#       collector.add(viz, typs=batch_typs, views=batch_views)
#   collector.generate_all_plots('./viz_output/', model=model)
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


# def _expert_utilization(logits, top_k=2):
#     """Lightweight expert utilization stats for logging."""
#     import math
#     if logits is None or logits.dim() != 2:
#         return {}
#     E = logits.size(1)
#     _, topk_idx = logits.topk(top_k, dim=-1)
#     usage = torch.zeros(E, device=logits.device)
#     for k in range(top_k):
#         for i in range(E):
#             usage[i] += (topk_idx[:, k] == i).float().sum()
#     usage = usage / (logits.size(0) * top_k)
#     entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum().item()
#     max_ent = math.log(E) if E > 1 else 1.0
#     return {
#         'usage': usage.detach().cpu(),
#         'balance_ratio': entropy / max_ent,
#     }


# class GaitMoEOracleV2Viz(BaseModel):
#     """GaitMoEOracleV2 with visualization hooks."""

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
#         self.temp_type = str(abl.get('temporal_type', 'cycle_aware'))
#         self.txt_gate_on = bool(abl.get('text_guidance_enabled', True))
#         self.nuis_on = bool(abl.get('nuisance_detach_enabled', True))
#         er = int(abl.get('expert_rank', 16))
#         ed = float(abl.get('expert_dropout', 0.0))
#         self.bal_w = float(abl.get('balance_loss_weight', 0.01))
#         self.div_w = float(abl.get('diversity_loss_weight', 0.01))
#         self.div_thr = float(abl.get('diversity_threshold', 0.5))
#         self.cond_w = float(abl.get('condition_cls_weight', 0.05))
#         cond_dim = int(abl.get('cond_dim', 32))
#         self.proj_dim = proj_dim

#         # Text
#         tcfg = model_cfg.get('text_adapter', {})
#         td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
#         tcd = int(tcfg.get('code_dim', 256))
#         thd = int(tcfg.get('hidden_dim', 1024))
#         tdp = float(tcfg.get('dropout', 0.1))

#         # Encoders
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

#         # Nuisance
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

#         # MoE layers
#         self.app_moe = TopKMoELayer(C, P, int(abl.get('appearance_num_experts', 8)),
#                                     int(abl.get('appearance_topk', 2)), er, ed) if self.app_moe_on else None
#         self.str_moe = TopKMoELayer(C, P, int(abl.get('structure_num_experts', 8)),
#                                     int(abl.get('structure_topk', 2)), er, ed) if self.str_moe_on else None

#         # Projectors
#         self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
#         self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
#         self.post_nuisance_adapter = (nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity())

#         # Condition
#         self.cond_enc = ConditionEncoder(6, cond_dim, int(abl.get('cond_hidden_dim', 64))) if self.cond_on else None

#         # Fusion
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

#         # Temporal
#         if self.temp_type == 'cycle_aware':
#             tc = model_cfg.get('cycle_temporal', {})
#             self.temporal = CycleAwareTemporalEncoder(
#                 proj_dim, P, tc.get('heads', 4), tc.get('layers', 1),
#                 tc.get('dropout', 0.1), tc.get('future_pred', True))
#         else:
#             self.temporal = None

#         # ID heads
#         fc, bn = model_cfg.get('SeparateFCs', {}), model_cfg.get('SeparateBNNecks', {})
#         self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim), fc.get('out_channels', 256))
#         self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256), bn.get('class_num', 250))

#         self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
#         self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
#         self.fut_w = float(model_cfg.get('future_weight', 0.01))
#         self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))

#         # ---- Visualization flag ----
#         self.collect_viz = bool(model_cfg.get('collect_visualization', False))

#     # --- helpers (unchanged) ---

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

#     # --- forward with visualization ---

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs
#         step = self._step()

#         rgb = self._rgb(ipts[self.rgb_index])
#         pc = ipts[self.pc_index].float()
#         rgb_map = self.rgb_enc(rgb)
#         ga = None
#         if self.pos_gate is not None:
#             rgb_map, ga = self.pos_gate(rgb_map)
#         rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(*rgb_map.shape[:2], self.channels, self.parts_num)
#         pc_ps = self.pc_enc(pc)
#         B, T, C, P = rgb_ps.shape

#         # Nuisance pre
#         npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
#         if self.nuis is not None:
#             tfs, mks = self._nuis_io(ipts)
#             rel, npre, ndiag = self.nuis.pre_temporal(rgb_ps.mean((1, 3)), tfs, mks, step)
#             rgb_ps = rgb_ps * rel

#         rf, pf = rgb_ps.reshape(B * T, C, P), pc_ps.reshape(B * T, C, P)

#         al, aw, sl, sw = None, None, None, None
#         if self.app_moe is not None:
#             rf, al, aw = self.app_moe(rf, seq_len=T)
#         if self.str_moe is not None:
#             pf, sl, sw = self.str_moe(pf, seq_len=T)

#         rp = self.rgb_proj(rf)
#         pp = self.pc_proj(pf)
#         tokens = torch.cat([rp.permute(0, 2, 1), pp.permute(0, 2, 1)], 1)

#         fl, fw = None, None
#         cond_loss = tokens.new_tensor(0.0)
#         if self.fus_moe is not None:
#             if self.cond_enc is not None:
#                 cs = compute_condition_stats(rgb, pc)
#                 ce, il, dl = self.cond_enc(cs)
#                 if self.training and self.cond_w > 0:
#                     ilbl, dlbl = ConditionEncoder.pseudo_labels(cs)
#                     cond_loss = (F.cross_entropy(il, ilbl) + F.cross_entropy(dl, dlbl)) * self.cond_w
#             else:
#                 ce = tokens.new_zeros(B, 0)
#             fused, fl, fw = self.fus_moe(tokens, ce, seq_len=T)
#         else:
#             fused, _, _ = self.fus_fb(tokens, seq_len=T)

#         Cp = fused.size(1)
#         fused_seq = fused.reshape(B, T, Cp, P)
#         fut_l = fused.new_tensor(0.0)
#         if self.temporal is not None:
#             feat, fut_l, _ = self.temporal(fused_seq)
#         else:
#             feat = fused_seq.max(1).values

#         e1 = self.FCs(feat)
#         e2, logits = self.BNNecks(e1)

#         tf = {'triplet': {'embeddings': e1, 'labels': labs},
#               'softmax': {'logits': logits, 'labels': labs}}
#         aux = {}
#         if self.anc_w > 0:
#             aux['anchor'] = symmetric_infonce(rgb_ps.mean((1, 3)), pc_ps.mean((1, 3))) * self.anc_w
#         if self.fut_w > 0:
#             aux['future'] = fut_l * self.fut_w
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
#             id_global = feat.mean(-1)
#             id_global = self.post_nuisance_adapter(id_global.float())
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

#         # ================================================================
#         #  NEW: Expert usage stats in visual_summary (training)
#         # ================================================================
#         if self.training and step % 100 == 0:
#             for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
#                 if logit is not None:
#                     stats = _expert_utilization(logit, 2)
#                     vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(stats['balance_ratio'])
#                     for i, u in enumerate(stats['usage']):
#                         vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())

#         # ================================================================
#         #  NEW: Visualization data collection (eval mode)
#         #  NOTE: viz data is stored in self._viz_buffer (NOT in
#         #  inference_feat) because OpenGait's base_model calls
#         #  ddp_all_gather on every value in inference_feat, which
#         #  requires Tensors — dicts would crash.
#         # ================================================================
#         inference_data = {'embeddings': e1}

#         if self.collect_viz and not self.training:
#             viz = {}

#             # 1) Expert routing logits & decisions
#             for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
#                 if logit is not None:
#                     _, topk_i = logit.topk(2, dim=-1)
#                     viz[f'{tag}_routing'] = topk_i.detach().cpu()
#                     viz[f'{tag}_logits'] = logit.detach().cpu()

#             # 2) Appearance expert part-attention
#             if self.app_moe is not None:
#                 app_attn = {}
#                 for i, expert in enumerate(self.app_moe.experts):
#                     if hasattr(expert, 'part_logit'):
#                         app_attn[f'expert_{i}'] = torch.sigmoid(
#                             expert.part_logit).detach().cpu()
#                 viz['app_part_attention'] = app_attn

#             # 3) Fusion expert token preference (RGB vs PC)
#             if self.fus_moe is not None:
#                 fus_pref = {}
#                 for i, expert in enumerate(self.fus_moe.experts):
#                     if hasattr(expert, 'token_logit'):
#                         logit = expert.token_logit.detach().cpu()
#                         half = logit.size(0) // 2
#                         attn = torch.sigmoid(logit)
#                         fus_pref[f'expert_{i}'] = {
#                             'rgb_preference': attn[:half].mean().item(),
#                             'pc_preference': attn[half:].mean().item(),
#                         }
#                 viz['fus_token_preference'] = fus_pref

#             # 4) Spatial gate
#             if ga is not None:
#                 viz['spatial_gate'] = ga.detach().cpu()

#             # Store in model buffer — collector reads from here
#             if not hasattr(self, '_viz_buffer'):
#                 self._viz_buffer = []
#             self._viz_buffer.append(viz)

#         return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': inference_data}




"""GaitMoEOracleV2 with visualization — drop-in replacement.

Identical to V2 except:
  - Per-sample viz tensors (routing logits, quality) in inference_feat
  - Overrides inference() to auto-save viz .pth + generate plots
  - Expert usage stats logged during training every 100/400 iters

Usage:
  # In config:  model: GaitMoEOracleV2Viz, collect_visualization: true
  # Weights fully compatible with GaitMoEOracleV2 (no new parameters)
"""

import os
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
    ConditionEncoder, CycleAwareTemporalEncoder, RuleBasedVerifier,
    symmetric_infonce, compute_condition_stats,
    moe_balance_loss, moe_diversity_loss, moe_token_diversity_loss,
)
import math


def _expert_utilization(logits, top_k=2):
    if logits is None or logits.dim() != 2:
        return {}
    E = logits.size(1)
    _, topk_idx = logits.topk(top_k, dim=-1)
    usage = torch.zeros(E, device=logits.device)
    for k in range(top_k):
        for i in range(E):
            usage[i] += (topk_idx[:, k] == i).float().sum()
    usage = usage / (logits.size(0) * top_k)
    entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum().item()
    max_ent = math.log(E) if E > 1 else 1.0
    return {'usage': usage.detach().cpu(), 'balance_ratio': entropy / max_ent}


class GaitMoEOracleV2(BaseModel):

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
        self.bal_w = float(abl.get('balance_loss_weight', 0.01))
        self.div_w = float(abl.get('diversity_loss_weight', 0.01))
        self.div_thr = float(abl.get('diversity_threshold', 0.5))
        self.cond_w = float(abl.get('condition_cls_weight', 0.05))
        cond_dim = int(abl.get('cond_dim', 32))
        self.proj_dim = proj_dim

        tcfg = model_cfg.get('text_adapter', {})
        td = int(tcfg.get('input_dim', model_cfg.get('text_dim', 3584)))
        tcd = int(tcfg.get('code_dim', 256))
        thd = int(tcfg.get('hidden_dim', 1024))
        tdp = float(tcfg.get('dropout', 0.1))

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

        self.app_moe = TopKMoELayer(C, P, int(abl.get('appearance_num_experts', 8)),
                                    int(abl.get('appearance_topk', 2)), er, ed) if self.app_moe_on else None
        self.str_moe = TopKMoELayer(C, P, int(abl.get('structure_num_experts', 8)),
                                    int(abl.get('structure_topk', 2)), er, ed) if self.str_moe_on else None

        self.rgb_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.pc_proj = nn.Sequential(nn.Conv1d(C, proj_dim, 1, bias=False), nn.BatchNorm1d(proj_dim), nn.GELU())
        self.post_nuisance_adapter = (nn.Linear(proj_dim, C, bias=False) if proj_dim != C else nn.Identity())

        self.cond_enc = ConditionEncoder(6, cond_dim, int(abl.get('cond_hidden_dim', 64))) if self.cond_on else None

        nit = 2 * P
        fus_rank = int(abl.get('fusion_expert_rank', 32))
        if self.fus_moe_on:
            self.fus_moe = FusionMoELayer(
                proj_dim, P, nit, int(abl.get('fusion_num_experts', 8)),
                int(abl.get('fusion_topk', 2)), cond_dim if self.cond_on else 0, fus_rank, ed)
        else:
            self.fus_moe = None
            if self.fus_fallback == 'cross_attention':
                self.fus_fb = CrossAttentionFusion(proj_dim, P)
            else:
                self.fus_fb = MLPFusion(proj_dim, P, nit)

        if self.temp_type == 'cycle_aware':
            tc = model_cfg.get('cycle_temporal', {})
            self.temporal = CycleAwareTemporalEncoder(
                proj_dim, P, tc.get('heads', 4), tc.get('layers', 1),
                tc.get('dropout', 0.1), tc.get('future_pred', True))
        else:
            self.temporal = None

        fc, bn = model_cfg.get('SeparateFCs', {}), model_cfg.get('SeparateBNNecks', {})
        self.FCs = SeparateFCs(fc.get('parts_num', P), fc.get('in_channels', proj_dim), fc.get('out_channels', 256))
        self.BNNecks = SeparateBNNecks(bn.get('parts_num', P), bn.get('in_channels', 256), bn.get('class_num', 250))

        self.verifier = RuleBasedVerifier(model_cfg.get('verifier', {}).get('attention_weight', 0.0))
        self.anc_w = float(model_cfg.get('rgb_anchor_weight', 0.1))
        self.fut_w = float(model_cfg.get('future_weight', 0.01))
        self.scalar_aux = bool(model_cfg.get('use_scalar_aux_loss', True))

        # ---- Viz config ----
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
        rgb_ps = self.HPP(rgb_map.flatten(0, 1)).reshape(*rgb_map.shape[:2], self.channels, self.parts_num)
        pc_ps = self.pc_enc(pc)
        B, T, C, P = rgb_ps.shape

        npre, npost, ndiag, tfs, mks = {}, {}, {}, {}, {}
        if self.nuis is not None:
            tfs, mks = self._nuis_io(ipts)
            rel, npre, ndiag = self.nuis.pre_temporal(rgb_ps.mean((1, 3)), tfs, mks, step)
            rgb_ps = rgb_ps * rel

        rf, pf = rgb_ps.reshape(B * T, C, P), pc_ps.reshape(B * T, C, P)

        al, aw, sl, sw = None, None, None, None
        if self.app_moe is not None:
            rf, al, aw = self.app_moe(rf, seq_len=T)
        if self.str_moe is not None:
            pf, sl, sw = self.str_moe(pf, seq_len=T)

        rp = self.rgb_proj(rf)
        pp = self.pc_proj(pf)
        tokens = torch.cat([rp.permute(0, 2, 1), pp.permute(0, 2, 1)], 1)

        fl, fw = None, None
        cond_loss = tokens.new_tensor(0.0)
        if self.fus_moe is not None:
            if self.cond_enc is not None:
                cs = compute_condition_stats(rgb, pc)
                ce, il, dl = self.cond_enc(cs)
                if self.training and self.cond_w > 0:
                    ilbl, dlbl = ConditionEncoder.pseudo_labels(cs)
                    cond_loss = (F.cross_entropy(il, ilbl) + F.cross_entropy(dl, dlbl)) * self.cond_w
            else:
                ce = tokens.new_zeros(B, 0)
            fused, fl, fw = self.fus_moe(tokens, ce, seq_len=T)
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

        tf = {'triplet': {'embeddings': e1, 'labels': labs},
              'softmax': {'logits': logits, 'labels': labs}}
        aux = {}
        if self.anc_w > 0:
            aux['anchor'] = symmetric_infonce(rgb_ps.mean((1, 3)), pc_ps.mean((1, 3))) * self.anc_w
        if self.fut_w > 0:
            aux['future'] = fut_l * self.fut_w
        if self.training:
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
            if cond_loss.abs() > 0: aux['cond'] = cond_loss
        if ga is not None:
            vl = self.verifier(ga, rgb)
            if vl.requires_grad or vl.item() != 0: aux['verifier'] = vl
        if self.nuis is not None:
            id_global = feat.mean(-1)
            id_global = self.post_nuisance_adapter(id_global.float())
            npost = self.nuis.post_temporal(id_global, tfs, mks, step)
            aux.update({f'n_{k}': v for k, v in npre.items()})
            aux.update({f'n_{k}': v for k, v in npost.items()})

        if self.scalar_aux:
            tf['gaitoracle_aux'] = {'loss': self._suml(aux, e1)}
        else:
            for k, v in aux.items(): tf[f'loss_{k}'] = v

        vs = {}
        for tag, w in [('app', aw), ('str', sw), ('fus', fw)]:
            if w is not None:
                for i in range(w.size(1)):
                    vs[f'scalar/{tag}_e{i}_w'] = w[:, i].mean().detach()
        if self.nuis is not None and self.training:
            for k, v in ndiag.items(): vs[f'scalar/{k}'] = v.detach() if torch.is_tensor(v) else v

        # ---- Training: expert usage stats ----
        if self.training and step % 100 == 0:
            for tag, logit in [('app', al), ('str', sl), ('fus', fl)]:
                if logit is not None:
                    stats = _expert_utilization(logit, 2)
                    vs[f'scalar/{tag}_balance_ratio'] = logit.new_tensor(stats['balance_ratio'])
                    for i, u in enumerate(stats['usage']):
                        vs[f'scalar/{tag}_usage_e{i}'] = logit.new_tensor(u.item())

            if step % 400 == 0:
                try:
                    import torch.distributed as dist
                    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    is_rank0 = True
                if is_rank0:
                    try:
                        from .gait_moe_visualize import format_expert_usage_for_log
                        msg = format_expert_usage_for_log(vs)
                        if msg:
                            print(f"\n[MoE Expert Usage] Iter {step:05d}\n{msg}", flush=True)
                    except Exception:
                        pass

        # ---- Inference: per-sample viz tensors in inference_feat ----
        inference_data = {'embeddings': e1}
        if self.collect_viz and not self.training:
            if al is not None:
                inference_data['viz_app_logits'] = al.detach()
            if sl is not None:
                inference_data['viz_str_logits'] = sl.detach()
            if fl is not None:
                inference_data['viz_fus_logits'] = fl.detach()

        return {'training_feat': tf, 'visual_summary': vs, 'inference_feat': inference_data}

    # ================================================================
    #  Override inference to auto-save viz data after evaluation
    # ================================================================
    def inference(self, rank):
        info_dict = super().inference(rank)

        if not self.collect_viz:
            return info_dict

        try:
            import torch.distributed as dist
            is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
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

        # 1) Per-sample routing logits (gathered from all batches/GPUs)
        for key in ['viz_app_logits', 'viz_str_logits', 'viz_fus_logits']:
            if key in info_dict:
                v = info_dict[key]
                viz_data[key] = torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v

        # 2) Model parameter: appearance expert part-attention
        if self.app_moe is not None:
            app_attn = {}
            for i, expert in enumerate(self.app_moe.experts):
                if hasattr(expert, 'part_logit'):
                    app_attn[f'expert_{i}'] = torch.sigmoid(expert.part_logit).detach().cpu().tolist()
            viz_data['app_part_attention'] = app_attn

        # 3) Model parameter: fusion expert token preference
        if self.fus_moe is not None:
            fus_pref = {}
            for i, expert in enumerate(self.fus_moe.experts):
                if hasattr(expert, 'token_logit'):
                    logit = expert.token_logit.detach().cpu()
                    half = logit.size(0) // 2
                    attn = torch.sigmoid(logit)
                    fus_pref[f'expert_{i}'] = {
                        'rgb_mean': float(attn[:half].mean()),
                        'pc_mean': float(attn[half:].mean()),
                        'rgb_per_part': attn[:half].tolist(),
                        'pc_per_part': attn[half:].tolist(),
                        'preference': 'RGB' if attn[:half].mean() > attn[half:].mean() else 'PC',
                        'ratio': float(attn[:half].mean() / max(attn[half:].mean(), 1e-8)),
                    }
            viz_data['fus_token_preference'] = fus_pref

        # Save .pth
        save_path = os.path.join(viz_dir, 'viz_data.pth')
        torch.save(viz_data, save_path)
        print(f"\n[Visualization] Saved viz data → {save_path}")

        # Generate plots
        try:
            from .gait_moe_visualize import (
                compute_expert_usage_from_logits, print_expert_usage_table,
                plot_expert_usage, plot_part_attention_heatmap,
                plot_fusion_token_preference,
            )
            # Expert usage
            usage = {}
            for tag in ['app', 'str', 'fus']:
                key = f'viz_{tag}_logits'
                if key in viz_data and isinstance(viz_data[key], torch.Tensor):
                    usage[tag] = compute_expert_usage_from_logits(viz_data[key], top_k=2)
            if usage:
                print_expert_usage_table(usage, title="Test-Set Expert Usage")
                plot_expert_usage(usage, os.path.join(viz_dir, 'expert_usage.png'))

            # Part attention heatmap
            if 'app_part_attention' in viz_data and viz_data['app_part_attention']:
                plot_part_attention_heatmap(
                    viz_data['app_part_attention'],
                    os.path.join(viz_dir, 'part_attention.png'),
                    parts_num=self.parts_num)

            # Fusion token preference
            if 'fus_token_preference' in viz_data and viz_data['fus_token_preference']:
                plot_fusion_token_preference(
                    viz_data['fus_token_preference'],
                    os.path.join(viz_dir, 'fusion_token_preference.png'))

            print(f"[Visualization] Plots saved in {viz_dir}/\n")
        except Exception as e:
            print(f"[Visualization] Plot generation failed: {e}")
