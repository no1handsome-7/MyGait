# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from ..base_model import BaseModel
# from ..modules import SetBlockWrapper, PackSequenceWrapper, SeparateFCs, SeparateBNNecks
# from ..modules import HorizontalPoolingPyramid
# from .lidargaitv2 import LidarGaitPlusPlusEncoder


# def _ensure_5d_video(x: torch.Tensor, channels: int) -> torch.Tensor:
#     """Normalize input to [N, C, S, H, W]."""
#     if x.dim() == 4:
#         x = x.unsqueeze(2)
#     if x.dim() != 5:
#         raise ValueError(f"Expected 4D/5D tensor, got shape={tuple(x.shape)}")
#     if x.shape[2] == channels:
#         x = rearrange(x, "n s c h w -> n c s h w")
#     elif x.shape[1] == channels:
#         pass
#     else:
#         raise ValueError(f"Channel mismatch: expected {channels}, got shape={tuple(x.shape)}")
#     return x.contiguous()


# def _ddp_safe_loss(x, device):
#     if not torch.is_tensor(x):
#         x = torch.as_tensor(x, device=device)
#     x = x.float()
#     if x.dim() == 0:
#         x = x.view(1)
#     return x


# class RunningNorm(nn.Module):
#     def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-6):
#         super().__init__()
#         self.momentum = momentum
#         self.eps = eps
#         self.register_buffer("mean", torch.zeros(dim))
#         self.register_buffer("var", torch.ones(dim))
#         self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

#     @torch.no_grad()
#     def update(self, x: torch.Tensor):
#         bmean = x.mean(dim=0)
#         bvar = x.var(dim=0, unbiased=False).clamp_min(self.eps)
#         if not bool(self.initialized.item()):
#             self.mean.copy_(bmean)
#             self.var.copy_(bvar)
#             self.initialized.fill_(True)
#         else:
#             self.mean.mul_(1.0 - self.momentum).add_(self.momentum * bmean)
#             self.var.mul_(1.0 - self.momentum).add_(self.momentum * bvar)

#     def normalize(self, x: torch.Tensor) -> torch.Tensor:
#         return (x - self.mean) / torch.sqrt(self.var + self.eps)


# class _QualityHead(nn.Module):
#     def __init__(self, in_channels: int):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels, max(in_channels // 2, 16)),
#             nn.ReLU(inplace=True),
#             nn.Linear(max(in_channels // 2, 16), 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, part_feat: torch.Tensor) -> torch.Tensor:
#         return self.mlp(part_feat.mean(dim=2))


# class _CrossAttnBlock(nn.Module):
#     def __init__(self, d: int, num_heads: int = 8, ffn_dim: int = 256, dropout: float = 0.0):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
#         self.ln1 = nn.LayerNorm(d)
#         self.ffn = nn.Sequential(
#             nn.Linear(d, ffn_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(ffn_dim, d),
#             nn.Dropout(dropout),
#         )
#         self.ln2 = nn.LayerNorm(d)

#     def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
#         x = self.ln1(q)
#         y = self.ln1(kv)
#         attn_out, _ = self.attn(query=x, key=y, value=y, need_weights=False)
#         x = q + attn_out
#         x = self.ln2(x + self.ffn(x))
#         return x


# class CarGaitReranker(nn.Module):
#     def __init__(self, d: int = 256, num_heads: int = 8, ffn_dim: int = 256, dropout: float = 0.0):
#         super().__init__()
#         self.block = _CrossAttnBlock(d=d, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)

#     @staticmethod
#     def _to_tokens(x_part: torch.Tensor) -> torch.Tensor:
#         return x_part.permute(0, 2, 1).contiguous()

#     def distance(self, q_part: torch.Tensor, g_part: torch.Tensor) -> torch.Tensor:
#         qp = self._to_tokens(q_part)
#         gp = self._to_tokens(g_part)
#         ep = self.block(qp, gp)
#         eg = self.block(gp, qp)
#         return (ep - eg).pow(2).sum(dim=-1).clamp_min(1e-12).sqrt().mean(dim=1)

#     def forward(self, q_part: torch.Tensor, g_part: torch.Tensor, *args, **kwargs) -> torch.Tensor:
#         return -self.distance(q_part, g_part)


# class PartProjector(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(out_dim),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.proj(x)


# class ConditionDPM(nn.Module):
#     """
#     Lightweight condition encoder:
#       - builds a condition embedding from handcrafted quality stats
#       - outputs biases for the first-layer 4x3 gate and second-layer 3-way gate
#     """
#     def __init__(self, stats_dim: int = 12, hidden_dim: int = 128, cond_dim: int = 16):
#         super().__init__()
#         self.norm = RunningNorm(stats_dim, momentum=0.01)
#         self.backbone = nn.Sequential(
#             nn.Linear(stats_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#         )
#         self.cond = nn.Linear(hidden_dim, cond_dim)
#         self.first_bias = nn.Linear(hidden_dim, 12)   # 4x3
#         self.second_bias = nn.Linear(hidden_dim, 3)
#         self.gate = nn.Linear(hidden_dim, 1)

#     def forward(self, stats_raw: torch.Tensor, update_norm: bool = False):
#         if self.training and update_norm:
#             self.norm.update(stats_raw.detach())
#         x = self.norm.normalize(stats_raw)
#         h = self.backbone(x)
#         return {
#             "cond_emb": self.cond(h),
#             "first_bias": self.first_bias(h).view(h.size(0), 4, 3),
#             "second_bias": self.second_bias(h),
#             "gate": torch.sigmoid(self.gate(h)),
#         }


# class FirstLayerGate(nn.Module):
#     """
#     Outputs A in R^{N x 4 x 3}
#     Each column is a modality recipe for one second-layer semantic expert.
#     Softmax is applied over the 4 modalities (dim=1).
#     """
#     def __init__(self, d: int, cond_dim: int, hidden: int = 256):
#         super().__init__()
#         in_dim = 4 * d + 4 + cond_dim
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, 12),
#         )

#     def forward(self, z_cat: torch.Tensor, avail: torch.Tensor, dpm_bias=None, dpm_gate=None):
#         logits = self.net(z_cat).view(z_cat.size(0), 4, 3)
#         if dpm_bias is not None:
#             if dpm_gate is None:
#                 dpm_gate = logits.new_ones((logits.size(0), 1))
#             logits = logits + dpm_gate.unsqueeze(1) * dpm_bias
#         mask = avail.unsqueeze(-1) > 0.0
#         minus_inf = torch.finfo(logits.dtype).min
#         logits = torch.where(mask, logits, torch.full_like(logits, minus_inf))
#         return torch.softmax(logits, dim=1)


# class PartSelfAttention(nn.Module):
#     def __init__(self, d: int, num_heads: int = 8, ffn_dim: int = 256, dropout: float = 0.0):
#         super().__init__()
#         self.block = _CrossAttnBlock(d=d, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         t = x.permute(0, 2, 1).contiguous()   # [N,P,C]
#         t = self.block(t, t)
#         return t.permute(0, 2, 1).contiguous()


# class AppearanceExpertParts(nn.Module):
#     def __init__(self, d: int, heads: int = 8, ffn_dim: int = 256, dropout: float = 0.0):
#         super().__init__()
#         self.pre = nn.Sequential(
#             nn.Conv1d(d, d, kernel_size=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#         )
#         self.attn = PartSelfAttention(d=d, num_heads=heads, ffn_dim=ffn_dim, dropout=dropout)
#         self.ffn = nn.Sequential(
#             nn.Conv1d(d, d, kernel_size=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(d, d, kernel_size=1, bias=False),
#         )
#         self.ln = nn.BatchNorm1d(d)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.pre(x)
#         h = self.attn(h)
#         h = h + self.ffn(h)
#         return self.ln(h)


# class StructureExpertParts(nn.Module):
#     def __init__(self, d: int, quality_dim: int = 4):
#         super().__init__()
#         self.gamma = nn.Linear(quality_dim, d)
#         self.beta = nn.Linear(quality_dim, d)
#         self.pre = nn.Sequential(
#             nn.Conv1d(d, d, kernel_size=1, groups=4, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(d, d, kernel_size=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#         )
#         self.attn = PartSelfAttention(d=d, num_heads=4, ffn_dim=d, dropout=0.0)
#         self.out = nn.Conv1d(d, d, kernel_size=1, bias=False)

#     def forward(self, x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
#         gamma = self.gamma(quality).unsqueeze(-1)
#         beta = self.beta(quality).unsqueeze(-1)
#         h = x * (1.0 + torch.tanh(gamma)) + beta
#         h = self.pre(h)
#         h = self.attn(h)
#         return self.out(h)


# class TemporalExpertParts(nn.Module):
#     """
#     Part-preserving temporal expert.
#     Inputs:
#       base_x:   [N,C,P]
#       sil_seq:  [N,S,C,P] from silhouette sequence branch
#     Output:
#       out:      [N,C,P]
#       temp_ctx: [N,C,P] explicit temporal cue for temporal-consistency regularization
#     """
#     def __init__(self, d: int):
#         super().__init__()
#         self.temp_conv = nn.Sequential(
#             nn.Conv1d(d, d, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(d, d, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#         )
#         self.part_attn = PartSelfAttention(d=d, num_heads=4, ffn_dim=d, dropout=0.0)
#         self.fuse = nn.Sequential(
#             nn.Conv1d(2 * d, d, kernel_size=1, bias=False),
#             nn.BatchNorm1d(d),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(d, d, kernel_size=1, bias=False),
#         )

#     def forward(self, base_x: torch.Tensor, sil_seq: torch.Tensor):
#         # sil_seq: [N,S,C,P] -> per-part temporal conv
#         n, s, c, p = sil_seq.shape
#         t = sil_seq.permute(0, 3, 2, 1).contiguous().view(n * p, c, s)   # [N*P,C,S]
#         t = self.temp_conv(t).mean(dim=2)                                 # [N*P,C]
#         temp_ctx = t.view(n, p, c).permute(0, 2, 1).contiguous()         # [N,C,P]
#         h = self.fuse(torch.cat([base_x, temp_ctx], dim=1))
#         h = self.part_attn(h)
#         return h, temp_ctx


# class SecondLayerGate(nn.Module):
#     """
#     Outputs w in R^{N x 3}; all second-layer experts participate in the decision.
#     """
#     def __init__(self, d: int, cond_dim: int, hidden: int = 256):
#         super().__init__()
#         in_dim = 3 * d + 4 + cond_dim
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, 3),
#         )

#     def forward(self, z_cat: torch.Tensor, dpm_bias=None, dpm_gate=None):
#         logits = self.net(z_cat)
#         if dpm_bias is not None:
#             if dpm_gate is None:
#                 dpm_gate = logits.new_ones((logits.size(0), 1))
#             logits = logits + dpm_gate * dpm_bias
#         return torch.softmax(logits, dim=1)


# class MultimodalHierarchicalGaitMoE(BaseModel):
#     """
#     Two-layer hierarchical MoE for multimodal gait:
#       Layer-1:
#         four modality encoders -> shared part space
#         first-layer gate outputs A in R^{4x3}
#       Layer-2:
#         appearance / structure / temporal experts
#         all experts output [N,C,P]
#         second-layer gate soft-fuses all three experts

#     Phase-1 keeps:
#       - quality awareness
#       - DPM / condition awareness
#       - SupCon on final fused output
#       - learnable rerank
#     """
#     def build_network(self, model_cfg):
#         self.bin_num = model_cfg.get("bin_num", [16, 8, 4, 2, 1])
#         self.parts_num = int(sum(self.bin_num))
#         self.embed_dim = int(model_cfg.get("embed_dim", 256))
#         self.shared_dim = int(model_cfg.get("shared_dim", self.embed_dim))
#         self.modality_order = model_cfg.get("modality_order", ["pc", "depth", "rgb", "sil"])
#         self.mod_drop_prob = float(model_cfg.get("modality_dropout_prob", 0.0))
#         self.router_temperature = float(model_cfg.get("router_temperature", 1.0))

#         self.rgb_backbone = self.get_backbone(model_cfg["rgb_backbone_cfg"])
#         self.rgb_backbone = SetBlockWrapper(self.rgb_backbone)
#         self.dep_backbone = self.get_backbone(model_cfg["dep_backbone_cfg"])
#         self.dep_backbone = SetBlockWrapper(self.dep_backbone)
#         self.sil_backbone = self.get_backbone(model_cfg["sil_backbone_cfg"])
#         self.sil_backbone = SetBlockWrapper(self.sil_backbone)

#         self.TP = PackSequenceWrapper(torch.max)
#         self.HPP = HorizontalPoolingPyramid(bin_num=self.bin_num)

#         self.rgb_fcs = SeparateFCs(in_channels=512, out_channels=self.embed_dim, parts_num=self.parts_num)
#         self.dep_fcs = SeparateFCs(in_channels=512, out_channels=self.embed_dim, parts_num=self.parts_num)
#         self.sil_fcs = SeparateFCs(in_channels=512, out_channels=self.embed_dim, parts_num=self.parts_num)

#         self.pc_encoder_type = model_cfg.get("pc_encoder_type", "lidargaitplusplus")
#         if self.pc_encoder_type != "lidargaitplusplus":
#             raise ValueError("This model expects pc_encoder_type='lidargaitplusplus'.")
#         self.pc_encoder = LidarGaitPlusPlusEncoder(model_cfg["pc_model_cfg"])
#         self.pc_proj = nn.Sequential(
#             nn.Linear(int(model_cfg.get("pc_input_dim", self.embed_dim)), self.embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )

#         # quality heads
#         self.q_pc = _QualityHead(self.embed_dim)
#         self.q_rgb = _QualityHead(self.embed_dim)
#         self.q_dep = _QualityHead(self.embed_dim)
#         self.q_sil = _QualityHead(self.embed_dim)

#         # shared projection for first-layer experts
#         self.proj_rgb = PartProjector(self.embed_dim, self.shared_dim)
#         self.proj_pc = PartProjector(self.embed_dim, self.shared_dim)
#         self.proj_dep = PartProjector(self.embed_dim, self.shared_dim)
#         self.proj_sil = PartProjector(self.embed_dim, self.shared_dim)

#         # DPM / condition
#         self.enable_dpm = bool(model_cfg.get("enable_dpm", True))
#         self.dpm_stats_dim = int(model_cfg.get("dpm_stats_dim", 12))
#         self.dpm_hidden_dim = int(model_cfg.get("dpm_hidden_dim", 128))
#         self.dpm_cond_dim = int(model_cfg.get("dpm_cond_dim", 16))
#         self.first_gate_hidden = int(model_cfg.get("first_gate_hidden", 256))
#         self.second_gate_hidden = int(model_cfg.get("second_gate_hidden", 256))
#         self.pretrained_dpm_ckpt = model_cfg.get("pretrained_dpm_ckpt", "")
#         self.dpm_freeze_iters = int(model_cfg.get("dpm_freeze_iters", 0))
#         # Note: dpm_lr_scale is config-only here; applying a smaller LR requires optimizer param groups in the training framework.
#         self.dpm_lr_scale = float(model_cfg.get("dpm_lr_scale", 1.0))
#         if self.enable_dpm:
#             self.aux_dpm = ConditionDPM(
#                 stats_dim=self.dpm_stats_dim,
#                 hidden_dim=self.dpm_hidden_dim,
#                 cond_dim=self.dpm_cond_dim,
#             )
#             if isinstance(self.pretrained_dpm_ckpt, str) and len(self.pretrained_dpm_ckpt) > 0:
#                 self.load_pretrained_dpm(self.pretrained_dpm_ckpt, strict=False)
#         else:
#             self.aux_dpm = None

#         self.first_gate = FirstLayerGate(d=self.embed_dim, cond_dim=self.dpm_cond_dim, hidden=self.first_gate_hidden)
#         self.appearance_expert = AppearanceExpertParts(
#             d=self.shared_dim,
#             heads=int(model_cfg.get("appearance_heads", 8)),
#             ffn_dim=int(model_cfg.get("appearance_ffn_dim", 256)),
#             dropout=float(model_cfg.get("appearance_dropout", 0.0)),
#         )
#         self.structure_expert = StructureExpertParts(d=self.shared_dim, quality_dim=4)
#         self.temporal_expert = TemporalExpertParts(d=self.shared_dim)
#         self.second_gate = SecondLayerGate(d=self.shared_dim, cond_dim=self.dpm_cond_dim, hidden=self.second_gate_hidden)

#         # final and auxiliary heads
#         self.final_fcs = SeparateFCs(in_channels=self.shared_dim, out_channels=self.embed_dim, parts_num=self.parts_num)
#         self.final_bnneck = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
#         self.app_bnneck = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
#         self.str_bnneck = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
#         self.temp_bnneck = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

#         # phase-1: fused SupCon only
#         self.enable_supcon = bool(model_cfg.get("enable_supcon", True))
#         if self.enable_supcon:
#             from ..losses.supcon_loss import SupConLoss
#             self.supcon = SupConLoss(temperature=float(model_cfg.get("supcon_temperature", 0.07)))
#             self.supcon_start_iter = int(model_cfg.get("supcon_start_iter", 10000))
#             self.supcon_warmup_iter = int(model_cfg.get("supcon_warmup_iter", 3000))
#             self.supcon_loss_weight = float(model_cfg.get("supcon_loss_weight", 0.05))

#         # learnable rerank
#         self.rerank_topk_train = int(model_cfg.get("rerank_topk_train", 8))
#         self.rerank_hard_negative_only = bool(model_cfg.get("rerank_hard_negative_only", True))
#         self.rerank_hard_negatives = int(model_cfg.get("rerank_hard_negatives", self.rerank_topk_train))
#         self.rerank_hard_neg_margin = float(model_cfg.get("rerank_hard_neg_margin", 0.0))
#         self.rerank_loss_weight = float(model_cfg.get("rerank_loss_weight", 0.1))
#         self.rerank_alpha = float(model_cfg.get("rerank_alpha", 0.1))  # inference fusion, evaluator uses it
#         self.rerank_hidden = int(model_cfg.get("rerank_hidden", 512))
#         self.reranker = CarGaitReranker(
#             d=self.embed_dim,
#             num_heads=int(model_cfg.get("rerank_heads", 8)),
#             ffn_dim=int(model_cfg.get("rerank_ffn_dim", 256)),
#             dropout=float(model_cfg.get("rerank_dropout", 0.0)),
#         )
#         self.rerank_start_iter = int(model_cfg.get("rerank_start_iter", 15000))
#         self.rerank_warmup_iter = int(model_cfg.get("rerank_warmup_iter", 3000))

#         self.app_aux_weight = float(model_cfg.get("app_aux_weight", 0.2))
#         self.str_aux_weight = float(model_cfg.get("str_aux_weight", 0.2))
#         self.temp_aux_weight = float(model_cfg.get("temp_aux_weight", 0.2))
#         self.structure_inv_weight = float(model_cfg.get("structure_inv_weight", 0.05))
#         self.temporal_cons_weight = float(model_cfg.get("temporal_cons_weight", 0.05))

#         self.register_buffer("global_step", torch.zeros((), dtype=torch.long), persistent=True)



#     def get_optimizer_parameter_groups(self, base_lr: float, weight_decay: float = 0.0):
#         """
#         Return optimizer param groups so DPM can use a smaller lr.
#         This method must be called by the training framework / optimizer builder.
#         """
#         main_params = []
#         dpm_params = []
#         for name, p in self.named_parameters():
#             if not p.requires_grad:
#                 continue
#             if name.startswith("aux_dpm."):
#                 dpm_params.append(p)
#             else:
#                 main_params.append(p)

#         groups = []
#         if len(main_params) > 0:
#             groups.append({
#                 "params": main_params,
#                 "lr": float(base_lr),
#                 "weight_decay": float(weight_decay),
#             })
#         if len(dpm_params) > 0:
#             groups.append({
#                 "params": dpm_params,
#                 "lr": float(base_lr) * float(getattr(self, "dpm_lr_scale", 1.0)),
#                 "weight_decay": float(weight_decay),
#             })
#         return groups

#     def load_pretrained_dpm(self, ckpt_path: str, strict: bool = False):
#         ckpt = torch.load(ckpt_path, map_location="cpu")
#         state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

#         mapped = {}
#         for k, v in state.items():
#             nk = k
#             if nk.startswith("module."):
#                 nk = nk[len("module."):]
#             if nk.startswith("net."):
#                 nk = nk[len("net."):]
#             mapped[nk] = v

#         missing, unexpected = self.aux_dpm.load_state_dict(mapped, strict=strict)
#         print(f"[DPM] Loaded ckpt from: {ckpt_path}")
#         print(f"[DPM] Missing keys: {missing}")
#         print(f"[DPM] Unexpected keys: {unexpected}")

#     def _use_trainable_dpm(self, step: int) -> bool:
#         if not self.enable_dpm or self.aux_dpm is None:
#             return False
#         return step >= int(getattr(self, "dpm_freeze_iters", 0))

#     def _modality_dropout_mask(self, N: int, device) -> torch.Tensor:
#         if (not self.training) or self.mod_drop_prob <= 0:
#             return torch.ones((N, 4), device=device)
#         a = (torch.rand((N, 4), device=device) > self.mod_drop_prob).float()
#         all_zero = (a.sum(dim=1) < 1.0)
#         if all_zero.any():
#             idx = torch.nonzero(all_zero, as_tuple=False).squeeze(1)
#             keep = torch.randint(low=0, high=4, size=(idx.numel(),), device=device)
#             a[idx, keep] = 1.0
#         return a

#     def _encode_image_modality(self, x, seqL, backbone, fcs, channels):
#         x = _ensure_5d_video(x, channels=channels)
#         outs = backbone(x)
#         outs = self.TP(outs, seqL, options={"dim": 2})[0]
#         part = self.HPP(outs)
#         emb = fcs(part)
#         pooled = emb.mean(dim=2)
#         return emb, pooled

#     def _encode_sil_temporal_parts(self, x, channels=1):
#         x = _ensure_5d_video(x, channels=channels)   # [N,C,S,H,W]
#         outs = self.sil_backbone(x)                  # [N,512,S,h,w]
#         n, c, s, h, w = outs.shape
#         frame_feat = outs.permute(0, 2, 1, 3, 4).contiguous().view(n * s, c, h, w)
#         part = self.HPP(frame_feat)                  # [N*S,512,P]
#         emb = self.sil_fcs(part)                     # [N*S,embed_dim,P]
#         emb = emb.view(n, s, self.embed_dim, self.parts_num).contiguous()
#         return emb

#     def _encode_pc_lidargaitpp(self, pc):
#         pc_emb, _ = self.pc_encoder(pc)
#         if pc_emb.dim() != 3:
#             raise ValueError(f"Expected PC embeddings [N,C,P], got shape={tuple(pc_emb.shape)}")
#         if pc_emb.size(2) != self.parts_num:
#             raise ValueError(
#                 f"PC parts mismatch: got P={pc_emb.size(2)} but expected {self.parts_num}."
#             )
#         if pc_emb.size(1) != self.embed_dim:
#             t = pc_emb.permute(0, 2, 1).contiguous()
#             if t.size(-1) != self.pc_proj[0].in_features:
#                 raise ValueError(
#                     f"PC channel mismatch: pc_emb C={t.size(-1)} but pc_input_dim={self.pc_proj[0].in_features}."
#                 )
#             t = self.pc_proj(t)
#             pc_emb = t.permute(0, 2, 1).contiguous()
#         return pc_emb, pc_emb.mean(dim=2)

#     def _extract_condition_stats(self, pc, rgb, dep, sil) -> torch.Tensor:
#         rgb5 = _ensure_5d_video(rgb, channels=3).float()
#         dep5 = _ensure_5d_video(dep, channels=3).float()
#         sil5 = _ensure_5d_video(sil, channels=1).float()
#         pcf = pc.float()

#         def _edge_strength(x):
#             dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=(1, 2, 3, 4))
#             dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean(dim=(1, 2, 3, 4))
#             return 0.5 * (dh + dw)

#         rgb_mean = rgb5.mean(dim=(1, 2, 3, 4))
#         rgb_std = rgb5.std(dim=(1, 2, 3, 4), unbiased=False)
#         rgb_edge = _edge_strength(rgb5)

#         dep_mean = dep5.mean(dim=(1, 2, 3, 4))
#         dep_std = dep5.std(dim=(1, 2, 3, 4), unbiased=False)
#         dep_valid = (dep5.abs() > 1e-6).float().mean(dim=(1, 2, 3, 4))

#         sil_area = sil5.mean(dim=(1, 2, 3, 4))
#         sil_std = sil5.std(dim=(1, 2, 3, 4), unbiased=False)
#         sil_edge = _edge_strength(sil5)

#         xyz = pcf[..., :3]
#         pc_valid = torch.isfinite(xyz).all(dim=-1).float().mean(dim=(1, 2))
#         xyz_clean = torch.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
#         pc_radius = torch.linalg.norm(xyz_clean, dim=-1).mean(dim=(1, 2))
#         pc_spread = xyz_clean.std(dim=2, unbiased=False).mean(dim=(1, 2))

#         stats = torch.stack([
#             rgb_mean, rgb_std, rgb_edge,
#             dep_mean, dep_std, dep_valid,
#             sil_area, sil_std, sil_edge,
#             pc_valid, pc_radius, pc_spread,
#         ], dim=1)
#         return torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

#     def _ce_over_parts(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # logits: [N, num_classes, P]
#         n, num_cls, p = logits.shape
#         logits_2d = logits.permute(0, 2, 1).reshape(n * p, num_cls)
#         labels_2d = labels.view(n, 1).expand(n, p).reshape(n * p)
#         return F.cross_entropy(logits_2d, labels_2d)

#     def _same_id_diff_type_structure_loss(self, feat: torch.Tensor, labels, typs):
#         # feat: [N,C,P]
#         x = feat.mean(dim=2)
#         if not isinstance(labels, torch.Tensor):
#             labels = torch.as_tensor(labels, device=x.device)
#         labels = labels.to(x.device)
#         loss_terms = []
#         N = x.size(0)
#         typs = [str(t) for t in typs]
#         for i in range(N):
#             for j in range(i + 1, N):
#                 if labels[i] == labels[j] and typs[i] != typs[j]:
#                     loss_terms.append((x[i] - x[j]).pow(2).mean())
#         if len(loss_terms) == 0:
#             return x.new_tensor(0.0)
#         return torch.stack(loss_terms).mean()



#     def _schedule_weight(self, base_weight: float, start_iter: int, warmup_iter: int, step: int) -> float:
#         if step < int(start_iter):
#             return 0.0
#         w = float(base_weight)
#         if int(warmup_iter) > 0:
#             w = w * min(1.0, float(step - int(start_iter)) / float(int(warmup_iter)))
#         return w

#     def _rerank_loss_inbatch(self, fused_part: torch.Tensor, labels, quality=None, avail=None):
#         """
#         In-batch learnable rerank loss.

#         For each anchor:
#         1) pick the hardest positive (largest base distance among same-ID samples)
#         2) pick top-K hardest negatives (smallest base distance among different-ID samples)
#         3) ask the reranker to rank the positive as index 0 using cross entropy
#         """
#         if fused_part is None or fused_part.dim() != 3:
#             return None

#         device = fused_part.device
#         if not isinstance(labels, torch.Tensor):
#             labels = torch.as_tensor(labels, device=device)
#         labels = labels.to(device).view(-1)
#         N = labels.numel()
#         if N < 3:
#             return None

#         base_feat = F.normalize(fused_part.mean(dim=2), dim=1)
#         base_dist = torch.cdist(base_feat, base_feat, p=2)  # [N,N]

#         losses = []
#         for i in range(N):
#             same = torch.nonzero(labels == labels[i], as_tuple=False).view(-1)
#             same = same[same != i]
#             if same.numel() == 0:
#                 continue

#             diff = torch.nonzero(labels != labels[i], as_tuple=False).view(-1)
#             if diff.numel() == 0:
#                 continue

#             # hardest positive: farthest same-ID sample in the current embedding space
#             pos_d = base_dist[i, same]
#             pos_idx = same[pos_d.argmax()]
#             pos_thresh = base_dist[i, pos_idx]

#             neg_d = base_dist[i, diff]
#             if bool(getattr(self, "rerank_hard_negative_only", True)):
#                 margin = float(getattr(self, "rerank_hard_neg_margin", 0.0))
#                 keep = neg_d <= (pos_thresh + margin)
#                 cand_neg = diff[keep]
#                 cand_neg_d = neg_d[keep]
#                 if cand_neg.numel() == 0:
#                     cand_neg = diff
#                     cand_neg_d = neg_d
#             else:
#                 cand_neg = diff
#                 cand_neg_d = neg_d

#             k = int(getattr(self, "rerank_hard_negatives", getattr(self, "rerank_topk_train", 8)))
#             k = max(1, min(int(k), int(cand_neg.numel())))
#             hard_order = torch.argsort(cand_neg_d, descending=False)[:k]
#             neg_idx = cand_neg[hard_order]

#             cand_idx = torch.cat([pos_idx.view(1), neg_idx], dim=0)
#             q_part = fused_part[i].unsqueeze(0).expand(cand_idx.numel(), -1, -1).contiguous()
#             g_part = fused_part[cand_idx].contiguous()

#             scores = self.reranker(q_part, g_part)
#             target = torch.zeros(1, dtype=torch.long, device=device)
#             loss_i = F.cross_entropy(scores.view(1, -1), target)
#             losses.append(loss_i)

#         if len(losses) == 0:
#             return None
#         return torch.stack(losses).mean()

#     def forward(self, inputs):
#         ipts, labs, typs, views, seqL = inputs
#         if self.training:
#             self.global_step += 1

#         name_to_idx = {name: i for i, name in enumerate(self.modality_order)}
#         pc = ipts[name_to_idx["pc"]]
#         rgb = ipts[name_to_idx["rgb"]]
#         dep = ipts[name_to_idx["depth"]]
#         sil = ipts[name_to_idx["sil"]]

#         rgb_emb, rgb_pool = self._encode_image_modality(rgb, seqL, self.rgb_backbone, self.rgb_fcs, channels=3)
#         dep_emb, dep_pool = self._encode_image_modality(dep, seqL, self.dep_backbone, self.dep_fcs, channels=3)
#         sil_emb, sil_pool = self._encode_image_modality(sil, seqL, self.sil_backbone, self.sil_fcs, channels=1)
#         sil_seq_emb = self._encode_sil_temporal_parts(sil, channels=1)  # [N,S,C,P]
#         pc_emb, pc_pool = self._encode_pc_lidargaitpp(pc)

#         N = rgb_emb.size(0)
#         device = rgb_emb.device
#         avail = self._modality_dropout_mask(N, device)

#         q_pc = self.q_pc(pc_emb)
#         q_rgb = self.q_rgb(rgb_emb)
#         q_dep = self.q_dep(dep_emb)
#         q_sil = self.q_sil(sil_emb)
#         quality = torch.cat([q_pc, q_rgb, q_dep, q_sil], dim=1) * avail

#         # apply availability mask to part features
#         pc_emb = pc_emb * avail[:, 0].view(N, 1, 1)
#         dep_emb = dep_emb * avail[:, 1].view(N, 1, 1)
#         rgb_emb = rgb_emb * avail[:, 2].view(N, 1, 1)
#         sil_emb = sil_emb * avail[:, 3].view(N, 1, 1)

#         # project all first-layer experts to the same part space
#         sh_pc = self.proj_pc(pc_emb)
#         sh_dep = self.proj_dep(dep_emb)
#         sh_rgb = self.proj_rgb(rgb_emb)
#         sh_sil = self.proj_sil(sil_emb)
#         shared = torch.stack([sh_pc, sh_dep, sh_rgb, sh_sil], dim=1)   # [N,4,C,P]

#         step = int(self.global_step.item())
#         if self.enable_dpm and self.aux_dpm is not None:
#             dpm_stats = self._extract_condition_stats(pc, rgb, dep, sil)
#             if self._use_trainable_dpm(step):
#                 dpm = self.aux_dpm(dpm_stats, update_norm=self.training)
#             else:
#                 with torch.no_grad():
#                     dpm = self.aux_dpm(dpm_stats, update_norm=False)
#             cond_sig = dpm["cond_emb"]
#         else:
#             dpm = None
#             cond_sig = torch.zeros((N, self.dpm_cond_dim), device=device, dtype=shared.dtype)

#         z_cat1 = torch.cat([pc_pool, dep_pool, rgb_pool, sil_pool, quality, cond_sig], dim=1)
#         first_weight = self.first_gate(
#             z_cat1,
#             avail=avail,
#             dpm_bias=(dpm["first_bias"] if dpm is not None else None),
#             dpm_gate=(dpm["gate"] if dpm is not None else None),
#         )  # [N,4,3]

#         F_app = (shared * first_weight[:, :, 0].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
#         F_str = (shared * first_weight[:, :, 1].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
#         F_temp = (shared * first_weight[:, :, 2].unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

#         E_app = self.appearance_expert(F_app)
#         E_str = self.structure_expert(F_str, quality)
#         E_temp, temp_ctx = self.temporal_expert(F_temp, sil_seq_emb)

#         z_app = E_app.mean(dim=2)
#         z_str = E_str.mean(dim=2)
#         z_temp = E_temp.mean(dim=2)
#         z_cat2 = torch.cat([z_app, z_str, z_temp, quality, cond_sig], dim=1)
#         second_weight = self.second_gate(
#             z_cat2,
#             dpm_bias=(dpm["second_bias"] if dpm is not None else None),
#             dpm_gate=(dpm["gate"] if dpm is not None else None),
#         )  # [N,3]

#         fused_shared = (
#             second_weight[:, 0].view(N, 1, 1) * E_app +
#             second_weight[:, 1].view(N, 1, 1) * E_str +
#             second_weight[:, 2].view(N, 1, 1) * E_temp
#         )

#         fused = self.final_fcs(fused_shared)
#         fused_bn, logits = self.final_bnneck(fused)

#         _, app_logits = self.app_bnneck(E_app)
#         _, str_logits = self.str_bnneck(E_str)
#         _, temp_logits = self.temp_bnneck(E_temp)

#         retval = {
#             "training_feat": {
#                 "triplet": {"embeddings": fused, "labels": labs},
#                 "softmax": {"logits": logits, "labels": labs},
#             },
#             "visual_summary": {
#                 "scalar/gate2_app_mean": second_weight[:, 0].mean().detach(),
#                 "scalar/gate2_str_mean": second_weight[:, 1].mean().detach(),
#                 "scalar/gate2_temp_mean": second_weight[:, 2].mean().detach(),
#                 "scalar/dpm_trainable": torch.as_tensor(float(self._use_trainable_dpm(step)), device=device, dtype=shared.dtype),
#             },
#             "inference_feat": {
#                 "embeddings": fused,
#                 "quality": quality,
#                 "router_prob": second_weight,
#                 "avail": avail,
#                 "labels": labs,
#                 "cond_sig": cond_sig,
#             },
#         }
#         if dpm is not None:
#             retval["visual_summary"]["scalar/dpm_cond_norm"] = dpm["cond_emb"].norm(dim=1).mean().detach()
#             retval["visual_summary"]["scalar/dpm_gate_mean"] = dpm["gate"].mean().detach()

#         if self.training:
#             app_aux = self._ce_over_parts(app_logits, labs) * self.app_aux_weight
#             str_aux = self._ce_over_parts(str_logits, labs) * self.str_aux_weight
#             temp_aux = self._ce_over_parts(temp_logits, labs) * self.temp_aux_weight
#             str_inv = self._same_id_diff_type_structure_loss(E_str, labs, typs) * self.structure_inv_weight
#             temp_cons = F.smooth_l1_loss(E_temp, temp_ctx.detach()) * self.temporal_cons_weight

#             retval["training_feat"]["app_aux"] = {"loss": app_aux}
#             retval["training_feat"]["str_aux"] = {"loss": str_aux}
#             retval["training_feat"]["temp_aux"] = {"loss": temp_aux}
#             retval["training_feat"]["str_inv"] = {"loss": str_inv}
#             retval["training_feat"]["temp_cons"] = {"loss": temp_cons}

#             retval["visual_summary"]["scalar/loss_app_aux"] = app_aux.detach()
#             retval["visual_summary"]["scalar/loss_str_aux"] = str_aux.detach()
#             retval["visual_summary"]["scalar/loss_temp_aux"] = temp_aux.detach()
#             retval["visual_summary"]["scalar/loss_str_inv"] = str_inv.detach()
#             retval["visual_summary"]["scalar/loss_temp_cons"] = temp_cons.detach()

#             if self.enable_supcon:
#                 start_iter = int(getattr(self, "supcon_start_iter", 10000))
#                 warmup = int(getattr(self, "supcon_warmup_iter", 3000))
#                 if step >= start_iter:
#                     w = 1.0 if warmup <= 0 else min(1.0, (step - start_iter) / warmup)
#                     sc_loss = self.supcon(fused, labs) * w * self.supcon_loss_weight
#                     retval["training_feat"]["supcon"] = {"loss": sc_loss}
#                     retval["visual_summary"]["scalar/loss_supcon"] = sc_loss.detach()

#             rr_w = self._schedule_weight(
#                 base_weight=float(getattr(self, "rerank_loss_weight", 0.0)),
#                 start_iter=int(getattr(self, "rerank_start_iter", 15000)),
#                 warmup_iter=int(getattr(self, "rerank_warmup_iter", 3000)),
#                 step=step,
#             )
#             if rr_w > 0.0:
#                 rr_loss = self._rerank_loss_inbatch(fused, labs, quality.detach(), avail.detach())
#                 if rr_loss is not None:
#                     rr_loss = rr_loss * rr_w
#                     retval["training_feat"]["rerank"] = {"loss": rr_loss}
#                     retval["visual_summary"]["scalar/loss_rerank"] = rr_loss.detach()

#             for _, d in retval["training_feat"].items():
#                 if isinstance(d, dict) and ("loss" in d):
#                     d["loss"] = _ddp_safe_loss(d["loss"], device=device)

#         return retval
