# -*- coding: utf-8 -*-
"""
StructGait — Structured Gait Representation Learning.

Architecture (proposal §5):
    GPF → ViewEstimator / PhaseEstimator
        → Stem + Backbone
        → framewise HPP → PhaseAlignedPool → MultiScaleTemp
        → ViewCondPartAttn
        → SeparateFCs → emb [N,256,P]
        → SubspaceHead → emb_id / emb_view / emb_cov
        → ProbabilisticHead → (μ, logσ²)
        → ViewRotaryEmbedding(μ, θ)

    Training:  ProbabilisticTripletLoss | ViewAdaptiveTripletLoss
             + CE(BNNeck(emb_id))
             + ConsistencyLoss
             + aux: orthogonality, decorrelation, view-pred, view-ssl, phase-reg
    Inference: emb_id (identity subspace of FC output)

Module switches (all independently toggleable):
    enable_view_module            M1
    enable_phase_module           M2
    enable_multiscale_module      M3  (requires M2)
    enable_subspace               M4
    enable_probabilistic          M5
    enable_view_adaptive_triplet  M6  (loss side, mutually exclusive with M5 for triplet)
    enable_consistency_loss       M7

Files
    Model  → opengait/modeling/models/struct_gait.py
    Losses → opengait/modeling/losses/struct_gait_losses.py
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

try:
    from .gpf_gait import (
        _to_ncthw, _select_channels, _as_list, _apply_framewise,
        ConvBNAct, BasicFrameBlock, FrameBackbone,
        GPFChannelNorm, TypedGPFStem, TemporalPrimitivePool,
    )
except ImportError:
    raise ImportError("StructGait requires gpf_gait.py in the same models/ package.")


# =====================================================================
#  M1 — View-Equivariant Components
# =====================================================================

class ViewEstimator(nn.Module):
    """Self-supervised viewpoint angle from GPF geo channels."""
    def __init__(self, geo_ch=5, hid=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(geo_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hid), nn.ReLU(True))
        self.head = nn.Linear(hid, 2)
        # Stable zero-angle start: sin=0, cos=1.  This avoids injecting
        # random/self-normalised theta into VRE or VAT at iteration 0.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[1] = 1.0

    def forward(self, geo):
        raw = self.head(self.enc(geo.mean(2)))
        sc = F.normalize(raw, dim=-1, eps=1e-6)
        return torch.atan2(sc[:, 0], sc[:, 1]), sc


class ViewConditionedPartAttention(nn.Module):
    """View-conditioned soft attention over HPP strips (gated residual)."""
    def __init__(self, parts, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(True),
                                 nn.Linear(hid, parts), nn.Sigmoid())
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, feat, sin_cos):
        return feat * (1. + self.gate * (self.net(sin_cos).unsqueeze(1) - .5))


class ViewRotaryEmbedding(nn.Module):
    """RoPE-style 2-D rotation in feature space conditioned on view angle."""
    def __init__(self, dim):
        super().__init__()
        n = dim // 2
        self.register_buffer("freq", 1. / (10000 ** (torch.arange(n).float() / max(n, 1))))

    def forward(self, feat, theta):
        N, C, P = feat.shape; h = C // 2
        a = theta.unsqueeze(-1) * self.freq[:h].unsqueeze(0)
        ca, sa = a.cos().unsqueeze(-1), a.sin().unsqueeze(-1)
        fe, fo = feat[:, 0::2], feat[:, 1::2]
        return torch.stack([fe*ca - fo*sa, fe*sa + fo*ca], dim=2).reshape(N, C, P)


class _ViewSelfSupLoss(nn.Module):
    """Geometry-correlation + dispersion self-supervision for the view estimator."""
    def __init__(self, geo_ch=5, dd=64):
        super().__init__()
        self.desc = nn.Sequential(nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(),
                                  nn.Linear(geo_ch * 32, dd))

    def forward(self, sc, geo_mean, labels):
        sd = F.normalize(self.desc(geo_mean), dim=-1)
        a_s, s_s = sc @ sc.t(), sd @ sd.t()
        m = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
        corr = (1. - F.cosine_similarity(
            (a_s * m).reshape(-1), (s_s * m).reshape(-1), dim=0)) if m.sum() > 0 else sc.sum() * 0.
        c = sc - sc.mean(0, keepdim=True)
        disp = F.mse_loss(c.t() @ c / max(sc.size(0) - 1, 1),
                          .25 * torch.eye(2, device=sc.device))
        return corr + .5 * disp


# =====================================================================
#  M2 — Implicit Phase Modeling
# =====================================================================

class ImplicitPhaseEstimator(nn.Module):
    def __init__(self, mot_ch=3, hid=64, H=4):
        super().__init__()
        self.H = H
        self.enc = nn.Sequential(
            nn.Conv1d(mot_ch, hid, 7, padding=3, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True),
            nn.Conv1d(hid, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True))
        self.ph = nn.Conv1d(hid, H * 2, 1)
        self.fq = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hid, 1), nn.Softplus())

    def forward(self, mt):
        h = self.enc(mt); raw = self.ph(h)
        return torch.cat([F.normalize(raw[:, 2*k:2*k+2], dim=1) for k in range(self.H)], 1), self.fq(h)


class PhaseAlignedPooling(nn.Module):
    def __init__(self, B=8, kappa=4.):
        super().__init__(); self.B, self.kappa = B, kappa
        self.register_buffer("ctr", torch.linspace(0, 2*math.pi*(1-1./B), B))

    def forward(self, feat, sc):
        N, C, T, P = feat.shape
        ph = torch.atan2(sc[:, 0], sc[:, 1]) + math.pi
        w = torch.exp(self.kappa * torch.cos(ph.unsqueeze(-1) - self.ctr))
        w = w / (w.sum(1, keepdim=True) + 1e-8)
        fr = feat.permute(0, 3, 1, 2).reshape(N*P, C, T)
        wr = w.unsqueeze(1).expand(-1, P, -1, -1).reshape(N*P, T, self.B)
        return torch.bmm(fr, wr).reshape(N, P, C, self.B).permute(0, 2, 3, 1)


class _PhaseRegLoss(nn.Module):
    def forward(self, sc, freq):
        ph = torch.atan2(sc[:, 0], sc[:, 1])
        d = ph[:, 1:] - ph[:, :-1]
        d = torch.where(d < -math.pi, d + 2*math.pi, d)
        d = torch.where(d > math.pi, d - 2*math.pi, d)
        return F.relu(-d).mean() + .5*d.var(1).mean() + .1*(F.relu(freq-3).mean() + F.relu(.5-freq).mean())


# =====================================================================
#  M3 — Multi-Scale Temporal Encoding
# =====================================================================

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, ch, B=8):
        super().__init__()
        self.micro = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True),
            nn.Conv1d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.BatchNorm1d(ch))
        self.me1 = nn.Sequential(nn.Conv1d(ch, ch, 3, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True))
        self.me2 = nn.Sequential(nn.Conv1d(ch, ch, 3, bias=False), nn.BatchNorm1d(ch))
        self.macro = nn.Sequential(nn.Linear(ch*3, ch, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True),
                                   nn.Linear(ch, ch))
        self.gate = nn.Sequential(nn.Linear(ch*3, 64), nn.ReLU(True), nn.Linear(64, 3), nn.Softmax(-1))

    @staticmethod
    def _cp(x, p=1): return torch.cat([x[..., -p:], x, x[..., :p]], -1)

    def forward(self, ft, fp):
        N, C, T, P = ft.shape; B = fp.shape[2]
        mi = self.micro(ft.permute(0,3,1,2).reshape(N*P,C,T)).max(2)[0].reshape(N,P,C).permute(0,2,1)
        me = fp.permute(0,3,1,2).reshape(N*P,C,B)
        me = self.me1(self._cp(me)); me = self.me2(self._cp(me))
        me = me.max(2)[0].reshape(N,P,C).permute(0,2,1)
        tm, ts = ft.mean(2), ft.std(2, unbiased=False)
        h = max(T // 2, 1)
        first = ft[:, :, :h].mean(2)
        second = ft[:, :, h:2*h]
        second = first if second.size(2) == 0 else second.mean(2)
        st = F.cosine_similarity(first, second, dim=1).unsqueeze(1).expand_as(tm)
        ma = self.macro(torch.cat([tm,ts,st],1).permute(0,2,1).reshape(N*P,3*C)).reshape(N,P,C).permute(0,2,1)
        gi = torch.cat([mi.mean(2), me.mean(2), ma.mean(2)], 1)
        sw = self.gate(gi)  # [N, 3]
        # Explicit [N,1,1] weights.  w[:, 0] would be [N,1] and does not
        # broadcast safely to [N,C,P].
        w0 = sw[:, 0].view(N, 1, 1)
        w1 = sw[:, 1].view(N, 1, 1)
        w2 = sw[:, 2].view(N, 1, 1)
        return w0 * mi + w1 * me + w2 * ma, sw


# =====================================================================
#  M4 — Subspace Disentanglement  (proposal §1)
# =====================================================================

class SubspaceHead(nn.Module):
    """
    Learned orthogonal projector that decomposes the embedding into
    identity / view / covariate subspaces.  Matching uses only emb_id.
    """
    def __init__(self, total_dim, id_dim, view_dim, cov_dim):
        super().__init__()
        assert id_dim + view_dim + cov_dim == total_dim
        self.id_dim, self.view_dim, self.cov_dim = id_dim, view_dim, cov_dim
        self.projector = nn.Linear(total_dim, total_dim, bias=False)
        nn.init.eye_(self.projector.weight)

    def forward(self, emb):
        """emb [N,C,P] → (emb_id, emb_view, emb_cov)"""
        N, C, P = emb.shape
        proj = self.projector(emb.permute(0, 2, 1)).permute(0, 2, 1)
        d1, d2 = self.id_dim, self.id_dim + self.view_dim
        return proj[:, :d1].contiguous(), proj[:, d1:d2].contiguous(), proj[:, d2:].contiguous()

    def orthogonality_loss(self):
        """W^T W should stay close to I."""
        W = self.projector.weight
        return F.mse_loss(W.t() @ W, torch.eye(W.size(0), device=W.device))

    @staticmethod
    def decorrelation(a, b):
        """Cross-covariance penalty (part-averaged) between two [N,D,P] sets."""
        if b is None or b.size(1) == 0:
            return a.sum() * 0.0
        af, bf = a.mean(2), b.mean(2)
        ac, bc = af - af.mean(0, keepdim=True), bf - bf.mean(0, keepdim=True)
        return ((ac.t() @ bc) / max(af.size(0) - 1, 1)).pow(2).mean()


# =====================================================================
#  M5 — Probabilistic Embedding  (proposal §3)
# =====================================================================

class ProbabilisticHead(nn.Module):
    """Output (μ, log σ²) for uncertainty-aware matching."""
    def __init__(self, id_dim, bb_ch, parts):
        super().__init__()
        self.logvar_fc = SeparateFCs(parts_num=parts, in_channels=bb_ch, out_channels=id_dim)

    def forward(self, emb_id, feat_out):
        return emb_id, self.logvar_fc(feat_out).clamp(-10, 2)


# =====================================================================
#  StructGait — Main Model
# =====================================================================

class StructGait(BaseModel):

    def build_network(self, model_cfg: Dict):
        cfg = model_cfg
        self.input_index = int(cfg.get("input_index", 0))
        self.source_channels = int(cfg.get("source_channels", 12))

        # ---- channels ----
        self.geo_idx = list(map(int, cfg.get("geo_indices", [0,1,2,3,4])))
        self.mot_idx = list(map(int, cfg.get("motion_indices", [5,8,9])))
        ri = cfg.get("reliability_index", 10)
        self.rel_idx = int(ri) if ri is not None else None
        self.res_idx = list(map(int, cfg.get("residual_indices", [])))

        # ---- switches ----
        self.en_view  = bool(cfg.get("enable_view_module", True))
        self.en_phase = bool(cfg.get("enable_phase_module", True))
        self.en_mst   = bool(cfg.get("enable_multiscale_module", True))
        self.en_sub   = bool(cfg.get("enable_subspace", True))
        self.en_prob  = bool(cfg.get("enable_probabilistic", False))
        self.en_vat   = bool(cfg.get("enable_view_adaptive_triplet", True))
        self.en_cons  = bool(cfg.get("enable_consistency_loss", True))

        # ---- safety / supervision knobs ----
        # By default all metric/CE/inference paths use the same embedding.
        # Only enable VRE for metric after the view estimator is supervised and stable.
        self.use_vre_metric = bool(cfg.get("use_view_rotary_for_metric", False))
        self.detach_theta_in_triplet = bool(cfg.get("detach_theta_in_triplet", True))

        # Phase teacher from GPF ch6/ch7 by default.  This lets phase pooling be
        # tested before asking the implicit phase estimator to discover phase.
        self.use_phase_teacher = bool(cfg.get("use_phase_teacher", True))
        self.phase_teacher_idx = list(map(int, cfg.get("phase_teacher_indices", [6, 7])))
        self.enable_implicit_phase_estimator = bool(cfg.get("enable_implicit_phase_estimator", True))

        # ---- aux weights / warmup ----
        self.aw = {
            'view':   float(cfg.get("aux_weight_view", 0.001)),
            'phase':  float(cfg.get("aux_weight_phase", 0.001)),
            'phase_teacher': float(cfg.get("aux_weight_phase_teacher", 0.001)),
            'ortho':  float(cfg.get("aux_weight_ortho", 0.001)),
            'decorr': float(cfg.get("aux_weight_decorr", 0.001)),
            'vpred':  float(cfg.get("aux_weight_vpred", 0.005)),
        }
        self.warmup_s = int(cfg.get("aux_warmup_start", 5000))
        self.warmup_e = int(cfg.get("aux_warmup_end", 10000))

        # ---- GPF channel norm ----
        self.norm = GPFChannelNorm(enabled=bool(cfg.get("enable_channel_norm", True)))

        # ---- Stem ----
        stem_ch = int(cfg.get("stem_out_channels", 32))
        self.adapter = TypedGPFStem(
            geo_channels=len(self.geo_idx), motion_channels=len(self.mot_idx),
            residual_channels=len(self.res_idx), out_channels=stem_ch,
            branch_channels=int(cfg.get("grouped_branch_channels", 24)),
            residual_branch_channels=int(cfg.get("residual_branch_channels", 12)),
            use_reliability_gate=bool(cfg.get("use_reliability_gate", True)),
            gate_strength=float(cfg.get("reliability_gate_strength", 0.75)),
            primitive_dropout=float(cfg.get("primitive_dropout", 0.05)))

        # ---- Backbone ----
        self.backbone = FrameBackbone(
            stem_ch,
            channels=cfg.get("backbone_channels", [64,128,256,512]),
            strides=cfg.get("backbone_strides", [1,2,2,1]),
            blocks_per_stage=cfg.get("blocks_per_stage", [1,1,1,1]))
        bb_ch = self.backbone.out_channels

        # ---- HPP / fallback TP ----
        bin_num = list(cfg.get("bin_num", [16]))
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
        P = int(sum(bin_num))
        self.TP = TemporalPrimitivePool(pool_type=cfg.get("temporal_pool", "max"))

        # ---- M1: View ----
        if self.en_view:
            self.view_est  = ViewEstimator(len(self.geo_idx), int(cfg.get("view_hidden_dim", 128)))
            self.view_attn = ViewConditionedPartAttention(P, int(cfg.get("view_attn_hidden", 32)))
            self.view_ssl  = _ViewSelfSupLoss(len(self.geo_idx))

        # ---- M2: Phase ----
        nB = int(cfg.get("num_phase_bins", 8))
        if self.en_phase:
            if self.enable_implicit_phase_estimator:
                self.phase_est  = ImplicitPhaseEstimator(len(self.mot_idx),
                                                         int(cfg.get("phase_hidden_dim", 64)),
                                                         int(cfg.get("phase_num_harmonics", 4)))
                self.phase_reg  = _PhaseRegLoss()
            self.phase_pool = PhaseAlignedPooling(nB, float(cfg.get("phase_kappa", 4.)))

        # ---- M3: Multi-Scale ----
        if self.en_mst:
            assert self.en_phase, "MST requires phase"
            self.mst = MultiScaleTemporalEncoder(bb_ch, nB)
            # Residual rollback: alpha=0 makes M3 exactly the baseline temporal max path.
            self.mst_alpha = nn.Parameter(torch.zeros(1))

        # ---- FCs ----
        emb_ch = int(cfg.get("emb_channels", 256))
        self.FCs = SeparateFCs(parts_num=P, in_channels=bb_ch, out_channels=emb_ch)

        # ---- M4: Subspace ----
        self.id_dim = emb_ch  # default when subspace disabled
        if self.en_sub:
            self.id_dim  = int(cfg.get("id_dim", emb_ch - 32))
            v_dim        = int(cfg.get("view_subspace_dim", 16))
            c_dim        = emb_ch - self.id_dim - v_dim
            assert c_dim >= 0
            self.subspace = SubspaceHead(emb_ch, self.id_dim, v_dim, c_dim)
            if self.en_view:
                self.view_pred = nn.Linear(v_dim, 2)

        # ---- M5: Probabilistic ----
        if self.en_prob:
            self.prob_head = ProbabilisticHead(self.id_dim, bb_ch, P)

        # ---- VRE on identity subspace ----
        if self.en_view:
            self.vre = ViewRotaryEmbedding(self.id_dim)

        # ---- BNNeck (on emb_id, for CE only) ----
        self.BNNecks = SeparateBNNecks(
            parts_num=P, in_channels=self.id_dim,
            class_num=int(cfg["class_num"]))

    # --------------------------------------------------------------- helpers
    def _aux_s(self):
        it = getattr(self, "iteration", 0)
        if it < self.warmup_s: return 0.
        if it >= self.warmup_e: return 1.
        return (it - self.warmup_s) / max(self.warmup_e - self.warmup_s, 1)

    def init_parameters(self):
        """Restore critical structured initialisation after BaseModel init.

        OpenGait initialises Conv/Linear/BN after build_network().  Without this
        override, SubspaceHead's identity projector and ViewEstimator's
        zero-angle head can be overwritten by the global Linear initialiser.
        """
        super().init_parameters()

        if hasattr(self, "subspace"):
            nn.init.eye_(self.subspace.projector.weight)

        if hasattr(self, "view_attn"):
            with torch.no_grad():
                self.view_attn.gate.zero_()

        if hasattr(self, "view_est"):
            nn.init.zeros_(self.view_est.head.weight)
            nn.init.zeros_(self.view_est.head.bias)
            with torch.no_grad():
                self.view_est.head.bias[1] = 1.0

        if hasattr(self, "mst_alpha"):
            with torch.no_grad():
                self.mst_alpha.zero_()

    def _framewise_hpp(self, f):
        N, C, T, H, W = f.shape
        p = self.HPP(f.permute(0,2,1,3,4).reshape(N*T, C, H, W))
        return p.reshape(N, T, C, -1).permute(0, 2, 1, 3).contiguous()

    def _phase_teacher_from_gpf(self, gpf, target_t):
        """Return teacher phase [N,2,T] from GPF sin/cos maps, if available."""
        if not self.use_phase_teacher or len(self.phase_teacher_idx) != 2:
            return None
        if any(i < 0 or i >= gpf.size(1) for i in self.phase_teacher_idx):
            return None

        teacher = gpf[:, self.phase_teacher_idx].mean(dim=(-1, -2)).float()
        if teacher.size(2) != target_t:
            teacher = F.interpolate(teacher, size=target_t, mode="linear", align_corners=False)
        return F.normalize(teacher, dim=1, eps=1e-6)

    # --------------------------------------------------------------- forward
    def forward(self, inputs):
        ipts, labs, typs, vies, seqL = inputs
        bs = labs.size(0) if torch.is_tensor(labs) else len(labs)

        # 0. GPF
        gpf = self.norm(_to_ncthw(ipts[self.input_index],
                                  expected_channels=self.source_channels))
        N = gpf.size(0)
        geo = _select_channels(gpf, self.geo_idx, "geo")
        mot = _select_channels(gpf, self.mot_idx, "mot")
        rel = gpf[:, self.rel_idx:self.rel_idx+1] \
              if self.rel_idx is not None and self.rel_idx < gpf.size(1) else None
        res = _select_channels(gpf, self.res_idx, "res") if self.res_idx else None
        fg  = gpf[:, 0:1] if gpf.size(1) > 0 else None

        # 1. Stem + Backbone
        stem, _ = self.adapter(geo, mot, reliability=rel, residual=res, foreground=fg)
        bb = self.backbone(stem)
        _, Cb, T, Hb, Wb = bb.shape

        # 2. View estimation
        theta, sin_cos = None, None
        if self.en_view:
            theta, sin_cos = self.view_est(geo)

        # 3. Phase estimation / teacher
        phase_sc, freq, phase_teacher = None, None, None
        if self.en_phase:
            phase_teacher = self._phase_teacher_from_gpf(gpf, T)
            if hasattr(self, "phase_est"):
                mt = mot.mean(dim=[3, 4])
                if mt.size(2) != T:
                    mt = F.adaptive_avg_pool1d(mt, T)
                phase_sc, freq = self.phase_est(mt)
                if phase_sc.size(2) != T:
                    phase_sc = F.interpolate(phase_sc, T, mode="linear", align_corners=False)

        # 4. Temporal path
        new_path = (self.en_phase or self.en_mst) and (N == bs)
        sw = None
        if new_path:
            pf = self._framewise_hpp(bb)                        # [N,Cb,T,P]
            if self.en_phase:
                # Prefer the explicit GPF phase teacher when present.  Fall back
                # to implicit phase only when teacher maps are unavailable.
                psc = phase_teacher if phase_teacher is not None else (phase_sc[:, :2] if phase_sc is not None else None)
                pa = self.phase_pool(pf, psc) if psc is not None else None  # [N,Cb,B,P]
            else:
                pa = None
            if self.en_mst and pa is not None:
                base = pf.max(2)[0]                             # baseline temporal max, [N,Cb,P]
                mst_out, sw = self.mst(pf, pa)                  # [N,Cb,P], [N,3]
                feat_out = base + torch.tanh(self.mst_alpha) * (mst_out - base)
            elif pa is not None:
                feat_out = pa.max(2)[0]
            else:
                feat_out = pf.max(2)[0]
        else:
            feat_out = self.HPP(self.TP(bb, seqL=seqL, batch_size=bs))

        # 5. View-conditioned part attention  (M1a)
        if self.en_view and sin_cos is not None:
            feat_out = self.view_attn(feat_out, sin_cos)

        # 6. SeparateFCs → full embedding
        emb = self.FCs(feat_out)                                # [N, emb_ch, P]

        # 7. Subspace decomposition  (M4)
        if self.en_sub:
            emb_id, emb_v, emb_c = self.subspace(emb)
        else:
            emb_id, emb_v, emb_c = emb, None, None

        # 8. Probabilistic head  (M5)
        logvar = None
        if self.en_prob:
            _, logvar = self.prob_head(emb_id, feat_out)

        # 9. Metric embedding
        # Keep Triplet / CE / Consistency / Inference in the same space.
        # Optional VRE is off by default; when enabled, use -theta as a
        # canonicalisation transform rather than training on one space and
        # evaluating another.
        if self.en_view and theta is not None and self.use_vre_metric:
            emb_metric = self.vre(emb_id, -theta)
        else:
            emb_metric = emb_id

        # 10. BNNeck → logits  (CE loss only — not used for inference)
        _, logits = self.BNNecks(emb_metric)

        # ============================================================
        #  Build training_feat
        # ============================================================
        training_feat = {}

        # --- primary triplet ---
        if self.en_prob and logvar is not None:
            # ProbabilisticTripletLoss interface: (mu, logvar, labels)
            training_feat["prob_triplet"] = {
                "mu": emb_metric, "logvar": logvar, "labels": labs}
        else:
            td = {"embeddings": emb_metric, "labels": labs}
            if self.en_vat and theta is not None:
                td["theta"] = theta.detach() if self.detach_theta_in_triplet else theta
            training_feat["triplet"] = td

        # --- CE ---
        training_feat["softmax"] = {"logits": logits, "labels": labs}

        # --- consistency ---
        if self.en_cons:
            training_feat["consistency"] = {"embeddings": emb_metric, "labels": labs}

        # --- auxiliary (pre-weighted tensors, added directly by LossAggregator) ---
        if self.training:
            s = self._aux_s()
            if s > 0:
                if self.en_view:
                    training_feat["_a_view"] = \
                        self.view_ssl(sin_cos, geo.mean(2), labs) * self.aw['view'] * s
                if self.en_phase and phase_sc is not None and freq is not None and hasattr(self, "phase_reg"):
                    training_feat["_a_phase"] = \
                        self.phase_reg(phase_sc[:, :2], freq) * self.aw['phase'] * s
                if self.en_phase and phase_sc is not None and phase_teacher is not None:
                    pred = F.normalize(phase_sc[:, :2], dim=1, eps=1e-6)
                    teacher = F.normalize(phase_teacher, dim=1, eps=1e-6)
                    training_feat["_a_phase_teacher"] = \
                        (1.0 - (pred * teacher).sum(dim=1)).mean() * self.aw['phase_teacher'] * s
                if self.en_sub:
                    training_feat["_a_ortho"] = \
                        self.subspace.orthogonality_loss() * self.aw['ortho'] * s
                    training_feat["_a_decorr"] = (
                        SubspaceHead.decorrelation(emb_id, emb_v)
                        + SubspaceHead.decorrelation(emb_id, emb_c)
                    ) * self.aw['decorr'] * s
                    if self.en_view and sin_cos is not None:
                        vp = self.view_pred(emb_v.mean(2))
                        training_feat["_a_vpred"] = \
                            F.mse_loss(vp, sin_cos) * self.aw['vpred'] * s

        # ============================================================
        #  Visual summary
        # ============================================================
        vis = {}
        if gpf.size(1) > 0:
            v = gpf[:, 0:1].detach().clamp(0, 1)
            idx = torch.linspace(0, v.size(2)-1, min(8, v.size(2)), device=v.device).long()
            v = v.index_select(2, idx)
            n_, c_, t_, h_, w_ = v.shape
            vis["image/sil"] = v.permute(0,2,1,3,4).reshape(n_*t_, c_, h_, w_)

        if self.en_view and theta is not None and sin_cos is not None:
            vis["scalar/view/theta_std"] = theta.std().detach()
            vis["scalar/view/sincos_norm"] = sin_cos.norm(dim=-1).mean().detach()
            if hasattr(self, "view_attn"):
                vis["scalar/view_attn/gate"] = self.view_attn.gate.detach()

        if self.en_sub:
            vis["scalar/subspace/id_norm"] = emb_id.norm(dim=1).mean().detach()
            if emb_v is not None:
                vis["scalar/subspace/view_norm"] = emb_v.norm(dim=1).mean().detach()
            if emb_c is not None:
                vis["scalar/subspace/cov_norm"] = emb_c.norm(dim=1).mean().detach()

        if self.en_mst and sw is not None:
            vis["scalar/mst/alpha"] = torch.tanh(self.mst_alpha).detach()
            vis["scalar/mst/gate_micro"] = sw[:, 0].mean().detach()
            vis["scalar/mst/gate_meso"] = sw[:, 1].mean().detach()
            vis["scalar/mst/gate_macro"] = sw[:, 2].mean().detach()

        # ============================================================
        #  Inference: same identity metric embedding used by training
        # ============================================================
        return {
            "training_feat": training_feat,
            "visual_summary": vis,
            "inference_feat": {"embeddings": emb_metric},
        }





# # -*- coding: utf-8 -*-
# """
# StructGait — Structured Gait Representation Learning.

# Architecture (proposal §5):
#     GPF → ViewEstimator / PhaseEstimator
#         → Stem + Backbone
#         → framewise HPP → PhaseAlignedPool → MultiScaleTemp
#         → ViewCondPartAttn
#         → SeparateFCs → emb [N,256,P]
#         → SubspaceHead → emb_id / emb_view / emb_cov
#         → ProbabilisticHead → (μ, logσ²)
#         → ViewRotaryEmbedding(μ, θ)

#     Training:  ProbabilisticTripletLoss | ViewAdaptiveTripletLoss
#              + CE(BNNeck(emb_id))
#              + ConsistencyLoss
#              + aux: orthogonality, decorrelation, view-pred, view-ssl, phase-reg
#     Inference: emb_id (identity subspace of FC output)

# Module switches (all independently toggleable):
#     enable_view_module            M1
#     enable_phase_module           M2
#     enable_multiscale_module      M3  (requires M2)
#     enable_subspace               M4
#     enable_probabilistic          M5
#     enable_view_adaptive_triplet  M6  (loss side, mutually exclusive with M5 for triplet)
#     enable_consistency_loss       M7

# Files
#     Model  → opengait/modeling/models/struct_gait.py
#     Losses → opengait/modeling/losses/struct_gait_losses.py
# """

# import math
# from typing import Dict, Optional

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from ..base_model import BaseModel
# from ..modules import HorizontalPoolingPyramid, SeparateFCs, SeparateBNNecks

# try:
#     from .gpf_gait import (
#         _to_ncthw, _select_channels, _as_list, _apply_framewise,
#         ConvBNAct, BasicFrameBlock, FrameBackbone,
#         GPFChannelNorm, TypedGPFStem, TemporalPrimitivePool,
#     )
# except ImportError:
#     raise ImportError("StructGait requires gpf_gait.py in the same models/ package.")


# # =====================================================================
# #  M1 — View-Equivariant Components
# # =====================================================================

# class ViewEstimator(nn.Module):
#     """Self-supervised viewpoint angle from GPF geo channels."""
#     def __init__(self, geo_ch=5, hid=128):
#         super().__init__()
#         self.enc = nn.Sequential(
#             nn.Conv2d(geo_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hid), nn.ReLU(True))
#         self.head = nn.Linear(hid, 2)
#         nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

#     def forward(self, geo):
#         sc = F.normalize(self.head(self.enc(geo.mean(2))), dim=-1)
#         return torch.atan2(sc[:, 0], sc[:, 1]), sc


# class ViewConditionedPartAttention(nn.Module):
#     """View-conditioned soft attention over HPP strips (gated residual)."""
#     def __init__(self, parts, hid=32):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(2, hid), nn.ReLU(True),
#                                  nn.Linear(hid, parts), nn.Sigmoid())
#         self.gate = nn.Parameter(torch.zeros(1))

#     def forward(self, feat, sin_cos):
#         return feat * (1. + self.gate * (self.net(sin_cos).unsqueeze(1) - .5))


# class ViewRotaryEmbedding(nn.Module):
#     """RoPE-style 2-D rotation in feature space conditioned on view angle."""
#     def __init__(self, dim):
#         super().__init__()
#         n = dim // 2
#         self.register_buffer("freq", 1. / (10000 ** (torch.arange(n).float() / max(n, 1))))

#     def forward(self, feat, theta):
#         N, C, P = feat.shape; h = C // 2
#         a = theta.unsqueeze(-1) * self.freq[:h].unsqueeze(0)
#         ca, sa = a.cos().unsqueeze(-1), a.sin().unsqueeze(-1)
#         fe, fo = feat[:, 0::2], feat[:, 1::2]
#         return torch.stack([fe*ca - fo*sa, fe*sa + fo*ca], dim=2).reshape(N, C, P)


# class _ViewSelfSupLoss(nn.Module):
#     """Geometry-correlation + dispersion self-supervision for the view estimator."""
#     def __init__(self, geo_ch=5, dd=64):
#         super().__init__()
#         self.desc = nn.Sequential(nn.AdaptiveAvgPool2d((8, 4)), nn.Flatten(),
#                                   nn.Linear(geo_ch * 32, dd))

#     def forward(self, sc, geo_mean, labels):
#         sd = F.normalize(self.desc(geo_mean), dim=-1)
#         a_s, s_s = sc @ sc.t(), sd @ sd.t()
#         m = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
#         corr = (1. - F.cosine_similarity(
#             (a_s * m).reshape(-1), (s_s * m).reshape(-1), dim=0)) if m.sum() > 0 else sc.sum() * 0.
#         c = sc - sc.mean(0, keepdim=True)
#         disp = F.mse_loss(c.t() @ c / max(sc.size(0) - 1, 1),
#                           .25 * torch.eye(2, device=sc.device))
#         return corr + .5 * disp


# # =====================================================================
# #  M2 — Implicit Phase Modeling
# # =====================================================================

# class ImplicitPhaseEstimator(nn.Module):
#     def __init__(self, mot_ch=3, hid=64, H=4):
#         super().__init__()
#         self.H = H
#         self.enc = nn.Sequential(
#             nn.Conv1d(mot_ch, hid, 7, padding=3, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True),
#             nn.Conv1d(hid, hid, 5, padding=2, bias=False), nn.BatchNorm1d(hid), nn.ReLU(True))
#         self.ph = nn.Conv1d(hid, H * 2, 1)
#         self.fq = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hid, 1), nn.Softplus())

#     def forward(self, mt):
#         h = self.enc(mt); raw = self.ph(h)
#         return torch.cat([F.normalize(raw[:, 2*k:2*k+2], dim=1) for k in range(self.H)], 1), self.fq(h)


# class PhaseAlignedPooling(nn.Module):
#     def __init__(self, B=8, kappa=4.):
#         super().__init__(); self.B, self.kappa = B, kappa
#         self.register_buffer("ctr", torch.linspace(0, 2*math.pi*(1-1./B), B))

#     def forward(self, feat, sc):
#         N, C, T, P = feat.shape
#         ph = torch.atan2(sc[:, 0], sc[:, 1]) + math.pi
#         w = torch.exp(self.kappa * torch.cos(ph.unsqueeze(-1) - self.ctr))
#         w = w / (w.sum(1, keepdim=True) + 1e-8)
#         fr = feat.permute(0, 3, 1, 2).reshape(N*P, C, T)
#         wr = w.unsqueeze(1).expand(-1, P, -1, -1).reshape(N*P, T, self.B)
#         return torch.bmm(fr, wr).reshape(N, P, C, self.B).permute(0, 2, 3, 1)


# class _PhaseRegLoss(nn.Module):
#     def forward(self, sc, freq):
#         ph = torch.atan2(sc[:, 0], sc[:, 1])
#         d = ph[:, 1:] - ph[:, :-1]
#         d = torch.where(d < -math.pi, d + 2*math.pi, d)
#         d = torch.where(d > math.pi, d - 2*math.pi, d)
#         return F.relu(-d).mean() + .5*d.var(1).mean() + .1*(F.relu(freq-3).mean() + F.relu(.5-freq).mean())


# # =====================================================================
# #  M3 — Multi-Scale Temporal Encoding
# # =====================================================================

# class MultiScaleTemporalEncoder(nn.Module):
#     def __init__(self, ch, B=8):
#         super().__init__()
#         self.micro = nn.Sequential(
#             nn.Conv1d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True),
#             nn.Conv1d(ch, ch, 3, padding=1, groups=ch, bias=False), nn.BatchNorm1d(ch))
#         self.me1 = nn.Sequential(nn.Conv1d(ch, ch, 3, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True))
#         self.me2 = nn.Sequential(nn.Conv1d(ch, ch, 3, bias=False), nn.BatchNorm1d(ch))
#         self.macro = nn.Sequential(nn.Linear(ch*3, ch, bias=False), nn.BatchNorm1d(ch), nn.ReLU(True),
#                                    nn.Linear(ch, ch))
#         self.gate = nn.Sequential(nn.Linear(ch*3, 64), nn.ReLU(True), nn.Linear(64, 3), nn.Softmax(-1))

#     @staticmethod
#     def _cp(x, p=1): return torch.cat([x[..., -p:], x, x[..., :p]], -1)

#     def forward(self, ft, fp):
#         N, C, T, P = ft.shape; B = fp.shape[2]
#         mi = self.micro(ft.permute(0,3,1,2).reshape(N*P,C,T)).max(2)[0].reshape(N,P,C).permute(0,2,1)
#         me = fp.permute(0,3,1,2).reshape(N*P,C,B)
#         me = self.me1(self._cp(me)); me = self.me2(self._cp(me))
#         me = me.max(2)[0].reshape(N,P,C).permute(0,2,1)
#         tm, ts = ft.mean(2), ft.std(2)
#         h = max(T//2, 1)
#         st = F.cosine_similarity(ft[:,:,:h].mean(2), ft[:,:,h:2*h].mean(2), dim=1).unsqueeze(1).expand_as(tm)
#         ma = self.macro(torch.cat([tm,ts,st],1).permute(0,2,1).reshape(N*P,3*C)).reshape(N,P,C).permute(0,2,1)
#         gi = torch.cat([mi.mean(2), me.mean(2), ma.mean(2)], 1)
#         sw = self.gate(gi); w = sw.unsqueeze(-1)
#         return w[:,0]*mi + w[:,1]*me + w[:,2]*ma, sw


# # =====================================================================
# #  M4 — Subspace Disentanglement  (proposal §1)
# # =====================================================================

# class SubspaceHead(nn.Module):
#     """
#     Learned orthogonal projector that decomposes the embedding into
#     identity / view / covariate subspaces.  Matching uses only emb_id.
#     """
#     def __init__(self, total_dim, id_dim, view_dim, cov_dim):
#         super().__init__()
#         assert id_dim + view_dim + cov_dim == total_dim
#         self.id_dim, self.view_dim, self.cov_dim = id_dim, view_dim, cov_dim
#         self.projector = nn.Linear(total_dim, total_dim, bias=False)
#         nn.init.eye_(self.projector.weight)

#     def forward(self, emb):
#         """emb [N,C,P] → (emb_id, emb_view, emb_cov)"""
#         N, C, P = emb.shape
#         proj = self.projector(emb.permute(0, 2, 1)).permute(0, 2, 1)
#         d1, d2 = self.id_dim, self.id_dim + self.view_dim
#         return proj[:, :d1].contiguous(), proj[:, d1:d2].contiguous(), proj[:, d2:].contiguous()

#     def orthogonality_loss(self):
#         """W^T W should stay close to I."""
#         W = self.projector.weight
#         return F.mse_loss(W.t() @ W, torch.eye(W.size(0), device=W.device))

#     @staticmethod
#     def decorrelation(a, b):
#         """Cross-covariance penalty (part-averaged) between two [N,D,P] sets."""
#         af, bf = a.mean(2), b.mean(2)
#         ac, bc = af - af.mean(0, keepdim=True), bf - bf.mean(0, keepdim=True)
#         return ((ac.t() @ bc) / max(af.size(0) - 1, 1)).pow(2).mean()


# # =====================================================================
# #  M5 — Probabilistic Embedding  (proposal §3)
# # =====================================================================

# class ProbabilisticHead(nn.Module):
#     """Output (μ, log σ²) for uncertainty-aware matching."""
#     def __init__(self, id_dim, bb_ch, parts):
#         super().__init__()
#         self.logvar_fc = SeparateFCs(parts_num=parts, in_channels=bb_ch, out_channels=id_dim)

#     def forward(self, emb_id, feat_out):
#         return emb_id, self.logvar_fc(feat_out).clamp(-10, 2)


# # =====================================================================
# #  StructGait — Main Model
# # =====================================================================

# class StructGait(BaseModel):

#     def build_network(self, model_cfg: Dict):
#         cfg = model_cfg
#         self.input_index = int(cfg.get("input_index", 0))
#         self.source_channels = int(cfg.get("source_channels", 12))

#         # ---- channels ----
#         self.geo_idx = list(map(int, cfg.get("geo_indices", [0,1,2,3,4])))
#         self.mot_idx = list(map(int, cfg.get("motion_indices", [5,8,9])))
#         ri = cfg.get("reliability_index", 10)
#         self.rel_idx = int(ri) if ri is not None else None
#         self.res_idx = list(map(int, cfg.get("residual_indices", [])))

#         # ---- switches ----
#         self.en_view  = bool(cfg.get("enable_view_module", True))
#         self.en_phase = bool(cfg.get("enable_phase_module", True))
#         self.en_mst   = bool(cfg.get("enable_multiscale_module", True))
#         self.en_sub   = bool(cfg.get("enable_subspace", True))
#         self.en_prob  = bool(cfg.get("enable_probabilistic", False))
#         self.en_vat   = bool(cfg.get("enable_view_adaptive_triplet", True))
#         self.en_cons  = bool(cfg.get("enable_consistency_loss", True))

#         # ---- aux weights / warmup ----
#         self.aw = {
#             'view':   float(cfg.get("aux_weight_view", 0.001)),
#             'phase':  float(cfg.get("aux_weight_phase", 0.001)),
#             'ortho':  float(cfg.get("aux_weight_ortho", 0.001)),
#             'decorr': float(cfg.get("aux_weight_decorr", 0.001)),
#             'vpred':  float(cfg.get("aux_weight_vpred", 0.005)),
#         }
#         self.warmup_s = int(cfg.get("aux_warmup_start", 5000))
#         self.warmup_e = int(cfg.get("aux_warmup_end", 10000))

#         # ---- GPF channel norm ----
#         self.norm = GPFChannelNorm(enabled=bool(cfg.get("enable_channel_norm", True)))

#         # ---- Stem ----
#         stem_ch = int(cfg.get("stem_out_channels", 32))
#         self.adapter = TypedGPFStem(
#             geo_channels=len(self.geo_idx), motion_channels=len(self.mot_idx),
#             residual_channels=len(self.res_idx), out_channels=stem_ch,
#             branch_channels=int(cfg.get("grouped_branch_channels", 24)),
#             residual_branch_channels=int(cfg.get("residual_branch_channels", 12)),
#             use_reliability_gate=bool(cfg.get("use_reliability_gate", True)),
#             gate_strength=float(cfg.get("reliability_gate_strength", 0.75)),
#             primitive_dropout=float(cfg.get("primitive_dropout", 0.05)))

#         # ---- Backbone ----
#         self.backbone = FrameBackbone(
#             stem_ch,
#             channels=cfg.get("backbone_channels", [64,128,256,512]),
#             strides=cfg.get("backbone_strides", [1,2,2,1]),
#             blocks_per_stage=cfg.get("blocks_per_stage", [1,1,1,1]))
#         bb_ch = self.backbone.out_channels

#         # ---- HPP / fallback TP ----
#         bin_num = list(cfg.get("bin_num", [16]))
#         self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)
#         P = int(sum(bin_num))
#         self.TP = TemporalPrimitivePool(pool_type=cfg.get("temporal_pool", "max"))

#         # ---- M1: View ----
#         if self.en_view:
#             self.view_est  = ViewEstimator(len(self.geo_idx), int(cfg.get("view_hidden_dim", 128)))
#             self.view_attn = ViewConditionedPartAttention(P, int(cfg.get("view_attn_hidden", 32)))
#             self.view_ssl  = _ViewSelfSupLoss(len(self.geo_idx))

#         # ---- M2: Phase ----
#         nB = int(cfg.get("num_phase_bins", 8))
#         if self.en_phase:
#             self.phase_est  = ImplicitPhaseEstimator(len(self.mot_idx),
#                                                      int(cfg.get("phase_hidden_dim", 64)),
#                                                      int(cfg.get("phase_num_harmonics", 4)))
#             self.phase_pool = PhaseAlignedPooling(nB, float(cfg.get("phase_kappa", 4.)))
#             self.phase_reg  = _PhaseRegLoss()

#         # ---- M3: Multi-Scale ----
#         if self.en_mst:
#             assert self.en_phase, "MST requires phase"
#             self.mst = MultiScaleTemporalEncoder(bb_ch, nB)

#         # ---- FCs ----
#         emb_ch = int(cfg.get("emb_channels", 256))
#         self.FCs = SeparateFCs(parts_num=P, in_channels=bb_ch, out_channels=emb_ch)

#         # ---- M4: Subspace ----
#         self.id_dim = emb_ch  # default when subspace disabled
#         if self.en_sub:
#             self.id_dim  = int(cfg.get("id_dim", emb_ch - 32))
#             v_dim        = int(cfg.get("view_subspace_dim", 16))
#             c_dim        = emb_ch - self.id_dim - v_dim
#             assert c_dim >= 0
#             self.subspace = SubspaceHead(emb_ch, self.id_dim, v_dim, c_dim)
#             if self.en_view:
#                 self.view_pred = nn.Linear(v_dim, 2)

#         # ---- M5: Probabilistic ----
#         if self.en_prob:
#             self.prob_head = ProbabilisticHead(self.id_dim, bb_ch, P)

#         # ---- VRE on identity subspace ----
#         if self.en_view:
#             self.vre = ViewRotaryEmbedding(self.id_dim)

#         # ---- BNNeck (on emb_id, for CE only) ----
#         self.BNNecks = SeparateBNNecks(
#             parts_num=P, in_channels=self.id_dim,
#             class_num=int(cfg["class_num"]))

#     # --------------------------------------------------------------- helpers
#     def _aux_s(self):
#         it = getattr(self, "iteration", 0)
#         if it < self.warmup_s: return 0.
#         if it >= self.warmup_e: return 1.
#         return (it - self.warmup_s) / max(self.warmup_e - self.warmup_s, 1)

#     def _framewise_hpp(self, f):
#         N, C, T, H, W = f.shape
#         p = self.HPP(f.permute(0,2,1,3,4).reshape(N*T, C, H, W))
#         return p.reshape(N, T, C, -1).permute(0, 2, 1, 3).contiguous()

#     # --------------------------------------------------------------- forward
#     def forward(self, inputs):
#         ipts, labs, typs, vies, seqL = inputs
#         bs = labs.size(0) if torch.is_tensor(labs) else len(labs)

#         # 0. GPF
#         gpf = self.norm(_to_ncthw(ipts[self.input_index],
#                                   expected_channels=self.source_channels))
#         N = gpf.size(0)
#         geo = _select_channels(gpf, self.geo_idx, "geo")
#         mot = _select_channels(gpf, self.mot_idx, "mot")
#         rel = gpf[:, self.rel_idx:self.rel_idx+1] \
#               if self.rel_idx is not None and self.rel_idx < gpf.size(1) else None
#         res = _select_channels(gpf, self.res_idx, "res") if self.res_idx else None
#         fg  = gpf[:, 0:1] if gpf.size(1) > 0 else None

#         # 1. Stem + Backbone
#         stem, _ = self.adapter(geo, mot, reliability=rel, residual=res, foreground=fg)
#         bb = self.backbone(stem)
#         _, Cb, T, Hb, Wb = bb.shape

#         # 2. View estimation
#         theta, sin_cos = None, None
#         if self.en_view:
#             theta, sin_cos = self.view_est(geo)

#         # 3. Phase estimation
#         phase_sc, freq = None, None
#         if self.en_phase:
#             mt = mot.mean(dim=[3, 4])
#             if mt.size(2) != T:
#                 mt = F.adaptive_avg_pool1d(mt, T)
#             phase_sc, freq = self.phase_est(mt)

#         # 4. Temporal path
#         new_path = (self.en_phase or self.en_mst) and (N == bs)
#         if new_path:
#             pf = self._framewise_hpp(bb)                        # [N,Cb,T,P]
#             if self.en_phase:
#                 psc = phase_sc[:, :2]
#                 if psc.size(2) != T:
#                     psc = F.interpolate(psc, T, mode="linear", align_corners=False)
#                 pa = self.phase_pool(pf, psc)                   # [N,Cb,B,P]
#             else:
#                 pa = None
#             if self.en_mst and pa is not None:
#                 feat_out, _ = self.mst(pf, pa)                  # [N,Cb,P]
#             elif pa is not None:
#                 feat_out = pa.max(2)[0]
#             else:
#                 feat_out = pf.max(2)[0]
#         else:
#             feat_out = self.HPP(self.TP(bb, seqL=seqL, batch_size=bs))

#         # 5. View-conditioned part attention  (M1a)
#         if self.en_view and sin_cos is not None:
#             feat_out = self.view_attn(feat_out, sin_cos)

#         # 6. SeparateFCs → full embedding
#         emb = self.FCs(feat_out)                                # [N, emb_ch, P]

#         # 7. Subspace decomposition  (M4)
#         if self.en_sub:
#             emb_id, emb_v, emb_c = self.subspace(emb)
#         else:
#             emb_id, emb_v, emb_c = emb, None, None

#         # 8. Probabilistic head  (M5)
#         logvar = None
#         if self.en_prob:
#             _, logvar = self.prob_head(emb_id, feat_out)

#         # 9. View Rotary Embedding on emb_id  (M1c — identity subspace only)
#         if self.en_view and theta is not None:
#             emb_id_rot = self.vre(emb_id, theta)
#         else:
#             emb_id_rot = emb_id

#         # 10. BNNeck → logits  (CE loss only — not used for inference)
#         _, logits = self.BNNecks(emb_id)

#         # ============================================================
#         #  Build training_feat
#         # ============================================================
#         training_feat = {}

#         # --- primary triplet ---
#         if self.en_prob and logvar is not None:
#             # ProbabilisticTripletLoss interface: (mu, logvar, labels)
#             training_feat["prob_triplet"] = {
#                 "mu": emb_id_rot, "logvar": logvar, "labels": labs}
#         else:
#             td = {"embeddings": emb_id_rot, "labels": labs}
#             if self.en_vat and theta is not None:
#                 td["theta"] = theta           # ViewAdaptiveTripletLoss
#             training_feat["triplet"] = td

#         # --- CE ---
#         training_feat["softmax"] = {"logits": logits, "labels": labs}

#         # --- consistency ---
#         if self.en_cons:
#             training_feat["consistency"] = {"embeddings": emb_id, "labels": labs}

#         # --- auxiliary (pre-weighted tensors, added directly by LossAggregator) ---
#         if self.training:
#             s = self._aux_s()
#             if s > 0:
#                 if self.en_view:
#                     training_feat["_a_view"] = \
#                         self.view_ssl(sin_cos, geo.mean(2), labs) * self.aw['view'] * s
#                 if self.en_phase and phase_sc is not None:
#                     training_feat["_a_phase"] = \
#                         self.phase_reg(phase_sc[:, :2], freq) * self.aw['phase'] * s
#                 if self.en_sub:
#                     training_feat["_a_ortho"] = \
#                         self.subspace.orthogonality_loss() * self.aw['ortho'] * s
#                     training_feat["_a_decorr"] = (
#                         SubspaceHead.decorrelation(emb_id, emb_v)
#                         + SubspaceHead.decorrelation(emb_id, emb_c)
#                     ) * self.aw['decorr'] * s
#                     if self.en_view and sin_cos is not None:
#                         vp = self.view_pred(emb_v.mean(2))
#                         training_feat["_a_vpred"] = \
#                             F.mse_loss(vp, sin_cos) * self.aw['vpred'] * s

#         # ============================================================
#         #  Visual summary
#         # ============================================================
#         vis = {}
#         if gpf.size(1) > 0:
#             v = gpf[:, 0:1].detach().clamp(0, 1)
#             idx = torch.linspace(0, v.size(2)-1, min(8, v.size(2)), device=v.device).long()
#             v = v.index_select(2, idx)
#             n_, c_, t_, h_, w_ = v.shape
#             vis["image/sil"] = v.permute(0,2,1,3,4).reshape(n_*t_, c_, h_, w_)

#         # ============================================================
#         #  Inference: emb_id (identity subspace from FC output)
#         # ============================================================
#         return {
#             "training_feat": training_feat,
#             "visual_summary": vis,
#             "inference_feat": {"embeddings": emb_id},
#         }
