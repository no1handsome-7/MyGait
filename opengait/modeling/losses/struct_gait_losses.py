# -*- coding: utf-8 -*-
"""
StructGait custom losses.

Drop into: OpenGait/opengait/modeling/losses/struct_gait_losses.py
Auto-discovered by the losses __init__.py.

Classes:
    ViewAdaptiveTripletLoss   — margin adapts to view-pair difficulty
    ProbabilisticTripletLoss  — MLS-based triplet on (μ, logσ²) embeddings
    ConsistencyLoss           — penalises intra-class max deviation
"""

import torch
import torch.nn.functional as F
from .base import BaseLoss, gather_and_scale_wrapper


# ======================================================================
#  ViewAdaptiveTripletLoss
# ======================================================================

class ViewAdaptiveTripletLoss(BaseLoss):
    """
    Triplet loss whose margin grows with the view-angle difference
    between anchor and positive.  Falls back to fixed margin when
    ``theta`` is absent from the kwargs.
    """

    def __init__(self, base_margin=0.2, view_margin_scale=0.15,
                 loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.base_margin = base_margin
        self.view_margin_scale = view_margin_scale

    # --- helpers (identical to OpenGait TripletLoss) ---
    @staticmethod
    def _euclidean(x, y):
        """x,y: [P, N, C] → [P, N, N]"""
        x2 = (x ** 2).sum(-1, keepdim=True)
        y2 = (y ** 2).sum(-1, keepdim=True)
        return torch.sqrt(F.relu(x2 + y2.transpose(1, 2) - 2 * x @ y.transpose(1, 2)))

    @staticmethod
    def _reduce(loss):
        s = loss.sum(-1);  n = (loss != 0).sum(-1).float()
        avg = s / (n + 1e-9);  avg[n == 0] = 0
        return avg, n

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, theta=None):
        """
        embeddings : [N, C, P]
        labels     : [N]
        theta      : [N]  estimated viewpoint angle (optional)
        """
        emb = embeddings.permute(2, 0, 1).contiguous().float()   # [P,N,C]
        dist = self._euclidean(emb, emb)                         # [P,N,N]
        P, N, _ = dist.shape
        mean_dist = dist.mean((1, 2))

        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        diffenc = ~matches

        ap_dist = dist[:, matches].view(P, N, -1, 1)
        an_dist = dist[:, diffenc].view(P, N, 1, -1)

        # ---- adaptive margin ----
        if theta is not None:
            # The model should learn identity geometry, not exploit the adaptive
            # margin by moving theta.  Detaching here keeps VAT safe even if the
            # caller forgets to detach.
            theta = theta.detach()
            delta = theta.unsqueeze(1) - theta.unsqueeze(0)         # [N,N]
            vdiff = torch.abs(torch.sin(delta / 2))                 # ∈[0,1]
            m_ap = (self.base_margin
                    + self.view_margin_scale * vdiff[matches].view(N, -1, 1))
            m_ap = m_ap.unsqueeze(0).expand(P, -1, -1, -1)          # [P,N,K+,1]
        else:
            m_ap = self.base_margin

        loss = F.relu(ap_dist - an_dist + m_ap)
        hard_loss = loss.reshape(P, -1).max(-1)[0]
        loss_avg, loss_num = self._reduce(loss.reshape(P, -1))

        self.info.update({
            'loss':      loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num':  loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
        })
        return loss_avg, self.info


# ======================================================================
#  ProbabilisticTripletLoss
# ======================================================================

class ProbabilisticTripletLoss(BaseLoss):
    """
    Triplet loss on probabilistic (μ, log σ²) embeddings.

    Uses Mutual-Likelihood-Score (MLS) as the similarity measure:
    dimensions with large σ contribute less, so the model naturally
    down-weights uncertain features.

    An auxiliary regulariser encourages:
      • mean σ ≈ 1  (prevents collapse to a point or explosion)
      • σ variance > 0 across samples (prevents constant σ)
    """

    def __init__(self, margin=0.2, sigma_reg=0.01, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.margin = margin
        self.sigma_reg = sigma_reg

    @staticmethod
    def _mls_score(mu_a, lv_a, mu_b, lv_b):
        """Mutual-Likelihood Score between two sets of Gaussian embeddings.
        All inputs [N, D].  Returns [N, N] (higher = more similar)."""
        d2 = (mu_a.unsqueeze(1) - mu_b.unsqueeze(0)) ** 2        # [N,N,D]
        vs = lv_a.unsqueeze(1).exp() + lv_b.unsqueeze(0).exp()   # [N,N,D]
        return -(d2 / vs + vs.log()).sum(-1)                      # [N,N]

    @gather_and_scale_wrapper
    def forward(self, mu, logvar, labels):
        """
        mu     : [N, C, P]
        logvar : [N, C, P]
        labels : [N]
        """
        N = mu.size(0)
        mu_f = mu.reshape(N, -1)                                  # [N, C*P]
        lv_f = logvar.reshape(N, -1)

        score = self._mls_score(mu_f, lv_f, mu_f, lv_f)          # [N, N]

        matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
        diffenc = ~matches

        # MLS is a similarity → positive pairs should score HIGHER
        ap = score[matches].view(N, -1, 1)
        an = score[diffenc].view(N, 1, -1)

        loss = F.relu(an - ap + self.margin)
        tri = loss[loss > 0].mean() if (loss > 0).any() else loss.mean()

        # σ regularisation
        sigma_mean = logvar.exp().mean()
        reg_mean   = (sigma_mean - 1.0).abs()
        reg_div    = -logvar.exp().mean(dim=[1, 2]).var()          # encourage diversity

        total = tri + self.sigma_reg * (reg_mean + reg_div)

        self.info.update({
            'loss':       total.detach().clone(),
            'tri':        tri.detach().clone(),
            'sigma_mean': sigma_mean.detach().clone(),
            'sigma_var':  (-reg_div).detach().clone(),
        })
        return total, self.info


# ======================================================================
#  ConsistencyLoss
# ======================================================================

class ConsistencyLoss(BaseLoss):
    """
    Penalise the **maximum** intra-class deviation per identity.

    Standard triplet only needs *one* gallery hit per query.  This loss
    requires *all* sequences of a person to cluster tightly, targeting
    mAP / mINP improvement rather than Rank-1.
    """

    def __init__(self, loss_term_weight=0.1):
        super().__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        embeddings : [N, C, P]
        labels     : [N]
        """
        feats = F.normalize(embeddings.reshape(embeddings.size(0), -1), dim=-1)
        devs = []
        for lab in labels.unique():
            mask = labels == lab
            if mask.sum() < 2:
                continue
            cls = feats[mask]
            ctr = cls.mean(0, keepdim=True)
            d2 = ((cls - ctr) ** 2).sum(-1)
            devs.append(d2.max())                     # worst outlier per class

        loss = torch.stack(devs).mean() if devs else embeddings.sum() * 0.0
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info




# # -*- coding: utf-8 -*-
# """
# StructGait custom losses.

# Drop into: OpenGait/opengait/modeling/losses/struct_gait_losses.py
# Auto-discovered by the losses __init__.py.

# Classes:
#     ViewAdaptiveTripletLoss   — margin adapts to view-pair difficulty
#     ProbabilisticTripletLoss  — MLS-based triplet on (μ, logσ²) embeddings
#     ConsistencyLoss           — penalises intra-class max deviation
# """

# import torch
# import torch.nn.functional as F
# from .base import BaseLoss, gather_and_scale_wrapper


# # ======================================================================
# #  ViewAdaptiveTripletLoss
# # ======================================================================

# class ViewAdaptiveTripletLoss(BaseLoss):
#     """
#     Triplet loss whose margin grows with the view-angle difference
#     between anchor and positive.  Falls back to fixed margin when
#     ``theta`` is absent from the kwargs.
#     """

#     def __init__(self, base_margin=0.2, view_margin_scale=0.15,
#                  loss_term_weight=1.0):
#         super().__init__(loss_term_weight)
#         self.base_margin = base_margin
#         self.view_margin_scale = view_margin_scale

#     # --- helpers (identical to OpenGait TripletLoss) ---
#     @staticmethod
#     def _euclidean(x, y):
#         """x,y: [P, N, C] → [P, N, N]"""
#         x2 = (x ** 2).sum(-1, keepdim=True)
#         y2 = (y ** 2).sum(-1, keepdim=True)
#         return torch.sqrt(F.relu(x2 + y2.transpose(1, 2) - 2 * x @ y.transpose(1, 2)))

#     @staticmethod
#     def _reduce(loss):
#         s = loss.sum(-1);  n = (loss != 0).sum(-1).float()
#         avg = s / (n + 1e-9);  avg[n == 0] = 0
#         return avg, n

#     @gather_and_scale_wrapper
#     def forward(self, embeddings, labels, theta=None):
#         """
#         embeddings : [N, C, P]
#         labels     : [N]
#         theta      : [N]  estimated viewpoint angle (optional)
#         """
#         emb = embeddings.permute(2, 0, 1).contiguous().float()   # [P,N,C]
#         dist = self._euclidean(emb, emb)                         # [P,N,N]
#         P, N, _ = dist.shape
#         mean_dist = dist.mean((1, 2))

#         matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
#         diffenc = ~matches

#         ap_dist = dist[:, matches].view(P, N, -1, 1)
#         an_dist = dist[:, diffenc].view(P, N, 1, -1)

#         # ---- adaptive margin ----
#         if theta is not None:
#             delta = theta.unsqueeze(1) - theta.unsqueeze(0)         # [N,N]
#             vdiff = torch.abs(torch.sin(delta / 2))                 # ∈[0,1]
#             m_ap = (self.base_margin
#                     + self.view_margin_scale * vdiff[matches].view(N, -1, 1))
#             m_ap = m_ap.unsqueeze(0).expand(P, -1, -1, -1)          # [P,N,K+,1]
#         else:
#             m_ap = self.base_margin

#         loss = F.relu(ap_dist - an_dist + m_ap)
#         hard_loss = loss.reshape(P, -1).max(-1)[0]
#         loss_avg, loss_num = self._reduce(loss.reshape(P, -1))

#         self.info.update({
#             'loss':      loss_avg.detach().clone(),
#             'hard_loss': hard_loss.detach().clone(),
#             'loss_num':  loss_num.detach().clone(),
#             'mean_dist': mean_dist.detach().clone(),
#         })
#         return loss_avg, self.info


# # ======================================================================
# #  ProbabilisticTripletLoss
# # ======================================================================

# class ProbabilisticTripletLoss(BaseLoss):
#     """
#     Triplet loss on probabilistic (μ, log σ²) embeddings.

#     Uses Mutual-Likelihood-Score (MLS) as the similarity measure:
#     dimensions with large σ contribute less, so the model naturally
#     down-weights uncertain features.

#     An auxiliary regulariser encourages:
#       • mean σ ≈ 1  (prevents collapse to a point or explosion)
#       • σ variance > 0 across samples (prevents constant σ)
#     """

#     def __init__(self, margin=0.2, sigma_reg=0.01, loss_term_weight=1.0):
#         super().__init__(loss_term_weight)
#         self.margin = margin
#         self.sigma_reg = sigma_reg

#     @staticmethod
#     def _mls_score(mu_a, lv_a, mu_b, lv_b):
#         """Mutual-Likelihood Score between two sets of Gaussian embeddings.
#         All inputs [N, D].  Returns [N, N] (higher = more similar)."""
#         d2 = (mu_a.unsqueeze(1) - mu_b.unsqueeze(0)) ** 2        # [N,N,D]
#         vs = lv_a.unsqueeze(1).exp() + lv_b.unsqueeze(0).exp()   # [N,N,D]
#         return -(d2 / vs + vs.log()).sum(-1)                      # [N,N]

#     @gather_and_scale_wrapper
#     def forward(self, mu, logvar, labels):
#         """
#         mu     : [N, C, P]
#         logvar : [N, C, P]
#         labels : [N]
#         """
#         N = mu.size(0)
#         mu_f = mu.reshape(N, -1)                                  # [N, C*P]
#         lv_f = logvar.reshape(N, -1)

#         score = self._mls_score(mu_f, lv_f, mu_f, lv_f)          # [N, N]

#         matches = (labels.unsqueeze(1) == labels.unsqueeze(0)).bool()
#         diffenc = ~matches

#         # MLS is a similarity → positive pairs should score HIGHER
#         ap = score[matches].view(N, -1, 1)
#         an = score[diffenc].view(N, 1, -1)

#         loss = F.relu(an - ap + self.margin)
#         tri = loss[loss > 0].mean() if (loss > 0).any() else loss.mean()

#         # σ regularisation
#         sigma_mean = logvar.exp().mean()
#         reg_mean   = (sigma_mean - 1.0).abs()
#         reg_div    = -logvar.exp().mean(dim=[1, 2]).var()          # encourage diversity

#         total = tri + self.sigma_reg * (reg_mean + reg_div)

#         self.info.update({
#             'loss':       total.detach().clone(),
#             'tri':        tri.detach().clone(),
#             'sigma_mean': sigma_mean.detach().clone(),
#             'sigma_var':  (-reg_div).detach().clone(),
#         })
#         return total, self.info


# # ======================================================================
# #  ConsistencyLoss
# # ======================================================================

# class ConsistencyLoss(BaseLoss):
#     """
#     Penalise the **maximum** intra-class deviation per identity.

#     Standard triplet only needs *one* gallery hit per query.  This loss
#     requires *all* sequences of a person to cluster tightly, targeting
#     mAP / mINP improvement rather than Rank-1.
#     """

#     def __init__(self, loss_term_weight=0.1):
#         super().__init__(loss_term_weight)

#     @gather_and_scale_wrapper
#     def forward(self, embeddings, labels):
#         """
#         embeddings : [N, C, P]
#         labels     : [N]
#         """
#         feats = F.normalize(embeddings.reshape(embeddings.size(0), -1), dim=-1)
#         devs = []
#         for lab in labels.unique():
#             mask = labels == lab
#             if mask.sum() < 2:
#                 continue
#             cls = feats[mask]
#             ctr = cls.mean(0, keepdim=True)
#             d2 = ((cls - ctr) ** 2).sum(-1)
#             devs.append(d2.max())                     # worst outlier per class

#         loss = torch.stack(devs).mean() if devs else embeddings.sum() * 0.0
#         self.info.update({'loss': loss.detach().clone()})
#         return loss, self.info
