import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


def _safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class ConditionalDecorrelationLoss(BaseLoss):
    """Decorrelate identity embeddings from nuisance/text factors.

    This is a lightweight extension of GaitMax-style decorrelation. It centers each
    mini-batch and penalizes squared cross-correlation. If labels are supplied, a
    within-identity centering term is added whenever an identity has more than one
    sample in the batch, making the penalty less likely to suppress identity cues.
    """

    def __init__(self, loss_term_weight=1.0, eps=1e-6, conditional_weight=0.5):
        super().__init__(loss_term_weight)
        self.eps = eps
        self.conditional_weight = conditional_weight

    def _corr_loss(self, x, y):
        if x is None or y is None or x.numel() == 0 or y.numel() == 0:
            return x.new_tensor(0.) if x is not None else torch.tensor(0.)
        x = x.float(); y = y.float()
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        x = x / x.std(0, keepdim=True).clamp_min(self.eps)
        y = y / y.std(0, keepdim=True).clamp_min(self.eps)
        c = x.T @ y / max(1, x.size(0) - 1)
        return (c ** 2).mean()

    @gather_and_scale_wrapper
    def forward(self, embeddings, nuisance_embeddings, labels=None):
        loss = self._corr_loss(embeddings, nuisance_embeddings)
        cond_loss = embeddings.new_tensor(0.)
        terms = 0
        if labels is not None:
            for lab in torch.unique(labels):
                mask = labels == lab
                if int(mask.sum()) > 1:
                    cond_loss = cond_loss + self._corr_loss(embeddings[mask], nuisance_embeddings[mask])
                    terms += 1
        if terms > 0:
            loss = loss + self.conditional_weight * cond_loss / terms
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class MotionAlignmentLoss(BaseLoss):
    """Align learned kinematic representation to structured motion text/pose descriptors."""

    def __init__(self, loss_term_weight=1.0, temperature=0.07, use_infonce=True):
        super().__init__(loss_term_weight)
        self.temperature = temperature
        self.use_infonce = use_infonce

    @gather_and_scale_wrapper
    def forward(self, kin_embeddings, motion_embeddings):
        kin = kin_embeddings.float()
        mot = motion_embeddings.float()
        if mot.abs().sum() < 1e-9:
            loss = kin.new_tensor(0.)
        elif self.use_infonce and kin.size(0) > 1:
            kin = _safe_normalize(kin)
            mot = _safe_normalize(mot)
            logits = kin @ mot.T / self.temperature
            labels = torch.arange(kin.size(0), device=kin.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        else:
            loss = 1 - F.cosine_similarity(kin, mot, dim=-1).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class CounterfactualConsistencyLoss(BaseLoss):
    """Pull together same-ID samples, emphasizing pairs with different nuisance text."""

    def __init__(self, loss_term_weight=1.0, temperature=0.2, min_nuisance_distance=0.05):
        super().__init__(loss_term_weight)
        self.temperature = temperature
        self.min_nuisance_distance = min_nuisance_distance

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, nuisance_embeddings=None):
        z = _safe_normalize(embeddings.float())
        sim = z @ z.T / self.temperature
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        pos = same & (~eye)
        if nuisance_embeddings is not None and nuisance_embeddings.abs().sum() > 1e-9:
            n = _safe_normalize(nuisance_embeddings.float())
            nd = 1 - n @ n.T
            pos = pos & (nd > self.min_nuisance_distance)
            # Fall back to any same-ID positive if batch lacks counterfactual pairs.
            if pos.sum() == 0:
                pos = same & (~eye)
        if pos.sum() == 0:
            loss = z.new_tensor(0.)
        else:
            sim = sim - sim.max(1, keepdim=True)[0].detach()
            exp_sim = torch.exp(sim) * (~eye).float()
            log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True).clamp_min(1e-6))
            loss = -(log_prob * pos.float()).sum(1) / pos.float().sum(1).clamp_min(1.0)
            loss = loss[pos.any(1)].mean()
        self.info.update({'loss': loss.detach().clone(), 'pairs': pos.sum().detach().clone().float()})
        return loss, self.info


class PhaseConsistencyLoss(BaseLoss):
    """Self-supervised gait phase prior.

    Encourages smooth phase evolution and an anti-phase relation between left/right
    feet. It is intentionally soft: the loss regularizes the representation but does
    not hard-code a normal gait template.
    """

    def __init__(self, loss_term_weight=1.0, smooth_weight=0.5, antiphase_weight=0.5, foot_indices=(7, 8)):
        super().__init__(loss_term_weight)
        self.smooth_weight = smooth_weight
        self.antiphase_weight = antiphase_weight
        self.foot_indices = tuple(foot_indices)

    def forward(self, phase_code, part_mu):
        # phase_code [N,T,2], part_mu [N,T,P,2]
        phase_code = phase_code.float()
        if phase_code.size(1) < 3:
            loss = phase_code.new_tensor(0.)
        else:
            norm_loss = (phase_code.norm(dim=-1) - 1).abs().mean()
            smooth = (phase_code[:, 1:] - phase_code[:, :-1]).pow(2).mean()
            li, ri = self.foot_indices
            li = min(li, part_mu.size(2) - 1); ri = min(ri, part_mu.size(2) - 1)
            l = part_mu[:, :, li, 0] - part_mu[:, :, li, 0].mean(1, keepdim=True)
            r = part_mu[:, :, ri, 0] - part_mu[:, :, ri, 0].mean(1, keepdim=True)
            corr = (l * r).mean(1) / (l.std(1).clamp_min(1e-6) * r.std(1).clamp_min(1e-6))
            anti = F.relu(corr + 0.2).mean()
            loss = norm_loss + self.smooth_weight * smooth + self.antiphase_weight * anti
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class PrivilegedDistillationLoss(BaseLoss):
    """Distill RGB kinematics from optional SMPL/PC/Event teachers."""

    def __init__(self, loss_term_weight=1.0, cosine_weight=0.5, mse_weight=0.5):
        super().__init__(loss_term_weight)
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    @gather_and_scale_wrapper
    def forward(self, student_embeddings, teacher_embeddings, teacher_valid=None):
        s = student_embeddings.float()
        t = teacher_embeddings.float().detach()
        if teacher_valid is None:
            valid = torch.ones(s.size(0), device=s.device, dtype=s.dtype)
        else:
            valid = (teacher_valid.float() > 0).float()
        if valid.sum() < 1:
            loss = s.new_tensor(0.)
        else:
            cos = 1 - F.cosine_similarity(s, t, dim=-1)
            mse = F.mse_loss(s, t, reduction='none').mean(-1)
            loss = (valid * (self.cosine_weight * cos + self.mse_weight * mse)).sum() / valid.sum().clamp_min(1.)
        self.info.update({'loss': loss.detach().clone(), 'valid_ratio': valid.mean().detach().clone()})
        return loss, self.info
