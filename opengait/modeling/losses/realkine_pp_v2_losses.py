"""OpenGait-compatible losses for RealKineGait++ v2.

Drop this file into ``opengait/modeling/losses``.  OpenGait's dynamic loss
registry imports every Python file in that package, so no manual edit to
``losses/__init__.py`` is required.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


def _safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _zero_like_ref(*refs):
    for ref in refs:
        if torch.is_tensor(ref):
            return ref.new_tensor(0.0)
    return torch.tensor(0.0)


class CausalDecorrelationLoss(BaseLoss):
    """Decorrelate identity embeddings from nuisance/text factors.

    This accepts the v2 config fields while remaining lightweight enough for
    OpenGait's standard loss aggregator.  The hard-negative and swap terms are
    soft regularizers instead of requiring an additional generator branch.
    """

    def __init__(self, loss_term_weight=1.0, eps=1e-6, conditional_weight=0.5,
                 hard_negative_weight=0.0, swap_consistency_weight=0.0,
                 nuisance_sim_threshold=0.8):
        super().__init__(loss_term_weight)
        self.eps = eps
        self.conditional_weight = conditional_weight
        self.hard_negative_weight = hard_negative_weight
        self.swap_consistency_weight = swap_consistency_weight
        self.nuisance_sim_threshold = nuisance_sim_threshold

    def _corr_loss(self, x, y):
        if x is None or y is None or (not torch.is_tensor(x)) or (not torch.is_tensor(y)):
            return _zero_like_ref(x, y)
        if x.numel() == 0 or y.numel() == 0 or y.abs().sum() < 1e-9:
            return x.new_tensor(0.0)
        x = x.float(); y = y.float()
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        x = x / x.std(0, keepdim=True).clamp_min(self.eps)
        y = y / y.std(0, keepdim=True).clamp_min(self.eps)
        c = x.T @ y / max(1, x.size(0) - 1)
        return c.pow(2).mean()

    @gather_and_scale_wrapper
    def forward(self, embeddings, nuisance_embeddings, labels=None):
        z = embeddings.float()
        n = nuisance_embeddings.float()
        if n.abs().sum() < 1e-9:
            loss = z.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone(), 'hard_pairs': loss.detach().clone()})
            return loss, self.info

        loss = self._corr_loss(z, n)
        cond_loss = z.new_tensor(0.0)
        terms = 0
        if labels is not None:
            for lab in torch.unique(labels):
                mask = labels == lab
                if int(mask.sum()) > 1:
                    cond_loss = cond_loss + self._corr_loss(z[mask], n[mask])
                    terms += 1
        if terms > 0:
            loss = loss + self.conditional_weight * cond_loss / terms

        # Penalize samples with very similar nuisance but different identities
        # from being too close in identity space.
        hard_pairs = z.new_tensor(0.0)
        if labels is not None and z.size(0) > 1 and self.hard_negative_weight > 0:
            zn = _safe_normalize(z.float())
            nn_ = _safe_normalize(n.float())
            n_sim = nn_ @ nn_.T
            z_sim = zn @ zn.T
            diff_id = ~labels[:, None].eq(labels[None, :])
            eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
            hard = (n_sim > self.nuisance_sim_threshold) & diff_id & (~eye)
            if hard.any():
                hard_pairs = hard.float().sum()
                loss = loss + self.hard_negative_weight * F.relu(z_sim[hard]).mean()

        if self.swap_consistency_weight > 0 and z.size(0) > 1:
            # Batch-level proxy: identity representation should be stable after
            # nuisance shuffling, implemented as covariance shrinkage.
            perm = torch.arange(z.size(0) - 1, -1, -1, device=z.device)
            loss = loss + self.swap_consistency_weight * self._corr_loss(z, n[perm])

        self.info.update({'loss': loss.detach().clone(), 'hard_pairs': hard_pairs.detach().clone()})
        return loss, self.info


# Backward-compatible alias used by the first RealKineGait patch.
ConditionalDecorrelationLoss = CausalDecorrelationLoss


class MotionAlignmentLoss(BaseLoss):
    """Align kinematic embeddings with generated motion-text descriptors."""

    def __init__(self, loss_term_weight=1.0, temperature=0.07, use_infonce=True):
        super().__init__(loss_term_weight)
        self.temperature = temperature
        self.use_infonce = use_infonce

    @gather_and_scale_wrapper
    def forward(self, kin_embeddings, motion_embeddings):
        kin = kin_embeddings.float()
        mot = motion_embeddings.float()
        if mot.abs().sum() < 1e-9:
            loss = kin.new_tensor(0.0)
        elif self.use_infonce and kin.size(0) > 1:
            kin = _safe_normalize(kin)
            mot = _safe_normalize(mot)
            logits = kin @ mot.T / self.temperature
            labels = torch.arange(kin.size(0), device=kin.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        else:
            loss = 1 - F.cosine_similarity(_safe_normalize(kin), _safe_normalize(mot), dim=-1).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class CounterfactualConsistencyLoss(BaseLoss):
    """Pull together same-ID samples, especially under different nuisance text."""

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
            cf_pos = pos & (nd > self.min_nuisance_distance)
            if cf_pos.sum() > 0:
                pos = cf_pos
        if pos.sum() == 0:
            loss = z.new_tensor(0.0)
        else:
            sim = sim - sim.max(1, keepdim=True)[0].detach()
            exp_sim = torch.exp(sim) * (~eye).float()
            log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True).clamp_min(1e-6))
            loss = -(log_prob * pos.float()).sum(1) / pos.float().sum(1).clamp_min(1.0)
            loss = loss[pos.any(1)].mean()
        self.info.update({'loss': loss.detach().clone(), 'pairs': pos.sum().detach().clone().float()})
        return loss, self.info


class PhaseCanonicalConsistencyLoss(BaseLoss):
    """Self-supervised gait phase regularization for v2.

    - unit-norm phase code
    - smooth temporal phase trajectory
    - soft anti-phase relation between left/right foot part centroids
    - optional diversity over canonical phase bins to discourage collapse
    """

    def __init__(self, loss_term_weight=1.0, smooth_weight=0.5,
                 antiphase_weight=0.5, diversity_weight=0.0,
                 foot_indices=(7, 8)):
        super().__init__(loss_term_weight)
        self.smooth_weight = smooth_weight
        self.antiphase_weight = antiphase_weight
        self.diversity_weight = diversity_weight
        self.foot_indices = tuple(foot_indices)

    def forward(self, phase_code, part_mu, cycle_tokens=None):
        phase_code = phase_code.float()
        if phase_code.size(1) < 3:
            loss = phase_code.new_tensor(0.0)
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
        diversity = phase_code.new_tensor(0.0)
        if cycle_tokens is not None and self.diversity_weight > 0 and cycle_tokens.size(1) > 1:
            # Encourage different phase bins to retain non-identical descriptors.
            x = _safe_normalize(cycle_tokens.float().mean(2), dim=-1)  # [N,K,C]
            sim = torch.einsum('nkc,nlc->nkl', x, x)
            eye = torch.eye(sim.size(-1), dtype=torch.bool, device=sim.device).unsqueeze(0)
            diversity = F.relu(sim.masked_select(~eye) - 0.5).mean()
            loss = loss + self.diversity_weight * diversity
        self.info.update({'loss': loss.detach().clone(), 'diversity': diversity.detach().clone()})
        return loss, self.info


# Backward-compatible alias used by the v1 patch.
PhaseConsistencyLoss = PhaseCanonicalConsistencyLoss


class UncertaintyAwareDistillationLoss(BaseLoss):
    """Distill the RGB kinematic student from optional privileged teachers.

    Expected inputs from ``RealKineGaitPP_v2.forward``:
    - student_embeddings: [N,D]
    - teacher_stack: [N,M,D]
    - log_var_stack: [N,M]
    - valid_stack: [N,M]
    """

    def __init__(self, loss_term_weight=1.0, cosine_weight=0.5,
                 mse_weight=0.5, calibration_weight=0.0):
        super().__init__(loss_term_weight)
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.calibration_weight = calibration_weight

    def forward(self, student_embeddings, teacher_stack, log_var_stack=None, valid_stack=None):
        s = student_embeddings.float()
        t = teacher_stack.float().detach()
        n, m, d = t.shape
        if valid_stack is None:
            valid = torch.ones(n, m, device=s.device, dtype=s.dtype)
        else:
            valid = (valid_stack.float() > 0).float()
        if valid.sum() < 1:
            loss = s.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone(), 'valid_ratio': valid.mean().detach().clone()})
            return loss, self.info
        if log_var_stack is None:
            precision = valid / valid.sum(1, keepdim=True).clamp_min(1.0)
        else:
            precision = torch.exp(-log_var_stack.float().detach()).clamp(max=1e4) * valid
            precision = precision / precision.sum(1, keepdim=True).clamp_min(1e-6)
        teacher = (precision.unsqueeze(-1) * t).sum(1)
        cos = 1 - F.cosine_similarity(_safe_normalize(s), _safe_normalize(teacher), dim=-1)
        mse = F.mse_loss(s, teacher, reduction='none').mean(-1)
        per = self.cosine_weight * cos + self.mse_weight * mse
        sample_valid = (valid.sum(1) > 0).float()
        loss = (per * sample_valid).sum() / sample_valid.sum().clamp_min(1.0)
        if self.calibration_weight > 0 and log_var_stack is not None:
            # Keep log variances numerically sane; teachers are detached, but the
            # term is useful for logging and for future trainable teachers.
            calib = (valid * log_var_stack.float().pow(2)).sum() / valid.sum().clamp_min(1.0)
            loss = loss + self.calibration_weight * calib
        self.info.update({'loss': loss.detach().clone(), 'valid_ratio': valid.mean().detach().clone()})
        return loss, self.info


class PrivilegedDistillationLoss(BaseLoss):
    """Compatibility wrapper for the v1 ``distill3d`` loss."""

    def __init__(self, loss_term_weight=1.0, cosine_weight=0.5, mse_weight=0.5):
        super().__init__(loss_term_weight)
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    @gather_and_scale_wrapper
    def forward(self, student_embeddings, teacher_embeddings, teacher_valid=None):
        s = student_embeddings.float()
        t = teacher_embeddings.float().detach()
        valid = torch.ones(s.size(0), device=s.device, dtype=s.dtype) if teacher_valid is None else (teacher_valid.float() > 0).float()
        if valid.sum() < 1:
            loss = s.new_tensor(0.0)
        else:
            cos = 1 - F.cosine_similarity(_safe_normalize(s), _safe_normalize(t), dim=-1)
            mse = F.mse_loss(s, t, reduction='none').mean(-1)
            loss = (valid * (self.cosine_weight * cos + self.mse_weight * mse)).sum() / valid.sum().clamp_min(1.0)
        self.info.update({'loss': loss.detach().clone(), 'valid_ratio': valid.mean().detach().clone()})
        return loss, self.info


class SilhouetteForegroundLoss(BaseLoss):
    """Keep learned part attention inside the silhouette foreground."""

    def __init__(self, loss_term_weight=1.0, eps=1e-6):
        super().__init__(loss_term_weight)
        self.eps = eps

    def forward(self, attn_map, sil_mask=None):
        if sil_mask is None:
            loss = attn_map.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone(), 'fg_ratio': loss.detach().clone()})
            return loss, self.info
        attn = attn_map.float()  # [N,T,P,H,W]
        sil = sil_mask.float()
        if sil.dim() == 5:
            sil = sil[:, 0].unsqueeze(2)  # [N,T,1,H,W]
        elif sil.dim() == 4:
            sil = sil.unsqueeze(2)
        sil = sil.clamp(0, 1)
        bg_mass = (attn * (1.0 - sil)).sum((-1, -2))
        loss = bg_mass.mean()
        fg_ratio = (attn * sil).sum((-1, -2)).mean()
        self.info.update({'loss': loss.detach().clone(), 'fg_ratio': fg_ratio.detach().clone()})
        return loss, self.info


class BackgroundAdversarialLoss(BaseLoss):
    """Suppress identity information carried by background embeddings.

    Without adding an auxiliary classifier to the model, the safest OpenGait
    compatible proxy is to shrink background embedding energy and batch
    discriminability.  The signature keeps ``labels`` for config compatibility.
    """

    def __init__(self, loss_term_weight=1.0, temperature=0.2):
        super().__init__(loss_term_weight)
        self.temperature = temperature

    def forward(self, bg_embed, labels=None):
        if bg_embed is None:
            loss = torch.tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        z = bg_embed.float()
        energy = z.pow(2).mean()
        loss = energy
        if labels is not None and z.size(0) > 1:
            zn = _safe_normalize(z)
            sim = zn @ zn.T / max(self.temperature, 1e-6)
            eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
            # Penalize non-uniform pairwise similarity structure.
            loss = loss + 0.05 * sim.masked_select(~eye).pow(2).mean()
        self.info.update({'loss': loss.detach().clone(), 'energy': energy.detach().clone()})
        return loss, self.info


class PoseAlignmentLoss(BaseLoss):
    """Align learned kinematic embeddings with pose-teacher embeddings."""

    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, kin_embeddings, pose_embeddings):
        kin = kin_embeddings.float()
        pose = pose_embeddings.float()
        if pose.abs().sum() < 1e-9:
            loss = kin.new_tensor(0.0)
        else:
            loss = 1 - F.cosine_similarity(_safe_normalize(kin), _safe_normalize(pose.detach()), dim=-1).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class PartJointAlignmentLoss(BaseLoss):
    """Align learned part centroids with high-confidence 2D pose joints."""

    def __init__(self, loss_term_weight=1.0, sigma=0.15, min_confidence=0.3):
        super().__init__(loss_term_weight)
        self.sigma = sigma
        self.min_confidence = min_confidence

    def forward(self, part_mu, pose=None):
        if part_mu is None or pose is None:
            loss = _zero_like_ref(part_mu, pose)
            self.info.update({'loss': loss.detach().clone(), 'coverage': loss.detach().clone()})
            return loss, self.info
        pose = pose.float()
        part_mu = part_mu.float()
        if pose.dim() == 3 and pose.size(-1) % 3 == 0:
            pose = pose.reshape(pose.size(0), pose.size(1), -1, 3)
        if pose.dim() != 4 or pose.size(-1) < 3:
            loss = part_mu.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone(), 'coverage': loss.detach().clone()})
            return loss, self.info

        # Pad/truncate temporal length when restored OpenGait batches differ by one
        # due to preprocessing quirks.
        t = min(part_mu.size(1), pose.size(1))
        part_mu = part_mu[:, :t]
        pose = pose[:, :t]

        joint_xy = pose[..., :2]
        joint_conf = pose[..., 2]
        valid = (joint_conf > self.min_confidence).float()
        dist = (part_mu.unsqueeze(3) - joint_xy.unsqueeze(2)).pow(2).sum(-1)  # [N,T,P,J]
        dist_masked = dist + (1.0 - valid.unsqueeze(2)) * 1e6
        min_dist, _ = dist_masked.min(dim=3)
        has_valid = valid.sum(-1, keepdim=True) > 0
        if has_valid.float().sum() < 1:
            loss = part_mu.new_tensor(0.0)
        else:
            loss = (min_dist * has_valid.float()).sum() / has_valid.float().sum().clamp_min(1.0) / part_mu.size(2)
        joint_min_dist, _ = dist.min(dim=2)
        covered = ((joint_min_dist < self.sigma ** 2) & (valid > 0)).float()
        coverage = covered.sum() / valid.sum().clamp_min(1.0)
        self.info.update({'loss': loss.detach().clone(), 'coverage': coverage.detach().clone()})
        return loss, self.info
