"""RealKineGait++ loss functions.

Covers three primary innovations and silhouette integration:
  1. Phase-Canonical losses — cycle-token consistency, phase-aligned triplet.
  2. Causal Nuisance Intervention — conditional decorrelation plus explicit
     do-operation counterfactual objectives.
  3. Uncertainty-aware Privileged Distillation.
  4. Silhouette — foreground attention penalty + background adversarial loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


def _safe_normalize(x, dim=-1, eps=1e-6):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# ===================================================================
# Innovation 1 : Phase-Canonical Losses
# ===================================================================

class PhaseCanonicalConsistencyLoss(BaseLoss):
    """Regularise the phase-canonical module.

    Objectives:
      1. **Unit-circle normalisation** — phase code should sit on the unit
         circle: ‖(sinθ, cosθ)‖ ≈ 1.
      2. **Temporal smoothness** — adjacent frames should have nearby phase
         values (no wild jumps).
      3. **Anti-phase foot prior** — left and right foot centroids should be
         roughly π apart in phase (anti-correlated horizontal motion).
      4. **Cycle-token diversity** — different phase bins should capture
         different information (avoid bin collapse).
    """

    def __init__(self, loss_term_weight=1.0,
                 smooth_weight=0.5, antiphase_weight=0.5,
                 diversity_weight=0.2, foot_indices=(7, 8)):
        super().__init__(loss_term_weight)
        self.smooth_weight = smooth_weight
        self.antiphase_weight = antiphase_weight
        self.diversity_weight = diversity_weight
        self.foot_indices = tuple(foot_indices)

    def forward(self, phase_code, part_mu, cycle_tokens=None):
        """
        Args:
            phase_code:    [N, T, 2]   (sin θ, cos θ)
            part_mu:       [N, T, P, 2]
            cycle_tokens:  [N, K, P, C] or None
        """
        phase_code = phase_code.float()

        if phase_code.size(1) < 3:
            loss = phase_code.new_tensor(0.)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        # 1. Unit-circle normalisation
        norm_loss = (phase_code.norm(dim=-1) - 1).abs().mean()

        # 2. Temporal smoothness
        smooth = (phase_code[:, 1:] - phase_code[:, :-1]).pow(2).mean()

        # 3. Anti-phase foot prior
        p = part_mu.size(2)
        li = min(self.foot_indices[0], p - 1)
        ri = min(self.foot_indices[1], p - 1)
        lx = part_mu[:, :, li, 0]
        rx = part_mu[:, :, ri, 0]
        lx = lx - lx.mean(1, keepdim=True)
        rx = rx - rx.mean(1, keepdim=True)
        corr = (lx * rx).mean(1) / (
            lx.std(1).clamp_min(1e-6) * rx.std(1).clamp_min(1e-6))
        anti = F.relu(corr + 0.2).mean()

        # 4. Cycle-token diversity (avoid bin collapse)
        div_loss = phase_code.new_tensor(0.)
        if cycle_tokens is not None and cycle_tokens.size(1) > 1:
            # Cosine sim between all pairs of phase bins → push apart
            ct = cycle_tokens.mean(2)              # [N, K, C]
            ct_n = _safe_normalize(ct, dim=-1)
            sim = torch.bmm(ct_n, ct_n.transpose(1, 2))   # [N, K, K]
            k = sim.size(1)
            eye = torch.eye(k, device=sim.device, dtype=sim.dtype)
            off_diag = sim * (1 - eye).unsqueeze(0)
            div_loss = off_diag.pow(2).mean()

        loss = (norm_loss
                + self.smooth_weight * smooth
                + self.antiphase_weight * anti
                + self.diversity_weight * div_loss)

        self.info.update({
            'loss': loss.detach().clone(),
            'norm': norm_loss.detach().clone(),
            'smooth': smooth.detach().clone(),
            'anti': anti.detach().clone(),
            'diversity': div_loss.detach().clone(),
        })
        return loss, self.info


# ===================================================================
# Innovation 2 : Causal Nuisance Intervention Losses
# ===================================================================

class CausalDecorrelationLoss(BaseLoss):
    """Enhanced decorrelation with explicit causal intervention semantics.

    Beyond standard cross-correlation penalisation (as in GaitMax), this loss:
      - Performs within-identity conditional decorrelation.
      - Applies a *cross-identity same-nuisance hard negative* penalty that
        pushes apart samples with similar clothing but different identities.
      - Adds a nuisance-swap consistency term: for same-ID pairs whose
        nuisance descriptions differ, z_id must remain close while
        z_nuisance must differ.

    These correspond to the causal interventions:
      do(clothing = changed), do(viewpoint = changed), do(background = changed)
    while requiring P(ID | z_id) invariant.
    """

    def __init__(self, loss_term_weight=1.0, eps=1e-6,
                 conditional_weight=0.5,
                 hard_negative_weight=0.3,
                 swap_consistency_weight=0.3,
                 nuisance_sim_threshold=0.8):
        super().__init__(loss_term_weight)
        self.eps = eps
        self.conditional_weight = conditional_weight
        self.hard_negative_weight = hard_negative_weight
        self.swap_consistency_weight = swap_consistency_weight
        self.nuisance_sim_threshold = nuisance_sim_threshold

    def _corr_loss(self, x, y):
        if x is None or y is None or x.numel() == 0 or y.numel() == 0:
            return x.new_tensor(0.) if x is not None else torch.tensor(0.)
        x = x.float()
        y = y.float()
        x = x - x.mean(0, keepdim=True)
        y = y - y.mean(0, keepdim=True)
        x = x / x.std(0, keepdim=True).clamp_min(self.eps)
        y = y / y.std(0, keepdim=True).clamp_min(self.eps)
        c = x.T @ y / max(1, x.size(0) - 1)
        return (c ** 2).mean()

    @gather_and_scale_wrapper
    def forward(self, embeddings, nuisance_embeddings, labels=None):
        z = embeddings.float()
        nz = nuisance_embeddings.float()

        # --- Standard decorrelation ---
        corr_loss = self._corr_loss(z, nz)

        # --- Conditional (within-identity) decorrelation ---
        cond_loss = z.new_tensor(0.)
        cond_terms = 0
        if labels is not None:
            for lab in torch.unique(labels):
                mask = labels == lab
                if int(mask.sum()) > 1:
                    cond_loss = cond_loss + self._corr_loss(
                        z[mask], nz[mask])
                    cond_terms += 1
        if cond_terms > 0:
            cond_loss = cond_loss / cond_terms

        # --- Cross-identity same-nuisance hard negative ---
        hn_loss = z.new_tensor(0.)
        if labels is not None and nz.abs().sum() > 1e-9:
            nz_norm = _safe_normalize(nz)
            nsim = nz_norm @ nz_norm.T           # [B, B]
            diff_id = labels[:, None] != labels[None, :]
            same_nuis = nsim > self.nuisance_sim_threshold
            # Hard negatives: different ID but very similar nuisance
            hard_neg_mask = diff_id & same_nuis
            if hard_neg_mask.any():
                z_norm = _safe_normalize(z)
                id_sim = z_norm @ z_norm.T       # [B, B]
                # These should be dissimilar in identity → penalise high sim
                hn_loss = F.relu(
                    id_sim[hard_neg_mask] - 0.0).mean()

        # --- Nuisance-swap consistency ---
        swap_loss = z.new_tensor(0.)
        if labels is not None and nz.abs().sum() > 1e-9:
            same_id = labels[:, None] == labels[None, :]
            eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
            same_id_pairs = same_id & ~eye
            if same_id_pairs.any():
                nz_norm = _safe_normalize(nz)
                nd = 1 - nz_norm @ nz_norm.T
                # Pairs: same ID, different nuisance
                diff_nuis_pairs = same_id_pairs & (nd > 0.1)
                if diff_nuis_pairs.any():
                    z_norm = _safe_normalize(z)
                    id_sim = z_norm @ z_norm.T
                    # z_id should stay close → pull
                    swap_loss = (1 - id_sim[diff_nuis_pairs]).mean()

        loss = (corr_loss
                + self.conditional_weight * cond_loss
                + self.hard_negative_weight * hn_loss
                + self.swap_consistency_weight * swap_loss)

        self.info.update({
            'loss': loss.detach().clone(),
            'corr': corr_loss.detach().clone(),
            'cond': cond_loss.detach().clone(),
            'hard_neg': hn_loss.detach().clone(),
            'swap': swap_loss.detach().clone(),
        })
        return loss, self.info


class CounterfactualConsistencyLoss(BaseLoss):
    """Pull same-ID samples together, emphasizing different-nuisance positives.

    Identical to v0.1 but now also works as an intervention objective
    when paired with silhouette/augmentation-based counterfactuals.
    """

    def __init__(self, loss_term_weight=1.0, temperature=0.2,
                 min_nuisance_distance=0.05):
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

        if (nuisance_embeddings is not None
                and nuisance_embeddings.abs().sum() > 1e-9):
            n = _safe_normalize(nuisance_embeddings.float())
            nd = 1 - n @ n.T
            pos_cf = pos & (nd > self.min_nuisance_distance)
            if pos_cf.sum() > 0:
                pos = pos_cf

        if pos.sum() == 0:
            loss = z.new_tensor(0.)
        else:
            sim = sim - sim.max(1, keepdim=True)[0].detach()
            exp_sim = torch.exp(sim) * (~eye).float()
            log_prob = sim - torch.log(
                exp_sim.sum(1, keepdim=True).clamp_min(1e-6))
            loss = -(log_prob * pos.float()).sum(1) / pos.float().sum(
                1).clamp_min(1.0)
            loss = loss[pos.any(1)].mean()

        self.info.update({
            'loss': loss.detach().clone(),
            'pairs': pos.sum().detach().clone().float(),
        })
        return loss, self.info


class MotionAlignmentLoss(BaseLoss):
    """Align kinematic embeddings to motion text / pose descriptors."""

    def __init__(self, loss_term_weight=1.0, temperature=0.07,
                 use_infonce=True):
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
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.T, labels))
        else:
            loss = 1 - F.cosine_similarity(kin, mot, dim=-1).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class PoseAlignmentLoss(BaseLoss):
    """Align kinematic embeddings to pose teacher embeddings.

    Uses a combination of cosine and MSE loss, only when the pose teacher
    is valid (non-zero).
    """

    def __init__(self, loss_term_weight=1.0, cosine_weight=0.5,
                 mse_weight=0.5):
        super().__init__(loss_term_weight)
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    @gather_and_scale_wrapper
    def forward(self, kin_embeddings, pose_embeddings):
        k = kin_embeddings.float()
        p = pose_embeddings.float().detach()
        if p.abs().sum() < 1e-9:
            loss = k.new_tensor(0.)
        else:
            cos = 1 - F.cosine_similarity(k, p, dim=-1).mean()
            mse = F.mse_loss(k, p)
            loss = self.cosine_weight * cos + self.mse_weight * mse
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


# ===================================================================
# Innovation 3 : Uncertainty-Aware Privileged Distillation
# ===================================================================

class UncertaintyAwareDistillationLoss(BaseLoss):
    """Distill RGB kinematics from M teachers, weighted by their
    self-assessed reliability.

    For each teacher m:
        w_m = valid_m · exp(−log_var_m)        (higher confidence → more weight)
        L_m = w_m · (α · (1 − cos(s, t_m)) + β · MSE(s, t_m))

    The total loss also includes an uncertainty calibration regulariser
    that encourages teachers not to uniformly claim high confidence:
        L_cal = −mean(log_var)  · λ_cal       (entropy bonus on uncertainty)
    """

    def __init__(self, loss_term_weight=1.0,
                 cosine_weight=0.5, mse_weight=0.5,
                 calibration_weight=0.01):
        super().__init__(loss_term_weight)
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight
        self.calibration_weight = calibration_weight

    @gather_and_scale_wrapper
    def forward(self, student_embeddings, teacher_stack,
                log_var_stack, valid_stack):
        """
        Args:
            student_embeddings: [N, C]
            teacher_stack:      [N, M, C]  (detached)
            log_var_stack:      [N, M]     (detached)
            valid_stack:        [N, M]     (0/1)
        """
        s = student_embeddings.float()
        t = teacher_stack.float()
        log_var = log_var_stack.float()
        valid = valid_stack.float()

        n, m, c = t.shape

        # Check there's at least one valid teacher
        total_valid = valid.sum()
        if total_valid < 1:
            loss = s.new_tensor(0.)
            self.info.update({
                'loss': loss.detach().clone(),
                'valid_ratio': s.new_tensor(0.),
            })
            return loss, self.info

        # Uncertainty weights
        unc_weight = valid * torch.exp(
            -log_var.clamp(-10, 10))                    # [N, M]
        weight_sum = unc_weight.sum(1, keepdim=True).clamp_min(1e-6)
        unc_weight = unc_weight / weight_sum            # normalised

        # Per-teacher losses
        s_exp = s.unsqueeze(1).expand_as(t)             # [N, M, C]
        cos_loss = 1 - F.cosine_similarity(s_exp, t, dim=-1)  # [N, M]
        mse_loss = (s_exp - t).pow(2).mean(-1)          # [N, M]
        per_teacher = (self.cosine_weight * cos_loss
                       + self.mse_weight * mse_loss)    # [N, M]

        # Weighted sum
        dist_loss = (unc_weight * per_teacher).sum(1)   # [N]
        dist_loss = dist_loss.mean()

        # Calibration regulariser: discourage all teachers from claiming
        # log_var = −∞ (i.e. perfect confidence); mild entropy bonus.
        cal_loss = -log_var[valid > 0].mean() if (valid > 0).any() \
            else s.new_tensor(0.)

        loss = dist_loss + self.calibration_weight * cal_loss

        self.info.update({
            'loss': loss.detach().clone(),
            'dist': dist_loss.detach().clone(),
            'cal': cal_loss.detach().clone(),
            'valid_ratio': valid.mean().detach().clone(),
            'mean_unc_weight': unc_weight[valid > 0].mean().detach().clone()
            if (valid > 0).any() else s.new_tensor(0.),
        })
        return loss, self.info


# ===================================================================
# Innovation 4 : Silhouette Losses
# ===================================================================

class SilhouetteForegroundLoss(BaseLoss):
    """Penalise part-attention that falls outside the silhouette mask.

    L_fg = mean( attn_map * (1 - sil_mask) )

    This encourages each learnable part query to focus on the foreground
    person rather than background clutter.
    """

    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, attn_map, sil_mask):
        """
        Args:
            attn_map:  [N, T, P, H, W]  part attention maps.
            sil_mask:  [N, 1, T, H, W]  or None.
        """
        if sil_mask is None:
            loss = attn_map.new_tensor(0.)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        attn = attn_map.float()
        # Reshape sil_mask to [N, T, 1, H, W] for broadcast with attn
        if sil_mask.dim() == 5 and sil_mask.size(1) == 1:
            sil = sil_mask.permute(0, 2, 1, 3, 4).float()  # [N,T,1,H,W]
        else:
            sil = sil_mask.float()

        bg_mask = 1.0 - sil.clamp(0, 1)
        # attn: [N,T,P,H,W],  bg_mask: [N,T,1,H,W]
        penalty = (attn * bg_mask).sum(dim=(-2, -1))       # [N, T, P]
        # Normalise by spatial extent
        total = attn.sum(dim=(-2, -1)).clamp_min(1e-6)
        loss = (penalty / total).mean()

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class BackgroundAdversarialLoss(BaseLoss):
    """Background features should NOT predict identity.

    We compute a softmax classification loss on the background embedding,
    then negate it (or equivalently, use gradient reversal). In practice
    we implement it as: maximise the cross-entropy of background embeddings
    w.r.t. the identity labels ≈ minimise negative cross-entropy ≈
    push the classifier toward uniform predictions.

    Since gradient reversal in OpenGait is awkward, we instead compute
    the *negative* of a soft InfoNCE-like identity loss — this pushes
    bg_embed to be un-informative about identity.
    """

    def __init__(self, loss_term_weight=1.0, temperature=0.2):
        super().__init__(loss_term_weight)
        self.temperature = temperature

    @gather_and_scale_wrapper
    def forward(self, bg_embed, labels):
        z = bg_embed.float()
        if z.abs().sum() < 1e-9:
            loss = z.new_tensor(0.)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        z = _safe_normalize(z)
        sim = z @ z.T / self.temperature        # [B, B]
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        pos = same & ~eye

        if pos.sum() == 0:
            loss = z.new_tensor(0.)
        else:
            sim = sim - sim.max(1, keepdim=True)[0].detach()
            exp_sim = torch.exp(sim) * (~eye).float()
            log_prob = sim - torch.log(
                exp_sim.sum(1, keepdim=True).clamp_min(1e-6))
            # *Negative* of the identity-pulling objective →
            # gradient pushes bg_embed away from identity-discriminative
            id_pull = (log_prob * pos.float()).sum(1) / pos.float().sum(
                1).clamp_min(1.0)
            loss = id_pull[pos.any(1)].mean()  # positive value = adversarial

        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
