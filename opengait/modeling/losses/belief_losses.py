# -*- coding: utf-8 -*-
"""Losses for BeliefGait. Drop into OpenGait/opengait/modeling/losses/."""

import math
import torch
import torch.nn.functional as F

from .base import BaseLoss


def _masked_mean(x, mask):
    mask = mask.to(x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return (x * mask).sum() / mask.expand_as(x).sum().clamp_min(1e-6)


def _entropy(p, eps=1e-8):
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=-1) / math.log(max(2, p.size(-1)))


def _sym_kl(p, q, eps=1e-8):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return 0.5 * ((p * (p.log() - q.log())).sum(dim=-1) + (q * (q.log() - p.log())).sum(dim=-1))


def _circular_convolution(p, q):
    out = torch.zeros_like(p)
    for shift in range(p.size(-1)):
        out = out + q[..., shift:shift + 1] * torch.roll(p, shifts=shift, dims=-1)
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _mirror(p):
    k = p.size(-1)
    index = torch.tensor([(-i) % k for i in range(k)], device=p.device, dtype=torch.long)
    return p.index_select(-1, index)


class BeliefCycleConsistencyLoss(BaseLoss):
    """Relative-view composition: delta(t,t+2) should match delta(t,t+1) o delta(t+1,t+2)."""
    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, transition_posterior, direct2_transition_posterior, temporal_mask):
        if transition_posterior.size(1) < 3:
            zero = transition_posterior.sum() * 0.0
            self.info.update({"loss": zero.detach()})
            return zero, self.info
        composed = _circular_convolution(transition_posterior[:, 1:-1].float(), transition_posterior[:, 2:].float())
        direct = direct2_transition_posterior.float()
        valid = temporal_mask[:, :-2] & temporal_mask[:, 1:-1] & temporal_mask[:, 2:]
        loss = _masked_mean(_sym_kl(composed, direct), valid)
        self.info.update({"loss": loss.detach(), "valid_ratio": valid.float().mean().detach()})
        return loss, self.info


class BeliefFlipEquivarianceLoss(BaseLoss):
    """Horizontal flip should mirror relative direction shifts."""
    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, transition_posterior, flip_transition_posterior, temporal_mask):
        if transition_posterior.size(1) <= 1:
            zero = transition_posterior.sum() * 0.0
            self.info.update({"loss": zero.detach()})
            return zero, self.info
        target = _mirror(transition_posterior[:, 1:].float()).detach()
        pred = flip_transition_posterior[:, 1:].float()
        valid = temporal_mask[:, 1:]
        loss = _masked_mean(_sym_kl(pred, target), valid)
        self.info.update({"loss": loss.detach(), "valid_ratio": valid.float().mean().detach()})
        return loss, self.info


class ActionGateSelfSupervisionLoss(BaseLoss):
    """Weak heuristic bootstrapping for walk/turn/interaction/uncertain gates."""
    def __init__(self, balance_weight=0.05, entropy_weight=0.01, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.balance_weight = float(balance_weight)
        self.entropy_weight = float(entropy_weight)

    def forward(self, action_probs, action_pseudo, temporal_mask):
        p = action_probs.float().clamp_min(1e-8)
        q = action_pseudo.float().detach().clamp_min(1e-8)
        kl = _masked_mean((q * (q.log() - p.log())).sum(dim=-1), temporal_mask)
        valid = temporal_mask.float().unsqueeze(-1)
        mean_action = (p * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        uniform = torch.full_like(mean_action, 1.0 / mean_action.numel())
        balance = (mean_action.clamp_min(1e-8) * (mean_action.clamp_min(1e-8).log() - uniform.log())).sum()
        entropy = _masked_mean(_entropy(p), temporal_mask)
        loss = kl + self.balance_weight * balance - self.entropy_weight * entropy
        self.info.update({"loss": loss.detach(), "pseudo_kl": kl.detach(), "balance": balance.detach(), "entropy": entropy.detach()})
        return loss, self.info


class BeliefRegularizationLoss(BaseLoss):
    """Prevent uniform collapse, over-sharp collapse and a single global-bin shortcut."""
    def __init__(self, entropy_min=0.10, entropy_max=0.82, observation_entropy_weight=0.20, belief_entropy_weight=0.20, batch_balance_weight=0.05, stable_smooth_weight=0.15, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.entropy_min = float(entropy_min)
        self.entropy_max = float(entropy_max)
        self.observation_entropy_weight = float(observation_entropy_weight)
        self.belief_entropy_weight = float(belief_entropy_weight)
        self.batch_balance_weight = float(batch_balance_weight)
        self.stable_smooth_weight = float(stable_smooth_weight)

    def _entropy_band(self, entropy, mask):
        penalty = F.relu(self.entropy_min - entropy) + F.relu(entropy - self.entropy_max)
        return _masked_mean(penalty, mask)

    def forward(self, observation_posterior, transition_posterior, belief, action_probs, temporal_mask):
        obs_entropy = _entropy(observation_posterior.float())
        belief_entropy = _entropy(belief.float().clamp_min(1e-8))
        obs_band = self._entropy_band(obs_entropy, temporal_mask)
        belief_band = self._entropy_band(belief_entropy, temporal_mask)
        valid = temporal_mask.float().unsqueeze(-1)
        mean_belief = (belief.float() * valid).sum(dim=(0, 1)) / valid.sum().clamp_min(1.0)
        mean_belief = mean_belief.clamp_min(1e-8)
        uniform = torch.full_like(mean_belief, 1.0 / mean_belief.numel())
        balance = (mean_belief * (mean_belief.log() - uniform.log())).sum()
        if belief.size(1) > 1:
            smooth_gap = 1.0 - F.cosine_similarity(belief[:, 1:].float(), belief[:, :-1].float(), dim=-1)
            stable = action_probs[:, 1:, 0].detach() * temporal_mask[:, 1:].float()
            smooth = (smooth_gap * stable).sum() / stable.sum().clamp_min(1e-6)
        else:
            smooth = belief.sum() * 0.0
        loss = self.observation_entropy_weight * obs_band + self.belief_entropy_weight * belief_band + self.batch_balance_weight * balance + self.stable_smooth_weight * smooth
        self.info.update({"loss": loss.detach(), "obs_entropy_band": obs_band.detach(), "belief_entropy_band": belief_band.detach(), "batch_balance": balance.detach(), "stable_smooth": smooth.detach(), "belief_entropy": _masked_mean(belief_entropy, temporal_mask).detach()})
        return loss, self.info


class MemoryCoverageLoss(BaseLoss):
    """Favor direction coverage while discouraging unbounded writes and dead slots."""
    def __init__(self, redundancy_weight=1.0, write_weight=0.10, usage_weight=0.10, target_write_rate=0.35, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.redundancy_weight = float(redundancy_weight)
        self.write_weight = float(write_weight)
        self.usage_weight = float(usage_weight)
        self.target_write_rate = float(target_write_rate)

    def forward(self, slot_direction, slot_quality, slot_usage, slot_valid, write_gate, temporal_mask):
        direction = F.normalize(slot_direction.float(), dim=-1, eps=1e-6)
        sim = torch.einsum("bsd,btd->bst", direction, direction)
        s = sim.size(1)
        eye = torch.eye(s, dtype=torch.bool, device=sim.device).unsqueeze(0)
        valid_pair = slot_valid.unsqueeze(2) & slot_valid.unsqueeze(1) & ~eye
        quality_pair = slot_quality.unsqueeze(2) * slot_quality.unsqueeze(1)
        redundancy = ((sim.clamp_min(0.0).pow(2) * quality_pair) * valid_pair.float()).sum() / valid_pair.float().sum().clamp_min(1.0)
        valid_time = temporal_mask.float()
        write_rate = (write_gate.float() * valid_time).sum() / valid_time.sum().clamp_min(1.0)
        write_penalty = (write_rate - self.target_write_rate).abs()
        if bool(slot_valid.any()):
            usage_penalty = F.relu(0.05 - slot_usage.float()[slot_valid]).mean()
        else:
            usage_penalty = slot_usage.sum() * 0.0
        loss = self.redundancy_weight * redundancy + self.write_weight * write_penalty + self.usage_weight * usage_penalty
        self.info.update({"loss": loss.detach(), "redundancy": redundancy.detach(), "write_rate": write_rate.detach(), "usage_penalty": usage_penalty.detach(), "valid_slots": slot_valid.float().sum(dim=-1).mean().detach()})
        return loss, self.info


class PrimitiveTransportRegularizationLoss(BaseLoss):
    """Keep latent primitives diverse and transport conservative."""
    def __init__(self, diversity_weight=1.0, covariance_weight=0.25, offset_weight=0.20, affine_weight=0.20, gate_weight=0.02, covariance_min=0.008, covariance_max=0.60, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.diversity_weight = float(diversity_weight)
        self.covariance_weight = float(covariance_weight)
        self.offset_weight = float(offset_weight)
        self.affine_weight = float(affine_weight)
        self.gate_weight = float(gate_weight)
        self.covariance_min = float(covariance_min)
        self.covariance_max = float(covariance_max)

    def forward(self, primitive_assign, primitive_cov, transport_offsets, transport_affine_delta, transport_gate, temporal_mask):
        b, t, m, h, w = primitive_assign.shape
        assign = primitive_assign.float().reshape(b * t, m, h * w)
        assign = F.normalize(assign, dim=-1, eps=1e-6)
        gram = assign @ assign.transpose(1, 2)
        eye = torch.eye(m, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        diversity_each = ((gram - eye).pow(2)).sum(dim=(1, 2)) / max(1, m * (m - 1))
        diversity = _masked_mean(diversity_each.reshape(b, t), temporal_mask)
        eig = torch.linalg.eigvalsh(primitive_cov.float())
        covariance = (F.relu(self.covariance_min - eig) + F.relu(eig - self.covariance_max)).mean()
        offset = transport_offsets.float().pow(2).mean()
        affine = transport_affine_delta.float().pow(2).mean()
        gate = transport_gate.float().mean()
        loss = self.diversity_weight * diversity + self.covariance_weight * covariance + self.offset_weight * offset + self.affine_weight * affine + self.gate_weight * gate
        self.info.update({"loss": loss.detach(), "diversity": diversity.detach(), "covariance": covariance.detach(), "offset": offset.detach(), "affine": affine.detach(), "transport_gate": gate.detach()})
        return loss, self.info


class CanonicalConsistencyLoss(BaseLoss):
    """Weak stop-gradient identity consistency: residual canonicalization must not erase native cues."""
    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, native, canonical, confidence=None):
        native = native.float().detach()
        canonical = canonical.float()
        gap = 1.0 - F.cosine_similarity(canonical, native, dim=-1)
        if confidence is not None:
            weight = (0.25 + 0.75 * confidence.float()).detach()
            while weight.dim() < gap.dim():
                weight = weight.unsqueeze(-1)
            loss = (gap * weight).sum() / weight.expand_as(gap).sum().clamp_min(1e-6)
        else:
            loss = gap.mean()
        self.info.update({"loss": loss.detach(), "cosine_gap": gap.mean().detach()})
        return loss, self.info


class MaskedCrossEntropyLoss(BaseLoss):
    """Cross entropy with invalid camera tags ignored."""
    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, logits, labels, valid_mask):
        valid_mask = valid_mask.bool()
        if not bool(valid_mask.any()):
            zero = logits.sum() * 0.0
            self.info.update({"loss": zero.detach(), "valid_ratio": zero.detach()})
            return zero, self.info
        loss = F.cross_entropy(logits[valid_mask].float(), labels[valid_mask].long())
        acc = (logits[valid_mask].argmax(dim=-1) == labels[valid_mask]).float().mean()
        self.info.update({"loss": loss.detach(), "accuracy": acc.detach(), "valid_ratio": valid_mask.float().mean().detach()})
        return loss, self.info
