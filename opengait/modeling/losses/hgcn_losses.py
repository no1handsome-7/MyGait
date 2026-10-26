
# -*- coding: utf-8 -*-
"""
Custom losses for Human-like Gait Cognition Network.

Drop-in path:
    OpenGait/opengait/modeling/losses/hgcn_losses.py

Register in OpenGait/opengait/modeling/losses/__init__.py:
    from .hgcn_losses import (
        CrossConditionSupConLoss,
        PrototypeClusterLoss,
        FactorDisentangleLoss,
        ViewObservableLoss,
        PartActionLoss,
    )

These classes follow the common OpenGait convention:
    forward(...) -> (loss, info_dict)

They are written defensively: if a required label is missing or degenerate,
the loss returns a zero tensor instead of crashing.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_like_loss(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _info(name: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {name: value.detach()}


class CrossConditionSupConLoss(nn.Module):
    """
    Supervised contrastive loss with higher positive weight for cross-condition
    and optionally cross-view positives.

    This directly optimizes "same ID should form a compact retrieval cluster",
    not just top-1 closed-set classification.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        cross_condition_weight: float = 2.0,
        cross_view_weight: float = 1.3,
        loss_term_weight: float = 0.2,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.cross_condition_weight = float(cross_condition_weight)
        self.cross_view_weight = float(cross_view_weight)
        self.loss_term_weight = float(loss_term_weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        condition_labels: Optional[torch.Tensor] = None,
        view_labels: Optional[torch.Tensor] = None,
    ):
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=2)
        z = F.normalize(embeddings.float(), dim=1)
        labels = labels.view(-1).long()
        n = z.size(0)
        if n <= 1 or labels.numel() != n:
            loss = _zero_like_loss(z)
            return loss, _info("cross_condition_supcon_loss", loss)

        sim = torch.div(torch.matmul(z, z.t()), self.temperature)
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(sim) - torch.eye(n, device=z.device, dtype=z.dtype)
        pos_mask = (labels[:, None] == labels[None, :]).float() * logits_mask

        if pos_mask.sum() <= 0:
            loss = _zero_like_loss(z)
            return loss, _info("cross_condition_supcon_loss", loss)

        weights = torch.ones_like(sim)
        if condition_labels is not None and condition_labels.numel() == n:
            cond = condition_labels.view(-1).long()
            cross_cond = (cond[:, None] != cond[None, :]).float()
            weights = weights + (self.cross_condition_weight - 1.0) * cross_cond * pos_mask
        if view_labels is not None and view_labels.numel() == n:
            view = view_labels.view(-1).long()
            cross_view = (view[:, None] != view[None, :]).float()
            weights = weights + (self.cross_view_weight - 1.0) * cross_view * pos_mask

        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        weighted_pos = pos_mask * weights
        mean_log_prob_pos = (weighted_pos * log_prob).sum(dim=1) / weighted_pos.sum(dim=1).clamp_min(1e-12)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean() * self.loss_term_weight
        return loss, {"cross_condition_supcon_loss": loss.detach()}


class PrototypeClusterLoss(nn.Module):
    """
    Cluster/prototype objective:
      - CE to global identity prototypes
      - pull z_gait to condition prototype
      - align condition prototype with global ID prototype

    The trainable prototypes live in the model; this loss consumes the logits
    and selected cosine similarities emitted by ClusterPrototypeHead.
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        condition_pull_weight: float = 0.2,
        global_pull_weight: float = 0.1,
        condition_global_weight: float = 0.1,
        loss_term_weight: float = 0.2,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.condition_pull_weight = float(condition_pull_weight)
        self.global_pull_weight = float(global_pull_weight)
        self.condition_global_weight = float(condition_global_weight)
        self.loss_term_weight = float(loss_term_weight)

    def forward(
        self,
        prototype_logits: torch.Tensor,
        labels: torch.Tensor,
        proto_sim_z_global: Optional[torch.Tensor] = None,
        proto_sim_z_condition: Optional[torch.Tensor] = None,
        proto_sim_condition_global: Optional[torch.Tensor] = None,
    ):
        labels = labels.view(-1).long()
        loss = F.cross_entropy(prototype_logits.float(), labels) * self.ce_weight
        if proto_sim_z_condition is not None:
            loss = loss + (1.0 - proto_sim_z_condition.float()).mean() * self.condition_pull_weight
        if proto_sim_z_global is not None:
            loss = loss + (1.0 - proto_sim_z_global.float()).mean() * self.global_pull_weight
        if proto_sim_condition_global is not None:
            loss = loss + (1.0 - proto_sim_condition_global.float()).mean() * self.condition_global_weight

        loss = loss * self.loss_term_weight
        return loss, {"prototype_cluster_loss": loss.detach()}


class FactorDisentangleLoss(nn.Module):
    """
    Factor disentanglement:
      z_gait: identity, but adversarially condition/view-invariant.
      z_nui : predicts nuisance condition, adversarially identity-weak.
      z_view: predicts view.
      Orthogonality separates z_gait from z_nui/z_view/z_act.

    GRL is applied in the model, so the CE terms here are normal CE.
    """
    def __init__(
        self,
        condition_weight: float = 0.2,
        adv_condition_weight: float = 0.2,
        view_weight: float = 0.1,
        adv_view_weight: float = 0.1,
        nui_id_adv_weight: float = 0.05,
        orth_weight: float = 0.05,
        act_orth_weight: float = 0.02,
        loss_term_weight: float = 1.0,
    ):
        super().__init__()
        self.condition_weight = float(condition_weight)
        self.adv_condition_weight = float(adv_condition_weight)
        self.view_weight = float(view_weight)
        self.adv_view_weight = float(adv_view_weight)
        self.nui_id_adv_weight = float(nui_id_adv_weight)
        self.orth_weight = float(orth_weight)
        self.act_orth_weight = float(act_orth_weight)
        self.loss_term_weight = float(loss_term_weight)

    def _orth(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a.float(), dim=1)
        b = F.normalize(b.float(), dim=1)
        return (a * b).sum(dim=1).pow(2).mean()

    def forward(
        self,
        z_gait: torch.Tensor,
        z_nui: torch.Tensor,
        z_view: torch.Tensor,
        z_act: Optional[torch.Tensor] = None,
        condition_logits: Optional[torch.Tensor] = None,
        adv_condition_logits: Optional[torch.Tensor] = None,
        view_logits: Optional[torch.Tensor] = None,
        adv_view_logits: Optional[torch.Tensor] = None,
        nui_id_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        condition_labels: Optional[torch.Tensor] = None,
        view_labels: Optional[torch.Tensor] = None,
    ):
        loss = _zero_like_loss(z_gait)

        if condition_logits is not None and condition_labels is not None and condition_labels.numel() == condition_logits.size(0):
            loss = loss + F.cross_entropy(condition_logits.float(), condition_labels.long()) * self.condition_weight

        if adv_condition_logits is not None and condition_labels is not None and condition_labels.numel() == adv_condition_logits.size(0):
            loss = loss + F.cross_entropy(adv_condition_logits.float(), condition_labels.long()) * self.adv_condition_weight

        if view_logits is not None and view_labels is not None and view_labels.numel() == view_logits.size(0):
            loss = loss + F.cross_entropy(view_logits.float(), view_labels.long()) * self.view_weight

        if adv_view_logits is not None and view_labels is not None and view_labels.numel() == adv_view_logits.size(0):
            loss = loss + F.cross_entropy(adv_view_logits.float(), view_labels.long()) * self.adv_view_weight

        if nui_id_logits is not None and labels is not None and labels.numel() == nui_id_logits.size(0):
            loss = loss + F.cross_entropy(nui_id_logits.float(), labels.long()) * self.nui_id_adv_weight

        loss = loss + self._orth(z_gait, z_nui) * self.orth_weight
        loss = loss + self._orth(z_gait, z_view) * self.orth_weight
        if z_act is not None:
            loss = loss + self._orth(z_nui, z_act) * self.act_orth_weight

        loss = loss * self.loss_term_weight
        return loss, {"factor_disentangle_loss": loss.detach()}


class ViewObservableLoss(nn.Module):
    """
    Self-supervised reliability alignment:
      - motion group gate should agree with motion reliability / periodic evidence
      - affinity/residual group gate should agree with texture-risk-inverse
    """
    def __init__(self, loss_term_weight: float = 0.05):
        super().__init__()
        self.loss_term_weight = float(loss_term_weight)

    def forward(
        self,
        primitive_group_gate: torch.Tensor,
        reliability_mean: torch.Tensor,
        periodic_mean: torch.Tensor,
        texture_inverse_mean: torch.Tensor,
    ):
        if primitive_group_gate is None or primitive_group_gate.size(-1) < 4:
            loss = _zero_like_loss(reliability_mean)
            return loss, {"view_observable_loss": loss.detach()}

        gate = primitive_group_gate.float().clamp(0, 1)  # [B,4] geo/motion/freq/aff
        rel = reliability_mean.float().view(-1).clamp(0, 1)
        per = periodic_mean.float().view(-1).clamp(0, 1)
        tex = texture_inverse_mean.float().view(-1).clamp(0, 1)

        target_motion = (0.5 * rel + 0.5 * per).detach()
        target_freq = per.detach()
        target_aff = (0.6 * tex + 0.4 * rel).detach()

        loss = F.mse_loss(gate[:, 1], target_motion)
        loss = loss + F.mse_loss(gate[:, 2], target_freq)
        loss = loss + F.mse_loss(gate[:, 3], target_aff)
        loss = loss * self.loss_term_weight
        return loss, {"view_observable_loss": loss.detach()}


class PartActionLoss(nn.Module):
    """
    Prevent part/action token collapse through diversity losses.
    Optionally extend this with VLM/MLLM vocabulary distillation later.
    """
    def __init__(self, part_div_weight: float = 0.03, action_div_weight: float = 0.03, loss_term_weight: float = 1.0):
        super().__init__()
        self.part_div_weight = float(part_div_weight)
        self.action_div_weight = float(action_div_weight)
        self.loss_term_weight = float(loss_term_weight)

    def _diversity(self, toks: torch.Tensor) -> torch.Tensor:
        # toks: [B,C,K]
        if toks is None or toks.dim() != 3 or toks.size(-1) <= 1:
            return _zero_like_loss(toks if toks is not None else torch.zeros((), requires_grad=True))
        x = F.normalize(toks.float(), dim=1).transpose(1, 2)  # [B,K,C]
        sim = torch.matmul(x, x.transpose(1, 2))  # [B,K,K]
        k = sim.size(-1)
        off = sim - torch.eye(k, device=sim.device, dtype=sim.dtype)[None]
        return off.pow(2).mean()

    def forward(self, part_tokens: torch.Tensor, action_tokens: torch.Tensor):
        loss = self._diversity(part_tokens) * self.part_div_weight
        loss = loss + self._diversity(action_tokens) * self.action_div_weight
        loss = loss * self.loss_term_weight
        return loss, {"part_action_loss": loss.detach()}
