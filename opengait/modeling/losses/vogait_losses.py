# -*- coding: utf-8 -*-
"""
Auxiliary losses for VOGait.

Drop this file into:
    OpenGait/opengait/modeling/losses/vogait_losses.py

Then add to OpenGait/opengait/modeling/losses/__init__.py:
    from .vogait_losses import ViewObserverLoss, DynamicalObservationLoss, StrategyDiversityLoss, ViewEquivarianceLoss

The classes intentionally accept both OpenGait-style keyword calls and a single dict argument:
    loss(**training_feat[log_prefix])
    loss(training_feat[log_prefix])
"""

import math
from contextlib import nullcontext
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _first_tensor(*args, **kwargs):
    for x in args:
        if torch.is_tensor(x):
            return x
        if isinstance(x, dict):
            t = _first_tensor(**x)
            if t is not None:
                return t
    for x in kwargs.values():
        if torch.is_tensor(x):
            return x
        if isinstance(x, dict):
            t = _first_tensor(**x)
            if t is not None:
                return t
    return None


def _zero_like(*args, **kwargs):
    """Return a graph-connected fp32 zero when possible.

    In AMP training, returning half zeros from auxiliary branches can still
    participate in mixed dtype arithmetic. Keeping the loss scalar fp32 makes
    GradScaler/backward more stable while preserving gradient connectivity.
    """
    t = _first_tensor(*args, **kwargs)
    if t is None:
        return torch.tensor(0.0)
    if torch.is_tensor(t) and t.is_floating_point():
        return t.float().sum() * 0.0
    return t.to(dtype=torch.float32).sum() * 0.0


def _disable_autocast(ref=None):
    """Disable CUDA autocast only for the small auxiliary-loss computation."""
    if torch.is_tensor(ref) and ref.is_cuda:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _f32(x, detach=False):
    """Move tensors used by losses to fp32 without changing labels."""
    if x is None:
        return None
    if not torch.is_tensor(x):
        return x
    if detach:
        x = x.detach()
    if x.is_floating_point():
        return x.float()
    return x


def _unpack(args, kwargs) -> Dict[str, Any]:
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        return args[0]
    if len(args) > 0 and isinstance(args[0], dict):
        out = dict(args[0])
        out.update(kwargs)
        return out
    return kwargs


def _match_time(x: Optional[torch.Tensor], target_t: int):
    if x is None:
        return None
    if x.size(1) == target_t:
        return x
    if x.size(1) > target_t:
        return x[:, :target_t].contiguous()
    pad = x[:, -1:].expand(x.size(0), target_t - x.size(1), *x.shape[2:]).contiguous()
    return torch.cat([x, pad], dim=1)


class _AuxLoss(nn.Module):
    def __init__(self, loss_term_weight=1.0, log_prefix="aux"):
        super().__init__()
        self.loss_term_weight = float(loss_term_weight)
        self.log_prefix = log_prefix

    def _finish(self, raw_loss: torch.Tensor, info: Dict[str, torch.Tensor]):
        loss = raw_loss * self.loss_term_weight
        out = {"loss": loss.detach()}
        for k, v in info.items():
            if torch.is_tensor(v):
                out[k] = v.detach()
            else:
                out[k] = v
        return loss, out


class ViewObserverLoss(_AuxLoss):
    """
    Supervises the View Observer.

    Strong labels:
        azimuth_labels/elevation_labels, parsed from dataset view ids when available.
    Weak labels:
        visibility/occlusion/part_confidence/foreshortening/phase pseudo-targets from GPF.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        log_prefix="view_observer",
        azimuth_weight=1.0,
        elevation_weight=0.2,
        visibility_weight=0.5,
        occlusion_weight=0.3,
        part_confidence_weight=0.3,
        foreshortening_weight=0.2,
        phase_weight=0.3,
    ):
        super().__init__(loss_term_weight, log_prefix)
        self.azimuth_weight = float(azimuth_weight)
        self.elevation_weight = float(elevation_weight)
        self.visibility_weight = float(visibility_weight)
        self.occlusion_weight = float(occlusion_weight)
        self.part_confidence_weight = float(part_confidence_weight)
        self.foreshortening_weight = float(foreshortening_weight)
        self.phase_weight = float(phase_weight)

    def _ce_seq(self, logits, labels):
        if logits is None or labels is None:
            device = None
            if torch.is_tensor(logits):
                device = logits.device
            elif torch.is_tensor(labels):
                device = labels.device
            return _zero_like(logits, labels), torch.tensor(-1.0, device=device)

        if logits.dim() == 3:
            logits_seq = logits.mean(dim=1)
        else:
            logits_seq = logits

        logits_seq = _f32(logits_seq)
        labels = labels.to(logits_seq.device).long().reshape(-1)

        n = min(logits_seq.size(0), labels.numel())
        logits_seq = logits_seq[:n]
        labels = labels[:n]

        valid = labels >= 0
        if valid.sum() == 0:
            return logits_seq.sum() * 0.0, torch.tensor(-1.0, device=logits_seq.device)

        loss = F.cross_entropy(logits_seq[valid], labels[valid])
        acc = (logits_seq[valid].argmax(dim=-1) == labels[valid]).float().mean()
        return loss, acc

    def _bce(self, logits, target):
        if logits is None or target is None:
            return _zero_like(logits, target)

        logits = _f32(logits)
        target = target.to(device=logits.device, dtype=logits.dtype)
        target = _match_time(target, logits.size(1)) if logits.dim() >= 3 else target

        n = min(logits.size(0), target.size(0))
        return F.binary_cross_entropy_with_logits(
            logits[:n],
            target[:n].clamp(0, 1),
        )

    def _smooth_l1(self, pred, target):
        if pred is None or target is None:
            return _zero_like(pred, target)

        pred = _f32(pred)
        target = target.to(device=pred.device, dtype=pred.dtype)
        target = _match_time(target, pred.size(1)) if pred.dim() >= 3 else target

        n = min(pred.size(0), target.size(0))
        return F.smooth_l1_loss(pred[:n], target[:n])

    def _phase(self, pred, target):
        if pred is None or target is None:
            return _zero_like(pred, target)

        pred = _f32(pred)
        target = target.to(device=pred.device, dtype=pred.dtype)
        target = _match_time(target, pred.size(1))

        n = min(pred.size(0), target.size(0))
        pred = F.normalize(pred[:n], dim=-1, eps=1e-6)
        target = F.normalize(target[:n], dim=-1, eps=1e-6)

        return (1.0 - (pred * target).sum(dim=-1)).mean()

    def forward(self, *args, **kwargs):
        x = _unpack(args, kwargs)

        with _disable_autocast(_first_tensor(x)):
            az_loss, az_acc = self._ce_seq(x.get("azimuth_logits"), x.get("azimuth_labels"))
            el_loss, el_acc = self._ce_seq(x.get("elevation_logits"), x.get("elevation_labels"))
            vis_loss = self._bce(x.get("visibility_logits"), x.get("visibility_targets"))
            occ_loss = self._bce(x.get("occlusion_logits"), x.get("occlusion_targets"))
            conf_loss = self._bce(x.get("part_confidence_logits"), x.get("part_confidence_targets"))
            fore_loss = self._smooth_l1(x.get("foreshortening"), x.get("foreshortening_targets"))
            phase_loss = self._phase(x.get("phase_pred"), x.get("phase_targets"))

            raw = (
                self.azimuth_weight * az_loss
                + self.elevation_weight * el_loss
                + self.visibility_weight * vis_loss
                + self.occlusion_weight * occ_loss
                + self.part_confidence_weight * conf_loss
                + self.foreshortening_weight * fore_loss
                + self.phase_weight * phase_loss
            )

        return self._finish(raw, {
            "azimuth_loss": az_loss,
            "azimuth_acc": az_acc,
            "elevation_loss": el_loss,
            "elevation_acc": el_acc,
            "visibility_loss": vis_loss,
            "occlusion_loss": occ_loss,
            "part_confidence_loss": conf_loss,
            "foreshortening_loss": fore_loss,
            "phase_loss": phase_loss,
        })


class DynamicalObservationLoss(_AuxLoss):
    """Next-step latent prediction plus view-conditioned observation consistency."""
    def __init__(
        self,
        loss_term_weight=1.0,
        log_prefix="dynamics",
        dynamics_weight=1.0,
        observation_weight=0.5,
        cosine_weight=0.2,
    ):
        super().__init__(loss_term_weight, log_prefix)
        self.dynamics_weight = float(dynamics_weight)
        self.observation_weight = float(observation_weight)
        self.cosine_weight = float(cosine_weight)

    def forward(self, *args, **kwargs):
        x = _unpack(args, kwargs)

        with _disable_autocast(_first_tensor(x)):
            zp = x.get("z_next_pred")
            zt = x.get("z_next_target")
            zop = x.get("z_obs_next_pred")
            zot = x.get("z_obs_next_target")

            if zp is None or zt is None or zp.numel() == 0:
                raw = _zero_like(zp, zt, zop, zot)
                return self._finish(raw, {
                    "dynamics_mse": raw,
                    "observation_mse": raw,
                    "dynamics_cos": raw,
                })

            n = min(zp.size(0), zt.size(0))
            t = min(zp.size(1), zt.size(1))

            zp = _f32(zp[:n, :t])
            zt = zt[:n, :t].detach().to(device=zp.device, dtype=zp.dtype)

            dyn_mse = F.mse_loss(zp, zt)
            dyn_cos = (1.0 - F.cosine_similarity(zp, zt, dim=-1)).mean()

            if zop is not None and zot is not None and zop.numel() > 0:
                n2 = min(zop.size(0), zot.size(0))
                t2 = min(zop.size(1), zot.size(1))

                zop = _f32(zop[:n2, :t2])
                zot = zot[:n2, :t2].detach().to(device=zop.device, dtype=zop.dtype)

                obs_mse = F.mse_loss(zop, zot)
            else:
                obs_mse = dyn_mse * 0.0

            raw = (
                self.dynamics_weight * dyn_mse
                + self.observation_weight * obs_mse
                + self.cosine_weight * dyn_cos
            )

        return self._finish(raw, {
            "dynamics_mse": dyn_mse,
            "observation_mse": obs_mse,
            "dynamics_cos": dyn_cos,
        })


class StrategyDiversityLoss(_AuxLoss):
    """
    Prevents the strategy bank from collapsing into a single expert while still
    allowing sample-wise soft/hard routing.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        log_prefix="strategy",
        load_balance_weight=1.0,
        sparsity_weight=0.1,
        part_orthogonality_weight=0.2,
        eps=1e-6,
    ):
        super().__init__(loss_term_weight, log_prefix)
        self.load_balance_weight = float(load_balance_weight)
        self.sparsity_weight = float(sparsity_weight)
        self.part_orthogonality_weight = float(part_orthogonality_weight)
        self.eps = eps

    def forward(self, *args, **kwargs):
        x = _unpack(args, kwargs)

        with _disable_autocast(_first_tensor(x)):
            alpha = x.get("alpha")
            usage = x.get("expert_part_usage")

            if alpha is None:
                raw = _zero_like(usage)
                return self._finish(raw, {
                    "load_balance": raw,
                    "routing_entropy": raw,
                    "part_orthogonality": raw,
                })

            alpha = _f32(alpha)
            e = alpha.size(-1)

            mean_alpha = alpha.mean(dim=(0, 1))
            uniform = torch.full_like(mean_alpha, 1.0 / float(e))

            load_balance = F.mse_loss(mean_alpha, uniform) * e

            alpha_safe = alpha.clamp_min(self.eps)
            entropy = -(alpha_safe * alpha_safe.log()).sum(dim=-1).mean() / math.log(max(e, 2))

            if usage is not None and torch.is_tensor(usage):
                usage = _f32(usage)
                u = F.normalize(usage, dim=-1, eps=1e-6)
                gram = torch.matmul(u, u.t())
                eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
                part_orth = ((gram - eye) ** 2).mean()
            else:
                part_orth = load_balance * 0.0

            raw = (
                self.load_balance_weight * load_balance
                + self.sparsity_weight * entropy
                + self.part_orthogonality_weight * part_orth
            )

        return self._finish(raw, {
            "load_balance": load_balance,
            "routing_entropy": entropy,
            "part_orthogonality": part_orth,
        })


class ViewEquivarianceLoss(_AuxLoss):
    """
    A lightweight training-time version of view equivariance.

    When a mini-batch contains same-ID sequences from different views, the canonical
    dynamics should be close, while the view-projected dynamics are allowed to differ.
    This returns zero if such pairs are absent, so it is safe with ordinary samplers.
    """
    def __init__(
        self,
        loss_term_weight=1.0,
        log_prefix="view_equiv",
        canonical_weight=1.0,
        projected_margin_weight=0.05,
        projected_margin=0.05,
    ):
        super().__init__(loss_term_weight, log_prefix)
        self.canonical_weight = float(canonical_weight)
        self.projected_margin_weight = float(projected_margin_weight)
        self.projected_margin = float(projected_margin)

    def forward(self, *args, **kwargs):
        x = _unpack(args, kwargs)

        with _disable_autocast(_first_tensor(x)):
            canon = x.get("canonical_embedding")
            proj = x.get("projected_embedding")
            labels = x.get("labels")
            views = x.get("azimuth_labels")

            if canon is None or labels is None or views is None:
                raw = _zero_like(canon, labels, views)
                return self._finish(raw, {
                    "canonical_pair_loss": raw,
                    "projected_margin_loss": raw,
                    "valid_pairs": 0.0,
                })

            canon = _f32(canon)
            labels = labels.to(canon.device).reshape(-1)
            views = views.to(canon.device).reshape(-1)

            n = min(canon.size(0), labels.numel(), views.numel())

            canon = F.normalize(canon[:n], dim=-1, eps=1e-6)
            labels = labels[:n]
            views = views[:n]

            valid_view = views >= 0
            same_id = labels[:, None].eq(labels[None, :])
            diff_view = views[:, None].ne(views[None, :]) & valid_view[:, None] & valid_view[None, :]
            upper = torch.triu(
                torch.ones(n, n, device=canon.device, dtype=torch.bool),
                diagonal=1,
            )
            mask = same_id & diff_view & upper

            if mask.sum() == 0:
                raw = canon.sum() * 0.0
                return self._finish(raw, {
                    "canonical_pair_loss": raw,
                    "projected_margin_loss": raw,
                    "valid_pairs": torch.tensor(0.0, device=canon.device),
                })

            sim = torch.matmul(canon, canon.t())
            canonical_pair = (1.0 - sim[mask]).mean()

            if proj is not None:
                proj = _f32(proj[:n])
                proj = F.normalize(proj, dim=-1, eps=1e-6)
                psim = torch.matmul(proj, proj.t())
                # Do not force view-projected evidence to be identical; only avoid excessive collapse.
                projected_margin = F.relu(
                    psim[mask] - (1.0 - self.projected_margin)
                ).mean()
            else:
                projected_margin = canonical_pair * 0.0

            raw = (
                self.canonical_weight * canonical_pair
                + self.projected_margin_weight * projected_margin
            )

        return self._finish(raw, {
            "canonical_pair_loss": canonical_pair,
            "projected_margin_loss": projected_margin,
            "valid_pairs": mask.sum().float(),
        })
