"""
Loss helpers for CQGaitMoE.

Drop this file into:
    OpenGait/opengait/modeling/losses/cqgait_moe_losses.py

The default CQGaitMoE config uses only ScalarLoss because the model already sums
MoE/CQE/text auxiliary losses into training_feat['cqgait_aux']['loss'].
PartMoEAuxLoss is provided for future ablations where router losses are emitted
as separate training_feat entries.
"""

import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class ScalarLoss(BaseLoss):
    """Pass-through scalar loss for auxiliary losses emitted by a model.

    Expected training_feat item:
        training_feat['cqgait_aux'] = {'loss': scalar_tensor}
    """

    def __init__(self, loss_term_weight=1.0):
        super(ScalarLoss, self).__init__(loss_term_weight)

    def forward(self, loss, labels=None):
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        loss = loss.float()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class PartMoEAuxLoss(BaseLoss):
    """Standalone top-k part-MoE auxiliary loss.

    This is optional. CQGaitMoE computes this internally by default.

    Args:
        router_prob: [B,P,E] softmax probability over experts.
        topk_idx:    [B,P,K] hard selected expert indices.
        router_logits: optional [B,P,E], for z-loss.
    """

    def __init__(self, importance_weight=1.0, load_weight=1.0, z_loss_weight=0.0, eps=1e-10, loss_term_weight=1.0):
        super(PartMoEAuxLoss, self).__init__(loss_term_weight)
        self.importance_weight = float(importance_weight)
        self.load_weight = float(load_weight)
        self.z_loss_weight = float(z_loss_weight)
        self.eps = float(eps)

    @staticmethod
    def _cv_squared(x, eps=1e-10):
        x = x.float()
        mean = x.mean().clamp_min(eps)
        return x.var(unbiased=False) / (mean * mean + eps)

    @gather_and_scale_wrapper
    def forward(self, router_prob, topk_idx=None, router_logits=None):
        if router_prob.dim() != 3:
            raise ValueError(f'router_prob must be [B,P,E], got {tuple(router_prob.shape)}')
        B, P, E = router_prob.shape
        loss = router_prob.new_tensor(0.0)

        importance = router_prob.float().mean(dim=(0, 1))
        if self.importance_weight > 0:
            loss = loss + self._cv_squared(importance, self.eps) * self.importance_weight

        load = None
        if topk_idx is not None and self.load_weight > 0:
            if topk_idx.dim() != 3 or topk_idx.shape[:2] != (B, P):
                raise ValueError(f'topk_idx must be [B,P,K], got {tuple(topk_idx.shape)}')
            load = F.one_hot(topk_idx.long(), num_classes=E).float().sum(dim=2).clamp(0, 1).mean(dim=(0, 1))
            loss = loss + self._cv_squared(load, self.eps) * self.load_weight

        if router_logits is not None and self.z_loss_weight > 0:
            loss = loss + torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean() * self.z_loss_weight

        entropy = -(importance * (importance + self.eps).log()).sum()
        self.info.update({
            'loss': loss.detach().clone(),
            'importance_entropy': entropy.detach().clone(),
        })
        for i in range(E):
            self.info[f'importance_{i}'] = importance[i].detach().clone()
            if load is not None:
                self.info[f'load_{i}'] = load[i].detach().clone()
        return loss, self.info
