"""MoE auxiliary losses for OpenGait loss_cfg integration.

These wrap the in-model computed MoE losses so that the LossAggregator
can route them via ``log_prefix``.
"""

import torch
import torch.nn as nn
from .base import BaseLoss


class ScalarLoss(BaseLoss):
    """Pass-through scalar loss.

    Expected training_feat item::

        {'loss': tensor_scalar}

    Used by GaitMoEOracle for ``gaitoracle_aux`` (aggregated auxiliary loss).
    """

    def __init__(self, loss_term_weight=1.0):
        super(ScalarLoss, self).__init__(loss_term_weight)

    def forward(self, loss, labels=None):
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        loss = loss.float()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class DirectLoss(BaseLoss):
    """Direct passthrough loss for pre-computed scalars.

    Same semantics as ScalarLoss but with optional reduce for non-scalar inputs.
    """

    def __init__(self, loss_term_weight=1.0, reduce="mean"):
        super(DirectLoss, self).__init__(loss_term_weight)
        self.reduce = reduce

    def forward(self, loss):
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss)
        loss = loss.float()
        if loss.dim() > 0:
            loss = loss.mean() if self.reduce == "mean" else loss.sum()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
