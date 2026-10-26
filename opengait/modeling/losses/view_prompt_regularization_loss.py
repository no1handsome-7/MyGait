# -*- coding: utf-8 -*-
"""
View prompt regularization loss for OpenGait.

Put this file into:
    OpenGait/opengait/modeling/losses/view_prompt_regularization_loss.py

Then add this import to:
    OpenGait/opengait/modeling/losses/__init__.py

    from .view_prompt_regularization_loss import ViewPromptRegularizationLoss
"""

import torch
import torch.nn as nn


class ViewPromptRegularizationLoss(nn.Module):
    """
    Combine prompt usage balance and prompt orthogonality regularizers.

    Expected training_feat item:
        training_feat["view_prompt_reg"] = {
            "balance_loss": scalar_tensor,
            "orthogonality_loss": scalar_tensor,
        }

    The model already computes:
      - balance_loss = KL(mean_alpha || uniform)
      - orthogonality_loss = mean squared off-diagonal prompt cosine similarity
    """
    def __init__(
        self,
        loss_term_weight: float = 1.0,
        balance_weight: float = 0.002,
        orthogonality_weight: float = 0.005,
        log_prefix: str = "view_prompt_reg",
    ):
        super().__init__()
        self.loss_term_weight = float(loss_term_weight)
        self.balance_weight = float(balance_weight)
        self.orthogonality_weight = float(orthogonality_weight)
        self.log_prefix = log_prefix

    def forward(self, balance_loss=None, orthogonality_loss=None, **kwargs):
        if balance_loss is None and orthogonality_loss is None:
            # Keep device consistent if possible.
            ref = None
            for v in kwargs.values():
                if torch.is_tensor(v):
                    ref = v
                    break
            if ref is None:
                loss = torch.tensor(0.0)
            else:
                loss = ref.sum() * 0.0
            info = {"loss": loss.detach()}
            return loss, info

        loss = 0.0
        info = {}

        if balance_loss is not None:
            loss = loss + self.balance_weight * balance_loss
            info["balance"] = balance_loss.detach()

        if orthogonality_loss is not None:
            loss = loss + self.orthogonality_weight * orthogonality_loss
            info["orthogonality"] = orthogonality_loss.detach()

        loss = self.loss_term_weight * loss
        info["loss"] = loss.detach()
        return loss, info
