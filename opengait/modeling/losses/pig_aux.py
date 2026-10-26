"""Optional auxiliary losses for PIGGait.

PIGGait computes phase/REx/counterfactual losses internally because they need
metadata such as sequence type/view. These generic losses are provided for
ablation experiments if you want to move scalar losses from the model into
OpenGait's LossAggregator.
"""

import torch
import torch.nn.functional as F
from .base import BaseLoss


class OrthogonalLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super().__init__(loss_term_weight)

    def forward(self, gait_feat, style_feat):
        gait_feat = F.normalize(gait_feat.float(), dim=1)
        style_feat = F.normalize(style_feat.float(), dim=1)
        loss = (gait_feat * style_feat).sum(dim=1).pow(2).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class EmbeddingConsistencyLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, detach_target=True):
        super().__init__(loss_term_weight)
        self.detach_target = detach_target

    def forward(self, embeddings, embeddings_cf):
        target = embeddings.detach() if self.detach_target else embeddings
        loss = F.mse_loss(F.normalize(embeddings_cf.float(), dim=1), F.normalize(target.float(), dim=1))
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
