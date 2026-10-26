import torch
import torch.nn.functional as F

from .base import BaseLoss


class CrossModalConsistencyLoss(BaseLoss):
    """Encourage RGB/Sil/PointCloud branches to describe the same identity motion.

    Args:
        modality_embeddings: Tensor [B, M, C, P], where M is the number of active modalities.
    """

    def __init__(self, loss_term_weight=1.0, mode='cosine'):
        super(CrossModalConsistencyLoss, self).__init__(loss_term_weight)
        self.mode = mode

    def forward(self, modality_embeddings):
        if modality_embeddings.dim() != 4:
            raise ValueError('modality_embeddings should be [B, M, C, P].')
        b, m, c, p = modality_embeddings.shape
        if m < 2:
            loss = modality_embeddings.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info

        x = modality_embeddings.flatten(2)  # [B,M,C*P]
        x = F.normalize(x, dim=-1)
        losses = []
        for i in range(m):
            for j in range(i + 1, m):
                if self.mode == 'mse':
                    losses.append(F.mse_loss(x[:, i], x[:, j], reduction='none').mean(-1))
                else:
                    losses.append(1.0 - (x[:, i] * x[:, j]).sum(-1))
        loss = torch.stack(losses, dim=0).mean()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
