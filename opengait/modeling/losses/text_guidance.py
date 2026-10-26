import torch
import torch.nn.functional as F

from .base import BaseLoss


class TextGuidanceContrastLoss(BaseLoss):
    """Contrast gait embeddings with Qwen2.5-VL-derived motion text features.

    embeddings:     [B, C, P]
    text_embeddings:[B, C]
    """

    def __init__(self, temperature=0.07, loss_term_weight=1.0):
        super(TextGuidanceContrastLoss, self).__init__(loss_term_weight)
        self.temperature = temperature

    def forward(self, embeddings, text_embeddings):
        if embeddings.dim() == 3:
            gait = embeddings.mean(-1)
        else:
            gait = embeddings
        gait = F.normalize(gait.float(), dim=-1)
        text = F.normalize(text_embeddings.float(), dim=-1)
        logits = gait @ text.t() / self.temperature
        labels = torch.arange(gait.size(0), device=gait.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
