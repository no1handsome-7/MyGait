import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


def _l2n(x, dim=-1, eps=1e-12):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class ScalarLoss(BaseLoss):
    """Pass-through scalar loss for auxiliary losses emitted by model.training_feat.

    Expected training_feat item:
      {'loss': tensor_scalar}

    Use in loss_cfg with log_prefix: gaitoracle_aux.
    """

    def __init__(self, loss_term_weight=1.0):
        super(ScalarLoss, self).__init__(loss_term_weight)

    def forward(self, loss, labels=None):
        if not torch.is_tensor(loss):
            loss = torch.tensor(float(loss))
        loss = loss.float()
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class SymmetricInfoNCELoss(BaseLoss):
    """Generic symmetric InfoNCE loss for cross-modal feature alignment."""

    def __init__(self, temperature=0.07, loss_term_weight=1.0):
        super(SymmetricInfoNCELoss, self).__init__(loss_term_weight)
        self.temperature = temperature

    @gather_and_scale_wrapper
    def forward(self, feat_a, feat_b, labels=None):
        if feat_a.dim() == 3:
            feat_a = feat_a.mean(-1)
        if feat_b.dim() == 3:
            feat_b = feat_b.mean(-1)
        if feat_a.size(0) <= 1:
            loss = feat_a.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        feat_a = _l2n(feat_a.float(), -1)
        feat_b = _l2n(feat_b.float(), -1)
        logits = feat_a @ feat_b.t() / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info


class MaskedTextInfoNCELoss(BaseLoss):
    """Masked text-image InfoNCE, useful for Qwen nuisance caption supervision."""

    def __init__(self, temperature=0.07, loss_term_weight=1.0):
        super(MaskedTextInfoNCELoss, self).__init__(loss_term_weight)
        self.temperature = temperature

    def forward(self, feat_img, feat_txt, mask=None):
        if feat_img.dim() == 3:
            feat_img = feat_img.mean(-1)
        if feat_txt.dim() == 3:
            feat_txt = feat_txt.mean(1)
        if mask is not None:
            mask = mask.float().view(mask.size(0), -1).mean(-1)
            valid = mask > 0
            if valid.sum() <= 1:
                loss = feat_img.new_tensor(0.0)
                self.info.update({'loss': loss.detach().clone()})
                return loss, self.info
            feat_img = feat_img[valid]
            feat_txt = feat_txt[valid]
        if feat_img.size(0) <= 1:
            loss = feat_img.new_tensor(0.0)
            self.info.update({'loss': loss.detach().clone()})
            return loss, self.info
        feat_img = _l2n(feat_img.float(), -1)
        feat_txt = _l2n(feat_txt.float(), -1)
        logits = feat_img @ feat_txt.t() / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
