"""Stable binary focal/ASL and adaptive, positive-preserving negative sampling."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def selected_cells(logits, targets, cfg):
    """All positives; high-scoring hard negatives plus random negatives."""
    positive = targets.bool()
    if cfg.negative_sampling == "all":
        return torch.ones_like(positive)
    mask = positive.clone()
    with torch.no_grad():
        for index in range(len(logits)):
            negatives = (~positive[index]).nonzero().flatten()
            budget = min(len(negatives), max(cfg.minimum_negatives,
                         math.ceil(int(positive[index].sum()) * cfg.negative_ratio)))
            hard_count = math.floor(budget * cfg.hard_fraction) if cfg.negative_sampling == "hard" else 0
            if hard_count:
                hard = negatives[logits[index, negatives].detach().topk(hard_count).indices]
                mask[index, hard] = True
            available = negatives[~mask[index, negatives]]
            random_count = budget - hard_count
            if random_count:
                random = available[torch.randperm(len(available), device=available.device)[:random_count]]
                mask[index, random] = True
    return mask


class NullLoss(nn.Module):
    def __init__(self, cfg, positive_weights):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("positive_weights", positive_weights)

    def elementwise(self, logits, targets):
        # Loss arithmetic stays FP32 even when BERT runs under autocast.
        logits, targets = logits.float(), targets.float()
        if self.cfg.null_loss != "asl":
            loss = F.binary_cross_entropy_with_logits(logits, targets,
                       pos_weight=self.positive_weights, reduction="none")
            if self.cfg.null_loss == "focal":
                probability = logits.sigmoid()
                p_target = torch.where(targets.bool(), probability, 1 - probability)
                loss = loss * (1 - p_target).pow(self.cfg.null_focal_gamma)
            return loss
        probability = logits.sigmoid()
        negative_probability = (1 - probability + self.cfg.asl_margin).clamp(max=1)
        positive_loss = -(1 - probability).pow(self.cfg.asl_gamma_positive) * F.logsigmoid(logits)
        negative_loss = -(1 - negative_probability).pow(self.cfg.asl_gamma_negative) * negative_probability.clamp_min(1e-8).log()
        # ASL does not silently stack inverse-frequency positive weights.
        return targets * positive_loss + (1 - targets) * negative_loss

    def forward(self, logits, targets):
        elements = self.elementwise(logits, targets)
        if self.cfg.negative_sampling == "all":
            return elements.mean()
        mask = selected_cells(logits, targets, self.cfg)
        positive = targets.bool()
        # Equal positive/negative-group contribution for sampled methods. The
        # random control uses the same normalization as hard-negative mining.
        pos_count = positive.sum(-1)
        negative = mask & ~positive
        negative_count = negative.sum(-1)
        pos_mean = (elements * positive).sum(-1) / pos_count.clamp_min(1)
        neg_mean = (elements * negative).sum(-1) / negative_count.clamp_min(1)
        groups = (pos_count > 0).int() + (negative_count > 0).int()
        return ((pos_mean + neg_mean) / groups.clamp_min(1)).mean()
