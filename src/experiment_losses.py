"""CE mixtures and focal loss with explicit per-head controls."""

import torch
from torch import nn
from torch.nn import functional as F

from src.losses import inverse_frequency_weights


class MixedCrossEntropy(nn.Module):
    def __init__(self, weights, ce_weight, weighted_ce_weight):
        super().__init__()
        self.register_buffer("weights", weights)
        total = ce_weight + weighted_ce_weight
        self.alpha = weighted_ce_weight / total

    def forward(self, logits, targets):
        valid = targets != -100
        if not valid.any():
            return logits.sum() * 0
        logits, targets = logits[valid], targets[valid]
        ce = F.cross_entropy(logits, targets)
        weighted = F.cross_entropy(logits, targets, weight=self.weights)
        return (1 - self.alpha) * ce + self.alpha * weighted


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        valid = targets != -100
        if not valid.any():
            return logits.sum() * 0
        ce = F.cross_entropy(logits[valid], targets[valid], reduction="none")
        probability = torch.exp(-ce)  # Unweighted CE gives the true target probability.
        return ((1 - probability).pow(self.gamma) * ce).mean()


def explicit_loss_functions(cfg, label_lists, vocabulary_sizes, device):
    functions = []
    for head, labels, size in zip(("bio", "category", "sentiment"), label_lists, vocabulary_sizes):
        kind = cfg.loss_type if head in cfg.loss_heads else "standard"
        weights = inverse_frequency_weights(labels, size)
        if kind == "standard":
            function = nn.CrossEntropyLoss(ignore_index=-100)
        elif kind == "weighted":
            function = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        elif kind == "mixed":
            function = MixedCrossEntropy(weights, cfg.ce_weight, cfg.weighted_ce_weight)
        else:
            function = FocalCrossEntropy(cfg.focal_gamma)
        functions.append(function.to(device))
    return tuple(functions)


def null_positive_weights(dataset, cap):
    """Train-only negative/positive ratios; absent positives receive weight 1."""
    if not len(dataset):
        raise ValueError("Cannot compute NULL weights from an empty training dataset")
    targets = dataset.null_targets
    positive = targets.sum(dim=0)
    ratio = (len(dataset) - positive) / positive.clamp_min(1)
    return torch.where(positive > 0, ratio.clamp(min=1, max=cap), torch.ones_like(positive))
