"""CE mixtures and focal loss with explicit per-head controls."""

from collections import Counter
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
        probability = torch.exp(-ce)
        return (
            (1 - probability).pow(self.gamma) * ce
        ).mean()


def class_balanced_weights(
    labels,
    num_classes,
    beta=0.9999
):
    """
    Class-Balanced Loss using the effective number of samples.

    weight_c = (1 - beta) / (1 - beta ** n_c)
    """

    valid_labels = [
        label
        for label in labels
        if 0 <= label < num_classes
    ]

    counts = Counter(valid_labels)

    weights = torch.ones(
        num_classes,
        dtype=torch.float
    )

    observed_classes = []

    for c in range(num_classes):
        n = counts.get(c, 0)

        if n > 0:
            effective_number = 1.0 - beta ** n

            weights[c] = (
                (1.0 - beta) / effective_number
            )

            observed_classes.append(c)

    # Normalize observed class weights so their mean is 1.
    if observed_classes:
        observed_weights = weights[observed_classes]

        weights[observed_classes] = (
            observed_weights /
            observed_weights.mean()
        )

    return weights


def explicit_loss_functions(
    cfg,
    label_lists,
    vocabulary_sizes,
    device
):
    functions = []

    for head, labels, size in zip(
        ("bio", "category", "sentiment"),
        label_lists,
        vocabulary_sizes
    ):
        kind = (
            cfg.loss_type
            if head in cfg.loss_heads
            else "standard"
        )

        weights = inverse_frequency_weights(
            labels,
            size
        )

        if kind == "standard":

            function = nn.CrossEntropyLoss(
                ignore_index=-100
            )

        elif kind == "weighted":

            function = nn.CrossEntropyLoss(
                weight=weights,
                ignore_index=-100
            )

        elif kind == "mixed":

            function = MixedCrossEntropy(
                weights,
                cfg.ce_weight,
                cfg.weighted_ce_weight
            )

        elif kind == "focal":

            function = FocalCrossEntropy(
                cfg.focal_gamma
            )

        elif kind == "class_balanced":

            cb_weights = class_balanced_weights(
                labels,
                size,
                beta=cfg.class_balance_beta
            )

            function = nn.CrossEntropyLoss(
                weight=cb_weights,
                ignore_index=-100
            )

        else:
            raise ValueError(
                f"Unknown loss type: {kind}"
            )

        functions.append(
            function.to(device)
        )

    return tuple(functions)
def null_positive_weights(dataset, cap):
    """Train-only negative/positive ratios; absent positives receive weight 1."""

    if not len(dataset):
        raise ValueError(
            "Cannot compute NULL weights from an empty training dataset"
        )

    targets = dataset.null_targets

    positive = targets.sum(dim=0)

    ratio = (
        len(dataset) - positive
    ) / positive.clamp_min(1)

    return torch.where(
        positive > 0,
        ratio.clamp(min=1, max=cap),
        torch.ones_like(positive)
    )