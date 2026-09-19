"""Loss functions for the primary comparison and optional head ablations."""
from collections import Counter
from typing import List

import torch
import torch.nn as nn


WEIGHTED_HEADS = {
    "standard": frozenset(),
    "weighted": frozenset({"bio", "category", "sentiment"}),
    "category_weighted": frozenset({"category"}),
}


def inverse_frequency_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    labels = [label for label in labels if 0 <= label < num_classes]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.ones(num_classes)
    for c in range(num_classes):
        count_c = counts.get(c, 0)
        if count_c > 0:
            weights[c] = total / (num_classes * count_c)
    return weights


def get_loss_fns(
    cfg,
    bio_label_counts,
    category_label_counts,
    sentiment_label_counts,
    num_categories=None,
    num_sentiments=None,
):
    if num_categories is None:
        num_categories = max(category_label_counts, default=-1) + 1
    if num_sentiments is None:
        num_sentiments = max(sentiment_label_counts, default=-1) + 1

    if cfg.loss_type not in WEIGHTED_HEADS:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")

    weighted = WEIGHTED_HEADS[cfg.loss_type]
    labels_and_sizes = (
        ("bio", bio_label_counts, 3),
        ("category", category_label_counts, num_categories),
        ("sentiment", sentiment_label_counts, num_sentiments),
    )
    return tuple(
        nn.CrossEntropyLoss(
            weight=(
                inverse_frequency_weights(labels, num_classes=size)
                if head in weighted else None
            ),
            ignore_index=-100,
        )
        for head, labels, size in labels_and_sizes
    )


def get_evaluation_loss_fns():
    """Return unweighted summed losses for corpus-level dev/test reporting."""
    return (
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
    )
