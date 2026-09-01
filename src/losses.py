"""unchanged from original scaffold — standard vs class-weighted loss
is still the one variable that must differ between the two configs."""
from collections import Counter
from typing import List

import torch
import torch.nn as nn


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

    if cfg.loss_type == "standard":
        bio_fn = nn.CrossEntropyLoss(ignore_index=-100)
        cat_fn = nn.CrossEntropyLoss(ignore_index=-100)
        sent_fn = nn.CrossEntropyLoss(ignore_index=-100)
    elif cfg.loss_type == "weighted":
        bio_w = inverse_frequency_weights(bio_label_counts, num_classes=3)
        cat_w = inverse_frequency_weights(category_label_counts, num_classes=num_categories)
        sent_w = inverse_frequency_weights(sentiment_label_counts, num_classes=num_sentiments)
        bio_fn = nn.CrossEntropyLoss(weight=bio_w, ignore_index=-100)
        cat_fn = nn.CrossEntropyLoss(weight=cat_w, ignore_index=-100)
        sent_fn = nn.CrossEntropyLoss(weight=sent_w, ignore_index=-100)
    else:
        raise ValueError(f"Unknown loss_type: {cfg.loss_type}")
    return bio_fn, cat_fn, sent_fn


def get_evaluation_loss_fns():
    """Return unweighted summed losses for corpus-level dev/test reporting."""
    return (
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
        nn.CrossEntropyLoss(ignore_index=-100, reduction="sum"),
    )
