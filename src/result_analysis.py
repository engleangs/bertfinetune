"""Reusable analysis helpers for completed experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple


Triplet = Tuple[str, str, str]


def load_prediction_file(path: Path) -> Tuple[List[List[Triplet]], List[List[Triplet]]]:
    """Load gold and predicted triplets from a saved JSONL artifact."""
    gold, predicted = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                gold.append([tuple(item) for item in row["gold_triplets"]])
                predicted.append([tuple(item) for item in row["predicted_triplets"]])
            except KeyError as exc:
                raise ValueError(f"Missing {exc.args[0]!r} on line {line_number} of {path}") from exc
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction example counts differ")
    return gold, predicted


def projected_micro_scores(
    gold: Sequence[Sequence[Triplet]],
    predicted: Sequence[Sequence[Triplet]],
    project: Callable[[Triplet], object] = lambda triplet: triplet,
    keep: Callable[[Triplet], bool] = lambda triplet: True,
) -> Dict[str, float | int]:
    """Calculate set-based micro scores after filtering/projecting triplets."""
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction example counts differ")
    true_positives = false_positives = false_negatives = 0
    for gold_items, predicted_items in zip(gold, predicted):
        gold_set = {project(item) for item in gold_items if keep(item)}
        predicted_set = {project(item) for item in predicted_items if keep(item)}
        true_positives += len(gold_set & predicted_set)
        false_positives += len(predicted_set - gold_set)
        false_negatives += len(gold_set - predicted_set)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def derive_crossdomain_metrics(
    gold: Sequence[Sequence[Triplet]],
    predicted: Sequence[Sequence[Triplet]],
    source_categories: Iterable[str],
) -> Dict[str, object]:
    """Derive the cross-domain views required by protocol section 5."""
    known = set(source_categories)
    gold_triplets = [set(items) for items in gold]
    gold_total = sum(len(items) for items in gold_triplets)
    gold_known = sum(sum(item[1] in known for item in items) for items in gold_triplets)
    full = projected_micro_scores(gold, predicted)
    seen = projected_micro_scores(gold, predicted, keep=lambda item: item[1] in known)
    aspect = projected_micro_scores(gold, predicted, project=lambda item: item[0])
    aspect_sentiment = projected_micro_scores(
        gold, predicted, project=lambda item: (item[0], item[2]),
    )
    unseen_categories = sorted({item[1] for items in gold_triplets for item in items if item[1] not in known})
    return {
        "full_exact_triplet": full,
        "source_known_exact_triplet": seen,
        "aspect_span": aspect,
        "aspect_plus_sentiment": aspect_sentiment,
        "gold_category_coverage": gold_known / gold_total if gold_total else 0.0,
        "gold_triplets": gold_total,
        "source_known_gold_triplets": gold_known,
        "unseen_gold_triplets": gold_total - gold_known,
        "unseen_categories": unseen_categories,
        "unseen_category_count": len(unseen_categories),
    }


def validate_complete_matrix(df, modes, configs, seeds) -> None:
    """Fail early unless exactly one complete result exists for every planned key."""
    keys = ["mode", "config", "seed"]
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate experiment keys: {df.loc[duplicates, keys].to_dict('records')}")
    complete = df[df["status"] == "complete"] if "status" in df else df
    actual = {(row.mode, row.config, int(row.seed)) for row in complete.itertuples()}
    expected = {(mode, config, int(seed)) for mode in modes for config in configs for seed in seeds}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"Incomplete experiment matrix; missing={missing}, unexpected={unexpected}")

