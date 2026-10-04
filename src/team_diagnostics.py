"""Set-based error diagnostics for saved ABSA prediction records.

These diagnostics use exact surface strings, like the current triplet metric.
They cannot distinguish repeated occurrences of the same term without offsets.
The primary error is the first failed component (term, category, sentiment),
while independent flags retain cases where several components are wrong.
"""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Iterable, Mapping, Sequence

from src.result_analysis import derive_crossdomain_metrics, projected_micro_scores


OUTCOMES = ("correct", "term", "category", "sentiment")
FREQUENCY_BANDS = ("unseen", "rare", "low", "medium", "frequent")


def category_frequency_band(count: int) -> str:
    """Use fixed training-only category frequency bands, including unseen = 0."""
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        raise ValueError("Category frequency must be a non-negative integer")
    if count == 0:
        return "unseen"
    if count <= 5:
        return "rare"
    if count <= 20:
        return "low"
    if count <= 100:
        return "medium"
    return "frequent"


def _triplets(record: Mapping, key: str) -> list[tuple[str, str, str]]:
    if key not in record:
        raise ValueError(f"Prediction record is missing {key!r}")
    items = record[key]
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{key} must be a list of three-string triplets")
    result = []
    for item in items:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
        ):
            raise ValueError(f"Invalid {key} triplet: {item!r}")
        result.append(tuple(item))
    return result


def _accuracy(correct: int, denominator: int) -> dict:
    return {
        "correct": correct,
        "denominator": denominator,
        "accuracy": correct / denominator if denominator else None,
    }


def _group_outcomes(rows: list[dict], key: str, values: Iterable) -> list[dict]:
    groups = []
    for value in values:
        members = [row for row in rows if row[key] == value]
        counts = Counter(row["outcome"] for row in members)
        groups.append({
            key: value,
            "gold_triplets": len(members),
            "primary_outcome_counts": {name: counts[name] for name in OUTCOMES},
            "error_rate": 1 - counts["correct"] / len(members) if members else None,
        })
    return groups


def analyze_records(
    records: Sequence[Mapping],
    category_counts: Mapping[str, int],
    source_categories: Iterable[str],
) -> tuple[dict, list[dict]]:
    """Return a JSON-friendly summary and one diagnostic row per unique gold.

    ``category_counts`` must count eligible training triplets from this run's
    source data. Its positive-frequency labels must equal ``source_categories``;
    do not use pooled train/dev/test frequencies for a cross-domain diagnosis.
    Records require ``gold_triplets`` and ``predicted_triplets``; domain,
    example_id, and sentence are optional metadata.

    Gold triplets and predictions are deduplicated within each example. Each
    unique gold receives exactly one primary outcome: correct, missing exact
    term, incorrect category on that term, or incorrect sentiment on that term
    and category. This hierarchy describes the first failed component, not a
    claim about the sole causal error. Predictions may support more than one
    gold row; this is a membership diagnosis, not a one-to-one span assignment.

    Independent correctness flags check any prediction on the exact term.
    For examples/confusion only, choose an exact triplet if present, otherwise
    a matching term+category, otherwise a matching term; break ties by sorting.
    """
    if isinstance(source_categories, str):
        raise ValueError("source_categories must be an iterable of label strings")
    known = set(source_categories)
    frequencies = {}
    for label, count in category_counts.items():
        category_frequency_band(count)  # Validate without silently coercing floats.
        if not isinstance(label, str):
            raise ValueError("Category labels must be strings")
        frequencies[label] = int(count)
    if known != {label for label, count in frequencies.items() if count > 0}:
        raise ValueError("Source categories must equal positive-frequency training categories")

    gold_batches, predicted_batches, rows = [], [], []
    gold_duplicates = prediction_duplicates = 0
    unique_predictions = incorrect_predictions = spurious_aspects = spurious_triplets = 0
    for index, record in enumerate(records):
        raw_gold = _triplets(record, "gold_triplets")
        raw_predicted = _triplets(record, "predicted_triplets")
        gold, predicted = set(raw_gold), set(raw_predicted)
        gold_duplicates += len(raw_gold) - len(gold)
        prediction_duplicates += len(raw_predicted) - len(predicted)
        gold_batches.append(sorted(gold))
        predicted_batches.append(sorted(predicted))
        unique_predictions += len(predicted)
        incorrect_predictions += len(predicted - gold)
        gold_terms = {item[0] for item in gold}
        spurious_aspects += len({item[0] for item in predicted} - gold_terms)
        spurious_triplets += sum(item[0] not in gold_terms for item in predicted)

        by_term = {}
        for item in sorted(predicted):
            by_term.setdefault(item[0], []).append(item)
        for item in sorted(gold):
            term, category, sentiment = item
            candidates = by_term.get(term, [])
            same_category = [candidate for candidate in candidates if candidate[1] == category]
            if item in predicted:
                outcome, chosen = "correct", item
            elif not candidates:
                outcome, chosen = "term", None
            elif not same_category:
                outcome, chosen = "category", candidates[0]
            else:
                outcome, chosen = "sentiment", same_category[0]
            frequency = frequencies.get(category, 0)
            rows.append({
                "domain": record.get("domain", "unknown"),
                "example_id": record.get("example_id", index),
                "sentence": record.get("sentence", ""),
                "gold": list(item),
                "predicted": list(chosen) if chosen is not None else None,
                "outcome": outcome,
                "term_correct": bool(candidates),
                "category_correct_on_term": bool(same_category),
                "sentiment_correct_on_term": any(candidate[2] == sentiment for candidate in candidates),
                "sentiment_correct_on_term_category": (
                    any(candidate[2] == sentiment for candidate in same_category)
                    if same_category else None
                ),
                "source_category_seen": category in known,
                "source_category_frequency": frequency,
                "rarity": category_frequency_band(frequency),
                "matching_term_prediction_count": len(candidates),
            })

    metrics = derive_crossdomain_metrics(gold_batches, predicted_batches, known)
    metrics["term_plus_category"] = projected_micro_scores(
        gold_batches, predicted_batches, project=lambda item: (item[0], item[1]),
    )
    matched = [row for row in rows if row["term_correct"]]
    category_matched = [row for row in matched if row["category_correct_on_term"]]
    outcomes = Counter(row["outcome"] for row in rows)

    # Rows are gold labels; columns are the deterministically chosen predictions.
    labels = sorted({row["gold"][2] for row in matched} | {row["predicted"][2] for row in matched})
    label_indices = {label: index for index, label in enumerate(labels)}
    confusion = [[0] * len(labels) for _ in labels]
    for row in matched:
        confusion[label_indices[row["gold"][2]]][label_indices[row["predicted"][2]]] += 1

    summary = {
        "schema_version": 1,
        "examples": len(records),
        "gold_triplets": len(rows),
        "metrics": metrics,
        "primary_outcome_counts": {name: outcomes[name] for name in OUTCOMES},
        "duplicate_counts": {"gold": gold_duplicates, "predicted": prediction_duplicates},
        "prediction_counts": {
            "unique_triplets": unique_predictions,
            "incorrect_triplets": incorrect_predictions,
            "spurious_aspects": spurious_aspects,
            "spurious_triplets": spurious_triplets,
            "incorrect_label_triplets_on_gold_term": incorrect_predictions - spurious_triplets,
        },
        "category_mismatch_with_correct_term_sentiment": sum(
            row["outcome"] == "category" and row["sentiment_correct_on_term"] for row in rows
        ),
        "conditional_accuracy": {
            "category_on_matched_term": _accuracy(
                sum(row["category_correct_on_term"] for row in matched), len(matched),
            ),
            "sentiment_on_matched_term": _accuracy(
                sum(row["sentiment_correct_on_term"] for row in matched), len(matched),
            ),
            "sentiment_on_matched_term_and_category": _accuracy(
                sum(row["sentiment_correct_on_term_category"] for row in category_matched),
                len(category_matched),
            ),
        },
        "sentiment_confusion": {
            "labels": labels,
            "matrix": confusion,
            "matched_gold": len(matched),
            "unmatched_gold": len(rows) - len(matched),
            "chosen_prediction_accuracy": _accuracy(
                sum(row["gold"][2] == row["predicted"][2] for row in matched), len(matched),
            ),
        },
        "by_rarity": _group_outcomes(rows, "rarity", FREQUENCY_BANDS),
        "by_source_seen": _group_outcomes(rows, "source_category_seen", (True, False)),
        "notes": {
            "matching": "Exact, case-sensitive surface strings; repeated occurrences require offsets.",
            "primary_error": "First failed component, with independent flags for other component failures.",
            "frequencies": "Eligible source training triplets only; unseen=0, rare=1-5, low=6-20, medium=21-100, frequent>100.",
            "confusion": "Conditional on exact term; exact triplet then same category then lexicographic selection.",
            "conditional_flags": "Membership accuracy over unique gold triplets; multiple predictions on a term may satisfy different flags.",
            "duplicates": "Counted separately and removed from scores and primary outcomes.",
        },
    }
    return summary, rows
