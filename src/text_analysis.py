"""Deterministic, pre-model text analysis for the English M-ABSA corpus.

The functions in this module deliberately operate on raw examples and the
already-defined baseline scope.  They do not load a model, inspect predictions,
or change any train/dev/test data.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np

from src.data import Example, summarize_task_scope


WORD_RE = re.compile(r"\b\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*\b", re.UNICODE)


@dataclass(frozen=True)
class CorpusRecord:
    """One parsed example together with its source partition."""

    domain: str
    split: str
    example: Example


def count_words(text: str) -> int:
    """Count word-like units consistently without requiring an NLP model."""
    return len(WORD_RE.findall(text))


def _length_summary(values: Sequence[int]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0}
    data = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "mean": round(float(data.mean()), 2),
        "median": round(float(np.median(data)), 2),
        "p95": round(float(np.percentile(data, 95)), 2),
        "max": int(data.max()),
    }


def summarize_corpus(
    records: Sequence[CorpusRecord],
    expected_files: int | None = None,
) -> Dict[str, object]:
    """Return presentation-oriented corpus statistics.

    ``summarize_task_scope`` is reused so the visual report and the training
    pipeline apply exactly the same definition of a model-ready triplet.
    """
    if not records:
        raise ValueError("Cannot summarize an empty corpus")

    domains = list(dict.fromkeys(record.domain for record in records))
    splits = list(dict.fromkeys(record.split for record in records))
    examples = [record.example for record in records]
    word_lengths = [count_words(example.sentence) for example in examples]
    character_lengths = [len(example.sentence) for example in examples]
    scope = summarize_task_scope(examples)

    by_domain_split = Counter((record.domain, record.split) for record in records)
    triplets_by_domain_split = Counter()
    word_lengths_by_domain = defaultdict(list)
    sentiment_by_domain: Dict[str, Counter] = defaultdict(Counter)
    category_by_domain: Dict[str, Counter] = defaultdict(Counter)
    overall_sentiments = Counter()
    overall_categories = Counter()

    for record in records:
        example = record.example
        triplets_by_domain_split[(record.domain, record.split)] += len(example.triplets)
        word_lengths_by_domain[record.domain].append(count_words(example.sentence))
        for _, category, sentiment in example.triplets:
            sentiment_by_domain[record.domain][sentiment] += 1
            category_by_domain[record.domain][category] += 1
            overall_sentiments[sentiment] += 1
            overall_categories[category] += 1

    loaded_partitions = len({(record.domain, record.split) for record in records})
    expected = expected_files if expected_files is not None else loaded_partitions
    total_triplets = int(scope["total_triplets"])
    ready_triplets = int(scope["included_explicit_triplets"])

    return {
        "schema_version": 1,
        "dataset": "M-ABSA English",
        "validation": {
            "expected_source_files": expected,
            "loaded_source_files": loaded_partitions,
            "parsed_rows": len(records),
            "parse_failures": 0,
        },
        "overall": {
            "domains": len(domains),
            "splits": len(splits),
            "examples": len(records),
            "annotated_examples": sum(bool(example.triplets) for example in examples),
            "empty_annotation_examples": sum(not example.triplets for example in examples),
            "total_triplets": total_triplets,
            "unique_categories": len(overall_categories),
            "unique_sentiments": len(overall_sentiments),
            "model_ready_triplets": ready_triplets,
            "model_ready_percent": round(100 * ready_triplets / total_triplets, 2)
            if total_triplets else 0.0,
        },
        "word_length": {
            "overall": _length_summary(word_lengths),
            "by_domain": {
                domain: _length_summary(word_lengths_by_domain[domain])
                for domain in domains
            },
        },
        "character_length": {"overall": _length_summary(character_lengths)},
        "by_domain_split": [
            {
                "domain": domain,
                "split": split,
                "examples": by_domain_split[(domain, split)],
                "triplets": triplets_by_domain_split[(domain, split)],
            }
            for domain in domains
            for split in splits
        ],
        "sentiment_counts": {
            "overall": dict(overall_sentiments),
            "by_domain": {
                domain: dict(sentiment_by_domain[domain]) for domain in domains
            },
        },
        "category_counts": {
            "overall": dict(overall_categories.most_common()),
            "by_domain": {
                domain: dict(category_by_domain[domain].most_common())
                for domain in domains
            },
        },
        "baseline_scope": scope,
        "notes": {
            "word_count": "Regex word count; not BERT WordPiece token count.",
            "labels": "Sentiment and category counts include every raw annotation, including NULL aspects.",
            "model_ready": "Uses the same aligned explicit, single-label-pair selection policy as training.",
        },
    }


def records_to_summary_rows(summary: Dict[str, object]) -> List[Dict[str, object]]:
    """Flatten the key domain/split measures for a shareable CSV."""
    length_by_domain = summary["word_length"]["by_domain"]
    rows = []
    for item in summary["by_domain_split"]:
        length = length_by_domain[item["domain"]]
        rows.append(
            {
                **item,
                "median_words_domain": length["median"],
                "p95_words_domain": length["p95"],
            }
        )
    return rows

