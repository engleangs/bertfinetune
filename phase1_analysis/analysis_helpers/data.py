"""Version-1 parsing and retained-gold rules, copied without training classes."""

import ast
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Example:
    sentence: str
    triplets: List[Tuple[str, str, str]]  # (aspect_term, category, sentiment)
    domain: str = "unknown"


SENTIMENT_ALIASES = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
    "conflict": "conflict",
}


def normalize_sentiment(sentiment: str) -> str:
    """Map M-ABSA polarity spelling/case variants to canonical labels."""
    cleaned = sentiment.strip().casefold()
    return SENTIMENT_ALIASES.get(cleaned, cleaned)


def parse_line(line: str, domain: str = "unknown") -> Example:
    parts = line.strip().split("####", 1)
    if len(parts) != 2:
        raise ValueError("Expected 'sentence####triplets' input format")

    sentence, raw_triplets = parts
    parsed = ast.literal_eval(raw_triplets)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Triplet annotation must be a list")

    triplets = []
    for index, triplet in enumerate(parsed):
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            raise ValueError(f"Annotation {index} is not a three-field triplet")
        if not all(isinstance(value, str) for value in triplet):
            raise ValueError(f"Annotation {index} contains a non-string field")

        aspect, category, sentiment = triplet
        triplets.append(
            (aspect.strip(), category.strip(), normalize_sentiment(sentiment))
        )

    return Example(sentence=sentence, triplets=triplets, domain=domain)


def load_domain_file(path: str, domain: str) -> List[Example]:
    with open(path, encoding="utf-8") as f:
        return [parse_line(line, domain=domain) for line in f if line.strip()]


def label_frequencies(examples: List[Example]) -> Tuple[dict, dict]:
    """Counts per category / sentiment label in the TRAINING set — this is what
    'rare label' means downstream (see evaluate.identify_rare_labels)."""
    cat_counts, sent_counts = {}, {}
    for ex in examples:
        for _, category, sentiment in aligned_explicit_triplets(ex):
            cat_counts[category] = cat_counts.get(category, 0) + 1
            sent_counts[sentiment] = sent_counts.get(sentiment, 0) + 1
    return cat_counts, sent_counts


def find_span(sentence: str, aspect_term: str) -> Tuple[int, int]:
    if aspect_term.strip().casefold() == "null":
        return -1, -1

    idx = sentence.find(aspect_term)
    if idx == -1:
        idx = sentence.casefold().find(aspect_term.casefold())
    if idx == -1:
        return -1, -1
    return idx, idx + len(aspect_term)


def _select_baseline_triplets(example: Example):
    """Select one label pair per non-overlapping aligned explicit span."""
    selected = []
    selected_spans = []
    seen_triplets = set()
    counts = {
        "total": 0,
        "implicit": 0,
        "unaligned": 0,
        "duplicate": 0,
        "additional_label_pair": 0,
        "overlapping": 0,
    }

    for aspect, category, sentiment in example.triplets:
        counts["total"] += 1
        if aspect.strip().casefold() == "null":
            counts["implicit"] += 1
            continue
        char_start, char_end = find_span(example.sentence, aspect)
        if char_start == -1:
            counts["unaligned"] += 1
            continue
        canonical = (
            example.sentence[char_start:char_end], category, sentiment,
        )
        if canonical in seen_triplets:
            counts["duplicate"] += 1
            continue
        seen_triplets.add(canonical)

        span = (char_start, char_end)
        if span in selected_spans:
            counts["additional_label_pair"] += 1
            continue
        if any(
            char_start < selected_end and selected_start < char_end
            for selected_start, selected_end in selected_spans
        ):
            counts["overlapping"] += 1
            continue

        selected_spans.append(span)
        selected.append(canonical)
    return selected, counts


def aligned_explicit_triplets(example: Example) -> List[Tuple[str, str, str]]:
    """Return targets representable by the first single-pair baseline.

    The first annotation for a character span is retained. Exact duplicates,
    additional label pairs for that span, overlapping spans, implicit aspects,
    and unaligned annotations are excluded and reported in run metadata.
    """
    return _select_baseline_triplets(example)[0]
