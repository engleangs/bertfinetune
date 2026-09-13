"""Reusable dataset-audit logic for the restricted M-ABSA baseline.

These functions inspect data only. They must not change target selection,
training, inference, or evaluation behaviour.
"""

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from src.data import (
    ABSADataset,
    Example,
    summarize_task_scope,
    summarize_tokenized_targets,
)


def count_empty_annotations(examples: Sequence[Example]) -> int:
    """Return the number of examples containing no annotated triplets."""
    return sum(not example.triplets for example in examples)


def find_all_spans(sentence: str, aspect_term: str) -> List[Tuple[int, int]]:
    """Return every non-overlapping, case-insensitive aspect occurrence.

    M-ABSA annotations contain surface text rather than character offsets. If
    the same text occurs more than once, this function exposes that ambiguity;
    it deliberately does not guess which occurrence an annotation refers to.
    """
    term = aspect_term.strip()
    if not term or term.casefold() == "null":
        return []

    folded_sentence = sentence.casefold()
    folded_term = term.casefold()
    spans = []
    start = 0
    while True:
        index = folded_sentence.find(folded_term, start)
        if index < 0:
            return spans
        spans.append((index, index + len(term)))
        start = index + max(1, len(term))


def repeated_term_summary(examples: Sequence[Example]) -> Dict[str, int]:
    """Count repeated annotations separately from repeated sentence text."""
    repeated_annotation_examples = 0
    ambiguous_occurrence_examples = 0
    ambiguous_occurrence_triplets = 0

    for example in examples:
        explicit_aspects = [
            aspect.strip().casefold()
            for aspect, _, _ in example.triplets
            if aspect.strip() and aspect.strip().casefold() != "null"
        ]
        if any(count > 1 for count in Counter(explicit_aspects).values()):
            repeated_annotation_examples += 1

        ambiguous_in_example = 0
        for aspect, _, _ in example.triplets:
            if len(find_all_spans(example.sentence, aspect)) > 1:
                ambiguous_in_example += 1
        if ambiguous_in_example:
            ambiguous_occurrence_examples += 1
            ambiguous_occurrence_triplets += ambiguous_in_example

    return {
        "repeated_annotated_aspect_text_examples": repeated_annotation_examples,
        "ambiguous_repeated_occurrence_examples": ambiguous_occurrence_examples,
        "ambiguous_repeated_occurrence_triplets": ambiguous_occurrence_triplets,
    }


def audit_split(
    domain: str,
    split: str,
    examples: Sequence[Example],
    tokenizer,
    category_vocab: Sequence[str],
    sentiment_vocab: Sequence[str],
    max_len: int,
) -> Dict[str, object]:
    """Build annotation- and tokenization-level statistics for one split."""
    task_scope = summarize_task_scope(
        examples,
        category_vocab=category_vocab,
        sentiment_vocab=sentiment_vocab,
    )
    dataset = ABSADataset(
        examples=list(examples),
        tokenizer=tokenizer,
        category_vocab=list(category_vocab),
        sentiment_vocab=list(sentiment_vocab),
        max_len=max_len,
    )
    token_summary = summarize_tokenized_targets(dataset)

    # Both existing summaries include `examples`. Keep one authoritative field.
    task_scope.pop("examples", None)
    token_summary.pop("examples", None)
    return {
        "domain": domain,
        "split": split,
        "examples": len(examples),
        "empty_annotation_examples": count_empty_annotations(examples),
        **repeated_term_summary(examples),
        **task_scope,
        **token_summary,
    }


def split_overlap_summary(
    train_examples: Sequence[Example],
    dev_examples: Sequence[Example],
    test_examples: Sequence[Example],
) -> Dict[str, int]:
    """Count repeated sentences and identical labeled rows across splits."""
    def sentence_set(examples: Iterable[Example]):
        return {example.sentence for example in examples}

    def labeled_set(examples: Iterable[Example]):
        return {
            (example.sentence, tuple(example.triplets)) for example in examples
        }

    train_sentences, dev_sentences, test_sentences = map(
        sentence_set, (train_examples, dev_examples, test_examples),
    )
    train_labeled, dev_labeled, test_labeled = map(
        labeled_set, (train_examples, dev_examples, test_examples),
    )
    return {
        "train_dev_repeated_sentences": len(train_sentences & dev_sentences),
        "train_test_repeated_sentences": len(train_sentences & test_sentences),
        "dev_test_repeated_sentences": len(dev_sentences & test_sentences),
        "train_dev_identical_labeled_examples": len(train_labeled & dev_labeled),
        "train_test_identical_labeled_examples": len(train_labeled & test_labeled),
        "dev_test_identical_labeled_examples": len(dev_labeled & test_labeled),
    }


def aggregate_audits(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate additive counts and union unseen-label names."""
    non_additive = {"domain", "split", "unseen_categories", "unseen_sentiments"}
    totals: Dict[str, object] = {}
    for row in rows:
        for key, value in row.items():
            if key not in non_additive and isinstance(value, int):
                totals[key] = int(totals.get(key, 0)) + value

    totals["unseen_categories"] = sorted({
        label for row in rows for label in row.get("unseen_categories", [])
    })
    totals["unseen_sentiments"] = sorted({
        label for row in rows for label in row.get("unseen_sentiments", [])
    })
    return totals
