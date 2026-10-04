"""Corrected annotation eligibility and optional sentence-level NULL targets."""

from collections import Counter
from pathlib import Path
import re

import torch

import config
from src.data import ABSADataset, Example, collate_fn, load_domain_file


def is_null(term):
    return term.strip().casefold() == "null"


def select_targets(example, drop_conflict=False):
    """Exclude all members of an overlap/conflict, independent of annotation order."""
    counts = Counter()
    spans = {}
    implicit = set()
    seen = set()
    for term, category, sentiment in example.triplets:
        counts["raw_annotations"] += 1
        if drop_conflict and sentiment == "conflict":
            counts["conflict_removed"] += 1
            continue
        if is_null(term):
            triplet = ("NULL", category, sentiment)
            if triplet in implicit:
                counts["duplicates"] += 1
            implicit.add(triplet)
            continue
        matches = [m.span(1) for m in re.finditer(f"(?=({re.escape(term)}))", example.sentence, re.IGNORECASE)] if term else []
        if not matches:
            counts["unaligned"] += 1
            continue
        if len(matches) != 1:
            counts["ambiguous_occurrence"] += 1
            continue
        start, end = matches[0]
        triplet = (example.sentence[start:end], category, sentiment)
        if triplet in seen:
            counts["duplicates"] += 1
            continue
        seen.add(triplet)
        spans.setdefault((start, end), set()).add(triplet)
    conflicting = {span for span, pairs in spans.items() if len(pairs) > 1}
    overlapping = {a for a in spans if any(a != b and a[0] < b[1] and b[0] < a[1] for b in spans)}
    retained = [next(iter(spans[span])) for span in sorted(spans) if span not in conflicting | overlapping]
    counts["conflicting_span_annotations"] = sum(len(spans[s]) for s in conflicting)
    counts["overlapping_span_annotations"] = sum(len(spans[s]) for s in overlapping)
    counts["retained_explicit"] = len(retained)
    counts["retained_null"] = len(implicit)
    # Exclusion reason diagnostics can overlap; retained counts have set semantics.
    return retained, sorted(implicit), dict(counts)


def load_source_splits(root, mode, held_out_domain=None):
    """Load train/dev only. Pilot execution never reads test files."""
    if mode not in ("indomain", "crossdomain"):
        raise ValueError("Unknown split mode")
    if mode == "crossdomain" and held_out_domain not in config.DOMAINS:
        raise ValueError("Cross-domain mode requires a valid held-out domain")
    sources = [d for d in config.DOMAINS if mode == "indomain" or d != held_out_domain]
    data_root = Path(root) / "data" / "m-absa"
    def load(split):
        return [ex for d in sources for ex in load_domain_file(str(data_root / config.DOMAIN_FILES[d][split]), d)]
    return load("train"), load("dev"), sources


def load_test_split(root, mode, held_out_domain=None):
    if mode not in ("indomain", "crossdomain") or (mode == "crossdomain" and held_out_domain not in config.DOMAINS):
        raise ValueError("Invalid test split request")
    domains = config.DOMAINS if mode == "indomain" else [held_out_domain]
    return [ex for d in domains for ex in load_domain_file(str(Path(root) / "data" / "m-absa" / config.DOMAIN_FILES[d]["test"]), d)]


def training_vocabulary(examples, cfg):
    explicit, implicit = [], []
    for example in examples:
        e, n, _ = select_targets(example, cfg.drop_conflict)
        explicit.extend(e)
        implicit.extend(n)
    include_null = cfg.vocabulary_scope == "explicit-null" or (cfg.vocabulary_scope == "auto" and cfg.null_head)
    targets = explicit + implicit if include_null else explicit
    categories = sorted({t[1] for t in targets})
    sentiments = sorted({t[2] for t in targets})
    if not categories or not sentiments:
        raise ValueError("Eligible source training targets produced an empty vocabulary")
    counts = Counter(t[1] for t in targets)
    return categories, sentiments, dict(counts)


class ExperimentDataset(ABSADataset):
    def __init__(self, examples, tokenizer, categories, sentiments, cfg):
        explicit, implicit, self.audit = [], [], Counter()
        self.gold_spans = []
        for example in examples:
            retained, nulls, counts = select_targets(example, cfg.drop_conflict)
            self.audit.update(counts)
            explicit.append(Example(example.sentence, retained, example.domain))
            implicit.append(nulls)
            self.gold_spans.append([
                {"start": example.sentence.find(t[0]),
                 "end": example.sentence.find(t[0]) + len(t[0]), "triplet": list(t)}
                for t in retained
            ])
        super().__init__(explicit, tokenizer, categories, sentiments, cfg.max_len)
        self.null_gold = implicit
        self.num_sentiments = len(sentiments)
        self.num_null_labels = len(categories) * len(sentiments)
        self.null_targets = torch.zeros((len(examples), self.num_null_labels), dtype=torch.float32)
        for index, targets in enumerate(implicit):
            for _, category, sentiment in targets:
                if category in self.cat2id and sentiment in self.sent2id:
                    self.null_targets[index, self.cat2id[category] * self.num_sentiments + self.sent2id[sentiment]] = 1

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item.update(null_targets=self.null_targets[index], gold_null_triplets=self.null_gold[index], gold_spans=self.gold_spans[index])
        return item


def experiment_collate(batch):
    result = collate_fn(batch)
    result["null_targets"] = torch.stack([item["null_targets"] for item in batch])
    result["gold_null_triplets"] = [item["gold_null_triplets"] for item in batch]
    result["gold_spans"] = [item["gold_spans"] for item in batch]
    return result
