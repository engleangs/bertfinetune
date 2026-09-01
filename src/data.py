"""
S3 (Data): parsing + BIO tagging  plus the two split
builders the new design needs:

  build_indomain_split()    -> official train/dev/test, all 7 domains mixed
  build_crossdomain_split()  -> train+dev from TRAIN_DOMAINS, test = HOLD_OUT_DOMAIN

Both return the same (train_examples, dev_examples, test_examples) shape, so
everything downstream (ABSADataset, train_one_config, evaluate) doesn't care
which mode it's running under.
"""
import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config as cfg


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


def build_indomain_split(data_dir: str = cfg.DATA_DIR) -> Tuple[List[Example], List[Example], List[Example]]:
    """Original design: official per-domain files, ALL 7 domains, combined.
    This is the protected core comparison — unchanged data condition."""
    train, dev, test = [], [], []
    for domain in cfg.DOMAINS:
        files = cfg.DOMAIN_FILES[domain]
        train += load_domain_file(os.path.join(data_dir, files["train"]), domain)
        dev += load_domain_file(os.path.join(data_dir, files["dev"]), domain)
        test += load_domain_file(os.path.join(data_dir, files["test"]), domain)
    return train, dev, test


def build_crossdomain_split(
    data_dir: str = cfg.DATA_DIR,
    train_domains: Optional[List[str]] = None,
    test_domain: Optional[str] = None,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Pool source-domain train/dev and hold one entire domain out.

    The current protocol combines the held-out domain's train, dev, and test
    files for evaluation. Freeze this choice before running cross-domain work;
    use the official test file only if that is the pre-registered decision.
    """
    train_domains = train_domains or cfg.TRAIN_DOMAINS
    test_domain = test_domain or cfg.HOLD_OUT_DOMAIN
    assert test_domain not in train_domains, (
        f"{test_domain} must NOT be in train_domains — that would leak the held-out "
        f"domain into training and silently invalidate the whole experiment."
    )

    train, dev = [], []
    for domain in train_domains:
        files = cfg.DOMAIN_FILES[domain]
        train += load_domain_file(os.path.join(data_dir, files["train"]), domain)
        dev += load_domain_file(os.path.join(data_dir, files["dev"]), domain)

    test_files = cfg.DOMAIN_FILES[test_domain]
    # This scaffold currently pools held-out train+dev+test for evaluation.
    # Keep cross-domain runs disabled until the team freezes this choice and
    # the policy for categories that do not occur in the source domains.
    test = (
        load_domain_file(os.path.join(data_dir, test_files["train"]), test_domain)
        + load_domain_file(os.path.join(data_dir, test_files["dev"]), test_domain)
        + load_domain_file(os.path.join(data_dir, test_files["test"]), test_domain)
    )
    return train, dev, test


def build_label_vocab(examples: List[Example]):
    triplets = [triplet for ex in examples for triplet in aligned_explicit_triplets(ex)]
    categories = sorted({triplet[1] for triplet in triplets})
    sentiments = sorted({triplet[2] for triplet in triplets})
    return categories, sentiments


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


def summarize_task_scope(
    examples: Sequence[Example],
    category_vocab: Optional[Sequence[str]] = None,
    sentiment_vocab: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Summarize retained and unsupported annotations for run metadata."""
    category_set = set(category_vocab) if category_vocab is not None else None
    sentiment_set = set(sentiment_vocab) if sentiment_vocab is not None else None
    total = implicit = unaligned = included = duplicate = 0
    additional_label_pair = overlapping = 0
    unseen_categories = set()
    unseen_sentiments = set()

    for example in examples:
        selected, counts = _select_baseline_triplets(example)
        total += counts["total"]
        implicit += counts["implicit"]
        unaligned += counts["unaligned"]
        duplicate += counts["duplicate"]
        additional_label_pair += counts["additional_label_pair"]
        overlapping += counts["overlapping"]
        included += len(selected)
        for _, category, sentiment in selected:
            if category_set is not None and category not in category_set:
                unseen_categories.add(category)
            if sentiment_set is not None and sentiment not in sentiment_set:
                unseen_sentiments.add(sentiment)

    return {
        "examples": len(examples),
        "total_triplets": total,
        "included_explicit_triplets": included,
        "implicit_triplets_excluded": implicit,
        "unaligned_triplets_excluded": unaligned,
        "duplicate_triplets_excluded": duplicate,
        "additional_label_pairs_excluded": additional_label_pair,
        "overlapping_triplets_excluded": overlapping,
        "unseen_categories": sorted(unseen_categories),
        "unseen_sentiments": sorted(unseen_sentiments),
    }


class ABSADataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer: AutoTokenizer,
                 category_vocab: List[str], sentiment_vocab: List[str], max_len: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.cat2id = {c: i for i, c in enumerate(category_vocab)}
        self.sent2id = {s: i for i, s in enumerate(sentiment_vocab)}
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.sentence, truncation=True, max_length=self.max_len,
            padding="max_length", return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        bio = [0] * len(offsets)
        span_boundaries, cat_labels, sent_labels = [], [], []

        gold_triplets = aligned_explicit_triplets(ex)
        for aspect_term, category, sentiment in gold_triplets:
            char_start, char_end = find_span(ex.sentence, aspect_term)
            if char_start == -1:
                continue  # NULL/implicit aspect — not handled by the core BIO head

            tok_start, tok_end = None, None
            for i, (s, e) in enumerate(offsets):
                if s == e:
                    continue
                if s <= char_start < e:
                    tok_start = i
                if s < char_end <= e:
                    tok_end = i
            if tok_start is None or tok_end is None:
                continue

            bio[tok_start] = 1
            for i in range(tok_start + 1, tok_end + 1):
                bio[i] = 2

            span_boundaries.append((tok_start, tok_end))
            cat_labels.append(self.cat2id.get(category, -100))
            sent_labels.append(self.sent2id.get(sentiment, -100))

        bio = [b if (s != e) else -100 for b, (s, e) in zip(bio, offsets)]

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["bio_labels"] = torch.tensor(bio)
        item["span_boundaries"] = span_boundaries
        item["category_labels"] = torch.tensor(cat_labels, dtype=torch.long)
        item["sentiment_labels"] = torch.tensor(sent_labels, dtype=torch.long)
        item["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
        item["example_id"] = idx
        item["sentence"] = ex.sentence
        item["gold_triplets"] = gold_triplets
        item["domain"] = ex.domain
        return item


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "bio_labels": torch.stack([b["bio_labels"] for b in batch]),
        "span_boundaries": [b["span_boundaries"] for b in batch],
        "category_labels": [b["category_labels"] for b in batch],
        "sentiment_labels": [b["sentiment_labels"] for b in batch],
        "offset_mapping": torch.stack([b["offset_mapping"] for b in batch]),
        "example_id": [b["example_id"] for b in batch],
        "sentence": [b["sentence"] for b in batch],
        "gold_triplets": [b["gold_triplets"] for b in batch],
        "domain": [b["domain"] for b in batch],
    }
