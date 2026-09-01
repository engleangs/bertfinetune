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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config as cfg


@dataclass
class Example:
    sentence: str
    triplets: List[Tuple[str, str, str]]  # (aspect_term, category, sentiment)
    domain: str = "unknown"


def parse_line(line: str, domain: str = "unknown") -> Example:
    sentence, raw_triplets = line.strip().split("####")
    triplets = ast.literal_eval(raw_triplets)
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
    """New design: train+dev pooled from train_domains (default: all except
    the held-out one), test = test_domain's OWN test file, entirely unseen
    during training. This is the generalization test the professor asked for."""
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
    # Use the held-out domain's train+dev+test ALL as the test set — during a
    # cross-domain test the model has never seen this domain at all, so there's
    # no reason to withhold part of it; using all of it gives a bigger, less
    # noisy test-set estimate. (If your team prefers a stricter design that only
    # uses the domain's own test file, swap the line below for just test_files["test"].)
    test = (
        load_domain_file(os.path.join(data_dir, test_files["train"]), test_domain)
        + load_domain_file(os.path.join(data_dir, test_files["dev"]), test_domain)
        + load_domain_file(os.path.join(data_dir, test_files["test"]), test_domain)
    )
    return train, dev, test


def build_label_vocab(examples: List[Example]):
    categories = sorted({t[1] for ex in examples for t in ex.triplets})
    sentiments = sorted({t[2] for ex in examples for t in ex.triplets})
    return categories, sentiments


def label_frequencies(examples: List[Example]) -> Tuple[dict, dict]:
    """Counts per category / sentiment label in the TRAINING set — this is what
    'rare label' means downstream (see evaluate.identify_rare_labels)."""
    cat_counts, sent_counts = {}, {}
    for ex in examples:
        for _, category, sentiment in ex.triplets:
            cat_counts[category] = cat_counts.get(category, 0) + 1
            sent_counts[sentiment] = sent_counts.get(sentiment, 0) + 1
    return cat_counts, sent_counts


def find_span(sentence: str, aspect_term: str) -> Tuple[int, int]:
    idx = sentence.find(aspect_term)
    if idx == -1:
        return -1, -1
    return idx, idx + len(aspect_term)


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

        for aspect_term, category, sentiment in ex.triplets:
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
            cat_labels.append(self.cat2id.get(category, -1))
            sent_labels.append(self.sent2id.get(sentiment, -1))

        bio = [b if (s != e) else -100 for b, (s, e) in zip(bio, offsets)]

        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["bio_labels"] = torch.tensor(bio)
        item["span_boundaries"] = span_boundaries
        item["category_labels"] = torch.tensor(cat_labels, dtype=torch.long)
        item["sentiment_labels"] = torch.tensor(sent_labels, dtype=torch.long)
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
        "domain": [b["domain"] for b in batch],
    }
