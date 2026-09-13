"""Generate a standalone audit report for the English M-ABSA dataset.

Examples:
    python3 data_audit.py
    python3 data_audit.py --mode crossdomain --output artifacts/crossdomain_audit.json
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer

import config as cfg
from src.artifacts import atomic_write_json
from src.data import build_label_vocab, load_domain_file
from src.dataset_audit import aggregate_audits, audit_split, split_overlap_summary


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "data_audit.json"


def load_split(data_dir: Path, domain: str, split: str):
    """Load one configured domain/split and fail with an actionable message."""
    path = data_dir / cfg.DOMAIN_FILES[domain][split]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Run `python3 download_data.py` first."
        )
    return load_domain_file(str(path), domain)


def print_split_report(result):
    """Print the most decision-relevant counts for one domain/split."""
    print(f"\n{'=' * 70}\n{result['domain'].upper()} — {result['split'].upper()}\n{'=' * 70}")
    labels = (
        ("Examples", "examples"),
        ("Total triplets", "total_triplets"),
        ("Included explicit triplets", "included_explicit_triplets"),
        ("Implicit / NULL excluded", "implicit_triplets_excluded"),
        ("Unaligned excluded", "unaligned_triplets_excluded"),
        ("Duplicate triplets excluded", "duplicate_triplets_excluded"),
        ("Additional label pairs excluded", "additional_label_pairs_excluded"),
        ("Overlapping triplets excluded", "overlapping_triplets_excluded"),
        ("Repeated annotation examples", "repeated_annotated_aspect_text_examples"),
        ("Ambiguous repeated-occurrence examples", "ambiguous_repeated_occurrence_examples"),
        ("Token alignment / truncation misses", "token_alignment_or_truncation_misses"),
        ("Unknown category targets", "unknown_category_targets"),
        ("Unknown sentiment targets", "unknown_sentiment_targets"),
    )
    for label, key in labels:
        print(f"{label}: {result[key]}")
    if result["unseen_categories"]:
        print(f"Unseen categories: {result['unseen_categories']}")
    if result["unseen_sentiments"]:
        print(f"Unseen sentiments: {result['unseen_sentiments']}")


def build_report(data_dir: Path, tokenizer, mode: str, max_len: int):
    """Load all 21 files and return one deterministic audit payload."""
    loaded = {
        domain: {
            split: load_split(data_dir, domain, split)
            for split in ("train", "dev", "test")
        }
        for domain in cfg.DOMAINS
    }

    vocab_domains = cfg.DOMAINS if mode == "indomain" else cfg.TRAIN_DOMAINS
    vocabulary_examples = [
        example
        for domain in vocab_domains
        for example in loaded[domain]["train"]
    ]
    category_vocab, sentiment_vocab = build_label_vocab(vocabulary_examples)
    if not category_vocab or not sentiment_vocab:
        raise ValueError("Training data produced an empty label vocabulary")

    rows = []
    for domain in cfg.DOMAINS:
        for split in ("train", "dev", "test"):
            row = audit_split(
                domain,
                split,
                loaded[domain][split],
                tokenizer,
                category_vocab,
                sentiment_vocab,
                max_len,
            )
            rows.append(row)
            print_split_report(row)

    all_train = [example for domain in cfg.DOMAINS for example in loaded[domain]["train"]]
    all_dev = [example for domain in cfg.DOMAINS for example in loaded[domain]["dev"]]
    all_test = [example for domain in cfg.DOMAINS for example in loaded[domain]["test"]]
    return {
        "schema_version": 1,
        "dataset": "M-ABSA English",
        "mode": mode,
        "data_dir": str(data_dir),
        "model_name": getattr(tokenizer, "name_or_path", cfg.MODEL_NAME),
        "max_len": max_len,
        "vocabulary_domains": list(vocab_domains),
        "category_vocab": category_vocab,
        "sentiment_vocab": sentiment_vocab,
        "overall": aggregate_audits(rows),
        "split_overlap": split_overlap_summary(all_train, all_dev, all_test),
        "by_domain_split": rows,
        "counting_note": (
            "Diagnostic exclusion/ambiguity counts may overlap and must not be "
            "assumed to partition total_triplets unless stated otherwise."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(cfg.DATA_DIR))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=cfg.MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=cfg.MAX_LEN)
    parser.add_argument("--mode", choices=("indomain", "crossdomain"), default="indomain")
    args = parser.parse_args()
    if args.max_len < 2:
        parser.error("--max-len must be at least 2")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("A fast tokenizer is required for offset-based auditing")

    report = build_report(args.data_dir.resolve(), tokenizer, args.mode, args.max_len)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    atomic_write_json(output, report)
    print(f"\n{'=' * 70}\nDATA AUDIT COMPLETE\n{'=' * 70}")
    print(f"Audit report saved to: {output}")


if __name__ == "__main__":
    main()
