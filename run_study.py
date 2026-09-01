"""
Once training + eval run, for one mode/config/seed
combination. run_all_study.py calls this in a loop for the full grid.

Usage:
    python run_study.py --mode indomain --config standard --seed 42
"""
import argparse
import csv
import os

import config as cfg
from src.data import (
    build_indomain_split, build_crossdomain_split, build_label_vocab,
    label_frequencies, ABSADataset,
)
from src.train import train_one_config
from transformers import AutoTokenizer

RESULTS_CSV = "results.csv"
RESULTS_FIELDS = ["mode", "config", "seed", "trainable_params", "training_time_sec"]


def append_result(row: dict):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run(mode: str, config_name: str, seed: int, device: str = "auto"):
    experiment_cfg = next(c for c in cfg.EXPERIMENTS if c.name == config_name)

    if mode == "indomain":
        train_examples, dev_examples, test_examples = build_indomain_split()
    elif mode == "crossdomain":
        train_examples, dev_examples, test_examples = build_crossdomain_split()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    category_vocab, sentiment_vocab = build_label_vocab(train_examples)
    cat_counts, sent_counts = label_frequencies(train_examples)  # TODO (S5): feed into rare_label_recall()

    tokenizer = AutoTokenizer.from_pretrained(experiment_cfg.model_name)
    train_ds = ABSADataset(train_examples, tokenizer, category_vocab, sentiment_vocab, experiment_cfg.max_len)
    dev_ds = ABSADataset(dev_examples, tokenizer, category_vocab, sentiment_vocab, experiment_cfg.max_len)

    result = train_one_config(experiment_cfg, train_ds, dev_ds, category_vocab, sentiment_vocab,
                               seed=seed, device=device)

    # TODO (S5): run real inference over test_examples here — decode BIO
    # predictions, map to spans, run category/sentiment heads on PREDICTED
    # (not gold) spans, then score with evaluate.complete_triplet_scores(),
    # evaluate.precision_recall_by_label(), and evaluate.rare_label_recall()
    # using identify_rare_labels(cat_counts) as the rare-label set. Append
    # those metrics to the row below once wired up.
    print(
        f"[{mode}/{config_name}/seed={seed}] trained on {result['device']} "
        f"in {result['training_time_sec']:.1f}s"
    )

    append_result({
        "mode": mode, "config": config_name, "seed": seed,
        "trainable_params": result["trainable_params"],
        "training_time_sec": result["training_time_sec"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=cfg.MODES)
    parser.add_argument("--config", required=True, choices=[c.name for c in cfg.EXPERIMENTS])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device (default: auto; selects CUDA, MPS, or CPU)",
    )
    args = parser.parse_args()
    run(args.mode, args.config, args.seed, args.device)
