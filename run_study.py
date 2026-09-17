"""Run one train/validation/test experiment and persist all artifacts.

Example:
    python run_study.py --mode indomain --config standard --seed 42 --device cuda
"""

import argparse
import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

import config as cfg
from src.artifacts import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    upsert_csv_row,
)
from src.data import (
    ABSADataset,
    aligned_explicit_triplets,
    build_crossdomain_split,
    build_indomain_split,
    build_label_vocab,
    label_frequencies,
    summarize_task_scope,
    summarize_tokenized_targets,
)
from src.evaluate import identify_rare_labels
from src.losses import get_evaluation_loss_fns
from src.train import evaluate_model, train_one_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results.csv"
RESULTS_FIELDS = [
    "schema_version",
    "mode",
    "config",
    "seed",
    "status",
    "task_scope",
    "best_epoch",
    "selection_metric",
    "best_dev_micro_f1",
    "dev_loss",
    "dev_micro_precision",
    "dev_micro_recall",
    "dev_micro_f1",
    "dev_macro_f1",
    "dev_rare_category_recall",
    "test_loss",
    "test_micro_precision",
    "test_micro_recall",
    "test_micro_f1",
    "test_macro_f1",
    "test_rare_category_recall",
    "train_size",
    "dev_size",
    "test_size",
    "trainable_params",
    "training_time_sec",
    "device",
    "artifact_dir",
    "completed_at",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_relative_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _split_examples(mode: str):
    if mode == "indomain":
        return build_indomain_split()
    if mode == "crossdomain":
        return build_crossdomain_split()
    raise ValueError(f"Unknown mode: {mode}")


def _completed_result_row(results_csv: Path, mode, config_name, seed):
    if not results_csv.exists():
        return None
    with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row.get("mode") == mode
                and row.get("config") == config_name
                and str(row.get("seed")) == str(seed)
                and row.get("status") == "complete"
            ):
                return row
    return None


def _split_overlap_summary(train_examples, dev_examples, test_examples):
    def sentence_set(examples):
        return {example.sentence for example in examples}

    def labeled_set(examples):
        return {
            (example.sentence, tuple(aligned_explicit_triplets(example)))
            for example in examples
        }

    train_sentences = sentence_set(train_examples)
    dev_sentences = sentence_set(dev_examples)
    test_sentences = sentence_set(test_examples)
    train_labeled = labeled_set(train_examples)
    dev_labeled = labeled_set(dev_examples)
    test_labeled = labeled_set(test_examples)
    return {
        "train_dev_repeated_sentences": len(train_sentences & dev_sentences),
        "train_test_repeated_sentences": len(train_sentences & test_sentences),
        "dev_test_repeated_sentences": len(dev_sentences & test_sentences),
        "train_dev_identical_labeled_examples": len(train_labeled & dev_labeled),
        "train_test_identical_labeled_examples": len(train_labeled & test_labeled),
        "dev_test_identical_labeled_examples": len(dev_labeled & test_labeled),
    }


def _summary_row(
    mode,
    config_name,
    seed,
    result,
    dev_metrics,
    test_metrics,
    split_sizes,
    run_dir,
    completed_at,
):
    dev_triplet = dev_metrics["complete_triplet"]
    test_triplet = test_metrics["complete_triplet"]
    return {
        "schema_version": 2,
        "mode": mode,
        "config": config_name,
        "seed": seed,
        "status": "complete",
        "task_scope": "single-pair-aligned-explicit-aspect",
        "best_epoch": result["best_epoch"],
        "selection_metric": result["selection_metric"],
        "best_dev_micro_f1": result["best_dev_metrics"]["complete_triplet"]["micro_f1"],
        "dev_loss": dev_metrics["loss"],
        "dev_micro_precision": dev_triplet["micro_precision"],
        "dev_micro_recall": dev_triplet["micro_recall"],
        "dev_micro_f1": dev_triplet["micro_f1"],
        "dev_macro_f1": dev_triplet["macro_f1"],
        "dev_rare_category_recall": dev_metrics["rare_category_recall"],
        "test_loss": test_metrics["loss"],
        "test_micro_precision": test_triplet["micro_precision"],
        "test_micro_recall": test_triplet["micro_recall"],
        "test_micro_f1": test_triplet["micro_f1"],
        "test_macro_f1": test_triplet["macro_f1"],
        "test_rare_category_recall": test_metrics["rare_category_recall"],
        "train_size": split_sizes["train"],
        "dev_size": split_sizes["dev"],
        "test_size": split_sizes["test"],
        "trainable_params": result["trainable_params"],
        "training_time_sec": result["training_time_sec"],
        "device": result["device"],
        "artifact_dir": str(run_dir),
        "completed_at": completed_at,
    }


def run(
    mode: str,
    config_name: str,
    seed: int,
    device: str = "auto",
    output_dir=DEFAULT_OUTPUT_ROOT,
    results_csv=DEFAULT_RESULTS_CSV,
    overwrite: bool = False,
):
    """Train, select on dev, test once, and store a reproducible run."""
    experiment_cfg = next(
        (item for item in cfg.EXPERIMENTS if item.name == config_name), None,
    )
    if experiment_cfg is None:
        raise ValueError(f"Unknown config: {config_name}")

    output_root = _project_relative_path(output_dir)
    results_csv = _project_relative_path(results_csv)
    run_dir = output_root / mode / config_name / f"seed_{seed}"
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.json"

    manifest_on_disk = None
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            manifest_on_disk = json.load(file)

    if manifest_on_disk and manifest_on_disk.get("status") == "complete" and not overwrite:
        required_artifacts = [
            metrics_path,
            checkpoint_path,
            history_path,
            run_dir / "dev_predictions.jsonl",
            run_dir / "test_predictions.jsonl",
            run_dir / "tokenizer",
        ]
        missing_artifacts = [
            str(path) for path in required_artifacts if not path.exists()
        ]
        if missing_artifacts:
            raise RuntimeError(
                f"Completed manifest has missing artifacts: {missing_artifacts}"
            )
        with metrics_path.open(encoding="utf-8") as file:
            saved = json.load(file)
        summary = saved.get("summary")
        if not summary:
            raise RuntimeError(f"Completed metrics file has no summary: {metrics_path}")
        upsert_csv_row(
            results_csv,
            summary,
            RESULTS_FIELDS,
            key_fields=("mode", "config", "seed"),
        )
        print(f"Skipping completed run at {run_dir}; use --overwrite to rerun.")
        return summary

    indexed_result = _completed_result_row(
        results_csv, mode, config_name, seed,
    )
    if indexed_result is not None and not overwrite:
        raise RuntimeError(
            "results.csv already contains this completed run key at "
            f"{indexed_result.get('artifact_dir')!r}. Use its artifacts or "
            "pass --overwrite explicitly."
        )
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"Incomplete artifacts already exist at {run_dir}. "
            "Inspect them or rerun with --overwrite."
        )
    if overwrite and metrics_path.exists():
        # A failed overwrite must not leave an older metrics payload in place.
        metrics_path.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    manifest = {
        "schema_version": 2,
        "status": "running",
        "mode": mode,
        "config_name": config_name,
        "seed": seed,
        "task_scope": "single-pair-aligned-explicit-aspect",
        "started_at": started_at,
        "configuration": asdict(experiment_cfg),
    }
    atomic_write_json(manifest_path, manifest)

    try:
        train_examples, dev_examples, test_examples = _split_examples(mode)
        category_vocab, sentiment_vocab = build_label_vocab(train_examples)
        if not category_vocab or not sentiment_vocab:
            raise ValueError("Training split produced an empty label vocabulary")

        category_counts, _ = label_frequencies(train_examples)
        rare_categories = identify_rare_labels(category_counts,max_count=cfg.RARE_CATEGORY_MAX_COUNT)
        data_summary = {
            "train": summarize_task_scope(
                train_examples, category_vocab, sentiment_vocab,
            ),
            "dev": summarize_task_scope(
                dev_examples, category_vocab, sentiment_vocab,
            ),
            "test": summarize_task_scope(
                test_examples, category_vocab, sentiment_vocab,
            ),
            "split_overlap": _split_overlap_summary(
                train_examples, dev_examples, test_examples,
            ),
        }

        tokenizer = AutoTokenizer.from_pretrained(experiment_cfg.model_name)
        if not getattr(tokenizer, "is_fast", True):
            raise ValueError("A fast tokenizer is required for offset-based inference")

        train_ds = ABSADataset(
            train_examples, tokenizer, category_vocab, sentiment_vocab,
            experiment_cfg.max_len,
        )
        dev_ds = ABSADataset(
            dev_examples, tokenizer, category_vocab, sentiment_vocab,
            experiment_cfg.max_len,
        )
        test_ds = ABSADataset(
            test_examples, tokenizer, category_vocab, sentiment_vocab,
            experiment_cfg.max_len,
        )
        data_summary["train"]["tokenized_targets"] = summarize_tokenized_targets(
            train_ds,
        )
        data_summary["dev"]["tokenized_targets"] = summarize_tokenized_targets(
            dev_ds,
        )
        data_summary["test"]["tokenized_targets"] = summarize_tokenized_targets(
            test_ds,
        )

        result = train_one_config(
            experiment_cfg,
            train_ds,
            dev_ds,
            category_vocab,
            sentiment_vocab,
            seed=seed,
            device=device,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            rare_category_labels=rare_categories,
        )
        resolved_device = torch.device(result["device"])
        evaluation_loss_fns = tuple(
            loss_fn.to(resolved_device) for loss_fn in get_evaluation_loss_fns()
        )

        dev_metrics, dev_predictions = evaluate_model(
            result["model"],
            dev_ds,
            category_vocab,
            sentiment_vocab,
            experiment_cfg,
            resolved_device,
            rare_category_labels=rare_categories,
            loss_fns=evaluation_loss_fns,
            description="selected dev",
        )
        test_metrics, test_predictions = evaluate_model(
            result["model"],
            test_ds,
            category_vocab,
            sentiment_vocab,
            experiment_cfg,
            resolved_device,
            rare_category_labels=rare_categories,
            loss_fns=evaluation_loss_fns,
            description="test",
        )

        completed_at = _utc_now()
        split_sizes = {
            "train": len(train_examples),
            "dev": len(dev_examples),
            "test": len(test_examples),
        }
        summary = _summary_row(
            mode,
            config_name,
            seed,
            result,
            dev_metrics,
            test_metrics,
            split_sizes,
            run_dir,
            completed_at,
        )
        metrics_payload = {
            "schema_version": 2,
            "task_scope": "single-pair-aligned-explicit-aspect",
            "selection": {
                "metric": result["selection_metric"],
                "best_epoch": result["best_epoch"],
                "best_epoch_dev": result["best_dev_metrics"],
            },
            "dev": dev_metrics,
            "test": test_metrics,
            "summary": summary,
        }

        atomic_write_jsonl(run_dir / "dev_predictions.jsonl", dev_predictions)
        atomic_write_jsonl(run_dir / "test_predictions.jsonl", test_predictions)
        atomic_write_json(metrics_path, metrics_payload)
        atomic_torch_save(
            checkpoint_path,
            {
                "schema_version": 2,
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in result["model"].state_dict().items()
                },
                "model_name": experiment_cfg.model_name,
                "mode": mode,
                "config": asdict(experiment_cfg),
                "seed": seed,
                "task_scope": "single-pair-aligned-explicit-aspect",
                "category_vocab": category_vocab,
                "sentiment_vocab": sentiment_vocab,
                "best_epoch": result["best_epoch"],
                "selection_metric": result["selection_metric"],
                "selection_score": result["best_dev_metrics"]["complete_triplet"]["micro_f1"],
            },
        )
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(run_dir / "tokenizer")

        manifest.update(
            {
                "status": "complete",
                "completed_at": completed_at,
                "device": result["device"],
                "best_epoch": result["best_epoch"],
                "selection_metric": result["selection_metric"],
                "category_vocab": category_vocab,
                "sentiment_vocab": sentiment_vocab,
                "rare_category_labels": rare_categories,
                "data_summary": data_summary,
                "split_sizes": split_sizes,
                "versions": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "artifacts": {
                    "checkpoint": "best_model.pt",
                    "history": "history.json",
                    "metrics": "metrics.json",
                    "dev_predictions": "dev_predictions.jsonl",
                    "test_predictions": "test_predictions.jsonl",
                    "tokenizer": "tokenizer/",
                },
            }
        )
        replaced = upsert_csv_row(
            results_csv,
            summary,
            RESULTS_FIELDS,
            key_fields=("mode", "config", "seed"),
        )
        # The completed manifest is the final commit marker for the run.
        atomic_write_json(manifest_path, manifest)

        print(
            f"[{mode}/{config_name}/seed={seed}] best_epoch={result['best_epoch']} "
            f"dev_f1={summary['dev_micro_f1']:.4f} "
            f"test_f1={summary['test_micro_f1']:.4f} "
            f"device={result['device']}"
        )
        print(f"Artifacts: {run_dir}")
        if replaced:
            print(f"Replaced {replaced} previous results.csv row(s) for this run key.")
        return summary
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=cfg.MODES)
    parser.add_argument(
        "--config", required=True, choices=[item.name for item in cfg.EXPERIMENTS],
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--device", default="auto",
        help="Training device (default: auto; selects CUDA, MPS, or CPU)",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_ROOT),
        help="root directory for per-run artifacts",
    )
    parser.add_argument(
        "--results-csv", default=str(DEFAULT_RESULTS_CSV),
        help="flat summary CSV updated after a successful run",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="rerun and replace artifacts for an existing run key",
    )
    args = parser.parse_args()
    run(
        args.mode,
        args.config,
        args.seed,
        args.device,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        overwrite=args.overwrite,
    )
