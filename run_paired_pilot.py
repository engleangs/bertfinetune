"""Run one matched, development-only tuning task per Slurm array index.

Examples:
    python run_paired_pilot.py --folds indomain restaurant --seeds 13 42 --dry-run
    python run_paired_pilot.py --folds indomain restaurant --seeds 13 42 --task-index 0 --device cuda
    python run_paired_pilot.py --folds indomain restaurant --seeds 13 42 --summarize

No held-out or in-domain test file is opened by this script. Each task writes
to its own directory so parallel Slurm array tasks never share a results CSV.
"""

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

import config as cfg
from src.artifacts import atomic_write_json
from src.data import (
    ABSADataset,
    build_crossdomain_train_dev,
    build_indomain_train_dev,
    build_label_vocab,
    label_frequencies,
)
from src.evaluate import identify_rare_labels
from src.train import train_one_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "paired_pilot"
PILOT_VERSION = "paired-pilot-v1"
CONFIGS = {item.name: item for item in (*cfg.EXPERIMENTS, *cfg.ABLATION_EXPERIMENTS)}
REQUIRED_CONFIGS = ("standard", "weighted", "category_weighted")
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class Recipe:
    lr: float
    epochs: int

    @property
    def tag(self):
        return f"lr_{self.lr:.8g}".replace(".", "p") + f"_epochs_{self.epochs}"


@dataclass(frozen=True)
class PilotTask:
    fold: str
    recipe: Recipe
    seed: int
    config_name: str

    def output_dir(self, root):
        return Path(root) / self.fold / self.recipe.tag / f"seed_{self.seed}" / self.config_name


def parse_recipe(value):
    try:
        lr_text, epochs_text = value.split(":", 1)
        recipe = Recipe(float(lr_text), int(epochs_text))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("recipe must have form learning-rate:epochs") from exc
    if not math.isfinite(recipe.lr) or recipe.lr <= 0 or recipe.epochs < 1:
        raise argparse.ArgumentTypeError("learning rate and epochs must be positive")
    return recipe


def build_plan(folds, recipes, seeds, configs=REQUIRED_CONFIGS):
    folds, recipes, seeds, configs = list(folds), list(recipes), list(seeds), list(configs)
    for name, values in (
        ("folds", folds), ("recipes", recipes), ("seeds", seeds), ("configs", configs),
    ):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be nonempty and unique")
    if set(folds) - {"indomain", *cfg.DOMAINS}:
        raise ValueError("unknown fold")
    if set(seeds) - set(cfg.SEEDS):
        raise ValueError("seeds must come from the existing study")
    if set(configs) - set(CONFIGS):
        raise ValueError("unknown loss configuration")
    shared_settings = None
    for config_name in configs:
        settings = {
            key: value for key, value in asdict(CONFIGS[config_name]).items()
            if key not in {"name", "loss_type"}
        }
        if shared_settings is None:
            shared_settings = settings
        elif settings != shared_settings:
            raise ValueError("loss configurations must share every non-loss setting")
    tags = [recipe.tag for recipe in recipes]
    if len(tags) != len(set(tags)):
        raise ValueError("recipe directory names must be unique")
    return [
        PilotTask(fold, recipe, seed, config_name)
        for fold in folds
        for recipe in recipes
        for seed in seeds
        for config_name in configs
    ]


def _train_dev(fold, data_dir):
    if fold == "indomain":
        return build_indomain_train_dev(str(data_dir))
    train_domains = [domain for domain in cfg.DOMAINS if domain != fold]
    return build_crossdomain_train_dev(
        str(data_dir), train_domains=train_domains, test_domain=fold,
    )


def run_task(task, data_dir, output_root, device):
    run_dir = task.output_dir(output_root)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Pilot output already exists: {run_dir}; choose a new output root")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    started_at = datetime.now(timezone.utc).isoformat()
    experiment = replace(
        CONFIGS[task.config_name],
        lr=task.recipe.lr,
        epochs=task.recipe.epochs,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
    )
    manifest = {
        "pilot_version": PILOT_VERSION,
        "status": "running",
        "started_at": started_at,
        "fold": task.fold,
        "seed": task.seed,
        "configuration": asdict(experiment),
        "data_dir": str(Path(data_dir).resolve()),
    }
    atomic_write_json(manifest_path, manifest)
    try:
        train_examples, dev_examples = _train_dev(task.fold, data_dir)
        category_vocab, sentiment_vocab = build_label_vocab(train_examples)
        if not train_examples or not dev_examples or not category_vocab or not sentiment_vocab:
            raise ValueError("Training or development data/vocabulary is empty")
        category_counts, _ = label_frequencies(train_examples)
        rare_categories = identify_rare_labels(
            category_counts, max_count=cfg.RARE_CATEGORY_MAX_COUNT,
        )
        tokenizer = AutoTokenizer.from_pretrained(experiment.model_name)
        if not getattr(tokenizer, "is_fast", True):
            raise ValueError("A fast tokenizer is required")
        train_ds = ABSADataset(
            train_examples, tokenizer, category_vocab, sentiment_vocab, experiment.max_len,
        )
        dev_ds = ABSADataset(
            dev_examples, tokenizer, category_vocab, sentiment_vocab, experiment.max_len,
        )
        result = train_one_config(
            experiment,
            train_ds,
            dev_ds,
            category_vocab,
            sentiment_vocab,
            seed=task.seed,
            device=device,
            history_path=run_dir / "history.json",
            rare_category_labels=rare_categories,
        )
        score = result["best_dev_metrics"]["complete_triplet"]["micro_f1"]
        summary = {
            "pilot_version": PILOT_VERSION,
            "fold": task.fold,
            "seed": task.seed,
            "config": task.config_name,
            "recipe": asdict(task.recipe),
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "selection_metric": result["selection_metric"],
            "best_dev_triplet_micro_f1": score,
            "best_epoch": result["best_epoch"],
            "train_size": len(train_examples),
            "dev_size": len(dev_examples),
            "training_time_sec": result["training_time_sec"],
            "device": result["device"],
        }
        atomic_write_json(run_dir / "dev_summary.json", summary)
        manifest.update({"status": "complete", "completed_at": datetime.now(timezone.utc).isoformat()})
        atomic_write_json(manifest_path, manifest)
        print(f"{task.fold}/{task.recipe.tag}/seed={task.seed}/{task.config_name}: dev triplet micro-F1={score:.4f}")
        return summary
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        atomic_write_json(manifest_path, manifest)
        raise


def summarize(plan, output_root):
    if set(task.config_name for task in plan) != set(REQUIRED_CONFIGS):
        raise ValueError("selection requires all three loss configurations")
    results = {}
    missing = []
    for task in plan:
        path = task.output_dir(output_root) / "dev_summary.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if (
            saved.get("pilot_version") != PILOT_VERSION
            or saved.get("fold") != task.fold
            or saved.get("seed") != task.seed
            or saved.get("config") != task.config_name
            or saved.get("recipe") != asdict(task.recipe)
            or saved.get("warmup_ratio") != WARMUP_RATIO
            or saved.get("max_grad_norm") != MAX_GRAD_NORM
            or saved.get("selection_metric") != "dev_complete_triplet_micro_f1"
        ):
            raise ValueError(f"Pilot summary does not match the plan: {path}")
        score = float(saved["best_dev_triplet_micro_f1"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid development F1 in {path}")
        results[task] = score
    if missing:
        raise RuntimeError(f"Pilot incomplete: {len(missing)} missing task(s); first: {missing[0]}")

    recipes = sorted({task.recipe for task in plan}, key=lambda recipe: (recipe.epochs, recipe.lr))
    scores = {}
    for recipe in recipes:
        score_rows = [results[task] for task in plan if task.recipe == recipe]
        scores[recipe] = statistics.mean(score_rows)
    selected = max(recipes, key=lambda recipe: (scores[recipe], -recipe.epochs, -recipe.lr))
    report = {
        "selection_rule": "highest equal-weight mean of development triplet micro-F1 across matched folds, seeds and all three losses; ties choose fewer epochs then lower learning rate",
        "selected_recipe": asdict(selected),
        "recipes": [
            {
                **asdict(recipe),
                "mean_dev_triplet_micro_f1": scores[recipe],
                "by_loss": {
                    config_name: statistics.mean(
                        results[task] for task in plan
                        if task.recipe == recipe and task.config_name == config_name
                    )
                    for config_name in REQUIRED_CONFIGS
                },
            }
            for recipe in recipes
        ],
        "task_count": len(plan),
    }
    atomic_write_json(Path(output_root) / "selection.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", default=["indomain", "restaurant"],
                        choices=["indomain", *cfg.DOMAINS])
    parser.add_argument("--recipes", nargs="+", type=parse_recipe,
                        default=[Recipe(2e-5, 5), Recipe(3e-5, 8)])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42])
    parser.add_argument("--configs", nargs="+", choices=REQUIRED_CONFIGS,
                        default=REQUIRED_CONFIGS)
    parser.add_argument("--data-dir", type=Path, default=Path(cfg.DATA_DIR))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--task-index", type=int)
    action.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.folds, args.recipes, args.seeds, args.configs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Paired development-only pilot: {len(plan)} tasks")
        for index, task in enumerate(plan):
            print(f"{index}: {task.fold} / {task.recipe.tag} / seed {task.seed} / {task.config_name}")
        return 0
    if args.summarize:
        print(summarize(plan, args.output_dir))
        return 0
    if not 0 <= args.task_index < len(plan):
        parser.error(f"--task-index must be between 0 and {len(plan) - 1}")
    run_task(plan[args.task_index], args.data_dir, args.output_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
