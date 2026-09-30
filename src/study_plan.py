"""Frozen, dependency-light plan for the cleaned-data study."""

from dataclasses import asdict, dataclass
from itertools import zip_longest
import math
import statistics

import config


VERSION = "clean-study-v1"
FOLDS = ("indomain", *config.DOMAINS)
LOSSES = ("standard", "weighted")
SEEDS = (13, 42, 123, 2024, 777)
SEARCH_SEEDS = SEEDS[:2]
LEARNING_RATES = (2e-5, 3e-5, 5e-5)


@dataclass(frozen=True)
class Task:
    stage: str
    data_version: str
    fold: str
    loss: str
    seed: int
    lr: float
    epochs: int
    warmup_ratio: float
    max_grad_norm: float | None

    @property
    def key(self):
        rate = f"{self.lr:.0e}".replace("-", "m").replace("+", "p")
        return f"{self.stage}/{self.data_version}/{self.fold}/{rate}/{self.loss}/seed_{self.seed}"

    def payload(self):
        return {**asdict(self), "key": self.key}


def baseline_tasks():
    return [Task("baseline", data, "indomain", loss, seed, 2e-5, 5, 0.0, None)
            for data in ("raw", "clean") for loss in LOSSES for seed in SEEDS]


def search_tasks():
    return [Task("search", "clean", fold, loss, seed, lr, 30, 0.1, 1.0)
            for fold in FOLDS for lr in LEARNING_RATES
            for loss in LOSSES for seed in SEARCH_SEEDS]


def smoke_tasks():
    return [Task("smoke", "clean", "indomain", loss, 13, 3e-5, 1, 0.1, 1.0)
            for loss in LOSSES]


def main_tasks():
    # A single array enforces a global concurrency cap for A and B together.
    return [task for pair in zip_longest(baseline_tasks(), search_tasks())
            for task in pair if task is not None]


def choose_recipes(summaries):
    """Choose within each fold; never pool held-out-domain information."""
    expected = {task.key: task for task in search_tasks()}
    by_key = {}
    for row in summaries:
        key = row["task"]["key"]
        if key in by_key or key not in expected:
            raise ValueError(f"Duplicate or unexpected search run: {key}")
        if row["task"] != expected[key].payload() or row.get("status") != "trained":
            raise ValueError(f"Search run does not match frozen plan: {key}")
        value = row["best_dev_micro_f1"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid development score: {key}")
        by_key[key] = row
    if set(by_key) != set(expected):
        raise ValueError(f"Search incomplete: missing {len(set(expected) - set(by_key))} runs")
    folds = {}
    for fold in FOLDS:
        candidates = []
        for lr in LEARNING_RATES:
            rows = [by_key[t.key] for t in expected.values() if t.fold == fold and t.lr == lr]
            candidates.append({"lr": lr, "mean_dev_micro_f1": statistics.mean(
                row["best_dev_micro_f1"] for row in rows), "run_keys": [row["task"]["key"] for row in rows]})
        best = max(candidates, key=lambda row: (row["mean_dev_micro_f1"], -row["lr"]))
        folds[fold] = {"lr": best["lr"], "selected_search_keys": best["run_keys"], "candidates": candidates}
    return {"version": VERSION, "rule": "per-fold mean development exact-triplet micro-F1 over two losses and seeds 13/42; ties use lower learning rate", "folds": folds}


def selected_search_tasks(selection):
    return [t for t in search_tasks() if t.lr == selection["folds"][t.fold]["lr"]]


def final_tasks(selection):
    return [Task("final", "clean", fold, loss, seed, selection["folds"][fold]["lr"], 30, 0.1, 1.0)
            for fold in FOLDS for loss in LOSSES for seed in SEEDS[2:]]


def completion_tasks(selection):
    # Search entries below are evaluation-only; they are never retrained.
    return [*selected_search_tasks(selection), *final_tasks(selection)]


def plan_payload():
    return {"version": VERSION, "baseline": [t.payload() for t in baseline_tasks()],
            "search": [t.payload() for t in search_tasks()],
            "main_array": [t.key for t in main_tasks()],
            "counts": {"baseline_training": 20, "search_training": 96,
                       "additional_final_training": 48, "reused_search_checkpoints": 32,
                       "total_training_excluding_smoke": 164, "final_results": 80}}
