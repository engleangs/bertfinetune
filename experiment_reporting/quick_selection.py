"""Validate the development freeze before evaluation or reporting."""

from dataclasses import asdict
import json
from pathlib import Path

from src.experiment_config import TrialConfig


def trial_from_dict(settings):
    settings = dict(settings)
    for key in ("loss_heads", "null_thresholds"):
        settings[key] = tuple(settings[key])
    trial = TrialConfig(**settings)
    trial.validate()
    return trial


def check_frozen_variants(frozen):
    seeds = frozen["seeds"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen seeds must be nonempty and unique")
    seen = set()
    for variant in frozen["variants"]:
        cfg = trial_from_dict(variant["config"])
        if cfg.trial_id != variant["trial_id"] or cfg.trial_id in seen:
            raise ValueError("Frozen trial identity is inconsistent or repeated")
        seen.add(cfg.trial_id)
        run_seeds = [run["seed"] for run in variant["runs"]]
        if sorted(run_seeds) != sorted(seeds):
            raise ValueError("Every frozen variant must have exactly the declared seeds")


def check_frozen_run(variant, run):
    directory = Path(run["directory"])
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cfg = trial_from_dict(variant["config"])
    if (directory.parent.name != cfg.trial_id or directory.name != f"seed_{run['seed']}"
            or manifest["status"] not in ("dev_complete", "complete")
            or manifest["fingerprint"] != run["fingerprint"]
            or manifest["checkpoint_sha256"] != run["checkpoint_sha256"]
            or manifest["training"]["best_epoch"] != run["best_epoch"]
            or manifest["identity"]["seed"] != run["seed"]
            or json.dumps(manifest["identity"]["config"], sort_keys=True)
            != json.dumps(asdict(cfg), sort_keys=True)):
        raise ValueError(f"Frozen run changed or is incomplete: {directory}")
    metrics = json.loads((directory / "dev_metrics.json").read_text(encoding="utf-8"))
    if metrics["null_threshold"] != run["null_threshold"]:
        raise ValueError(f"Frozen NULL threshold changed: {directory}")
    return manifest, metrics
