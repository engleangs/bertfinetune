"""Plan a staged development search; execute only with --execute.

Uses the existing version-2 trainer and its provenance checks. No training
code, environment, model architecture, or completed run is changed.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

import config
from src.experiment_config import TrialConfig, candidate_recipes


ROOT = Path(__file__).resolve().parent
SCREEN = ("standard", "weighted", "mixed_025", "mixed_033", "mixed_050", "focal_2")
RECIPES = {recipe["name"]: recipe for recipe in candidate_recipes("all")}
EXPLICIT_RECIPES = tuple(name for name in RECIPES if not name.startswith("standard_null"))


@dataclass
class Job:
    trial: TrialConfig
    seed: int
    directory: Path
    command: list[str]


def pinned_revision():
    """Read the runner's pin without importing torch or loading a model."""
    tree = ast.parse((ROOT / "run_experiments.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BERT_REVISION" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("The version-2 runner is missing its pinned model revision")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step", choices=("screen", "lr", "confirm", "null"), default="screen")
    p.add_argument("--winner", choices=EXPLICIT_RECIPES, help="Candidate chosen from completed development screening")
    p.add_argument("--recipes", nargs="+", choices=EXPLICIT_RECIPES,
                   help="Screen these candidates; standard/weighted controls are always retained")
    p.add_argument("--learning-rates", nargs="+", type=float, default=[1e-5, 2e-5, 3e-5], help="Used only in the lr step")
    p.add_argument("--lr", type=float, default=2e-5, help="Fixed rate for screen/null; chosen rate for confirm")
    p.add_argument("--epochs", type=int, default=5, help="Equal budget for each candidate; shorter screening uses a separate output")
    p.add_argument("--first-seeds", type=int, help="Defaults to one for screening/LR/NULL, two for confirmation; use five after shortlisting")
    p.add_argument("--mode", choices=config.MODES, default="indomain")
    p.add_argument("--held-out-domain", choices=config.DOMAINS, default="restaurant")
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, help="Defaults to the existing v2 output for five epochs, a separate folder for shorter budgets")
    p.add_argument("--python", type=Path, help="Training interpreter; defaults to this repository's virtual environment")
    p.add_argument("--retry-failed", action="store_true", help="Restart failed runs from scratch using the existing runner")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def make_jobs(args):
    if args.step in ("lr", "confirm") and not args.winner:
        raise ValueError(f"--step {args.step} requires --winner chosen on development results")
    if args.recipes and args.step != "screen":
        raise ValueError("--recipes applies only to screening")
    count = args.first_seeds if args.first_seeds is not None else (2 if args.step == "confirm" else 1)
    if not 1 <= count <= len(config.SEEDS):
        raise ValueError("--first-seeds must be between 1 and 5")
    if args.epochs < 1 or (args.step == "confirm" and args.epochs != 5):
        raise ValueError("Use positive epochs; confirmation retains the registered five-epoch budget")
    rates = args.learning_rates if args.step == "lr" else [args.lr]
    if not rates or len(set(rates)) != len(rates) or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be unique, positive, finite values")
    if args.step == "screen":
        names = list(dict.fromkeys(("standard", "weighted", *(args.recipes or SCREEN))))
    elif args.step == "lr":
        names = [args.winner]
    elif args.step == "confirm":
        names = list(dict.fromkeys(("standard", "weighted", args.winner)))
    else:
        names = ["standard_null_vocab", "standard_null"]
    output = (args.output or ROOT / "artifacts" / (
        "experiments_v2" if args.epochs == 5 else f"experiments_screen_e{args.epochs}"
    )).resolve()
    environment_python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python = str(args.python or (environment_python if environment_python.is_file() else Path(sys.executable)))
    domain = args.held_out_domain if args.mode == "crossdomain" else "all_domains"
    revision = pinned_revision()
    jobs = []
    for rate in rates:
        for name in names:
            trial = TrialConfig(**RECIPES[name], lr=rate, epochs=args.epochs, model_revision=revision, offline=True)
            trial.validate()
            for seed in config.SEEDS[:count]:
                command = [python, str(ROOT / "run_experiments.py"), "--stage", "pilot", "--suite", "all",
                           "--recipes", name, "--seeds", str(seed), "--lr", str(rate), "--epochs", str(args.epochs),
                           "--mode", args.mode, "--device", args.device, "--output", str(output), "--execute"]
                if args.mode == "crossdomain":
                    command.extend(["--held-out-domains", args.held_out_domain])
                if args.retry_failed:
                    command.append("--retry-failed")
                directory = output / "research" / args.mode / domain / trial.trial_id / f"seed_{seed}"
                jobs.append(Job(trial, seed, directory, command))
    return jobs, output


def recorded_status(job):
    path = job.directory / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))["status"] if path.is_file() else "pending"


def minutes_per_epoch(output):
    """Approximate observed speed from complete-run file times, not GPU claims."""
    values = []
    for path in (output / "research").glob("**/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        audit, history = path.parent / "audit.json", path.parent / "history.json"
        if manifest["status"] not in ("dev_complete", "complete") or not audit.is_file() or not history.is_file():
            continue
        epochs = manifest["identity"]["config"]["epochs"]
        elapsed = (history.stat().st_mtime - audit.stat().st_mtime) / 60
        if elapsed > 0:
            values.append(elapsed / epochs)
    return statistics.median(values) if values else None


def running_manifest(output):
    """Check both the requested output and the original queue's output."""
    for root in dict.fromkeys((output, ROOT / "artifacts" / "experiments_v2")):
        for path in (root / "research").glob("**/manifest.json"):
            if json.loads(path.read_text(encoding="utf-8"))["status"] == "running":
                return path
    return None


def execute_jobs(jobs, output, p):
    """Use the original trainer's provenance checks and stop on any failure."""
    active = running_manifest(output)
    if active is not None:
        p.error(f"A run is recorded as running: {active.parent}. Finish/stop the existing queue before launching another. "
                "An interrupted run restarts from scratch; this trainer does not save optimizer state for resumption.")
    for job in jobs:
        print(f"Checking/executing {job.trial.name}, seed {job.seed}, lr={job.trial.lr:g}", flush=True)
        result = subprocess.run(job.command, cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        jobs, output = make_jobs(args)
    except ValueError as exc:
        p.error(str(exc))
    print(f"Step: {args.step}; {len(jobs)} runs; {sum(j.trial.epochs for j in jobs)} model-epochs; test access disabled")
    print("Settings: batch=16, max_len=128, weight_decay=.01, warmup=.10, clipping=1.0")
    print(f"Output: {output}")
    for job in jobs:
        print(f"  {job.trial.name:22s} seed={job.seed:<4} lr={job.trial.lr:g} epochs={job.trial.epochs} "
              f"status={recorded_status(job)} id={job.trial.trial_id}")
    waiting = [job for job in jobs if recorded_status(job) in ("pending", "failed")]
    estimate = minutes_per_epoch(output)
    if estimate is not None:
        print(f"Observed approximately {estimate:.1f} min/epoch; pending/failed jobs approximately "
              f"{estimate * sum(j.trial.epochs for j in waiting):.0f} min, excluding any running job.")
    print("Existing runs are verified/skipped by the original trainer only when their provenance matches.")
    if not args.execute or args.dry_run:
        print("Plan only. Add --execute after stopping the old queue; completed models stay available.")
        return
    execute_jobs(jobs, output, p)


if __name__ == "__main__":
    main()
