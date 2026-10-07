"""Plan a matched vocabulary control and a small NULL-head parameter search.

Uses the original trainer and saved-result checks. Plan only unless --execute.
"""

import argparse
from pathlib import Path
import sys

import config
import run_fast_pilot as fast
from src.experiment_config import TrialConfig, candidate_recipes

RECIPES = {r["name"]: r for r in candidate_recipes("loss")}
THRESHOLDS = (.02, .05, .1, .2, .3, .4, .5, .6, .7, .8)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--winner", choices=tuple(RECIPES), default="mixed_025")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--first-seeds", type=int, default=1)
    p.add_argument("--null-pos-weight-caps", nargs="+", type=float, default=[1.0, 10.0, 30.0])
    p.add_argument("--null-loss-weights", nargs="+", type=float, default=[.5])
    p.add_argument("--null-thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path)
    p.add_argument("--python", type=Path)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def make_jobs(args):
    if not 1 <= args.first_seeds <= len(config.SEEDS):
        raise ValueError("--first-seeds must be between 1 and 5")
    for values, name in ((args.null_pos_weight_caps, "caps"), (args.null_loss_weights, "weights"),
                         (args.null_thresholds, "thresholds")):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"NULL {name} must be nonempty and unique")
    output = (args.output or fast.ROOT / "artifacts" / (
        "experiments_v2" if args.epochs == 5 else f"experiments_null_screen_e{args.epochs}"
    )).resolve()
    environment_python = fast.ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    python = str(args.python or (environment_python if environment_python.is_file() else Path(sys.executable)))
    common = {**RECIPES[args.winner], "lr": args.lr, "epochs": args.epochs,
              "model_revision": fast.pinned_revision(), "offline": True, "vocabulary_scope": "explicit-null"}
    # NULL settings have no effect on the disabled head. Retain its original defaults
    # so this control can reuse an existing matched-vocabulary run.
    trials = [TrialConfig(**common, null_head=False)]
    trials.extend(TrialConfig(**common, null_head=True, null_pos_weight_cap=float(cap),
                              null_loss_weight=float(weight), null_thresholds=tuple(map(float, args.null_thresholds)))
                  for cap in args.null_pos_weight_caps for weight in args.null_loss_weights)
    jobs = []
    for trial in trials:
        trial.validate()
        for seed in config.SEEDS[:args.first_seeds]:
            command = [python, str(fast.ROOT / "run_experiments.py"), "--stage", "pilot", "--suite", "loss",
                       "--recipes", args.winner, "--seeds", str(seed), "--lr", str(args.lr), "--epochs", str(args.epochs),
                       "--mode", "indomain", "--device", args.device, "--output", str(output),
                       "--vocabulary-scope", "explicit-null", "--selection-metric", "explicit",
                       "--null-head" if trial.null_head else "--no-null-head", "--execute"]
            if trial.null_head:
                command.extend(["--null-pos-weight-cap", str(trial.null_pos_weight_cap),
                                "--null-loss-weight", str(trial.null_loss_weight),
                                "--null-thresholds", *map(str, trial.null_thresholds)])
            if args.retry_failed:
                command.append("--retry-failed")
            directory = output / "research/indomain/all_domains" / trial.trial_id / f"seed_{seed}"
            jobs.append(fast.Job(trial, seed, directory, command))
    return jobs, output


def label(job):
    if not job.trial.null_head:
        return "Vocabulary control, NULL off"
    return f"NULL on, cap={job.trial.null_pos_weight_cap:g}, weight={job.trial.null_loss_weight:g}"


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        jobs, output = make_jobs(args)
    except ValueError as exc:
        p.error(str(exc))
    print(f"NULL pilot: {args.winner}; {len(jobs)} runs; first {args.first_seeds} registered seed(s); test access disabled")
    print(f"LR={args.lr:g}; epochs={args.epochs}; checkpoint selection=explicit F1")
    print(f"NULL threshold grid (development only): {args.null_thresholds}")
    print(f"Output: {output}")
    for job in jobs:
        print(f"  {label(job):34s} seed={job.seed:<4} status={fast.recorded_status(job)} id={job.trial.trial_id}")
    estimate = fast.minutes_per_epoch(output)
    if estimate is not None:
        pending = [j for j in jobs if fast.recorded_status(j) in ("pending", "failed")]
        print(f"Pending/failed jobs: approximately {estimate * sum(j.trial.epochs for j in pending):.0f} minutes.")
    print("Compare combined development F1, explicit F1 and NULL P/R/F1 before expanding seeds.")
    if not args.execute or args.dry_run:
        print("Plan only. Add --execute to train after the existing queue finishes.")
        return
    fast.execute_jobs(jobs, output, p)
    print("Finished. Refresh with .venv/Scripts/python.exe build_finetuning_report.py")


if __name__ == "__main__":
    main()
