"""Reuse completed pilots, select NULL-off/on settings on dev, then test and report.

Default is a plan. --execute runs the selected step; no training source changes.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

import config
import run_fast_pilot as fast
import run_null_pilot as null_pilot
from src.experiment_config import TrialConfig
from experiment_reporting.quick_selection import check_frozen_run, check_frozen_variants, trial_from_dict

SELECTION_FILE = "selection.json"
NUMERIC_FLAGS = ("ce_weight", "weighted_ce_weight", "focal_gamma", "bio_loss_weight",
                 "category_loss_weight", "sentiment_loss_weight", "null_loss_weight",
                 "null_pos_weight_cap", "null_threshold", "warmup_ratio", "max_grad_norm",
                 "lr", "weight_decay", "epochs", "batch_size", "max_len")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step", choices=("all", "search", "final", "report"), default="all")
    p.add_argument("--first-seeds", type=int, default=2, help="Final shortlist only; screening always uses seed 13")
    p.add_argument("--null-pos-weight-caps", nargs="+", type=float, default=[10.0, 30.0])
    p.add_argument("--null-loss-weights", nargs="+", type=float, default=[.5])
    p.add_argument("--output", type=Path, default=fast.ROOT / "artifacts/experiments_v2")
    p.add_argument("--study-output", type=Path, default=fast.ROOT / "artifacts/quick_tuning")
    p.add_argument("--device", default="auto")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def development_rows(output):
    """Selection never opens test metrics, test predictions or raw test data."""
    rows = []
    for path in sorted((output / "research/indomain/all_domains").glob("*/seed_*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = manifest["identity"]
        if manifest["status"] not in ("dev_complete", "complete") or identity.get("smoke"):
            continue
        metrics = json.loads((path.parent / "dev_metrics.json").read_text(encoding="utf-8"))
        rows.append({"directory": str(path.parent.resolve()), "manifest": manifest,
                     "config": identity["config"], "seed": identity["seed"], "metrics": metrics,
                     "trial_id": path.parent.parent.name})
    return rows


def cohort_signature(row):
    """Allow the searched loss/LR/NULL settings to differ, keeping data and provenance fixed."""
    identity = row["manifest"]["identity"]
    cfg = row["config"]
    payload = {key: identity.get(key) for key in ("source_sha256", "source_data_sha256", "environment",
                                                 "resolved_device", "mode", "held_out_domain", "split_limits")}
    payload["fixed"] = {key: cfg[key] for key in ("model_name", "model_revision", "epochs", "batch_size",
                       "max_len", "weight_decay", "warmup_ratio", "max_grad_norm", "drop_conflict",
                       "selection_metric", "bio_loss_weight", "category_loss_weight", "sentiment_loss_weight",
                       "loss_heads")}
    return json.dumps(payload, sort_keys=True)


def best_row(rows, metric):
    if not rows:
        raise ValueError("The required development candidates have not completed")
    return max(rows, key=lambda r: (r["metrics"][metric]["micro_f1"],
                                   r["metrics"]["explicit"]["micro_f1"],
                                   -r["metrics"]["unweighted_loss"], r["trial_id"]))


def job_for_config(cfg, seed, args, stage="pilot"):
    python = fast.ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    command = [str(python), str(fast.ROOT / "run_experiments.py"), "--stage", stage,
               "--suite", "loss", "--recipes", cfg.name, "--seeds", str(seed), "--mode", "indomain",
               "--output", str(args.output.resolve()), "--device", args.device,
               "--model-name", cfg.model_name, "--model-revision", cfg.model_revision,
               "--loss", cfg.loss_type, "--loss-heads", *cfg.loss_heads,
               "--vocabulary-scope", cfg.vocabulary_scope, "--selection-metric", cfg.selection_metric,
               "--null-thresholds", *map(str, cfg.null_thresholds),
               "--null-head" if cfg.null_head else "--no-null-head",
               "--drop-conflict" if cfg.drop_conflict else "--no-drop-conflict", "--execute"]
    for field in NUMERIC_FLAGS:
        command.extend(["--" + field.replace("_", "-"), str(getattr(cfg, field))])
    if not cfg.offline:
        command.append("--online")
    if args.retry_failed and stage != "final":
        command.append("--retry-failed")
    directory = args.output.resolve() / "research/indomain/all_domains" / cfg.trial_id / f"seed_{seed}"
    return fast.Job(cfg, seed, directory, command)


def matching_rows(output, jobs):
    keys = {(j.trial.trial_id, j.seed) for j in jobs}
    rows = [r for r in development_rows(output) if (r["trial_id"], r["seed"]) in keys]
    if {(r["trial_id"], r["seed"]) for r in rows} != keys:
        raise ValueError("Some requested development runs are incomplete; retry the search before testing")
    configs = {j.trial.trial_id: asdict(j.trial) for j in jobs}
    if any(json.dumps(r["config"], sort_keys=True) != json.dumps(configs[r["trial_id"]], sort_keys=True) for r in rows):
        raise ValueError("Completed development configuration does not match the requested settings")
    return rows


def search_result(stage, row):
    cfg, metrics = row["config"], row["metrics"]
    return {"stage": stage, "trial_id": row["trial_id"], "seed": row["seed"], "config": cfg,
            "explicit_f1": metrics["explicit"]["micro_f1"], "null_f1": metrics["null"]["micro_f1"],
            "combined_f1": metrics["combined"]["micro_f1"], "null_threshold": metrics["null_threshold"]}


def write_freeze(path, frozen):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def fast_jobs(args, step, winner=None):
    flags = ["--step", step, "--first-seeds", "1", "--output", str(args.output), "--device", args.device]
    if winner:
        flags.extend(["--winner", winner])
    if args.retry_failed:
        flags.append("--retry-failed")
    return fast.make_jobs(fast.parser().parse_args(flags))[0]


def null_jobs(args, explicit):
    flags = ["--winner", explicit["config"]["name"], "--lr", str(explicit["config"]["lr"]),
             "--first-seeds", "1", "--output", str(args.output), "--device", args.device,
             "--null-pos-weight-caps", *map(str, args.null_pos_weight_caps),
             "--null-loss-weights", *map(str, args.null_loss_weights)]
    if args.retry_failed:
        flags.append("--retry-failed")
    return null_pilot.make_jobs(null_pilot.parser().parse_args(flags))[0]


def search(args, p):
    freeze_path = args.study_output / SELECTION_FILE
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
        if frozen["seeds"] != config.SEEDS[:args.first_seeds]:
            raise ValueError("This study is already frozen with different seeds; use a separate --study-output")
        print(f"Reuse frozen selection: {freeze_path}", flush=True)
        return frozen

    screen = fast_jobs(args, "screen")
    print("1/4: Complete the professor's six-loss screen at LR 2e-5, seed 13.", flush=True)
    fast.execute_jobs(screen, args.output, p)
    screen_rows = matching_rows(args.output, screen)
    screen_winner = best_row(screen_rows, "explicit")
    print(f"2/4: Learning-rate search for {screen_winner['config']['name']} only.", flush=True)
    rates = fast_jobs(args, "lr", screen_winner["config"]["name"])
    fast.execute_jobs(rates, args.output, p)

    signature = cohort_signature(screen_rows[0])
    eligible = [r for r in development_rows(args.output)
                if r["seed"] == 13 and r["config"]["name"] in fast.SCREEN
                and not r["config"]["null_head"] and r["config"]["vocabulary_scope"] in ("auto", "explicit")
                and cohort_signature(r) == signature]
    explicit = best_row(eligible, "explicit")
    print(f"3/4: NULL search for {explicit['config']['name']}, LR={explicit['config']['lr']:g}.", flush=True)
    null_grid = null_jobs(args, explicit)
    fast.execute_jobs(null_grid, args.output, p)
    null_rows = matching_rows(args.output, null_grid)
    implicit = best_row([r for r in null_rows if r["config"]["null_head"]], "combined")
    control = null_grid[0].trial
    configs = [trial_from_dict(explicit["config"]), control, trial_from_dict(implicit["config"])]

    print(f"4/4: Check only the selected settings in seeds {config.SEEDS[:args.first_seeds]}.", flush=True)
    confirmations = {}
    for cfg in configs:
        for seed in config.SEEDS[:args.first_seeds]:
            job = job_for_config(cfg, seed, args)
            confirmations[(cfg.trial_id, seed)] = job
    fast.execute_jobs(list(confirmations.values()), args.output, p)
    rows = matching_rows(args.output, list(confirmations.values()))
    no_null = [cfg for cfg in configs[:2] if not cfg.null_head]
    def mean_explicit(cfg):
        values = [r["metrics"]["explicit"]["micro_f1"] for r in rows if r["trial_id"] == cfg.trial_id]
        return sum(values) / len(values)
    winner_off = max(no_null, key=mean_explicit)
    roles = [("Best NULL off", winner_off), ("Best NULL on", configs[2]), ("Vocabulary control", control)]
    variants = {}
    for role, cfg in roles:
        variant = variants.setdefault(cfg.trial_id, {"trial_id": cfg.trial_id, "config": asdict(cfg), "roles": [], "runs": []})
        variant["roles"].append(role)
    for variant in variants.values():
        for row in rows:
            if row["trial_id"] == variant["trial_id"]:
                manifest = row["manifest"]
                variant["runs"].append({"seed": row["seed"], "directory": row["directory"],
                                       "fingerprint": manifest["fingerprint"],
                                       "checkpoint_sha256": manifest["checkpoint_sha256"],
                                       "best_epoch": manifest["training"]["best_epoch"],
                                       "null_threshold": row["metrics"]["null_threshold"]})
    search_results = ([search_result("Loss screen", r) for r in screen_rows]
                      + [search_result("LR search", r) for r in matching_rows(args.output, rates)]
                      + [search_result("NULL search", r) for r in null_rows])
    frozen = {"created_at": datetime.now(timezone.utc).isoformat(),
              "seeds": config.SEEDS[:args.first_seeds], "variants": list(variants.values()),
              "search_results": search_results,
              "selection": {"screen_seed": 13, "screen": list(fast.SCREEN), "learning_rates": [1e-5, 2e-5, 3e-5],
                            "null_caps": args.null_pos_weight_caps, "null_loss_weights": args.null_loss_weights,
                            "null_threshold_grid": list(null_pilot.THRESHOLDS),
                            "checkpoint_metric": "explicit development F1",
                            "NULL_config_metric": "combined development F1 on seed 13",
                            "NULL_threshold_metric": "NULL development F1",
                            "NULL_off_metric": "mean explicit development F1 over confirmation seeds",
                            "test_note": "Exploratory evaluation: historical test sets were already inspected."}}
    check_frozen_variants(frozen)
    write_freeze(freeze_path, frozen)
    print(f"Frozen development-selected settings: {freeze_path}", flush=True)
    return frozen


def final_jobs(frozen, args):
    check_frozen_variants(frozen)
    jobs = []
    for variant in frozen["variants"]:
        cfg = trial_from_dict(variant["config"])
        for run in variant["runs"]:
            job = job_for_config(cfg, run["seed"], args, "final")
            if job.directory.resolve() != Path(run["directory"]).resolve():
                raise ValueError("The frozen study uses a different --output directory")
            check_frozen_run(variant, run)
            jobs.append(job)
    return jobs


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    args.study_output = args.study_output.resolve()
    if not 1 <= args.first_seeds <= len(config.SEEDS):
        p.error("--first-seeds must be between 1 and 5")
    try:
        # Validate all declared NULL knobs before running any screen model.
        null_pilot.make_jobs(null_pilot.parser().parse_args([
            "--null-pos-weight-caps", *map(str, args.null_pos_weight_caps),
            "--null-loss-weights", *map(str, args.null_loss_weights)]))
        if not args.execute or args.dry_run:
            frozen_path = args.study_output / SELECTION_FILE
            if args.step in ("final", "report") or frozen_path.exists():
                if not frozen_path.is_file():
                    print("No frozen selection yet. Complete --step search --execute first.")
                else:
                    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
                    jobs = final_jobs(frozen, args)
                    print(f"Frozen shortlist: {len(frozen['variants'])} variants, {len(jobs)} saved checkpoints.")
                    print(f"Next step: {args.step}. Report: {args.study_output / 'report.md'}")
                print("Plan only. Add --execute.")
                return
            screen = fast_jobs(args, "screen")
            pending = sum(fast.recorded_status(j) in ("pending", "failed") for j in screen)
            print(f"Quick plan: {pending} unfinished screen model(s); reuse the completed loss/mixture pilots.")
            print("Tune three learning rates only for the screen winner; then test two NULL caps plus one vocabulary control.")
            print(f"Check the selected settings in {args.first_seeds} seed(s), freeze, evaluate tests, and build the report.")
            print("Typical new training: about 2 hours; a different screen winner can add learning-rate/seed runs.")
            print("Plan only. Add --execute; use --step search to stop before test evaluation.")
            return
        frozen = None
        if args.step in ("all", "search"):
            frozen = search(args, p)
        else:
            path = args.study_output / SELECTION_FILE
            if not path.is_file():
                raise ValueError("No frozen selection. Run --step search --execute first")
            frozen = json.loads(path.read_text(encoding="utf-8"))
        if args.step in ("all", "final"):
            print("Evaluate the frozen shortlist on tests once; no test-driven parameter changes.", flush=True)
            fast.execute_jobs(final_jobs(frozen, args), args.output, p)
        from experiment_reporting.quick_report import write_quick_report
        write_quick_report(frozen, args.study_output)
        print(f"Report: {args.study_output / 'report.md'}")
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
