"""Wait for seed-13 LODO, finish the remaining seeds, freeze, test and report.

This orchestrates the existing trainer without changing its training protocol.
Run with --wait-for-first --execute to queue behind the user's first seven folds.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DOMAINS = ("coursera", "hotel", "laptop", "restaurant", "phone", "sight", "food")
NUMERIC_FLAGS = (
    "lr", "epochs", "batch_size", "max_len", "weight_decay", "ce_weight",
    "weighted_ce_weight", "focal_gamma", "bio_loss_weight", "category_loss_weight",
    "sentiment_loss_weight", "null_loss_weight", "null_pos_weight_cap",
    "null_threshold", "warmup_ratio", "max_grad_norm",
)
DEV_FILES = ("best.pt", "dev_metrics.json", "dev_predictions.jsonl",
             "dev_taxonomy.jsonl", "history.json", "audit.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a small status file atomically; no model files are rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_variant(path):
    frozen = read(path)
    variant = next(v for v in frozen["variants"] if "Best NULL off" in v.get("roles", []))
    cfg = variant["config"]
    expected = f"{cfg['name']}_" + hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    if cfg["null_head"] or cfg["selection_metric"] != "explicit" or variant["trial_id"] != expected:
        raise ValueError("Expected the valid frozen, explicit-selected NULL-off configuration")
    return frozen, variant


def directory(output, trial_id, domain, seed):
    return output / "research" / "crossdomain" / domain / trial_id / f"seed_{seed}"


def cohort_status(output, variant, seeds):
    """Inspect manifests only; this gate never opens test metrics or test data."""
    statuses = []
    for domain in DOMAINS:
        for seed in seeds:
            path = directory(output, variant["trial_id"], domain, seed)
            manifest = read(path / "manifest.json") if (path / "manifest.json").is_file() else None
            status = "pending" if manifest is None else manifest["status"]
            if manifest:
                identity = manifest["identity"]
                if (identity["config"] != variant["config"] or identity["seed"] != seed
                        or identity["mode"] != "crossdomain" or identity["held_out_domain"] != domain
                        or identity.get("smoke")):
                    raise ValueError(f"LODO run identity changed: {path}")
                if status == "failed":
                    raise ValueError(f"LODO fold failed: {path}: {manifest.get('error', 'see manifest')}")
                if status in ("dev_complete", "complete"):
                    missing = [name for name in DEV_FILES if not (path / name).is_file()]
                    if missing:
                        raise ValueError(f"Completed fold is missing {missing}: {path}")
            statuses.append({"domain": domain, "seed": seed, "status": status})
    return statuses


def wait_for_first(output, variant, seed, *, wait, hours, poll=30, update=lambda value: None):
    deadline = time.monotonic() + hours * 3600
    while True:
        statuses = cohort_status(output, variant, [seed])
        ready = sum(row["status"] in ("dev_complete", "complete") for row in statuses)
        update({"status": "waiting_for_first", "first_seed": seed,
                "completed_first_folds": ready, "expected_first_folds": len(DOMAINS), "folds": statuses})
        if ready == len(DOMAINS):
            return
        if not wait:
            raise ValueError("The first seven folds are incomplete; use --wait-for-first to queue behind them")
        if time.monotonic() >= deadline:
            raise ValueError("Timed out waiting for the user's first folds; their process was left untouched")
        print(f"Waiting for seed {seed}: {ready}/{len(DOMAINS)} development folds complete; no continuation training started.", flush=True)
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def command(args, variant, stage, seeds):
    cfg = variant["config"]
    result = [sys.executable, "-u", str(ROOT / "run_experiments.py"), "--stage", stage,
              "--mode", "crossdomain", "--held-out-domains", *DOMAINS,
              "--suite", "loss", "--recipes", cfg["name"], "--seeds", *map(str, seeds),
              "--output", str(args.output), "--device", args.device,
              "--model-name", cfg["model_name"], "--model-revision", cfg["model_revision"],
              "--loss", cfg["loss_type"], "--loss-heads", *cfg["loss_heads"],
              "--vocabulary-scope", cfg["vocabulary_scope"], "--selection-metric", cfg["selection_metric"],
              "--null-thresholds", *map(str, cfg["null_thresholds"]),
              "--no-null-head", "--drop-conflict" if cfg["drop_conflict"] else "--no-drop-conflict",
              "--execute"]
    for key in NUMERIC_FLAGS:
        result.extend(["--" + key.replace("_", "-"), str(cfg[key])])
    if not cfg["offline"]:
        result.append("--online")
    if args.retry_failed and stage != "final":
        result.append("--retry-failed")
    return result


def freeze(output, variant, seeds, path):
    """Freeze and hash every development checkpoint before any new test evaluation."""
    statuses = cohort_status(output, variant, seeds)
    if any(row["status"] not in ("dev_complete", "complete") for row in statuses):
        raise ValueError("Cannot freeze an incomplete LODO development cohort")
    runs, source_by_domain = [], {}
    for row in statuses:
        domain, seed = row["domain"], row["seed"]
        folder = directory(output, variant["trial_id"], domain, seed)
        manifest, metric = read(folder / "manifest.json"), read(folder / "dev_metrics.json")
        identity = manifest["identity"]
        if (domain in identity["source_domains"]
                or set(identity["source_domains"]) != set(DOMAINS) - {domain}):
            raise ValueError(f"Held-out domain appeared in source data: {folder}")
        for name, expected in identity["source_sha256"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Training code changed: {name}")
        for name, expected in identity["source_data_sha256"].items():
            if file_hash(ROOT / "data/m-absa" / name) != expected:
                raise ValueError(f"Source data changed: {name}")
        signature = {key: identity.get(key) for key in (
            "source_sha256", "source_data_sha256", "environment", "resolved_device", "split_limits")}
        if domain in source_by_domain and source_by_domain[domain] != signature:
            raise ValueError(f"Seed provenance differs within fold {domain}")
        source_by_domain[domain] = signature
        if file_hash(folder / "best.pt") != manifest["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint contents changed: {folder}")
        if len(read(folder / "history.json")) != variant["config"]["epochs"]:
            raise ValueError(f"Incomplete epoch history: {folder}")
        runs.append({"domain": domain, "seed": seed, "directory": str(folder.resolve()),
                     "fingerprint": manifest["fingerprint"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                     "best_epoch": manifest["training"]["best_epoch"], "null_threshold": metric["null_threshold"]})
    frozen = {"created_at_utc": now(), "trial_id": variant["trial_id"], "config": variant["config"],
              "domains": list(DOMAINS), "seeds": seeds, "runs": runs,
              "selection": "Explicit source-development F1; parameters transferred from frozen in-domain selection",
              "exploratory": True}
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k != "created_at_utc"} != {
                k: v for k, v in frozen.items() if k != "created_at_utc"}:
            raise ValueError("Existing LODO freeze changed; preserve it and use a separate study-output")
        return previous
    write_json(path, frozen)
    return frozen


@contextmanager
def continuation_lock(output):
    """One continuation per study; stale locks can be reclaimed after process exit."""
    from null_experiments.storage import process_alive
    output.mkdir(parents=True, exist_ok=True)
    path = output / "active.lock"
    payload = {"pid": os.getpid(), "host": socket.gethostname(), "started_at_utc": now()}
    if path.exists():
        previous = read(path)
        if previous["host"] != payload["host"] or process_alive(previous["pid"]):
            raise ValueError(f"A continuation already holds {path}; PID {previous['pid']}")
        path.unlink()
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle)
    try:
        yield
    finally:
        if path.exists() and read(path) == payload:
            path.unlink()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", choices=("finish", "report"), default="finish")
    p.add_argument("--selection", type=Path, default=ROOT / "artifacts/quick_tuning_5seeds/selection.json")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments_v2")
    p.add_argument("--study-output", type=Path, default=ROOT / "artifacts/lodo_continuation")
    p.add_argument("--report-output", type=Path, default=ROOT / "lodo_finetuning_results")
    p.add_argument("--first-seeds", type=int, choices=range(1, 6), default=5)
    p.add_argument("--wait-for-first", action="store_true")
    p.add_argument("--wait-hours", type=float, default=48)
    p.add_argument("--poll-seconds", type=float, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 0 < args.wait_hours <= 72 or not 5 <= args.poll_seconds <= 60:
        p.error("Use wait-hours in (0,72] and poll-seconds in [5,60]")
    for key in ("selection", "output", "study_output", "report_output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.step == "report":
        from experiment_reporting.lodo_report import write_report
        print(write_report(args.study_output / "selection.json", args.report_output))
        return
    origin, variant = selected_variant(args.selection)
    seeds = origin["seeds"][:args.first_seeds]
    remaining = seeds[1:]
    print(f"LODO continuation: wait for seed {seeds[0]}; {len(DOMAINS)*len(remaining)} remaining training runs; then {len(DOMAINS)*len(seeds)} test evaluations.", flush=True)
    print(f"Fixed configuration: {variant['trial_id']}; no new hyperparameter search; NULL head off.", flush=True)
    if not args.execute:
        print("Plan only. Add --wait-for-first --execute to queue behind the first seven folds.")
        return
    started = now()

    def update(fields):
        write_json(args.study_output / "status.json", {
            "pid": os.getpid(), "host": socket.gethostname(), "started_at_utc": started,
            "updated_at_utc": now(), "trial_id": variant["trial_id"], "domains": list(DOMAINS),
            "seeds": seeds, "report": str(args.report_output / "report.md"), **fields})

    with continuation_lock(args.study_output):
        try:
            write_json(args.study_output / "plan.json", {
                "created_at_utc": started, "origin_selection": str(args.selection),
                "trial_id": variant["trial_id"], "config": variant["config"],
                "domains": list(DOMAINS), "seeds": seeds, "remaining_seeds": remaining,
                "training_command": command(args, variant, "full", remaining) if remaining else None,
                "test_command": command(args, variant, "final", seeds)})
            wait_for_first(args.output, variant, seeds[0], wait=args.wait_for_first,
                           hours=args.wait_hours, poll=args.poll_seconds, update=update)
            # Give the original runner time to finish its final summary/cleanup.
            time.sleep(5)
            cohort = cohort_status(args.output, variant, seeds)
            training_needed = any(row["status"] not in ("dev_complete", "complete") for row in cohort)
            if remaining and training_needed:
                update({"status": "training_remaining", "remaining_seeds": remaining})
                subprocess.run(command(args, variant, "full", remaining), cwd=ROOT, check=True)
            elif not training_needed:
                print("All development folds are complete; reusing all trained models.", flush=True)
            update({"status": "freezing_development_checkpoints"})
            freeze(args.output, variant, seeds, args.study_output / "selection.json")
            update({"status": "evaluating_test"})
            test_command = command(args, variant, "final", seeds)
            test_command[2] = str(ROOT / "evaluate_lodo_frozen.py")
            test_command[3:3] = ["--freeze", str(args.study_output / "selection.json")]
            subprocess.run(test_command, cwd=ROOT, check=True)
            update({"status": "writing_report"})
            from experiment_reporting.lodo_report import write_report
            result = write_report(args.study_output / "selection.json", args.report_output)
            update({"status": "complete", "completed_at_utc": now()})
            print(f"LODO continuation complete: {result}", flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise


if __name__ == "__main__":
    main()
