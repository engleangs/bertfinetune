"""Run the requested three-loss, five-seed full study without changing the trainer."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from experiment_reporting import full_study as study

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts/class_balance_full_2026_10_09"
DEFAULT_REPORT = ROOT / "phase2/full_study"


def make_jobs(backend, output, device="cuda", retry=False):
    jobs = []
    for mode in ("indomain", "crossdomain"):
        flags = ["--stage", "full", "--suite", "loss", "--recipes", *study.RECIPES,
                 "--first-seeds", "5", "--mode", mode, "--output", str(output), "--device", device,
                 "--lr", "2e-5", "--epochs", "5", "--batch-size", "16", "--max-len", "128",
                 "--weight-decay", ".01", "--warmup-ratio", ".10", "--max-grad-norm", "1.0",
                 "--class-balance-beta", ".9999", "--loss-heads", "bio", "category", "sentiment",
                 "--no-null-head", "--vocabulary-scope", "auto",
                 "--selection-metric", "explicit", "--no-drop-conflict"]
        if mode == "crossdomain":
            flags += ["--held-out-domains", *study.DOMAINS]
        if retry:
            flags.append("--retry-failed")
        args = backend.parser().parse_args(flags)
        for cfg, seed, domain in backend.make_plan(args):
            key = f"{mode}/{domain or 'all_domains'}/{cfg.name}/seed_{seed}"
            directory = f"research/{mode}/{domain or 'all_domains'}/{cfg.trial_id}/seed_{seed}"
            jobs.append((args, cfg, seed, domain, {
                "key": key, "mode": mode, "held_out_domain": domain, "recipe": cfg.name,
                "seed": seed, "trial_id": cfg.trial_id, "config": json.loads(json.dumps(asdict(cfg))),
                "relative_directory": directory,
            }))
    # Give each setting all three seed-13 arms before progressing to later seeds.
    jobs.sort(key=lambda r: (study.SEEDS.index(r[2]),
                            0 if r[0].mode == "indomain" else 1,
                            study.DOMAINS.index(r[3]) if r[3] else -1,
                            study.RECIPES.index(r[1].name)))
    study.validate_matrix([r[4] for r in jobs])
    return jobs


class ProgressStream:
    def __init__(self, original, update):
        self.original, self.update, self.pending = original, update, ""

    def write(self, value):
        self.original.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if line.strip():
                self.update(line[-2000:])
        return len(value)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        # Transformers also queries isatty/encoding while printing its loading report.
        return getattr(self.original, name)


def execute(args, backend, jobs):
    control = args.output / "controls"
    plan = [row[4] for row in jobs]
    from src.experiment_runtime import code_inventory, data_inventory, environment_inventory
    environment, code = environment_inventory(), code_inventory(ROOT)
    if args.device == "cuda" and not environment["cuda_available"]:
        raise ValueError("CUDA is unavailable; no CPU training was started")
    inventory = {"source_sha256": code["source_sha256"], "environment": environment,
                 "train_dev_sha256": data_inventory(ROOT, study.DOMAINS)}
    payload = {"schema": 1, "plan": plan, "inventory": inventory, "device": args.device}
    lock_path = control / "study_plan.json"
    with study.queue_lock(control) as previous:
        if lock_path.exists():
            if study.read(lock_path)["payload"] != payload:
                raise ValueError("Study parameters, source or environment changed; use a new --output")
        else:
            if (args.output / "research").exists():
                raise ValueError("Output contains unowned training results; choose a new --output")
            study.write_json(lock_path, {"created_at_utc": study.now(), "payload": payload})
        study.recover_owned_interruption(args.output, plan, previous, args.retry_failed)
        started = time.monotonic()
        status = {"pid": os.getpid(), "started_at_utc": study.now(), "phase": "training",
                  "expected_runs": 120, "current_job": None}
        durations_path = control / "durations.json"
        durations = study.read(durations_path) if durations_path.exists() else {}

        def update(line=None):
            states = study.inspect_runs(args.output, plan)
            status.update(updated_at_utc=study.now(), elapsed_this_process_seconds=time.monotonic()-started,
                          development_complete=sum(r["status"] in ("dev_complete", "complete") for r in states),
                          test_complete=sum(r["status"] == "complete" for r in states))
            if line:
                status["last_progress"] = line
            trained_seconds = [v["seconds"] for key, v in durations.items() if key.startswith("full/")]
            if trained_seconds:
                status["rough_training_hours_remaining"] = (
                    (120-status["development_complete"]) * sum(trained_seconds)/len(trained_seconds)/3600)
            study.write_json(control / "status.json", status)

        update("Started isolated 120-run full study; test access disabled until the cohort freeze")
        frozen = None
        stages = ("full", "final") if args.stage == "all" else (("full",) if args.stage == "train" else ("final",))
        try:
            for stage in stages:
                if stage == "final":
                    frozen = study.freeze_development(args.output, plan, inventory, ROOT)
                    tests = data_inventory(ROOT, study.DOMAINS, ("test",))
                    target_lock = control / "test_cohort.json"
                    if target_lock.exists() and study.read(target_lock) != tests:
                        raise ValueError("Frozen test data changed; evaluation remains blocked")
                    study.write_json(target_lock, tests)
                    status["phase"] = "frozen_test_evaluation"
                for original_args, cfg, seed, domain, job in jobs:
                    study.verify_inputs(ROOT, inventory)
                    if frozen is not None:
                        study.verify_frozen_job(args.output, job, frozen)
                    if stage == "final" and data_inventory(ROOT, study.DOMAINS, ("test",)) != tests:
                        raise ValueError("Test data changed during the frozen evaluation")
                    run_args = SimpleNamespace(**{**vars(original_args), "stage": stage})
                    status["current_job"] = job["key"]
                    update(f"{stage}: {job['key']}")
                    before = study.inspect_runs(args.output, [job])[0]["status"]
                    run_start = time.monotonic()
                    with redirect_stdout(ProgressStream(sys.stdout, update)):
                        backend.run_trial(run_args, cfg, seed, domain, environment, code)
                    after = study.inspect_runs(args.output, [job])[0]["status"]
                    if before != after:
                        durations[stage+"/"+job["key"]] = {"seconds": time.monotonic()-run_start,
                                                           "finished_at_utc": study.now()}
                        study.write_json(durations_path, durations)
                    study.build_report(args.output, args.report_output, plan)
                    update(f"Finished {stage}: {job['key']}")
                if stage == "full":
                    frozen = study.freeze_development(args.output, plan, inventory, ROOT)
            status.update(phase="complete" if args.stage != "train" else "development_frozen",
                          current_job=None, completed_at_utc=study.now())
            update()
        except BaseException as exc:
            status.update(phase="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            update()
            raise


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--stage", choices=("all", "train", "final"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args(argv)
    args.output, args.report_output = args.output.resolve(), args.report_output.resolve()
    path = args.output / "controls/study_plan.json"
    if args.status:
        state = args.output / "controls/status.json"
        print(json.dumps(study.read(state) if state.exists() else {"phase": "not_started"}, indent=2))
        return
    if args.report and path.exists():
        study.build_report(args.output, args.report_output, study.read(path)["payload"]["plan"])
        print(f"Report: {args.report_output / 'report.md'}")
        return
    import run_experiments as backend
    jobs = make_jobs(backend, args.output, args.device, args.retry_failed)
    print("Full study: 120 full-data runs; three losses; five seeds; in-domain + seven LODO folds.")
    print("Controls: LR=2e-5, epochs=5, batch=16, max_len=128, beta=.9999, NULL=False.")
    print(f"Model output: {args.output}\nShareable report: {args.report_output}")
    if args.report:
        study.build_report(args.output, args.report_output, [r[4] for r in jobs])
    elif args.execute:
        execute(args, backend, jobs)
    else:
        print("Plan only. --execute trains all, freezes development, tests all and refreshes the report.")


if __name__ == "__main__":
    main()
