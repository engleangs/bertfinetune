"""Finish the frozen NULL study and combine it with completed quick tuning."""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick-study", type=Path, default=ROOT / "artifacts/quick_tuning_5seeds")
    p.add_argument("--null-study", type=Path, default=ROOT / "artifacts/null_fast_track")
    p.add_argument("--experiments-output", type=Path, default=ROOT / "artifacts/experiments_v2", help="Existing quick-screen checkpoints")
    p.add_argument("--output", type=Path, default=ROOT / "final_presentation_results")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--finish-null", action="store_true", help="Complete finalist seeds, evaluate the freeze, then report")
    action.add_argument("--evaluate-null", action="store_true", help="Evaluate the existing freeze without more training, then report")
    p.add_argument("--first-seeds", type=int, choices=range(1, 6), default=5, help="Finalist cohort for --finish-null")
    p.add_argument("--wait-for-null", action="store_true", help="Wait for the existing NULL queue to exit before continuing")
    p.add_argument("--wait-hours", type=float, default=12)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--execute", action="store_true", help="Required to train/evaluate; reporting alone needs no flag")
    p.add_argument("--ablation-tests", action="store_true", help="Freeze and test every registered trained quick/NULL screen candidate")
    return p


def wait_for_queue(output, wait=False, hours=12):
    from null_experiments.storage import process_alive
    deadline, last_message = time.monotonic() + hours * 3600, 0
    path = output / "active.lock"
    while path.exists():
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return  # The current queue just removed its completed lock.
        except json.JSONDecodeError:
            if not wait or time.monotonic() >= deadline:
                raise ValueError("The queue lock is being written; retry when the current queue settles")
            time.sleep(1)
            continue
        if lock["host"] != socket.gethostname():
            raise ValueError("The NULL lock belongs to another host; finish that queue there first")
        if not process_alive(lock["pid"]):
            return  # The original runner safely removes its stale lock.
        if not wait:
            raise ValueError("NULL training is active. Use --wait-for-null, or run after that queue finishes")
        now = time.monotonic()
        if now >= deadline:
            raise ValueError("Waiting timed out; the existing training process has been left running")
        if now - last_message >= 60:
            print(f"Waiting for the existing NULL queue (PID {lock['pid']}); no second training queue started.", flush=True)
            last_message = now
        time.sleep(min(15, deadline - now))


def remaining_commands(args):
    """Use only the completed search; never restart a hyperparameter grid."""
    state_path = args.null_study / "search_state.json"
    if not state_path.exists() or "selected_null" not in json.loads(state_path.read_text(encoding="utf-8")):
        raise ValueError("NULL search has not finished. Resume its original command first; do not change its search plan")
    freeze_path = args.null_study / "selection.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else None
    common = [sys.executable, str(ROOT / "run_null_fast_track.py"), "--output", str(args.null_study),
              "--device", args.device, "--execute"]
    commands = []
    if args.finish_null:
        seeds = max(args.first_seeds, len(frozen["seeds"]) if frozen else 0)
        null_candidates = sum(v["config"]["null_head"] for v in frozen["variants"]) if frozen else 1
        epochs = frozen["variants"][0]["config"]["epochs"] if frozen else 5
        command = common + ["--step", "confirm", "--first-seeds", str(seeds), "--epochs", str(epochs),
                            "--confirm-top", str(null_candidates)]
        if args.retry_failed:
            command.append("--retry-failed")
        commands.append(command)
    elif not frozen:
        raise ValueError("No development freeze exists; finish confirmation before evaluating test data")
    commands.append(common + ["--step", "final"])
    return commands


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not 0 < args.wait_hours <= 48:
        p.error("--wait-hours must be positive and no more than 48")
    args.quick_study, args.null_study, args.output, args.experiments_output = [v.resolve() for v in
        (args.quick_study, args.null_study, args.output, args.experiments_output)]
    try:
        if args.finish_null or args.evaluate_null or args.ablation_tests:
            if not args.execute:
                print("Plan: reuse completed quick-tuning evaluations; wait for the NULL search/confirmation if requested.")
                print(f"NULL: {'confirm at least ' + str(args.first_seeds) + ' finalist seeds, then' if args.finish_null else 'use the existing freeze to'} evaluate tests, then combine reports.")
                print("Only finalist seeds can need training. No new parameter screen; test scores do not select settings.")
                if args.ablation_tests:
                    print("Also freeze/test all registered trained screens, separately labeled by seed and epoch budget; no screen retraining.")
                print("Add --execute to run this flow. --wait-for-null can wait for the current queue.")
                return
            wait_for_queue(args.null_study, args.wait_for_null, args.wait_hours)
            if args.ablation_tests:
                from experiment_reporting.ablation_evaluation import finish_with_ablations
                finish_with_ablations(ROOT, args)
            else:
                for command in remaining_commands(args):
                    print("Run: " + subprocess.list2cmdline(command), flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
        from experiment_reporting.presentation_report import write_report
        result = write_report(args.quick_study, args.null_study, args.output)
        print(f"Combined report: {result}")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
