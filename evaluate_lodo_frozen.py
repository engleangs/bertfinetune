"""Evaluate frozen LODO checkpoints while tolerating a locked summary CSV.

Training, checkpoint selection and metrics use the unchanged experiment runner.
Only its derived CSV index writer is redirected when Windows locks that index.
"""

import argparse
from pathlib import Path

from continue_lodo_finetuning import DOMAINS, freeze, now, read, write_json


def resilient_csv_writer(original, recovery_metadata):
    redirects = {}

    def write(destination, *args, **kwargs):
        destination = Path(destination)
        if destination in redirects:
            return original(redirects[destination], *args, **kwargs)
        try:
            return original(destination, *args, **kwargs)
        except PermissionError:
            if destination.name != "development_results.csv":
                raise
            fallback = destination.with_name("development_results_lodo_recovery.csv")
            # Make the alternate index first; propagate errors if it cannot be written.
            result = original(fallback, *args, **kwargs)
            redirects[destination] = fallback
            write_json(recovery_metadata, {
                "recorded_at_utc": now(), "original": str(destination.resolve()),
                "alternate_summary": str(fallback.resolve()),
                "reason": "PermissionError updating the derived development-results CSV",
                "model_metrics_preserved": True,
            })
            print(f"Summary CSV is locked; preserving {destination} and writing {fallback} instead.", flush=True)
            return result

    return write


def validate_plan(args, plan, frozen):
    expected = {(domain, seed) for domain in frozen["domains"] for seed in frozen["seeds"]}
    observed = {(domain, seed) for _, seed, domain in plan}
    if (args.stage != "final" or args.mode != "crossdomain" or args.smoke
            or set(frozen["domains"]) != set(DOMAINS) or observed != expected
            or len(plan) != len(expected)):
        raise ValueError("Recovery evaluates the entire frozen LODO test matrix only; training is disabled")
    for cfg, _, _ in plan:
        if cfg.trial_id != frozen["trial_id"]:
            raise ValueError("Recovery parameters differ from the frozen development configuration")


def main(argv=None):
    wrapper = argparse.ArgumentParser(description=__doc__, add_help=False)
    wrapper.add_argument("--freeze", type=Path, required=True)
    wrapper_args, trainer_flags = wrapper.parse_known_args(argv)
    import run_experiments as trainer
    args = trainer.parser().parse_args(trainer_flags)
    plan = trainer.make_plan(args)
    frozen = read(wrapper_args.freeze)
    validate_plan(args, plan, frozen)
    if args.execute:
        # Checks training hashes, source data, checkpoint contents and unchanged
        # development operating points. It does not read test scores for selection.
        freeze(args.output.resolve(), {"config": frozen["config"], "trial_id": frozen["trial_id"]},
               frozen["seeds"], wrapper_args.freeze)
    trainer.upsert_csv_row = resilient_csv_writer(
        trainer.upsert_csv_row, wrapper_args.freeze.parent / "csv_recovery.json")
    trainer.main(trainer_flags)


if __name__ == "__main__":
    main()
