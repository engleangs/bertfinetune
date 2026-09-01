"""Run a filtered experiment matrix.

The default is the first in-domain matrix: two loss configurations x five
seeds = ten runs. Use ``--mode all`` only after the cross-domain protocol is
frozen.
"""

import argparse

import config as cfg
from run_study import DEFAULT_OUTPUT_ROOT, DEFAULT_RESULTS_CSV, run


def main():
    config_names = [item.name for item in cfg.EXPERIMENTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["indomain", "crossdomain", "all"],
        default="indomain",
        help="data condition to run (default: indomain)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=config_names,
        default=config_names,
        help="one or more loss configurations",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=cfg.SEEDS,
        help="one or more random seeds",
    )
    parser.add_argument(
        "--device", default="auto",
        help="training device (default: auto)",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument(
        "--overwrite", action="store_true",
        help="rerun completed keys instead of skipping them",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="continue the matrix after a failed run",
    )
    args = parser.parse_args()

    modes = cfg.MODES if args.mode == "all" else [args.mode]
    selected_configs = [
        item for item in cfg.EXPERIMENTS if item.name in args.configs
    ]
    combinations = [
        (mode, experiment_cfg, seed)
        for mode in modes
        for experiment_cfg in selected_configs
        for seed in args.seeds
    ]

    failures = []
    for index, (mode, experiment_cfg, seed) in enumerate(combinations, start=1):
        print(
            f"\n=== [{index}/{len(combinations)}] mode={mode} "
            f"config={experiment_cfg.name} seed={seed} ==="
        )
        try:
            run(
                mode,
                experiment_cfg.name,
                seed,
                device=args.device,
                output_dir=args.output_dir,
                results_csv=args.results_csv,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((mode, experiment_cfg.name, seed, str(exc)))
            if not args.continue_on_error:
                raise
            print(f"FAILED: {type(exc).__name__}: {exc}")

    print(f"\nMatrix finished: {len(combinations) - len(failures)} succeeded/skipped, {len(failures)} failed.")
    if failures:
        for mode, config_name, seed, error in failures:
            print(f"- {mode}/{config_name}/seed={seed}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
