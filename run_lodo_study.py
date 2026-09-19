"""Run the remaining leave-one-domain-out (LODO) experiment matrix.

By default this excludes ``config.HOLD_OUT_DOMAIN`` because that domain already
has completed post-pilot runs. With the current configuration, the default is:

    6 remaining domains x 2 loss configurations x 5 seeds = 60 runs

This runner only adds safe multi-domain orchestration. It does not itself fix
the warmup, gradient-clipping, alignment, or other protocol deviations recorded
in ``current-progress.md``. Complete those gates before treating new outputs as
version-2 final evidence.
"""

import argparse
from pathlib import Path

import config as cfg
from run_study import PROJECT_ROOT, run


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lodo_runs"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results_lodo.csv"


def default_remaining_domains():
    """Return every configured domain except the already completed holdout."""
    return [domain for domain in cfg.DOMAINS if domain != cfg.HOLD_OUT_DOMAIN]


def build_run_matrix(domains, config_names, seeds):
    """Validate and return deterministic (domain, config, seed) run keys."""
    domains = list(domains)
    config_names = list(config_names)
    seeds = [int(seed) for seed in seeds]
    if not domains:
        raise ValueError("At least one held-out domain is required")
    if len(domains) != len(set(domains)):
        raise ValueError("Held-out domains must be unique")
    unknown_domains = sorted(set(domains) - set(cfg.DOMAINS))
    if unknown_domains:
        raise ValueError(f"Unknown held-out domains: {unknown_domains}")
    if len(config_names) != len(set(config_names)):
        raise ValueError("Configuration names must be unique")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be unique")
    return [
        (domain, config_name, seed)
        for domain in domains
        for config_name in config_names
        for seed in seeds
    ]


def main(argv=None):
    config_names = [item.name for item in cfg.EXPERIMENTS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=cfg.DOMAINS,
        default=default_remaining_domains(),
        help=(
            "domains to hold out one at a time; default is every domain except "
            f"the completed {cfg.HOLD_OUT_DOMAIN!r} fold"
        ),
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
        help="rerun completed domain/config/seed keys instead of skipping them",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="continue the matrix after a failed run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the planned run keys without training",
    )
    args = parser.parse_args(argv)

    combinations = build_run_matrix(args.domains, args.configs, args.seeds)
    print(
        f"LODO matrix: {len(args.domains)} domains x {len(args.configs)} configs "
        f"x {len(args.seeds)} seeds = {len(combinations)} runs"
    )
    print(f"Held-out domains: {', '.join(args.domains)}")
    print(f"Artifacts: {Path(args.output_dir)}")
    print(f"Results: {Path(args.results_csv)}")

    if args.dry_run:
        for index, (domain, config_name, seed) in enumerate(combinations, start=1):
            print(
                f"[{index}/{len(combinations)}] held_out={domain} "
                f"config={config_name} seed={seed}"
            )
        return 0

    failures = []
    for index, (domain, config_name, seed) in enumerate(combinations, start=1):
        print(
            f"\n=== [{index}/{len(combinations)}] held_out={domain} "
            f"config={config_name} seed={seed} ==="
        )
        try:
            run(
                mode="crossdomain",
                config_name=config_name,
                seed=seed,
                device=args.device,
                output_dir=args.output_dir,
                results_csv=args.results_csv,
                overwrite=args.overwrite,
                held_out_domain=domain,
            )
        except Exception as exc:
            failures.append((domain, config_name, seed, str(exc)))
            if not args.continue_on_error:
                raise
            print(f"FAILED: {type(exc).__name__}: {exc}")

    succeeded_or_skipped = len(combinations) - len(failures)
    print(
        f"\nLODO matrix finished: {succeeded_or_skipped} succeeded/skipped, "
        f"{len(failures)} failed."
    )
    if failures:
        for domain, config_name, seed, error in failures:
            print(f"- {domain}/{config_name}/seed={seed}: {error}")
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    main()
