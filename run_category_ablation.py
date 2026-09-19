"""Run an optional category-only weighting ablation in separate artifacts.

This experiment changes only which prediction head receives inverse-frequency
class weights. It is exploratory and is not part of the primary two-config
study matrix.
"""

import argparse
from pathlib import Path

import config as cfg
from run_study import PROJECT_ROOT, run


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "category_ablation"
DEFAULT_RESULTS_CSV = PROJECT_ROOT / "results_category_ablation.csv"
CONFIG_NAME = "category_weighted"


def build_plan(mode, held_out_domain, seeds):
    """Validate a matched-seed ablation plan before any training begins."""
    seeds = [int(seed) for seed in seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if set(seeds) - set(cfg.SEEDS):
        raise ValueError("Ablation seeds must come from the existing study")
    if mode == "indomain":
        if held_out_domain is not None:
            raise ValueError("Held-out domain is valid only in crossdomain mode")
    elif mode == "crossdomain":
        if held_out_domain not in cfg.DOMAINS:
            raise ValueError("Choose one of the seven configured held-out domains")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return [(mode, held_out_domain, CONFIG_NAME, seed) for seed in seeds]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=cfg.MODES, default="indomain")
    parser.add_argument("--held-out-domain", choices=cfg.DOMAINS)
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        plan = build_plan(args.mode, args.held_out_domain, args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Category-only weighting ablation: {len(plan)} run(s)")
    print(f"Artifacts: {args.output_dir}")
    print(f"Results: {args.results_csv}")
    for index, (mode, domain, config_name, seed) in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {mode} / {domain or 'all-domains'} / {config_name} / seed {seed}")
        if not args.dry_run:
            run(
                mode=mode,
                held_out_domain=domain,
                config_name=config_name,
                seed=seed,
                device=args.device,
                output_dir=args.output_dir,
                results_csv=args.results_csv,
            )
    return 0


if __name__ == "__main__":
    main()
