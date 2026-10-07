"""Plan/run techniques 1–5 in a small staged study; test is a separate step."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import config
from null_experiments.config import NullConfig, VERSION

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--step", choices=("all", "calibrate", "search", "confirm", "final", "report", "custom"), default="all")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/null_fast_track")
    p.add_argument("--share-report", type=Path, help="Small Git-ready report folder; default null_fast_track_reports/<output name>")
    p.add_argument("--first-seeds", type=int, default=None, help="Confirmation seed prefix; default 2 (custom default 1)")
    p.add_argument("--seeds", nargs="+", type=int, help="Custom step only; explicit unique seeds")
    p.add_argument("--screen-epochs", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5, help="Confirmation/custom epoch budget")
    p.add_argument("--confirm-top", type=int, choices=(1, 2), default=1, help="NULL candidates to confirm, plus NULL-off control")
    p.add_argument("--tune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tune-lr", action="store_true", help="Optional additional LR 2e-5 candidate; default keeps the established LR")
    p.add_argument("--skip-calibration", action="store_true", help="Skip optional legacy inference; new models still calibrate each epoch")
    p.add_argument("--legacy-selection", type=Path, default=ROOT / "artifacts/quick_tuning_5seeds/selection.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--retry-failed", action="store_true", help="Resume failed runs; interruptions resume automatically")
    p.add_argument("--smoke", action="store_true", help="Tiny integration runs, isolated from research and never eligible for test")
    p.add_argument("--train-limit", type=int, default=64)
    p.add_argument("--dev-limit", type=int, default=32)
    p.add_argument("--recipe", choices=("bce", "focal", "asl", "ia", "attention", "hard", "null_off", "custom"), default="bce")
    p.add_argument("--null-loss", choices=("bce", "focal", "asl"))
    p.add_argument("--representation", choices=("cls", "ia", "ia_attention"))
    p.add_argument("--negative-sampling", choices=("all", "random", "hard"))
    p.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    p.add_argument("--selection-metric", choices=("combined", "null", "explicit"), default="combined")
    for name, kind in (("lr", float), ("batch-size", int), ("max-len", int), ("weight-decay", float),
                       ("warmup-ratio", float), ("max-grad-norm", float), ("ce-weight", float),
                       ("weighted-ce-weight", float), ("null-loss-weight", float), ("null-pos-weight-cap", float),
                       ("null-focal-gamma", float), ("asl-gamma-positive", float), ("asl-gamma-negative", float),
                       ("asl-margin", float), ("negative-ratio", float), ("minimum-negatives", int),
                       ("hard-fraction", float), ("attention-size", int), ("amp-initial-scale", float)):
        p.add_argument("--" + name, type=kind)
    p.add_argument("--null-thresholds", nargs="+", type=float, help="Development grid; abstention is always included")
    return p


def settings(args):
    fields = {k: getattr(args, k) for k in NullConfig.__dataclass_fields__ if hasattr(args, k) and getattr(args, k) is not None}
    fields["epochs"] = 1 if args.smoke else (args.epochs if args.step == "custom" else args.screen_epochs)
    if "null_thresholds" in fields:
        fields["null_thresholds"] = tuple(fields["null_thresholds"])
    if args.step == "custom":
        presets = {"bce": {}, "focal": {"null_loss": "focal"}, "asl": {"null_loss": "asl"},
                   "ia": {"representation": "ia"}, "attention": {"representation": "ia_attention"},
                   "hard": {"representation": "ia_attention", "negative_sampling": "hard"},
                   "null_off": {"null_head": False}, "custom": {}}
        fields = {"name": args.recipe, **presets[args.recipe], **fields}
    cfg = NullConfig(**fields)
    cfg.validate()
    return cfg


def validate_args(args):
    if args.first_seeds is not None and args.seeds is not None:
        raise ValueError("Use --seeds or --first-seeds, not both")
    args.first_seeds = args.first_seeds if args.first_seeds is not None else (1 if args.step == "custom" else 2)
    if not 1 <= args.first_seeds <= len(config.SEEDS):
        raise ValueError("--first-seeds must be between 1 and 5")
    if args.seeds is not None and (args.step != "custom" or len(set(args.seeds)) != len(args.seeds)
                                  or any(not 0 <= seed < 2**32 for seed in args.seeds)):
        raise ValueError("Unique nonnegative --seeds are available for custom steps only")
    if min(args.screen_epochs, args.epochs, args.train_limit, args.dev_limit) < 1:
        raise ValueError("Epoch/split limits must be positive")
    if args.smoke and args.step == "final":
        raise ValueError("Smoke studies never evaluate test data")
    if args.smoke:
        args.epochs = 1
        args.screen_epochs = 1
    if args.step != "custom" and any(getattr(args, k) is not None for k in ("null_loss", "representation", "negative_sampling")):
        raise ValueError("Use --step custom for a manually chosen NULL loss/representation/sampling combination")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        validate_args(args)
        cfg = settings(args)
    except ValueError as exc:
        p.error(str(exc))
    output = args.output.resolve()
    share = args.share_report or ROOT / "null_fast_track_reports" / output.name
    print(f"{VERSION}; step={args.step}; {'SMOKE ONLY' if args.smoke else 'exploratory research'}")
    print("Test access: frozen evaluation only" if args.step == "final" else "Test access: disabled")
    if args.step == "report":
        from null_experiments.reporting import write_report
        print(write_report(output, share, args.smoke))
        return
    print(f"Settings: mixed CE={cfg.ce_weight:g}/{cfg.weighted_ce_weight:g}; LR={cfg.lr:g}; batch={cfg.batch_size}; "
          f"max_len={cfg.max_len}; precision={cfg.precision}; selection={cfg.selection_metric}")
    if args.step == "custom":
        seeds = args.seeds or config.SEEDS[:args.first_seeds]
        print(f"Custom: {cfg.name}; seeds={seeds}; epochs={cfg.epochs}; NULL loss={cfg.null_loss}; "
              f"representation={cfg.representation}; sampling={cfg.negative_sampling}; id={cfg.trial_id}")
    elif args.step in ("all", "search"):
        candidates = 8 + (2 + int(args.tune_lr) if args.tune else 0)
        confirmation = (args.confirm_top + 1) * args.first_seeds * args.epochs if args.step == "all" else 0
        print(f"Adaptive search: {candidates} short-budget runs on seed 13, {cfg.epochs} epochs each.")
        print(f"Confirmation: {args.confirm_top} NULL candidate(s) + NULL-off control, first {args.first_seeds} seeds, {args.epochs} epochs.")
        print(f"Maximum planned budget: {candidates * cfg.epochs + confirmation} model-epochs; verified completed runs skip.")
    print(f"Output: {output}")
    if not args.execute or args.dry_run:
        print("Plan only. Add --execute to run; --step final is required to open test data.")
        return
    from null_experiments.training import Runtime
    from null_experiments.storage import study_lock
    from null_experiments.reporting import write_report
    from null_experiments.study import calibrate_legacy, search, confirm, final_evaluation
    try:
        with study_lock(output):
            runtime = Runtime(ROOT, output, args.device, args.smoke, args.train_limit, args.dev_limit)
            refresh = lambda: write_report(output, share, args.smoke)
            if args.step in ("all", "calibrate") and not args.skip_calibration:
                if args.legacy_selection.exists():
                    calibrate_legacy(runtime, args.legacy_selection)
                    refresh()
                elif args.step == "calibrate":
                    raise ValueError("Legacy selection is missing; supply --legacy-selection")
                else:
                    print("Legacy checkpoint freeze unavailable; calibration remains enabled for every new model.", flush=True)
            if args.step in ("all", "search"):
                state = search(runtime, cfg, args, refresh)
            if args.step in ("all", "confirm"):
                if args.step == "confirm":
                    state = json.loads((output / "search_state.json").read_text(encoding="utf-8"))
                    if "selected_null" not in state:
                        raise ValueError("Search is not complete; run --step search first")
                confirm(runtime, state, args, refresh)
            elif args.step == "final":
                frozen = json.loads((output / "selection.json").read_text(encoding="utf-8"))
                final_evaluation(runtime, frozen, refresh)
            elif args.step == "custom":
                for seed in args.seeds or config.SEEDS[:args.first_seeds]:
                    runtime.run(cfg, seed, "custom", args.retry_failed)
                    refresh()
            print(f"Finished. Report: {refresh()}")
            if not args.smoke:
                print(f"Small shareable report: {share}")
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
