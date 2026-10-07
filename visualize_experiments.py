"""Compare saved development trials, visualize progress and show actual settings."""

import argparse
from pathlib import Path
import time

from experiment_reporting.analysis import build_analysis
from experiment_reporting.plots import make_figures
from experiment_reporting.report import headline, write_report
from src.experiment_config import candidate_recipes

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "artifacts" / "experiments_v2")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experiment_comparison")
    p.add_argument("--suite", choices=("all", "loss", "heads", "null", "baseline", "custom"), default="all",
                   help="Expected recipes for the progress grid; reads all saved trials in the selected mode/fold")
    p.add_argument("--expected-seeds", nargs="+", type=int, default=[13, 42])
    p.add_argument("--mode", choices=("indomain", "crossdomain"), default="indomain")
    p.add_argument("--held-out-domain", default=None)
    p.add_argument("--minimum-effect", type=float, default=.02, help="Absolute F1 target; .02 means two percentage points")
    p.add_argument("--include-smoke", action="store_true", help="Show smoke cohorts separately; excluded by default")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--watch", action="store_true", help="Regenerate when saved artifacts change; stop with Ctrl+C")
    p.add_argument("--interval", type=float, default=30, help="Seconds between change checks in watch mode")
    return p


def snapshot_signature(root):
    return [(str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.glob("**/*.json"))
            if p.name in {"manifest.json", "history.json", "dev_metrics.json", "audit.json"}]


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.input.is_dir():
        parser().error(f"Input directory does not exist: {args.input}")
    if args.mode == "crossdomain" and args.held_out_domain is None:
        parser().error("--mode crossdomain requires --held-out-domain")
    if args.mode == "indomain" and args.held_out_domain is not None:
        parser().error("--held-out-domain applies only to crossdomain reports")
    if args.interval < 1 or args.dpi < 50 or not 0 <= args.minimum_effect <= 1:
        parser().error("Invalid refresh interval, DPI or effect target")
    recipes = [r["name"] for r in candidate_recipes(args.suite)]
    previous = None
    try:
        while True:
            signature = snapshot_signature(args.input)
            if signature != previous:
                analysis = build_analysis(args.input, args.mode, args.held_out_domain, recipes, args.expected_seeds,
                                          args.minimum_effect, args.include_smoke)
                figures = make_figures(analysis, args.output, args.dpi)
                write_report(analysis, figures, args.output)
                print(f"{analysis['generated_at']}: {analysis['completed_runs']} finished runs; {headline(analysis)}", flush=True)
                print(f"Open {args.output.resolve() / 'report.html'}", flush=True)
                previous = signature
            if not args.watch:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped refreshing. The last report is saved.")


if __name__ == "__main__":
    main()
