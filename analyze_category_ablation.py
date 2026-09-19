"""Compare category-only weighting with matched primary-study runs.

The comparison is exploratory. Scores are paired by the same random seed and
reported only for seeds present in all three configurations.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import config as cfg


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ABLATION_RESULTS = PROJECT_ROOT / "results_category_ablation.csv"
METRICS = ("test_micro_precision", "test_micro_recall", "test_micro_f1")
CONFIGS = ("standard", "weighted", "category_weighted")


def load_scores(path, mode, held_out_domain, selected_configs, legacy_holdout=None):
    """Read completed rows and reject duplicate keys or malformed scores."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"mode", "config", "seed", "status", *METRICS}
        if mode == "crossdomain" and legacy_holdout is None:
            required.add("held_out_domain")
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

        scores = {name: {} for name in selected_configs}
        for row in reader:
            if row["status"] != "complete" or row["mode"] != mode:
                continue
            row_domain = row.get("held_out_domain") or legacy_holdout
            if mode == "crossdomain" and row_domain != held_out_domain:
                continue
            name = row["config"]
            if name not in scores:
                continue
            seed = int(row["seed"])
            if seed in scores[name]:
                raise ValueError(f"Duplicate completed key: {mode}/{held_out_domain}/{name}/{seed}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
                raise ValueError(f"Invalid precision, recall, or F1: {name}/{seed}")
            scores[name][seed] = values
    return scores


def summarize(reference, ablation, planned_seeds):
    """Produce paired summaries without treating a partial matrix as final."""
    combined = {**reference, **ablation}
    planned = set(planned_seeds)
    if set(combined) != set(CONFIGS):
        raise ValueError(f"Expected configurations: {CONFIGS}")
    unexpected = {name: sorted(set(rows) - planned) for name, rows in combined.items()}
    if any(unexpected.values()):
        raise ValueError(f"Unexpected seeds: {unexpected}")
    missing = {name: sorted(planned - set(rows)) for name, rows in combined.items()}
    matched = sorted(set.intersection(*(set(rows) for rows in combined.values())))

    metrics = {}
    for metric in METRICS:
        means = {
            name: statistics.mean(combined[name][seed][metric] for seed in matched)
            for name in CONFIGS
        } if matched else {}
        metrics[metric] = {
            "means_on_matched_seeds": means,
            "category_minus_standard": (
                statistics.mean(
                    combined["category_weighted"][seed][metric]
                    - combined["standard"][seed][metric]
                    for seed in matched
                ) if matched else None
            ),
            "category_minus_all_head_weighted": (
                statistics.mean(
                    combined["category_weighted"][seed][metric]
                    - combined["weighted"][seed][metric]
                    for seed in matched
                ) if matched else None
            ),
        }
    return {
        "status": "complete" if not any(missing.values()) else "partial",
        "planned_seeds": sorted(planned),
        "matched_seeds": matched,
        "missing_seeds": missing,
        "metrics": metrics,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=cfg.MODES, default="indomain")
    parser.add_argument("--held-out-domain", choices=cfg.DOMAINS)
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--ablation-results", type=Path, default=DEFAULT_ABLATION_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "crossdomain" and args.held_out_domain is None:
        parser.error("--held-out-domain is required for crossdomain analysis")
    if args.mode == "indomain" and args.held_out_domain is not None:
        parser.error("--held-out-domain is valid only for crossdomain analysis")

    reference_path = args.reference_results or (
        PROJECT_ROOT / ("results.csv" if args.mode == "indomain" or args.held_out_domain == cfg.HOLD_OUT_DOMAIN else "results_lodo.csv")
    )
    reference = load_scores(
        reference_path, args.mode, args.held_out_domain, ("standard", "weighted"),
        legacy_holdout=(
            cfg.HOLD_OUT_DOMAIN
            if args.mode == "crossdomain"
            and args.held_out_domain == cfg.HOLD_OUT_DOMAIN
            and reference_path.name == "results.csv"
            else None
        ),
    )
    ablation = load_scores(
        args.ablation_results, args.mode, args.held_out_domain,
        ("category_weighted",),
    )
    result = summarize(reference, ablation, cfg.SEEDS)
    result.update({"mode": args.mode, "held_out_domain": args.held_out_domain})

    output = args.output or (
        PROJECT_ROOT / "artifacts" / "category_ablation_analysis" / args.mode
        / (args.held_out_domain or "all-domains") / "summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    f1 = result["metrics"]["test_micro_f1"]
    print(f"Ablation status: {result['status']}; matched seeds: {result['matched_seeds']}")
    print(f"Missing seeds: {result['missing_seeds']}")
    print(f"Mean exact-triplet F1: {f1['means_on_matched_seeds']}")
    print(f"Category-only minus standard: {f1['category_minus_standard']}")
    print(f"Category-only minus all-head weighted: {f1['category_minus_all_head_weighted']}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    main()
