"""Analyze a complete or partial leave-one-domain-out experiment matrix.

The script never changes raw run artifacts or ``results_lodo.csv``. It writes
derived, reproducible interim summaries and marks a domain as final-ready only
when both configurations contain every planned seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config as cfg
from src.result_analysis import derive_crossdomain_metrics, load_prediction_file
from src.stats import bootstrap_ci, paired_significance_test, passes_min_effect


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = PROJECT_ROOT / "results_lodo.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "lodo_analysis"
MEASURE_COLUMNS = [
    "test_micro_precision",
    "test_micro_recall",
    "test_micro_f1",
    "test_macro_f1",
    "test_rare_category_recall",
]
DERIVED_COLUMNS = [
    "known_category_f1",
    "aspect_f1",
    "aspect_sentiment_f1",
    "category_coverage",
    "gold_triplets",
    "source_known_gold_triplets",
    "unseen_gold_triplets",
    "unseen_category_count",
    "aspect_missed",
    "category_wrong",
    "sentiment_wrong",
    "extra_predicted",
]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _atomic_write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No LODO results file found at {path}")
    frame = pd.read_csv(path)
    required = {
        "held_out_domain", "config", "seed", "status", "artifact_dir",
        *MEASURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")
    frame = frame[frame["status"] == "complete"].copy()
    keys = ["held_out_domain", "config", "seed"]
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate completed LODO keys: "
            f"{frame.loc[duplicates, keys].to_dict('records')}"
        )
    for column in ["seed", *MEASURE_COLUMNS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def expected_and_missing(frame, domains, configs, seeds):
    expected = {
        (domain, config_name, int(seed))
        for domain in domains
        for config_name in configs
        for seed in seeds
    }
    actual = {
        (str(row.held_out_domain), str(row.config), int(row.seed))
        for row in frame.itertuples()
    }
    unexpected = sorted(actual - expected)
    if unexpected:
        raise ValueError(f"Unexpected completed LODO keys: {unexpected}")
    return expected, sorted(expected - actual)


def enrich_runs(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = []
    for row in frame.to_dict("records"):
        run_dir = Path(row["artifact_dir"])
        manifest_path = run_dir / "manifest.json"
        predictions_path = run_dir / "test_predictions.jsonl"
        metrics_path = run_dir / "metrics.json"
        for required_path in (manifest_path, predictions_path, metrics_path):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Completed row is missing artifact: {required_path}"
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        gold, predicted = load_prediction_file(predictions_path)
        derived = derive_crossdomain_metrics(
            gold, predicted, manifest["category_vocab"],
        )
        preliminary_errors = metrics["test"]["error_analysis"]["overall"]
        enriched.append(
            {
                **row,
                "known_category_f1": derived["source_known_exact_triplet"]["f1"],
                "aspect_f1": derived["aspect_span"]["f1"],
                "aspect_sentiment_f1": derived["aspect_plus_sentiment"]["f1"],
                "category_coverage": derived["gold_category_coverage"],
                "gold_triplets": derived["gold_triplets"],
                "source_known_gold_triplets": derived["source_known_gold_triplets"],
                "unseen_gold_triplets": derived["unseen_gold_triplets"],
                "unseen_category_count": derived["unseen_category_count"],
                **preliminary_errors,
            }
        )
    return pd.DataFrame(enriched)


def build_domain_summary(frame, domains, configs, seeds):
    rows = []
    planned_seed_count = len(seeds)
    for domain in domains:
        for config_name in configs:
            group = frame[
                (frame["held_out_domain"] == domain)
                & (frame["config"] == config_name)
            ]
            row = {
                "held_out_domain": domain,
                "config": config_name,
                "completed_seeds": len(group),
                "planned_seeds": planned_seed_count,
                "status": "complete" if len(group) == planned_seed_count else "partial",
            }
            for column in [*MEASURE_COLUMNS, *DERIVED_COLUMNS]:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[f"{column}_mean"] = float(values.mean()) if len(values) else None
                row[f"{column}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    if len(values) == 1 else None
                )
            rows.append(row)
    return rows


def build_paired_summary(frame, domains, seeds):
    rows = []
    planned_seed_count = len(seeds)
    for domain in domains:
        domain_frame = frame[frame["held_out_domain"] == domain]
        paired = domain_frame.pivot(
            index="seed", columns="config", values="test_micro_f1",
        )
        if not {"standard", "weighted"}.issubset(paired.columns):
            matched = paired.iloc[0:0]
        else:
            matched = paired[["standard", "weighted"]].dropna().sort_index()
        differences = (
            matched["weighted"] - matched["standard"]
            if len(matched) else pd.Series(dtype=float)
        )
        row = {
            "held_out_domain": domain,
            "matched_seeds": len(matched),
            "planned_seeds": planned_seed_count,
            "status": "complete" if len(matched) == planned_seed_count else "partial",
            "seed_ids": ";".join(str(int(seed)) for seed in matched.index),
            "standard_mean_f1": float(matched["standard"].mean()) if len(matched) else None,
            "weighted_mean_f1": float(matched["weighted"].mean()) if len(matched) else None,
            "mean_difference": float(differences.mean()) if len(matched) else None,
            "std_difference": (
                float(differences.std(ddof=1)) if len(matched) > 1 else 0.0
                if len(matched) == 1 else None
            ),
            "p_value": None,
            "ci_low": None,
            "ci_high": None,
            "passes_positive_0_02_effect": False,
        }
        if len(matched) >= 2:
            standard = matched["standard"].tolist()
            weighted = matched["weighted"].tolist()
            interval = bootstrap_ci(standard, weighted)
            # A paired t-test is undefined when every paired difference is
            # exactly zero. Preserve the zero bootstrap interval and report no
            # p-value instead of serializing NaN.
            significance = (
                None if bool((differences == 0).all())
                else paired_significance_test(standard, weighted)
            )
            row.update({
                "p_value": (
                    significance["p_value"] if significance is not None else None
                ),
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "passes_positive_0_02_effect": passes_min_effect(
                    float(differences.mean()), cfg.MIN_EFFECT_SIZE,
                ),
            })
        rows.append(row)
    return rows


def _number(value, digits=3):
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def build_markdown_report(
    completed_count,
    expected_count,
    missing,
    domain_summary,
    paired_summary,
):
    complete_domains = [
        row["held_out_domain"] for row in paired_summary
        if row["status"] == "complete"
    ]
    partial_domains = [
        row["held_out_domain"] for row in paired_summary
        if row["status"] != "complete"
    ]
    lines = [
        "# Interim LODO Result Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Completion status",
        "",
        f"- Completed runs: **{completed_count}/{expected_count} "
        f"({100 * completed_count / expected_count:.1f}%)**",
        f"- Complete five-seed domains: {', '.join(complete_domains) or 'none'}",
        f"- Partial domains: {', '.join(partial_domains) or 'none'}",
        f"- Missing logical run keys: {len(missing)}",
        "",
        "## Paired exact-triplet F1",
        "",
        "| Held-out domain | Status | Matched seeds | Standard F1 | Weighted F1 | Difference | 95% bootstrap CI | p-value |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired_summary:
        interval = (
            f"[{_number(row['ci_low'])}, {_number(row['ci_high'])}]"
            if row["ci_low"] is not None else "—"
        )
        lines.append(
            f"| {row['held_out_domain']} | {row['status']} | "
            f"{row['matched_seeds']}/{row['planned_seeds']} | "
            f"{_number(row['standard_mean_f1'])} | "
            f"{_number(row['weighted_mean_f1'])} | "
            f"{_number(row['mean_difference'], 4)} | {interval} | "
            f"{_number(row['p_value'], 4)} |"
        )
    lines.extend(
        [
            "",
            "Partial-domain statistics are descriptive only. Do not compare a "
            "single weighted seed with five standard seeds.",
            "",
            "## Configuration-level metrics",
            "",
            "| Domain | Config | Status | n | Precision | Recall | F1 | Aspect F1 | Aspect+sentiment F1 | Category coverage |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in domain_summary:
        lines.append(
            f"| {row['held_out_domain']} | {row['config']} | {row['status']} | "
            f"{row['completed_seeds']} | "
            f"{_number(row['test_micro_precision_mean'])} | "
            f"{_number(row['test_micro_recall_mean'])} | "
            f"{_number(row['test_micro_f1_mean'])} | "
            f"{_number(row['aspect_f1_mean'])} | "
            f"{_number(row['aspect_sentiment_f1_mean'])} | "
            f"{_number(100 * row['category_coverage_mean'], 1)}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is a live interim analysis. Only complete five-seed domain pairs support the planned paired comparison.",
            "- Error counts are the pipeline's existing preliminary counts; they are not yet the planned mutually exclusive gold-triplet taxonomy.",
            "- Category coverage is the percentage of held-out gold triplets whose category exists in source training.",
            "- These runs still use zero warmup and no gradient clipping, so they remain post-pilot evidence unless a protocol amendment accepts those settings.",
            "- `NULL` aspects remain outside the explicit-span model and are not included in the restricted-task F1.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--domains", nargs="+",
        default=[domain for domain in cfg.DOMAINS if domain != cfg.HOLD_OUT_DOMAIN],
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=[item.name for item in cfg.EXPERIMENTS],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    args = parser.parse_args(argv)

    frame = load_results(args.results_csv)
    expected, missing = expected_and_missing(
        frame, args.domains, args.configs, args.seeds,
    )
    enriched = enrich_runs(frame)
    domain_summary = build_domain_summary(
        enriched, args.domains, args.configs, args.seeds,
    )
    paired_summary = build_paired_summary(enriched, args.domains, args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_fields = list(enriched.columns)
    _atomic_write_csv(
        args.output_dir / "interim_run_metrics.csv",
        enriched.to_dict("records"),
        run_fields,
    )
    _atomic_write_csv(
        args.output_dir / "interim_domain_summary.csv",
        domain_summary,
        list(domain_summary[0]),
    )
    _atomic_write_csv(
        args.output_dir / "interim_paired_summary.csv",
        paired_summary,
        list(paired_summary[0]),
    )
    missing_rows = [
        {"held_out_domain": domain, "config": config_name, "seed": seed}
        for domain, config_name, seed in missing
    ]
    _atomic_write_csv(
        args.output_dir / "missing_runs.csv",
        missing_rows,
        ["held_out_domain", "config", "seed"],
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "completed_runs": len(frame),
        "expected_runs": len(expected),
        "completion_percent": 100 * len(frame) / len(expected),
        "missing_runs": missing_rows,
        "domain_summary": domain_summary,
        "paired_summary": paired_summary,
        "status": "complete" if not missing else "interim",
    }
    _atomic_write_json(args.output_dir / "interim_summary.json", payload)
    report = build_markdown_report(
        len(frame), len(expected), missing, domain_summary, paired_summary,
    )
    _atomic_write_text(args.output_dir / "INTERIM_REPORT.md", report)

    print(report)
    print(f"Analysis artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    main()
