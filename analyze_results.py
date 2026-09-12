"""Summarize completed runs and compare losses on exactly matched seeds."""

import argparse
from pathlib import Path

import pandas as pd

import config as cfg
from src.stats import (
    bootstrap_ci,
    paired_significance_test,
    passes_min_effect,
    summarize,
)


DEFAULT_RESULTS = Path(__file__).resolve().parent / "results.csv"


def analyze(metric_col: str = "test_micro_f1", results_csv=DEFAULT_RESULTS):
    results_csv = Path(results_csv)
    if not results_csv.exists():
        print(f"No results file found at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    required = {"mode", "config", "seed", metric_col}
    missing = required - set(df.columns)
    if missing:
        print(
            f"Cannot analyze {metric_col!r}; missing columns: {sorted(missing)}. "
            "Run the new train/validation/test pipeline first."
        )
        return

    if "status" in df.columns:
        df = df[df["status"] == "complete"]
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df[df[metric_col].notna()].copy()
    if df.empty:
        print(f"No completed numeric {metric_col} values are available.")
        return

    key_columns = ["mode", "config", "seed"]
    duplicate_mask = df.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, key_columns].to_dict("records")
        raise ValueError(f"Duplicate run keys in results.csv: {duplicates}")

    for mode in cfg.MODES:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            continue
        print(f"\n=== {mode} — {metric_col} ===")

        for config_name in sorted(mode_df["config"].unique()):
            values = mode_df.loc[
                mode_df["config"] == config_name, metric_col,
            ].astype(float).tolist()
            print(f"{config_name}: {summarize(values)}")

        standard = mode_df[mode_df["config"] == "standard"][
            ["seed", metric_col]
        ].rename(columns={metric_col: "standard"})
        weighted = mode_df[mode_df["config"] == "weighted"][
            ["seed", metric_col]
        ].rename(columns={metric_col: "weighted"})
        paired = standard.merge(
            weighted, on="seed", how="inner", validate="one_to_one",
        ).sort_values("seed")

        missing_standard = sorted(set(weighted["seed"]) - set(standard["seed"]))
        missing_weighted = sorted(set(standard["seed"]) - set(weighted["seed"]))
        if missing_standard or missing_weighted:
            print(
                "unmatched seeds: "
                f"missing standard={missing_standard}, missing weighted={missing_weighted}"
            )
        if len(paired) < 2:
            print("paired comparison requires at least two matched seeds")
            continue

        standard_scores = paired["standard"].astype(float).tolist()
        weighted_scores = paired["weighted"].astype(float).tolist()
        significance = paired_significance_test(
            standard_scores, weighted_scores,
        )
        interval = bootstrap_ci(standard_scores, weighted_scores)
        effect_ok = passes_min_effect(
            significance["mean_diff"], cfg.MIN_EFFECT_SIZE,
        )
        # Report the seeds that have results for both standard and weighted configs.
        print(f"matched seeds: {paired['seed'].astype(int).tolist()}")

        # Print the paired result for each seed so that the comparison is auditable.
        # The difference is calculated as weighted - standard.
        for _, row in paired.iterrows():
            seed = int(row["seed"])
            standard_score = float(row["standard"])
            weighted_score = float(row["weighted"])
            diff = weighted_score - standard_score

            print(
                f"seed {seed}: "
                f"standard {standard_score:.4f}, "
                f"weighted {weighted_score:.4f}, "
                f"diff {diff:+.4f}"
            )

        # Check whether any planned experimental seeds are missing from the
        # paired comparison and warn the user before interpreting the results.
        matched_seeds = set(paired["seed"].astype(int).tolist())
        missing_planned_seeds = sorted(set(cfg.SEEDS) - matched_seeds)

        if missing_planned_seeds:
            print(f"WARNING: missing planned seeds: {missing_planned_seeds}")

        print(
            f"mean diff (weighted - standard): "
            f"{significance['mean_diff']:.4f}"
        )
        print(f"paired t-test p-value: {significance['p_value']:.4f}")
        print(
            f"bootstrap 95% CI: "
            f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
        )
        print(
            f"minimum effect {cfg.MIN_EFFECT_SIZE:.4f}: "
            f"{'PASSES' if effect_ok else 'does NOT pass'}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="test_micro_f1")
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()
    analyze(args.metric, args.results_csv)
