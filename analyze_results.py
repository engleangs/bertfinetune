"""
Turns raw per-seed rows in results.csv into the table ,   mean +/- std, significance test, and an explicit
pass/fail against the pre-registered MIN_EFFECT_SIZE.
Run this AFTER run_all_study.py, and after wiring in real micro_f1/macro_f1/
rare_label_recall columns (see run_study.py's TODO).
"""
import pandas as pd

import config as cfg
from src.stats import summarize, paired_significance_test, bootstrap_ci, passes_min_effect


def analyze(metric_col: str = "micro_f1"):
    df = pd.read_csv("results.csv")

    if metric_col not in df.columns:
        print(f"'{metric_col}' not in results.csv yet — this file only has "
              f"{list(df.columns)}. Wire up real evaluation in run_study.py's "
              f"TODO first (see its comment block), then re-run.")
        return

    for mode in cfg.MODES:
        mode_df = df[df["mode"] == mode]
        if mode_df.empty:
            continue

        standard = mode_df[mode_df["config"] == "standard"].sort_values("seed")[metric_col].tolist()
        weighted = mode_df[mode_df["config"] == "weighted"].sort_values("seed")[metric_col].tolist()

        if len(standard) != len(weighted) or len(standard) < 2:
            print(f"[{mode}] need matched seeds for both configs (>=2) — have "
                  f"{len(standard)} standard, {len(weighted)} weighted. Skipping.")
            continue

        print(f"\n=== {mode} — {metric_col} ===")
        print(f"standard: {summarize(standard)}")
        print(f"weighted: {summarize(weighted)}")

        sig = paired_significance_test(standard, weighted)
        ci = bootstrap_ci(standard, weighted)
        effect_ok = passes_min_effect(sig["mean_diff"], cfg.MIN_EFFECT_SIZE)

        print(f"mean diff (weighted - standard): {sig['mean_diff']:.4f}")
        print(f"paired t-test p-value: {sig['p_value']:.4f} "
              f"({'significant' if sig['significant_at_0.05'] else 'NOT significant'} at 0.05)")
        print(f"bootstrap 95% CI on diff: [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] "
              f"({'excludes zero' if ci['excludes_zero'] else 'includes zero'})")
        print(f"pre-registered min effect size ({cfg.MIN_EFFECT_SIZE}): "
              f"{'PASSES' if effect_ok else 'does NOT pass'}")

        if sig["significant_at_0.05"] and not effect_ok:
            print("  -> Statistically significant but below your pre-registered bar. "
                  "Report both numbers honestly — this is a real, defensible finding "
                  "either way, exactly what the professor asked for.")


if __name__ == "__main__":
    analyze("micro_f1")
    # TODO (S5): also call analyze("macro_f1") and a rare-label-recall variant
    # once those columns exist in results.csv.
