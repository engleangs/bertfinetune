"""Create protocol-aware presentation figures from completed run artifacts.

Usage:
    python visualize_results.py
    python visualize_results.py --results-csv results.csv --output-dir artifacts/result_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

import config as cfg
from src.result_analysis import derive_crossdomain_metrics, load_prediction_file, validate_complete_matrix
from src.stats import bootstrap_ci, paired_significance_test, passes_min_effect, summarize


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "result_analysis"
CONFIGS = ("standard", "weighted")
NAVY = "#16324F"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
CORAL = "#E76F51"
GOLD = "#E9C46A"
GREY = "#667085"
PALE = "#EEF4F8"
CONFIG_COLORS = {"standard": BLUE, "weighted": CORAL}


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titleweight": "bold", "axes.titlesize": 15,
        "axes.labelcolor": NAVY, "axes.edgecolor": "#D0D5DD",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GREY, "ytick.color": GREY,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def add_title(fig, title, subtitle):
    fig.text(0.055, 0.955, title, ha="left", va="top", fontsize=24, weight="bold", color=NAVY)
    fig.text(0.055, 0.895, subtitle, ha="left", va="top", fontsize=11.5, color=GREY)


def add_footer(fig):
    fig.text(0.055, 0.025, "Observed saved artifacts • Protocol deviations are documented in PROTOCOL_REVIEW.md", color="#98A2B3", fontsize=8.5)


def save_figure(fig, output_dir, filename, dpi):
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def paired_frame(df, mode, metric="test_micro_f1"):
    part = df[df["mode"] == mode]
    standard = part[part["config"] == "standard"][["seed", metric]].rename(columns={metric: "standard"})
    weighted = part[part["config"] == "weighted"][["seed", metric]].rename(columns={metric: "weighted"})
    return standard.merge(weighted, on="seed", validate="one_to_one").sort_values("seed")


def comparison_summary(df, mode):
    paired = paired_frame(df, mode)
    standard = paired["standard"].astype(float).tolist()
    weighted = paired["weighted"].astype(float).tolist()
    significance = paired_significance_test(standard, weighted)
    interval = bootstrap_ci(standard, weighted)
    return {
        "standard": summarize(standard),
        "weighted": summarize(weighted),
        "paired": paired.to_dict("records"),
        "mean_difference": significance["mean_diff"],
        "sd_difference": significance["std_diff"],
        "t_statistic": significance["t_stat"],
        "p_value": significance["p_value"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "passes_minimum_effect": passes_min_effect(significance["mean_diff"], cfg.MIN_EFFECT_SIZE),
    }


def artifact_dir(row):
    path = Path(row.artifact_dir)
    if path.is_dir():
        return path
    return ROOT / "artifacts" / "runs" / row.mode / row.config / f"seed_{int(row.seed)}"


def load_artifact_details(df):
    histories, cross_rows, errors = {}, [], []
    for row in df.itertuples():
        directory = artifact_dir(row)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        histories[(row.mode, row.config, int(row.seed))] = json.loads((directory / "history.json").read_text(encoding="utf-8"))
        error = metrics["test"]["error_analysis"]["overall"]
        errors.append({"mode": row.mode, "config": row.config, "seed": int(row.seed), **error})
        if row.mode == "crossdomain":
            gold, predicted = load_prediction_file(directory / "test_predictions.jsonl")
            derived = derive_crossdomain_metrics(gold, predicted, manifest["category_vocab"])
            cross_rows.append({
                "mode": row.mode, "config": row.config, "seed": int(row.seed),
                "full_exact_f1": derived["full_exact_triplet"]["f1"],
                "known_category_f1": derived["source_known_exact_triplet"]["f1"],
                "aspect_f1": derived["aspect_span"]["f1"],
                "aspect_sentiment_f1": derived["aspect_plus_sentiment"]["f1"],
                "category_coverage": derived["gold_category_coverage"],
                "gold_triplets": derived["gold_triplets"],
                "known_gold_triplets": derived["source_known_gold_triplets"],
                "unseen_gold_triplets": derived["unseen_gold_triplets"],
                "unseen_category_count": derived["unseen_category_count"],
                "unseen_categories": derived["unseen_categories"],
            })
    return histories, pd.DataFrame(cross_rows), pd.DataFrame(errors)


def draw_paired(ax, paired, title):
    for row in paired.itertuples():
        ax.plot([0, 1], [row.standard, row.weighted], color="#B8C7D1", linewidth=1.5, zorder=1)
        ax.scatter(0, row.standard, color=BLUE, s=65, zorder=2)
        ax.scatter(1, row.weighted, color=CORAL, s=65, zorder=2)
        ax.text(1.04, row.weighted, str(int(row.seed)), va="center", fontsize=8.5, color=GREY)
    ax.set_xticks([0, 1], ["Standard", "Weighted"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("Exact-triplet micro-F1")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#E4E7EC")
    ax.set_axisbelow(True)
    ax.text(1.04, ax.get_ylim()[1], "seed", va="top", fontsize=8, color="#98A2B3")


def summary_card(ax, comparison, heading, verdict, verdict_color):
    ax.axis("off")
    card = FancyBboxPatch((0.02, 0.03), 0.96, 0.94, boxstyle="round,pad=0.025,rounding_size=0.03", facecolor=PALE, edgecolor="#D9E5ED", transform=ax.transAxes)
    ax.add_patch(card)
    ax.text(0.08, 0.88, heading, transform=ax.transAxes, fontsize=15, weight="bold", color=NAVY)
    ax.text(0.08, 0.72, f"Standard  {comparison['standard']['mean']:.3f} ± {comparison['standard']['std']:.3f}", transform=ax.transAxes, fontsize=14, color=BLUE, weight="bold")
    ax.text(0.08, 0.61, f"Weighted  {comparison['weighted']['mean']:.3f} ± {comparison['weighted']['std']:.3f}", transform=ax.transAxes, fontsize=14, color=CORAL, weight="bold")
    ax.text(0.08, 0.45, f"Mean paired difference  {comparison['mean_difference']:+.4f}", transform=ax.transAxes, fontsize=13, color=NAVY)
    ax.text(0.08, 0.35, f"95% bootstrap CI  [{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]", transform=ax.transAxes, fontsize=11, color=GREY)
    ax.text(0.08, 0.27, f"Paired t-test  p = {comparison['p_value']:.4f}", transform=ax.transAxes, fontsize=11, color=GREY)
    ax.text(0.08, 0.11, verdict, transform=ax.transAxes, fontsize=13, weight="bold", color=verdict_color)


def plot_primary(df, comparison, output_dir, dpi):
    paired = paired_frame(df, "indomain")
    fig, (ax_pair, ax_card) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.08, right=0.95, top=0.80, bottom=0.12, wspace=0.25)
    add_title(fig, "Class weighting reduced the confirmatory in-domain score", "Five matched seeds; exact-triplet micro-F1 on the official in-domain test splits")
    draw_paired(ax_pair, paired, "Every paired seed moved downward")
    summary_card(ax_card, comparison, "Confirmatory comparison", "Does not meet the +0.02 improvement criterion", CORAL)
    add_footer(fig)
    save_figure(fig, output_dir, "slide_01_primary_indomain_outcome.png", dpi)


def plot_crossdomain(df, comparison, cross, output_dir, dpi):
    paired = paired_frame(df, "crossdomain")
    fig, (ax_pair, ax_metrics) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [0.9, 1.35]})
    fig.subplots_adjust(left=0.08, right=0.96, top=0.74, bottom=0.15, wspace=0.28)
    add_title(fig, "Weighting improved the secondary restaurant hold-out result", "Exploratory cross-domain result; full score remains constrained by unseen categories")
    draw_paired(ax_pair, paired, "Full exact-triplet F1 by seed")
    metrics = [
        ("full_exact_f1", "Full exact triplet"),
        ("known_category_f1", "Known-category triplet"),
        ("aspect_f1", "Aspect span"),
        ("aspect_sentiment_f1", "Aspect + sentiment"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    for offset, config in ((-width / 2, "standard"), (width / 2, "weighted")):
        means = [cross[cross["config"] == config][column].mean() for column, _ in metrics]
        stds = [cross[cross["config"] == config][column].std(ddof=1) for column, _ in metrics]
        bars = ax_metrics.bar(x + offset, means, width, yerr=stds, capsize=4, color=CONFIG_COLORS[config], label=config.title())
        for bar, value, std in zip(bars, means, stds):
            ax_metrics.text(bar.get_x() + bar.get_width() / 2, value + std + 0.008, f"{value:.3f}", ha="center", fontsize=8.5, color=NAVY)
    coverage = cross["category_coverage"].iloc[0]
    ax_metrics.set_xticks(x, [label for _, label in metrics], rotation=16, ha="right")
    ax_metrics.set_ylabel("Mean F1 across seeds")
    ax_metrics.set_title(f"Protocol views • source vocabulary covers {coverage:.1%} of gold triplets", loc="left")
    ax_metrics.grid(axis="y", color="#E4E7EC")
    ax_metrics.set_axisbelow(True)
    ax_metrics.legend(frameon=False)
    fig.text(0.53, 0.84, f"Full-score Δ = {comparison['mean_difference']:+.4f}  •  95% CI [{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]  •  p={comparison['p_value']:.4f}", fontsize=10, color=GREY)
    add_footer(fig)
    save_figure(fig, output_dir, "slide_02_crossdomain_outcome.png", dpi)


def plot_precision_recall(df, output_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.80, bottom=0.15, wspace=0.18)
    add_title(fig, "Class weighting changed the precision–recall balance", "Mean test metrics across five seeds; error bars show one standard deviation")
    fields = [("test_micro_precision", "Precision"), ("test_micro_recall", "Recall"), ("test_micro_f1", "F1")]
    x = np.arange(len(fields))
    width = 0.34
    for ax, mode in zip(axes, cfg.MODES):
        part = df[df["mode"] == mode]
        for offset, config in ((-width / 2, "standard"), (width / 2, "weighted")):
            subset = part[part["config"] == config]
            means = [subset[column].mean() for column, _ in fields]
            stds = [subset[column].std(ddof=1) for column, _ in fields]
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4, color=CONFIG_COLORS[config], label=config.title())
            for bar, value in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9, color=NAVY)
        ax.set_xticks(x, [label for _, label in fields])
        ax.set_title("In-domain (confirmatory)" if mode == "indomain" else "Cross-domain (secondary)", loc="left")
        ax.grid(axis="y", color="#E4E7EC")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Mean score")
    add_footer(fig)
    save_figure(fig, output_dir, "slide_03_precision_recall_tradeoff.png", dpi)


def plot_learning_curves(histories, output_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.96, top=0.74, bottom=0.13, wspace=0.18)
    add_title(fig, "Development performance remained separated during training", "Exact-triplet development micro-F1, averaged over the five seeded runs")
    for ax, mode in zip(axes, cfg.MODES):
        for config in CONFIGS:
            curves = []
            for seed in cfg.SEEDS:
                history = histories[(mode, config, int(seed))]
                curves.append([row["dev_micro_f1"] for row in history])
            values = np.asarray(curves, dtype=float)
            epochs = np.arange(1, values.shape[1] + 1)
            mean = values.mean(axis=0)
            std = values.std(axis=0, ddof=1)
            ax.plot(epochs, mean, color=CONFIG_COLORS[config], linewidth=2.5, marker="o", label=config.title())
            ax.fill_between(epochs, mean - std, mean + std, color=CONFIG_COLORS[config], alpha=0.16)
        ax.set_xticks(epochs)
        ax.set_xlabel("Epoch")
        ax.set_title("In-domain" if mode == "indomain" else "Cross-domain", loc="left")
        ax.grid(color="#E4E7EC")
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Development micro-F1")
    add_footer(fig)
    save_figure(fig, output_dir, "slide_04_development_learning_curves.png", dpi)


def plot_protocol_review(output_dir, dpi):
    rows = [
        ("PASS", "Complete paired matrix", "20/20 runs; 5 matched seeds per mode"),
        ("PASS", "Restaurant hold-out", "Official restaurant test only; 544 records"),
        ("PASS", "Primary metric and threshold", "Exact-triplet micro-F1; +0.02 minimum effect"),
        ("PASS", "Rare-label definition", "96 in-domain categories at 1–5 training examples"),
        ("FIX", "Warmup", "Artifacts used 0%; protocol specifies 10%"),
        ("FIX", "Gradient clipping", "No max-norm clipping is implemented"),
        ("FIX", "Ambiguous alignment", "Repeated occurrences are assigned to the first match"),
        ("FIX", "Reproducibility gates", "Audit JSON absent; dependencies unpinned; git commit not in manifests"),
        ("POST-HOC", "Cross-domain protocol metrics", "Derived here from saved predictions; absent from original metrics.json"),
    ]
    colors = {"PASS": TEAL, "FIX": CORAL, "POST-HOC": GOLD}
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.80, bottom=0.08)
    add_title(fig, "Protocol review: the matrix is complete, but deviations need disclosure", "Do not describe the current artifacts as fully protocol-conformant without a dated amendment")
    ax.axis("off")
    row_height = 0.095
    start = 0.96
    for index, (status, check, evidence) in enumerate(rows):
        y = start - index * row_height
        if index % 2:
            ax.add_patch(FancyBboxPatch((0.0, y - 0.065), 1.0, 0.083, boxstyle="round,pad=0.004", facecolor="#F8FAFC", edgecolor="none", transform=ax.transAxes))
        ax.text(0.015, y - 0.02, status, transform=ax.transAxes, color=colors[status], weight="bold", fontsize=10.5, va="center")
        ax.text(0.14, y - 0.02, check, transform=ax.transAxes, color=NAVY, weight="bold", fontsize=12, va="center")
        ax.text(0.42, y - 0.02, evidence, transform=ax.transAxes, color=GREY, fontsize=11, va="center")
    save_figure(fig, output_dir, "slide_05_protocol_review.png", dpi)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def write_reports(output_dir, comparisons, cross, df):
    serializable_cross = cross.copy()
    serializable_cross["unseen_categories"] = serializable_cross["unseen_categories"].apply(lambda value: "; ".join(value))
    payload = {
        "schema_version": 1,
        "run_matrix": {"completed_runs": len(df), "expected_runs": len(cfg.MODES) * len(CONFIGS) * len(cfg.SEEDS)},
        "comparisons": comparisons,
        "crossdomain_protocol_metrics": serializable_cross.to_dict("records"),
        "interpretation_status": "Observed results; protocol deviations documented separately.",
    }
    summary_path = output_dir / "result_analysis_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {summary_path}")

    paired_rows = []
    for mode, comparison in comparisons.items():
        for row in comparison["paired"]:
            paired_rows.append({"mode": mode, **row, "difference_weighted_minus_standard": row["weighted"] - row["standard"]})
    write_csv(output_dir / "paired_seed_results.csv", paired_rows)
    write_csv(output_dir / "crossdomain_protocol_metrics.csv", serializable_cross.to_dict("records"))

    primary, secondary = comparisons["indomain"], comparisons["crossdomain"]
    coverage = cross["category_coverage"].iloc[0]
    notes = f"""# Presentation notes: completed experiment matrix

## Confirmatory in-domain result

Across five matched seeds, standard loss achieved {primary['standard']['mean']:.3f} ± {primary['standard']['std']:.3f} exact-triplet micro-F1 and weighted loss achieved {primary['weighted']['mean']:.3f} ± {primary['weighted']['std']:.3f}. The paired difference was {primary['mean_difference']:+.4f} with a 95% paired-bootstrap interval of [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}] and paired t-test p={primary['p_value']:.4f}. Class weighting therefore did not meet the pre-specified +{cfg.MIN_EFFECT_SIZE:.2f} improvement criterion.

## Secondary cross-domain result

On the held-out restaurant test split, standard loss achieved {secondary['standard']['mean']:.3f} ± {secondary['standard']['std']:.3f}, while weighted loss achieved {secondary['weighted']['mean']:.3f} ± {secondary['weighted']['std']:.3f}. The paired difference was {secondary['mean_difference']:+.4f}; this secondary result is exploratory. The source category vocabulary covers {coverage:.1%} of retained restaurant gold triplets.

## Interpretation

Weighting increases recall in both modes. In-domain, the precision loss is large enough to reduce F1. Cross-domain, the recall gain is large enough to increase F1. This is evidence of a precision–recall trade-off, not a universal improvement.

## Disclosure

These are observed saved-artifact results, not a fully protocol-conformant final analysis. See `PROTOCOL_REVIEW.md` before presenting conclusions.
"""
    notes_path = output_dir / "PRESENTATION_NOTES.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"Saved {notes_path}")

    review = """# Protocol review of the completed artifacts

## Overall status

The experiment matrix is complete, paired, and internally consistent, but it is **not fully compliant with protocol.md version 1.0**. Treat the current outputs as observed/post-pilot results unless a dated amendment explicitly accepts the deviations below.

## Confirmed alignment

- All 20 planned `(mode, configuration, seed)` runs completed exactly once.
- Both modes use seeds 13, 42, 123, 2024, and 777.
- The cross-domain test artifacts contain 544 restaurant records, matching the official restaurant test file only.
- Model, batch size, epochs, learning rate, weight decay, maximum length, loss-component weights, held-out domain, rare-label threshold, and minimum effect threshold match the protocol.
- Development exact-triplet micro-F1 is used for checkpoint selection, with lower evaluation loss breaking exact score ties.
- `python -m pip check` currently passes.

## Deviations and missing gates

1. **Warmup mismatch:** `src/train.py` uses `num_warmup_steps=0`; section 10 specifies 10% of total steps.
2. **Gradient clipping missing:** no `clip_grad_norm_` call is present; section 10 specifies maximum norm 1.0.
3. **Ambiguous alignment policy mismatch:** repeated aspect surface occurrences are assigned to the first match rather than excluded. The current data contain 486/114/225 ambiguous examples in the in-domain train/dev/test splits and 466/109/10 in the cross-domain train/dev/test splits.
4. **Standalone audits absent:** `artifacts/data_audit.json` and `artifacts/crossdomain_audit.json` were not present after the run.
5. **Environment not locked:** `requirements.txt` uses broad lower bounds or unpinned packages. Manifests record only Python, Torch, and Transformers versions.
6. **Experiment commit not recorded:** manifests do not include a git commit identifier.
7. **Macro-F1 universe mismatch:** the implemented macro-F1 averages over labels present in each run's gold/prediction union, rather than the protocol's fixed category universe.
8. **Original cross-domain metric gap:** `metrics.json` lacks known-category exact F1, category coverage, aspect-span F1, and aspect-plus-sentiment F1. `visualize_results.py` derives them post hoc from immutable saved predictions and labels them accordingly.

## Recommended handling

Do not silently rerun and replace these results. Either present them with this disclosure, or create a dated protocol amendment, fix the implementation, record a clean commit and locked environment, regenerate audits, and rerun the full matrix as a new experiment version.
"""
    review_path = output_dir / "PROTOCOL_REVIEW.md"
    review_path.write_text(review, encoding="utf-8")
    print(f"Saved {review_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    numeric = [column for column in df.columns if column.startswith(("test_", "dev_"))]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    validate_complete_matrix(df, cfg.MODES, CONFIGS, cfg.SEEDS)
    histories, cross, _ = load_artifact_details(df)
    comparisons = {mode: comparison_summary(df, mode) for mode in cfg.MODES}

    set_style()
    plot_primary(df, comparisons["indomain"], output_dir, args.dpi)
    plot_crossdomain(df, comparisons["crossdomain"], cross, output_dir, args.dpi)
    plot_precision_recall(df, output_dir, args.dpi)
    plot_learning_curves(histories, output_dir, args.dpi)
    plot_protocol_review(output_dir, args.dpi)
    write_reports(output_dir, comparisons, cross, df)
    print(f"\nComplete: {len(df)} runs visualized; outputs in {output_dir}")


if __name__ == "__main__":
    main()
