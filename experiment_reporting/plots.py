"""Export readable research figures from the same snapshot used by the report."""

from collections import defaultdict
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_reporting.analysis import mean

NAVY, BLUE, TEAL, ORANGE = "#18354a", "#3978b9", "#168b7d", "#d58b27"
GREY, RED, PALE = "#64748b", "#d86762", "#f1f5f9"


def label(row):
    cfg = row["config"]
    name = row["recipe"]
    names = {"standard": "Standard CE", "weighted": "Weighted CE: all heads",
             "weighted_bio": "Weighted CE: BIO", "weighted_category": "Weighted CE: category",
             "weighted_sentiment": "Weighted CE: sentiment", "standard_null_vocab": "CE: NULL head off",
             "standard_null": "CE: NULL head on"}
    if name.startswith("mixed_"):
        alpha = cfg["weighted_ce_weight"] / (cfg["ce_weight"] + cfg["weighted_ce_weight"])
        text = f"Mixed CE: alpha={alpha:.3g}"
    elif name.startswith("focal_"):
        text = f"Focal: gamma={cfg['focal_gamma']:g}"
    else:
        text = names.get(name, name)
    return text + (" *" if row.get("provisional") else "")


def empty(ax, message):
    ax.set_axis_off()
    ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center", color=GREY, fontsize=12, wrap=True)


def finish(fig, output, name, title, subtitle, footer, dpi):
    fig.suptitle(title, x=.035, ha="left", y=.98, fontsize=19, weight="bold", color=NAVY)
    fig.text(.035, .935, subtitle, color=GREY, fontsize=10, ha="left", va="top")
    fig.text(.035, .012, footer, color=GREY, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(.015, .045, .99, .89))
    fig.savefig(output / f"{name}.png", dpi=dpi, facecolor="white")
    fig.savefig(output / f"{name}.svg", facecolor="white")
    plt.close(fig)
    return {"name": name, "title": title, "png": f"{name}.png", "svg": f"{name}.svg"}


def cohorts(analysis):
    groups = defaultdict(list)
    for row in analysis["summaries"]:
        groups[row["cohort"]].append(row)
    return list(groups.values()) or [[]]


def selected_runs(analysis, summary):
    return [r for r in analysis["runs"] if r["cohort"] == summary["cohort"] and r["trial_id"] == summary["trial_id"]
            and r["seed"] in summary["scored_seeds"] and r["metrics"] is not None
            and (summary["provisional"] or r["completed"])]


def score_chart(analysis, output, dpi):
    groups = cohorts(analysis)
    height = max(4.8, max(len(group) for group in groups) * .50 + 2.2)
    fig, axes = plt.subplots(1, len(groups), figsize=(max(10, 7 * len(groups)), height), squeeze=False)
    for ax, group in zip(axes[0], groups):
        if not group:
            empty(ax, "No evaluated development epochs yet.")
            continue
        group = sorted(group, key=lambda row: row["explicit_f1_mean"] or 0, reverse=True)
        for index, row in enumerate(group):
            for offset, metric, color in ((-.20, "precision", BLUE), (0, "explicit_f1", TEAL), (.20, "recall", ORANGE)):
                value, deviation = row[f"{metric}_mean"], row[f"{metric}_std"]
                ax.errorbar(100 * value, index + offset, xerr=100 * deviation if deviation is not None else None,
                            fmt="o", markersize=7, color=color, capsize=3,
                            markerfacecolor="white" if row["provisional"] else color)
            ax.text(min(100, row["explicit_f1_mean"] * 100 + 2), index - .04,
                    f"{100 * row['explicit_f1_mean']:.2f}", color=TEAL, fontsize=9)
        ax.set_yticks(range(len(group)), [label(row) for row in group])
        ax.invert_yaxis()
        ax.set_xlim(0, 103)
        ax.set_xlabel("Development score (%)")
        ax.set_title(group[0]["cohort_label"].split(" / ", 2)[-1], loc="left", fontsize=12)
        ax.grid(axis="x", alpha=.2)
        for metric, color in (("Precision", BLUE), ("F1", TEAL), ("Recall", ORANGE)):
            ax.scatter([], [], label=metric, color=color, s=35)
        ax.legend(loc="lower right", frameon=False, ncol=3, fontsize=9)
    return finish(fig, output, "01_scores", "F1, precision and recall", "Compare loss settings within each vocabulary and training budget.",
                  "Whiskers: sample SD across seeds, not confidence intervals. * Hollow markers: provisional best epoch of unfinished runs.", dpi)


def delta_chart(analysis, output, dpi):
    summaries = {(s["cohort"], s["trial_id"]): s for s in analysis["summaries"]}
    rows = [r for r in analysis["comparisons"] if r.get("pairs") and r["trial_id"] in {s["trial_id"] for s in analysis["summaries"]}
            and summaries[(r["cohort"], r["trial_id"])]["recipe"] != r["control"]]
    fig, ax = plt.subplots(figsize=(11, max(4.8, .55 * len(rows) + 2)))
    if not rows:
        empty(ax, "Awaiting completed candidates and their matched CE controls.\nUnfinished scores cannot establish an improvement.")
    else:
        rows = sorted(rows, key=lambda row: row["delta_f1"], reverse=True)
        for index, row in enumerate(rows):
            values = [100 * p["delta_f1"] for p in row["pairs"]]
            color = TEAL if row["ready"] and row["delta_f1"] > 0 else ORANGE if not row["ready"] else RED
            ax.plot([min(values), max(values)], [index, index], color=color, alpha=.5, linewidth=3)
            ax.scatter(values, [index] * len(values), color=color, s=35, alpha=.7)
            ax.scatter(100 * row["delta_f1"], index, marker="D", color=color, s=70)
        ax.set_yticks(range(len(rows)), [label(summaries[(r["cohort"], r["trial_id"])]) for r in rows])
        ax.invert_yaxis()
        ax.axvline(0, color=GREY, linewidth=1)
        ax.axvline(100 * analysis["minimum_effect"], color=TEAL, linestyle="--", label="Minimum effect target")
        ax.set_xlabel("Paired explicit F1 change versus compatible CE control (percentage points)")
        ax.grid(axis="x", alpha=.2)
        ax.legend(frameon=False)
    return finish(fig, output, "02_paired_changes", "Does the change improve F1?", "Each dot is one completed seed pair; diamonds show the paired mean.",
                  "Match source data, vocabulary, task, budget and seed. The NULL extension uses its own CE control. Two seeds are screening evidence.", dpi)


def component_chart(analysis, output, dpi):
    rows = analysis["summaries"]
    fig, ax = plt.subplots(figsize=(12, max(4.8, .52 * len(rows) + 2)))
    if not rows:
        empty(ax, "Component scores will appear after the first development evaluation.")
    else:
        fields = ("term_f1", "boundary_f1", "term_category_f1", "term_sentiment_f1", "explicit_f1")
        values = np.array([[row[f"{metric}_mean"] * 100 for metric in fields] for row in rows])
        chart = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_xticks(range(len(fields)), ["Term surface", "Exact boundary", "Term + category", "Term + sentiment", "Full triplet"])
        ax.set_yticks(range(len(rows)), [label(row) for row in rows])
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value > 55 else NAVY, fontsize=9)
        fig.colorbar(chart, ax=ax, label="F1 (%)", shrink=.8)
    return finish(fig, output, "03_components", "Where does useful prediction get lost?", "Component F1 supplements exact-triplet F1; the columns do not add up.",
                  "Rows retain their own vocabulary scope. * Provisional scores from unfinished runs. Exact boundaries use saved character offsets.", dpi)


def taxonomy_chart(analysis, output, dpi):
    rows = analysis["summaries"]
    fig, ax = plt.subplots(figsize=(12, max(4.8, .55 * len(rows) + 2)))
    if not rows:
        empty(ax, "The first failed component taxonomy will appear after development evaluation.")
    else:
        left = np.zeros(len(rows))
        for name, color in (("correct", TEAL), ("term", BLUE), ("category", ORANGE), ("sentiment", RED)):
            values = np.array([100 * row["outcomes"][name] / row["gold_triplets"] if row["gold_triplets"] else 0 for row in rows])
            ax.barh(range(len(rows)), values, left=left, color=color, label=name.capitalize(), height=.65)
            for index, value in enumerate(values):
                if value >= 5:
                    ax.text(left[index] + value / 2, index, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=9)
            left += values
        ax.set_yticks(range(len(rows)), [label(row) for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of all retained explicit gold triplets (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=4, frameon=False)
    return finish(fig, output, "04_taxonomy", "Which component is the largest obstacle?", "Exclusive outcomes: correct, then term failure, category failure or sentiment failure.",
                  "Denominator: all retained explicit gold, pooled over displayed seeds. NULL has separate category/sentiment diagnostics. * Provisional.", dpi)


def learning_chart(analysis, output, dpi):
    groups = defaultdict(list)
    for run in analysis["runs"]:
        if run["history"]:
            groups[(run["cohort"], run["trial_id"])].append(run)
    columns = min(3, max(1, len(groups)))
    rows = max(1, math.ceil(len(groups) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(max(10, columns * 4.3), max(5, rows * 2.9 + 1.5)), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    if not groups:
        empty(axes.flat[0], "No completed epoch evaluations have been recorded yet.")
    for ax, members in zip(axes.flat, groups.values()):
        ax.set_axis_on()
        for run in members:
            ax.plot([r["epoch"] for r in run["history"]], [100 * r["f1"] for r in run["history"]],
                    marker="o", markersize=4, linewidth=1.7, label=f"seed {run['seed']}" + (" (unfinished)" if not run["completed"] else ""))
        ax.set_title(label(members[0]), loc="left", fontsize=11)
        ax.set_xlim(.7, members[0]["config"]["epochs"] + .3)
        ax.set_ylim(0, 100)
        ax.set_xticks(range(1, members[0]["config"]["epochs"] + 1))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Explicit F1 (%)")
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=8)
    return finish(fig, output, "05_learning_curves", "Is development F1 still improving?", "Saved per-epoch evaluations, including unfinished runs. No missing epochs are filled in.",
                  "Use curves to decide whether a separate learning-rate or epoch-budget experiment is justified. Checkpoint selection stays development-only.", dpi)


def null_chart(analysis, output, dpi):
    rows = [row for row in analysis["summaries"] if row["config"].get("null_head") or row["config"].get("vocabulary_scope") == "explicit-null"]
    fig, ax = plt.subplots(figsize=(11, max(4.8, .8 * len(rows) + 2)))
    if not rows:
        empty(ax, "The matched NULL-head control and extension have not been evaluated yet.\nTheir explicit, NULL-only and combined F1 will appear here.")
    else:
        for offset, name, text, color in ((-.23, "explicit_f1", "Explicit", BLUE), (0, "null_f1", "NULL only", ORANGE), (.23, "combined_f1", "Combined", TEAL)):
            ax.barh(np.arange(len(rows)) + offset, [100 * row[f"{name}_mean"] for row in rows], height=.22, label=text, color=color)
        ax.set_yticks(range(len(rows)), [label(row) for row in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Development F1 (%)")
        ax.legend(frameon=False, loc="lower right")
    return finish(fig, output, "06_null_comparison", "Does the NULL head help its intended task?", "The head-off control and extension share the explicit+NULL source-training vocabulary.",
                  "Combined = eligible explicit plus deduplicated NULL gold; it is not full-annotation TASD. Explicit F1 remains primary. * Provisional.", dpi)


def domain_chart(analysis, output, dpi):
    rows = analysis["summaries"]
    domains = sorted({d for r in analysis["runs"] for d in r["per_domain"]})
    fig, ax = plt.subplots(figsize=(12, max(4.8, .55 * len(rows) + 2)))
    if not rows or not domains:
        empty(ax, "Per-domain scores will appear after the first development evaluation.")
    else:
        values = np.array([[mean([r["per_domain"].get(domain, {}).get("explicit", {}).get("micro_f1")
                                 for r in selected_runs(analysis, row)]) for domain in domains] for row in rows], dtype=float) * 100
        chart = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(domains)), domains)
        ax.set_yticks(range(len(rows)), [label(row) for row in rows])
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f"{value:.2f}" if math.isfinite(value) else "N/A", ha="center", va="center",
                    color="white" if value > 55 else NAVY, fontsize=9)
        fig.colorbar(chart, ax=ax, label="Explicit F1 (%)", shrink=.8)
    return finish(fig, output, "07_domains", "Which development domains are hardest?", "Exact explicit-triplet F1 per domain; interpret with source-category coverage in the report exports.",
                  "Development rankings are descriptive. Missing domain support is N/A. * Provisional scores from unfinished runs.", dpi)


def make_figures(analysis, output, dpi=150):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": NAVY, "xtick.color": GREY,
                         "ytick.color": GREY, "svg.fonttype": "none"})
    return [function(analysis, output, dpi) for function in
            (score_chart, delta_chart, component_chart, taxonomy_chart, learning_chart, null_chart, domain_chart)]
