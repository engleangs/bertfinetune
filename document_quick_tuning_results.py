"""Verify a completed study and document final findings without training."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics

from experiment_reporting.phase_comparison import compare_phase1, verify_study
from experiment_reporting.quick_report import load_selected, write_quick_report

ROOT = Path(__file__).resolve().parent
LABELS = {"Best NULL off": "Development-selected NULL off", "Best NULL on": "NULL on",
          "Vocabulary control": "Vocabulary control"}


def label(row):
    return "; ".join(LABELS.get(role, role) for role in row["roles"])


def role_row(comparison, role):
    return next(r for r in comparison["summaries"] if role in r["roles"])


def format_score(values):
    return f"{values['f1']:.4f} +/- {values['sd']:.4f}" if values["sd"] is not None else f"{values['f1']:.4f}"


def plots(comparison, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = comparison["summaries"]
    names = ["Phase 1\nre-scored" if r["model"] == "phase1" else
             "Selected\nNULL off" if "Best NULL off" in r["roles"] else
             "NULL on" if "Best NULL on" in r["roles"] else "Vocabulary\ncontrol" for r in summaries]
    colors = ["#8c949e", "#31688e", "#e89a45", "#63957c"][:len(summaries)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, scope in zip(axes, ("explicit", "combined")):
        values = [r[scope]["f1"] for r in summaries]
        deviations = [r[scope]["sd"] or 0 for r in summaries]
        ax.bar(range(len(summaries)), values, color=colors, yerr=deviations, capsize=4)
        for i, (value, sd) in enumerate(zip(values, deviations)):
            ax.text(i, value + sd + .012, f"{value:.4f}", ha="center", fontsize=10)
        ax.set(title="Explicit exact-triplet F1" if scope == "explicit" else "Eligible explicit + NULL F1",
               xticks=range(len(names)), xticklabels=names, ylim=(0, .5))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Micro-F1")
    fig.suptitle(f"Test comparison on identical gold: {len(comparison['seeds'])} matched seeds")
    fig.text(.5, .02, "Means and sample SD across seeds; whiskers are not confidence intervals.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save_figure(fig, output, "phase1_comparison", plt)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    left = np.zeros(len(summaries))
    for outcome, color, title in (("correct", "#63957c", "Correct"), ("term", "#31688e", "Term failure"),
                                  ("category", "#e89a45", "Category failure"), ("sentiment", "#bf6b70", "Sentiment failure")):
        values = np.array([r["taxonomy"][outcome] / r["explicit"]["gold"] for r in summaries])
        ax.barh(range(len(summaries)), values, left=left, color=color, label=title)
        for i, value in enumerate(values):
            if value >= .05:
                ax.text(left[i] + value / 2, i, f"{100 * value:.1f}%", ha="center", va="center", color="white", fontsize=10)
        left += values
    ax.set(yticks=range(len(names)), yticklabels=[n.replace("\n", " ") for n in names], xlim=(0, 1),
           xlabel="Share of eligible explicit gold", title="One primary outcome per gold triplet")
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncols=4, frameon=False)
    fig.text(.5, .02, "First-failure outcomes partition gold; they do not decompose F1 or establish isolated head effects.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, 1))
    save_figure(fig, output, "taxonomy_comparison", plt)

    domains = sorted(summaries[0]["domains"])
    matrix = np.array([[r["domains"][d]["f1"] for d in domains] for r in summaries])
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    chart = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=.65, aspect="auto")
    for i in range(len(summaries)):
        for j in range(len(domains)):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > .4 else "#172938")
    ax.set(xticks=range(len(domains)), xticklabels=domains, yticks=range(len(names)),
           yticklabels=[n.replace("\n", " ") for n in names], title="In-domain test F1 by domain, on identical eligible gold")
    fig.colorbar(chart, ax=ax, label="Mean explicit exact-triplet F1")
    fig.text(.5, .02, "Every model was trained in-domain. This figure does not measure leave-one-domain-out transfer.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save_figure(fig, output, "domain_comparison", plt)


def save_figure(fig, output, name, plt):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def document(frozen, comparison, verification, rows, artifact_prefix="artifacts/quick_tuning_5seeds"):
    selected = role_row(comparison, "Best NULL off")
    implicit = role_row(comparison, "Best NULL on")
    control = role_row(comparison, "Vocabulary control")
    historical = next(r for r in comparison["summaries"] if r["model"] == "phase1")
    original_f1 = statistics.mean(r["f1"] for r in comparison["original_phase1"])
    off_gain = 100 * (selected["explicit"]["f1"] - historical["explicit"]["f1"])
    null_delta = 100 * (implicit["combined"]["f1"] - control["combined"]["f1"])
    c = selected["taxonomy"]
    failures = c["term"] + c["category"] + c["sentiment"]
    domain, domain_metrics = min(selected["domains"].items(), key=lambda item: item[1]["f1"])
    lines = ["# Final fine-tuning results", "", "Verified 6 October 2026, Pacific/Auckland. Presentation: Thursday, 8 October.", "",
             f"The continuation is complete: **{verification['verified_runs']} trained and test-evaluated runs**, "
             f"three configurations with matched seeds {frozen['seeds']}. Six earlier completed evaluations were reused; "
             "nine additional models were trained. Every run completed all five training epochs.", "",
             "## Final result and model choice", "",
             f"Keep the development-selected NULL-off model as the primary result: explicit test F1 **{selected['explicit']['f1']:.4f}**. "
             f"The matching vocabulary control has the highest observed test F1, **{control['explicit']['f1']:.4f}**, "
             "but it was not the development winner. Report it as a control and retain the selection made before looking at test scores.", "",
             f"The NULL extension reaches **{implicit['null']['f1']:.4f} NULL F1**. Its combined F1 is "
             f"**{implicit['combined']['f1']:.4f}**, {abs(null_delta):.2f} percentage points below the matching NULL-off "
             "vocabulary control. It recovers some implicit annotations while adding many false positives.", "",
             "Scores are means +/- sample SD across five seeds. F1 is calculated within each seed before averaging.", "",
             "| Configuration | Explicit precision | Explicit recall | Explicit test F1 | NULL test F1 | Combined test F1 |",
             "|---|---:|---:|---|---|---|"]
    for r in (selected, implicit, control):
        lines.append(f"| {label(r)} | {r['explicit']['precision']:.4f} | {r['explicit']['recall']:.4f} | "
                     f"{format_score(r['explicit'])} | {format_score(r['null'])} | {format_score(r['combined'])} |")
    lines += ["", "## Frozen settings", "",
              "Shared: BERT-base-uncased, revision `86b5e0934494bd15c9632b12f734a8a67f723594`; "
              "LR `3e-5`; five epochs; batch 16; max length 128; weight decay .01; warmup .10; clipping 1.0; "
              "all three explicit heads weighted 1; all four sentiment classes retained.", "",
              "Loss: normalized `.75 * CE + .25 * inverse-frequency CE`. Training annotations alone determine frequencies and vocabulary.", "",
              "| Configuration | Vocabulary | NULL cap | NULL loss weight | Prediction threshold |",
              "|---|---|---:|---:|---:|",
              "| Development-selected NULL off | auto (explicit) | N/A | N/A | N/A |",
              "| NULL on | explicit-null | 10 | 0.5 | 0.2 in every seed |",
              "| Vocabulary control | explicit-null | N/A | N/A | N/A |", "",
              "Checkpoints were chosen by explicit development F1; NULL thresholds by development NULL F1. "
              "The cap was chosen by seed-13 combined development F1. Selected epochs are 4 or 5; this is checkpoint selection, not early stopping.", "",
              "## Individual test results", "",
              "| Seed | Selected NULL-off F1 | NULL-on explicit F1 | NULL F1 | Vocabulary-control F1 |",
              "|---:|---:|---:|---:|---:|"]
    for seed in frozen["seeds"]:
        members = {r["model"]: r for r in comparison["runs"] if r["seed"] == seed}
        a, b, d = (members[r["model"]] for r in (selected, implicit, control))
        lines.append(f"| {seed} | {a['explicit']['f1']:.4f} | {b['explicit']['f1']:.4f} | {b['null']['f1']:.4f} | {d['explicit']['f1']:.4f} |")
    lines += ["", "## Improvement over Phase 1", "",
              f"The originally reported Phase 1 standard-CE F1 was {original_f1:.4f}; the updated selected model scores "
              f"{selected['explicit']['f1']:.4f}, a reported difference of {100 * (selected['explicit']['f1'] - original_f1):+.2f} percentage points. "
              "Gold eligibility differs, so use the matched comparison below for the main claim.", "",
              "All five seeds use the same 3,794 test sentences. Phase 1 used 4,165 eligible explicit triplets per seed; "
              "the corrected evaluation uses 3,728. Re-scoring keeps every prediction, including false positives, "
              "and applies the current gold to both phases. Combined gold adds 1,300 deduplicated NULL annotations.", "",
              "| Model, on current gold | Explicit F1 | Gain over re-scored Phase 1 | Combined F1 |",
              "|---|---|---:|---|"]
    for r in comparison["summaries"]:
        delta = 100 * (r["explicit"]["f1"] - historical["explicit"]["f1"])
        lines.append(f"| {label(r)} | {format_score(r['explicit'])} | {delta:+.2f} pp | {format_score(r['combined'])} |")
    lines += ["", f"The primary selected model improves explicit F1 by **{off_gain:.2f} percentage points** on identical gold and matched seeds. "
              f"Recall rises from {historical['explicit']['recall']:.4f} to {selected['explicit']['recall']:.4f}; "
              f"precision changes from {historical['explicit']['precision']:.4f} to {selected['explicit']['precision']:.4f}. "
              "Loss, LR, optimization and training eligibility changed together; this measures the updated pipeline, not an isolated loss effect.", "",
              "![Same-gold Phase 1 comparison](artifacts/quick_tuning_5seeds/phase1_comparison.png)", "",
              "## Taxonomy: which component is hardest?", "",
              f"For the selected model, **term failures remain largest: {c['term']:.1f} per seed**, "
              f"{100 * c['term'] / failures:.1f}% of incorrect explicit gold. The remaining mean failures are "
              f"category {c['category']:.1f} and sentiment {c['sentiment']:.1f}. These are first-failed components, not sole causal errors.", "",
              "| Model, on current explicit gold | Correct | Term failures | Category failures | Sentiment failures |",
              "|---|---:|---:|---:|---:|"]
    for r in comparison["summaries"]:
        counts = r["taxonomy"]
        lines.append(f"| {label(r)} | {counts['correct']:.1f} | {counts['term']:.1f} | {counts['category']:.1f} | {counts['sentiment']:.1f} |")
    for r in (historical, selected):
        counts = r["taxonomy"]
        matched = r["explicit"]["gold"] - counts["term"]
        category_matched = matched - counts["category"]
        lines.append(f"\n{label(r)}: pooled category failure given a matched term "
                     f"{100 * counts['category'] / matched:.2f}%; sentiment failure given term and category "
                     f"{100 * counts['sentiment'] / category_matched:.2f}%. "
                     "Raw downstream counts depend on how many terms were recovered.")
    lines += ["", "![First-failure taxonomy](artifacts/quick_tuning_5seeds/taxonomy_comparison.png)", "",
              "## Category mismatch, rare labels and hardest domain", ""]
    selected_rows = [r for r in rows if r["trial_id"] == selected["model"] and r["split"] == "test"]
    term_sentiment = statistics.mean(r["metrics"]["term_sentiment_f1"] for r in selected_rows)
    term = statistics.mean(r["metrics"]["term_f1"] for r in selected_rows)
    mismatches = statistics.mean(r["diagnostics"]["category_mismatch"] for r in selected_rows)
    lines += [f"Term F1 is {term:.4f}, term+sentiment F1 {term_sentiment:.4f}, and full-triplet F1 {selected['explicit']['f1']:.4f}. "
              f"There are {mismatches:.1f} mean cases with the correct term and sentiment but a category mismatch. "
              "Projected scores show useful predictions hidden by exact category matching; they supplement full-triplet F1 and do not replace it.", ""]
    for bucket in ("rare", "unseen"):
        items = [b for r in selected_rows for b in r["diagnostics"]["rarity"] if b["rarity"] == bucket]
        recall = statistics.mean(1 - b["error_rate"] for b in items)
        lines.append(f"- {bucket.capitalize()} categories: {statistics.mean(b['gold_triplets'] for b in items):.0f} explicit gold per seed; "
                     f"exact-triplet recall {100 * recall:.2f}%. Frequencies use eligible training annotations; unseen and rare are separate.")
    lines += ["", f"The hardest in-domain test domain is **{domain}**, F1 **{domain_metrics['f1']:.4f}**, "
              f"with {domain_metrics['gold']} eligible explicit triplets. Domain rankings here are separate from historical LODO rankings.", "",
              "![Domain comparison](artifacts/quick_tuning_5seeds/domain_comparison.png)", "",
              "## NULL finding", "",
              f"Mean NULL precision is {100 * implicit['null']['precision']:.2f}%, recall {100 * implicit['null']['recall']:.2f}%, "
              f"and F1 {100 * implicit['null']['f1']:.2f}%. Mean true positives are {implicit['null']['tp']:.1f}, "
              f"false positives {implicit['null']['fp']:.1f}, and false negatives {implicit['null']['fn']:.1f} per seed. "
              "The head adds implicit-aspect capability, but sparse recovery and false positives limit its benefit. "
              "Any follow-up should diagnose category+sentiment pair sparsity and calibrate on development data or a new validation split; keep these test results frozen.", "",
              "## Professor recommendations and reporting limits", "",
              "- Standard CE, inverse-frequency CE, mixtures .25/1/3/.5 and focal gamma=2 were compared at LR 2e-5 on seed 13. "
              "The normalized mixtures include the professor's 1/.5 and .5/.5 examples. Mixed .25 won that screen.",
              "- Only the winning mixture received the LR sweep; all loss families were not individually optimized at LR 3e-5.",
              "- Five seeds confirmed the three shortlisted mixture configurations. This does not turn the one-seed loss screen into a five-seed comparison of every loss family.",
              "- These test sets were inspected historically; seeds 13/42 were also evaluated before the five-seed freeze. "
              "Their completed evaluations were reused. Selection reads development results only; the five-seed study remains exploratory.",
              "- Combined F1 uses eligible explicit plus deduplicated NULL gold. Ambiguous, conflicting and overlapping explicit annotations remain excluded; this is not evaluation of every raw annotation.",
              "- Five-seed SD describes seed variation, not a confidence interval or significance test. No new claim of corrected LODO improvement is made.", "",
              "## Verification and presentation wording", "",
              "Verified all 15 statuses, frozen configurations, actual checkpoint SHA-256 hashes, training-code and corpus hashes, "
              "complete five-epoch histories, development checkpoint choices, and prediction-derived explicit/NULL/combined metrics and taxonomy. "
              "Training code, configurations, checkpoints and predictions were preserved.", "",
              f"> On five matched seeds and identical evaluation gold, the updated pipeline improved explicit test F1 by {off_gain:.2f} percentage points. "
              "The strongest gain was term recovery. Rare and unseen categories remain difficult. The NULL head recovers implicit annotations, "
              "but false positives lower its combined score relative to the matching vocabulary control.", "",
              "[Full development/test report](artifacts/quick_tuning_5seeds/report.md) | "
              "[Frozen settings](artifacts/quick_tuning_5seeds/selection.json) | "
              "[Verification and Phase 1 data](artifacts/quick_tuning_5seeds/final_documentation.json)", ""]
    return "\n".join(lines).replace("artifacts/quick_tuning_5seeds", artifact_prefix)


def export_documents(frozen, comparison, verification, rows, study, summary, share_output):
    prefix = os.path.relpath(study, summary.parent).replace("\\", "/")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(document(frozen, comparison, verification, rows, prefix), encoding="utf-8")
    share_output.mkdir(parents=True, exist_ok=True)
    shared = document(frozen, comparison, verification, rows, ".")
    shared += ("\n## Sharing and regeneration\n\nThis folder contains reports, figures and frozen metadata. "
               "Checkpoints, tokenizers and prediction files are excluded. Directory references in selection.json "
               "describe the original workspace and do not need to exist to read this report.\n\n"
               "From the original workspace, regenerate and verify with:\n\n```powershell\n"
               ".venv/Scripts/python.exe document_quick_tuning_results.py\n```\n")
    (share_output / "README.md").write_text(shared, encoding="utf-8")
    for name in ("selection.json", "results.json", "final_documentation.json", "comparison.png", "comparison.svg",
                 "phase1_comparison.png", "phase1_comparison.svg", "taxonomy_comparison.png", "taxonomy_comparison.svg",
                 "domain_comparison.png", "domain_comparison.svg"):
        shutil.copyfile(study / name, share_output / name)
    full_report = (study / "report.md").read_text(encoding="utf-8")
    phase1_path = os.path.relpath(ROOT / "phase1_analysis/TEAM_FINDINGS.md", share_output).replace("\\", "/")
    (share_output / "report.md").write_text(full_report.replace("../../phase1_analysis/TEAM_FINDINGS.md", phase1_path), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=ROOT / "artifacts/quick_tuning_5seeds")
    p.add_argument("--summary", type=Path, default=ROOT / "final-finetuning-results-2026-10-06.md")
    p.add_argument("--share-output", type=Path, default=ROOT / "final_results", help="Small reports/figures only; no model weights")
    args = p.parse_args()
    frozen = json.loads((args.study / "selection.json").read_text(encoding="utf-8"))
    if len(frozen["seeds"]) != 5 or len(frozen["variants"]) != 3:
        p.error("This dated final document requires the completed five-seed, three-configuration study")
    print("Verify completed predictions, histories, provenance and checkpoint hashes...", flush=True)
    verification = verify_study(frozen, ROOT)
    comparison = compare_phase1(frozen, ROOT)
    rows = load_selected(frozen)
    write_quick_report(frozen, args.study)
    plots(comparison, args.study)
    result = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "verification": verification, "phase1": comparison}
    (args.study / "final_documentation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    export_documents(frozen, comparison, verification, rows, args.study, args.summary, args.share_output)
    print(f"Verified {verification['verified_runs']} completed runs. Summary: {args.summary}", flush=True)
    print(f"Shareable reports and figures: {args.share_output}", flush=True)


if __name__ == "__main__":
    main()
