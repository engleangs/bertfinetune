"""Generate Phase 1, Phase 2 and versus CSV reports in the supplied column format."""

import argparse
import json
from pathlib import Path
import zipfile

try:
    from .compare_results import read, sha256, verify_bundle
    from .csv_export import (CSV_COLUMNS, comparison_stems, native_summaries, paired_comparisons,
                             phase1_runs, phase2_runs, registered_cohorts, save_csv,
                             summarize_pairs, template_rows)
except ImportError:
    from compare_results import read, sha256, verify_bundle
    from csv_export import (CSV_COLUMNS, comparison_stems, native_summaries, paired_comparisons,
                            phase1_runs, phase2_runs, registered_cohorts, save_csv,
                            summarize_pairs, template_rows)

ROOT = Path(__file__).resolve().parent


def show(value):
    return "N/A" if value is None else f"{100*value:.2f}"


def write_phase_report(output, phase, exports, parameters):
    summaries = native_summaries(exports)
    save_csv(output / "summary.csv", summaries)
    save_csv(output / "run_parameters.csv", parameters)
    lines = [f"# {phase} CSV report", "",
             "The development_results.csv and test_results.csv headers match the supplied example exactly.",
             "Scores are fractions in CSV and percentages below. Empty cells mean unavailable or undefined; they are not zeros.",
             "", "| Split | Mode | Domain / fold | Recipe | Trial | Seeds | Explicit F1 (%) | NULL F1 (%) | Combined F1 (%) |",
             "|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['split']} | {row['mode']} | {row['domain']} | {row['recipe']} | {row['trial_id']} | "
                     f"{row['n']} | {show(row['explicit_f1_mean'])} | {show(row['null_f1_mean'])} | {show(row['combined_f1_mean'])} |")
    lines += ["", "- [Development runs](development_results.csv), [test runs](test_results.csv).",
              "- [Means and sample SD](summary.csv), [run settings and provenance](run_parameters.csv).",
              "- LODO development scores evaluate the six source domains; the domain column identifies the held-out fold. Only the corresponding test scores evaluate the held-out domain.",
              "- run_directory identifies an archive and member prefix relative to the phase2 handoff root; no original machine path is required."]
    if phase == "Phase 1":
        lines += ["- Historical NULL and offset-boundary F1 were not measured and remain blank. NULL prevalence is a separate audit.",
                  "- These native historical scores use older gold eligibility. Use the versus report for same-gold gains."]
    else:
        incomplete = [row for row in parameters if not row["included_in_score_exports"]]
        if incomplete:
            save_csv(output / "incomplete_runs.csv", incomplete)
        lines += ["- One row per unique trained model, even when a checkpoint appears in both screening and finalist cohorts.",
                  "- run_parameters.csv marks finalists, screens and other saved research runs. Archive membership alone does not imply finalist status.",
                  "- No missing test score is filled with a development result; development-only models simply have no test row.",
                  "- [Interrupted/failed runs](incomplete_runs.csv) are excluded from completed score exports; their status and settings are retained separately."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_versus_report(output, summary):
    lines = ["# Phase 1 versus Phase 2: same-gold comparison", "",
             "The Phase 2 arm is the previously development-selected NULL-off mixture: "
             "0.75 CE + 0.25 inverse-frequency CE, LR 3e-5. Both historical standard and weighted CE are retained as baselines.",
             "Each seed keeps its original selected checkpoint. Historical predictions are unchanged; corrected Phase 2 gold is matched by domain, example ID and exact sentence text.",
             "All predictions on excluded annotations still count as false positives. The comparison does not choose new settings from test scores.",
             "", "| Split | Mode | Domain / fold | Phase 1 baseline | Seeds | Phase 1 explicit F1 (%) | Phase 2 explicit F1 (%) | Difference (pp) |",
             "|---|---|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['split']} | {row['mode']} | {row['domain']} | {row['baseline']} | {row['n']} | "
                     f"{show(row['explicit_f1_phase1_mean'])} | {show(row['explicit_f1_phase2_mean'])} | "
                     f"{row['explicit_f1_delta_pp_mean']:+.2f} |")
    lines += ["", "- [Development seed pairs](development_results.csv), [test seed pairs](test_results.csv), [means and sample SD](summary.csv).",
              "- delta_pp fields are Phase 2 minus Phase 1 in percentage points. Counts are not percentage differences.",
              "- Sample SD is across the five matched seeds, not a confidence interval. Component projections are separate diagnostic tasks.",
              "- LODO development data contains only the source domains. The held-out target domain appears in test evaluation.",
              "- Rare recall uses each model's original source-training rarity labels; its label subset can differ between pipelines.",
              "- A Phase 1 NULL-off baseline has zero NULL F1 when re-scored on known current NULL gold. Historical native NULL F1 remains unmeasured.",
              "- Pipeline eligibility, loss and optimizer settings differ. The gains do not isolate a loss effect.",
              "- Historical tests were inspected before follow-up, and common LODO settings came from all-domain development tuning. Report the study as exploratory."]
    if (output / "in_domain_development_and_test.png").is_file():
        lines += ["", "![In-domain development and frozen test comparison](in_domain_development_and_test.png)"]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def charts(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, split in zip(axes, ("dev", "test")):
        rows = {row["baseline"]: row for row in summary if row["split"] == split and row["mode"] == "indomain"}
        values = [rows["standard"]["explicit_f1_phase1_mean"], rows["weighted"]["explicit_f1_phase1_mean"],
                  rows["standard"]["explicit_f1_phase2_mean"]]
        sd = [rows["standard"]["explicit_f1_phase1_sd"], rows["weighted"]["explicit_f1_phase1_sd"],
              rows["standard"]["explicit_f1_phase2_sd"]]
        bars = ax.bar(["Phase 1\nstandard", "Phase 1\nweighted", "Phase 2\nmixed"], [100*v for v in values],
                      yerr=[100*v for v in sd], color=["#94a3b8", "#14b8a6", "#2563eb"], capsize=4)
        ax.bar_label(bars, labels=[f"{100*v:.2f}" for v in values], padding=6)
        ax.set_title("Development" if split == "dev" else "Frozen test")
        ax.set_ylabel("Explicit F1 (%)")
        ax.set_ylim(0, max(100*(v+s) for v, s in zip(values, sd))*1.18)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("In-domain Phase 1 vs Phase 2 on matching corrected gold\nFive-seed means with sample SD")
    fig.savefig(output / "in_domain_development_and_test.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "comparison/csv_reports")
    parser.add_argument("--no-plots", action="store_true", help="Use only Python's standard library")
    args = parser.parse_args(argv)
    checked = verify_bundle(ROOT)
    source = ROOT / "report_templates/development_results.csv"
    examples = template_rows(source)
    source_digest = sha256(source)
    data, lodo = read(ROOT / "results/in_domain/results.json"), read(ROOT / "results/lodo/results.json")
    if not data["complete"] or not lodo["complete"]:
        raise ValueError("Both studies must be complete before export")
    with zipfile.ZipFile(ROOT / "evidence/phase1_outputs.zip") as legacy, zipfile.ZipFile(ROOT / "evidence/phase2_outputs.zip") as current:
        old, parameters1 = phase1_runs(legacy)
        new, parameters2 = phase2_runs(current, registered_cohorts(data, lodo))
        paired = paired_comparisons(legacy, current, comparison_stems(data, lodo))
    output = args.output.resolve()
    if output == ROOT or output in ROOT.parents:
        raise ValueError("Use a report subfolder, not the handoff or repository root")
    for name, exports, parameters in (("phase1", old, parameters1), ("phase2", new, parameters2)):
        directory = output / name
        for split, rows in exports.items():
            filename = "development_results.csv" if split == "dev" else "test_results.csv"
            save_csv(directory / filename, rows, CSV_COLUMNS)
        write_phase_report(directory, "Phase 1" if name == "phase1" else "Phase 2", exports, parameters)
    versus = output / "vs"
    for split, rows in paired.items():
        save_csv(versus / ("development_results.csv" if split == "dev" else "test_results.csv"), rows)
    summary = summarize_pairs(paired, lodo["freeze"]["seeds"])
    save_csv(versus / "summary.csv", summary)
    if not args.no_plots:
        try:
            charts(summary, versus)
        except ImportError:
            print("CSV reports complete. Install requirements-analysis.txt to regenerate the chart.")
    write_versus_report(versus, summary)
    smoke = output / "supplied_smoke"
    save_csv(smoke / "development_results.csv", examples, CSV_COLUMNS)
    (smoke / "README.md").write_text(
        "# Supplied smoke example\n\n"
        "These three development rows come from the supplied development_results.csv. "
        "They use seed 13, two epochs and scope smoke_256_128 on another machine. "
        "They establish formatting and a small-run check, rather than a full-study Phase 2 improvement. "
        "The full-data reports do not aggregate or subtract these scores.\n", encoding="utf-8")
    validation = {"verified_input_files": checked, "template_sha256": source_digest,
                  "columns_match_supplied_file": True,
                  "phase1_rows": {k: len(v) for k, v in old.items()},
                  "phase2_unique_rows": {k: len(v) for k, v in new.items()},
                  "excluded_incomplete_models": sum(not row["included_in_score_exports"] for row in parameters2),
                  "paired_rows": {k: len(v) for k, v in paired.items()},
                  "smoke_rows_separate": len(examples),
                  "native_prediction_counts_verified": True, "same_gold_sentence_id_checks": True}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# CSV reports matching development_results.csv\n\n"
        "Generated from the frozen model-free handoff; no training or inference.\n\n"
        "- [Phase 1 report](phase1/README.md): 80 historical runs, with separate development/test CSVs.\n"
        "- [Phase 2 report](phase2/README.md): all unique bundled development/test models and their settings.\n"
        "- [Phase 1 versus Phase 2](vs/README.md): matched five-seed development/test comparisons on corrected gold.\n"
        "- [Supplied smoke example](supplied_smoke/README.md): kept separate from full-data results.\n\n"
        "The phase CSVs preserve the original 25 columns and their order. The versus CSVs put "
        "Phase 1/Phase 2 metrics side by side, with percentage-point differences. "
        "Blank cells mean a metric was unavailable or had no denominator.\n\n"
        "explicit_f1 is exact (term, category, sentiment) micro F1; null_f1 is implicit "
        "(category, sentiment) F1. Combined F1 uses combined TP/FP/FN, not an average of scores. "
        "term_f1, term_category_f1 and term_sentiment_f1 project and deduplicate within each sentence. "
        "boundary_f1 uses offsets and stays blank for historical runs without offset records. "
        "known_category_coverage uses each run's source vocabulary. Error counts assign the first "
        "failed component per gold triplet. Rare means source category count 1-5; unseen labels "
        "are separate. A saved NULL threshold is inactive when null_head is False in run_parameters.csv.\n\n"
        "Regenerate from the repository root with:\n\n"
        "    python phase2/export_csv_reports.py\n\n"
        "Use --no-plots for the standard-library-only path. Inside a copied phase2 folder, "
        "run python export_csv_reports.py. Originals are preserved; these derived reports are replaceable.\n",
        encoding="utf-8")
    print(f"Reports ready: {output / 'README.md'}")
    print(f"Phase 1: {len(old['dev'])} development / {len(old['test'])} test rows")
    print(f"Phase 2: {len(new['dev'])} development / {len(new['test'])} test rows; one row per model")
    print(f"Versus: {len(paired['dev'])} development / {len(paired['test'])} test pairs")


if __name__ == "__main__":
    main()
