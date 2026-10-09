"""Render only the development-selected, frozen quick-study variants."""

import json
from pathlib import Path
import statistics

from experiment_reporting.analysis import flatten_metrics
from experiment_reporting.quick_selection import check_frozen_run, check_frozen_variants


def load_selected(frozen):
    check_frozen_variants(frozen)
    rows = []
    for variant in frozen["variants"]:
        for run in variant["runs"]:
            directory = Path(run["directory"])
            manifest, _ = check_frozen_run(variant, run)
            for split in ("dev", "test"):
                if split == "test" and manifest["status"] != "complete":
                    continue
                raw = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
                metrics = flatten_metrics(raw)
                if sum(metrics["outcomes"].values()) != metrics["gold_triplets"]:
                    raise ValueError(f"Taxonomy does not partition gold: {directory}")
                rows.append({"trial_id": variant["trial_id"], "roles": variant["roles"], "config": variant["config"],
                             "seed": run["seed"], "split": split, "metrics": metrics, "directory": str(directory),
                             "diagnostics": {"rarity": raw["taxonomy"].get("by_rarity", []),
                                             "domains": raw.get("per_domain", {}),
                                             "category_mismatch": raw["taxonomy"].get("category_mismatch_with_correct_term_sentiment", 0),
                                             "null_counts": raw["null"],
                                             "sentiments": raw.get("sentiment_by_label", {})}})
    return rows


def groups(frozen, rows):
    summaries = []
    for variant in frozen["variants"]:
        for split in ("dev", "test"):
            members = [r for r in rows if r["trial_id"] == variant["trial_id"] and r["split"] == split]
            if not members:
                continue
            scores = [r["metrics"]["explicit_f1"] for r in members]
            values = {k: statistics.mean(r["metrics"][k] for r in members)
                      for k in ("explicit_f1", "precision", "recall", "combined_f1", "null_f1",
                                "term_f1", "term_category_f1", "term_sentiment_f1")}
            summaries.append({"trial_id": variant["trial_id"], "roles": variant["roles"], "split": split,
                              "seeds": sorted(r["seed"] for r in members), **values,
                              "sd": statistics.stdev(scores) if len(scores) > 1 else None,
                              "outcomes": {k: statistics.mean(r["metrics"]["outcomes"][k] for r in members)
                                           for k in ("correct", "term", "category", "sentiment")}})
    return summaries


def figure(frozen, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    variants = frozen["variants"]
    for ax, split in zip(axes, ("dev", "test")):
        for i, variant in enumerate(variants):
            members = [r for r in rows if r["trial_id"] == variant["trial_id"] and r["split"] == split]
            if not members:
                ax.text(i, .03, "Pending", ha="center", color="#666666")
                continue
            for shift, metric, color, name in ((-.18, "explicit_f1", "#31688e", "Explicit F1"),
                                               (.18, "combined_f1", "#e89a45", "Combined F1")):
                values = [r["metrics"][metric] for r in members]
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else None
                ax.bar(i + shift, mean, width=.34, yerr=sd, capsize=3, color=color, label=name if i == 0 else None)
                ax.text(i + shift, mean + (sd or 0) + .02, f"{mean:.3f}", ha="center", fontsize=9)
        names = ["NULL off" if "Best NULL off" in v["roles"] else
                 "NULL on" if "Best NULL on" in v["roles"] else "Vocabulary\ncontrol" for v in variants]
        ax.set(title="Development" if split == "dev" else "Frozen test evaluation",
               xticks=range(len(variants)), xticklabels=names, ylim=(0, 1), xlim=(-.6, len(variants) - .4))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Micro-F1")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Quick fine-tuning comparison: {len(frozen['seeds'])} development-selected seed(s)")
    fig.text(.5, .02, "Means and sample SD over seeds. Combined scoring includes NULL gold for every model.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.savefig(output / "comparison.png", dpi=160)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)


def write_quick_report(frozen, output):
    rows = load_selected(frozen)
    summary = groups(frozen, rows)
    output.mkdir(parents=True, exist_ok=True)
    lines = ["# Quick fine-tuning final comparison", "",
             "Settings and checkpoints were selected on development data and frozen for test evaluation. "
             "Completed test evaluations may be reused from an earlier study. "
             "These are the best tested candidates in a small search, not proof of a global optimum.", "",
             f"Seeds: {frozen['seeds']}. Checkpoints use explicit development F1; "
             "NULL thresholds use NULL development F1. NULL configurations use combined development F1.", "",
             "## Results", "",
             "| Split | Model | Seeds | Precision | Recall | Explicit F1 | Sample SD | NULL F1 | Combined F1 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        sd = f"{r['sd']:.4f}" if r["sd"] is not None else "N/A"
        lines.append(f"| {r['split']} | {'; '.join(r['roles'])} | {r['seeds']} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['explicit_f1']:.4f} | {sd} | {r['null_f1']:.4f} | {r['combined_f1']:.4f} |")
    lines.extend(paired_null_table(frozen, rows))
    expected_tests = sum(len(v["runs"]) for v in frozen["variants"])
    completed_tests = sum(r["split"] == "test" for r in rows)
    if completed_tests < expected_tests:
        lines.extend(["", "Test evaluation is pending; no test performance is inferred from development scores."])
    lines.extend(["", "![Frozen comparison](comparison.png)", "", "## Selected parameters", "",
                  "BERT base uncased; batch 16; max length 128; five epochs; weight decay .01; warmup .10; clipping 1.0.", "",
                  "| Model | Loss | LR | CE / weighted CE | NULL cap / loss weight | Selected thresholds by seed |",
                  "|---|---|---:|---|---|---|"])
    for variant in frozen["variants"]:
        cfg = variant["config"]
        mixture = f"{cfg['ce_weight']:g} / {cfg['weighted_ce_weight']:g}" if cfg["loss_type"] == "mixed" else "N/A"
        loss = f"focal gamma={cfg['focal_gamma']:g}" if cfg["loss_type"] == "focal" else cfg["loss_type"]
        null = f"{cfg['null_pos_weight_cap']:g} / {cfg['null_loss_weight']:g}" if cfg["null_head"] else "Head off"
        thresholds = ", ".join(f"{r['seed']}: {r['null_threshold']:g}" for r in variant["runs"]) if cfg["null_head"] else "N/A"
        lines.append(f"| {'; '.join(variant['roles'])} | {loss} | {cfg['lr']:g} | {mixture} | {null} | {thresholds} |")
    lines.extend(["", "## Development search", "",
                  "The loss screen reuses completed models. Only the winning loss family receives an LR sweep. "
                  "NULL caps and thresholds are selected on development data, before test access.", "",
                  "| Stage | Recipe | LR | NULL | Positive-weight cap | Explicit F1 | NULL F1 | Combined F1 |",
                  "|---|---|---:|---|---:|---:|---:|---:|"])
    for r in frozen.get("search_results", []):
        cfg = r["config"]
        cap = f"{cfg['null_pos_weight_cap']:g}" if cfg["null_head"] else "N/A"
        lines.append(f"| {r['stage']} | {cfg['name']} | {cfg['lr']:g} | {cfg['null_head']} | {cap} | "
                     f"{r['explicit_f1']:.4f} | {r['null_f1']:.4f} | {r['combined_f1']:.4f} |")
    lines.extend(["", "## Taxonomy and component F1", "",
                  "Mean first-failure counts per seed; these partition explicit gold. NULL failures are separate. "
                  "Component metrics diagnose errors and do not replace full-triplet F1.", "",
                  "| Split | Model | Term errors | Category errors | Sentiment errors | Term F1 | Term+category F1 | Term+sentiment F1 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"])
    for r in summary:
        c = r["outcomes"]
        lines.append(f"| {r['split']} | {'; '.join(r['roles'])} | {c['term']:.1f} | {c['category']:.1f} | {c['sentiment']:.1f} | "
                     f"{r['term_f1']:.4f} | {r['term_category_f1']:.4f} | {r['term_sentiment_f1']:.4f} |")
    lines.extend(diagnostic_tables(frozen, rows))
    lines.extend(["", "## Interpretation boundaries", "",
                  "- Compare combined F1 across both NULL settings against the same eligible explicit+NULL gold. Other excluded annotations remain outside this score.",
                  "- The vocabulary control isolates the richer label vocabulary from adding a NULL head.",
                  "- Screening uses seed 13; only shortlisted settings are checked in the final seed prefix. This is exploratory selection, not a significance claim.",
                  "- Standard CE, inverse-frequency CE, normalized mixtures .25/1/3/.5, and focal gamma=2 are screened at the same LR before the selected-family LR search.",
                  "- Historical test sets were already inspected. Do not retune from these final test outcomes.",
                  "- Domain scores here are in-domain evaluations. Use the separate 80-run Phase 1 analysis for historical LODO findings.", "",
                  "[Frozen settings](selection.json) · [Machine-readable report](results.json) · "
                  "[Phase 1 analysis](../../phase1_analysis/TEAM_FINDINGS.md)", ""])
    (output / "results.json").write_text(json.dumps({"selection": frozen, "runs": rows, "summaries": summary}, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    figure(frozen, rows, output)


def diagnostic_tables(frozen, rows):
    lines = ["", "## Rare and unseen categories", "",
             "Each bucket uses its model's eligible training-annotation counts. The explicit+NULL vocabulary "
             "control is needed for a fair bucket comparison with the NULL head. Recall below counts recovered exact triplets.", "",
             "| Split | Model | Bucket | Mean gold | Exact-triplet recall |",
             "|---|---|---|---:|---:|"]
    for variant in frozen["variants"]:
        for split in ("dev", "test"):
            members = [r for r in rows if r["trial_id"] == variant["trial_id"] and r["split"] == split]
            for bucket in ("rare", "unseen"):
                items = [b for r in members for b in r["diagnostics"]["rarity"] if b["rarity"] == bucket]
                if not items:
                    continue
                gold = statistics.mean(b["gold_triplets"] for b in items)
                recalls = [1 - b["error_rate"] for b in items if b["error_rate"] is not None]
                recall = f"{statistics.mean(recalls):.4f}" if recalls else "N/A (no gold)"
                lines.append(f"| {split} | {'; '.join(variant['roles'])} | {bucket} | {gold:.1f} | {recall} |")
    lines.extend(["", "## Hardest domains", "",
                  "Mean explicit exact-triplet F1 across the selected seeds, with mean explicit gold support.", "",
                  "| Split | Model | Domain | Explicit F1 | Mean gold |",
                  "|---|---|---|---:|---:|"])
    for variant in frozen["variants"]:
        for split in ("dev", "test"):
            members = [r for r in rows if r["trial_id"] == variant["trial_id"] and r["split"] == split]
            domains = sorted({d for r in members for d in r["diagnostics"]["domains"]})
            scores = []
            for domain in domains:
                items = [r["diagnostics"]["domains"][domain] for r in members]
                scores.append((statistics.mean(d["explicit"]["micro_f1"] for d in items), domain,
                               statistics.mean(d["components"]["gold_triplets"] for d in items)))
            for score, domain, gold in sorted(scores):
                lines.append(f"| {split} | {'; '.join(variant['roles'])} | {domain} | {score:.4f} | {gold:.1f} |")
    lines.extend(["", "## NULL recovery and category mismatches", "",
                  "Category-mismatch counts have the correct term and sentiment but fail the exact category label. "
                  "NULL true/false positives and false negatives count category+sentiment pairs without a text span.", "",
                  "| Split | Model | Category mismatches | NULL TP | NULL FP | NULL FN |",
                  "|---|---|---:|---:|---:|---:|"])
    for variant in frozen["variants"]:
        for split in ("dev", "test"):
            members = [r for r in rows if r["trial_id"] == variant["trial_id"] and r["split"] == split]
            if not members:
                continue
            mismatch = statistics.mean(r["diagnostics"]["category_mismatch"] for r in members)
            counts = [statistics.mean(r["diagnostics"]["null_counts"][key] for r in members)
                      for key in ("true_positives", "false_positives", "false_negatives")]
            lines.append(f"| {split} | {'; '.join(variant['roles'])} | {mismatch:.1f} | "
                         + " | ".join(f"{value:.1f}" for value in counts) + " |")
    return lines


def paired_null_table(frozen, rows):
    implicit = next((v for v in frozen["variants"] if "Best NULL on" in v["roles"]), None)
    control = next((v for v in frozen["variants"] if "Vocabulary control" in v["roles"]), None)
    if implicit is None or control is None:
        return []
    lines = ["", "NULL head contribution against the matching vocabulary control (head on minus head off). "
             "Deltas are percentage points averaged within matched seeds.", "",
             "| Split | Matched seeds | Explicit F1 delta | Combined F1 delta | NULL F1 delta |",
             "|---|---|---:|---:|---:|"]
    for split in ("dev", "test"):
        on = {r["seed"]: r for r in rows if r["trial_id"] == implicit["trial_id"] and r["split"] == split}
        off = {r["seed"]: r for r in rows if r["trial_id"] == control["trial_id"] and r["split"] == split}
        seeds = sorted(set(on) & set(off))
        if not seeds:
            continue
        deltas = [100 * statistics.mean(on[s]["metrics"][key] - off[s]["metrics"][key] for s in seeds)
                  for key in ("explicit_f1", "combined_f1", "null_f1")]
        lines.append(f"| {split} | {seeds} | " + " | ".join(f"{v:+.2f}" for v in deltas) + " |")
    return lines
