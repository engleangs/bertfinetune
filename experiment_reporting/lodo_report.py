"""Report frozen LODO test results, components and coverage without model inference."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics

from experiment_reporting.analysis import flatten_metrics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_results(freeze_path):
    frozen = read(freeze_path)
    expected = {(domain, seed) for domain in frozen["domains"] for seed in frozen["seeds"]}
    if {(r["domain"], r["seed"]) for r in frozen["runs"]} != expected or len(frozen["runs"]) != len(expected):
        raise ValueError("LODO freeze has an incomplete or duplicated domain/seed matrix")
    rows = []
    for run in frozen["runs"]:
        folder = Path(run["directory"])
        m = read(folder / "manifest.json")
        identity = m["identity"]
        if (m["status"] != "complete" or m["fingerprint"] != run["fingerprint"]
                or m["checkpoint_sha256"] != run["checkpoint_sha256"]
                or identity["config"] != frozen["config"] or identity["seed"] != run["seed"]
                or identity["mode"] != "crossdomain" or identity["held_out_domain"] != run["domain"]
                or identity.get("smoke") or run["domain"] in identity["source_domains"]
                or set(identity["source_domains"]) != set(frozen["domains"]) - {run["domain"]}
                or m["training"]["best_epoch"] != run["best_epoch"]):
            raise ValueError(f"Frozen LODO metadata changed or test evaluation is incomplete: {folder}")
        dev, test = read(folder / "dev_metrics.json"), read(folder / "test_metrics.json")
        if dev["null_threshold"] != run["null_threshold"] or test["null_threshold"] != run["null_threshold"]:
            raise ValueError(f"Frozen operating point changed: {folder}")
        flat = flatten_metrics(test)
        if sum(flat["outcomes"].values()) != flat["gold_triplets"]:
            raise ValueError(f"Taxonomy does not partition explicit gold: {folder}")
        rows.append({**run, "flat": flat, "metrics": test,
                     "dev_explicit_f1": dev["explicit"]["micro_f1"],
                     "source_domains": identity["source_domains"]})
    return frozen, rows


def scores(values):
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def summarize(frozen, rows):
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    summaries = []
    for domain in frozen["domains"]:
        members = by_domain[domain]
        if sorted(r["seed"] for r in members) != sorted(frozen["seeds"]):
            raise ValueError(f"Incomplete seed cohort in {domain}")
        values = {name: scores([r["flat"][name] for r in members]) for name in
                  ("explicit_f1", "term_f1", "term_category_f1", "term_sentiment_f1", "coverage")}
        outcomes = {key: statistics.mean(r["flat"]["outcomes"].get(key, 0) for r in members)
                    for key in ("correct", "term", "category", "sentiment")}
        rarity = {}
        for bucket in ("rare", "unseen"):
            entries = [next((r for r in row["metrics"]["taxonomy"]["by_rarity"] if r["rarity"] == bucket),
                            {"gold_triplets": 0, "primary_outcome_counts": {}}) for row in members]
            gold = sum(e["gold_triplets"] for e in entries)
            correct = sum(e["primary_outcome_counts"].get("correct", 0) for e in entries)
            rarity[bucket] = {"mean_gold": gold / len(members), "recall": correct / gold if gold else None}
        summaries.append({"domain": domain, "seeds": frozen["seeds"], "scores": values,
                          "mean_gold": statistics.mean(r["flat"]["gold_triplets"] for r in members),
                          "outcomes": outcomes, "rarity": rarity})
    pooled, equal_domain = [], []
    for seed in frozen["seeds"]:
        members = [r for r in rows if r["seed"] == seed]
        tp, fp, fn = (sum(r["metrics"]["explicit"][key] for r in members)
                      for key in ("true_positives", "false_positives", "false_negatives"))
        pooled.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
        equal_domain.append(statistics.mean(r["flat"]["explicit_f1"] for r in members))
    return summaries, {"pooled_fold_micro_f1": scores(pooled),
                       "equal_domain_mean_f1": scores(equal_domain)}


def fmt(value):
    return f"{value:.4f}" if value is not None else "N/A"


def mean_sd(value):
    return f"{value['mean']:.4f} +/- {fmt(value['sd'])}"


def draw(summaries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    domains = [r["domain"] for r in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = list(range(len(domains)))
    for offset, key, label, color in [(-.24, "explicit_f1", "Exact triplet", "#2563eb"),
                                    (0, "term_f1", "Term", "#14b8a6"),
                                    (.24, "term_sentiment_f1", "Term + sentiment", "#eab308")]:
        axes[0].bar([v + offset for v in x], [100*r["scores"][key]["mean"] for r in summaries],
                    .24, yerr=[100*(r["scores"][key]["sd"] or 0) for r in summaries],
                    label=label, color=color, capsize=2)
    axes[0].set_ylabel("F1 (%)")
    axes[0].set_title("LODO transfer: exact and projected scores")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [100*r["scores"]["coverage"]["mean"] for r in summaries], color="#64748b")
    axes[1].set_ylabel("Eligible test gold with source-known category (%)")
    axes[1].set_title("Category coverage limits exact-triplet recovery")
    for ax in axes:
        ax.set_xticks(x, domains, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(freeze_path, output):
    frozen, rows = load_results(freeze_path)
    summaries, overall = summarize(frozen, rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "complete": True,
               "freeze": frozen, "test_runs": len(rows), "summaries": summaries,
               "overall": overall, "runs": rows}
    (output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(freeze_path, output / "selection.json")
    draw(summaries, output / "comparison.png")
    cfg = frozen["config"]
    lines = ["# LODO fine-tuning results", "",
             f"Complete: **{len(rows)}/{len(frozen['domains'])*len(frozen['seeds'])} test evaluations**. Seeds: {frozen['seeds']}.", "",
             "Exploratory transfer of the frozen in-domain parameters. Each fold trains on six source domains, selects checkpoints on their development data, and evaluates only the held-out domain's official test set. The original hyperparameter selection used all-domain development data, so this is not a strict source-only hyperparameter search.", "",
             f"Model: {cfg['model_name']}; explicit loss {cfg['ce_weight']} CE + {cfg['weighted_ce_weight']} inverse-frequency CE; LR {cfg['lr']}; epochs {cfg['epochs']}; batch {cfg['batch_size']}; maximum length {cfg['max_len']}; weight decay {cfg['weight_decay']}; warmup {cfg['warmup_ratio']}; clipping {cfg['max_grad_norm']}; NULL head off. Frequencies and category vocabularies are rebuilt from each fold's source training annotations. Conflict is retained when present in source training targets.", "",
             "## Test results by held-out domain", "",
             "Scores use the 0–1 scale and are averaged across seeds. SD is sample SD, not a confidence interval. Projected metrics supplement exact-triplet F1.", "",
             "| Domain | Seeds | Exact-triplet F1 +/- SD | Term F1 | Term+sentiment F1 | Source-category coverage | Mean explicit gold |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        s = r["scores"]
        lines.append(f"| {r['domain']} | {len(r['seeds'])} | {mean_sd(s['explicit_f1'])} | {s['term_f1']['mean']:.4f} | {s['term_sentiment_f1']['mean']:.4f} | {s['coverage']['mean']:.4f} | {r['mean_gold']:.1f} |")
    lines += ["", f"Equal-domain mean exact F1: **{mean_sd(overall['equal_domain_mean_f1'])}**. Each domain contributes equally within a seed, then seed means are averaged.", "",
              f"Pooled-fold micro F1: **{mean_sd(overall['pooled_fold_micro_f1'])}**. TP/FP/FN are summed across the seven held-out evaluations within each seed before averaging. These are evaluations from seven separately trained models, not one all-domain checkpoint.", "",
              "![Exact/projected F1 and category coverage](comparison.png)", "",
              "## First-failed-component taxonomy and rarity", "",
              "Counts below are means per seed. Correct + term + category + sentiment partition eligible explicit gold. They are ordered first failures, not isolated causal effects.", "",
              "| Domain | Correct | Term failures | Category failures | Sentiment failures | Rare gold / recall | Unseen gold / recall |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        o, b = r["outcomes"], r["rarity"]
        lines.append(f"| {r['domain']} | {o['correct']:.1f} | {o['term']:.1f} | {o['category']:.1f} | {o['sentiment']:.1f} | {b['rare']['mean_gold']:.1f} / {fmt(b['rare']['recall'])} | {b['unseen']['mean_gold']:.1f} / {fmt(b['unseen']['recall'])} |")
    lines += ["", "## Interpretation boundaries", "",
              "- A closed-set category head cannot emit categories absent from its source training vocabulary. Zero category coverage forces exact-triplet F1 to zero even when terms or sentiment transfer.",
              "- Rare categories have 1–5 source training annotations; unseen categories have zero. Their recalls use pooled gold counts across seeds.",
              "- NULL prediction is disabled in this study. This report does not establish cross-domain implicit extraction performance.",
              "- The historical Phase 1 LODO results use an earlier eligibility protocol. Direct score differences require rescoring historical predictions against the same corrected eligible gold; this report does not claim that comparison has been performed.",
              "- Test results do not select hyperparameters, folds or checkpoints. Training source hashes, source data hashes and checkpoint contents were verified when the development cohort was frozen; the existing trainer checks provenance before evaluation.",
              "- The report builder reads saved metrics and frozen metadata only. The raw per-seed metrics, thresholds, configuration and paths appear in results.json; weights remain in the ignored artifacts folder.", "",
              "[Saved evidence](results.json) · [Development checkpoint freeze](selection.json)"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output / "report.md"
