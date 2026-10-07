"""Read-only train/dev diagnostics for the completed NULL study; no test access."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.experiment_data import load_source_splits, select_targets


ROOT = Path(__file__).resolve().parent


def scores(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def sentence_scope(explicit, nulls):
    if explicit and nulls:
        return "mixed"
    if nulls:
        return "null_only"
    return "explicit_only" if explicit else "neither_eligible"


def analyze(study):
    selection = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    variant, = [v for v in selection["variants"] if v["config"]["null_head"]]
    if variant["config"]["drop_conflict"]:
        raise ValueError("This diagnostic expects the completed four-sentiment protocol")
    train, dev, _ = load_source_splits(ROOT, "indomain")
    targets = {}
    for split, examples in (("train", train), ("dev", dev)):
        targets[split] = [select_targets(ex)[:2] for ex in examples]
    pair_counts = Counter((c, s) for _, nulls in targets["train"] for _, c, s in nulls)
    first = Path(variant["runs"][0]["directory"])
    audit = json.loads((first / "audit.json").read_text(encoding="utf-8"))
    categories, sentiments = audit["categories"], audit["sentiments"]
    num_labels = len(categories) * len(sentiments)
    report = {
        "scope": "Completed study train/dev only; diagnostic, no model or threshold changes",
        "trial_id": variant["trial_id"],
        "categories": len(categories), "sentiments": sentiments,
        "null_output_labels": num_labels,
        "positive_train_pairs": len(pair_counts),
        "train_pairs_support_1_to_5": sum(n <= 5 for n in pair_counts.values()),
        "labels_with_no_null_training_positive": num_labels - len(pair_counts),
        "train_null_targets": sum(pair_counts.values()),
        "positive_target_density": sum(pair_counts.values()) / (len(train) * num_labels),
        "sentence_scopes": {
            split: dict(Counter(sentence_scope(e, n) for e, n in rows))
            for split, rows in targets.items()
        },
        "dev_null_pair_coverage": dict(Counter(
            "unseen_category" if c not in categories else
            "unseen_sentiment" if s not in sentiments else
            "no_null_training_positive" if (c, s) not in pair_counts else
            "rare_null_training_pair" if pair_counts[c, s] <= 5 else "frequent_null_training_pair"
            for _, nulls in targets["dev"] for _, c, s in nulls
        )),
        "pair_support": [{"category": c, "sentiment": s, "count": n}
                         for (c, s), n in sorted(pair_counts.items())],
        "runs": [],
    }
    for run in sorted(variant["runs"], key=lambda r: r["seed"]):
        directory = Path(run["directory"])
        metric_file = directory / "dev_metrics.json"
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (directory / "dev_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(dev):
            raise ValueError("Development predictions do not match the source split length")
        diagnostics = Counter()
        for row, example, (explicit, nulls) in zip(rows, dev, targets["dev"]):
            if row["sentence"] != example.sentence or set(map(tuple, row["gold_null_triplets"])) != set(nulls):
                raise ValueError("Development sentence or NULL gold differs from the source")
            gold, predicted = set(nulls), set(map(tuple, row["predicted_null_triplets"]))
            for _, category, sentiment in predicted - gold:
                diagnostics["false_positive_pairs"] += 1
                if (category, sentiment) not in pair_counts:
                    diagnostics["fp_pair_without_null_training_positive"] += 1
                if not gold:
                    diagnostics["fp_on_sentence_without_gold_null"] += 1
            diagnostics["sentences_with_predicted_null"] += bool(predicted)
            diagnostics["sentences_with_gold_null"] += bool(gold)
            diagnostics["sentence_null_presence_tp"] += bool(gold and predicted)
            diagnostics["sentence_null_presence_fp"] += bool(predicted and not gold)
            diagnostics["sentence_null_presence_fn"] += bool(gold and not predicted)
        explicit = metrics["explicit"]
        curve = []
        for point in metrics["null_threshold_curve"]:
            joint = scores(*[explicit[key] + point[key] for key in
                             ("true_positives", "false_positives", "false_negatives")])
            curve.append({"threshold": point["threshold"], "null_f1": point["micro_f1"],
                          "null_tp": point["true_positives"], "null_fp": point["false_positives"],
                          "null_fn": point["false_negatives"], "combined": joint})
        selected, = [p for p in curve if p["threshold"] == metrics["null_threshold"]]
        if abs(selected["combined"]["f1"] - metrics["combined"]["micro_f1"]) > 1e-12:
            raise ValueError("Derived combined F1 differs from the saved development metric")
        best = max(curve, key=lambda p: (p["combined"]["f1"], -abs(p["threshold"] - metrics["null_threshold"])))
        report["runs"].append({
            "seed": run["seed"], "best_epoch": run["best_epoch"],
            "metrics_sha256": hashlib.sha256(metric_file.read_bytes()).hexdigest(),
            "saved_threshold": metrics["null_threshold"], "saved_point": selected,
            "best_combined_point_in_saved_grid": best,
            "prediction_diagnostics": dict(diagnostics),
            "sentence_null_presence": scores(diagnostics["sentence_null_presence_tp"],
                                             diagnostics["sentence_null_presence_fp"],
                                             diagnostics["sentence_null_presence_fn"]),
            "threshold_curve": curve,
        })
    return report


def plot_diagnostics(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from statistics import mean

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
    labels = ["With positive\nNULL supervision", "Without positive\nNULL supervision"]
    values = [report["positive_train_pairs"], report["labels_with_no_null_training_positive"]]
    bars = axes[0].bar(labels, values, color=["#257a82", "#b6c1ce"], width=0.55)
    axes[0].bar_label(bars, labels=[f"{v:,} ({v/report['null_output_labels']:.1%})" for v in values], padding=5)
    axes[0].set_ylim(0, 1040)
    axes[0].set_ylabel("Category–sentiment output pairs")
    axes[0].set_title("1,040 outputs; 201 observed NULL pairs", loc="left", fontsize=11)
    thresholds = [p["threshold"] for p in report["runs"][0]["threshold_curve"] if .1 <= p["threshold"] <= .5]
    for key, label, color in (("null_f1", "NULL F1", "#c87527"), ("combined", "Combined F1", "#257a82")):
        means = []
        for threshold in thresholds:
            points = [next(p for p in r["threshold_curve"] if p["threshold"] == threshold) for r in report["runs"]]
            means.append(mean(p[key]["f1"] if key == "combined" else p[key] for p in points))
        axes[1].plot(thresholds, means, "o-", color=color, label=label, linewidth=2)
    axes[1].axvline(.2, color="#555", linestyle="--", linewidth=1, label="Saved threshold")
    axes[1].set_xticks(thresholds)
    axes[1].set_ylim(0, .36)
    axes[1].set_xlabel("Threshold (saved development grid)")
    axes[1].set_ylabel("Mean F1 across five seeds")
    axes[1].set_title("Different selection objectives", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("NULL diagnosis: sparse supervision and a costly operating point", fontsize=14, x=.04, ha="left")
    fig.text(.04, .02, "Train/dev only. Lines connect evaluated points; intermediate thresholds have not been measured.", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, .055, 1, .92])
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "artifacts/quick_tuning_5seeds")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/literature_review/null_development_diagnostics.json")
    parser.add_argument("--plot", action="store_true", help="Also save a PNG next to the diagnostic JSON")
    args = parser.parse_args()
    report = analyze(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        plot_diagnostics(report, args.output.with_suffix(".png"))
    print(json.dumps({k: v for k, v in report.items() if k not in ("pair_support", "runs")}, indent=2))
    for run in report["runs"]:
        selected, best = run["saved_point"], run["best_combined_point_in_saved_grid"]
        print(f"Seed {run['seed']}: saved combined dev F1={selected['combined']['f1']:.4f}; "
              f"best saved-grid threshold={best['threshold']:g}, combined F1={best['combined']['f1']:.4f}; "
              f"NULL TP={best['null_tp']}, FP={best['null_fp']}")
        print(run["prediction_diagnostics"])
    print(f"Diagnostics: {args.output}")


if __name__ == "__main__":
    main()
