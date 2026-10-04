"""Evaluation with paired seeds and explicit diagnostics.

Reads completed runs and raw data; never trains or changes saved predictions.
Run from the repository: .venv/Scripts/python.exe analyze_team_notebook.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from src.data import aligned_explicit_triplets, label_frequencies, load_domain_file
from src.result_analysis import projected_micro_scores
from src.team_diagnostics import analyze_records
from src.team_report import write_report
from src.team_taxonomy import summarize_taxonomy


ROOT = Path(__file__).resolve().parent
SENTIMENTS = ["positive", "negative", "neutral", "conflict"]
OUTCOMES = ["correct", "term", "category", "sentiment"]


def load_corpus(root: Path):
    return {
        (domain, split): load_domain_file(
            str(root / "data" / "m-absa" / config.DOMAIN_FILES[domain][split]), domain
        )
        for domain in config.DOMAINS
        for split in ("train", "dev", "test")
    }


def audit_corpus(corpus):
    """Separate raw, deduplicated and model-retained annotation denominators."""
    sentiment_rows, null_rows, null_label_rows = [], [], []
    for (domain, split), examples in sorted(corpus.items()):
        raw = [t for ex in examples for t in ex.triplets]
        unique = [t for ex in examples for t in set(ex.triplets)]
        retained = [t for ex in examples for t in aligned_explicit_triplets(ex)]
        null = [t for t in unique if t[0].strip().casefold() == "null"]
        for scope, triplets in [("raw", raw), ("deduplicated", unique), ("retained", retained)]:
            counts = Counter(t[2] for t in triplets)
            for label in SENTIMENTS:
                sentiment_rows.append({
                    "domain": domain, "split": split, "scope": scope,
                    "sentiment": label, "count": counts[label],
                    "total": len(triplets),
                    "percentage": 100 * counts[label] / len(triplets) if triplets else 0.0,
                })
        types = Counter()
        for ex in examples:
            gold = set(ex.triplets)
            has_null = any(t[0].strip().casefold() == "null" for t in gold)
            has_explicit = any(t[0].strip().casefold() != "null" for t in gold)
            kind = "mixed" if has_null and has_explicit else "null_only" if has_null else "explicit_only" if has_explicit else "empty"
            types[kind] += 1
        null_rows.append({
            "domain": domain, "split": split, "examples": len(examples),
            "raw_triplets": len(raw), "unique_triplets": len(unique),
            "null_triplets": len(null),
            "null_percentage": 100 * len(null) / len(unique) if unique else 0.0,
            "retained_explicit_triplets": len(retained),
            # Annotation-scope bound, not a tokenizer-aware achievable ceiling.
            "scope_recall_upper_bound": len(retained) / len(unique) if unique else 0.0,
            "null_only_examples": types["null_only"],
            "explicit_only_examples": types["explicit_only"],
            "mixed_examples": types["mixed"], "empty_examples": types["empty"],
        })
        for (category, sentiment), count in sorted(Counter((t[1], t[2]) for t in null).items()):
            null_label_rows.append({
                "domain": domain, "split": split, "category": category,
                "sentiment": sentiment, "count": count,
            })
    return pd.DataFrame(sentiment_rows), pd.DataFrame(null_rows), pd.DataFrame(null_label_rows)


def pair_runs(frame: pd.DataFrame):
    """Pair by condition, domain and seed; never align by domain alone."""
    keys = ["setting", "held_out_domain", "model", "seed"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate logical run keys; do not average duplicate copies")
    standard = frame[frame.model == "standard"]
    weighted = frame[frame.model == "weighted"]
    paired = standard.merge(
        weighted, on=["setting", "held_out_domain", "seed"],
        suffixes=("_standard", "_weighted"), how="outer", validate="one_to_one", indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        raise ValueError("Unmatched standard/weighted seeds; complete pairs before comparison")
    paired = paired.drop(columns="_merge")
    for metric in ["precision", "recall", "f1", "aspect_f1", "aspect_sentiment_f1"]:
        paired[f"{metric}_change"] = paired[f"{metric}_weighted"] - paired[f"{metric}_standard"]
    return paired


def summarize_runs(frame: pd.DataFrame):
    values = ["precision", "recall", "f1", "aspect_f1", "aspect_sentiment_f1", "term_category_f1", "known_category_f1", "category_coverage"]
    rows = []
    for keys, group in frame.groupby(["setting", "held_out_domain", "model"], dropna=False):
        row = dict(zip(["setting", "held_out_domain", "model"], keys))
        row.update(n=len(group), seed_ids=",".join(str(s) for s in sorted(group.seed)),
                   complete=set(group.seed) == set(config.SEEDS))
        for metric in values:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_analysis(root: Path = ROOT, progress=None):
    """Validate saved runs; optionally report progress to a callable."""
    root = Path(root).resolve()
    if progress is not None:
        progress("Loading and auditing 21 domain/split data files...")
    corpus = load_corpus(root)
    sentiments, nulls, null_labels = audit_corpus(corpus)
    run_rows, error_rows, rarity_rows, confusion_rows, condition_rows, domain_rows = [], [], [], [], [], []
    summary_json, frequencies, data_hashes = {}, {}, {}
    for domain in config.DOMAINS:
        for split in ("train", "dev", "test"):
            path = root / "data" / "m-absa" / config.DOMAIN_FILES[domain][split]
            data_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()

    indexes = [pd.read_csv(root / name) for name in ["results.csv", "results_lodo.csv"]]
    total_runs = sum(int((index.status == "complete").sum()) for index in indexes)
    if progress is not None:
        progress(f"Validating predictions and computing diagnostics for {total_runs} completed runs...")
    for index in indexes:
        for row in index[index.status == "complete"].to_dict("records"):
            run_dir = Path(row["artifact_dir"])
            # Allow relocating the repository without trusting an old drive path.
            if not run_dir.exists():
                if row["mode"] == "indomain" or "held_out_domain" not in row:
                    run_dir = root / "artifacts" / "runs" / row["mode"] / row["config"] / f"seed_{int(row['seed'])}"
                else:
                    run_dir = root / "artifacts" / "lodo_runs" / row["held_out_domain"] / row["config"] / f"seed_{int(row['seed'])}"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            records = [json.loads(line) for line in (run_dir / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if manifest["status"] != "complete" or manifest["seed"] != int(row["seed"]) or manifest["config_name"] != row["config"] or manifest["mode"] != row["mode"]:
                raise ValueError(f"Index/manifest mismatch: {run_dir}")
            setting = "In-domain" if row["mode"] == "indomain" else "LODO"
            domains = {r["domain"] for r in records}
            if setting == "LODO" and len(domains) != 1:
                raise ValueError(f"LODO predictions must contain one held-out domain: {run_dir}")
            held_out = next(iter(domains)) if setting == "LODO" else "all"
            if setting == "LODO" and (manifest.get("held_out_domain") not in (None, held_out) or ("held_out_domain" in row and row["held_out_domain"] != held_out)):
                raise ValueError(f"Held-out domain mismatch: {run_dir}")
            source_domains = [d for d in config.DOMAINS if d != held_out] if setting == "LODO" else config.DOMAINS
            if held_out not in frequencies:
                train = [ex for d in source_domains for ex in corpus[(d, "train")]]
                frequencies[held_out] = label_frequencies(train)[0]
            counts = frequencies[held_out]
            if set(counts) != set(manifest["category_vocab"]):
                raise ValueError(f"Training vocabulary differs from the saved manifest: {run_dir}")
            if sum(counts.values()) != manifest["data_summary"]["train"]["included_explicit_triplets"]:
                raise ValueError(f"Training scope differs from the saved manifest: {run_dir}")
            test_examples = [ex for d in (config.DOMAINS if setting == "In-domain" else [held_out]) for ex in corpus[(d, "test")]]
            if len(records) != len(test_examples):
                raise ValueError(f"Test sentence count differs: {run_dir}")
            for position, (record, example) in enumerate(zip(records, test_examples)):
                if record["example_id"] != position or record["sentence"] != example.sentence or record["domain"] != example.domain or {tuple(t) for t in record["gold_triplets"]} != set(aligned_explicit_triplets(example)):
                    raise ValueError(f"Test gold/data mismatch at {position}: {run_dir}")
            diagnostics, gold_rows = analyze_records(records, counts, manifest["category_vocab"])
            derived = diagnostics["metrics"]
            full = derived["full_exact_triplet"]
            for saved_name, name in [("micro_precision", "precision"), ("micro_recall", "recall"), ("micro_f1", "f1")]:
                saved = metrics["test"]["complete_triplet"][saved_name]
                if not math.isclose(full[name], saved, abs_tol=1e-10) or not math.isclose(full[name], float(row[f"test_{saved_name}"]), abs_tol=1e-10):
                    raise ValueError(f"Saved score differs from recomputed predictions: {run_dir}")
            identity = {"setting": setting, "held_out_domain": held_out, "model": row["config"], "seed": int(row["seed"])}
            run_rows.append({
                **identity, **full,
                "aspect_f1": derived["aspect_span"]["f1"],
                "aspect_sentiment_f1": derived["aspect_plus_sentiment"]["f1"],
                "term_category_f1": derived["term_plus_category"]["f1"],
                "known_category_f1": derived["source_known_exact_triplet"]["f1"] if derived["source_known_gold_triplets"] else float("nan"),
                "category_coverage": derived["gold_category_coverage"],
                "gold_triplets": derived["gold_triplets"],
                "unseen_gold_triplets": derived["unseen_gold_triplets"],
                "artifact_dir": str(run_dir),
                **diagnostics["primary_outcome_counts"],
                "spurious_aspects": diagnostics["prediction_counts"]["spurious_aspects"],
                "duplicate_predictions": diagnostics["duplicate_counts"]["predicted"],
                "category_mismatch_correct_term_sentiment": diagnostics["category_mismatch_with_correct_term_sentiment"],
            })
            error_rows.extend({**identity, **item} for item in gold_rows)
            for details in diagnostics["by_rarity"]:
                rarity_rows.append({**identity, "rarity": details["rarity"],
                                    "gold_triplets": details["gold_triplets"],
                                    "error_rate": details["error_rate"],
                                    **details["primary_outcome_counts"]})
            conditional = {f"{condition}_{measure}": value
                           for condition, statistics in diagnostics["conditional_accuracy"].items()
                           for measure, value in statistics.items()}
            condition_rows.append({**identity, **conditional})
            confusion = diagnostics["sentiment_confusion"]
            for gi, gold_label in enumerate(confusion["labels"]):
                for pi, pred_label in enumerate(confusion["labels"]):
                    confusion_rows.append({**identity, "gold_sentiment": gold_label, "predicted_sentiment": pred_label, "count": confusion["matrix"][gi][pi]})
            summary_json["/".join(str(identity[k]) for k in identity)] = diagnostics
            for domain in sorted(domains):
                domain_records = [r for r in records if r["domain"] == domain]
                score = projected_micro_scores(
                    [[tuple(t) for t in r["gold_triplets"]] for r in domain_records],
                    [[tuple(t) for t in r["predicted_triplets"]] for r in domain_records],
                )
                outcomes = Counter(r["outcome"] for r in gold_rows if r["domain"] == domain)
                domain_rows.append({**identity, "domain": domain, **score,
                                    **{outcome: outcomes[outcome] for outcome in OUTCOMES}})
            if progress is not None and (len(run_rows) % 10 == 0 or len(run_rows) == total_runs):
                progress(f"Validated {len(run_rows)}/{total_runs} runs")

    runs = pd.DataFrame(run_rows)
    paired = pair_runs(runs)
    errors = pd.DataFrame(error_rows)
    expected = {(s, d, m, seed) for s, domains in [("In-domain", ["all"]), ("LODO", config.DOMAINS)] for d in domains for m in ["standard", "weighted"] for seed in config.SEEDS}
    actual = set(runs[["setting", "held_out_domain", "model", "seed"]].itertuples(index=False, name=None))
    return {
        "runs": runs, "summary": summarize_runs(runs), "paired": paired,
        "taxonomy_summary": summarize_taxonomy(runs),
        "errors": errors, "domain_metrics": pd.DataFrame(domain_rows),
        "rarity": pd.DataFrame(rarity_rows), "conditional": pd.DataFrame(condition_rows),
        "sentiment_confusion": pd.DataFrame(confusion_rows),
        "sentiment_counts": sentiments, "null_prevalence": nulls,
        "null_labels": null_labels, "run_diagnostics": summary_json,
        "metadata": {"review_date": "2026-10-04", "analysis_protocol_version": "1.1",
                     "data_sha256": data_hashes,
                     "missing_run_keys": sorted(expected - actual),
                     "unexpected_run_keys": sorted(actual - expected),
                     "counts_basis": "Current version-1 retained annotation scope; raw NULL counts deduplicated per sentence"},
    }



def write_outputs(analysis, output: Path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in analysis.items():
        if isinstance(value, pd.DataFrame):
            if name == "errors":
                value.to_json(output / "gold_error_records.jsonl", orient="records", lines=True, force_ascii=False)
            else:
                value.to_csv(output / f"{name}.csv", index=False)
        else:
            (output / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "team_analysis")
    parser.add_argument("--report", type=Path, default=ROOT / "detailed-analysis.md")
    args = parser.parse_args()
    def progress(message):
        print(message, flush=True)

    result = build_analysis(args.root, progress=progress)
    progress("Writing analysis tables, diagnostic records, and the report...")
    write_outputs(result, args.output)
    write_report(result, args.report)
    columns = ["setting", "held_out_domain", "model", "n", "f1_mean", "f1_std", "aspect_f1_mean", "aspect_sentiment_f1_mean", "category_coverage_mean"]
    print(result["summary"][columns].round(4).to_string(index=False))
    print(f"Validated {len(result['runs'])} runs; missing planned keys: {len(result['metadata']['missing_run_keys'])}")
    print(f"Outputs: {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
