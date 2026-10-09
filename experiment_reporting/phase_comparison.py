"""Verify completed quick-study outputs and re-score Phase 1 on current gold."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from experiment_reporting.quick_selection import check_frozen_run, check_frozen_variants


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["domain"], row["example_id"])
            if key in records:
                raise ValueError(f"Duplicate example identity: {path}: {key}")
            records[key] = row
    return records


def align(reference, predictions, same_gold=False):
    if reference.keys() != predictions.keys():
        raise ValueError("Prediction files contain different sentence identities")
    for key, row in reference.items():
        candidate = predictions[key]
        if row["sentence"] != candidate["sentence"]:
            raise ValueError(f"Sentence contents differ: {key}")
        if same_gold:
            for field in ("gold_triplets", "gold_null_triplets"):
                if {tuple(t) for t in row[field]} != {tuple(t) for t in candidate[field]}:
                    raise ValueError(f"Current gold annotations differ: {key}, {field}")


def score(reference, predictions, scope="explicit"):
    if scope not in ("explicit", "null", "combined"):
        raise ValueError("Choose explicit, null or combined scope")
    counts = Counter(tp=0, fp=0, fn=0)
    for key, row in reference.items():
        candidate = predictions[key]
        gold = list(row["gold_triplets"]) if scope != "null" else []
        pred = list(candidate["predicted_triplets"]) if scope != "null" else []
        if scope != "explicit":
            gold += row["gold_null_triplets"]
            pred += candidate.get("predicted_null_triplets", [])
        gold, pred = {tuple(t) for t in gold}, {tuple(t) for t in pred}
        counts.update(tp=len(gold & pred), fp=len(pred - gold), fn=len(gold - pred))
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {**counts, "gold": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def primary_outcomes(reference, predictions):
    counts = Counter(correct=0, term=0, category=0, sentiment=0)
    for key, row in reference.items():
        predicted = {tuple(t) for t in predictions[key]["predicted_triplets"]}
        terms, pairs = {t[0] for t in predicted}, {t[:2] for t in predicted}
        for gold in {tuple(t) for t in row["gold_triplets"]}:
            outcome = ("correct" if gold in predicted else "term" if gold[0] not in terms
                       else "category" if gold[:2] not in pairs else "sentiment")
            counts[outcome] += 1
    return dict(counts)


def verify_metrics(records, metrics):
    for scope in ("explicit", "null", "combined"):
        actual, saved = score(records, records, scope), metrics[scope]
        for short, long in (("tp", "true_positives"), ("fp", "false_positives"), ("fn", "false_negatives")):
            if actual[short] != saved[long]:
                raise ValueError(f"Saved {scope} {long} does not match predictions")
        for name in ("f1", "precision", "recall"):
            if abs(actual[name] - saved["micro_" + name]) > 1e-12:
                raise ValueError(f"Saved {scope} {name} does not match predictions")
    taxonomy = primary_outcomes(records, records)
    if taxonomy != metrics["taxonomy"]["primary_outcome_counts"]:
        raise ValueError("Saved taxonomy does not match predictions")
    if sum(taxonomy.values()) != metrics["taxonomy"]["gold_triplets"]:
        raise ValueError("Taxonomy does not partition explicit gold")


def verify_study(frozen, root):
    """Check weights, provenance, saved predictions and development selection."""
    check_frozen_variants(frozen)
    checks, hashes = [], {}
    def cached_hash(path):
        key = str(Path(path).resolve())
        if key not in hashes:
            hashes[key] = file_hash(path)
        return hashes[key]
    for variant in frozen["variants"]:
        for run in variant["runs"]:
            directory = Path(run["directory"])
            manifest, metrics = check_frozen_run(variant, run)
            if manifest["status"] != "complete":
                raise ValueError(f"Test evaluation is incomplete: {directory}")
            if manifest["identity"]["mode"] != "indomain" or manifest["identity"]["smoke"]:
                raise ValueError("Phase comparison requires research in-domain runs")
            if cached_hash(directory / "best.pt") != run["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint contents changed: {directory}")
            for name, expected in manifest["identity"]["source_sha256"].items():
                if cached_hash(root / name) != expected:
                    raise ValueError(f"Training source changed: {name}")
            for inventory in (manifest["identity"]["source_data_sha256"], manifest["test_data_sha256"]):
                for name, expected in inventory.items():
                    if cached_hash(root / "data/m-absa" / name) != expected:
                        raise ValueError(f"Corpus contents changed: {name}")
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            if [r["epoch"] for r in history] != list(range(1, variant["config"]["epochs"] + 1)):
                raise ValueError(f"Training history is incomplete: {directory}")
            best = max(history, key=lambda r: (r["development"][variant["config"]["selection_metric"]]["micro_f1"],
                                               -r["development"]["unweighted_loss"], -r["epoch"]))
            if best["epoch"] != run["best_epoch"] or best["development"] != metrics:
                raise ValueError(f"Checkpoint does not match development selection: {directory}")
            for split in ("dev", "test"):
                records = read_records(directory / f"{split}_predictions.jsonl")
                report = json.loads((directory / f"{split}_metrics.json").read_text(encoding="utf-8"))
                verify_metrics(records, report)
            checks.append({"trial_id": variant["trial_id"], "seed": run["seed"], "status": "complete",
                           "best_epoch": run["best_epoch"], "threshold": run["null_threshold"],
                           "checkpoint_verified": True, "source_and_corpus_verified": True,
                           "predictions_and_metrics_verified": True})
    return {"verified_runs": len(checks), "runs": checks}


def compare_phase1(frozen, root):
    baseline = next(v for v in frozen["variants"] if "Best NULL off" in v["roles"])
    references = {run["seed"]: read_records(Path(run["directory"]) / "test_predictions.jsonl")
                  for run in baseline["runs"]}
    original, rows = [], []
    for seed in frozen["seeds"]:
        reference = references[seed]
        historical = read_records(root / "phase1_analysis/indomain_runs/standard" / f"seed_{seed}/test_predictions.jsonl")
        align(reference, historical)
        original.append({"seed": seed, **score(historical, historical)})
        models = [("phase1", ["Phase 1 standard, re-scored"], historical)]
        for variant in frozen["variants"]:
            run = next(r for r in variant["runs"] if r["seed"] == seed)
            predictions = read_records(Path(run["directory"]) / "test_predictions.jsonl")
            align(reference, predictions, same_gold=True)
            models.append((variant["trial_id"], variant["roles"], predictions))
        for model, roles, predictions in models:
            domains = {domain: score({key: row for key, row in reference.items() if row["domain"] == domain}, predictions)
                       for domain in sorted({row["domain"] for row in reference.values()})}
            rows.append({"model": model, "roles": roles, "seed": seed,
                         **{scope: score(reference, predictions, scope) for scope in ("explicit", "null", "combined")},
                         "taxonomy": primary_outcomes(reference, predictions), "domains": domains})
    summaries = []
    for model in dict.fromkeys(r["model"] for r in rows):
        members = [r for r in rows if r["model"] == model]
        summary = {"model": model, "roles": members[0]["roles"], "seeds": frozen["seeds"]}
        for scope in ("explicit", "null", "combined"):
            summary[scope] = {key: statistics.mean(r[scope][key] for r in members)
                              for key in ("precision", "recall", "f1", "gold", "tp", "fp", "fn")}
            summary[scope]["sd"] = statistics.stdev(r[scope]["f1"] for r in members) if len(members) > 1 else None
        summary["taxonomy"] = {key: statistics.mean(r["taxonomy"][key] for r in members)
                               for key in ("correct", "term", "category", "sentiment")}
        summary["domains"] = {domain: {"f1": statistics.mean(r["domains"][domain]["f1"] for r in members),
                                       "gold": members[0]["domains"][domain]["gold"]}
                              for domain in members[0]["domains"]}
        summaries.append(summary)
    return {"seeds": frozen["seeds"], "original_phase1": original, "runs": rows, "summaries": summaries,
            "method": "Saved predictions re-scored against identical current explicit+NULL gold; retain all predictions.",
            "scope": "In-domain; eligible explicit and deduplicated NULL; not all raw annotations."}
