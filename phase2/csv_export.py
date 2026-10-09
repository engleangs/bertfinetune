"""Export saved run scores with the supplied CSV schema and matched-gold comparisons."""

from collections import defaultdict
import csv
import json
import math

try:
    from .analysis import align_gold, counts_to_scores, jsonl, mean_sd, score_records
except ImportError:
    from analysis import align_gold, counts_to_scores, jsonl, mean_sd, score_records

CSV_COLUMNS = [
    "mode", "domain", "trial_id", "recipe", "seed", "scope", "smoke", "best_epoch",
    "explicit_f1", "explicit_precision", "explicit_recall", "null_f1", "combined_f1",
    "boundary_f1", "null_threshold", "run_directory", "term_f1", "term_category_f1",
    "term_sentiment_f1", "known_category_coverage", "term_errors", "category_errors",
    "sentiment_errors", "rare_gold", "rare_recall",
]
RATE_COLUMNS = [
    "explicit_f1", "explicit_precision", "explicit_recall", "null_f1", "combined_f1",
    "term_f1", "term_category_f1", "term_sentiment_f1", "known_category_coverage", "rare_recall",
]


def read_json(archive, name):
    return json.loads(archive.read(name))


def save_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def legacy_identity(stem, manifest):
    parts = stem.strip("/").split("/")
    held_out = parts[1] if parts[0] == "lodo_runs" else None
    if manifest.get("held_out_domain") not in (None, held_out):
        raise ValueError("Historical manifest and canonical fold path disagree")
    return {"mode": "crossdomain" if parts[0] == "lodo_runs" else "indomain",
            "domain": held_out if parts[0] == "lodo_runs" else "all_domains",
            "trial_id": "phase1_" + manifest["config_name"], "recipe": manifest["config_name"],
            "seed": manifest["seed"], "scope": "phase1_historical_explicit", "smoke": False,
            "best_epoch": manifest["best_epoch"],
            "run_directory": "evidence/phase1_outputs.zip::" + stem.rstrip("/")}


def verify_counts(derived, saved, context):
    for our_key, saved_key in (("tp", "true_positives"), ("fp", "false_positives"), ("fn", "false_negatives")):
        if derived[our_key] != saved[saved_key]:
            raise ValueError(f"Saved prediction counts differ: {context}/{our_key}")
    if not math.isclose(derived["f1"], saved["micro_f1"], abs_tol=1e-12):
        raise ValueError(f"Saved F1 differs from predictions: {context}")


def legacy_row(identity, manifest, records, *, null_gold=None):
    """Native historical NULL/boundary metrics stay blank when never measured."""
    scored = score_records(records, manifest["category_vocab"])
    explicit = scored["explicit"]
    rare_labels = set(manifest["rare_category_labels"])
    rare_gold = rare_correct = 0
    for record in records:
        gold = {tuple(t) for t in record["gold_triplets"] if t[1] in rare_labels}
        predicted = {tuple(t) for t in record["predicted_triplets"]}
        rare_gold += len(gold)
        rare_correct += len(gold & predicted)
    combined = None if null_gold is None else counts_to_scores(
        explicit["tp"], explicit["fp"], explicit["fn"] + null_gold)["f1"]
    return {**identity, "explicit_f1": explicit["f1"],
            "explicit_precision": explicit["precision"], "explicit_recall": explicit["recall"],
            "null_f1": None if null_gold is None else 0., "combined_f1": combined,
            "boundary_f1": None, "null_threshold": None,
            "term_f1": scored["term"]["f1"], "term_category_f1": scored["term_category"]["f1"],
            "term_sentiment_f1": scored["term_sentiment"]["f1"], "known_category_coverage": scored["coverage"],
            "term_errors": scored["taxonomy"]["term"], "category_errors": scored["taxonomy"]["category"],
            "sentiment_errors": scored["taxonomy"]["sentiment"],
            "rare_gold": rare_gold, "rare_recall": rare_correct/rare_gold if rare_gold else None}


def phase1_runs(archive):
    exports, parameters = {"dev": [], "test": []}, []
    stems = sorted(name.removesuffix("manifest.json") for name in archive.namelist()
                   if name.endswith("/manifest.json"))
    for stem in stems:
        manifest = read_json(archive, stem + "manifest.json")
        identity = legacy_identity(stem, manifest)
        metrics = read_json(archive, stem + "metrics.json")
        for split in ("dev", "test"):
            records = jsonl(archive, stem + split + "_predictions.jsonl")
            scored = score_records(records, manifest["category_vocab"])
            verify_counts(scored["explicit"], metrics[split]["complete_triplet"], stem + split)
            exports[split].append(legacy_row(identity, manifest, records))
        parameters.append({**identity, **manifest["configuration"],
                           "null_head": False, "selection_metric": manifest["selection_metric"]})
    if len(stems) != 80:
        raise ValueError("Phase 1 requires the full historical 80-run matrix")
    return exports, parameters


def current_identity(stem, manifest):
    identity, config = manifest["identity"], manifest["identity"]["config"]
    parts = stem.rstrip("/").split("/")
    mode = identity["mode"]
    return {"mode": mode, "domain": identity.get("held_out_domain") or "all_domains",
            "trial_id": parts[-2], "recipe": config["name"], "seed": identity["seed"],
            "scope": "research", "smoke": identity["smoke"],
            "best_epoch": manifest.get("training", {}).get("best_epoch"),
            "run_directory": "evidence/phase2_outputs.zip::" + stem.rstrip("/")}


def current_row(identity, metrics):
    taxonomy = metrics["taxonomy"]
    components, outcomes = taxonomy["metrics"], taxonomy["primary_outcome_counts"]
    rare = next(row for row in taxonomy["by_rarity"] if row["rarity"] == "rare")
    if sum(outcomes.values()) != taxonomy["gold_triplets"]:
        raise ValueError("Saved taxonomy does not partition gold")
    explicit = metrics["explicit"]
    return {**identity, "explicit_f1": explicit["micro_f1"],
            "explicit_precision": explicit["micro_precision"], "explicit_recall": explicit["micro_recall"],
            "null_f1": metrics["null"]["micro_f1"], "combined_f1": metrics["combined"]["micro_f1"],
            "boundary_f1": metrics["explicit_boundary"]["f1"], "null_threshold": metrics["null_threshold"],
            "term_f1": components["aspect_span"]["f1"], "term_category_f1": components["term_plus_category"]["f1"],
            "term_sentiment_f1": components["aspect_plus_sentiment"]["f1"],
            "known_category_coverage": components["gold_category_coverage"],
            "term_errors": outcomes["term"], "category_errors": outcomes["category"],
            "sentiment_errors": outcomes["sentiment"], "rare_gold": rare["gold_triplets"],
            "rare_recall": rare["primary_outcome_counts"]["correct"]/rare["gold_triplets"]
                           if rare["gold_triplets"] else None}


def registered_cohorts(data, lodo):
    result = defaultdict(set)
    for collection, cohort in (("runs", "finalist"), ("screen_runs", "screen")):
        for row in data[collection]:
            result[row["trial_id"], row["seed"]].add(cohort + ": " + row["study"] + "/" + row["role"])
    for row in lodo["runs"]:
        result[lodo["freeze"]["trial_id"], row["seed"], row["domain"]].add("LODO finalist")
    return result


def phase2_runs(archive, cohorts):
    exports, parameters = {"dev": [], "test": []}, []
    members = set(archive.namelist())
    stems = sorted(name.removesuffix("manifest.json") for name in members if name.endswith("/manifest.json"))
    for stem in stems:
        manifest = read_json(archive, stem + "manifest.json")
        identity = current_identity(stem, manifest)
        if identity["smoke"]:
            raise ValueError("Full research archive unexpectedly contains smoke runs")
        key = (identity["trial_id"], identity["seed"], identity["domain"]) if identity["mode"] == "crossdomain" else (
            identity["trial_id"], identity["seed"])
        completed = manifest["status"] in {"complete", "dev_complete"}
        parameters.append({**identity, **manifest["identity"]["config"], "status": manifest["status"],
                           "cohort": " | ".join(sorted(cohorts.get(key, {"other bundled run"}))),
                           "has_test_results": stem + "test_metrics.json" in members,
                           "included_in_score_exports": completed})
        if not completed:
            continue
        if identity["best_epoch"] is None:
            raise ValueError(f"Completed run is missing its selected checkpoint epoch: {stem}")
        audit = read_json(archive, stem + "audit.json")
        for split in ("dev", "test"):
            metric_name = stem + split + "_metrics.json"
            if metric_name not in members:
                continue
            metrics = read_json(archive, metric_name)
            records = jsonl(archive, stem + split + "_predictions.jsonl")
            scored = score_records(records, audit["categories"])
            verify_counts(scored["explicit"], metrics["explicit"], stem + split)
            exports[split].append(current_row(identity, metrics))
    for split, rows in exports.items():
        keys = [(r["mode"], r["domain"], r["trial_id"], r["seed"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate Phase 2 logical runs: {split}")
    return exports, parameters


def comparison_stems(data, lodo):
    selected = next(row for row in data["runs"] if row["study"] == "Quick tuning"
                    and row["role"] == "Best NULL off" and row["split"] == "test")
    trial = selected["trial_id"]
    if selected["config"]["null_head"] or trial != lodo["freeze"]["trial_id"]:
        raise ValueError("LODO and in-domain comparison must use the frozen NULL-off recipe")
    jobs = []
    for mode, domain, seeds in [("indomain", "all_domains", selected["expected_seeds"])] + [
        ("crossdomain", domain, lodo["freeze"]["seeds"]) for domain in lodo["freeze"]["domains"]]:
        for seed in seeds:
            jobs.append((mode, domain, seed,
                         f"experiments_v2/research/{mode}/{domain}/{trial}/seed_{seed}/"))
    return jobs


def paired_comparisons(legacy, current, jobs):
    exports = {"dev": [], "test": []}
    for mode, domain, seed, stem in jobs:
        manifest2 = read_json(current, stem + "manifest.json")
        for split in ("dev", "test"):
            records = jsonl(current, stem + split + "_predictions.jsonl")
            metrics = read_json(current, stem + split + "_metrics.json")
            new = current_row(current_identity(stem, manifest2), metrics)
            null_gold = sum(len({tuple(t[1:]) for t in r["gold_null_triplets"]}) for r in records)
            for baseline in ("standard", "weighted"):
                prefix = (f"indomain_runs/{baseline}/seed_{seed}/" if mode == "indomain" else
                          f"lodo_runs/{domain}/{baseline}/seed_{seed}/")
                manifest1 = read_json(legacy, prefix + "manifest.json")
                old_records = jsonl(legacy, prefix + split + "_predictions.jsonl")
                aligned = align_gold(old_records, records)
                old = legacy_row(legacy_identity(prefix, manifest1), manifest1, aligned, null_gold=null_gold)
                identity = {"mode": mode, "domain": domain, "baseline": baseline, "seed": seed,
                            "evaluation_scope": "same_corrected_phase2_gold",
                            "phase2_trial_id": new["trial_id"], "phase2_recipe": new["recipe"],
                            "explicit_gold": sum(len({tuple(t) for t in r["gold_triplets"]}) for r in records),
                            "null_gold": null_gold, "phase1_best_epoch": old["best_epoch"],
                            "phase2_best_epoch": new["best_epoch"]}
                row = dict(identity)
                for field in RATE_COLUMNS + ["term_errors", "category_errors", "sentiment_errors", "rare_gold"]:
                    row["phase1_" + field], row["phase2_" + field] = old[field], new[field]
                    if field in RATE_COLUMNS:
                        row[field + "_delta_pp"] = (100*(new[field]-old[field])
                                                   if old[field] is not None and new[field] is not None else None)
                exports[split].append(row)
    return exports


def summarize_pairs(exports, expected_seeds):
    groups = defaultdict(list)
    for split, rows in exports.items():
        for row in rows:
            groups[split, row["mode"], row["domain"], row["baseline"]].append(row)
    result = []
    for (split, mode, domain, baseline), cohort in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in cohort)
        if seeds != sorted(expected_seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("Same-gold comparison has missing or duplicated paired seeds")
        if len({(r["explicit_gold"], r["null_gold"]) for r in cohort}) != 1:
            raise ValueError("Eligible gold counts differ between paired seeds")
        row = {"split": split, "mode": mode, "domain": domain, "baseline": baseline,
               "n": len(cohort), "seeds": " ".join(map(str, seeds)),
               "explicit_gold": cohort[0]["explicit_gold"], "null_gold": cohort[0]["null_gold"]}
        for field in ("explicit_f1", "combined_f1", "term_f1", "term_sentiment_f1"):
            for label, key in (("phase1", "phase1_" + field), ("phase2", "phase2_" + field),
                               ("delta_pp", field + "_delta_pp")):
                summary = mean_sd([r[key] for r in cohort])
                row[field + "_" + label + "_mean"] = summary["mean"]
                row[field + "_" + label + "_sd"] = summary["sd"]
        result.append(row)
    return result


def native_summaries(exports):
    groups = defaultdict(list)
    for split, rows in exports.items():
        for row in rows:
            groups[split, row["mode"], row["domain"], row["trial_id"], row["recipe"]].append(row)
    result = []
    for (split, mode, domain, trial, recipe), cohort in sorted(groups.items()):
        row = {"split": split, "mode": mode, "domain": domain, "trial_id": trial,
               "recipe": recipe, "n": len(cohort), "seeds": " ".join(map(str, sorted(r["seed"] for r in cohort)))}
        for field in ("explicit_f1", "null_f1", "combined_f1", "term_f1", "term_sentiment_f1"):
            values = [r[field] for r in cohort if r[field] is not None]
            score = mean_sd(values) if values else {"mean": None, "sd": None}
            row[field + "_mean"], row[field + "_sd"] = score["mean"], score["sd"]
        result.append(row)
    return result


def template_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError("The supplied development_results.csv schema changed")
        rows = list(reader)
    if any(row["smoke"].casefold() != "true" for row in rows):
        raise ValueError("The preserved example must stay separate from full research runs")
    return rows
