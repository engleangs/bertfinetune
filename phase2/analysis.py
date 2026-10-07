"""Exact saved-prediction comparisons with matching gold and matched seeds."""

from collections import Counter
import json
import math
import statistics


def mean_sd(values):
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError("Scores must be a nonempty finite cohort")
    return {"mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def jsonl(archive, member):
    with archive.open(member) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def align_gold(legacy, current):
    """Keep every legacy prediction; replace gold only after sentence/id matching."""
    def indexed(rows):
        result = {}
        for row in rows:
            key = row["domain"], row["example_id"]
            if key in result:
                raise ValueError(f"Duplicate prediction example: {key}")
            result[key] = row
        return result

    old, new = indexed(legacy), indexed(current)
    if old.keys() != new.keys():
        raise ValueError("Historical and current test example identities differ")
    result = []
    for row in current:
        previous = old[row["domain"], row["example_id"]]
        if previous["sentence"] != row["sentence"]:
            raise ValueError("Historical/current sentence text differs")
        result.append({"domain": row["domain"], "example_id": row["example_id"],
                       "sentence": row["sentence"], "gold_triplets": row["gold_triplets"],
                       "predicted_triplets": previous["predicted_triplets"]})
    return result


def counts_to_scores(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn, "gold": tp + fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def score_records(records, source_categories):
    """Deduplicate within sentence; assign the first failed component per gold."""
    known = set(source_categories)
    projections = {"explicit": lambda t: t, "term": lambda t: t[0],
                   "term_category": lambda t: (t[0], t[1]),
                   "term_sentiment": lambda t: (t[0], t[2])}
    counts = {name: Counter() for name in projections}
    outcomes, known_gold = Counter(), 0
    for row in records:
        gold = {tuple(t) for t in row["gold_triplets"]}
        predicted = {tuple(t) for t in row["predicted_triplets"]}
        for name, projection in projections.items():
            g, p = {projection(t) for t in gold}, {projection(t) for t in predicted}
            counts[name].update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
        for item in gold:
            term, category, _ = item
            known_gold += category in known
            outcome = ("correct" if item in predicted else
                       "term" if not any(p[0] == term for p in predicted) else
                       "category" if not any(p[:2] == item[:2] for p in predicted) else "sentiment")
            outcomes[outcome] += 1
    metrics = {name: counts_to_scores(c["tp"], c["fp"], c["fn"]) for name, c in counts.items()}
    gold = metrics["explicit"]["gold"]
    if sum(outcomes.values()) != gold:
        raise ValueError("First-failure taxonomy does not partition gold")
    return {**metrics, "coverage": known_gold / gold if gold else 0.,
            "taxonomy": {key: outcomes[key] for key in ("correct", "term", "category", "sentiment")}}


def compare_lodo(saved, phase2_archive, phase1_archive):
    frozen = saved["freeze"]
    domains, seeds, trial = frozen["domains"], frozen["seeds"], frozen["trial_id"]
    saved_rows = {(r["domain"], r["seed"]): r for r in saved["runs"]}
    expected = {(d, s) for d in domains for s in seeds}
    if set(saved_rows) != expected or len(saved_rows) != len(saved["runs"]):
        raise ValueError("Saved LODO result matrix is incomplete or duplicated")
    rows = []
    for domain in domains:
        for seed in seeds:
            stem = f"experiments_v2/research/crossdomain/{domain}/{trial}/seed_{seed}/"
            current = jsonl(phase2_archive, stem + "test_predictions.jsonl")
            audit = json.loads(phase2_archive.read(stem + "audit.json"))
            result = score_records(current, audit["categories"])
            target = saved_rows[domain, seed]["metrics"]["explicit"]
            for ours, theirs in [("tp", "true_positives"), ("fp", "false_positives"), ("fn", "false_negatives")]:
                if result["explicit"][ours] != target[theirs]:
                    raise ValueError(f"Prediction-derived metrics differ from saved test metrics: {domain}/{seed}")
            rows.append({"domain": domain, "seed": seed, "model": "phase2_mixed_025", **result})
            for model in ("standard", "weighted"):
                stem_old = f"lodo_runs/{domain}/{model}/seed_{seed}/"
                legacy = jsonl(phase1_archive, stem_old + "test_predictions.jsonl")
                manifest = json.loads(phase1_archive.read(stem_old + "manifest.json"))
                # Verify historical metrics first, then rescore unchanged predictions.
                original = score_records(legacy, manifest["category_vocab"])
                previous = json.loads(phase1_archive.read(stem_old + "metrics.json"))["test"]["complete_triplet"]
                if not math.isclose(original["explicit"]["f1"], previous["micro_f1"], abs_tol=1e-12):
                    raise ValueError(f"Historical prediction metrics changed: {domain}/{model}/{seed}")
                matched = score_records(align_gold(legacy, current), manifest["category_vocab"])
                rows.append({"domain": domain, "seed": seed, "model": "phase1_" + model,
                             "original_gold": original["explicit"]["gold"],
                             "original_f1": original["explicit"]["f1"], **matched})
    summaries, paired = [], []
    for domain in domains:
        for model in ("phase1_standard", "phase1_weighted", "phase2_mixed_025"):
            cohort = [r for r in rows if r["domain"] == domain and r["model"] == model]
            if sorted(r["seed"] for r in cohort) != sorted(seeds):
                raise ValueError("Matched seed cohort is incomplete")
            metrics = {key: mean_sd([r[key]["f1"] for r in cohort])
                       for key in ("explicit", "term", "term_category", "term_sentiment")}
            summaries.append({"domain": domain, "model": model, "seeds": seeds,
                              "scores": metrics, "coverage": statistics.mean(r["coverage"] for r in cohort),
                              "gold": statistics.mean(r["explicit"]["gold"] for r in cohort),
                              "taxonomy": {key: statistics.mean(r["taxonomy"][key] for r in cohort)
                                           for key in ("correct", "term", "category", "sentiment")}})
        for baseline in ("phase1_standard", "phase1_weighted"):
            for seed in seeds:
                on = next(r for r in rows if (r["domain"], r["seed"], r["model"]) == (domain, seed, "phase2_mixed_025"))
                off = next(r for r in rows if (r["domain"], r["seed"], r["model"]) == (domain, seed, baseline))
                if on["explicit"]["gold"] != off["explicit"]["gold"]:
                    raise ValueError("Paired arms have different eligible gold")
                paired.append({"domain": domain, "seed": seed, "baseline": baseline,
                               "explicit_delta_pp": 100 * (on["explicit"]["f1"] - off["explicit"]["f1"]),
                               "term_delta_pp": 100 * (on["term"]["f1"] - off["term"]["f1"]),
                               "term_sentiment_delta_pp": 100 * (on["term_sentiment"]["f1"] - off["term_sentiment"]["f1"])})
    return {"method": "Replace historical gold with matched Phase 2 eligible explicit gold; keep every prediction and false positive. No model inference or test tuning.",
            "domains": domains, "seeds": seeds, "runs": rows, "summaries": summaries, "paired": paired}
