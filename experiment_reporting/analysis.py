"""Load saved development artifacts and compare only compatible, matched runs."""

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo


COMPLETE = {"dev_complete", "complete"}
SCORES = {
    "explicit_f1": ("explicit", "micro_f1"),
    "precision": ("explicit", "micro_precision"),
    "recall": ("explicit", "micro_recall"),
    "macro_f1": ("explicit", "macro_f1"),
    "null_f1": ("null", "micro_f1"),
    "combined_f1": ("combined", "micro_f1"),
    "boundary_f1": ("explicit_boundary", "f1"),
}
COMPONENTS = {"term_f1": "aspect_span", "term_category_f1": "term_plus_category",
              "term_sentiment_f1": "aspect_plus_sentiment"}


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def effective_scope(cfg):
    scope = cfg.get("vocabulary_scope", "auto")
    return ("explicit-null" if cfg.get("null_head") else "explicit") if scope == "auto" else scope


def cohort_identity(identity, audit):
    """Different task, budget, vocabulary, data or training versions stay separate."""
    cfg = identity["config"]
    budget = {key: cfg.get(key) for key in ("model_name", "model_revision", "lr", "epochs", "batch_size",
              "max_len", "weight_decay", "warmup_ratio", "max_grad_norm", "drop_conflict", "selection_metric")}
    payload = {"budget": budget, "scope": effective_scope(cfg),
               "mode": identity["mode"], "domain": identity.get("held_out_domain"),
               "smoke": identity.get("smoke", False), "split_limits": identity.get("split_limits"),
               "data": identity.get("source_data_sha256"), "code": identity.get("source_sha256"),
               "environment": identity.get("environment"), "device": identity.get("resolved_device"),
               "categories": audit.get("categories"), "sentiments": audit.get("sentiments")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]


def flatten_metrics(report):
    result = {name: report.get(group, {}).get(key) for name, (group, key) in SCORES.items()}
    taxonomy = report.get("taxonomy", {})
    components = taxonomy.get("metrics", {})
    result.update({name: components.get(key, {}).get("f1") for name, key in COMPONENTS.items()})
    result["coverage"] = components.get("gold_category_coverage")
    known = components.get("source_known_exact_triplet")
    result["known_f1"] = known.get("f1") if known is not None else None
    result["gold_triplets"] = taxonomy.get("gold_triplets", 0)
    result["outcomes"] = taxonomy.get("primary_outcome_counts", {})
    result["conditional"] = taxonomy.get("conditional_accuracy", {})
    rare = next((r for r in taxonomy.get("by_rarity", []) if r["rarity"] == "rare"), {})
    result["rare_recall"] = 1 - rare["error_rate"] if rare.get("error_rate") is not None else None
    result["rare_gold"] = rare.get("gold_triplets", 0)
    result["null_gold"] = report.get("null_gold", 0)
    result["null_threshold"] = report.get("null_threshold")
    result["unweighted_loss"] = report.get("unweighted_loss")
    for name in (*SCORES, *COMPONENTS):
        value = result[name]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid {name}: {value}")
    return result


def load_runs(root, mode, domain, include_smoke=False):
    runs, notices = [], []
    for path in sorted(root.glob("**/manifest.json")):
        try:
            manifest = read_json(path)
            identity = manifest["identity"]
            if identity["mode"] != mode or identity.get("held_out_domain") != domain:
                continue
            if identity.get("smoke") and not include_smoke:
                continue
            cfg = identity["config"]
            audit = read_json(path.parent / "audit.json", {})
            history = read_json(path.parent / "history.json", [])
            completed = manifest["status"] in COMPLETE
            report = read_json(path.parent / "dev_metrics.json") if completed else None
            # History is atomically written before checkpoint artifacts. For a running
            # trial it is a consistent snapshot of fully evaluated epochs.
            best_epoch = manifest.get("training", {}).get("best_epoch")
            if not completed and history:
                best = max(history, key=lambda r: (r["development"][cfg["selection_metric"]]["micro_f1"],
                           -r["development"]["unweighted_loss"], -r["epoch"]))
                report, best_epoch = best["development"], best["epoch"]
            if completed and report is None:
                raise ValueError("Completed run is missing dev_metrics.json")
            cohort = cohort_identity(identity, audit)
            scope = effective_scope(cfg)
            label = f"{mode} / {domain or 'all domains'} / {scope} vocabulary"
            if identity.get("smoke"):
                label += f" / SMOKE {identity.get('split_limits')}"
            training = manifest.get("training", {})
            runs.append({"trial_id": path.parent.parent.name, "recipe": cfg["name"], "seed": identity["seed"],
                         "cohort": cohort, "cohort_label": label, "config": cfg, "status": manifest["status"],
                         "completed": completed, "provisional": report is not None and not completed,
                         "smoke": identity.get("smoke", False),
                         "epochs_recorded": len(history), "best_epoch": best_epoch,
                         "metrics": flatten_metrics(report) if report else None,
                         "history": [{"epoch": r["epoch"], "f1": r["development"]["explicit"]["micro_f1"],
                                      "precision": r["development"]["explicit"]["micro_precision"],
                                      "recall": r["development"]["explicit"]["micro_recall"],
                                      "loss": r["development"]["unweighted_loss"]} for r in history],
                         "per_domain": report.get("per_domain", {}) if report else {},
                         "vocabulary_scope": scope, "num_categories": len(audit.get("categories", [])),
                         "sentiments": audit.get("sentiments", []), "training": training,
                         "audit": {key: audit.get(key) for key in ("train", "dev", "tokenized_train", "tokenized_dev", "frequency_scope")},
                         "directory": str(path.parent.resolve()), "fingerprint": manifest["fingerprint"],
                         "error": manifest.get("error")})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            notices.append(f"Could not read {path}: {exc}")
    keys = [(r["cohort"], r["trial_id"], r["seed"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate trial/seed within the same comparable cohort")
    return runs, notices


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def sample_std(values):
    values = [value for value in values if value is not None]
    return statistics.stdev(values) if len(values) >= 2 else None


def summaries(runs):
    groups = defaultdict(list)
    for run in runs:
        if run["metrics"] is not None:
            groups[(run["cohort"], run["trial_id"])].append(run)
    output = []
    for (cohort, trial), members in sorted(groups.items()):
        complete = [r for r in members if r["completed"]]
        # Never pool a provisional epoch into a finished seed mean.
        selected = complete or members
        fields = (*SCORES, *COMPONENTS, "coverage", "known_f1", "rare_recall")
        row = {"cohort": cohort, "cohort_label": members[0]["cohort_label"], "trial_id": trial,
               "recipe": members[0]["recipe"], "config": members[0]["config"],
               "completed_seeds": sorted(r["seed"] for r in complete),
               "scored_seeds": sorted(r["seed"] for r in selected), "provisional": not complete,
               "n": len(selected)}
        row.update({f"{name}_mean": mean([r["metrics"][name] for r in selected]) for name in fields})
        row.update({f"{name}_std": sample_std([r["metrics"][name] for r in selected]) for name in fields})
        row["outcomes"] = {name: sum(r["metrics"]["outcomes"].get(name, 0) for r in selected)
                           for name in ("correct", "term", "category", "sentiment")}
        row["gold_triplets"] = sum(r["metrics"]["gold_triplets"] for r in selected)
        output.append(row)
    return output


def paired_comparisons(runs, expected_seeds, minimum_effect):
    groups = defaultdict(list)
    for run in runs:
        groups[run["cohort"]].append(run)
    comparisons = []
    for cohort, members in groups.items():
        # The explicit-null control isolates NULL architecture with the same vocabulary.
        control_name = "standard_null_vocab" if members[0]["vocabulary_scope"] == "explicit-null" else "standard"
        controls = {r["trial_id"] for r in members if r["recipe"] == control_name
                    and r["config"]["loss_type"] == "standard" and not r["config"]["null_head"]}
        candidates = sorted({r["trial_id"] for r in members})
        if len(controls) != 1:
            for trial in candidates:
                comparisons.append({"cohort": cohort, "trial_id": trial, "control": control_name, "pairs": [],
                                    "delta_f1": None, "verdict": "No unique compatible CE control"})
            continue
        control_id = next(iter(controls))
        baseline = {r["seed"]: r for r in members if r["trial_id"] == control_id and r["completed"] and r["metrics"]}
        for trial in candidates:
            candidate = {r["seed"]: r for r in members if r["trial_id"] == trial and r["completed"] and r["metrics"]}
            pairs = [{"seed": seed, "candidate_f1": candidate[seed]["metrics"]["explicit_f1"],
                      "control_f1": baseline[seed]["metrics"]["explicit_f1"],
                      "delta_f1": candidate[seed]["metrics"]["explicit_f1"] - baseline[seed]["metrics"]["explicit_f1"],
                      "delta_null_f1": candidate[seed]["metrics"]["null_f1"] - baseline[seed]["metrics"]["null_f1"],
                      "delta_combined_f1": candidate[seed]["metrics"]["combined_f1"] - baseline[seed]["metrics"]["combined_f1"]}
                     for seed in sorted(set(candidate) & set(baseline))]
            delta = mean([p["delta_f1"] for p in pairs])
            ready = set(expected_seeds).issubset({p["seed"] for p in pairs})
            smoke = members[0].get("smoke", False)
            if smoke:
                verdict = "Smoke check only; no performance claim"
            elif trial == control_id:
                verdict = "CE control" if ready else "CE control still incomplete"
            elif not ready:
                verdict = "Awaiting matched completed seeds"
            elif all(p["delta_f1"] > 0 for p in pairs) and delta >= minimum_effect:
                verdict = "Promising pilot; meets +2 pp target" if minimum_effect == .02 else "Promising pilot; meets effect target"
            elif delta > 0:
                verdict = "Mixed seed outcomes" if any(p["delta_f1"] <= 0 for p in pairs) else "Positive pilot; below effect target"
            else:
                verdict = "Lower or equal pilot F1"
            comparisons.append({"cohort": cohort, "trial_id": trial, "control": control_name, "pairs": pairs,
                                "delta_f1": delta, "delta_f1_std": sample_std([p["delta_f1"] for p in pairs]),
                                "delta_null_f1": mean([p["delta_null_f1"] for p in pairs]),
                                "delta_combined_f1": mean([p["delta_combined_f1"] for p in pairs]),
                                "ready": ready, "smoke": smoke, "verdict": verdict})
    return comparisons


def build_analysis(root, mode, domain, expected_recipes, expected_seeds, minimum_effect=.02, include_smoke=False):
    runs, notices = load_runs(root, mode, domain, include_smoke)
    grid = []
    for recipe in expected_recipes:
        for seed in expected_seeds:
            matches = [r for r in runs if r["recipe"] == recipe and r["seed"] == seed]
            grid.append({"recipe": recipe, "seed": seed, "status": matches[0]["status"] if len(matches) == 1
                         else "multiple variants" if matches else "not started",
                         "epochs_recorded": matches[0]["epochs_recorded"] if len(matches) == 1 else None})
    return {"generated_at": datetime.now(ZoneInfo("Pacific/Auckland")).isoformat(timespec="seconds"),
            "input": str(root.resolve()), "mode": mode, "domain": domain,
            "expected_recipes": expected_recipes, "expected_seeds": expected_seeds,
            "minimum_effect": minimum_effect, "runs": runs, "summaries": summaries(runs),
            "comparisons": paired_comparisons(runs, expected_seeds, minimum_effect), "grid": grid,
            "status_counts": dict(Counter(r["status"] for r in runs)), "notices": notices,
            "completed_runs": sum(r["completed"] for r in runs), "expected_runs": len(grid),
            "evaluation": "Development only; seed variation is sample SD, not a confidence interval.",
            "research_status": "Exploratory two-seed screening; no claim of statistical significance."}
