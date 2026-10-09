"""Adaptive search, development freeze and separately requested test evaluation."""

from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

import config
from src.artifacts import atomic_write_json, atomic_write_jsonl, atomic_torch_save
from src.experiment_runtime import file_hash

from .config import VERSION, THRESHOLDS, from_dict, loss_candidates, representation_candidates, negative_candidates, tuning_candidates
from .storage import append_csv, digest, event, utc_now, verify_completed


def read_result(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in ("dev_complete", "complete"):
        raise ValueError(f"Development run is incomplete: {directory}")
    metrics = json.loads((directory / "dev_metrics.json").read_text(encoding="utf-8"))
    return {"directory": str(directory.resolve()), "manifest": manifest,
            "config": manifest["identity"]["config"], "metrics": metrics}


def best_result(rows, objective="combined"):
    if not rows:
        raise ValueError("No completed development candidates")
    return max(rows, key=lambda row: (row["metrics"][objective]["micro_f1"],
                                     row["metrics"]["null"]["micro_f1"],
                                     row["metrics"]["explicit"]["micro_f1"],
                                     -row["metrics"]["unweighted_loss"],
                                     digest(row["config"])))


def search(runtime, base, args, refresh=lambda: None):
    path = runtime.output / "search_plan.json"
    plan = {"version": VERSION, "screen_seed": config.SEEDS[0], "base_config": asdict(base),
            "tune": args.tune, "tune_lr": args.tune_lr}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
        raise ValueError("Search plan differs from this output folder; use a new --output for changed search settings")
    atomic_write_json(path, plan)
    state = {"version": VERSION, "created_at": utc_now(), "stages": []}

    def stage(name, candidates):
        print(f"\n{name}: {len(candidates)} candidate(s), seed {config.SEEDS[0]}", flush=True)
        rows = []
        for cfg in candidates:
            directory = runtime.run(cfg, config.SEEDS[0], name, args.retry_failed)
            rows.append(read_result(directory))
            refresh()
        state["stages"].append({"name": name, "candidates": [
            {"directory": row["directory"], "config": row["config"],
             "combined_f1": row["metrics"]["combined"]["micro_f1"],
             "null_f1": row["metrics"]["null"]["micro_f1"]} for row in rows]})
        atomic_write_json(runtime.output / "search_state.json", state)
        return rows

    losses = stage("2-3_loss_screen", loss_candidates(base))
    null_rows = [row for row in losses if row["config"]["null_head"]]
    control, = [row for row in losses if not row["config"]["null_head"]]
    winner = best_result(null_rows, base.selection_metric)
    representations = stage("4_implicit_representation", representation_candidates(from_dict(winner["config"])))
    null_rows.extend(representations)
    winner = best_result(null_rows, base.selection_metric)
    sampled = stage("5_negative_sampling", negative_candidates(from_dict(winner["config"])))
    null_rows.extend(sampled)
    winner = best_result(null_rows, base.selection_metric)
    if args.tune:
        tuned = tuning_candidates(from_dict(winner["config"]))
        if args.tune_lr:
            tuned.append(replace(from_dict(winner["config"]), name=f"{winner['config']['name']}_lr", lr=2e-5))
        null_rows.extend(stage("parameter_tuning", tuned))
    winner = best_result(null_rows, base.selection_metric)
    ranked = sorted(null_rows, key=lambda row: (row["metrics"][base.selection_metric]["micro_f1"],
                    row["metrics"]["null"]["micro_f1"], row["metrics"]["explicit"]["micro_f1"]), reverse=True)
    state.update(selected_null=winner["config"], selected_control=control["config"],
                 ranked_null_candidates=[row["config"] for row in ranked])
    atomic_write_json(runtime.output / "search_state.json", state)
    event(runtime.output, "search_selected", selected_null=winner["config"], objective=base.selection_metric)
    refresh()
    return state


def confirm(runtime, state, args, refresh=lambda: None):
    candidates = [state["selected_null"]]
    if args.confirm_top > 1:
        for raw in state["ranked_null_candidates"]:
            if raw not in candidates:
                candidates.append(raw)
                if len(candidates) == args.confirm_top:
                    break
    candidates.append(state["selected_control"])
    seeds = config.SEEDS[:args.first_seeds]
    variants = []
    for raw in candidates:
        cfg = replace(from_dict(raw), epochs=args.epochs)
        runs = []
        for seed in seeds:
            directory = runtime.run(cfg, seed, "confirmation", args.retry_failed)
            row = read_result(directory)
            manifest = row["manifest"]
            runs.append({"seed": seed, "directory": row["directory"], "fingerprint": manifest["fingerprint"],
                         "checkpoint_sha256": manifest["checkpoint_sha256"],
                         "best_epoch": manifest["training"]["best_epoch"],
                         "null_threshold": row["metrics"]["null_threshold"],
                         "development": {scope: row["metrics"][scope] for scope in ("explicit", "null", "combined")}})
            refresh()
        variants.append({"trial_id": cfg.trial_id, "config": asdict(cfg), "runs": runs})
    head_variants = [v for v in variants if v["config"]["null_head"]]
    selected = max(head_variants, key=lambda v: (
        sum(r["development"][v["config"]["selection_metric"]]["micro_f1"] for r in v["runs"]) / len(v["runs"]),
        sum(r["development"]["null"]["micro_f1"] for r in v["runs"]) / len(v["runs"])))
    for variant in variants:
        variant["role"] = "Best NULL on" if variant is selected else "NULL off control" if not variant["config"]["null_head"] else "NULL on confirmation control"
    frozen = {"version": VERSION, "created_at": utc_now(), "seeds": seeds, "variants": variants,
              "selection_rule": state["selected_null"]["selection_metric"] + " development F1",
              "scope": "Exploratory M-ABSA, eligible explicit + deduplicated NULL; no new cross-domain claim",
              "smoke": runtime.smoke, "source_sha256": runtime.code["source_sha256"],
              "source_data_sha256": runtime.data_hashes, "environment": runtime.environment}
    # Immutable archives permit extending 1 -> 2 -> 5 seeds without losing the
    # earlier freeze. selection.json points to the latest complete cohort.
    content_id = digest({"seeds": seeds, "variants": variants, "version": VERSION})[:12]
    archive = runtime.output / "selections" / f"selection_{len(seeds)}seeds_{content_id}.json"
    if not archive.exists():
        atomic_write_json(archive, frozen)
    atomic_write_json(runtime.output / "selection.json", frozen)
    event(runtime.output, "development_frozen", seeds=seeds, archive=str(archive), selected=selected["trial_id"])
    refresh()
    return frozen


def check_freeze(runtime, frozen):
    if runtime.smoke or frozen.get("smoke"):
        raise ValueError("Smoke studies never evaluate test data")
    if frozen["version"] != VERSION or frozen["source_sha256"] != runtime.code["source_sha256"]:
        raise ValueError("Frozen source changed; preserve this study and use a new output")
    if frozen["source_data_sha256"] != runtime.data_hashes or frozen["environment"] != runtime.environment:
        raise ValueError("Frozen train/dev data or environment changed")
    seen = set()
    for variant in frozen["variants"]:
        cfg = from_dict(variant["config"])
        if cfg.trial_id != variant["trial_id"] or cfg.trial_id in seen:
            raise ValueError("Frozen trial identity is invalid")
        seen.add(cfg.trial_id)
        if sorted(r["seed"] for r in variant["runs"]) != sorted(frozen["seeds"]):
            raise ValueError("Frozen cohort is missing or repeats seeds")
        for run in variant["runs"]:
            directory = runtime.destination(cfg, run["seed"])
            manifest = verify_completed(directory, runtime.identity(cfg, run["seed"]))
            metrics = json.loads((directory / "dev_metrics.json").read_text(encoding="utf-8"))
            if manifest["status"] not in ("dev_complete", "complete") or manifest["fingerprint"] != run["fingerprint"]:
                raise ValueError("Frozen development run is incomplete or changed")
            if manifest["checkpoint_sha256"] != run["checkpoint_sha256"] or metrics["null_threshold"] != run["null_threshold"]:
                raise ValueError("Frozen checkpoint or threshold changed")
            if manifest["training"]["best_epoch"] != run["best_epoch"] or manifest["training"]["null_threshold"] != run["null_threshold"]:
                raise ValueError("Frozen checkpoint selection metadata changed")


def final_evaluation(runtime, frozen, refresh=lambda: None):
    import torch
    from src.experiment_data import load_test_split
    from .data import CachedDataset
    from .evaluation import evaluate
    from .model import NullModel

    # Validate every variant before opening any raw test file.
    check_freeze(runtime, frozen)
    tests = load_test_split(runtime.root, "indomain")
    from src.experiment_runtime import data_inventory
    test_hashes = data_inventory(runtime.root, config.DOMAINS, ("test",))
    for variant in frozen["variants"]:
        cfg = from_dict(variant["config"])
        train, _, categories, sentiments, counts = runtime.data(cfg)
        test_ds = CachedDataset(tests, runtime.tokenizer, categories, sentiments, cfg)
        for run in variant["runs"]:
            directory = runtime.destination(cfg, run["seed"])
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["status"] == "complete":
                if manifest.get("test_data_sha256") != test_hashes:
                    raise ValueError("Previously evaluated test data changed")
                if not all((directory / name).exists() for name in
                           ("test_metrics.json", "test_predictions.jsonl", "test_taxonomy.jsonl")):
                    raise ValueError("Completed test evaluation is missing artifacts")
                metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
                append_test_result(runtime.output, cfg, run, manifest, metrics)
                continue
            checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
            if json.loads(json.dumps(checkpoint["config"])) != json.loads(json.dumps(asdict(cfg))) or checkpoint["seed"] != run["seed"]:
                raise ValueError("Checkpoint settings differ from the development freeze")
            if checkpoint["null_threshold"] != run["null_threshold"] or checkpoint["best_epoch"] != run["best_epoch"]:
                raise ValueError("Checkpoint selection differs from the development freeze")
            model = NullModel(cfg, categories, sentiments, runtime.tokenizer).to(runtime.device)
            model.load_state_dict(checkpoint["state_dict"])
            metrics, records, outcomes, _ = evaluate(model, test_ds, categories, sentiments, counts, cfg,
                   runtime.device, train.pair_counts, threshold=run["null_threshold"])
            atomic_write_json(directory / "test_metrics.json", metrics)
            atomic_write_jsonl(directory / "test_predictions.jsonl", records)
            atomic_write_jsonl(directory / "test_taxonomy.jsonl", outcomes)
            manifest.update(status="complete", test_data_sha256=test_hashes, test_evaluated_at=utc_now())
            atomic_write_json(directory / "manifest.json", manifest)
            append_test_result(runtime.output, cfg, run, manifest, metrics)
            event(runtime.output, "test_evaluated", trial_id=cfg.trial_id, seed=run["seed"])
            del model, checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            refresh()


def append_test_result(output, cfg, run, manifest, metrics):
    # Replaying a completed evaluation repairs a crash between its manifest
    # commit and CSV append without duplicating the logical result.
    append_csv(output / "test_results.csv", {"at": manifest["test_evaluated_at"], "trial_id": cfg.trial_id,
        "seed": run["seed"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "explicit_f1": metrics["explicit"]["micro_f1"], "null_f1": metrics["null"]["micro_f1"],
        "combined_f1": metrics["combined"]["micro_f1"], "null_threshold": run["null_threshold"]},
        ("trial_id", "seed", "checkpoint_sha256"))


def calibrate_legacy(runtime, selection_path, seed=13):
    """Technique 1: separate development-only inference, legacy files preserved."""
    import torch
    from transformers import AutoTokenizer
    from src.experiment_config import TrialConfig
    from src.experiment_data import ExperimentDataset, load_source_splits, select_targets
    from src.experiment_model import ExperimentModel
    from .evaluation import evaluate

    if runtime.smoke:
        return None
    frozen = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    variant, = [v for v in frozen["variants"] if v["config"]["null_head"]]
    run, = [r for r in variant["runs"] if r["seed"] == seed]
    source = Path(run["directory"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["identity"]["source_sha256"].items():
        if file_hash(runtime.root / rel) != expected:
            raise ValueError("Legacy training source changed; calibration needs its original code")
    if manifest["identity"]["source_data_sha256"] != runtime.data_hashes or file_hash(source / "best.pt") != run["checkpoint_sha256"]:
        raise ValueError("Legacy corpus or selected checkpoint changed")
    cfg_raw = dict(variant["config"])
    for key in ("loss_heads", "null_thresholds"):
        cfg_raw[key] = tuple(cfg_raw[key])
    cfg = TrialConfig(**cfg_raw)
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    categories, sentiments = checkpoint["categories"], checkpoint["sentiments"]
    output = runtime.output / "calibration" / f"seed_{seed}"
    identity = {"checkpoint_sha256": run["checkpoint_sha256"], "source_data_sha256": runtime.data_hashes,
                "source_sha256": runtime.code["source_sha256"], "thresholds": list(THRESHOLDS),
                "device": str(runtime.device), "environment": runtime.environment}
    if (output / "calibration.json").exists():
        previous = json.loads((output / "calibration.json").read_text(encoding="utf-8"))
        if previous["identity"] == identity:
            print(f"Reuse verified development calibration: {output}", flush=True)
            return previous
        raise ValueError("Calibration provenance differs; use a new --output")
    cfg.null_thresholds, cfg.selection_metric = THRESHOLDS, "combined"
    train, dev, _ = load_source_splits(runtime.root, "indomain")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, local_files_only=True, use_fast=True)
    dataset = ExperimentDataset(dev, tokenizer, categories, sentiments, cfg)
    pair_counts = Counter((c, s) for ex in train for _, c, s in select_targets(ex)[1])
    audit = json.loads((source / "audit.json").read_text(encoding="utf-8"))
    model = ExperimentModel(cfg, len(categories), len(sentiments)).to(runtime.device)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, records, outcomes, probabilities = evaluate(model, dataset, categories, sentiments,
                    audit["source_category_counts"], cfg, runtime.device, pair_counts)
    old = json.loads((source / "dev_metrics.json").read_text(encoding="utf-8"))
    for key in ("true_positives", "false_positives", "false_negatives"):
        if metrics["explicit"][key] != old["explicit"][key]:
            raise ValueError("Legacy explicit predictions changed during calibration; inspect before comparing")
    report = {"identity": identity, "source": str(source), "seed": seed,
              "scope": "Historical checkpoint diagnostic; different input/precision from new arms",
              "saved": {"threshold": old["null_threshold"], "combined": old["combined"], "null": old["null"]},
              "calibrated": {"threshold": metrics["null_threshold"], "combined": metrics["combined"], "null": metrics["null"]},
              "threshold_curve": metrics["null_threshold_curve"], "diagnostics": metrics["null_diagnostics"]}
    atomic_write_json(output / "calibration.json", report)
    atomic_write_json(output / "dev_metrics.json", metrics)
    atomic_write_jsonl(output / "dev_predictions.jsonl", records)
    atomic_write_jsonl(output / "dev_taxonomy.jsonl", outcomes)
    atomic_torch_save(output / "dev_probabilities.pt", probabilities)
    event(runtime.output, "legacy_calibrated", seed=seed, threshold=metrics["null_threshold"])
    print(f"Development calibration: threshold {old['null_threshold']:g} -> {metrics['null_threshold']:g}; "
          f"combined F1 {old['combined']['micro_f1']:.4f} -> {metrics['combined']['micro_f1']:.4f}; "
          f"NULL F1={metrics['null']['micro_f1']:.4f}", flush=True)
    return report
