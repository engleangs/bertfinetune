"""Freeze and test existing screen checkpoints; no optimizer or training calls."""

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def screen_variants(entries, directory_for):
    """Dedupe reused screen/LR runs and retain saved dev operating points."""
    variants = {}
    for entry in entries:
        directory = directory_for(entry)
        manifest = read(directory / "manifest.json")
        cfg = manifest["identity"]["config"]
        if manifest["identity"].get("smoke") or manifest["status"] not in ("dev_complete", "complete"):
            raise ValueError(f"Screen is incomplete or smoke-only: {directory}")
        if cfg != entry["config"]:
            raise ValueError(f"Screen config changed: {directory}")
        metric = read(directory / "dev_metrics.json")
        if metric["null_threshold"] != manifest["training"]["null_threshold"]:
            raise ValueError(f"Screen threshold metadata changed: {directory}")
        trial, seed = directory.parent.name, manifest["identity"]["seed"]
        variant = variants.setdefault(trial, {"trial_id": trial, "config": cfg,
            "roles": [f"{cfg['name']}; LR={cfg['lr']:g}; {cfg['epochs']} epochs"], "runs": []})
        if any(r["seed"] == seed for r in variant["runs"]):
            continue
        variant["runs"].append({"seed": seed, "directory": str(directory.resolve()),
            "fingerprint": manifest["fingerprint"], "checkpoint_sha256": manifest["checkpoint_sha256"],
            "best_epoch": manifest["training"]["best_epoch"], "null_threshold": metric["null_threshold"]})
    if not variants or any([r["seed"] for r in v["runs"]] != [13] for v in variants.values()):
        raise ValueError("This registered screen ablation requires one seed-13 checkpoint per candidate")
    return list(variants.values())


def save_freeze(path, frozen):
    from src.artifacts import atomic_write_json
    content = {k: v for k, v in frozen.items() if k != "created_at"}
    if path.exists():
        old = read(path)
        if {k: v for k, v in old.items() if k != "created_at"} != content:
            raise ValueError("Ablation freeze changed; preserve this report and choose a new --output")
        return old
    atomic_write_json(path, frozen)
    return frozen


def freeze_screens(root, quick_study, null_study, experiments_output, output, device):
    """Freeze BOTH screen sets, and check them before opening any test split."""
    from null_experiments.config import VERSION
    from null_experiments.training import Runtime
    from null_experiments.study import check_freeze
    from src.experiment_runtime import code_inventory, data_inventory, environment_inventory, file_hash
    from src.utils import resolve_device
    import config

    quick = read(quick_study / "selection.json")
    state = read(null_study / "search_state.json")
    if "selected_null" not in state:
        raise ValueError("Finish the NULL search before freezing screening ablations")
    old_entries = quick["search_results"]
    old_variants = screen_variants(old_entries, lambda e: experiments_output / "research/indomain/all_domains" / e["trial_id"] / f"seed_{e['seed']}")
    new_entries = [c for stage in state["stages"] for c in stage["candidates"]]
    new_variants = screen_variants(new_entries, lambda e: Path(e["directory"]))
    environment = environment_inventory()
    code = code_inventory(root)["source_sha256"]
    data = data_inventory(root, config.DOMAINS)
    resolved = str(resolve_device(device))
    for variant in old_variants:
        for run in variant["runs"]:
            manifest = read(Path(run["directory"]) / "manifest.json")
            identity = manifest["identity"]
            if (identity["source_sha256"] != code or identity["source_data_sha256"] != data
                    or identity["environment"] != environment or identity["resolved_device"] != resolved):
                raise ValueError("Historical screen source/data/environment/device changed")
            if file_hash(Path(run["directory"]) / "best.pt") != run["checkpoint_sha256"]:
                raise ValueError("Historical screen checkpoint changed")
    runtime = Runtime(root, null_study, device)
    for variant in new_variants:
        variant["role"] = variant["roles"][0]
    evaluation_sources = {str(p.relative_to(root)): file_hash(p) for p in
                          (root / "finalize_finetuning_report.py", Path(__file__))}
    common = {"created_at": datetime.now(timezone.utc).isoformat(), "seeds": [13], "smoke": False,
              "selection_rule": "All registered trained screen candidates; retain each saved development checkpoint and threshold",
              "scope": "Exploratory screening ablation, not a five-seed finalist comparison",
              "evaluation_source_sha256": evaluation_sources}
    old = {**common, "variants": old_variants, "source_sha256": code,
           "source_data_sha256": data, "environment": environment}
    new = {**common, "version": VERSION, "variants": new_variants, "source_sha256": runtime.code["source_sha256"],
           "source_data_sha256": runtime.data_hashes, "environment": runtime.environment}
    check_freeze(runtime, new)
    from experiment_reporting.quick_selection import check_frozen_variants, check_frozen_run
    check_frozen_variants(old)
    for variant in old["variants"]:
        for run in variant["runs"]:
            check_frozen_run(variant, run)
    # Both files are committed before the first new raw-test read.
    old = save_freeze(output / "ablation_freezes/quick/selection.json", old)
    new = save_freeze(output / "ablation_freezes/null/selection.json", new)
    print(f"Ablations frozen before testing: {len(old_variants)} quick candidates; {len(new_variants)} NULL candidates, seed 13 only.", flush=True)
    return old, new, runtime


def evaluate_quick_screens(root, frozen, output, device):
    import torch
    from transformers import AutoTokenizer
    import config
    from src.artifacts import atomic_write_json, atomic_write_jsonl
    from src.data import summarize_tokenized_targets
    from src.experiment_data import ExperimentDataset, load_source_splits, load_test_split, training_vocabulary
    from src.experiment_model import ExperimentModel
    from src.experiment_runtime import data_inventory
    from src.experiment_train import evaluate_trial
    from src.utils import resolve_device
    from experiment_reporting.quick_selection import check_frozen_variants, check_frozen_run, trial_from_dict
    from null_experiments.storage import append_csv, utc_now

    check_frozen_variants(frozen)
    for variant in frozen["variants"]:
        for run in variant["runs"]:
            check_frozen_run(variant, run)
    device = resolve_device(str(device))
    train, _, _ = load_source_splits(root, "indomain")
    test_hashes = data_inventory(root, config.DOMAINS, ("test",))
    tests = None
    for variant in frozen["variants"]:
        cfg = trial_from_dict(variant["config"])
        for run in variant["runs"]:
            directory = Path(run["directory"])
            manifest, _ = check_frozen_run(variant, run)
            if manifest["status"] == "complete":
                if manifest.get("test_data_sha256") != test_hashes:
                    raise ValueError("Previously evaluated historical test data changed")
                metrics = read(directory / "test_metrics.json")
            else:
                categories, sentiments, counts = training_vocabulary(train, cfg)
                checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
                if (checkpoint["config"] != asdict(cfg) or checkpoint["categories"] != categories
                        or checkpoint["sentiments"] != sentiments or checkpoint["null_threshold"] != run["null_threshold"]
                        or checkpoint["best_epoch"] != run["best_epoch"]):
                    raise ValueError("Historical checkpoint differs from the ablation freeze")
                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, local_files_only=cfg.offline, use_fast=True)
                model = ExperimentModel(cfg, len(categories), len(sentiments)).to(device)
                model.load_state_dict(checkpoint["state_dict"])
                if tests is None:
                    tests = load_test_split(root, "indomain")
                dataset = ExperimentDataset(tests, tokenizer, categories, sentiments, cfg)
                metrics, records, outcomes = evaluate_trial(model, dataset, categories, sentiments, counts, cfg, device,
                                                            threshold=run["null_threshold"])
                atomic_write_json(directory / "test_metrics.json", metrics)
                atomic_write_jsonl(directory / "test_predictions.jsonl", records)
                atomic_write_jsonl(directory / "test_taxonomy.jsonl", outcomes)
                atomic_write_json(directory / "test_audit.json", {"annotation_scope": dict(dataset.audit),
                                                                    "tokenized_targets": summarize_tokenized_targets(dataset)})
                manifest.update(status="complete", test_data_sha256=test_hashes, test_evaluated_at=utc_now())
                atomic_write_json(directory / "manifest.json", manifest)
                del model, checkpoint
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if metrics["null_threshold"] != run["null_threshold"]:
                raise ValueError("Historical test threshold differs from its development freeze")
            append_csv(output / "ablation_test_results.csv", {"study": "Quick screens", "trial_id": cfg.trial_id,
                "seed": run["seed"], "checkpoint_sha256": run["checkpoint_sha256"],
                **{s + "_f1": metrics[s]["micro_f1"] for s in ("explicit", "null", "combined")},
                "null_threshold": run["null_threshold"]}, ("study", "trial_id", "seed", "checkpoint_sha256"))
            print(f"Quick ablation: {cfg.name}, LR={cfg.lr:g}; test combined F1={metrics['combined']['micro_f1']:.4f}", flush=True)


def finish_with_ablations(root, args):
    from null_experiments.study import confirm, final_evaluation
    from null_experiments.storage import append_csv, study_lock
    from null_experiments.reporting import write_report

    # One process and one lock: avoid a second CUDA context on a small GPU.
    with study_lock(args.null_study):
        quick, screens, runtime = freeze_screens(root, args.quick_study, args.null_study,
                                                args.experiments_output, args.output, args.device)
        refresh = lambda: write_report(args.null_study, root / "null_fast_track_reports" / args.null_study.name)
        if args.finish_null:
            existing = read(args.null_study / "selection.json") if (args.null_study / "selection.json").exists() else None
            confirmation = SimpleNamespace(first_seeds=max(args.first_seeds, len(existing["seeds"]) if existing else 0),
                epochs=existing["variants"][0]["config"]["epochs"] if existing else 5,
                confirm_top=sum(v["config"]["null_head"] for v in existing["variants"]) if existing else 1,
                retry_failed=args.retry_failed)
            confirm(runtime, read(args.null_study / "search_state.json"), confirmation, refresh)
        if args.finish_null or args.evaluate_null:
            final_evaluation(runtime, read(args.null_study / "selection.json"), refresh)
        evaluate_quick_screens(root, quick, args.output, runtime.device)
        final_evaluation(runtime, screens, refresh)
        for variant in screens["variants"]:
            for run in variant["runs"]:
                metrics = read(Path(run["directory"]) / "test_metrics.json")
                append_csv(args.output / "ablation_test_results.csv", {"study": "NULL screens", "trial_id": variant["trial_id"],
                    "seed": run["seed"], "checkpoint_sha256": run["checkpoint_sha256"],
                    **{s + "_f1": metrics[s]["micro_f1"] for s in ("explicit", "null", "combined")},
                    "null_threshold": run["null_threshold"]}, ("study", "trial_id", "seed", "checkpoint_sha256"))
