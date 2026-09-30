"""Clean-data study: prepare, search on source dev, evaluate, export small results."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import config as cfg
from src.artifacts import atomic_write_json, atomic_write_jsonl_gzip, upsert_csv_row
from src.study_data import (file_digest, json_digest, prepare_clean_data, split_paths, verify_source_splits)
from src.study_plan import (VERSION, FOLDS, LOSSES, SEEDS, baseline_tasks, search_tasks,
                            smoke_tasks, main_tasks, selected_search_tasks, final_tasks,
                            completion_tasks, choose_recipes, plan_payload)
from src.study_storage import (read_json, code_fingerprint, create_code_snapshot,
                               lightweight_export, safe_checkpoint_path)

PROJECT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT / "artifacts" / "clean_study_v1"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt")


def now():
    return datetime.now(timezone.utc).isoformat()


def model_directory():
    return Path(os.environ.get("BERT_MODEL_PATH", PROJECT / "model_cache" / "bert-base-uncased")).resolve()


def prepare(output):
    output = Path(output).resolve()
    raw, clean = PROJECT / "data/m-absa", PROJECT / "data/m-absa-clean-v1"
    audit = prepare_clean_data(raw, clean)
    code = code_fingerprint(PROJECT)
    model = {name: file_digest(model_directory() / name) for name in MODEL_FILES}
    meta = {"version": VERSION, "plan": plan_payload(), "code": code,
            "model_files": model, "data_roots": {"raw": "data/m-absa", "clean": "data/m-absa-clean-v1"},
            "cleaning_audit_sha256": json_digest(audit),
            "task_scope": "single-pair-aligned-explicit-aspect; original first-occurrence/first-pair selection retained",
            "storage_policy": "results-only local export; keep selected search weights until evaluated; archive 16 dev-selected representatives"}
    marker = output / "study.json"
    if marker.exists():
        if read_json(marker) != meta:
            raise ValueError("Study code/data/plan changed. Use a new versioned output root.")
        print("Prepared study verified:", output)
        return meta
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not an empty new study")
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "cleaning_audit.json", audit)
    create_code_snapshot(PROJECT, output / "reproducibility/code")
    # Raw data are small; save one copy for exact regeneration in the result ZIP.
    for relative in split_paths():
        target = output / "reproducibility/raw_data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((raw / relative).read_bytes())
    for name in MODEL_FILES:
        if name != "model.safetensors":
            target = output / "reproducibility/tokenizer_and_config" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((model_directory() / name).read_bytes())
    atomic_write_json(marker, meta)
    print(json.dumps({"study": str(output), "removed_sentences": audit["removed_sentence_count"],
                      "removed_conflict_triplets": audit["removed_conflict_triplets"],
                      "removed_other_triplets": audit["removed_other_triplets"],
                      **meta["plan"]["counts"]}, indent=2))
    return meta


def load_study(output, verify_code=False):
    output = Path(output).resolve()
    meta = read_json(output / "study.json")
    audit = read_json(output / "cleaning_audit.json")
    if meta.get("version") != VERSION or meta.get("plan") != plan_payload():
        raise ValueError("Study does not match this runner's frozen plan")
    if json_digest(audit) != meta["cleaning_audit_sha256"]:
        raise ValueError("Cleaning audit has changed")
    if verify_code and code_fingerprint(PROJECT) != meta["code"]:
        raise ValueError("Code changed after preparation; create a new study version")
    return meta, audit


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_dir(output, task):
    return Path(output) / "runs" / task.key


def runtime_versions():
    import torch
    import transformers
    import numpy
    import tqdm
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "transformers": transformers.__version__,
              "numpy": numpy.__version__, "tqdm": tqdm.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name()
    return result


def ensure_evidence(directory, manifest, require_test=False):
    names = ["dev_summary.json", "history.json", "dev_metrics.json", "dev_predictions.jsonl.gz"]
    if require_test:
        names += ["metrics.json", "test_predictions.jsonl.gz"]
    for name in names:
        if not (directory / name).is_file():
            raise ValueError(f"Missing evidence: {directory / name}")
        expected = manifest.get("evidence_sha256", {}).get(name)
        if not expected or file_digest(directory / name) != expected:
            raise ValueError(f"Evidence hash mismatch: {directory / name}")


def _extra_metrics(metrics, records, categories):
    from src.result_analysis import derive_crossdomain_metrics
    gold = [[tuple(t) for t in row["gold_triplets"]] for row in records]
    pred = [[tuple(t) for t in row["predicted_triplets"]] for row in records]
    metrics["projections"] = derive_crossdomain_metrics(gold, pred, categories)
    metrics["macro_f1_source_training_categories"] = statistics.mean(
        metrics["category_by_label"].get(label, {}).get("f1", 0.0) for label in categories)
    metrics["metric_note"] = "Exact surface-string triplets in the original explicit/single-pair scope. Projection aspect_span uses surface text, not occurrence offsets. Legacy error counts may overlap."
    return metrics


def run_task(task, output, device="cuda", resume=False, evaluate_only=False):
    import torch
    from transformers import AutoTokenizer
    from src.data import (ABSADataset, build_indomain_train_dev, build_crossdomain_train_dev,
                          load_domain_file, build_label_vocab, label_frequencies,
                          summarize_task_scope, summarize_tokenized_targets)
    from src.evaluate import identify_rare_labels
    from src.losses import get_evaluation_loss_fns
    from src.model import build_model
    from src.train import train_one_config, evaluate_model
    from src.utils import resolve_device

    output = Path(output).resolve()
    meta, audit = load_study(output, verify_code=True)
    for name, digest in meta["model_files"].items():
        if file_digest(model_directory() / name) != digest:
            raise ValueError(f"Shared model/tokenizer asset changed: {name}")
    directory = run_dir(output, task)
    with run_lock(directory):
        marker = directory / "manifest.json"
        manifest = read_json(marker) if marker.exists() else {}
        if manifest and manifest.get("task") != task.payload():
            raise ValueError("Existing run identity differs")
        do_test = task.stage in ("baseline", "final") or evaluate_only
        if manifest.get("evaluation_complete"):
            ensure_evidence(directory, manifest, require_test=True)
            print("Skipping evaluated run:", task.key)
            return
        if manifest.get("training_complete") and not do_test:
            ensure_evidence(directory, manifest)
            print("Skipping trained run:", task.key)
            return
        if evaluate_only and not manifest.get("training_complete"):
            raise ValueError("Selected search model is not complete; evaluation cannot retrain it")
        data_root = PROJECT / meta["data_roots"][task.data_version]
        source_hashes = verify_source_splits(data_root, audit, task.data_version, task.fold)
        experiment = cfg.ExperimentConfig(name=task.loss, loss_type=task.loss,
            model_name=str(model_directory()), lr=task.lr, epochs=task.epochs,
            warmup_ratio=task.warmup_ratio, max_grad_norm=task.max_grad_norm)
        identity = json_digest({"task": task.payload(), "code": meta["code"]["sha256"],
                                "source_hashes": source_hashes, "model": meta["model_files"]})
        if manifest and manifest.get("identity") != identity:
            raise ValueError("Run data/code/model identity changed")
        if not manifest:
            manifest = {"version": VERSION, "task": task.payload(), "identity": identity,
                        "started_at": now(), "configuration": asdict(experiment),
                        "source_hashes": source_hashes, "versions": runtime_versions(),
                        "training_complete": False, "evaluation_complete": False}
        else:
            current_versions = runtime_versions()
            for field in ("python", "torch", "transformers", "numpy", "tqdm"):
                if manifest["versions"][field] != current_versions[field]:
                    raise ValueError(f"Environment changed while resuming/evaluating: {field}")
        atomic_write_json(marker, manifest)
        checkpoint, resume_file = directory / "best_model.pt", directory / "resume.pt"
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(model_directory()), local_files_only=True)
            if task.fold == "indomain":
                train, dev = build_indomain_train_dev(str(data_root))
            else:
                train, dev = build_crossdomain_train_dev(str(data_root),
                    train_domains=[d for d in cfg.DOMAINS if d != task.fold], test_domain=task.fold)
            if task.stage == "smoke":
                train, dev = train[:64], dev[:32]
            categories, sentiments = build_label_vocab(train)
            if task.data_version == "clean" and "conflict" in sentiments:
                raise ValueError("Clean training data still contain conflict")
            counts, _ = label_frequencies(train)
            rare = identify_rare_labels(counts, cfg.RARE_CATEGORY_MAX_COUNT)
            train_ds = ABSADataset(train, tokenizer, categories, sentiments, experiment.max_len)
            dev_ds = ABSADataset(dev, tokenizer, categories, sentiments, experiment.max_len)
            resolved = resolve_device(device)
            losses = tuple(fn.to(resolved) for fn in get_evaluation_loss_fns())
            if not manifest.get("training_complete"):
                if resume_file.exists() and not resume:
                    raise ValueError("Incomplete run has a resume checkpoint; use --resume")
                result = train_one_config(experiment, train_ds, dev_ds, categories, sentiments,
                    task.seed, device=device, checkpoint_path=checkpoint,
                    history_path=directory / "history.json", rare_category_labels=rare,
                    resume_path=resume_file, resume=resume and resume_file.exists(), run_identity=identity)
                model = result["model"]
                dev_metrics, dev_records = evaluate_model(model, dev_ds, categories, sentiments,
                    experiment, resolved, rare_category_labels=rare, loss_fns=losses)
                _extra_metrics(dev_metrics, dev_records, categories)
                atomic_write_jsonl_gzip(directory / "dev_predictions.jsonl.gz", dev_records)
                atomic_write_json(directory / "dev_metrics.json", dev_metrics)
                summary = {"task": task.payload(), "status": "trained", "identity": identity,
                           "best_dev_micro_f1": result["best_dev_metrics"]["complete_triplet"]["micro_f1"],
                           "best_epoch": result["best_epoch"], "training_time_sec": result["training_time_sec"],
                           "train_size": len(train), "dev_size": len(dev)}
                atomic_write_json(directory / "dev_summary.json", summary)
                manifest.update({"training_complete": True, "training_completed_at": now(),
                    "checkpoint_status": "retained", "category_vocab": categories, "sentiment_vocab": sentiments,
                    "rare_categories": rare, "training_label_statistics": result["training_label_statistics"],
                    "train_scope": summarize_task_scope(train, categories, sentiments),
                    "train_tokenized_targets": summarize_tokenized_targets(train_ds),
                    "dev_scope": summarize_task_scope(dev, categories, sentiments),
                    "evidence_sha256": {name: file_digest(directory / name) for name in
                        ("dev_summary.json", "history.json", "dev_metrics.json", "dev_predictions.jsonl.gz")}})
                atomic_write_json(marker, manifest)
            else:
                ensure_evidence(directory, manifest)
                if not checkpoint.is_file():
                    raise ValueError("Required selected model checkpoint is missing or pruned")
                model = build_model(experiment, len(categories), len(sentiments)).to(resolved)
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
            # An epoch recovery file is no longer needed once durable trained evidence exists.
            if resume_file.exists():
                resume_file.unlink()
            if do_test:
                domains = cfg.DOMAINS if task.fold == "indomain" else [task.fold]
                test = []
                for domain in domains:
                    relative = f"{domain}/en/test.txt"
                    if file_digest(data_root / relative) != audit["files"][relative][f"{task.data_version}_sha256"]:
                        raise ValueError("Test file changed after preparation")
                    test.extend(load_domain_file(str(data_root / relative), domain))
                test_ds = ABSADataset(test, tokenizer, categories, sentiments, experiment.max_len)
                metrics, records = evaluate_model(model, test_ds, categories, sentiments, experiment,
                    resolved, rare_category_labels=rare, loss_fns=losses, description="final test")
                _extra_metrics(metrics, records, categories)
                atomic_write_jsonl_gzip(directory / "test_predictions.jsonl.gz", records)
                atomic_write_json(directory / "metrics.json", {"task": task.payload(), "test": metrics,
                    "test_scope": summarize_task_scope(test, categories, sentiments),
                    "dev": read_json(directory / "dev_metrics.json")})
                manifest.update({"evaluation_complete": True, "evaluated_at": now()})
                for name in ("metrics.json", "test_predictions.jsonl.gz"):
                    manifest["evidence_sha256"][name] = file_digest(directory / name)
                atomic_write_json(marker, manifest)
            print("Complete:", task.key, "evaluated=" + str(do_test))
        except Exception as exc:
            manifest["last_error"] = {"time": now(), "type": type(exc).__name__, "message": str(exc)}
            atomic_write_json(marker, manifest)
            raise


def select(output):
    load_study(output, verify_code=True)
    summaries = []
    for task in search_tasks():
        directory = run_dir(output, task)
        manifest = read_json(directory / "manifest.json")
        if not manifest.get("training_complete"):
            raise ValueError(f"Search incomplete: {task.key}")
        ensure_evidence(directory, manifest)
        summaries.append(read_json(directory / "dev_summary.json"))
    selection = choose_recipes(summaries)
    selection["source_summary_hashes"] = {t.key: file_digest(run_dir(output, t) / "dev_summary.json") for t in search_tasks()}
    target = Path(output) / "selection.json"
    if target.exists() and read_json(target) != selection:
        raise ValueError("Parameter selection is already frozen with different inputs")
    for task in selected_search_tasks(selection):
        manifest = read_json(run_dir(output, task) / "manifest.json")
        if not (run_dir(output, task) / "best_model.pt").exists() and not manifest.get("evaluation_complete"):
            raise ValueError(f"Selected search checkpoint is unavailable: {task.key}")
    atomic_write_json(target, selection)
    print(json.dumps({fold: row["lr"] for fold, row in selection["folds"].items()}, indent=2))
    return selection


def load_selection(output):
    selection = read_json(Path(output) / "selection.json")
    if selection.get("version") != VERSION or set(selection["folds"]) != set(FOLDS):
        raise ValueError("Invalid selection file")
    summaries = []
    for task in search_tasks():
        path = run_dir(output, task) / "dev_summary.json"
        if file_digest(path) != selection["source_summary_hashes"][task.key]:
            raise ValueError("Search summary changed after parameter selection")
        summaries.append(read_json(path))
    expected = choose_recipes(summaries)
    if {k: v for k, v in selection.items() if k != "source_summary_hashes"} != expected:
        raise ValueError("Selection differs from the frozen source-development rule")
    return selection


def summarize(output):
    load_study(output, verify_code=True)
    selection = load_selection(output)
    rows = []
    tasks = [*baseline_tasks(), *completion_tasks(selection)]
    for task in tasks:
        directory = run_dir(output, task)
        manifest = read_json(directory / "manifest.json")
        if not manifest.get("evaluation_complete"):
            raise ValueError(f"Evaluation incomplete: {task.key}")
        ensure_evidence(directory, manifest, require_test=True)
        summary = read_json(directory / "dev_summary.json")
        metrics = read_json(directory / "metrics.json")["test"]
        rows.append({"group": "baseline" if task.stage == "baseline" else "final",
                     "data": task.data_version, "fold": task.fold, "loss": task.loss,
                     "seed": task.seed, "lr": task.lr, "epochs": task.epochs,
                     "best_epoch": summary["best_epoch"], "dev_f1": summary["best_dev_micro_f1"],
                     "test_f1": metrics["complete_triplet"]["micro_f1"],
                     "test_precision": metrics["complete_triplet"]["micro_precision"],
                     "test_recall": metrics["complete_triplet"]["micro_recall"],
                     "rare_category_recall": metrics["rare_category_recall"],
                     "category_coverage": metrics["projections"]["gold_category_coverage"],
                     "aspect_f1": metrics["projections"]["aspect_span"]["f1"],
                     "aspect_sentiment_f1": metrics["projections"]["aspect_plus_sentiment"]["f1"],
                     "known_category_f1": metrics["projections"]["source_known_exact_triplet"]["f1"],
                     "key": task.key})
    fields = list(rows[0])
    destination = Path(output) / "results.csv"
    for row in rows:
        upsert_csv_row(destination, row, fields, ["key"])
    comparisons = []
    for group, data, fold in sorted({(r["group"], r["data"], r["fold"]) for r in rows}):
        matched = [r for r in rows if (r["group"], r["data"], r["fold"]) == (group, data, fold)]
        differences = [next(r["test_f1"] for r in matched if r["seed"] == seed and r["loss"] == "weighted") -
                       next(r["test_f1"] for r in matched if r["seed"] == seed and r["loss"] == "standard") for seed in SEEDS]
        comparisons.append({"group": group, "data": data, "fold": fold,
            "mean_paired_difference": statistics.mean(differences), "paired_difference_std": statistics.stdev(differences),
            "paired_differences": dict(zip(map(str, SEEDS), differences)),
            "by_loss": {loss: {"mean_f1": statistics.mean(r["test_f1"] for r in matched if r["loss"] == loss),
                               "std_f1": statistics.stdev(r["test_f1"] for r in matched if r["loss"] == loss)} for loss in LOSSES}})
    cleanup = {}
    for loss in LOSSES:
        differences = [next(r["test_f1"] for r in rows if r["group"] == "baseline" and r["data"] == "clean" and r["loss"] == loss and r["seed"] == seed) -
                       next(r["test_f1"] for r in rows if r["group"] == "baseline" and r["data"] == "raw" and r["loss"] == loss and r["seed"] == seed) for seed in SEEDS]
        cleanup[loss] = {"mean_clean_minus_raw": statistics.mean(differences), "paired_differences": dict(zip(map(str, SEEDS), differences))}
    report = {"version": VERSION, "result_count": len(rows), "comparisons": comparisons,
              "cleanup_effect": cleanup, "search_seed_reuse": [13, 42],
              "interpretation": "Exploratory follow-up after prior test inspection; learning rates selected only from each fold's source dev. Cleanup changes examples and the sentiment output vocabulary."}
    atomic_write_json(Path(output) / "summary.json", report)
    print("Saved 20 baseline and 80 final results:", destination)
    return report


def prune(output, scope="unselected", apply=False, keep_models="representatives"):
    """Delete only named checkpoints in this new study, after verifying their evidence."""
    output = Path(output).resolve()
    load_study(output, verify_code=True)
    selection = load_selection(output)
    if scope == "unselected":
        selected = {t.key for t in selected_search_tasks(selection)}
        tasks = [t for t in search_tasks() if t.key not in selected]
        keep = set()
    else:
        summarize(output)  # Requires all 100 evaluations and intact prediction files.
        tasks = [*baseline_tasks(), *search_tasks(), *final_tasks(selection), *smoke_tasks()]
        keep = set()
        if keep_models == "representatives":
            for fold in FOLDS:
                for loss in LOSSES:
                    candidates = [t for t in completion_tasks(selection) if t.fold == fold and t.loss == loss]
                    best = max(candidates, key=lambda t: (
                        read_json(run_dir(output, t) / "dev_summary.json")["best_dev_micro_f1"], -SEEDS.index(t.seed)))
                    keep.add(best.key)
    removals = []
    for task in tasks:
        directory = run_dir(output, task)
        if not (directory / "manifest.json").exists() and task.stage == "smoke":
            continue
        manifest = read_json(directory / "manifest.json")
        if not manifest.get("training_complete"):
            raise ValueError(f"Refusing to prune unfinished run: {task.key}")
        ensure_evidence(directory, manifest, require_test=manifest.get("evaluation_complete", False))
        for name in ("best_model.pt", "resume.pt"):
            if task.key in keep and name == "best_model.pt":
                if not (directory / name).is_file():
                    raise ValueError(f"Representative checkpoint unavailable: {task.key}")
                continue
            relative = (directory / name).relative_to(output)
            path = safe_checkpoint_path(output, relative)
            if path.exists():
                removals.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "key": task.key})
    report = {"scope": scope, "keep_models": keep_models, "retained_model_keys": sorted(keep),
              "removals": removals, "bytes_to_remove": sum(r["bytes"] for r in removals), "applied": False}
    if apply:
        journal = output / f"prune_{scope}.json"
        atomic_write_json(journal, {**report, "state": "prepared"})
        for row in removals:
            path = safe_checkpoint_path(output, row["path"])
            marker = path.parent / "manifest.json"
            with run_lock(path.parent):
                manifest = read_json(marker)
                ensure_evidence(path.parent, manifest, require_test=manifest.get("evaluation_complete", False))
                if path.name == "best_model.pt":
                    manifest["checkpoint_status"] = "pruned"
                    manifest["checkpoint_prune_reason"] = scope
                    atomic_write_json(marker, manifest)
                path.unlink(missing_ok=True)
        report["applied"] = True
        atomic_write_json(journal, {**report, "state": "complete"})
    print(json.dumps({"scope": scope, "files": len(removals), "bytes_to_remove": report["bytes_to_remove"],
                      "retained_models": len(keep), "applied": apply}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "plan", "run", "select", "summarize", "export", "prune"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("smoke", "main", "completion"), default="main")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scope", choices=("unselected", "archive"), default="unselected")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--keep-models", choices=("representatives", "none"), default="representatives")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "plan":
        print(json.dumps(plan_payload(), indent=2))
    elif args.action == "run":
        selection = load_selection(root) if args.phase == "completion" else None
        tasks = smoke_tasks() if args.phase == "smoke" else (main_tasks() if args.phase == "main" else completion_tasks(selection))
        if args.task_index is None or not 0 <= args.task_index < len(tasks):
            parser.error(f"--task-index must be between 0 and {len(tasks) - 1}")
        task = tasks[args.task_index]
        run_task(task, root, args.device, args.resume,
                 evaluate_only=args.phase == "completion" and task.stage == "search")
    elif args.action == "select":
        select(root)
    elif args.action == "summarize":
        summarize(root)
    elif args.action == "prune":
        prune(root, args.scope, args.apply, args.keep_models)
    elif args.action == "export":
        if args.legacy_root is None:
            summarize(root)
        destination = args.destination or root.parent / ("legacy_results_light.zip" if args.legacy_root else "clean_study_results_light.zip")
        print(json.dumps(lightweight_export(args.legacy_root or root, destination,
                                           legacy=args.legacy_root is not None), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
