"""Versioned development experiments. Print a plan by default; execute explicitly."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from collections import defaultdict
import statistics

import torch
from transformers import AutoTokenizer

import config
from src.artifacts import atomic_write_json, atomic_write_jsonl, upsert_csv_row
from src.data import summarize_tokenized_targets
from src.experiment_config import TrialConfig, candidate_recipes
from src.experiment_data import ExperimentDataset, load_source_splits, load_test_split, training_vocabulary
from src.experiment_model import ExperimentModel
from src.experiment_runtime import code_inventory, data_inventory, environment_inventory, file_hash
from src.experiment_train import evaluate_trial, train_trial
from src.utils import resolve_device

ROOT = Path(__file__).resolve().parent
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
VERSION = "2.0-exploratory-2026-10-05"


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("pilot", "full", "final"), default="pilot",
                   help="pilot: first seeds/dev only; full: five seeds/dev only; final: existing checkpoints/test once")
    p.add_argument("--suite", choices=("baseline", "loss", "heads", "null", "all", "custom"), default="loss")
    p.add_argument("--recipes", nargs="+", help="Run only these named members of the suite")
    p.add_argument("--execute", action="store_true", help="Execute the printed grid")
    p.add_argument("--dry-run", action="store_true", help="Print the grid without loading data or models")
    p.add_argument("--first-seeds", type=int, default=None, help="Matched prefix of the five registered seeds (pilot defaults to two)")
    p.add_argument("--seeds", nargs="+", type=int, help="Explicit exploratory seed list; cannot combine with --first-seeds")
    p.add_argument("--mode", choices=config.MODES, default="indomain")
    p.add_argument("--held-out-domains", nargs="+", choices=config.DOMAINS, default=["restaurant"])
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts" / "experiments_v2")
    p.add_argument("--retry-failed", action="store_true", help="Restart failed training with identical provenance; completed runs are never retrained")
    p.add_argument("--smoke", action="store_true", help="Integration check only, not a research score; saves to a distinct subdirectory")
    p.add_argument("--train-limit", type=int, default=64, help="Used only with --smoke")
    p.add_argument("--dev-limit", type=int, default=32, help="Used only with --smoke")
    p.add_argument("--model-name", default=config.MODEL_NAME)
    p.add_argument("--model-revision", default=BERT_REVISION, help="Immutable model/tokenizer commit SHA; custom models need their own revision")
    p.add_argument("--online", action="store_true", help="Permit downloading the pinned model instead of using its local cache")
    # None means retain the named recipe's setting, rather than silently overriding it.
    p.add_argument("--loss",dest="loss_type",choices=("standard", "weighted", "mixed", "focal", "class_balanced"),default=None)
    p.add_argument("--loss-heads", nargs="+", choices=("bio", "category", "sentiment"), default=None)
    p.add_argument("--null-head", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--vocabulary-scope", choices=("auto", "explicit", "explicit-null"), default=None)
    p.add_argument("--drop-conflict", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--selection-metric", choices=("explicit", "combined"), default=None)
    p.add_argument("--null-thresholds", nargs="+", type=float, default=None)
    for flag, kind in (
        ("ce-weight", float), ("weighted-ce-weight", float), ("focal-gamma", float),
        ("bio-loss-weight", float), ("category-loss-weight", float), ("sentiment-loss-weight", float),
        ("null-loss-weight", float), ("null-pos-weight-cap", float), ("null-threshold", float),
        ("warmup-ratio", float), ("max-grad-norm", float), ("lr", float), ("weight-decay", float),
        ("epochs", int), ("batch-size", int), ("max-len", int),
        ("class-balance-beta", float),
    ):
        p.add_argument(f"--{flag}", type=kind, default=None)
    return p


def make_plan(args):
    if args.seeds is not None and args.first_seeds is not None:
        raise ValueError("Use --seeds or --first-seeds, not both")
    count = args.first_seeds if args.first_seeds is not None else (2 if args.stage == "pilot" else len(config.SEEDS))
    if not 1 <= count <= len(config.SEEDS):
        raise ValueError("--first-seeds must be between 1 and 5")
    seeds = args.seeds if args.seeds is not None else config.SEEDS[:count]
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Seeds must be unique integers in [0, 2**32)")
    if args.smoke and args.stage == "final":
        raise ValueError("Smoke checks never evaluate test data")
    if args.smoke and min(args.train_limit, args.dev_limit) < 1:
        raise ValueError("Smoke split limits must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        raise ValueError("Pin --model-revision to an immutable 40-character commit SHA")
    recipes = candidate_recipes(args.suite)
    if args.recipes:
        missing = set(args.recipes) - {r["name"] for r in recipes}
        if missing:
            raise ValueError(f"Recipes outside this suite: {sorted(missing)}")
        recipes = [r for r in recipes if r["name"] in args.recipes]
    overrides = {key: getattr(args, key) for key in TrialConfig.__dataclass_fields__
                 if hasattr(args, key) and getattr(args, key) is not None}
    overrides.update(offline=not args.online)
    for key in ("loss_heads", "null_thresholds"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    trials = [TrialConfig(**{**recipe, **overrides}) for recipe in recipes]
    if args.smoke and args.epochs is None:
        for trial in trials:
            trial.epochs = 1
    for trial in trials:
        trial.validate()
    domains = args.held_out_domains if args.mode == "crossdomain" else [None]
    if len(set(domains)) != len(domains):
        raise ValueError("Held-out domains must be unique")
    return [(trial, seed, domain) for domain in domains for trial in trials for seed in seeds]


def run_trial(args, cfg, seed, domain, environment, code):
    device = resolve_device(args.device)
    train_examples, dev_examples, sources = load_source_splits(ROOT, args.mode, domain)
    if args.smoke:
        train_examples, dev_examples = train_examples[:args.train_limit], dev_examples[:args.dev_limit]
    categories, sentiments, category_counts = training_vocabulary(train_examples, cfg)
    identity = {"experiment_version": VERSION, "config": asdict(cfg), "seed": seed,
                "mode": args.mode, "held_out_domain": domain, "source_domains": sources,
                "source_data_sha256": data_inventory(ROOT, sources), "source_sha256": code["source_sha256"],
                "environment": environment, "requested_device": args.device, "resolved_device": str(device),
                "smoke": args.smoke, "split_limits": [args.train_limit, args.dev_limit] if args.smoke else None}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve() / (f"smoke_{args.train_limit}_{args.dev_limit}" if args.smoke else "research")
    destination = output / args.mode / (domain or "all_domains") / cfg.trial_id / f"seed_{seed}"
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest and manifest["fingerprint"] != fingerprint:
        raise ValueError(f"Provenance changed for {destination}. Choose a new --output directory; prior results are preserved.")
    if manifest and manifest["status"] in ("dev_complete", "complete"):
        for filename in ("best.pt", "dev_metrics.json", "dev_predictions.jsonl", "dev_taxonomy.jsonl", "history.json", "audit.json"):
            if not (destination / filename).is_file():
                raise ValueError(f"Completed run is missing {filename}: {destination}")
        if manifest["status"] == "complete":
            for filename in ("test_metrics.json", "test_predictions.jsonl", "test_taxonomy.jsonl", "test_audit.json"):
                if not (destination / filename).is_file():
                    raise ValueError(f"Completed evaluation is missing {filename}: {destination}")
        if args.stage != "final" or manifest["status"] == "complete":
            print(f"Skip completed {cfg.name}, seed {seed}, {domain or args.mode}", flush=True)
            return
    elif args.stage == "final":
        raise ValueError(f"Final evaluation requires a completed development run: {destination}. Run --stage full first.")
    elif manifest and (manifest["status"] != "failed" or not args.retry_failed):
        raise ValueError(f"Incomplete run at {destination}; use --retry-failed for a recorded failure or a new --output")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision, local_files_only=cfg.offline, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("Offset-based tagging requires a fast tokenizer")
    if manifest is None:
        manifest = {"fingerprint": fingerprint, "identity": identity, "code": code, "status": "planned"}
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Run {cfg.name}, seed {seed}, {domain or args.mode}, stage {args.stage}", flush=True)
    if args.stage != "final":
        manifest["status"] = "running"
        atomic_write_json(manifest_path, manifest)
        try:
            train_ds = ExperimentDataset(train_examples, tokenizer, categories, sentiments, cfg)
            dev_ds = ExperimentDataset(dev_examples, tokenizer, categories, sentiments, cfg)
            audits = {"source_domains": sources, "categories": categories, "sentiments": sentiments,
                      "source_category_counts": category_counts,
                      "frequency_scope": "Eligible training explicit+NULL" if cfg.vocabulary_scope == "explicit-null" or cfg.null_head else "Eligible training explicit",
                      "train": dict(train_ds.audit), "dev": dict(dev_ds.audit),
                      "tokenized_train": summarize_tokenized_targets(train_ds),
                      "tokenized_dev": summarize_tokenized_targets(dev_ds),
                      "null_train_positive_targets": int(train_ds.null_targets.sum()),
                      "null_dev_positive_targets": int(dev_ds.null_targets.sum()),
                      "policy": "Exclude ambiguous occurrences, all conflicting spans and all members of overlaps. Keep NULL separately."}
            atomic_write_json(destination / "audit.json", audits)
            tokenizer.save_pretrained(destination / "tokenizer")
            model, training = train_trial(cfg, train_ds, dev_ds, categories, sentiments, category_counts, seed, device, destination)
            manifest.update(training=training, status="dev_complete", checkpoint_sha256=file_hash(destination / "best.pt"))
            manifest.pop("error", None)
            atomic_write_json(manifest_path, manifest)
            del model
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_write_json(manifest_path, manifest)
            raise
    else:
        if file_hash(destination / "best.pt") != manifest["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint contents changed")
        checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
        if checkpoint["categories"] != categories or checkpoint["sentiments"] != sentiments or checkpoint["config"] != asdict(cfg):
            raise ValueError("Frozen checkpoint vocabulary/config mismatch")
        model = ExperimentModel(cfg, len(categories), len(sentiments)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        tests = load_test_split(ROOT, args.mode, domain)  # Only this explicit stage opens test data.
        test_ds = ExperimentDataset(tests, tokenizer, categories, sentiments, cfg)
        report, records, outcomes = evaluate_trial(model, test_ds, categories, sentiments, category_counts, cfg, device,
                                                  threshold=checkpoint["null_threshold"])
        atomic_write_json(destination / "test_metrics.json", report)
        atomic_write_jsonl(destination / "test_predictions.jsonl", records)
        atomic_write_jsonl(destination / "test_taxonomy.jsonl", outcomes)
        atomic_write_json(destination / "test_audit.json", {"annotation_scope": dict(test_ds.audit),
                                                           "tokenized_targets": summarize_tokenized_targets(test_ds)})
        manifest.update(status="complete", test_data_sha256=data_inventory(ROOT, [domain] if domain else config.DOMAINS, ("test",)))
        atomic_write_json(manifest_path, manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_development_summary(output):
    """Give a compact comparison without hiding individual seed metrics."""
    manifests = sorted(output.glob("**/manifest.json"))
    rows = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] not in ("dev_complete", "complete"):
            continue
        identity = manifest["identity"]
        metrics = json.loads((path.parent / "dev_metrics.json").read_text(encoding="utf-8"))
        row = {"mode": identity["mode"], "domain": identity["held_out_domain"] or "all_domains",
               "trial_id": path.parent.parent.name, "recipe": identity["config"]["name"], "seed": identity["seed"],
               "scope": f"smoke_{identity['split_limits'][0]}_{identity['split_limits'][1]}" if identity["smoke"] else "research",
               "smoke": identity["smoke"], "best_epoch": manifest["training"]["best_epoch"],
               "explicit_f1": metrics["explicit"]["micro_f1"], "explicit_precision": metrics["explicit"]["micro_precision"],
               "explicit_recall": metrics["explicit"]["micro_recall"], "null_f1": metrics["null"]["micro_f1"],
               "combined_f1": metrics["combined"]["micro_f1"], "boundary_f1": metrics["explicit_boundary"]["f1"],
               "null_threshold": metrics["null_threshold"], "run_directory": str(path.parent)}
        components = metrics["taxonomy"]["metrics"]
        row.update(term_f1=components["aspect_span"]["f1"],
                   term_category_f1=components["term_plus_category"]["f1"],
                   term_sentiment_f1=components["aspect_plus_sentiment"]["f1"],
                   known_category_coverage=components["gold_category_coverage"])
        row.update({f"{name}_errors": metrics["taxonomy"]["primary_outcome_counts"][name]
                    for name in ("term", "category", "sentiment")})
        rare = next(group for group in metrics["taxonomy"]["by_rarity"] if group["rarity"] == "rare")
        row.update(rare_gold=rare["gold_triplets"], rare_recall=1 - rare["error_rate"] if rare["error_rate"] is not None else None)
        rows.append(row)
    if rows:
        for row in rows:
            upsert_csv_row(output / "development_results.csv", row, list(row), ("mode", "domain", "trial_id", "seed", "scope"))
        groups = defaultdict(list)
        for row in rows:
            groups[(row["mode"], row["domain"], row["trial_id"], row["scope"])].append(row)
        summaries = []
        for (mode, domain, trial_id, scope), members in sorted(groups.items()):
            values = [r["explicit_f1"] for r in members]
            summaries.append({"mode": mode, "domain": domain, "trial_id": trial_id, "scope": scope, "smoke": members[0]["smoke"],
                              "recipe": members[0]["recipe"], "seeds": [r["seed"] for r in members],
                              "explicit_f1_mean": statistics.mean(values),
                              "explicit_f1_sample_std": statistics.stdev(values) if len(values) > 1 else None,
                              "null_f1_mean": statistics.mean(r["null_f1"] for r in members),
                              "combined_f1_mean": statistics.mean(r["combined_f1"] for r in members)})
        atomic_write_json(output / "development_summary.json", summaries)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan = make_plan(args)
    except ValueError as exc:
        parser().error(str(exc))
    print(f"Version {VERSION}; {len(plan)} runs; {'SMOKE ONLY' if args.smoke else 'exploratory research'}; stage={args.stage}")
    print("Test access: frozen checkpoint evaluation" if args.stage == "final" else "Test access: disabled")
    for trial, seed, domain in plan:
        alpha = trial.weighted_ce_weight / (trial.ce_weight + trial.weighted_ce_weight)
        print(f"  {trial.name}: seed={seed}, held_out={domain}, loss={trial.loss_type}, heads={','.join(trial.loss_heads)}, "
              f"alpha={alpha:.3f}, gamma={trial.focal_gamma:g}, NULL={trial.null_head}, epochs={trial.epochs}, lr={trial.lr:g}; id={trial.trial_id}")
    if not args.execute or args.dry_run:
        print("Plan only. Add --execute to run. Use --smoke for a short integration check.")
        return
    environment, code = environment_inventory(), code_inventory(ROOT)
    environment_id = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[:12]
    environment_dir = args.output.resolve() / "environments" / environment_id
    atomic_write_json(environment_dir / "environment.json", environment)
    (environment_dir / "requirements.lock.txt").write_text("\n".join(environment["packages"]) + "\n", encoding="utf-8")
    for trial, seed, domain in plan:
        run_trial(args, trial, seed, domain, environment, code)
        write_development_summary(args.output.resolve())
    print(f"Finished. Development summary: {args.output.resolve() / 'development_results.csv'}")


if __name__ == "__main__":
    main()
