"""Cached-data AMP training with epoch-boundary recovery and full provenance."""

from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config
from src.artifacts import atomic_torch_save, atomic_write_json, atomic_write_jsonl
from src.data import summarize_tokenized_targets
from src.experiment_data import experiment_collate, load_source_splits, training_vocabulary
from src.experiment_losses import explicit_loss_functions, null_positive_weights
from src.experiment_runtime import data_inventory, environment_inventory, file_hash
from src.experiment_train import training_loss
from src.train import _collect_training_label_counts
from src.utils import resolve_device, set_seed

from .config import VERSION
from .data import CachedDataset, IA_TOKEN
from .evaluation import autocast_context, evaluate
from .losses import NullLoss
from .model import NullModel
from .storage import append_csv, digest, event, source_inventory, utc_now, verify_completed


def cpu_state(model):
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def rng_state(generator):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def save_best(directory, best):
    atomic_torch_save(directory / "best.pt", best["checkpoint"])
    atomic_write_json(directory / "dev_metrics.json", best["metrics"])
    atomic_write_jsonl(directory / "dev_predictions.jsonl", best["records"])
    atomic_write_jsonl(directory / "dev_taxonomy.jsonl", best["outcomes"])
    atomic_torch_save(directory / "dev_probabilities.pt", best["probabilities"])


def result_row(cfg, seed, stage, directory, manifest, metrics):
    return {"completed_at": utc_now(), "stage": stage, "trial_id": cfg.trial_id, "seed": seed,
            "smoke": manifest["identity"]["smoke"], "fingerprint": manifest["fingerprint"],
            "checkpoint_sha256": manifest["checkpoint_sha256"], "best_epoch": manifest["training"]["best_epoch"],
            "explicit_f1": metrics["explicit"]["micro_f1"], "null_f1": metrics["null"]["micro_f1"],
            "null_precision": metrics["null"]["micro_precision"], "null_recall": metrics["null"]["micro_recall"],
            "combined_f1": metrics["combined"]["micro_f1"], "null_threshold": metrics["null_threshold"],
            "null_loss": cfg.null_loss, "representation": cfg.representation, "sampling": cfg.negative_sampling,
            "null_head": cfg.null_head, "lr": cfg.lr, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
            "max_len": cfg.max_len, "ce_weight": cfg.ce_weight, "weighted_ce_weight": cfg.weighted_ce_weight,
            "null_loss_weight": cfg.null_loss_weight, "null_pos_weight_cap": cfg.null_pos_weight_cap,
            "precision": cfg.precision, "selection_metric": cfg.selection_metric,
            "parameters_json": json.dumps(asdict(cfg), sort_keys=True), "run_directory": str(directory)}


class Runtime:
    """One process, one GPU; tokenized train/dev data is reused across trials."""
    def __init__(self, root, output, device="auto", smoke=False, train_limit=64, dev_limit=32):
        self.root, self.output = Path(root), Path(output)
        self.device = resolve_device(device)
        self.requested_device = device
        self.smoke, self.train_limit, self.dev_limit = smoke, train_limit, dev_limit
        self.environment = environment_inventory()
        self.code = source_inventory(self.root)
        self.data_hashes = data_inventory(self.root, config.DOMAINS)
        self.train, self.dev, self.domains = load_source_splits(self.root, "indomain")
        if smoke:
            self.train, self.dev = self.train[:train_limit], self.dev[:dev_limit]
        self.datasets = {}
        self.tokenizer = None
        atomic_write_json(self.output / "environment.json", self.environment)

    def destination(self, cfg, seed):
        scope = f"smoke_{self.train_limit}_{self.dev_limit}" if self.smoke else "research"
        return self.output / scope / cfg.trial_id / f"seed_{seed}"

    def identity(self, cfg, seed):
        return {"version": VERSION, "config": asdict(cfg), "seed": seed, "mode": "indomain",
                "source_sha256": self.code["source_sha256"], "source_data_sha256": self.data_hashes,
                "environment": self.environment, "resolved_device": str(self.device),
                "smoke": self.smoke, "limits": [self.train_limit, self.dev_limit] if self.smoke else None,
                "input_protocol": "One reserved IA slot after CLS in all arms; text budget max_len-3"}

    def data(self, cfg):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, revision=cfg.model_revision,
                                                          local_files_only=cfg.offline, use_fast=True)
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IA_TOKEN]})
        key = (cfg.max_len, cfg.null_head and cfg.representation != "cls")
        if key not in self.datasets:
            print(f"Cache tokenization: max_len={cfg.max_len}, active_IA={key[1]}", flush=True)
            categories, sentiments, counts = training_vocabulary(self.train, cfg)
            train = CachedDataset(self.train, self.tokenizer, categories, sentiments, cfg)
            dev = CachedDataset(self.dev, self.tokenizer, categories, sentiments, cfg)
            self.datasets[key] = train, dev, categories, sentiments, counts
        return self.datasets[key]

    def run(self, cfg, seed, stage, retry_failed=False):
        cfg.validate()
        directory = self.destination(cfg, seed)
        identity = self.identity(cfg, seed)
        path = directory / "manifest.json"
        manifest = verify_completed(directory, identity) if path.exists() else {
            "fingerprint": digest(identity), "identity": identity, "status": "planned"}
        if manifest["status"] in ("dev_complete", "complete"):
            metrics = json.loads((directory / "dev_metrics.json").read_text(encoding="utf-8"))
            append_csv(self.output / "development_results.csv", result_row(cfg, seed, stage, directory, manifest, metrics),
                       ("fingerprint", "checkpoint_sha256"))
            print(f"Skip verified {cfg.name}, seed {seed}", flush=True)
            return directory
        if manifest["status"] == "failed" and not retry_failed:
            raise ValueError(f"Recorded failure at {directory}; use --retry-failed to resume")
        directory.mkdir(parents=True, exist_ok=True)
        manifest.update(status="running", stage=stage, started_at=utc_now(), pid=os.getpid())
        manifest.pop("error", None)
        atomic_write_json(path, manifest)
        event(self.output, "run_started", trial_id=cfg.trial_id, seed=seed, stage=stage)
        try:
            training = self._train(cfg, seed, stage, directory)
            manifest.update(status="dev_complete", training=training, completed_at=utc_now(),
                            checkpoint_sha256=file_hash(directory / "best.pt"))
            atomic_write_json(path, manifest)
            metrics = json.loads((directory / "dev_metrics.json").read_text(encoding="utf-8"))
            append_csv(self.output / "development_results.csv", result_row(cfg, seed, stage, directory, manifest, metrics),
                       ("fingerprint", "checkpoint_sha256"))
            event(self.output, "run_completed", trial_id=cfg.trial_id, seed=seed, stage=stage,
                  combined_f1=metrics["combined"]["micro_f1"], null_f1=metrics["null"]["micro_f1"])
            # The large optimizer recovery file is temporary; keep best.pt.
            resume = directory / "resume.pt"
            if resume.exists():
                resume.unlink()
            return directory
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error=f"{type(exc).__name__}: {exc}", stopped_at=utc_now())
            atomic_write_json(path, manifest)
            event(self.output, manifest["status"], trial_id=cfg.trial_id, seed=seed, error=manifest["error"])
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _train(self, cfg, seed, stage, directory):
        train, dev, categories, sentiments, category_counts = self.data(cfg)
        set_seed(seed)
        model = NullModel(cfg, categories, sentiments, self.tokenizer).to(self.device)
        labels = _collect_training_label_counts(train, cfg.batch_size)
        functions = explicit_loss_functions(cfg, labels, (3, len(categories), len(sentiments)), self.device)
        weights = null_positive_weights(train, cfg.null_pos_weight_cap).to(self.device)
        null_loss = NullLoss(cfg, weights)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=generator,
                            collate_fn=experiment_collate, pin_memory=self.device.type == "cuda")
        total_steps = len(loader) * cfg.epochs
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer, math.floor(cfg.warmup_ratio * total_steps), total_steps)
        amp = cfg.precision == "amp" and self.device.type == "cuda"
        # BERT's initial gradients can overflow at the default 65536 scale.
        # A smaller starting scale is still adjusted automatically thereafter.
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=cfg.amp_initial_scale)
        audit = {"train": train.audit, "dev": dev.audit, "categories": categories, "sentiments": sentiments,
                 "source_category_counts": category_counts, "tokenized_train": summarize_tokenized_targets(train),
                 "tokenized_dev": summarize_tokenized_targets(dev), "positive_null_pairs": len(train.pair_counts),
                 "null_output_labels": train.num_null_labels, "null_train_targets": int(train.null_targets.sum()),
                 "text_budget_tokens": cfg.max_len - 3, "active_IA": cfg.representation != "cls" and cfg.null_head,
                 "null_positive_weights": weights.cpu().tolist() if cfg.null_loss != "asl" else None,
                 "sampling_reduction": "cell_mean" if cfg.negative_sampling == "all" else "mean_of_positive_and_sampled_negative_means"}
        atomic_write_json(directory / "audit.json", audit)
        self.tokenizer.save_pretrained(directory / "tokenizer")
        history, best, best_key, start_epoch, optimizer_steps = [], None, None, 1, 0
        resume_path = directory / "resume.pt"
        if resume_path.exists():
            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
            if saved["fingerprint"] != digest(self.identity(cfg, seed)):
                raise ValueError("Resume checkpoint provenance changed")
            model.load_state_dict(saved["state_dict"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            history, best, best_key = saved["history"], saved["best"], saved["best_key"]
            start_epoch, optimizer_steps = saved["epoch_completed"] + 1, saved["optimizer_steps"]
            restore_rng(saved["rng"], generator)
            save_best(directory, best)
            atomic_write_json(directory / "history.json", history)
            del saved
            print(f"Resume {cfg.name}, seed {seed}, from epoch {start_epoch}/{cfg.epochs}", flush=True)
        attempt = uuid.uuid4().hex[:12]
        for epoch in range(start_epoch, cfg.epochs + 1):
            started = time.monotonic()
            last_update = started
            model.train()
            total, examples, skipped_updates = 0.0, 0, 0
            for step, batch in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(cfg, self.device):
                    hidden, bio = model.encode_and_tag(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
                    loss = training_loss(model, hidden, bio, batch, functions, null_loss, cfg, self.device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite objective in epoch {epoch}, batch {step}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if not amp and not torch.isfinite(norm):
                    raise RuntimeError("Non-finite FP32 gradients")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped_updates += 1
                else:
                    scheduler.step()
                    optimizer_steps += 1
                total += float(loss.detach()) * len(batch["input_ids"])
                examples += len(batch["input_ids"])
                if time.monotonic() - last_update > 30:
                    print(f"  {cfg.name}, seed {seed}, epoch {epoch}: batch {step}/{len(loader)}, objective={total/examples:.4f}", flush=True)
                    last_update = time.monotonic()
            metrics, records, outcomes, probabilities = evaluate(model, dev, categories, sentiments,
                category_counts, cfg, self.device, train.pair_counts)
            key = (metrics[cfg.selection_metric]["micro_f1"], metrics["null"]["micro_f1"], -metrics["unweighted_loss"], -epoch)
            current = cpu_state(model)
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best = {"checkpoint": {"state_dict": current, "config": asdict(cfg), "categories": categories,
                        "sentiments": sentiments, "seed": seed, "best_epoch": epoch,
                        "null_threshold": metrics["null_threshold"]}, "metrics": metrics, "records": records,
                        "outcomes": outcomes, "probabilities": probabilities}
            elapsed = time.monotonic() - started
            history.append({"epoch": epoch, "attempt": attempt, "seconds": elapsed, "objective": total/examples,
                            "skipped_amp_updates": skipped_updates, "development": metrics})
            # Atomic recovery is committed before publishing epoch outputs. A
            # resumed run restores both current and selected-best state together.
            atomic_torch_save(resume_path, {"fingerprint": digest(self.identity(cfg, seed)), "epoch_completed": epoch,
                "state_dict": current, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": rng_state(generator), "history": history,
                "best": best, "best_key": best_key, "optimizer_steps": optimizer_steps})
            atomic_write_json(directory / "history.json", history)
            if improved:
                save_best(directory, best)
            append_csv(self.output / "epoch_results.csv", {"at": utc_now(), "trial_id": cfg.trial_id,
                "seed": seed, "stage": stage, "smoke": self.smoke, "attempt": attempt, "epoch": epoch,
                "seconds": elapsed, "objective": total/examples, "skipped_amp_updates": skipped_updates,
                "explicit_f1": metrics["explicit"]["micro_f1"], "null_f1": metrics["null"]["micro_f1"],
                "null_precision": metrics["null"]["micro_precision"], "null_recall": metrics["null"]["micro_recall"],
                "combined_f1": metrics["combined"]["micro_f1"], "null_threshold": metrics["null_threshold"]})
            event(self.output, "epoch_completed", trial_id=cfg.trial_id, seed=seed, epoch=epoch, seconds=elapsed)
            print(f"  {cfg.name}, seed {seed}, epoch {epoch}/{cfg.epochs}: explicit F1={metrics['explicit']['micro_f1']:.4f}; "
                  f"NULL F1={metrics['null']['micro_f1']:.4f}; combined F1={metrics['combined']['micro_f1']:.4f}; "
                  f"threshold={metrics['null_threshold']:g}; {elapsed/60:.1f} min", flush=True)
        if not best:
            raise ValueError("No completed training epoch")
        if optimizer_steps == 0:
            raise RuntimeError("No optimizer updates succeeded. Reduce --amp-initial-scale, or use --precision fp32 in a new run.")
        # Covers recovery after the final epoch but before run completion.
        save_best(directory, best)
        return {"best_epoch": best["checkpoint"]["best_epoch"], "null_threshold": best["checkpoint"]["null_threshold"],
                "optimizer_steps": optimizer_steps, "completed_epochs": len(history),
                "effective_precision": "fp16_amp" if amp else "fp32", "epoch_seconds": [r["seconds"] for r in history]}
