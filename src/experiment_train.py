"""Corrected training, development selection, and separate NULL evaluation."""

from collections import Counter
from dataclasses import asdict
import math
import time

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.artifacts import atomic_torch_save, atomic_write_json, atomic_write_jsonl
from src.evaluate import complete_triplet_scores, decode_bio_to_spans, precision_recall_by_label
from src.experiment_data import experiment_collate
from src.experiment_losses import explicit_loss_functions, null_positive_weights
from src.experiment_model import ExperimentModel
from src.losses import get_evaluation_loss_fns
from src.result_analysis import projected_micro_scores
from src.team_diagnostics import analyze_records
from src.train import _classification_loss, _collect_training_label_counts, _flatten_labels, compute_evaluation_loss_components
from src.utils import set_seed


def training_loss(model, hidden, bio_logits, batch, losses, null_loss, cfg, device):
    category_logits, sentiment_logits = model.classify_spans(hidden, batch["span_boundaries"])
    labels = batch["bio_labels"].to(device).reshape(-1)
    valid = labels != -100
    flat_logits = bio_logits.reshape(-1, 3)
    total = cfg.bio_loss_weight * losses[0](flat_logits[valid], labels[valid]) if valid.any() else hidden.sum() * 0
    for logits, key, loss_fn, weight in (
        (category_logits, "category_labels", losses[1], cfg.category_loss_weight),
        (sentiment_logits, "sentiment_labels", losses[2], cfg.sentiment_loss_weight),
    ):
        value = _classification_loss(logits, _flatten_labels(batch[key]), loss_fn, device)
        if value is not None:
            total = total + weight * value
    if cfg.null_head:
        total = total + cfg.null_loss_weight * null_loss(model.classify_null(hidden), batch["null_targets"].to(device))
    return total


def decode_predictions(model, hidden, bio_logits, batch, categories, sentiments, minimum_threshold):
    tags = bio_logits.argmax(-1).cpu().tolist()
    offsets = batch["offset_mapping"].tolist()
    spans = [decode_bio_to_spans(row, [bool(mask) and end > start for mask, (start, end) in zip(attention, off)])
             for row, attention, off in zip(tags, batch["attention_mask"].tolist(), offsets)]
    category_logits, sentiment_logits = model.classify_spans(hidden, spans)
    category_ids = category_logits.argmax(-1).cpu().tolist() if category_logits is not None else []
    sentiment_ids = sentiment_logits.argmax(-1).cpu().tolist() if sentiment_logits is not None else []
    null_logits = model.classify_null(hidden)
    probabilities = null_logits.sigmoid().cpu() if null_logits is not None else None
    records, flat_index = [], 0
    for index, boundaries in enumerate(spans):
        sentence = batch["sentence"][index]
        predicted, predicted_spans = [], []
        for token_start, token_end in boundaries:
            start, end = offsets[index][token_start][0], offsets[index][token_end][1]
            triplet = [sentence[start:end], categories[category_ids[flat_index]], sentiments[sentiment_ids[flat_index]]]
            flat_index += 1
            predicted.append(triplet)
            predicted_spans.append({"start": start, "end": end, "token_start": token_start,
                                    "token_end": token_end, "triplet": triplet})
        candidates = []
        if probabilities is not None:
            for pair in (probabilities[index] >= minimum_threshold).nonzero().flatten().tolist():
                candidates.append([categories[pair // len(sentiments)], sentiments[pair % len(sentiments)],
                                   float(probabilities[index, pair])])
        records.append({"example_id": batch["example_id"][index], "domain": batch["domain"][index],
                        "sentence": sentence, "gold_triplets": [list(t) for t in batch["gold_triplets"][index]],
                        "predicted_triplets": predicted, "gold_spans": batch["gold_spans"][index],
                        "predicted_spans": predicted_spans,
                        "gold_null_triplets": [list(t) for t in batch["gold_null_triplets"][index]],
                        "null_candidates": candidates})
    return records


def apply_null_threshold(records, threshold):
    for row in records:
        row["predicted_null_triplets"] = [["NULL", category, sentiment] for category, sentiment, probability
                                          in row["null_candidates"] if probability >= threshold]


def choose_null_threshold(records, categories, cfg):
    """Development only; ties prefer the predeclared reference threshold."""
    gold = [row["gold_null_triplets"] for row in records]
    curve = []
    for threshold in sorted(set((*cfg.null_thresholds, cfg.null_threshold))):
        apply_null_threshold(records, threshold)
        scores = complete_triplet_scores(gold, [row["predicted_null_triplets"] for row in records], categories)
        curve.append({"threshold": threshold, **scores})
    if not any(gold):
        return cfg.null_threshold, curve  # No positive support: threshold is unidentifiable.
    best = max(curve, key=lambda row: (row["micro_f1"], -abs(row["threshold"] - cfg.null_threshold), -row["threshold"]))
    return best["threshold"], curve


def score_records(records, categories, category_counts):
    gold = [row["gold_triplets"] for row in records]
    predicted = [row["predicted_triplets"] for row in records]
    null_gold = [row["gold_null_triplets"] for row in records]
    null_predicted = [row["predicted_null_triplets"] for row in records]
    taxonomy, outcomes = analyze_records(records, category_counts, categories)
    if taxonomy["metrics"]["source_known_gold_triplets"] == 0:
        taxonomy["metrics"]["source_known_exact_triplet"] = None
    boundary_gold = [[(span["start"], span["end"]) for span in row["gold_spans"]] for row in records]
    boundary_predicted = [[(span["start"], span["end"]) for span in row["predicted_spans"]] for row in records]
    null_outcomes = Counter()
    for g, p in zip(null_gold, null_predicted):
        p = {tuple(t) for t in p}
        for triplet in {tuple(t) for t in g}:
            outcome = "correct" if triplet in p else "category_missing" if not any(t[1] == triplet[1] for t in p) else "sentiment"
            null_outcomes[outcome] += 1
    report = {
        "num_examples": len(records),
        "explicit": complete_triplet_scores(gold, predicted, categories),
        "null": complete_triplet_scores(null_gold, null_predicted, categories),
        "combined": complete_triplet_scores([g + n for g, n in zip(gold, null_gold)],
                                             [p + n for p, n in zip(predicted, null_predicted)], categories),
        "explicit_boundary": projected_micro_scores(boundary_gold, boundary_predicted),
        "taxonomy": taxonomy,
        "category_by_label": precision_recall_by_label(gold, predicted, 1),
        "sentiment_by_label": precision_recall_by_label(gold, predicted, 2),
        "null_outcome_counts": dict(null_outcomes),
        "null_gold": sum(len({tuple(t) for t in row}) for row in null_gold),
        "null_unseen_category_gold": sum(t[1] not in categories for row in null_gold for t in {tuple(t) for t in row}),
        "macro_category_universe": categories,
        "scope": "Eligible explicit annotations and deduplicated NULL; other annotation exclusions remain. Not full-annotation TASD.",
    }
    report["per_domain"] = {}
    for domain in sorted({row["domain"] for row in records}):
        rows = [row for row in records if row["domain"] == domain]
        report["per_domain"][domain] = {
            scope: complete_triplet_scores([r[gold_key] for r in rows], [r[pred_key] for r in rows], categories)
            for scope, gold_key, pred_key in (("explicit", "gold_triplets", "predicted_triplets"),
                                              ("null", "gold_null_triplets", "predicted_null_triplets"))
        }
        domain_taxonomy, _ = analyze_records(rows, category_counts, categories)
        if domain_taxonomy["metrics"]["source_known_gold_triplets"] == 0:
            domain_taxonomy["metrics"]["source_known_exact_triplet"] = None
        report["per_domain"][domain]["components"] = domain_taxonomy["metrics"]
        report["per_domain"][domain]["primary_outcome_counts"] = domain_taxonomy["primary_outcome_counts"]
    return report, outcomes


def evaluate_trial(model, dataset, categories, sentiments, category_counts, cfg, device, *, tune_threshold=False, threshold=None):
    if not len(dataset):
        raise ValueError("The evaluation split is empty")
    model.eval()
    records, sums, counts = [], Counter(), Counter()
    loss_fns = get_evaluation_loss_fns()
    minimum = min((*cfg.null_thresholds, cfg.null_threshold)) if tune_threshold else (threshold or cfg.null_threshold)
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=experiment_collate):
            hidden, bio_logits = model.encode_and_tag(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            cat_logits, sent_logits = model.classify_spans(hidden, batch["span_boundaries"])
            components = compute_evaluation_loss_components(bio_logits, cat_logits, sent_logits, batch, loss_fns, device)
            if cfg.null_head:
                targets = batch["null_targets"].to(device)
                components["null"] = (float(F.binary_cross_entropy_with_logits(model.classify_null(hidden), targets, reduction="sum")), targets.numel())
            for task, (value, count) in components.items():
                sums[task] += value
                counts[task] += count
            records.extend(decode_predictions(model, hidden, bio_logits, batch, categories, sentiments, minimum))
    threshold = cfg.null_threshold if threshold is None else threshold
    curve = []
    if cfg.null_head and tune_threshold:
        threshold, curve = choose_null_threshold(records, categories, cfg)
    apply_null_threshold(records, threshold)
    for row in records:
        del row["null_candidates"]
    report, outcomes = score_records(records, categories, category_counts)
    means = {task: sums[task] / count if count else None for task, count in counts.items()}
    report.update(unweighted_loss=sum(value for value in means.values() if value is not None),
                  unweighted_loss_by_head=means, loss_label_counts=dict(counts),
                  null_threshold=threshold, null_threshold_curve=curve)
    return report, records, outcomes


def train_trial(cfg, train_ds, dev_ds, categories, sentiments, category_counts, seed, device, destination,
                model_factory=None):
    cfg.validate()
    if not len(train_ds) or not len(dev_ds):
        raise ValueError("Training and development splits must be nonempty")
    set_seed(seed)
    model = (model_factory or ExperimentModel)(cfg, len(categories), len(sentiments)).to(device)
    labels = _collect_training_label_counts(train_ds, cfg.batch_size)
    loss_fns = explicit_loss_functions(cfg, labels, (3, len(categories), len(sentiments)), device)
    null_weights = null_positive_weights(train_ds, cfg.null_pos_weight_cap).to(device) if cfg.null_head else None
    null_loss = torch.nn.BCEWithLogitsLoss(pos_weight=null_weights) if cfg.null_head else None
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=experiment_collate,
                        generator=torch.Generator().manual_seed(seed))
    total_steps = len(loader) * cfg.epochs
    warmup_steps = math.floor(cfg.warmup_ratio * total_steps)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_key, history = None, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, examples_seen, clipped_norm = 0.0, 0, 0.0
        last_update = time.monotonic()
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            hidden, bio_logits = model.encode_and_tag(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = training_loss(model, hidden, bio_logits, batch, loss_fns, null_loss, cfg, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
            clipped_norm = max(clipped_norm, float(norm))
            optimizer.step()
            scheduler.step()
            size = len(batch["example_id"])
            total_loss += float(loss.detach()) * size
            examples_seen += size
            if time.monotonic() - last_update >= 30:
                print(f"  seed {seed}, epoch {epoch}: batch {step}/{len(loader)}, mean objective {total_loss / examples_seen:.4f}", flush=True)
                last_update = time.monotonic()
        report, records, outcomes = evaluate_trial(model, dev_ds, categories, sentiments, category_counts, cfg, device, tune_threshold=True)
        key = (report[cfg.selection_metric]["micro_f1"], -report["unweighted_loss"], -epoch)
        history.append({"epoch": epoch, "mean_batch_training_objective": total_loss / examples_seen,
                        "max_gradient_norm_before_clipping": clipped_norm, "development": report})
        print(f"  seed {seed}, epoch {epoch}/{cfg.epochs}: dev explicit F1={report['explicit']['micro_f1']:.4f}; "
              f"NULL F1={report['null']['micro_f1']:.4f}; combined F1={report['combined']['micro_f1']:.4f}", flush=True)
        atomic_write_json(destination / "history.json", history)
        if best_key is None or key > best_key:
            best_key = key
            atomic_torch_save(destination / "best.pt", {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "config": asdict(cfg), "categories": categories, "sentiments": sentiments,
                "seed": seed, "best_epoch": epoch, "null_threshold": report["null_threshold"],
            })
            atomic_write_json(destination / "dev_metrics.json", report)
            atomic_write_jsonl(destination / "dev_predictions.jsonl", records)
            atomic_write_jsonl(destination / "dev_taxonomy.jsonl", outcomes)
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return model, {"best_epoch": checkpoint["best_epoch"], "null_threshold": checkpoint["null_threshold"],
                   "optimizer_steps": total_steps, "warmup_steps": warmup_steps,
                   "training_label_counts": {head: dict(Counter(values)) for head, values in zip(("bio", "category", "sentiment"), labels)},
                   "null_positive_weights": null_weights.cpu().tolist() if null_weights is not None else None}
