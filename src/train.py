"""Training, validation, checkpoint selection, and predicted-span inference."""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.artifacts import atomic_torch_save, atomic_write_json
from src.data import collate_fn
from src.evaluate import (
    complete_triplet_scores,
    decode_bio_to_spans,
    error_analysis,
    per_domain_scores,
    precision_recall_by_label,
    rare_label_recall,
)
from src.losses import get_evaluation_loss_fns, get_loss_fns
from src.model import build_model, count_trainable_params
from src.utils import resolve_device, set_seed


def _flatten_labels(label_tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    nonempty = [labels for labels in label_tensors if labels.numel() > 0]
    if not nonempty:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(nonempty)


def _classification_loss(logits, labels, loss_fn, device):
    if logits is None or labels.numel() == 0:
        return None
    labels = labels.to(device)
    valid = labels != -100
    if not valid.any():
        return None
    return loss_fn(logits[valid], labels[valid])


def compute_batch_loss(
    bio_logits,
    category_logits,
    sentiment_logits,
    batch,
    loss_fns,
    cfg,
    device,
):
    """Compute the joint loss while safely ignoring unknown class labels."""
    bio_fn, category_fn, sentiment_fn = loss_fns
    bio_labels = batch["bio_labels"].to(device)
    loss = bio_fn(
        bio_logits.reshape(-1, bio_logits.size(-1)), bio_labels.reshape(-1),
    )

    category_labels = _flatten_labels(batch["category_labels"])
    category_loss = _classification_loss(
        category_logits, category_labels, category_fn, device,
    )
    if category_loss is not None:
        loss = loss + cfg.category_loss_weight * category_loss

    sentiment_labels = _flatten_labels(batch["sentiment_labels"])
    sentiment_loss = _classification_loss(
        sentiment_logits, sentiment_labels, sentiment_fn, device,
    )
    if sentiment_loss is not None:
        loss = loss + cfg.sentiment_loss_weight * sentiment_loss

    return loss


def _summed_classification_loss(logits, labels, loss_fn, device):
    if logits is None or labels.numel() == 0:
        return 0.0, 0
    labels = labels.to(device)
    valid = labels != -100
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0
    return float(loss_fn(logits[valid], labels[valid]).item()), count


def compute_evaluation_loss_components(
    bio_logits,
    category_logits,
    sentiment_logits,
    batch,
    loss_fns,
    device,
):
    """Return summed task losses and exact valid-label denominators."""
    bio_fn, category_fn, sentiment_fn = loss_fns
    bio_labels = batch["bio_labels"].to(device).reshape(-1)
    flat_bio_logits = bio_logits.reshape(-1, bio_logits.size(-1))
    valid_bio = bio_labels != -100
    bio_count = int(valid_bio.sum().item())
    bio_sum = (
        float(bio_fn(flat_bio_logits[valid_bio], bio_labels[valid_bio]).item())
        if bio_count else 0.0
    )

    category_sum, category_count = _summed_classification_loss(
        category_logits,
        _flatten_labels(batch["category_labels"]),
        category_fn,
        device,
    )
    sentiment_sum, sentiment_count = _summed_classification_loss(
        sentiment_logits,
        _flatten_labels(batch["sentiment_labels"]),
        sentiment_fn,
        device,
    )
    return {
        "bio": (bio_sum, bio_count),
        "category": (category_sum, category_count),
        "sentiment": (sentiment_sum, sentiment_count),
    }


def _collect_training_label_counts(dataset, batch_size: int):
    """Collect all three task labels in one deterministic dataset pass."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    bio_counts: List[int] = []
    category_counts: List[int] = []
    sentiment_counts: List[int] = []
    for batch in tqdm(loader, desc="label counts", leave=False):
        bio_counts.extend(
            int(label) for label in batch["bio_labels"].reshape(-1).tolist()
            if label != -100
        )
        category_counts.extend(
            int(label) for labels in batch["category_labels"]
            for label in labels.tolist() if label != -100
        )
        sentiment_counts.extend(
            int(label) for labels in batch["sentiment_labels"]
            for label in labels.tolist() if label != -100
        )
    return bio_counts, category_counts, sentiment_counts


def _decode_batch_predictions(
    model,
    hidden_states,
    bio_logits,
    batch,
    category_vocab: Sequence[str],
    sentiment_vocab: Sequence[str],
):
    tag_ids = bio_logits.argmax(dim=-1).detach().cpu()
    offsets = batch["offset_mapping"].cpu()
    attention_mask = batch["attention_mask"].cpu()
    predicted_spans = []

    for example_index in range(tag_ids.size(0)):
        valid_mask = [
            bool(attention_mask[example_index, token_index])
            and int(offsets[example_index, token_index, 1])
            > int(offsets[example_index, token_index, 0])
            for token_index in range(tag_ids.size(1))
        ]
        predicted_spans.append(
            decode_bio_to_spans(tag_ids[example_index].tolist(), valid_mask)
        )

    category_logits, sentiment_logits = model.classify_spans(
        hidden_states, predicted_spans,
    )
    category_ids = (
        category_logits.argmax(dim=-1).detach().cpu().tolist()
        if category_logits is not None else []
    )
    sentiment_ids = (
        sentiment_logits.argmax(dim=-1).detach().cpu().tolist()
        if sentiment_logits is not None else []
    )

    records = []
    flat_index = 0
    for batch_index, spans in enumerate(predicted_spans):
        predicted_triplets = []
        sentence = batch["sentence"][batch_index]
        for token_start, token_end in spans:
            category_id = category_ids[flat_index]
            sentiment_id = sentiment_ids[flat_index]
            flat_index += 1

            char_start = int(offsets[batch_index, token_start, 0])
            char_end = int(offsets[batch_index, token_end, 1])
            if char_end <= char_start:
                continue
            predicted_triplets.append(
                (
                    sentence[char_start:char_end],
                    category_vocab[category_id],
                    sentiment_vocab[sentiment_id],
                )
            )

        records.append(
            {
                "example_id": int(batch["example_id"][batch_index]),
                "domain": batch["domain"][batch_index],
                "sentence": sentence,
                "gold_triplets": [
                    list(triplet) for triplet in batch["gold_triplets"][batch_index]
                ],
                "predicted_triplets": [
                    list(triplet) for triplet in predicted_triplets
                ],
            }
        )
    return records


def _evaluation_report(records, average_loss, rare_category_labels):
    gold = [record["gold_triplets"] for record in records]
    predicted = [record["predicted_triplets"] for record in records]
    domains = [record["domain"] for record in records]
    return {
        "num_examples": len(records),
        "loss": average_loss,
        "complete_triplet": complete_triplet_scores(gold, predicted),
        "category_by_label": precision_recall_by_label(gold, predicted, 1),
        "sentiment_by_label": precision_recall_by_label(gold, predicted, 2),
        "rare_category_labels": list(rare_category_labels),
        "rare_category_recall": rare_label_recall(
            gold, predicted, list(rare_category_labels), 1,
        ),
        "per_domain": per_domain_scores(gold, predicted, domains),
        "error_analysis": error_analysis(gold, predicted, domains),
    }


def evaluate_model(
    model,
    dataset,
    category_vocab: Sequence[str],
    sentiment_vocab: Sequence[str],
    cfg,
    device,
    rare_category_labels: Sequence[str] = (),
    loss_fns=None,
    description: str = "evaluate",
):
    """Evaluate with predicted BIO spans and return metrics plus JSONL rows."""
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn,
    )
    model.eval()
    records = []
    task_loss_sums = {"bio": 0.0, "category": 0.0, "sentiment": 0.0}
    task_label_counts = {"bio": 0, "category": 0, "sentiment": 0}

    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden_states, bio_logits = model.encode_and_tag(
                input_ids, attention_mask,
            )

            if loss_fns is not None:
                category_logits, sentiment_logits = model.classify_spans(
                    hidden_states, batch["span_boundaries"],
                )
                components = compute_evaluation_loss_components(
                    bio_logits, category_logits, sentiment_logits, batch,
                    loss_fns, device,
                )
                for task, (loss_sum, label_count) in components.items():
                    task_loss_sums[task] += loss_sum
                    task_label_counts[task] += label_count

            records.extend(
                _decode_batch_predictions(
                    model, hidden_states, bio_logits, batch,
                    category_vocab, sentiment_vocab,
                )
            )

    if loss_fns is None or task_label_counts["bio"] == 0:
        average_loss = None
    else:
        average_loss = task_loss_sums["bio"] / task_label_counts["bio"]
        if task_label_counts["category"]:
            average_loss += cfg.category_loss_weight * (
                task_loss_sums["category"] / task_label_counts["category"]
            )
        if task_label_counts["sentiment"]:
            average_loss += cfg.sentiment_loss_weight * (
                task_loss_sums["sentiment"] / task_label_counts["sentiment"]
            )
    return _evaluation_report(records, average_loss, rare_category_labels), records


def train_one_config(
    cfg,
    train_ds,
    val_ds,
    category_vocab,
    sentiment_vocab,
    seed: int,
    device="auto",
    checkpoint_path: Optional[Path] = None,
    history_path: Optional[Path] = None,
    rare_category_labels: Sequence[str] = (),
):
    """Train for the fixed epoch budget and restore the best dev checkpoint."""
    set_seed(seed)
    device = resolve_device(device)
    model = build_model(cfg, len(category_vocab), len(sentiment_vocab)).to(device)
    n_trainable = count_trainable_params(model)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    if not train_loader:
        raise ValueError("Training dataset is empty")

    start_time = time.time()
    bio_counts, category_counts, sentiment_counts = _collect_training_label_counts(
        train_ds, cfg.batch_size,
    )
    training_loss_fns = tuple(
        loss_fn.to(device) for loss_fn in get_loss_fns(
            cfg,
            bio_counts,
            category_counts,
            sentiment_counts,
            num_categories=len(category_vocab),
            num_sentiments=len(sentiment_vocab),
        )
    )
    evaluation_loss_fns = tuple(
        loss_fn.to(device) for loss_fn in get_evaluation_loss_fns()
    )

    optimizer = AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps,
    )

    history = []
    best_state = None
    best_epoch = None
    best_score = -math.inf
    best_dev_loss = math.inf
    best_dev_metrics: Dict = {}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(
            train_loader,
            desc=f"[{cfg.name} seed={seed}] epoch {epoch + 1}/{cfg.epochs}",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hidden_states, bio_logits = model.encode_and_tag(
                input_ids, attention_mask,
            )
            category_logits, sentiment_logits = model.classify_spans(
                hidden_states, batch["span_boundaries"],
            )
            loss = compute_batch_loss(
                bio_logits, category_logits, sentiment_logits, batch,
                training_loss_fns, cfg, device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item())

        train_loss = epoch_loss / len(train_loader)
        dev_metrics, _ = evaluate_model(
            model,
            val_ds,
            category_vocab,
            sentiment_vocab,
            cfg,
            device,
            rare_category_labels=rare_category_labels,
            loss_fns=evaluation_loss_fns,
            description=f"dev epoch {epoch + 1}",
        )
        dev_score = dev_metrics["complete_triplet"]["micro_f1"]
        dev_loss = (
            dev_metrics["loss"]
            if dev_metrics["loss"] is not None else math.inf
        )
        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_micro_precision": dev_metrics["complete_triplet"]["micro_precision"],
            "dev_micro_recall": dev_metrics["complete_triplet"]["micro_recall"],
            "dev_micro_f1": dev_score,
            "dev_macro_f1": dev_metrics["complete_triplet"]["macro_f1"],
            "dev_rare_category_recall": dev_metrics["rare_category_recall"],
        }
        history.append(history_row)

        improved = dev_score > best_score + 1e-12
        tied_with_lower_loss = (
            abs(dev_score - best_score) <= 1e-12 and dev_loss < best_dev_loss
        )
        if improved or tied_with_lower_loss:
            best_score = dev_score
            best_dev_loss = dev_loss
            best_epoch = epoch + 1
            best_dev_metrics = dev_metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if checkpoint_path is not None:
                atomic_torch_save(
                    checkpoint_path,
                    {
                        "model_state_dict": best_state,
                        "best_epoch": best_epoch,
                        "selection_metric": "dev_complete_triplet_micro_f1",
                        "selection_score": best_score,
                        "dev_loss": best_dev_metrics["loss"],
                    },
                )

        if history_path is not None:
            atomic_write_json(history_path, history)
        dev_loss_text = (
            f"{dev_metrics['loss']:.4f}"
            if dev_metrics["loss"] is not None else "n/a"
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"dev_loss={dev_loss_text} dev_micro_f1={dev_score:.4f}"
        )

    if best_state is None:
        raise RuntimeError("Training completed without producing a checkpoint")
    model.load_state_dict(best_state)

    return {
        "model": model,
        "trainable_params": n_trainable,
        "training_time_sec": time.time() - start_time,
        "device": str(device),
        "history": history,
        "best_epoch": best_epoch,
        "selection_metric": "dev_complete_triplet_micro_f1",
        "best_dev_metrics": best_dev_metrics,
    }
