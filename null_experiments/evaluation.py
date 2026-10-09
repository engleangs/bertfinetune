"""Development calibration, exact scoring and NULL-specific diagnostic slices."""

from contextlib import nullcontext
from collections import Counter
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.evaluate import complete_triplet_scores
from src.experiment_data import experiment_collate
from src.experiment_train import decode_predictions, score_records
from src.losses import get_evaluation_loss_fns
from src.train import compute_evaluation_loss_components


def autocast_context(cfg, device):
    return torch.autocast("cuda", dtype=torch.float16) if getattr(cfg, "precision", "fp32") == "amp" and device.type == "cuda" else nullcontext()


def count_scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "micro_precision": precision, "micro_recall": recall,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def threshold_curve(records, probabilities, categories, sentiments, thresholds):
    """Count-based sweep; unknown gold labels remain false negatives."""
    category_ids = {c: i for i, c in enumerate(categories)}
    sentiment_ids = {s: i for i, s in enumerate(sentiments)}
    gold = torch.zeros_like(probabilities, dtype=torch.bool)
    total_gold = 0
    for index, row in enumerate(records):
        targets = set(map(tuple, row["gold_null_triplets"]))
        total_gold += len(targets)
        for _, category, sentiment in targets:
            if category in category_ids and sentiment in sentiment_ids:
                gold[index, category_ids[category] * len(sentiments) + sentiment_ids[sentiment]] = True
    explicit = complete_triplet_scores([r["gold_triplets"] for r in records],
                                      [r["predicted_triplets"] for r in records], categories)
    curve = []
    for threshold in sorted(set((*thresholds, 1.0))):
        # 1.0 is an explicit abstention option, including saturated logits.
        predicted = probabilities >= threshold if threshold < 1 else torch.zeros_like(gold)
        tp, predicted_count = int((predicted & gold).sum()), int(predicted.sum())
        null = count_scores(tp, predicted_count - tp, total_gold - tp)
        combined = count_scores(*[explicit[key] + null[key] for key in
                                ("true_positives", "false_positives", "false_negatives")])
        curve.append({"threshold": threshold, "null": null, "combined": combined})
    return curve


def choose_threshold(curve, objective="combined"):
    metric = "null" if objective == "null" else "combined"
    return max(curve, key=lambda p: (p[metric]["micro_f1"], p["null"]["micro_f1"], p["threshold"]))["threshold"]


def apply_threshold(records, probabilities, categories, sentiments, threshold):
    for index, row in enumerate(records):
        pairs = (probabilities[index] >= threshold).nonzero().flatten().tolist() if threshold < 1 else []
        row["predicted_null_triplets"] = [["NULL", categories[pair // len(sentiments)],
                                          sentiments[pair % len(sentiments)]] for pair in pairs]
        row.pop("null_candidates", None)


def null_diagnostics(records, pair_counts, categories):
    def projection(rows):
        gold = [[("NULL", t[1], "category") for t in row["gold_null_triplets"]] for row in rows]
        predicted = [[("NULL", t[1], "category") for t in row["predicted_null_triplets"]] for row in rows]
        return complete_triplet_scores(gold, predicted, categories)
    groups = {"null_only": [], "mixed": [], "without_gold_null": []}
    rarity = {name: Counter() for name in ("unseen_pair", "rare_pair", "frequent_pair")}
    presence, errors = Counter(), Counter()
    for row in records:
        gold, predicted = set(map(tuple, row["gold_null_triplets"])), set(map(tuple, row["predicted_null_triplets"]))
        name = "mixed" if gold and row["gold_triplets"] else "null_only" if gold else "without_gold_null"
        groups[name].append(row)
        presence["tp"] += bool(gold and predicted)
        presence["fp"] += bool(predicted and not gold)
        presence["fn"] += bool(gold and not predicted)
        errors["fp_on_sentence_without_gold_null"] += len(predicted - gold) if not gold else 0
        for _, category, sentiment in predicted - gold:
            errors["fp_without_positive_training_pair"] += (category, sentiment) not in pair_counts
        for triplet in gold:
            support = pair_counts.get(tuple(triplet[1:]), 0)
            bucket = "unseen_pair" if support == 0 else "rare_pair" if support <= 5 else "frequent_pair"
            rarity[bucket]["gold"] += 1
            rarity[bucket]["correct"] += triplet in predicted
    return {
        "presence": count_scores(presence["tp"], presence["fp"], presence["fn"]),
        "category_projection": projection(records),
        "errors": dict(errors),
        "by_rarity": {name: {**dict(counts), "recall": counts["correct"] / counts["gold"] if counts["gold"] else None}
                      for name, counts in rarity.items()},
        "by_sentence_scope": {name: {"sentences": len(rows), "null": complete_triplet_scores(
                                  [r["gold_null_triplets"] for r in rows], [r["predicted_null_triplets"] for r in rows], categories),
                                  "category_projection": projection(rows)} for name, rows in groups.items()},
    }


def evaluate(model, dataset, categories, sentiments, category_counts, cfg, device,
             pair_counts, *, threshold=None):
    """If threshold is supplied, never tune it (used for frozen test evaluation)."""
    if not len(dataset):
        raise ValueError("Evaluation split is empty")
    model.eval()
    records, probability_batches, sums, counts = [], [], Counter(), Counter()
    functions = get_evaluation_loss_fns()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=experiment_collate,
                                pin_memory=device.type == "cuda"):
            with autocast_context(cfg, device):
                hidden, bio = model.encode_and_tag(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                category, sentiment = model.classify_spans(hidden, batch["span_boundaries"])
                components = compute_evaluation_loss_components(bio, category, sentiment, batch, functions, device)
                # A threshold above 1 yields explicit records without building
                # millions of Python candidate lists at low NULL thresholds.
                records.extend(decode_predictions(model, hidden, bio, batch, categories, sentiments, 1.01))
                logits = model.classify_null(hidden)
            if logits is None:
                probabilities = torch.zeros((len(batch["input_ids"]), len(categories) * len(sentiments)))
            else:
                probabilities = logits.float().sigmoid().cpu()
                targets = batch["null_targets"].to(device)
                components["null"] = (float(F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="sum")), targets.numel())
            probability_batches.append(probabilities)
            for name, (value, count) in components.items():
                sums[name] += value
                counts[name] += count
    probabilities = torch.cat(probability_batches)
    curve = threshold_curve(records, probabilities, categories, sentiments, cfg.null_thresholds) if threshold is None else []
    if threshold is None:
        threshold = choose_threshold(curve, cfg.selection_metric) if cfg.null_head else 1.0
    apply_threshold(records, probabilities, categories, sentiments, threshold)
    metrics, outcomes = score_records(records, categories, category_counts)
    means = {name: sums[name] / count for name, count in counts.items() if count}
    metrics.update(null_threshold=threshold, null_threshold_curve=curve, unweighted_loss_by_head=means,
                   unweighted_loss=sum(means.values()), selection_metric=cfg.selection_metric,
                   null_diagnostics=null_diagnostics(records, pair_counts, categories))
    return metrics, records, outcomes, probabilities
