"""
S5 (Evaluation): unchanged complete-triplet Micro/Macro-F1 from before, PLUS
the additions :
  - precision_recall_by_label(): so you can see recall, not just F1
  - identify_rare_labels() + rare_label_recall(): the direct test of whether
    class-weighting helps the labels it's meant to help, or just moves errors
  - error_analysis() extended with a `domains` argument for per-domain breakdown
"""
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

Triplet = Tuple[str, str, str]


def decode_bio_to_spans(
    bio_preds: Sequence[int],
    valid_token_mask: Optional[Sequence[bool]] = None,
) -> List[Tuple[int, int]]:
    """Decode BIO tags while excluding special and padding tokens.

    An orphan ``I`` is repaired as the beginning of a span. Returned indices
    always refer to positions in the original token sequence.
    """
    if valid_token_mask is None:
        valid_token_mask = [True] * len(bio_preds)
    if len(bio_preds) != len(valid_token_mask):
        raise ValueError("BIO tags and valid-token mask must have equal lengths")

    spans = []
    start = None
    span_end = None
    for i, (tag, is_valid) in enumerate(zip(bio_preds, valid_token_mask)):
        if tag not in (0, 1, 2):
            raise ValueError(f"Unknown BIO tag id: {tag}")

        if not is_valid or tag == 0:
            if start is not None:
                spans.append((start, span_end))
                start = span_end = None
            continue

        if tag == 1:
            if start is not None:
                spans.append((start, span_end))
            start = i
            span_end = i
        elif tag == 2:
            if start is None:
                start = i
            span_end = i
    if start is not None:
        spans.append((start, span_end))
    return spans


def _validate_parallel(*collections) -> None:
    lengths = {len(collection) for collection in collections}
    if len(lengths) > 1:
        raise ValueError(f"Parallel evaluation inputs have unequal lengths: {sorted(lengths)}")


def _triplet_set(triplets) -> set:
    return {tuple(triplet) for triplet in triplets}


def complete_triplet_scores(gold: List[List[Triplet]], pred: List[List[Triplet]]) -> Dict:
    """Score exact aspect/category/sentiment matches with set semantics.

    ``macro_f1`` is the mean exact-triplet F1 across category labels.
    Duplicate annotations within one example are intentionally deduplicated.
    """
    _validate_parallel(gold, pred)
    tp, fp, fn = 0, 0, 0
    per_class_tp, per_class_fp, per_class_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    for g_list, p_list in zip(gold, pred):
        g_set, p_set = _triplet_set(g_list), _triplet_set(p_list)
        matched = g_set & p_set
        tp += len(matched)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
        for t in matched:
            per_class_tp[t[1]] += 1
        for t in (p_set - g_set):
            per_class_fp[t[1]] += 1
        for t in (g_set - p_set):
            per_class_fn[t[1]] += 1

    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    classes = set(per_class_tp) | set(per_class_fp) | set(per_class_fn)
    class_f1s = []
    for c in classes:
        p = per_class_tp[c] / (per_class_tp[c] + per_class_fp[c]) if (per_class_tp[c] + per_class_fp[c]) else 0.0
        r = per_class_tp[c] / (per_class_tp[c] + per_class_fn[c]) if (per_class_tp[c] + per_class_fn[c]) else 0.0
        class_f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def precision_recall_by_label(gold: List[List[Triplet]], pred: List[List[Triplet]],
                               label_index: int = 1) -> Dict[str, Dict]:
    """Return exact-triplet metrics grouped by category or sentiment label.

    These are not isolated category-head or sentiment-head accuracies: the
    aspect, category, and sentiment must all match for a true positive.
    """
    if label_index not in (1, 2):
        raise ValueError("label_index must be 1 (category) or 2 (sentiment)")
    _validate_parallel(gold, pred)
    tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    for g_list, p_list in zip(gold, pred):
        g_set, p_set = _triplet_set(g_list), _triplet_set(p_list)
        for t in g_set:
            support[t[label_index]] += 1
        for t in (g_set & p_set):
            tp[t[label_index]] += 1
        for t in (p_set - g_set):
            fp[t[label_index]] += 1
        for t in (g_set - p_set):
            fn[t[label_index]] += 1

    labels = set(tp) | set(fp) | set(fn) | set(support)
    out = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[label] = {"precision": p, "recall": r, "f1": f1, "support": support[label]}
    return out


def identify_rare_labels(label_counts: Dict[str, int], bottom_fraction: float = 0.34) -> List[str]:
    """'Rare' = bottom third of labels by training-set frequency, by default.
    TODO (S5): eyeball the actual distribution once you have real counts —
    if there's a natural cliff (e.g. a few labels with <20 examples vs. the
    rest with hundreds), use that cliff instead of a fixed fraction."""
    sorted_labels = sorted(label_counts.items(), key=lambda kv: kv[1])
    cutoff = max(1, int(len(sorted_labels) * bottom_fraction))
    return [label for label, _ in sorted_labels[:cutoff]]


def rare_label_recall(gold: List[List[Triplet]], pred: List[List[Triplet]],
                       rare_labels: List[str], label_index: int = 1) -> float:
    """Exact-triplet recall restricted to gold triplets with rare labels."""
    by_label = precision_recall_by_label(gold, pred, label_index)
    rare_present = [l for l in rare_labels if l in by_label]
    if not rare_present:
        return 0.0
    total_tp = sum(by_label[l]["recall"] * by_label[l]["support"] for l in rare_present)
    total_support = sum(by_label[l]["support"] for l in rare_present)
    return total_tp / total_support if total_support else 0.0


def per_domain_scores(gold: List[List[Triplet]], pred: List[List[Triplet]], domains: List[str]) -> Dict[str, Dict]:
    _validate_parallel(gold, pred, domains)
    by_domain_gold, by_domain_pred = defaultdict(list), defaultdict(list)
    for g, p, d in zip(gold, pred, domains):
        by_domain_gold[d].append(g)
        by_domain_pred[d].append(p)
    return {d: complete_triplet_scores(by_domain_gold[d], by_domain_pred[d]) for d in by_domain_gold}


def error_analysis(gold: List[List[Triplet]], pred: List[List[Triplet]],
                    domains: List[str] = None) -> Dict:
    """Extended per professor feedback: this is now a real breakdown, not a
    wrap-up paragraph. Reports aspect/category/sentiment error counts overall
    AND per domain if domains is provided."""
    _validate_parallel(gold, pred)

    def _counts(g_sub, p_sub):
        counts = {"aspect_missed": 0, "category_wrong": 0, "sentiment_wrong": 0, "extra_predicted": 0}
        for g_list, p_list in zip(g_sub, p_sub):
            g_list = list(_triplet_set(g_list))
            p_list = list(_triplet_set(p_list))
            g_terms = {t[0] for t in g_list}
            p_terms = {t[0] for t in p_list}
            counts["aspect_missed"] += len(g_terms - p_terms)
            counts["extra_predicted"] += len(p_terms - g_terms)
            for term in g_terms & p_terms:
                gold_categories = {t[1] for t in g_list if t[0] == term}
                predicted_categories = {t[1] for t in p_list if t[0] == term}
                gold_sentiments = {t[2] for t in g_list if t[0] == term}
                predicted_sentiments = {t[2] for t in p_list if t[0] == term}
                counts["category_wrong"] += len(
                    gold_categories - predicted_categories
                )
                counts["sentiment_wrong"] += len(
                    gold_sentiments - predicted_sentiments
                )
        return counts

    result = {"overall": _counts(gold, pred)}
    if domains is not None:
        _validate_parallel(gold, pred, domains)
        by_domain_gold, by_domain_pred = defaultdict(list), defaultdict(list)
        for g, p, d in zip(gold, pred, domains):
            by_domain_gold[d].append(g)
            by_domain_pred[d].append(p)
        result["by_domain"] = {d: _counts(by_domain_gold[d], by_domain_pred[d]) for d in by_domain_gold}
    return result
