"""
S5 (Evaluation): unchanged complete-triplet Micro/Macro-F1 from before, PLUS
the additions :
  - precision_recall_by_label(): so you can see recall, not just F1
  - identify_rare_labels() + rare_label_recall(): the direct test of whether
    class-weighting helps the labels it's meant to help, or just moves errors
  - error_analysis() extended with a `domains` argument for per-domain breakdown
"""
from collections import defaultdict
from typing import Dict, List, Tuple

Triplet = Tuple[str, str, str]


def decode_bio_to_spans(bio_preds: List[int]) -> List[Tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(bio_preds):
        if tag == 1:
            if start is not None:
                spans.append((start, i - 1))
            start = i
        elif tag == 2:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, len(bio_preds) - 1))
    return spans


def complete_triplet_scores(gold: List[List[Triplet]], pred: List[List[Triplet]]) -> Dict:
    tp, fp, fn = 0, 0, 0
    per_class_tp, per_class_fp, per_class_fn = defaultdict(int), defaultdict(int), defaultdict(int)

    for g_list, p_list in zip(gold, pred):
        g_set, p_set = set(g_list), set(p_list)
        matched = g_set & p_set
        tp += len(matched)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
        for t in matched:
            per_class_tp[t[2]] += 1
        for t in (p_set - g_set):
            per_class_fp[t[2]] += 1
        for t in (g_set - p_set):
            per_class_fn[t[2]] += 1

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

    return {"micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1, "macro_f1": macro_f1}


def precision_recall_by_label(gold: List[List[Triplet]], pred: List[List[Triplet]],
                               label_index: int = 1) -> Dict[str, Dict]:
    """label_index: 1 = category, 2 = sentiment. Returns per-label precision,
    recall, F1, and support (gold count) — support is what lets you tell
    'rare' from 'common' downstream."""
    tp, fp, fn, support = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)

    for g_list, p_list in zip(gold, pred):
        g_set, p_set = set(g_list), set(p_list)
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
    """The direct answer to the professor's original worry: does the recall
    of JUST the rare labels go up under weighted training, or not."""
    by_label = precision_recall_by_label(gold, pred, label_index)
    rare_present = [l for l in rare_labels if l in by_label]
    if not rare_present:
        return 0.0
    total_tp = sum(by_label[l]["recall"] * by_label[l]["support"] for l in rare_present)
    total_support = sum(by_label[l]["support"] for l in rare_present)
    return total_tp / total_support if total_support else 0.0


def per_domain_scores(gold: List[List[Triplet]], pred: List[List[Triplet]], domains: List[str]) -> Dict[str, Dict]:
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
    def _counts(g_sub, p_sub):
        counts = {"aspect_missed": 0, "category_wrong": 0, "sentiment_wrong": 0, "extra_predicted": 0}
        for g_list, p_list in zip(g_sub, p_sub):
            g_terms = {t[0] for t in g_list}
            p_terms = {t[0] for t in p_list}
            counts["aspect_missed"] += len(g_terms - p_terms)
            counts["extra_predicted"] += len(p_terms - g_terms)
            for term in g_terms & p_terms:
                g_t = next(t for t in g_list if t[0] == term)
                p_t = next(t for t in p_list if t[0] == term)
                if g_t[1] != p_t[1]:
                    counts["category_wrong"] += 1
                if g_t[2] != p_t[2]:
                    counts["sentiment_wrong"] += 1
        return counts

    result = {"overall": _counts(gold, pred)}
    if domains is not None:
        by_domain_gold, by_domain_pred = defaultdict(list), defaultdict(list)
        for g, p, d in zip(gold, pred, domains):
            by_domain_gold[d].append(g)
            by_domain_pred[d].append(p)
        result["by_domain"] = {d: _counts(by_domain_gold[d], by_domain_pred[d]) for d in by_domain_gold}
    return result
