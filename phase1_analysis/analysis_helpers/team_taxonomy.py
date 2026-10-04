"""Compact summaries of the first failed triplet component.

Keep primary error shares separate from conditional component failure rates.
Neither statistic establishes the cause of an error or isolated head accuracy.
"""

import pandas as pd


ERRORS = ("term", "category", "sentiment")
GROUPS = ("setting", "held_out_domain", "model")


def summarize_taxonomy(runs: pd.DataFrame) -> pd.DataFrame:
    """Average counts across seeds and calculate rates from pooled outcomes.

    Primary shares divide by all incorrect gold triplets. Term failure divides
    by all gold; category failure by gold with a predicted term; sentiment
    failure by gold with a predicted term and category. Empty denominators are
    undefined (NaN). Counts must partition gold exactly, including correct.
    """
    if runs.duplicated([*GROUPS, "seed"]).any():
        raise ValueError("Duplicate taxonomy run keys")
    counts = runs[["correct", *ERRORS]]
    if (counts < 0).any().any() or not (counts.sum(axis=1) == runs.gold_triplets).all():
        raise ValueError("Primary outcomes must partition retained gold triplets")

    rows = []
    for keys, group in runs.groupby(list(GROUPS), dropna=False):
        totals = group[["gold_triplets", "correct", *ERRORS]].sum()
        failures = sum(totals[name] for name in ERRORS)
        largest = max(totals[name] for name in ERRORS)

        def rate(numerator, denominator):
            return numerator / denominator if denominator else float("nan")

        row = dict(zip(GROUPS, keys))
        row.update(
            n=len(group),
            dominant_primary_error=" / ".join(name for name in ERRORS if totals[name] == largest) if largest else "none",
            gold_triplets_mean=group.gold_triplets.mean(),
            correct_mean=group.correct.mean(),
            gold_error_rate=rate(failures, totals.gold_triplets),
            term_failure_rate=rate(totals.term, totals.gold_triplets),
            category_failure_given_term=rate(totals.category, totals.gold_triplets - totals.term),
            sentiment_failure_given_term_category=rate(totals.sentiment, totals.correct + totals.sentiment),
        )
        for name in ERRORS:
            row[f"{name}_mean"] = group[name].mean()
            row[f"{name}_error_share"] = rate(totals[name], failures)
        rows.append(row)
    return pd.DataFrame(rows)
