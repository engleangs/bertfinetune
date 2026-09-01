"""
the "basic rigour"  — turns "0.2 point wobble"
into an actual defensible claim.
"""
from typing import List, Dict

import numpy as np
from scipy import stats


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("values must contain at least one finite score")
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": len(arr), "min": float(arr.min()), "max": float(arr.max())}


def paired_significance_test(standard_scores: List[float], weighted_scores: List[float]) -> Dict:
    """Paired t-test across matched seeds (seed i's standard run vs seed i's
    weighted run — same seed means same data shuffling/init noise source,
    which is why paired is the right test here, not independent-samples).
    Falls back to a paired bootstrap if you'd rather not assume normality —
    swap the TODO below  ifprefers that."""
    if len(standard_scores) != len(weighted_scores):
        raise ValueError("need one score per matched seed for both configs")
    if len(standard_scores) < 2:
        raise ValueError("paired significance testing requires at least two seeds")
    standard_scores = np.asarray(standard_scores, dtype=float)
    weighted_scores = np.asarray(weighted_scores, dtype=float)
    if not np.isfinite(standard_scores).all() or not np.isfinite(weighted_scores).all():
        raise ValueError("paired scores must be finite")
    diffs = weighted_scores - standard_scores
    t_stat, p_value = stats.ttest_rel(weighted_scores, standard_scores)
    return {
        "mean_diff": float(diffs.mean()),
        "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def bootstrap_ci(standard_scores: List[float], weighted_scores: List[float],
                  n_boot: int = 10000, seed: int = 0) -> Dict:
    """Alternative/companion to the t-test — doesn't assume normality, useful
    with only 4-5 seeds where a t-test's normality assumption is shaky.
    Returns a 95% CI on the paired difference; if it excludes 0, that's your
    significance evidence."""
    if len(standard_scores) != len(weighted_scores) or not standard_scores:
        raise ValueError("bootstrap requires non-empty matched score lists")
    rng = np.random.default_rng(seed)
    diffs = np.asarray(weighted_scores, dtype=float) - np.asarray(standard_scores, dtype=float)
    if not np.isfinite(diffs).all():
        raise ValueError("paired score differences must be finite")
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return {"ci_low": float(lower), "ci_high": float(upper), "excludes_zero": bool(lower > 0 or upper < 0)}


def passes_min_effect(mean_diff: float, min_effect_size: float) -> bool:
    """The pre-registered bar from config.MIN_EFFECT_SIZE. A statistically
    significant but tiny difference (e.g. p=0.03, diff=0.003) still fails
    this — significance and 'big enough to matter' are different questions,
    and the professor's feedback was specifically about not confusing them."""
    return mean_diff >= min_effect_size
