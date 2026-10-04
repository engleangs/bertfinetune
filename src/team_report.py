"""Readable Markdown reporting for the saved-run diagnostic analysis."""

import math
from pathlib import Path

import pandas as pd

from config import SENTIMENTS


def markdown_table(frame: pd.DataFrame, digits: int = 4):
    """Render small tables without adding the optional tabulate dependency."""
    def value(v):
        if isinstance(v, float):
            return f"{v:.{digits}f}" if math.isfinite(v) else "N/A"
        return str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(value(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)



def write_report(analysis, path: Path):
    """Write the reviewed findings and proposed experiments, with live tables."""
    summary = analysis["summary"]
    in_summary = summary[summary.setting == "In-domain"]
    columns = ["model", "n", "precision_mean", "recall_mean", "f1_mean", "f1_std"]
    in_table = markdown_table(in_summary[columns].rename(columns={"model": "Loss", "n": "Seeds", "precision_mean": "Precision", "recall_mean": "Recall", "f1_mean": "F1 mean", "f1_std": "F1 SD"}))
    run_frame = analysis["runs"]
    error_table = markdown_table(run_frame[run_frame.setting == "In-domain"].groupby("model")[["correct", "term", "category", "sentiment", "spurious_aspects"]].mean().reset_index(), digits=1)
    transfer = summary[summary.setting == "LODO"]
    transfer_table = markdown_table(transfer[["held_out_domain", "model", "n", "f1_mean", "aspect_f1_mean", "aspect_sentiment_f1_mean", "category_coverage_mean"]].rename(columns={"held_out_domain": "Domain", "model": "Loss", "n": "Seeds", "f1_mean": "Triplet F1", "aspect_f1_mean": "Term F1", "aspect_sentiment_f1_mean": "Term+sentiment F1", "category_coverage_mean": "Coverage"}))
    nulls = analysis["null_prevalence"]
    null_table = markdown_table(nulls[nulls.split == "test"][["domain", "unique_triplets", "null_triplets", "null_percentage", "null_only_examples", "mixed_examples", "scope_recall_upper_bound"]], digits=3)
    sentiments = analysis["sentiment_counts"]
    sentiment_table = sentiments[sentiments.scope == "retained"].groupby(["split", "sentiment"])["count"].sum().unstack(fill_value=0).reindex(columns=SENTIMENTS).reset_index()
    sentiment_text = markdown_table(sentiment_table, digits=0)
    rarity = analysis["rarity"]
    rarity = rarity[(rarity.setting == "In-domain") & (rarity.rarity.isin(["rare", "unseen"]))].groupby(["model", "rarity"])[["gold_triplets", "correct", "term", "category", "sentiment"]].mean().reset_index()
    rarity["recall"] = rarity.correct / rarity.gold_triplets
    rarity_table = markdown_table(rarity)
    domain = analysis["domain_metrics"]
    domain_table = domain[domain.setting == "In-domain"].groupby(["domain", "model"]).f1.mean().unstack().sort_values("standard").reset_index()
    domain_text = markdown_table(domain_table)
    conditional = analysis["conditional"]
    conditional = conditional[conditional.setting == "In-domain"].groupby("model")[["category_on_matched_term_accuracy", "sentiment_on_matched_term_accuracy", "sentiment_on_matched_term_and_category_accuracy"]].mean().reset_index()
    conditional_text = markdown_table(conditional.rename(columns={"category_on_matched_term_accuracy": "Category given term", "sentiment_on_matched_term_accuracy": "Sentiment given term", "sentiment_on_matched_term_and_category_accuracy": "Sentiment given term+category"}))
    taxonomy = analysis["taxonomy_summary"]
    primary_shares = taxonomy[taxonomy.setting == "In-domain"][["model", "dominant_primary_error", "term_error_share", "category_error_share", "sentiment_error_share"]].copy()
    for name in ["term", "category", "sentiment"]:
        primary_shares[f"{name}_error_share"] *= 100
    taxonomy_text = markdown_table(primary_shares.rename(columns={"model": "Loss", "dominant_primary_error": "Largest primary error", "term_error_share": "Term %", "category_error_share": "Category %", "sentiment_error_share": "Sentiment %"}), digits=2)
    text = f"""# Team notebook review and detailed error analysis

Review date: 2026-10-04, Pacific/Auckland. Dataset: English M-ABSA.
Reviewed notebook: `Initial_Evaluation_and_Error_Analysis_CS_760.ipynb`.
Results are descriptive analyses of inspected version-1 test predictions.
Current reporting rules and staged training requirements:
[protocol amendment 1.1](protocol-amendment-2026-10-04.md).

## Main findings

The team's in-domain calculations are correct: weighting increases recall
but reduces precision enough to lower exact-triplet F1. Cross-domain zeros
often reflect unavailable categories, while term and sentiment predictions
still carry useful information. NULL targets are structurally unsupported.
These three findings need separate explanations and experiments.

The current repository contains {len(run_frame)} validated completed runs:
10 in-domain and 70 LODO runs, with five matched seeds for each condition.
Missing planned keys: {len(analysis['metadata']['missing_run_keys'])}.
Restaurant runs come from the original cross-domain rows in `results.csv`;
the other six folds come from `results_lodo.csv`. They are included once.
Saved metrics were checked against predictions and indexed results; test
sentences, gold scope, training vocabulary and counts were checked against
the local data. This checks consistency, not historical data immutability.
Current data hashes are saved in `artifacts/team_analysis/metadata.json`.

## What was correct and what needed correction

| Original notebook part | Review | Correction |
|---|---|---|
| Exact-triplet precision, recall and F1 | Correct for supplied artifacts | Recompute and verify against metrics and CSV indexes |
| In-domain means, sample SD and seed differences | Correct for its five matched seeds | Reject duplicate or unmatched run keys explicitly |
| LODO comparison | Correct only for the supplied 12 seed-13 runs | Include all seven folds and pair within domain AND seed |
| Pivot on domain alone | Fails with multiple seeds | Summarize seeds before domain tables; pair seed-level rows |
| Coverage table | Correct in the one-seed notebook | Derive per run; avoid repeated domain rows and many-to-many joins |
| Existing error counts | Useful preliminary overlapping counts | Assign one outcome per gold triplet and retain independent flags |
| Category/sentiment label F1 | Exact-triplet F1 grouped by a label | Add component F1 and conditional component accuracy |
| Colab paths and ZIP extraction | Specific to the old uploaded archives | Use local repository data/results and one reusable analyzer |
| Titles saying five seeds | Hard-coded assumption | Display actual counts and completeness status |
| NULL, rare and sentiment imbalance analyses | Missing or incomplete | Add training-only frequency bands and raw-data audits |
| Sample successes/failures | Valid examples, not representative statistics | Export all gold diagnostic rows; inspect examples after counts |

The original notebook is preserved at
`artifacts/team_analysis/original_notebook.ipynb`. The cleaned notebook keeps
the main results, removes repeated inspection cells and stale outputs, and
uses `analyze_team_notebook.py` to load runs, `src/team_diagnostics.py` for
metrics and taxonomy, and `src/team_report.py` for report formatting.
It requires this repository and its local artifacts/data. It is not a
standalone Colab archive notebook.

## In-domain result

All F1 values use a 0-1 scale; 0.02 means two percentage points.
Tables average per-seed scores. SD is the sample SD across seeds.

{in_table}

The weighted model's F1 is lower by about 0.0885, or 8.85 percentage points.
It produces more detected gold aspects and many more spurious aspects.
This is an observed precision/recall tradeoff; an ablation is needed to
identify which weighted head produces it.

## One primary error per retained gold triplet

Apply this fixed order within each sentence:

1. **Correct:** the full triplet exists in predictions.
2. **Term error:** no prediction has the exact gold term surface text.
3. **Category error:** the term exists, but no prediction on it has the gold category.
4. **Sentiment error:** the term and category exist together, but sentiment differs.

This is a first-failed-component label, not proof of a sole causal explanation.
If category and sentiment are both wrong, the primary label is category;
the independent flags preserve both failures. Correct plus the three error
counts partitions retained gold support. Spurious predictions and duplicates
are prediction-side counts and do not belong in that partition.

Mean counts across five in-domain seeds:

{error_table}

The old overlapping sentiment counts were 112.8 for standard and 360.4 for
weighted. The exclusive sentiment counts are 76.8 and 189.4. Neither should
be called the total number of sentiment-head mistakes: the exclusive count
is conditional on the term and category already matching.

Mean conditional membership accuracies:

{conditional_text}

These accuracies exclude missing terms. Multiple predictions on one term can
satisfy different component flags. The confusion matrix selects one candidate
deterministically: exact triplet first, then matching term+category, then a
matching term; ties use lexicographic order. Its denominator and missing-term
count are saved in each run's diagnostics.

## Which component is hardest, and should F1 be split?

The largest primary error group answers where complete extraction first fails.
Shares below divide by all incorrect retained gold, pooling outcomes across
seeds. They are descriptive; repeated test gold across seeds is not independent
evidence for a statistical test.

{taxonomy_text}

Term errors dominate standard loss (about 80.75% of its gold errors).
Category errors dominate all-head weighting (about 50.00%). The weighting
experiment detects more terms, exposing more category failures. This does
not establish that the category head caused the change. An isolated loss-head
ablation is needed for that claim.

Use three complementary views: full-triplet F1 for complete task performance,
component F1 for successful partial extraction, and conditional error rates
for labels among detected terms. Splitting F1 is useful for diagnosis; it is
not a better replacement for the exact-triplet endpoint. A small exclusive
sentiment error share does not mean sentiment is always easy, because failures
behind missed terms or wrong categories are assigned earlier in the hierarchy.
The independent flags and conditional confusion retain that information.
`taxonomy_summary.csv` reports primary shares and conditional failure rates
for every condition/domain/loss pair with their different denominators.

## Evaluation beyond full-triplet matching

Keep exact-triplet micro-F1 as the primary task score. Add these secondary
views so category mismatch does not hide successful term/sentiment transfer:

| View | Required match | Interpretation |
|---|---|---|
| Term F1 | Term surface text | Aspect extraction |
| Term+category F1 | Term and category | Extraction plus category |
| Term+sentiment F1 | Term and sentiment | Category-independent transfer |
| Full-triplet F1 | All three fields | Complete task performance |
| Source-known-category triplet F1 | Full triplet, restricted gold categories | Closed-set performance; always show coverage |
| Conditional accuracy/confusion | Category or sentiment on a matched gold term | Diagnosis among detected terms; show denominator |

For example, predicting `(screen, DEVICE, positive)` for gold
`(screen, DISPLAY, positive)` earns credit under term F1 and term+sentiment
F1, but fails full-triplet F1. These projected metrics deduplicate after
projection and are distinct tasks. Their differences cannot be interpreted
as an additive decomposition of full-triplet errors.

Five-seed means; coverage is a proportion of retained gold triplets:

{transfer_table}

Coursera's standard term+sentiment F1 is about 0.531 despite zero full F1.
Laptop, phone and sight have zero source-category coverage, forcing full
triplet recall and F1 to zero for this classifier. Coursera covers only
1/409 retained gold triplets. Unseen category names must remain false
negatives in the full score. Category-free metrics reveal transfer; they
do not make unseen category prediction possible. Known-category F1 is
reported as N/A when there are no known-category gold targets.

Saved artifacts have surface strings, not predicted character offsets.
Consequently these are exact surface-text metrics, not validated boundary
or occurrence metrics. Repeated identical terms cannot be distinguished.
Future runs need saved character/token offsets for boundary F1 or overlap
analysis; do not label substring matching as exact extraction.

## Rare versus unseen labels

Freeze category bands from each fold's source-training retained annotations:
unseen=0, rare=1-5, low=6-20, medium=21-100, frequent>100.
Unseen labels are not rare labels. Test gold support is shown so a high rate
from very few examples is not mistaken for stable performance.

In-domain mean counts and exact-triplet recall:

{rarity_table}

Rare-category recall changes from 0 to about 2.84%, while overall F1 declines.
The in-domain test has 14 retained unseen-category triplets, even though all
domains appear in training. Every cross-domain fold uses its own training
counts; the common in-domain rare set must not be reused across folds.
`rarity.csv` and `gold_error_records.jsonl` include all frequency bands.

## NULL prevalence and representational limit

Count exact duplicates once within each sentence. Keep raw annotation counts
separately. NULL is an implicit target with a category and sentiment, not an
unlabelled example and not automatically a term error in the explicit task.

{null_table}

NULL percentages use all deduplicated gold annotations in the domain's test
split. Food has the highest test NULL prevalence, about 54.74%.
`scope_recall_upper_bound` is retained explicit annotations divided by all
deduplicated annotations. It is an annotation-scope bound; token truncation,
ambiguous occurrences and unseen categories can reduce achievable recall
further. The primary explicit-only F1 remains unchanged by this raw-data audit.
Training, development and test counts, sentence types, and NULL
category/sentiment distributions are exported separately.

A full-annotation evaluation including NULL would count each NULL gold
triplet as a false negative for this architecture. This notebook exports
prevalence and scope bounds; it does not claim to have trained a NULL model
or computed a new full-annotation benchmark. A sentence-level multi-label
NULL `(category, sentiment)` head is a separately amended extension.

## Sentiment imbalance and whether to remove Conflict

Retained explicit targets, summed over all seven domains:

{sentiment_text}

Training targets are approximately 69.93% positive, 21.64% negative,
8.41% neutral and 0.012% Conflict. There are five raw/deduplicated Conflict
training annotations: two Coursera, one hotel and two food; three are NULL.
Only one survives the current explicit selection rules. Development and
test have no Conflict gold annotations. The earlier configuration comment
claiming Conflict occurred only in restaurant has been corrected using this audit.

**Recommendation:** preserve the four-class original baseline. A three-class
task is a reasonable separately named sensitivity experiment, because this
corpus cannot estimate Conflict generalization from zero dev/test support.
If adopted, filter Conflict annotations before rebuilding vocabularies,
frequency weights and datasets. Do not map Conflict to neutral, remove whole
mixed-label sentences, or simply delete a label from `config.SENTIMENTS`:
the actual vocabulary is built dynamically from training annotations.

Under pre-tokenization counts the nominal inverse-frequency Conflict weight
would be 8335/(4*1)=2083.75. Actual training weights use tokenized targets;
PyTorch weighted mean CE divides by the sum of target weights in a batch,
so this is not an assertion of a 2083.75-fold gradient amplification.
The weighted in-domain models predict Conflict only twice in total across
five test runs. Removing those predictions from the old output would not
explain the large observed precision loss. Retraining a three-class model
could still change behavior, which is why it needs its own experiment.
([PyTorch CE reduction](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html))

## Which domain is hardest?

In-domain exact-triplet mean F1, ordered by standard loss:

{domain_text}

Food is lowest under both losses in this condition. For LODO full-triplet F1,
Coursera, laptop, phone and sight tie at zero, so a unique hardest domain
cannot be identified from that score. Use coverage-aware component rankings:
phone has the lowest standard term F1; food has the lowest weighted term F1.
Among folds with nonzero category coverage, food has the lowest full-triplet
F1 under both losses. These are descriptive rankings without a demonstrated
significant difference between every domain pair. High NULL prevalence,
category support, language length and other associations are explanations
to investigate, not demonstrated causes of these rankings.

## Professor's feedback: a small, controlled loss sweep

The historical trainer supports standard CE or all-head inverse-frequency CE.
The separate `run_experiments.py` runner now implements mixed CE, focal loss,
head ablations and an optional NULL head. See
[the 5 October experiment guide](experiment-guide-2026-10-05.md) for flags
and the two-seed development pilot. This report's tables remain historical;
they do not establish an F1 improvement from the new configurations.

For head h, freeze a normalized mixture:

    L_h = (1-alpha) * CE_h + alpha * weighted_CE_h
    L_total = L_BIO + L_category + L_sentiment

Normalizing by w1+w2 keeps the overall head multiplier at one:
`w1=1, w2=0.5` becomes alpha=1/3; `w1=0.5, w2=0.5` becomes alpha=1/2.
State this normalization explicitly: it changes the scale of the professor's
literal unnormalized formula. Use the existing mean reduction consistently
for CE and weighted CE; they have different denominators, which must remain
fixed across the sweep. Do not confuse mixture coefficients with class weights
or the outer BIO/category/sentiment loss multipliers.

Initial development-only candidate budget (eight settings, seed 13):

| Family | Settings | Heads |
|---|---|---|
| CE / weighted CE mixture | alpha = 0, 0.25, 1/3, 0.5, 0.75, 1 | Same alpha for all three heads |
| Unweighted focal | gamma = 1, 2 | All three heads; outer multipliers unchanged |

Focal loss downweights easy examples using `(1-p_t)^gamma`; the original
paper introduced it for dense object detection. Whether it helps this TASD
model is an empirical question. Calculate p_t from unweighted log-softmax
probabilities, not `exp(-weighted_CE)`, and ignore -100 targets. Gamma=0 with
no class weighting should reduce to ordinary CE. Start without an additional
inverse-frequency focal factor so two imbalance mechanisms are not mixed
in one unexplained setting.
([Lin et al., Focal Loss](https://arxiv.org/abs/1708.02002))

Use source development exact-triplet F1 to select a shortlist. Inspect
precision, recall, rare recall and the exclusive taxonomy alongside it.
Then isolate the strongest weighted/focal variant to BIO, category and
sentiment heads in a second ablation before making a causal head claim.
Run selected designs with all five matched seeds. Do not tune on the saved
test examples reviewed here. Re-evaluation on these inspected tests must be
reported as exploratory; a fresh confirmation needs untouched evaluation.

The separate runner implements the corrections recorded in
`finetuning-readiness-2026-10-04.md`: warmup, clipping, eligibility rules,
fixed macro-F1, reproducible environment and distinct experiment versions.
Validate the small pilot before expanding shortlisted configurations.
Do not reuse the old result keys for corrected or new loss configurations.

## Reproduction and outputs

Run from the repository with its virtual environment:

```powershell
.venv/Scripts/python.exe analyze_team_notebook.py
```

Open the cleaned notebook and run cells in order for the presentation tables
and charts. No model download, GPU training or ZIP extraction is needed.
Reproducible analysis exports are in `artifacts/team_analysis/`:

- `runs.csv`, `summary.csv`, `paired.csv`: full/component scores and seed comparisons.
- `gold_error_records.jsonl`: every retained unique gold triplet with primary
  outcome, independent flags, frequency and selected prediction.
- `domain_metrics.csv`, `rarity.csv`, `conditional.csv`: domain and label diagnostics.
- `taxonomy_summary.csv`: largest primary error, mean counts, primary error
  shares, and conditional failure rates with empty denominators reported as N/A.
- `sentiment_confusion.csv`: conditional sentiment counts by run.
- `sentiment_counts.csv`, `null_prevalence.csv`, `null_labels.csv`: raw-data audits.
- `run_diagnostics.json`, `metadata.json`: denominators, duplicate/spurious
  counts, coverage, notes and current data hashes.

This historical report uses the original metrics, predictions and eligibility.
New training lives in separate experiment modules and versioned artifacts;
its optional NULL scores must not be mixed into these explicit-only tables.
"""
    Path(path).write_text(text, encoding="utf-8")

