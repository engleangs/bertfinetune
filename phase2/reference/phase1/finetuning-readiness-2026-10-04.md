# Fine-tuning readiness review

Reviewed: 2026-10-04 (Pacific/Auckland).
Status: proposed continuation plan; not a new protocol freeze or completed training run.

Implementation update, 2026-10-05: use
[the experiment guide](experiment-guide-2026-10-05.md) for the corrected
runner, optional NULL head and first-seed pilot. The findings and proposed
sequence below are retained as the 4 October review.

Current reporting rules and next-stage sequence are recorded in
[protocol amendment 1.1](protocol-amendment-2026-10-04.md). That amendment
incorporates the professor's mixed-CE and focal-loss suggestions; the earlier
candidate ordering below is preserved as a readiness-review note.

## What the saved results show

Five-seed means from results.csv:

| In-domain configuration | Precision | Recall | Exact-triplet micro-F1 |
|---|---:|---:|---:|
| Standard | 0.3899 | 0.3210 | 0.3520 |
| All-head inverse-frequency weighting | 0.1909 | 0.4255 | 0.2635 |

Weighting improves recall but loses enough precision to reduce F1 by 0.0885.
This establishes the observed tradeoff, not which head causes it. Isolate heads
before attributing the effect to BIO, category, or sentiment weighting.

Source-category coverage from standard seed-13 held-out manifests:

| Domain | Covered retained gold triplets | Coverage |
|---|---:|---:|
| coursera | 1 / 409 | 0.24% |
| food | 69 / 249 | 27.71% |
| hotel | 207 / 543 | 38.12% |
| laptop | 0 / 525 | 0% |
| phone | 0 / 1179 | 0% |
| sight | 0 / 696 | 0% |

Zero coverage forces zero exact-triplet recall and F1 for a classifier whose
outputs are restricted to source-training categories. These results cannot be
fixed by extra epochs or class weighting. Label strings differ across domain
ontologies; do not invent mappings from inspected test results. Category-free
aspect and aspect-plus-sentiment metrics can still measure transfer.

## Protocol decisions that must be locked

1. **Task and annotation eligibility:** explicit aspects only; unique character
   alignment; exact duplicate collapse; deterministic overlap exclusion;
   exclude all spans with multiple distinct category/sentiment pairs. Keep
   counts of NULL, ambiguous, conflicting, overlapping, and unaligned targets.
   Current code chooses the first occurrence and first label pair, contrary
   to the eligibility rules in protocol.md.
2. **Splits and ontology:** official train/dev/test files; source-train-only
   vocabularies and frequency counts; held-out train/dev unused for tuning.
   Freeze data hashes, domain list, normalization, and tokenizer revision.
3. **Metrics:** exact sentence-level triplet sets; micro-F1 primary, precision
   and recall alongside it. Freeze category macro-F1 over the source-training
   universe with zero-denominator classes scored zero. Unseen gold categories
   remain false negatives in full micro-F1 and are separately reported.
   Current macro-F1 uses the gold/prediction union, which varies by run.
   Explicitly distinguish surface-text matching from character-offset matching.
4. **Training:** BERT-base-uncased, full fine-tuning, length 128, batch 16,
   AdamW, LR 2e-5, weight decay 0.01, five epochs, linear schedule,
   10% warmup (freeze rounding rule), gradient norm clipping 1.0, equal head
   multipliers. Current training has zero warmup and no clipping.
5. **Comparison:** retain standard vs all-head inverse-frequency CE as the
   original comparison. Freeze N/(K*count) and absent-class behavior. New
   head-specific weighting variants are exploratory amendments.
6. **Selection and replication:** dev exact-triplet micro-F1; ties by lower
   unweighted dev loss, then earlier epoch; matched seeds
   13, 42, 123, 2024, 777. Never select using test F1.
7. **Analysis:** preserve the 0.02 absolute minimum effect, paired seed
   differences, uncertainty, and primary/secondary distinction. Freeze rare
   categories at training count 1-5; unseen categories separate.
8. **Provenance and execution gates:** dated amendment, exact package lock,
   full environment inventory, code commit and dirty-state record, model
   revision, data hashes, standalone audits for all intended folds, passing
   tests and pip check, distinct versioned results/artifact paths. Current
   requirements are unpinned and manifests record only limited versions.

Existing test results have been inspected. Preserve them as version-1 evidence;
do not describe later improvements on those same test files as an untouched
confirmatory test. Report subsequent experiments transparently as exploratory,
or use a genuinely untouched evaluation source for a fresh confirmation.

## Order for continuing fine-tuning

1. Implement and verify protocol corrections together; preserve all old runs.
   Run a tiny end-to-end check into a separate v2 artifact directory.
2. Start with corrected in-domain development experiments. Compare standard,
   all-head weighting, and category-only weighting on seed 13 using identical
   training settings. Save per-epoch precision, recall, F1, span F1, category
   accuracy given a correct span, sentiment accuracy given a correct span,
   rare-label recall, and counts of spurious/missed aspects.
3. Evaluate this ablation on development data only. Category-only weighting is
   a diagnostic candidate, not an established F1 improvement. Separate BIO-only
   and sentiment-only weighting if needed to identify the tradeoff.
4. If development learning curves still improve at epoch five, register an
   exploratory budget comparison of 5 vs 10 epochs, crossed with LR 1e-5 vs
   2e-5. Apply the same candidate budget to compared losses, choose by dev
   micro-F1, and document the search before running it. Do not silently alter
   the frozen five-epoch core experiment.
5. Expand the selected development design to all five matched seeds, then
   evaluate the frozen checkpoints and report the exploratory status.
6. Continue seven-domain transfer with coverage-aware diagnostics. A model
   that can score unseen category descriptions or generate categories is a
   separate architectural extension requiring its own protocol; preserve the
   original closed-set results for comparison.

Do not launch another full GPU matrix until the implementation matches the
amended protocol. Corrections improve validity; an F1 increase is not guaranteed.
