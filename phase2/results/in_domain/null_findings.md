# Findings after NULL evaluation — 7 October 2026

The overnight workflow completed: **25/25 finalist test rows and 21/21 screening test rows**. Some screening checkpoints also appear among the finalists; these are not 46 independent models. The five-seed confirmation uses seeds 13, 42, 123, 2024 and 777.

**The new NULL adaptation did not demonstrate a combined-F1 benefit against its matching NULL-off control.** Its development-selected operating point suppresses every NULL prediction. The successful result for the presentation remains the established explicit fine-tuning improvement.

## Frozen test results

Scores below are percentages, averaged over five seeds. Each seed's score is micro F1. Combined F1 includes eligible explicit triples and NULL category/sentiment pairs; it is not the average of explicit and NULL F1.

| Study / arm | Explicit F1 | NULL F1 | Combined F1 |
|---|---:|---:|---:|
| Established quick tuning: selected NULL off | 40.28 | 0.00 | 35.20 |
| Historical quick tuning: BCE/CLS NULL on | 40.59 | 8.89 | 33.44 |
| Historical quick tuning: matching vocabulary control | 40.74 | 0.00 | 35.60 |
| New adaptation: focal + IA attention | 40.52 | 0.00 | 35.43 |
| New adaptation: matching NULL-off control | 40.55 | 0.00 | 35.43 |

The new matched combined-F1 difference is **+0.002 percentage points** on average: effectively tied. Three seeds favor the control and two favor the adaptation. This does not establish a benefit from the auxiliary head; no significance claim is made.

The historical NULL head does recover some implicit pairs, but its mean precision is only **9.50%** and recall **8.95%**. It lowers combined F1 by **2.15 percentage points** relative to its matching vocabulary control. The historical and new studies differ in input budget, numerical precision and selection objective, so their differences are descriptive comparisons.

## Why the new NULL F1 is zero

All five selected attention models chose **threshold 1.0**, the protocol's explicit abstention option, using combined development F1. Therefore each emits zero NULL predictions on test: **0 true positives, 0 false positives and 1,300 false negatives**. This is the saved evaluation outcome, rather than evidence that the overnight jobs stopped early.

The label "Best NULL on" in the generated report means the enabled-head candidate selected by **combined development F1**. It does not mean the strongest model for NULL-only extraction.

The test misses include **1,091 frequent**, **176 rare** and **33 unseen** NULL pairs under the training-count definitions. Thus the observed failure extends beyond unseen labels. Presence and category-only NULL projections also score zero because the chosen operating point emits nothing.

## What the development curves reveal

The following table reads the **saved development threshold grid at each method's combined-dev-selected checkpoint**. It reports the threshold with the highest NULL F1 as a diagnostic. These alternative thresholds were not applied to test and do not replace the frozen results above.

All methods in this table use seed 13 and a three-epoch screening budget.

| NULL method | Best NULL dev F1 in saved grid (%) | Threshold | NULL precision (%) | NULL recall (%) |
|---|---:|---:|---:|---:|
| BCE + CLS | 6.50 | 0.18 | 4.09 | 15.83 |
| Focal + CLS | 8.87 | 0.38 | 5.86 | 18.26 |
| ASL + CLS | 3.54 | 0.38 | 1.96 | 18.78 |
| Focal + dedicated IA token | 8.59 | 0.38 | 5.60 | 18.43 |
| Focal + IA attention | 0.21 | 0.14 | 0.10 | 36.87 |
| IA attention + hard negatives | 0.50 | 0.80 | 0.33 | 1.04 |
| IA attention + random negatives | 0.75 | 0.80 | 0.38 | 16.35 |

**Focal with CLS shows a promising NULL-specific development signal relative to BCE in this screen.** Its precision remains low: at that diagnostic threshold, combined development F1 is 24.63%, compared with 28.83% when abstaining. A nonzero NULL score therefore does not imply improved combined extraction.

The attention variant's slight advantage in abstaining combined development F1 did not translate into stronger implicit predictions. In its five-epoch confirmation checkpoints, even the best NULL F1 in each saved development grid ranges from **0.217% to 0.295%**, averaging **0.254%**. Lowering the threshold alone does not resolve its weak precision. For example, seed 13 at threshold 0.14 produces 212 correct NULL pairs but 194,239 false positives on development data.

The tested ASL setting, hard mining and random sampling did not demonstrate an improvement in NULL extraction over focal/CLS. These are short, single-seed screens, not general conclusions about the techniques. All five attention confirmation histories record zero skipped AMP updates, so numerical update skipping does not explain this result.

## Sparsity and the remaining research problem

The training audit records **2,395 positive NULL targets** across **1,040 category/sentiment outputs**; only **201 outputs** have a positive NULL training example. This is severe sparse supervision. It supports investigating label coverage and representation quality, but does not by itself identify the cause of the attention failure.

For explicit extraction, the selected new model's first-failed-component taxonomy attributes about **58.5% of missed gold triples to term extraction**, followed by category and sentiment. These are ordered error assignments, not proof of a sole cause. Food remains the hardest evaluated in-domain domain, at approximately **23.99% explicit F1**. These results do not establish a new cross-domain/LODO improvement.

## What to present and use now

Use the established development-selected explicit model as the main fine-tuning result: **0.75 standard CE + 0.25 inverse-frequency CE**, learning rate **3e-5**, batch **16**, maximum length **128**, **5 epochs**, weight decay **0.01**, warmup **0.10**, gradient clipping **1.0**, NULL head **off**. Keep all four sentiment classes, including Conflict.

Its five-seed explicit test F1 improves from the same-gold Phase 1 score of **33.41% to 40.28%: +6.88 percentage points**. This is a pipeline comparison, rather than an isolated loss effect.

Present the NULL extension as a measured limitation: the historical head recovers a small number of implicit pairs at a combined-score cost, while the new joint-score-selected adaptation abstains and ties its matched control. For a future NULL-focused study, focal/CLS is a reasonable development candidate; specify NULL-specific selection and precision/recall goals before a new evaluation cohort. Do not select new parameters or thresholds from the already inspected test results.

Suggested presentation sentence: **"Explicit fine-tuning improved F1 by 6.88 percentage points over Phase 1. NULL extraction remained unresolved: low precision made abstention preferable under combined F1, and the new adaptation showed no benefit against its matched control."**

## Evidence

- [Complete report, comparison charts and selected parameters](report.md).
- [Saved finalist and screen results](results.json) and [individual development/test rows](seed_results.csv).
- [Five-seed NULL selection](null_selection.json).
- [Attention seed-13 development curve](../../evidence/null_diagnostics/focal_ia_attention_e6801d313247/seed_13/dev_metrics.json), [test slices](../../evidence/null_diagnostics/focal_ia_attention_e6801d313247/seed_13/test_metrics.json) and [training audit](../../evidence/null_diagnostics/focal_ia_attention_e6801d313247/seed_13/audit.json).
- [Focal/CLS development curve](../../evidence/null_diagnostics/focal_cls_cb0f8a9e7ef7/seed_13/dev_metrics.json) and [BCE/CLS development curve](../../evidence/null_diagnostics/bce_cls_72b9117e2b7f/seed_13/dev_metrics.json).

This findings note reads existing metrics only. It does not retrain models, recalibrate test thresholds or change the frozen evaluation protocol. Both studies remain exploratory because historical test results were inspected before the follow-up.
