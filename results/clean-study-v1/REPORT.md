# CS760 Experiment Results Summary

Study version: clean-study-v1 | Report date: 3 October 2026

This report summarizes the conflict-label cleanup, the original-configuration replication, and the experiments using the new training configuration. It compares Standard loss with Weighted loss, which applies inverse-frequency class weighting to all three prediction heads.

**Main findings:** Removing conflict-containing examples produced only small mean changes under the original configuration. The new training configuration substantially narrowed the observed performance gap. Final in-domain exact-triplet F1 was **49.68 for Standard** and **48.98 for Weighted**. Cross-domain exact-triplet performance was strongly constrained by category-vocabulary coverage and should be interpreted alongside aspect-extraction metrics. These are descriptive findings; no statistical significance tests are reported.

## 1 Reading the metrics

- All F1, Precision, Recall, and coverage values use a **0–100 scale**. A value of 0.4968 in the source JSON is presented as 49.68 here.
- **Mean ± standard deviation** summarizes five runs with seeds 13, 42, 123, 2024, and 777. Standard deviation describes variation across seeds; it is not a confidence interval.
- The primary metric is **test exact-triplet micro-F1**. The aspect text, category, and sentiment must all match. F1 is not accuracy.
- Precision, Recall, and F1 are averaged separately across the five runs. Mean F1 is not recalculated from mean Precision and mean Recall.
- A gain of 0.29 F1 points is an absolute score difference, not a relative improvement of 0.29%. Loss-comparison differences consistently use **Weighted minus Standard**.

| Metric | Meaning |
| --- | --- |
| Precision | The proportion of predicted triplets that are correct |
| Recall | The proportion of gold triplets recovered by the model |
| F1 | The harmonic mean of Precision and Recall |
| Aspect F1 | Matches aspect text without requiring a category or sentiment match |
| Aspect + sentiment F1 | Matches aspect text and sentiment without requiring a category match |
| Category coverage | The proportion of gold test triplets whose category belongs to the source-training category vocabulary |
| Rare-category Recall | The recall measure for categories defined as rare by the implementation; distinct from exact-triplet F1 |

## 2 Experiment completion and configuration

| Stage | Design | Count |
| --- | --- | --- |
| Original-configuration replication | Raw/clean data × 2 losses × 5 seeds; pooled in-domain setting | 20 training runs and test evaluations |
| Learning-rate search | 1 pooled in-domain + 7 leave-one-domain-out settings × 2 losses × 3 learning rates × 2 seeds | 96 training runs with development evaluation |
| Final comparison | Reuse 32 selected search checkpoints; train the remaining 3 seeds in 48 additional runs | 80 final test results |
| Smoke checks | Small runs to verify the training workflow | 2 runs |
| Total | Every training history reaches its configured epoch budget; all planned test results are present | 164 full training runs + 2 smoke checks; 100 test results |

In-domain is one training condition that pools all seven domains while retaining the official train/development/test partitions. Leave-one-domain-out (LODO) uses six domains for training and tests on the remaining domain, yielding seven cross-domain conditions.

| Setting | Original configuration | Selected final configuration |
| --- | --- | --- |
| Data | Raw and clean variants evaluated separately | Clean |
| Learning rate | 2e-5 | 5e-5 |
| Training budget | 5 epochs | 30 epochs |
| Warmup | 0 | 10% |
| Gradient clipping | None | 1.0 |
| Batch size | 16 | 16 |
| Maximum sequence length | 128 | 128 |
| Weight decay | 0.01 | 0.01 |
| Checkpoint selection | Best development exact-triplet F1 | Best development exact-triplet F1 |

Cleanup removed five complete training examples containing conflict labels: five conflict annotations and two additional positive annotations. Original files were preserved; development and test files were unchanged. Learning rates were selected separately within each condition using source-domain development scores, without using held-out target-domain test performance.

## 3 Original-configuration replication and final in-domain results

Precision, Recall, and F1 below refer to exact triplets on the test set. Each row summarizes n = 5 seeds.

| Experiment | Loss | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Raw data, 5 epochs | Standard | 38.83 ± 0.74 | 31.98 ± 1.04 | 35.07 ± 0.75 |
| Raw data, 5 epochs | Weighted | 19.25 ± 0.29 | 42.89 ± 0.40 | 26.57 ± 0.27 |
| Clean data, 5 epochs | Standard | 39.14 ± 1.03 | 32.24 ± 0.99 | 35.36 ± 0.98 |
| Clean data, 5 epochs | Weighted | 19.57 ± 0.32 | 43.15 ± 0.38 | 26.93 ± 0.36 |
| Clean data, new configuration | Standard | 51.98 ± 0.84 | 47.58 ± 0.29 | 49.68 ± 0.36 |
| Clean data, new configuration | Weighted | 49.24 ± 0.50 | 48.73 ± 0.44 | 48.98 ± 0.25 |

### 3.1 Effect of removing conflict-containing examples

| Loss | Raw-data F1 | Clean-data F1 | Clean minus raw | Seeds with an increase |
| --- | --- | --- | --- | --- |
| Standard | 35.07 | 35.36 | +0.29 | 3/5 |
| Weighted | 26.57 | 26.93 | +0.36 | 4/5 |

The mean changes were small, and improvement was not consistent across every seed. After cleanup, the original-configuration F1 gap remained 8.43 points. These results do not support the explanation that the few conflict examples were the main cause of Weighted loss underperforming. Cleanup changed both the examples and the sentiment output vocabulary, so the effect cannot be attributed solely to one large class weight.

### 3.2 Effect of the new training configuration

| Loss | Clean data with original configuration | Clean data with new configuration | New minus original F1 |
| --- | --- | --- | --- |
| Standard | 35.36 | 49.68 | +14.32 |
| Weighted | 26.93 | 48.98 | +22.05 |

The gap between the losses narrowed from 8.43 to 0.70 F1 points. Standard achieved higher final exact-triplet F1 in all five paired seeds. Learning rate, training duration, warmup, and gradient clipping changed together; the gains cannot be attributed separately to a longer training budget or a higher learning rate.

### 3.3 Final in-domain F1 by seed

| Seed | Standard | Weighted | Weighted minus Standard |
| --- | --- | --- | --- |
| 13 | 49.69 | 49.22 | -0.47 |
| 42 | 49.32 | 48.98 | -0.35 |
| 123 | 49.93 | 48.59 | -1.34 |
| 2024 | 49.32 | 49.20 | -0.12 |
| 777 | 50.13 | 48.92 | -1.22 |

## 4 Final cross-domain results

The test domain is held out from training. Precision, Recall, and F1 are reported as mean ± standard deviation across five seeds.

| Test domain | Loss | Precision | Recall | Exact-triplet F1 |
| --- | --- | --- | --- | --- |
| Coursera | Standard | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Coursera | Weighted | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Hotel | Standard | 24.29 ± 1.62 | 27.88 ± 0.48 | 25.95 ± 1.07 |
| Hotel | Weighted | 22.93 ± 0.27 | 27.85 ± 0.44 | 25.15 ± 0.21 |
| Laptop | Standard | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Laptop | Weighted | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Restaurant | Standard | 29.08 ± 1.70 | 13.76 ± 1.64 | 18.65 ± 1.75 |
| Restaurant | Weighted | 28.20 ± 1.84 | 13.90 ± 0.30 | 18.61 ± 0.50 |
| Phone | Standard | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Phone | Weighted | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Sight | Standard | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Sight | Weighted | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Food | Standard | 6.62 ± 0.49 | 8.11 ± 0.34 | 7.29 ± 0.41 |
| Food | Weighted | 5.42 ± 0.42 | 7.79 ± 0.88 | 6.39 ± 0.57 |

### 4.1 Interpreting zero scores in four domains

| Test domain | Category coverage |
| --- | --- |
| Coursera | 0.24% |
| Hotel | 38.12% |
| Laptop | 0.00% |
| Restaurant | 71.28% |
| Phone | 0.00% |
| Sight | 0.00% |
| Food | 27.71% |

The current category head can only predict categories in the source-training vocabulary. Category coverage is 0% for Laptop, Phone, and Sight, making exact-triplet matches impossible under this setup. Coursera coverage is approximately 0.24%: only one of its 409 gold triplets has a category in the source vocabulary. Its final exact-triplet F1 is also zero.

These zero scores do not establish that the model learned no aspect or sentiment information, and they cannot distinguish the overall quality of the losses in those settings. They expose a cross-domain category-vocabulary limitation. When coverage is zero, an evaluation restricted to source-known categories also has no matching gold categories for a meaningful comparison.

## 5 Supplementary aspect and sentiment extraction results

The following values are five-seed means. Aspect matching uses surface text rather than occurrence offsets, so separate occurrences of the same text are not distinguished by this projection.

| Condition | Standard Aspect F1 | Weighted Aspect F1 | Standard Aspect + sentiment F1 | Weighted Aspect + sentiment F1 |
| --- | --- | --- | --- | --- |
| In-domain | 66.05 | 66.63 | 61.53 | 61.66 |
| Coursera | 61.72 | 64.38 | 57.43 | 58.20 |
| Hotel | 66.48 | 67.37 | 63.57 | 64.24 |
| Laptop | 47.45 | 54.20 | 44.37 | 50.03 |
| Restaurant | 46.71 | 47.28 | 43.67 | 44.34 |
| Phone | 31.26 | 33.69 | 29.42 | 31.25 |
| Sight | 36.44 | 38.94 | 33.29 | 34.98 |
| Food | 52.66 | 53.65 | 43.78 | 43.55 |

Weighted had higher mean aspect F1 in all seven cross-domain settings. Mean aspect + sentiment F1 was higher in six of seven, with Food slightly lower. For Laptop, aspect F1 increased from 47.45 to 54.20 and aspect + sentiment F1 from 44.37 to 50.03, while exact-triplet F1 remained zero. Thus, the category-vocabulary limitation obscures some extraction gains in the exact-triplet score. These observations do not establish statistical significance or isolate the causal contribution of an individual prediction head.

### 5.1 In-domain rare-category recall

| Experiment | Standard rare-category Recall | Weighted rare-category Recall |
| --- | --- | --- |
| Raw data, 5 epochs | 0.00 ± 0.00 | 2.84 ± 0.97 |
| Clean data, 5 epochs | 0.00 ± 0.00 | 2.60 ± 0.99 |
| Clean data, new configuration | 8.64 ± 2.08 | 8.05 ± 0.79 |

Weighted had higher rare-category recall under the original configuration, but did not retain a mean advantage under the new configuration. Rarity uses the training-frequency threshold of 5 in the implementation. Category sets differ across cross-domain conditions, so this is not performance on one shared set of rare categories across all domains.

## 6 Learning rate and training duration

| Condition | Selected learning rate |
| --- | --- |
| In-domain | 5e-5 |
| Coursera | 5e-5 |
| Hotel | 5e-5 |
| Laptop | 5e-5 |
| Restaurant | 5e-5 |
| Phone | 5e-5 |
| Sight | 5e-5 |
| Food | 5e-5 |

Each condition independently selected 5e-5 from 2e-5, 3e-5, and 5e-5. Selection used the mean best development exact-triplet F1 over both losses and seeds 13 and 42; both losses then shared the selected learning rate. This identifies the choice among the tested candidates, not a global optimum or evidence that still higher rates would improve performance.

Across the 80 final models, best development F1 occurred between epochs 18 and 30, with a median of epoch 28. In 15 models, the best score occurred at epoch 30.

| Loss | Epoch of minimum development loss | Epoch of best development F1 |
| --- | --- | --- |
| Standard | 5–6 | 28–29 |
| Weighted | 7–8 | 27–30 |

Minimum development loss and best F1 occurred at different times. Checkpoints were selected by development F1 rather than automatically using the last epoch. Epoch 8 within a 30-epoch schedule is not equivalent to the previous standalone 8-epoch schedule. A best score at epoch 30 does not guarantee that extending training would improve results.

## 7 Conclusions for the group report

1. **Conflict cleanup had a small effect.** Under the original configuration, Standard and Weighted improved by 0.29 and 0.36 mean F1 points, respectively; the performance gap remained.
2. **The new configuration substantially narrowed the exact-triplet F1 gap.** Final in-domain scores were 49.68 versus 48.98, a difference of approximately 0.70 points in favor of Standard.
3. **Category coverage limits cross-domain evaluation.** Four domains had zero exact-triplet F1; interpretation requires category-coverage and aspect-extraction results.
4. **Weighted showed gains in some extraction metrics.** Mean aspect F1 was higher in all seven cross-domain conditions, without a corresponding mean exact-triplet F1 advantage.

**Scope and limitations:** This is an exploratory follow-up after earlier test results had been inspected. The final five-seed results reuse selection seeds 13 and 42, rather than providing five confirmation seeds independent of selection. No significance tests were performed; the report describes means, standard deviations, and observed differences. The task retains the original explicit-aspect and first-usable-label-pair filtering rules. NULL implicit aspects remain outside the evaluated task.

## 8 Sources and verification

Source: [published experiment evidence](experiment_results.zip), derived from the verified cluster export. All metrics and predictions are unchanged; see [publication notes](README.md#archive-provenance) for the documented omission of one historical Chinese guide.

| File inside the archive | Use in this report |
| --- | --- |
| results.csv | 100 test results; metric means, standard deviations, and paired differences recalculated from individual rows |
| summary.json | Cross-check of F1 summaries and cleanup effects |
| selection.json | Selected learning rate and selection basis for each condition |
| runs/…/history.json | Per-epoch losses and development F1; epoch-budget and best-epoch checks |
| runs/…/metrics.json | Component metrics, category coverage, and unseen categories |
| runs/…/manifest.json | Completion flags, configuration, label vocabulary, and environment |
| runs/…/test_predictions.jsonl.gz | Per-example gold and predicted triplets for subsequent error analysis |

**Verification:** All 100 results form complete five-seed groups. Recalculated group mean F1 values agree with summary.json. All 166 training records reach their configured epoch budgets, and 100 records have completed test evaluation. Standard deviations use the sample definition with denominator n−1.

Example source path: `runs/search/clean/indomain/5em05/standard/seed_13/history.json`. Selected search checkpoints were reused for final evaluation, so some final results legitimately reside under search directories.
