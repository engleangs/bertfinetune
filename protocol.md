# Experimental Protocol

Protocol version: 1.0
Frozen date: 2026-09-17
Project: Class-weighted BERT for explicit-aspect TASD
Dataset: English M-ABSA

## Protocol status

This is a post-pilot protocol freeze, not a strict pre-registration.

Before this protocol was frozen, aggregate test results from two in-domain
standard-loss runs, seeds 42 and 123, had already been produced. No weighted
or cross-domain model results had been produced.

These earlier runs are considered pilot runs. All final experiments will be
executed from the frozen code and configuration. No further changes will be
made based on test-set results.

## 1. Research question

Does inverse-frequency class-weighted cross-entropy improve exact triplet
extraction compared with standard cross-entropy when all other model,
training, and data settings are held constant?

The confirmatory experiment uses the in-domain M-ABSA condition.

A secondary experiment investigates whether the result is preserved when
generalizing to an unseen domain.

## 2. Task definition

The task is English explicit-aspect Target–Aspect–Sentiment Detection
(EA-TASD).

For each sentence, the model predicts a set of:

    (explicit aspect surface span, aspect category, sentiment)

A prediction is correct only when all three fields exactly match a gold
triplet.

This study is not a full TASD or conventional opinion-term ASTE system.
Implicit NULL aspects are outside the primary model scope.

## 3. Representable annotation scope

The baseline includes only aspect annotations that are:

1. explicit in the sentence;
2. uniquely alignable to a character span;
3. non-overlapping;
4. associated with exactly one unique category–sentiment pair.

Exact duplicate annotations are collapsed.

NULL aspects, unaligned aspects, ambiguously repeated surface terms,
overlapping spans, and spans with multiple distinct category–sentiment pairs
are excluded from the restricted task. Their counts will be reported for
every split.

No excluded annotation will silently disappear from reporting.

## 4. Data conditions

### 4.1 In-domain condition

Training, development, and test data use the official corresponding splits
from all seven English M-ABSA domains.

### 4.2 Cross-domain condition

The held-out domain is:

    restaurant

Training uses the official training splits from the other six domains.

Development uses the official development splits from the other six domains.

Final evaluation uses only:

    data/m-absa/restaurant/en/test.txt

The held-out domain's training and development examples will not be pooled
into the test set and will not be used for model training, model selection,
threshold selection, or hyperparameter tuning.

Restaurant was selected before cross-domain model training because it has the
highest category-label overlap with the source domains. This selection was
based on annotation-schema coverage, not model performance.

## 5. Category ontology policy

Category and sentiment vocabularies are constructed exclusively from the
source training split.

No held-out-domain example may add a class to the model vocabulary.

No manual category mapping will be created after model results are inspected.

A held-out gold triplet whose category is absent from the source vocabulary
remains in the overall gold evaluation set. Because the closed-set model
cannot predict that category, it contributes a false negative.

Cross-domain reporting will include:

1. full held-out exact-triplet micro-F1;
2. source-known-category exact-triplet micro-F1;
3. percentage of gold triplets covered by the source category vocabulary;
4. aspect-span F1;
5. aspect-plus-sentiment F1 independent of category;
6. number of unseen categories and affected triplets.

The full held-out score is the main cross-domain descriptive result. The
seen-category score must never be reported without the coverage percentage.

## 6. Primary outcome

The primary outcome is exact-triplet micro-F1 on the in-domain official test
set.

Exact matching requires the aspect surface span, category, and sentiment to
all match.

Micro-F1 is selected because it is stable for the highly sparse 256-category
label space and is the model-selection metric already used by the pipeline.

## 7. Secondary outcomes

The planned secondary outcomes are:

- exact-triplet macro-F1 over a fixed category universe;
- exact-triplet micro precision and recall;
- rare-category recall;
- category-level precision, recall, and F1;
- sentiment-level precision, recall, and F1;
- per-domain exact-triplet F1;
- aspect boundary F1;
- error counts for missed aspects, extra aspects, category errors, and
  sentiment errors;
- cross-domain full and source-known-category metrics.

Secondary outcomes are descriptive and exploratory. They will not be used to
change the primary conclusion.

## 8. Rare-label definition

A rare category is a category appearing between one and five times,
inclusive, among retained triplets in the relevant training split.

The rare-category set is calculated from training data only and frozen before
development or test predictions are analyzed.

For the current in-domain training data, this definition identifies 96 of 256
categories and 244 retained training triplets. The in-domain test set contains
169 retained triplets belonging to these rare categories.

Unseen categories are reported separately and are not classified as rare.

## 9. Configurations

Two configurations will be compared:

### Standard

Unweighted cross-entropy for BIO, category, and sentiment prediction.

### Weighted

Inverse-frequency class-weighted cross-entropy for BIO, category, and
sentiment prediction.

For class c:

    weight_c = N / (K * count_c)

where N is the number of valid training targets and K is the classifier
vocabulary size.

Classes absent from the training targets receive weight 1.0, although they
cannot be learned from that training split.

Loss weights are:

    BIO:       1.0
    Category:  1.0
    Sentiment: 1.0

The loss functions are the only intended experimental difference.

## 10. Fixed model and training settings

Model: bert-base-uncased
Fine-tuning: all BERT parameters
Maximum sequence length: 128
Batch size: 16
Epochs: 5
Learning rate: 2e-5
Weight decay: 0.01
Optimizer: AdamW
Scheduler: linear decay
Warmup: 10% of total training steps
Gradient clipping: maximum norm 1.0
Category loss weight: 1.0
Sentiment loss weight: 1.0

Random seeds:

    13, 42, 123, 2024, 777

Every configuration and mode uses the same seeds and hyperparameters.

No hyperparameter will be changed after final test evaluation begins.

## 11. Model selection

The model is evaluated on source-domain development data after every epoch.

The selected checkpoint is the epoch with the highest development
exact-triplet micro-F1.

Ties are resolved using:

1. lower unweighted development loss;
2. earlier epoch if both F1 and loss remain tied.

The test set is not used for checkpoint selection.

## 12. Practical effect threshold

The minimum practically meaningful improvement is:

    0.02 absolute micro-F1

For example, an improvement from 0.350 to 0.370 meets the threshold.

The weighted model is considered practically better only when:

    mean(weighted - standard) >= 0.02

Statistical significance alone is insufficient.

## 13. Statistical analysis

Runs are paired by random seed.

For the primary in-domain micro-F1 comparison, the report will include:

- each seed's standard score;
- each seed's weighted score;
- each paired difference;
- mean and standard deviation for both configurations;
- mean paired difference;
- standard deviation of paired differences;
- two-sided paired t-test;
- 95% paired bootstrap confidence interval;
- minimum-effect verdict.

The significance level is alpha = 0.05.

There is one confirmatory hypothesis: the in-domain exact-triplet micro-F1
comparison. Therefore, no multiple-comparison correction is applied to that
single primary test.

All macro-F1, rare-label, per-domain, error-analysis, and cross-domain
comparisons are secondary. Their p-values, if shown, are exploratory and
cannot independently establish the primary conclusion.

If several secondary p-values are formally interpreted, Holm correction will
be applied within that family.

## 14. Test-set handling

The complete five-seed matrix will be run before final aggregate results are
interpreted.

Individual test results will not be used to:

- change hyperparameters;
- choose the rare-label threshold;
- change the held-out domain;
- modify category mappings;
- select a different checkpoint;
- remove difficult examples.

Any deviation from this protocol will be dated and documented as an amendment.

## 15. Required experiment gates

Experiments may begin only after:

1. `python -m pip check` passes;
2. the complete unit-test suite passes;
3. the data audit is saved;
4. ambiguous-alignment policy is implemented;
5. the primary metric uses a fixed definition;
6. the environment versions are locked;
7. the git commit used for experiments is recorded.