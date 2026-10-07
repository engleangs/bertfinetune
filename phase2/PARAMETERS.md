# Settings and evaluation protocol

The full per-row settings are in [parameters.csv](parameters.csv) and
[parameters.json](parameters.json). Each row includes the trial ID, seed,
chosen checkpoint epoch and development-selected NULL threshold. The
configurations in results/*/results.json and the original manifests in
the evidence ZIPs provide the underlying provenance.

## Selected explicit fine-tuning settings

| Setting | Value |
|---|---|
| Backbone | bert-base-uncased |
| Immutable model revision | 86b5e0934494bd15c9632b12f734a8a67f723594 |
| Loss | 0.75 standard CE + 0.25 inverse-frequency weighted CE |
| Heads with this loss | BIO, category, sentiment |
| Head objective weights | 1.0, 1.0, 1.0 |
| Learning rate | 3e-5 |
| Maximum epochs | 5; checkpoint selected on explicit development F1 |
| Batch size / maximum sequence length | 16 / 128 |
| Weight decay / warmup ratio | 0.01 / 0.10 |
| Gradient clipping | 1.0 |
| NULL head / vocabulary | Off / auto (explicit source-training vocabulary) |
| Sentiments | Positive, Negative, Neutral, Conflict retained where supported by training |
| Seeds | 13, 42, 123, 2024, 777 |

These settings were frozen before the final evaluation and reused in the
35 Phase 2 LODO runs. Each LODO fold trains on six source domains, chooses
its checkpoint using source development data and evaluates the held-out
domain. Category vocabularies and frequency counts are source-only in
each fold. The choice of the common hyperparameters itself used all-domain
in-domain development results; therefore this is exploratory transfer,
rather than a target-free LODO hyperparameter search.

## Professor's loss recommendations

The quick search compared standard CE, inverse-frequency weighted CE,
their mixtures, and focal loss with gamma 2 at LR 2e-5, seed 13 and five
epochs. Mixtures used CE/weighted-CE coefficients 0.75/0.25, 2/3 and 1/3,
and 0.50/0.50. These cover the teacher's suggested relative mixture weights:
1.0/0.5 is the same ratio as 2/3 and 1/3 after normalization.
The selected mixture then received a LR sweep of 1e-5, 2e-5 and 3e-5.

The implemented mixture uses the stated normalized coefficients. Scaling
the entire objective can affect optimization, so normalized and unnormalized
objectives are not claimed to be identical training runs. Standard and
weighted losses remain separate controls. Focal was screened; it was not
chosen as the main explicit loss.

Read [development screens](comparison/development_screens.csv) for the
selection evidence and [test ablations](comparison/screen_ablations.csv)
for the separately reported exploratory test scores. Different loss families
did not all receive a complete LR sweep. An inactive parameter may still
appear in a configuration, for example focal_gamma in a mixed-loss run;
loss_type determines whether it is used.

## NULL arms and their controls

| Arm | Explicit objective | NULL representation/loss | Vocabulary | Selection | Frozen NULL threshold |
|---|---|---|---|---|---|
| Established NULL off | Selected mixture | Disabled | auto | Explicit dev F1 | Inactive |
| Historical quick NULL on | Selected mixture | CLS / BCE; weight 0.5, positive-weight cap 10 | explicit-null | Explicit checkpoint; development threshold search | 0.2 across five seeds |
| Historical vocabulary control | Selected mixture | Disabled | explicit-null | Explicit dev F1 | Inactive |
| New attention adaptation | Selected mixture | Dedicated IA attention / focal gamma 2; weight 0.5, cap 10 | explicit-null | Combined dev F1 | 1.0 across five seeds (abstention) |
| New adaptation control | Selected mixture | Disabled | explicit-null | Combined dev F1 | Inactive |

The newer adaptation and its control use AMP; attention size is 128 and the
selected negative-sampling setting is all. The explicit settings are the
same LR, epochs, batch size and length as the main fine-tuning recipe.
Historical and new NULL studies differ in precision and selection objective.
Compare each enabled head with its matching control.

The NULL screen additionally evaluated BCE, focal, asymmetric loss (ASL),
CLS, dedicated IA token, IA attention, random/hard negative sampling and
NULL weight/gamma variants. These are short seed-13, three-epoch screens.
All actual values and frozen thresholds are recorded in parameters.csv.
Do not apply the best diagnostic NULL-only development threshold to the
inspected tests: the reported tests use the pre-test frozen selection.

## Score definitions

Exact explicit scoring compares sentence-local sets of (term, category,
sentiment). NULL scoring compares sentence-local sets of (category,
sentiment) for implicit targets. Combined precision/recall/F1 comes from
summed explicit and NULL TP/FP/FN; it is not the mean of the two F1 values.

For component F1, project and deduplicate the gold and predictions within
each sentence to term, (term, category), or (term, sentiment). These are
different tasks with different denominators. Their scores do not sum
to exact-triplet F1 and are not isolated classification-head accuracies.

The primary taxonomy assigns each retained gold triplet its first failed
component: correct, missing term, wrong category given term, or wrong
sentiment given term and category. It partitions gold, not prediction
false positives; it is an ordered diagnostic and does not prove a sole cause.
Use the original result taxonomies for independent flags and prediction-side
diagnostics.

Rare means source training count 1-5; unseen means 0. Never estimate these
counts from the target test set. Category coverage measures the fraction
of gold with a category supported by the model's own source vocabulary.
Zero coverage is a closed-set representational limitation under exact
triplet scoring, even if terms or sentiment transfer.

For Phase 1 comparisons, use the consolidated report's same-gold rescoring.
The historical snapshots use older eligibility rules. New and old pipelines
also differ in loss and optimizer settings, so their differences do not
identify a loss-only effect. Sample SD describes seed variation; repeated
evaluation of the same test examples across seeds does not create independent
new datasets. These follow-up analyses are exploratory.
