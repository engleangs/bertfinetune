# Plan to Address Professor Feedback

## Objective

Produce a defensible evaluation of whether class-weighted BERT generalizes
across M-ABSA domains and explain *why* each configuration succeeds or fails.

The work has two connected research questions:

1. Does the standard-versus-weighted comparison hold when each M-ABSA domain
   is excluded from training and used only for testing?
2. Are failures primarily caused by aspect-term extraction, category
   classification, sentiment classification, rare labels, or annotations the
   architecture cannot represent—especially implicit `NULL` aspects?

## Success criteria

The project will be considered ready to defend when it can show:

- results for every held-out domain, not only one aggregate score;
- five matched seeds for standard and weighted loss;
- exact definitions and counts for term, category, and sentiment errors;
- the hardest domain and evidence explaining why it is difficult;
- recall for rare categories versus common categories;
- the prevalence and impact of `NULL` aspects, clearly separated from model
  errors;
- category-vocabulary coverage for every held-out domain;
- reproducible tables, figures, prediction-level error records, and example
  cases generated from saved artifacts.

## Workstream 0 — Freeze a corrected experiment version

The existing results remain **version 1 post-pilot evidence**. They must not be
overwritten.

Before new training, create a dated protocol amendment and fix the known
implementation differences:

1. Implement 10% learning-rate warmup.
2. Apply gradient clipping with maximum norm 1.0.
3. Exclude ambiguous repeated aspect occurrences instead of assigning the
   annotation to the first textual match.
4. Calculate macro-F1 over a frozen training-category universe.
5. Save the complete cross-domain metric set during evaluation.
6. Pin the software environment and save the full environment inventory.
7. Record the Git commit, protocol version, and experiment version in every
   run manifest.
8. Save standalone data audits before training.

Version the new outputs separately:

```text
artifacts/v1_post_pilot/
artifacts/v2_lodo/
results_v1.csv
results_v2_lodo.csv
```

**Gate:** the protocol amendment, audit files, locked environment, passing
tests, and frozen commit all exist before version-2 training begins.

## Workstream 1 — Seven-domain leave-one-domain-out evaluation

### Experimental design

Run leave-one-domain-out (LODO) evaluation over all seven English domains:

```text
Held out: coursera   Train/dev: other six domains   Test: coursera test
Held out: food       Train/dev: other six domains   Test: food test
Held out: hotel      Train/dev: other six domains   Test: hotel test
Held out: laptop     Train/dev: other six domains   Test: laptop test
Held out: phone      Train/dev: other six domains   Test: phone test
Held out: restaurant Train/dev: other six domains   Test: restaurant test
Held out: sight      Train/dev: other six domains   Test: sight test
```

For every fold:

- training uses only official training files from the six source domains;
- development uses only official development files from the six sources;
- testing uses only the official test file from the held-out domain;
- category and sentiment vocabularies come from source training data only;
- the held-out domain is never used for checkpoint, threshold, mapping, or
  hyperparameter decisions;
- both loss configurations use identical settings and seeds
  `13, 42, 123, 2024, 777`.

The complete matrix is:

```text
7 held-out domains × 2 losses × 5 seeds = 70 runs
```

Based on the existing restaurant fold, budget approximately 20 GPU-hours for
the full LODO matrix. First run a 14-run, one-seed integration matrix to expose
data- or ontology-specific failures before committing to the other seeds.

### Cross-domain metrics

Save these metrics for every run:

- full held-out exact-triplet micro precision, recall, and F1;
- source-known-category exact-triplet precision, recall, and F1;
- source-category coverage of held-out gold triplets;
- fixed-universe macro-F1;
- aspect-span precision, recall, and F1;
- aspect-plus-sentiment precision, recall, and F1;
- rare-category recall;
- unseen category count and affected gold triplets;
- error counts from the taxonomy below.

### Statistical reporting

The original in-domain comparison remains the confirmatory analysis. LODO is a
planned secondary analysis.

Report:

- mean ± standard deviation across five seeds for each held-out domain;
- each paired seed difference, weighted minus standard;
- paired bootstrap confidence interval for every domain;
- a seven-domain macro-average for each seed, followed by a paired comparison
  of those five seed-level macro-averages;
- Holm-adjusted p-values only if individual domain p-values are formally
  interpreted;
- the pre-specified 0.02 practical-effect threshold separately from
  statistical significance.

**Deliverable:** a domain-by-configuration results table and a heatmap that
ranks held-out domains by full exact-triplet F1.

## Workstream 2 — First-class error analysis

### 2.1 Create a mutually exclusive error taxonomy

For each retained gold triplet, assign exactly one outcome using this order:

1. **Correct triplet** — aspect, category, and sentiment all match.
2. **Term/aspect error** — no prediction matches the gold aspect span.
3. **Category error** — the aspect matches, but the gold category does not.
4. **Sentiment error** — aspect and category match, but sentiment does not.

Separately classify prediction-side errors:

- extra/spurious aspect;
- duplicate prediction;
- correct aspect with unsupported category;
- boundary-only mismatch, where the predicted and gold spans overlap but are
  not identical.

This hierarchy prevents one failed triplet from being counted simultaneously
as a term, category, and sentiment error.

Save one machine-readable record per evaluated example:

```json
{
  "mode": "lodo",
  "held_out_domain": "restaurant",
  "config": "weighted",
  "seed": 42,
  "sentence": "...",
  "gold_triplet": ["service", "service general", "negative"],
  "matched_prediction": null,
  "error_type": "term_missed",
  "category_frequency": 231,
  "rarity_group": "common",
  "category_known_to_source": true,
  "aspect_type": "explicit"
}
```

### 2.2 Analyze aspect-term failures

Measure by domain and configuration:

- exact aspect-span recall;
- missed aspects;
- extra aspects;
- partial/boundary overlaps;
- misses by aspect length;
- misses by sentence length;
- misses involving repeated surface terms;
- misses caused by truncation or token alignment.

Use deterministic qualitative sampling: show the most frequent failure
patterns and fixed examples selected by support/error frequency, rather than
hand-picking unusual cases.

### 2.3 Analyze category failures

For every domain:

- category precision, recall, F1, and support;
- most-confused category pairs;
- source-known versus unseen held-out categories;
- category-vocabulary coverage;
- recall by training-frequency band;
- relationship between ontology coverage and exact-triplet F1.

Recommended frequency bands calculated from the relevant source training fold:

```text
rare:       1–5 retained training triplets
low:        6–20
medium:     21–100
frequent:   >100
unseen:     0, reported separately
```

Do not classify unseen categories as rare.

### 2.4 Analyze sentiment failures

Report:

- sentiment precision, recall, F1, and support;
- confusion matrices for positive, negative, neutral, and conflict;
- sentiment accuracy conditional on a correct aspect span;
- sentiment accuracy conditional on a correct aspect and category;
- sentiment errors by domain and category-frequency band.

Conditional reporting is important because a sentiment head cannot receive
credit or blame when the aspect itself was never extracted.

### 2.5 Analyze rare categories

For each fold and configuration:

- freeze the rare-category set from source training data only;
- report rare-category support in the held-out test set;
- calculate micro recall over rare-category gold triplets;
- compare rare and common recall using matched seeds;
- show whether weighting improves rare recall while damaging precision or
  frequent-category performance;
- identify rare categories that are consistently recovered or consistently
  missed across all five seeds.

**Required figure:** rare-versus-common recall for standard and weighted loss,
faceted by held-out domain.

### 2.6 Treat NULL aspects honestly

The current explicit-aspect BIO architecture cannot emit `NULL` aspects.
Therefore, excluded `NULL` annotations are a **structural coverage limitation**,
not ordinary false negatives from the trained model.

For every domain and split, report:

- total `NULL` triplets;
- percentage of raw triplets that are `NULL`;
- their category and sentiment distributions;
- the theoretical full-TASD recall ceiling created by excluding them;
- results both on the explicit-aspect task and, as a clearly labelled
  diagnostic, against the full raw annotation set where every NULL triplet is
  necessarily missed.

The report must say explicitly:

> The model has zero representational coverage for implicit NULL aspects; this
> is an architectural scope decision, not evidence that weighting failed to
> learn them.

If the project must claim full TASD performance, add a separate NULL-aware
component—such as a sentence-level implicit category/sentiment head—and treat
that as a new model extension. Do not mix that extension into the protected
standard-versus-weighted comparison without another protocol amendment.

## Workstream 3 — Identify and explain the hardest domain

Rank domains using five-seed mean full exact-triplet F1. For each domain, place
performance beside possible explanatory variables:

- source-category coverage;
- unseen-category triplet percentage;
- rare-category percentage;
- NULL-aspect percentage;
- median and 95th-percentile sentence length;
- aspect-span recall;
- category-error rate;
- sentiment-error rate;
- training/test label-distribution divergence.

Do not conclude that a variable *causes* difficulty from correlation alone.
Use language such as “consistent with” or “associated with.”

**Required figures:**

1. held-out-domain F1 ranking with mean ± standard deviation;
2. 100% stacked error composition by domain;
3. domain × error-type heatmap;
4. category coverage versus full exact-triplet F1;
5. rare/common recall comparison;
6. NULL prevalence and representational ceiling by domain.

## Workstream 4 — Reporting package

Generate the following artifacts directly from saved predictions:

```text
artifacts/v2_lodo/analysis/
├── lodo_summary.csv
├── paired_seed_differences.csv
├── error_records.jsonl
├── error_type_by_domain.csv
├── category_confusions.csv
├── sentiment_confusions.csv
├── rare_category_analysis.csv
├── null_scope_analysis.csv
├── qualitative_examples.md
├── analysis_summary.json
└── figures/
```

The professor-facing presentation should use this sequence:

1. research question and LODO design;
2. data and ontology coverage by fold;
3. result across all seven held-out domains;
4. hardest-domain ranking;
5. term/category/sentiment error decomposition;
6. rare-category effect of weighting;
7. NULL limitation and full-task ceiling;
8. conclusion: where weighting helps, where it hurts, and why.

## Implementation tasks

### Data and experiment configuration

- Replace the single `HOLD_OUT_DOMAIN` workflow with an explicit fold iterator.
- Give every run a key containing experiment version and held-out domain.
- Assert that the held-out domain never appears in train or development data.
- Save source and held-out domain lists in every manifest.

### Evaluation

- Add span-projected and aspect-plus-sentiment metrics to `src/evaluate.py`.
- Add fixed-universe macro-F1.
- Implement mutually exclusive gold-side error attribution.
- Add boundary-overlap diagnostics.
- Emit prediction-level error records.
- Add rare-frequency bands and NULL scope summaries.

### Orchestration and storage

- Extend `run_all_study.py` with `--held-out-domain` and `--lodo-all`.
- Prevent `--overwrite` from replacing another experiment version.
- Add checkpoint-only evaluation so metrics can be recalculated without
  retraining.
- Add an analysis command that validates all 70 run keys before aggregation.

### Tests

- Test fold membership and absence of held-out leakage.
- Test train-only vocabulary construction for every fold.
- Test mutually exclusive error attribution with known examples.
- Test boundary mismatch, category error, and sentiment error cases.
- Test rare, unseen, and NULL classification.
- Test fixed-universe macro-F1.
- Test complete 70-run matrix validation.
- Test that analysis values reproduce saved `metrics.json` results.

## Proposed one-week sequence

### Day 1 — Protocol and definitions

- Agree on the dated amendment.
- Freeze the LODO aggregation rule and error hierarchy.
- Decide whether NULL remains a reported scope limitation or becomes a new
  model extension.

### Day 2 — Engineering fixes

- Implement warmup, clipping, ambiguous-span exclusion, manifests, audit
  gates, and dependency locking.
- Add unit tests.

### Day 3 — Evaluation pipeline

- Implement projected metrics, fixed macro-F1, error attribution, rare bands,
  NULL summaries, and prediction-level records.
- Run synthetic known-answer tests.

### Day 4 — LODO smoke matrix

- Run all seven folds with one seed and both losses: 14 runs.
- Verify dataset sizes, category coverage, and artifacts without interpreting
  test performance to tune the model.

### Days 5–6 — Full matrix

- Run the remaining four seeds for all folds: 56 runs.
- Validate all 70 keys and artifact packages.

### Day 7 — Analysis and presentation

- Produce tables, figures, qualitative examples, and the eight-slide
  professor update.
- Lead with the cross-domain pattern and use error analysis to explain it.

## Final decision points for the professor

Ask for approval on three items before starting version 2:

1. Is the seven-fold LODO macro-average the correct headline secondary result?
2. Should NULL aspects remain a quantified architectural limitation, or must a
   NULL-aware model extension be implemented?
3. Does the professor approve the dated protocol amendment and separate
   version-2 rerun rather than replacing the existing post-pilot artifacts?

