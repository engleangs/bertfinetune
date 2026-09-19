# Next Plan

**Goal:** Convert the working version-1 pipeline into a defensible,
protocol-compliant seven-domain generalization study with first-class error
analysis.

## Priority 1: Freeze and correct version 2

Preserve all current results as post-pilot evidence. Do not overwrite them.

Before new training:

- write a dated protocol amendment explaining the version-1 deviations;
- implement 10% learning-rate warmup;
- implement gradient clipping with maximum norm 1.0;
- exclude ambiguous repeated surface-term alignments;
- calculate macro-F1 over the fixed source-training category universe;
- save all cross-domain metrics during evaluation rather than deriving them
  afterward;
- generate standalone audits before training;
- lock package versions and save a complete environment inventory;
- record Git commit, protocol version, and experiment version in every run;
- keep version-2 outputs separate from version 1.

**Completion gate:** 26 or more tests pass, `pip check` passes, both data audits
exist, the environment is locked, and a clean experiment commit is recorded.

## Priority 2: Add seven-domain leave-one-domain-out execution

For each domain in `coursera`, `food`, `hotel`, `laptop`, `phone`,
`restaurant`, and `sight`:

1. exclude that domain from training and development;
2. build vocabularies from the six source training splits only;
3. select checkpoints using only the six source development splits;
4. evaluate once on the official held-out test split;
5. run both standard and weighted loss with seeds
   `13, 42, 123, 2024, 777`.

The full design is:

```text
7 held-out domains x 2 loss configurations x 5 seeds = 70 runs
```

First run a one-seed integration matrix:

```text
Remaining work after restaurant:
6 domains x 2 configurations x 1 seed = 12 smoke runs
```

Run it with:

```powershell
python run_lodo_study.py --seeds 13 --device cuda --continue-on-error
```

Only start the other 48 remaining-domain runs after the smoke matrix passes.
The complete seven-domain study still contains 70 logical runs when the 10
completed restaurant runs are included. At the current per-run duration,
reserve approximately 20 GPU-hours for the entire seven-domain matrix.

## Priority 3: Make error analysis a pipeline output

Create one machine-readable record for every retained gold triplet and assign
exactly one primary outcome:

1. correct triplet;
2. term/aspect error: no exact aspect span match;
3. category error: aspect matches but category does not;
4. sentiment error: aspect and category match but sentiment does not.

Record prediction-side errors separately:

- extra or spurious aspect;
- duplicate prediction;
- partial boundary overlap;
- unsupported or unseen category.

Every error record should include held-out domain, loss configuration, seed,
sentence, gold and matched predicted triplets, training category frequency,
rarity band, source-category coverage, and explicit/`NULL` aspect type.

## Priority 4: Answer the professor's diagnostic questions

### Term failures

Report missed, extra, and boundary-mismatched aspects by domain, aspect length,
sentence length, repeated term, alignment, and truncation status.

### Category failures

Report category precision, recall, F1, common confusions, and recall by
source-training frequency:

```text
rare: 1-5       low: 6-20       medium: 21-100
frequent: >100  unseen: 0, reported separately
```

### Sentiment failures

Report the four-class confusion matrix and sentiment performance conditional on
the correct aspect, and conditional on both the correct aspect and category.

### Rare labels

Freeze the rare set from each fold's source training data. Compare rare versus
common recall for standard and weighted loss, and check whether improved rare
recall is purchased with lower precision or weaker frequent-label performance.

### `NULL` aspects

Report count, prevalence, category distribution, sentiment distribution, and
the theoretical full-TASD recall ceiling for every domain. State clearly that
the current architecture has zero representational coverage for `NULL`
aspects. Add a NULL-aware model only as a separately amended extension.

## Priority 5: Identify the hardest domain

Rank domains by five-seed mean full exact-triplet F1. Interpret the ranking
beside:

- category-vocabulary coverage;
- unseen and rare triplet percentages;
- `NULL` prevalence;
- sentence-length distribution;
- aspect-span recall;
- category and sentiment error rates;
- source/test label-distribution divergence.

Use association language rather than causal language unless a controlled
experiment supports causality.

## Required final outputs

### Tables

- domain x configuration mean and standard deviation;
- paired differences and confidence intervals;
- category coverage and unseen-category counts;
- term/category/sentiment error counts;
- rare versus common recall;
- `NULL` prevalence and representational ceiling.

### Figures

1. held-out-domain F1 ranking with uncertainty;
2. standard-versus-weighted paired differences by domain;
3. 100% stacked term/category/sentiment error composition;
4. domain x error-type heatmap;
5. category coverage versus F1;
6. rare-versus-common recall;
7. `NULL` prevalence by domain;
8. representative failure cases.

### Reproducible artifacts

```text
artifacts/v2_lodo/<held_out>/<config>/seed_<seed>/
results_v2_lodo.csv
error_records.jsonl
error_summary.csv
domain_summary.csv
rare_category_summary.csv
null_aspect_summary.csv
figures/
```

## Proposed sequence for the next seven working days

| Day | Work | Exit condition |
|---|---|---|
| 1 | Protocol amendment and versioned output design | Decisions and deviations are frozen. |
| 2 | Training/evaluation corrections | Unit tests cover warmup, clipping, alignment, and fixed macro-F1. |
| 3 | LODO runner and error-record generator | One local end-to-end fold succeeds. |
| 4 | Fourteen-run smoke matrix | All domains and both losses produce valid artifacts. |
| 5-6 | Remaining full matrix | All 70 logical run keys complete exactly once. |
| 7 | Statistical/error analysis and professor slides | Tables, figures, examples, and conclusions regenerate from saved artifacts. |

## Definition of done

The next stage is complete when:

- all seven domains have five matched seeds for both configurations;
- the corrected experiment version passes every protocol gate;
- each gold triplet has a deterministic error classification;
- the hardest domain is identified and explained with measured factors;
- rare, unseen, and `NULL` aspects are reported separately;
- conclusions distinguish confirmatory in-domain evidence from exploratory
  cross-domain evidence;
- the professor presentation can be regenerated entirely from saved data and
  prediction artifacts.
