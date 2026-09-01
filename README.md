# ABSA Triplet Extraction — Group07 Scaffold

## What changed from the LoRA version
- **Core comparison unchanged**: BERT with standard loss vs. class-weighted
  loss. Still the one variable that differs between configs.
- **New research question, layered on top**: does the comparison hold when
  generalizing to an unseen domain? Same two configs, run under two DATA
  conditions instead of two MODEL variants:
  - `indomain` — original split, all 7 domains mixed (protected core)
  - `crossdomain` — train on 6 domains, test on 1 fully held-out domain
- **Mandatory rigor, both conditions**: 5 seeds per config, mean +/- std
  (not a single number), a paired significance test, a pre-registered
  minimum effect size, precision/recall by label (not just F1), and a
  direct check of whether class-weighting improves *rare-label recall*
  specifically — not just the headline score.

## Before running anything: pre-register two decisions

In `config.py`:
```python
HOLD_OUT_DOMAIN = "coursera"   # which domain gets held out for crossdomain
MIN_EFFECT_SIZE = 0.02          # F1 points that count as "improvement"
```
Write these down somewhere dated (commit message, shared doc) BEFORE you run
a single experiment. Changing them after seeing results defeats the point —
that's what "pre-registered" means.

## Layout

```
absa-triplet/
├── config.py              # experiment configs, seeds, domain split, pre-registration
├── src/
│   ├── data.py             # M-ABSA parsing, BIO tagging, in-domain + cross-domain splits
│   ├── model.py             # BERT + 3 heads
│   ├── losses.py             # standard vs class-weighted loss
│   ├── train.py              # training, dev selection, and predicted-span evaluation
│   ├── artifacts.py          # atomic checkpoints, JSON/JSONL, and CSV upserts
│   ├── evaluate.py           # complete-triplet F1, precision/recall by label, rare-label recall
│   ├── stats.py               # paired significance test + bootstrap CI + effect-size check
│   └── utils.py                # seeding
├── run_study.py            # one train/dev/test (mode, config, seed) run
├── run_all_study.py        # filtered matrix; defaults to the 10 in-domain runs
└── analyze_results.py      # mean/std, significance, effect-size verdict — the final table
```

## Data format & paths

Same M-ABSA `sentence####[(term, category, sentiment), ...]` format as
before. Files are stored locally under
`data/m-absa/<domain>/en/{train,dev,test}.txt`; for example,
`data/m-absa/coursera/en/train.txt`. `config.DATA_DIR` points to
`data/m-absa`, and `config.DOMAIN_FILES` maps all seven English domains and
three splits to that layout. `python download_data.py` downloads only missing
files from https://github.com/swaggy66/M-ABSA; the generated `data/` directory
is intentionally ignored by Git.

The current local dataset passes the acquisition/loading smoke check: all 21
expected files match upstream, all 14,776 non-empty rows parse, and both the
in-domain and cross-domain split builders load successfully.

## Running it

```powershell
pip install -r requirements.txt
python download_data.py
python -m unittest discover -s tests -v

# One complete train -> validation -> test run
python run_study.py --mode indomain --config standard --seed 42 --device cuda

# First experiment matrix: 2 losses x 5 seeds = 10 in-domain runs
python run_all_study.py --mode indomain --device cuda

# Summarize runs and compare standard vs. weighted on exactly matched seeds
python analyze_results.py --metric test_micro_f1
```

Training selects CUDA, Apple MPS, or CPU automatically. To force CPU, add
`--device cpu` to either training command. An explicitly requested CUDA device
requires a CUDA-enabled PyTorch installation. Completed run keys are skipped by
default; use `--overwrite` only when you intentionally want to retrain and
replace that run.

The default `run_all_study.py` scope is in-domain only. Cross-domain execution
remains available with `--mode crossdomain` or `--mode all`, but should wait
until the held-out evaluation and category-ontology policy is frozen.

### Saved run artifacts

Each run is stored under
`artifacts/runs/<mode>/<config>/seed_<seed>/`:

```text
manifest.json           # status, configuration, versions, vocabularies, data audit
history.json            # train loss and dev metrics for every epoch
best_model.pt           # best dev-selected model plus label vocabularies
metrics.json            # selected-dev and one-time test metrics
dev_predictions.jsonl   # per-example gold and predicted triplets
test_predictions.jsonl  # per-example gold and predicted triplets
tokenizer/               # tokenizer needed to reload the checkpoint
```

`results.csv` is an atomic flat index with one row per
`(mode, config, seed)`. Detailed per-label and per-domain results remain in the
run's `metrics.json`.

## Current baseline assessment

The repository now provides an executable **single-pair, aligned
explicit-aspect** English BERT baseline. For each non-overlapping aligned
aspect span, the first annotated category/sentiment pair is retained; all
implicit, unaligned, duplicate, additional-pair, and overlapping exclusions
are counted in the manifest. The pipeline trains for the fixed epoch budget, evaluates the
development split after every epoch, restores the checkpoint with the best
end-to-end development micro-F1, evaluates the test split once, and saves the
model, predictions, metrics, and run metadata. It is not a full TASD baseline:
implicit `NULL` aspects and multiple label pairs for one aspect remain outside
the architecture's representational scope.

| Component | Status | Assessment |
|---|---|---|
| Research grid | Partial | Two losses, two data conditions, and five seeds are defined, but several protocol choices are not frozen. |
| Data acquisition & split loading | Implemented | The downloader and configured 21-file mapping are aligned; all files are present locally, match upstream, and load through both split builders. |
| Data audit & validation | Partial | Triplets and sentiment aliases are normalized and retained/excluded counts are saved, but overlap policy, full truncation reporting, and a persistent standalone audit remain unfinished. |
| Target construction | Implemented for restricted scope | One non-overlapping label pair per aligned explicit span becomes a BIO target; every excluded annotation type is counted. Repeated surface terms still use the first text match. |
| Model | Partial | BERT and three heads are implemented, but one category and sentiment prediction per extracted span cannot represent all M-ABSA cases, including multiple triplets for one aspect and implicit aspects. |
| Weighted loss | Implemented | Weight vectors use the full classifier vocabulary, safely handle unobserved/unknown labels, and move to the selected device. |
| Training & validation | Implemented | Both losses use the same end-to-end dev micro-F1 selection rule; the best checkpoint and per-epoch history are saved. |
| Inference & test evaluation | Implemented | BIO is decoded only over real tokens, predicted spans feed the category/sentiment heads, and exact-triplet plus label/domain-stratified metrics and JSONL predictions are saved. |
| Statistical analysis | Implemented | Results are uniquely upserted, duplicate keys fail analysis, and standard/weighted scores are joined on exact matched seeds. |
| Reproducibility | Partial | Manifests record configuration and package versions and tests cover the pipeline, but dependency versions are still broad lower bounds rather than a locked environment. |

The scope must also be stated precisely: this code is an English-only BERT
multi-head baseline. The official M-ABSA baseline is multilingual mT5 with an
extraction paradigm, so results from this repository should not be described
as a reproduction of the official baseline.

## Next steps, in order

### 1. Freeze the research protocol before looking at test results

- Decide whether the task is full TASD or explicitly **explicit-aspect-only
  TASD**. Full TASD requires support for `NULL` aspects and multiple triplets
  associated with the same aspect; filtering them changes the task and must be
  reported with retained/dropped counts.
- Pre-register `HOLD_OUT_DOMAIN`, `MIN_EFFECT_SIZE`, the primary metric, the
  rare-label definition, and whether cross-domain evaluation uses only the
  held-out official test set or all three held-out splits.
- Define one hyperparameter-selection rule using source-domain development
  data only. Apply it identically to standard and weighted loss, and do not
  tune on either test set.
- Treat macro-F1, rare-category recall, and the two modes as planned secondary
  analyses, or pre-register a multiple-comparison policy.

**Gate:** a dated protocol record exists and `config.py` matches it.

### 2. Audit and harden data processing

Already implemented: repeatable missing-file download, the correct 21-file
path mapping, both split builders, the held-out-domain membership guard,
triplet schema validation, tuple conversion, and sentiment alias normalization.

- Add a standalone data-audit command that reports, by domain and split:
  examples, triplets,
  empty annotations, `NULL` aspects, duplicate triplets, repeated aspect text,
  labels, unseen development/test labels, and examples or triplets lost to
  token truncation/alignment.
- Add assertions for file existence, disjoint configured domains, and zero
  train/test sentence overlap where the protocol requires it. The current
  cross-domain smoke check finds eight repeated generic sentence strings, so
  their handling must be documented even though domain membership is disjoint.

**Acquisition/loading gate: passed.** All 21 English files are present, match
upstream, and load successfully. Per-run manifests now save the main retained,
excluded, duplicate, and unseen-label counts. **Remaining validation gate:** a
standalone audit report covers alignment/truncation and the final overlap
policy.

### 3. Harden the training baseline

- Pin and record a tested Python/Torch/Transformers/NumPy environment.
- Keep automated CPU/GPU smoke tests for device selection and checkpoint
  serialization. The current unit/integration suite uses fast CPU substitutes,
  and standard plus weighted real-BERT GPU smoke runs have completed.
- Decide how to model one-to-many triplets. If retaining the current
  architecture, use a representation and loss that can emit multiple category
  and sentiment combinations for a span; otherwise adopt a sequence-to-
  sequence baseline aligned with the upstream TASD formulation.

**Baseline gate: passed.** Standard and weighted smoke runs train, validate,
restore the best epoch, and perform predicted-span evaluation. A tiny overfit
test reaching near-perfect predictions remains a useful stronger diagnostic.

### 4. Extend end-to-end inference and metrics

- Add support for implicit `NULL` aspects and multiple category/sentiment pairs
  for one aspect, or explicitly retain the current restricted task scope.
- Add prediction-only checkpoint evaluation so a saved run can be rescored on
  an approved dataset without retraining.
- Add truncation/alignment ceiling metrics and more one-to-many error-analysis
  tests.

**Explicit-baseline gate: passed.** `run_study.py` saves dev/test predictions
and structured metrics. Checkpoint-only rescoring remains to be implemented.

### 5. Make the comparison statistically safe

- Report mean, standard deviation, every paired seed difference, paired
  confidence intervals, p-values, and the pre-registered practical-effect
  verdict. With only five seeds, interpret both the t-test and bootstrap
  interval cautiously.

**Storage/pairing gate: passed.** Run rows are atomically upserted by
`(mode, config, seed)`, and analysis fails on duplicates and reports unmatched
seeds. A synthetic known-answer test for the complete statistics table remains
open.

### 6. Run experiments without contaminating the test sets

1. Run data and metric unit tests.
2. Run one in-domain seed for both losses as an integration smoke test.
3. Freeze code, configuration, task scope, and the in-domain primary metric.
4. Run the ten-run in-domain matrix.
5. Run the in-domain analysis once, then perform qualitative error analysis
   from saved predictions.
6. Freeze the held-out ontology/evaluation policy before enabling the
   cross-domain matrix.
7. Run the cross-domain matrix and final combined analysis without tuning on
   saved predictions.

After the core study, useful extensions are leave-one-domain-out evaluation
across all seven domains, ablations that weight BIO/category/sentiment losses
separately, and comparison with the official mT5 extraction baseline. These
should not replace the pre-registered core comparison.
