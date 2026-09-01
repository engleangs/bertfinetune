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
│   ├── train.py              # training loop (now seed-parameterized)
│   ├── evaluate.py           # complete-triplet F1, precision/recall by label, rare-label recall
│   ├── stats.py               # paired significance test + bootstrap CI + effect-size check
│   └── utils.py                # seeding
├── run_study.py            # ONE (mode, config, seed) run
├── run_all_study.py        # all 2 x 2 x len(SEEDS) runs
└── analyze_results.py      # mean/std, significance, effect-size verdict — the final table
```

## Data format & paths

Same M-ABSA `sentence####[(term, category, sentiment), ...]` format as
before. The upstream repository uses
`data/<domain>/<language>/{train,dev,test}.txt`; for example,
`data/coursera/en/train.txt`. The current `config.DOMAIN_FILES` values do not
match that layout and must be corrected after cloning
https://github.com/swaggy66/M-ABSA.

## Running it

```bash
pip install -r requirements.txt
python download_data.py                                            # download missing English M-ABSA files
python run_study.py --mode indomain --config standard --seed 42   # one run
python run_all_study.py                                            # all 20 runs
python analyze_results.py                                          # final report
```

Training selects CUDA, Apple MPS, or CPU automatically. To force CPU, add
`--device cpu` to either training command. An explicitly requested CUDA device
requires a CUDA-enabled PyTorch installation.

## Current baseline assessment

This repository is a useful experiment scaffold, but it is not yet an
executable or research-valid TASD baseline. It currently reaches model
training only; `run_study.py` writes time and parameter count, not predictions
or task metrics.

| Component | Status | Main issue |
|---|---|---|
| Research grid | Partial | Two losses, two data conditions, and five seeds are defined, but several protocol choices are not frozen. |
| Data loading | Blocked | `DOMAIN_FILES` does not match the upstream directory layout, and no local M-ABSA data is present. |
| Target construction | Partial | Explicit aspects become BIO spans, but `NULL`/implicit aspects are silently discarded, repeated terms use only the first text match, and truncation losses are not reported. |
| Model | Partial | BERT and three heads are implemented, but one category and sentiment prediction per extracted span cannot represent all M-ABSA cases, including multiple triplets for one aspect and implicit aspects. |
| Weighted loss | Implemented | Class-weight tensors move with the loss modules to the selected training device. |
| Training | Partial | Device selection is portable, but the validation loader is never used and there is no checkpoint selection or early stopping. |
| Inference and evaluation | Blocked | No end-to-end prediction path exists. The metric helpers also expect hashable triplet tuples, while `parse_line` currently returns lists. |
| Statistical analysis | Partial | Summary code exists, but `results.csv` has no metric columns. Pairing is based on separately sorted rows rather than an exact seed join, and reruns append duplicates. |
| Reproducibility | Blocked | Dependency lower bounds are too broad. The current environment resolves Torch 2.2.2, Transformers 5.16.1, and NumPy 2.5.2, which are not mutually usable here. |

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

### 2. Make data loading correct and auditable

- Clone M-ABSA, set `DATA_DIR`, and change each split path to
  `<domain>/<language>/<split>.txt`.
- Convert parsed triplets to tuples and validate every input row.
- Add a data audit that reports, by domain and split: examples, triplets,
  empty annotations, `NULL` aspects, duplicate triplets, repeated aspect text,
  labels, unseen development/test labels, and examples or triplets lost to
  token truncation/alignment.
- Add assertions for file existence, disjoint configured domains, and zero
  train/test sentence overlap where the protocol requires it.

**Gate:** all 21 English files load, audit totals are saved, and a small
dataset test verifies parsing, BIO alignment, truncation, and the cross-domain
leakage guard.

### 3. Repair the training baseline

- Pin and record a tested Python/Torch/Transformers/NumPy environment.
- Keep automated CPU/GPU smoke tests for device selection and weighted-loss
  tensor placement.
- Use the development set for checkpoint selection and return the best model,
  not merely the final epoch. Record train/dev losses, epoch, seed, runtime,
  package versions, and the complete configuration.
- Decide how to model one-to-many triplets. If retaining the current
  architecture, use a representation and loss that can emit multiple category
  and sentiment combinations for a span; otherwise adopt a sequence-to-
  sequence baseline aligned with the upstream TASD formulation.

**Gate:** a tiny overfit test reaches near-perfect predictions, and one
standard plus one weighted smoke run completes on the target device.

### 4. Implement end-to-end inference and metrics

- Build a test dataset and decode BIO predictions only over real,
  non-special tokens. Recover aspect text from tokenizer offsets, then run the
  category and sentiment heads on **predicted**, never gold, spans.
- Save per-example gold and predicted triplets to JSONL so every aggregate
  score is reproducible and errors can be inspected.
- Write micro precision/recall/F1, a clearly defined macro-F1, per-category
  and per-sentiment precision/recall/F1/support, rare-category recall,
  per-domain scores, and error counts to a structured result file.
- Unit-test exact matches, duplicate triplets, no-prediction cases, malformed
  BIO sequences, implicit aspects, and wrong aspect/category/sentiment cases.

**Gate:** `run_study.py` produces predictions and metrics that can be
recomputed from its saved JSONL without retraining.

### 5. Make the comparison statistically safe

- Store one uniquely keyed row per `(mode, config, seed)` and refuse or
  explicitly overwrite duplicates.
- In `analyze_results.py`, inner-join standard and weighted runs on exact
  `(mode, seed)` keys and report missing pairs instead of relying on row order.
- Report mean, standard deviation, every paired seed difference, paired
  confidence intervals, p-values, and the pre-registered practical-effect
  verdict. With only five seeds, interpret both the t-test and bootstrap
  interval cautiously.

**Gate:** analysis fails loudly for duplicated or unmatched runs and produces
the complete planned table from synthetic known-answer results.

### 6. Run experiments without contaminating the test sets

1. Run data and metric unit tests.
2. Run the tiny overfit test.
3. Run one seed for both losses in both modes as an integration smoke test.
4. Freeze code and configuration.
5. Run all 20 planned experiments.
6. Run the final analysis once, then perform qualitative error analysis from
   saved predictions.

After the core study, useful extensions are leave-one-domain-out evaluation
across all seven domains, ablations that weight BIO/category/sentiment losses
separately, and comparison with the official mT5 extraction baseline. These
should not replace the pre-registered core comparison.
