# Small development experiments before the full run

Use `run_experiments.py` for the corrected pipeline. The original runners and
80 historical runs remain version-1 evidence. This extension implements the
training corrections specified in [the 4 October protocol](protocol-amendment-2026-10-04.md).
Its research status is exploratory because the historical tests have already
been inspected. Improving development F1 is a goal, not a promised result.

## First run: check the flags and two seeds

Run commands from `D:\UoA\bertfinetune` with the existing virtual environment.
Plain `python` currently selects a different interpreter without PyTorch.

```powershell
.\.venv\Scripts\python.exe run_experiments.py --suite all --first-seeds 2 --dry-run
```

This prints 26 runs: 13 settings with matched seeds 13 and 42. No model or
dataset is loaded. A short BERT integration check is separate from the
research pilot:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --suite null --first-seeds 2 --smoke --batch-size 4 --max-len 64 --execute
```

The smoke check uses the first 64 source training and 32 source development
sentences, one epoch, and a separate output subdirectory. It verifies loading,
training, selection and saved artifacts. Its F1 is not a research result.
`--train-limit`, `--dev-limit` and `--epochs` can change that smoke budget.
Keep batch size 16 and maximum length 128 for research unless registering a
separate budget experiment. CUDA is detected automatically; `--device cpu`
forces CPU. Models are loaded from the pinned local cache by default;
`--online` permits downloading that same revision.

## Research pilot: two seeds, official train/dev, five epochs

Start with the loss sweep, then the head ablation and the NULL comparison.
Each suite is independently selectable so the work can stay small.

```powershell
.\.venv\Scripts\python.exe run_experiments.py --suite loss --first-seeds 2 --execute
.\.venv\Scripts\python.exe run_experiments.py --suite heads --first-seeds 2 --execute
.\.venv\Scripts\python.exe run_experiments.py --suite null --first-seeds 2 --execute
```

Alternatively, `--suite all --first-seeds 2 --execute` runs the same 26 unique
configurations in one command. Standard/weighted runs shared between suites
are skipped once complete. A recipe can be isolated with `--recipes`.
All pilot and full-stage training uses source train/dev only. Test files are
opened exclusively in the explicit `final` stage.

| Suite | Settings | Why run it? |
|---|---|---|
| `baseline` | standard CE, inverse-frequency weighted CE | Corrected controls |
| `loss` | controls; mixed alpha=.25, 1/3, .5, .75; focal gamma=1, 2 | Professor's loss proposals |
| `heads` | controls; weighting on BIO, category or sentiment alone | Locate the precision/recall tradeoff |
| `null` | CE with NULL head off/on and the same explicit+NULL training vocabulary | Measure implicit target prediction separately |
| `all` | All 13 unique settings | Complete initial candidate pool |
| `custom` | One configurable setting | Follow-up sensitivity experiments |

Prefer mixed CE or category-only weighting as candidates to inspect first:
the old all-head weighting increased recall while sharply reducing precision.
This is motivation for an ablation, not evidence those candidates improve F1.
Focal loss remains an empirical alternative. Keep LR=2e-5 and five epochs
fixed during the initial loss comparison; inspect the saved learning curves
before registering a further LR or epoch sweep.

## NULL extension and experiment flags

The explicit BIO head keeps ignoring NULL spans. The extension preserves
NULL annotations separately and trains a shared-BERT `[CLS]` head to predict
multiple `(category, sentiment)` pairs using sigmoid and binary cross-entropy.
Sentences with no NULL annotation provide all-zero negative targets.
NULL-only labels enter the source-training vocabulary for this extension;
the `null` suite gives its control the same vocabulary to isolate the head.
Unseen target categories remain false negatives.

```powershell
.\.venv\Scripts\python.exe run_experiments.py --suite custom --loss mixed --ce-weight 1 --weighted-ce-weight 0.5 --loss-heads category --null-head --null-loss-weight 0.5 --first-seeds 2 --execute
```

| Flags | Meaning / default |
|---|---|
| `--null-head`, `--no-null-head` | Enable/disable implicit prediction; default depends on recipe |
| `--loss standard\|weighted\|mixed\|focal` | Explicit-head loss family |
| `--loss-heads bio category sentiment` | Heads receiving the chosen loss; remaining heads use CE |
| `--ce-weight`, `--weighted-ce-weight` | Normalized mixture `(w1*CE+w2*WCE)/(w1+w2)`; custom default 1 and .5 |
| `--focal-gamma` | Unweighted focal exponent; default 2 |
| `--bio-loss-weight`, `--category-loss-weight`, `--sentiment-loss-weight` | Outer head multipliers; each defaults to 1 |
| `--null-loss-weight` | NULL BCE multiplier; default .5 |
| `--null-pos-weight-cap` | Cap train-only negative/positive BCE weights; default 1 means no positive weighting |
| `--null-threshold`, `--null-thresholds .2 .3 .4 .5 .6 .7 .8` | Reference and development-only global threshold search |
| `--selection-metric explicit\|combined` | Checkpoint objective; explicit is primary; combined is a separate NULL experiment |
| `--vocabulary-scope auto\|explicit\|explicit-null` | Vocabulary source; auto includes NULL when its head is enabled |
| `--drop-conflict` | Separate three-class task; removes individual annotations before vocab/weights, preserves other targets |
| `--lr`, `--epochs`, `--batch-size`, `--max-len`, `--weight-decay` | Training budget; changes create distinct trial IDs |
| `--warmup-ratio`, `--max-grad-norm` | Defaults .10 and 1.0; preserve these for the corrected protocol |
| `--first-seeds N`, `--seeds ...` | Matched registered prefix or a declared exploratory seed list |
| `--mode crossdomain --held-out-domains restaurant` | Fold source train/dev exclude target domain; test only at final stage |
| `--output PATH`, `--retry-failed` | Separate artifacts and safe retry of a failed run with identical provenance |

Changing multiple flags together prevents attribution to one change. Keep
Conflict removal, NULL architecture, loss weighting and training-budget
experiments separately labelled. `--help` lists all flags.

## Scores to inspect and what to lock

Use development exact explicit-triplet micro-F1 as the primary selection
metric. Compare seeds on the same vocabulary/task and training budget.
Inspect precision and recall together: recall gains alone can reduce F1.
The runner saves these additional views:

- Explicit, NULL-only and combined exact-triplet scores, each with TP/FP/FN,
  precision, recall and F1. Combined scope is eligible explicit plus NULL;
  it excludes ambiguous, conflicting and overlapping explicit annotations.
- Term surface, exact character boundary, term+category and term+sentiment
  F1; fixed-source-category macro-F1; known-category coverage/F1.
- First failed component taxonomy, spurious predictions, conditional label
  diagnostics, sentiment confusion, training-frequency bands and domain scores.
- Per-epoch development scores and loss components, saved best checkpoint,
  development predictions with offsets, training/dev eligibility audits.

Splitting metrics helps explain the hardest component; it does not replace
full-triplet F1. Exact boundary F1 checks occurrence positions, while term
surface F1 matches strings. Component F1 values do not add up to triplet F1.
NULL diagnostics use category/sentiment outcomes and never invent a term error.
When NULL enters the vocabulary, rarity counts include eligible source NULL;
that scope is recorded and differs from explicit-only rarity counts.

Checkpoint ties use lower unweighted development loss (sum of available
per-head means), then earlier epoch. The NULL threshold is chosen by NULL
development F1 for each checkpoint; ties prefer the reference .5, then the
smaller threshold. With no development NULL gold, use the reference unchanged.
When checkpoint selection is explicit, threshold tuning cannot change which
epoch wins on F1. The frozen checkpoint carries the threshold into testing.

Warmup is `floor(.10 * optimizer_steps)`. Clipping occurs after backward and
before each optimizer step. Eligibility excludes ambiguous occurrences,
the entire span with distinct label pairs, and all members of overlapping
spans regardless of annotation order. Train-only vocabularies and weights
are rebuilt after filtering; unknown gold remains in evaluation. Duplicate
triplets are deduplicated within a sentence. Macro-F1 uses the frozen source
category list and zero for unsupported denominator categories.

Manifests record the full configuration, seeds, source data hashes, code
hashes, git commit/dirty state, device, package inventory and pinned
model/tokenizer revision. Package locks are written under `environments/`.
These locks describe this installed environment, including its CUDA wheel;
recreating it also needs a compatible Python/CUDA wheel source. A changed
code/data/environment fingerprint requires a new output root. Freeze the
code before promoting pilot runs; never reuse legacy CSV keys.

## Expand a shortlist, then evaluate frozen checkpoints

After the two-seed pilot, select on development data and retain the controls.
For example, if `mixed_033` is worth continuing:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --stage full --suite loss --recipes standard weighted mixed_033 --execute
```

This uses all five matched seeds, skips completed seeds 13/42 with identical
provenance, and trains 123/2024/777. It still does not load test files.
Reproduce exactly the same model/loss/budget flags and output root when
promoting a custom candidate.

Once the shortlist and checkpoints are frozen:

```powershell
.\.venv\Scripts\python.exe run_experiments.py --stage final --suite loss --recipes standard weighted mixed_033 --execute
```

The final stage refuses to train missing runs, verifies checkpoint hashes,
uses saved thresholds, and evaluates each completed checkpoint once.
Previously inspected historical test sets remain exploratory evaluation.
Five seeds support mean/standard-deviation and paired comparisons; two-seed
pilots are screening evidence, not significance claims. Preserve the .02
absolute F1 effect threshold when interpreting improvements.

Default outputs live in `artifacts/experiments_v2/`. Open
`development_results.csv` for per-seed components/taxonomy and
`development_summary.json` for seed means and sample standard deviations.
Do not compare different task/vocabulary/smoke scopes as if they were one study.

## Validation completed on 5 October

All 59 tests pass; `pip check` finds no broken requirements. All 13 recipes
completed real cached-BERT GPU smoke training with seeds 13 and 42 (26 runs),
using 64 training/32 development sentences and one epoch. No test files were
loaded. Checkpoints, offsets, taxonomy, fixed macro universe and warmup were
verified. These checks establish execution correctness; the tiny scores do
not establish a performance improvement or select a loss family.

Validation outputs are in `artifacts/experiments_v2_smoke_20261005/`:
`test_validation.json`, `smoke_validation.json`, `development_results.csv`
and per-run artifacts. The next performance step is the official train/dev
loss pilot above with two seeds and the unchanged five-epoch budget.

## Visualize the pilot and current hyperparameters

The standalone reporting command reads saved artifacts and works while the
pilot is still running:

```powershell
.\.venv\Scripts\python.exe visualize_experiments.py
```

Open `artifacts/experiment_comparison/report.html`. It contains filters for
cohort, score and seed; the progress grid; current training settings; paired
changes against CE; learning curves; component F1; error taxonomy; domain
scores; and the matched NULL comparison. Figures export as PNG/SVG, and
`hyperparameters.csv` records each run's actual parameters and selected epoch.

Keep the report regenerating as training produces new artifacts:

```powershell
.\.venv\Scripts\python.exe visualize_experiments.py --watch --interval 30
```

Reload the HTML page to see the latest generated snapshot. Unfinished runs
show their best fully evaluated epoch as provisional. They are excluded from
paired improvement verdicts. A recipe's finished seed mean never pools an
unfinished seed. Missing runs remain pending rather than receiving zero F1.
Comparisons require the same data, vocabulary, task, budget and completed
seeds; NULL uses the vocabulary-matched control. The +.02 effect target is
shown as +2 percentage points. Two-seed gains are screening evidence.

Other inputs use `--input PATH --output PATH`. Smoke results are excluded by
default; `--include-smoke` displays them in separate cohorts with no performance
claim. For transfer use `--mode crossdomain --held-out-domain restaurant`.
After extending to five seeds, add
`--expected-seeds 13 42 123 2024 777` to the visualization command.

Reporting helpers live in `experiment_reporting/`, separate from the frozen
trainer. Generating the report preserves the training source hashes used for
checkpoint promotion.
