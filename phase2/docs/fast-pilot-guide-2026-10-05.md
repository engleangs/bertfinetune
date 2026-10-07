# Faster staged fine-tuning pilot

The full command `run_experiments.py --suite all --first-seeds 2 --execute`
trains 13 settings with two seeds and five epochs: 26 models, 130 model-epochs.
The four completed runs currently take approximately 21 minutes each on the
RTX 3050, estimated from audit-to-history file times. The full grid is roughly
nine hours at that observed pace. This estimate includes training/evaluation
and checkpoint saving; it is not a GPU benchmark or guaranteed runtime.

Reduce the candidate grid before changing the training budget. Standard CE
seed 13 improves from development explicit F1 .2800 at epoch 3 to .3082 at
epoch 5; seed 42 also continues improving through epoch 5. Stopping every
candidate at epoch 3 could rank slower-learning configurations unfairly.

`run_fast_pilot.py` is a separate launcher around the existing trainer. It
does not modify the trainer, source hashes or experiment definitions. It
prints a plan by default and never loads a model while planning. With
`--execute`, the original runner verifies provenance before skipping completed
runs. It refuses to launch when the requested output or original v2 output
records another running job.
Stop the previous queue before launching a replacement. Ctrl+C records the
active run as failed; `--retry-failed` restarts it from scratch. The current
checkpoint lacks optimizer/scheduler state for continuing an interrupted run.
Prefer letting the current model finish before switching queues.

## 1. Screen a small loss comparison

```powershell
python run_fast_pilot.py
.venv/Scripts/python.exe run_fast_pilot.py --execute
```

This screens six settings with seed 13: CE, weighted CE, mixed alpha .25,
1/3 and .5, and focal gamma 2. It uses the official source train/dev data,
five epochs and the original training settings. It tests both mixture settings
suggested by the professor and one focal configuration, with completed
compatible runs reused. The grid has 30 model-epochs versus 130: **77% less
scheduled training**, before reuse. Actual runtime savings vary by recipe.

At the time of this review, CE/weighted seed 13 are complete and mixed .25
seed 13 is recorded as running. After that model completes, three new models
remain in this screen: mixed 1/3, mixed .5 and focal 2, roughly another hour
at the observed pace. Run the plan again for current status.

For an even smaller selection, controls are always included:

```powershell
python run_fast_pilot.py --recipes mixed_033 focal_2
```

Heads, the second focal gamma, remaining mixture coefficients, NULL and
Conflict removal are deferred. These can still be run through the original
runner after evidence justifies the next comparison. The six-setting screen
does not claim to exhaust the professor's proposed parameter sweep.

## 2. Tune learning rate only for the selected recipe

Select a candidate from **completed development runs**, then hold its loss
and all other settings fixed while trying 1e-5, 2e-5 and 3e-5 with seed 13.
For example, if `mixed_033` earns the shortlist:

```powershell
python run_fast_pilot.py --step lr --winner mixed_033
.venv/Scripts/python.exe run_fast_pilot.py --step lr --winner mixed_033 --execute
```

The existing compatible 2e-5 run is reused; only the other two rates are new.
Choose using explicit-triplet development micro-F1, inspecting precision,
recall, spurious predictions and taxonomy alongside it. LR=2e-5 remains a
reference, not an assertion that it is optimal. This is an exploratory range;
do not take the Cartesian product of LR, loss, batch size, epochs and seeds.

The screen can also choose CE if the alternatives do not improve it. A
one-seed winner is provisional; these development comparisons are selection
evidence, not independent confirmation of a performance improvement.

## 3. Check the shortlist with a second seed

```powershell
python run_fast_pilot.py --step confirm --winner mixed_033 --lr 2e-5
.venv/Scripts/python.exe run_fast_pilot.py --step confirm --winner mixed_033 --lr 2e-5 --execute
```

The launcher compares standard, weighted and the selected recipe with seeds
13/42 at the **same chosen LR and five-epoch budget**. If LR changes, matched
CE/weighted controls at that rate are needed; old 2e-5 controls are not a
comparison that isolates loss at another rate. At unchanged settings, the
four existing control runs and the candidate's seed-13 run can be reused.

`confirm` names this second-seed development check. It is still exploratory
selection, reads no test data, and does not imply untouched confirmatory
evaluation. After that check, use `--first-seeds 5` for the final shortlist.
Only the original runner's explicit `--stage final` evaluates frozen tests.

## 4. Keep the NULL experiment separate

```powershell
python run_fast_pilot.py --step null
.venv/Scripts/python.exe run_fast_pilot.py --step null --execute
```

This tests the matched-vocabulary CE control and NULL head with seed 13.
Inspect NULL and combined scores alongside explicit F1. NULL F1=0 is expected
for recipes with no NULL head; it does not indicate a broken explicit model.
Do not select the explicit loss winner by these unsupported NULL scores.

## Optional shorter screening and hardware work

`--epochs 3` uses a distinct default output, `artifacts/experiments_screen_e3/`.
It is a separate screening budget, not a continuation of the five-epoch runs.
The scheduler/warmup change with the budget, and shortlisted candidates must
be rerun at five epochs. It saves epoch work but cannot reuse the current
five-epoch checkpoints. The default therefore keeps five epochs and fewer
settings. Keep batch=16, max_len=128, weight_decay=.01, warmup=.10 and
clipping=1.0 fixed while comparing losses/LR.

The current trainer uses full precision, pads every example to 128 tokens,
and tokenizes examples again on dataset access. Future per-run speed work
could benchmark cached tokenization, dynamic padding and automatic mixed
precision. AMP needs autocast and, for FP16, gradient scaling with unscaling
before gradient clipping. These are not implemented by this launcher; a
separate training version and matched controls are needed before mixing their
results with the existing runs. See [PyTorch's AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

Use [the experiment comparison report](experiment-guide-2026-10-05.md#visualize-the-pilot-and-current-hyperparameters)
to inspect the saved settings and learning curves. Reports distinguish
training-budget/LR cohorts; compare candidates with compatible controls.
The [4 October reporting amendment](protocol-amendment-2026-10-04.md) remains
the protocol for taxonomy, primary F1, four sentiment classes and inspected
tests. No learning-rate choice or loss is guaranteed to increase test F1.
