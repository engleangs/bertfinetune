# Quick fine-tuning, test evaluation and final report

**Completed, 6 October 2026:** the user expanded the shortlist to all five
registered seeds. All 15 selected runs have completed training and test
evaluation. Read [the verified final results](final-finetuning-results-2026-10-06.md)
or [the shareable report folder](../results/quick_tuning/README.md). The original
two-seed search procedure below is retained for reproducibility.

To re-check and document the completed five-seed study:

```powershell
.venv/Scripts/python.exe document_quick_tuning_results.py
```

This is the current plan for Thursday's presentation. It replaces the earlier
four-loss/five-seed continuation with a small development search and two-seed
confirmation. The historical Phase 1 study stays separate.

Run from `D:\UoA\bertfinetune` with one training queue at a time:

```powershell
.venv/Scripts/python.exe run_quick_tuning.py --retry-failed --execute
```

Without `--execute`, the script prints a plan and does not train or evaluate.
With `--execute`, it completes these steps automatically:

1. Reuse completed seed-13 development runs at LR `2e-5` for standard CE,
   inverse-frequency CE, mixed weights .25, 1/3 and .5, and focal gamma=2.
   Currently only the explicit-only focal run is missing.
2. Search LR `1e-5`, `2e-5`, `3e-5` for the loss-screen winner. Reuse the
   existing compatible LR runs. Keep the best tested explicit candidate
   on development F1.
3. Train a matching explicit+NULL-vocabulary control with the NULL head off,
   and two NULL-head variants with positive-weight caps `10` and `30`.
   Use NULL loss weight `.5`. Tune thresholds `.02, .05, .1, .2, .3, .4,
   .5, .6, .7, .8` on development NULL F1. Choose the NULL configuration
   by combined development F1. Compare explicit and NULL scores separately.
4. Check the explicit winner, vocabulary control and NULL winner on seeds
   `13` and `42`, reusing completed checkpoints. Select the better NULL-off
   candidate by mean explicit development F1. Save configurations, checkpoints,
   epochs and thresholds in a freeze record.
5. Evaluate only the frozen shortlist on the official tests. Generate the
   final comparison report and figures from saved metrics.

Expected extra training with the current mixture winner: six models, roughly
two hours at the observed 21 minutes/model. A different focal/LR winner can
require additional runs. This estimate excludes test evaluation and startup.
It is a small search for the best tested parameters, not a global optimization
or a five-seed significance study. No improvement in NULL F1 is guaranteed.

## Outputs

- `artifacts/quick_tuning/selection.json`: frozen settings and full development
  search evidence; saved before test evaluation.
- `artifacts/quick_tuning/report.md`: development/test scores, selected parameters,
  error taxonomy, component F1, rare/unseen categories, hardest domains and NULL
  recovery counts.
- `artifacts/quick_tuning/comparison.png` and `.svg`: comparison charts.
- `artifacts/quick_tuning/results.json`: report data, including each seed.
- `artifacts/experiments_v2/development_results.csv`: retains earlier logical
  results and adds new completed runs. Reruns do not duplicate the same identity.

The mixed loss is `(w1 * CE + w2 * inverse_frequency_CE) / (w1 + w2)`.
The screen includes the professor's `1/.5` and `.5/.5` examples as normalized
mixtures. Training frequencies determine class weights and vocabulary; development
data alone selects parameters, checkpoints and thresholds. Keep all four sentiment
classes, including Conflict, for this search.

Locked budget: pinned BERT-base-uncased; five epochs; batch 16; max length 128;
weight decay .01; warmup .10; gradient clipping 1.0. All candidates choose
checkpoints by explicit development F1. NULL threshold selection is separate.
Combined F1 counts the same explicit+NULL gold for both head settings, so use
it to compare total annotation coverage. Component F1 is diagnostic and does
not replace exact-triplet F1.

Historical test sets were already inspected. Describe the final scores as
exploratory test evaluation and do not tune parameters from those scores.

## Optional: stop after development

```powershell
.venv/Scripts/python.exe run_quick_tuning.py --step search --retry-failed --execute
.venv/Scripts/python.exe run_quick_tuning.py --step final --execute
```

The second command evaluates the frozen checkpoints and creates the report.
To regenerate the report without training or evaluation:

```powershell
.venv/Scripts/python.exe run_quick_tuning.py --step report --execute
```

Repeating the main command reuses the frozen selection and completed evaluations.
A changed seed count needs a separate `--study-output` directory. Interrupted
training can restart from scratch with `--retry-failed`; this trainer does not
save optimizer state for epoch resumption. A stale `running` manifest requires
checking that its process stopped before marking that particular run failed.
