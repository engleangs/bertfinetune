# Finish training, compare test results, produce one report

The NULL runner already uses the quick-tuning winner's explicit settings: `.75 CE + .25 inverse-frequency CE`, LR `3e-5`, batch 16, max length 128, weight decay `.01`, warmup `.10`, clipping `1`. Both heads are trained jointly in the new NULL experiments. Combine the studies' reports; no extra model-weight merge is needed.

The completed quick study has 15 final test evaluations across three variants and five seeds. Preserve/reuse them. The current NULL queue completes its registered screen and two-seed confirmation before the continuation runs.

## Requested overnight continuation

```powershell
.venv/Scripts/python.exe finalize_finetuning_report.py --finish-null --first-seeds 5 --ablation-tests --wait-for-null --execute
```

This single command:

1. Waits for the current NULL queue to exit. It starts no overlapping GPU training.
2. Freezes every registered trained screen checkpoint and its development-selected threshold: 11 unique quick-tuning candidates and normally 10 NULL fast-track candidates. Reused LR/screen checkpoints are deduplicated. Both screen sets are frozen before new test inference.
3. Completes the selected NULL candidate and matching NULL-off control on five seeds. Existing seed 13/42 confirmation runs skip after verification; only missing seeds need training. A two-to-five-seed extension adds 30 model-epochs.
4. Evaluates the frozen five-seed NULL finalists on test data with fixed thresholds.
5. Evaluates every registered quick/NULL screen checkpoint on test data as a separate exploratory ablation. This performs inference only, and reuses completed evaluations.
6. Builds `final_presentation_results/report.md`, plots, full parameter CSV, individual dev/test scores, frozen metadata and machine-readable evidence. No model weights go into this shareable folder.

The continuation calls the existing confirmation/evaluation functions. It preserves training-source fingerprints and stops if the queue failed before completing its search, provenance changed, or a freeze/checkpoint is inconsistent. No fresh parameter grid is launched. Rerunning the same command reuses completed work; true failed training needs `--retry-failed` after resolving its cause.

You asked for all trained screening candidates to have test comparisons. These test results are a fixed exploratory ablation; they do not change the development-selected finalist settings. NULL screens use three-epoch budgets, quick screens use their recorded five-epoch budgets, and screen tests normally have only seed 13. Those rows remain separate from the five-seed final comparisons.

## Shortest alternative: no additional training

If the deadline requires using the current NULL cohort without extending it:

```powershell
.venv/Scripts/python.exe finalize_finetuning_report.py --evaluate-null --ablation-tests --wait-for-null --execute
```

This uses the existing NULL freeze (normally two seeds), evaluates all screens, and combines it with the completed five-seed quick study. The report states each cohort size; it makes no five-seed claim for the two-seed NULL result.

## Reporting only

Run this at any time to read saved results and refresh the combined snapshot:

```powershell
.venv/Scripts/python.exe finalize_finetuning_report.py
```

It loads no model or raw test data and starts no training/evaluation. Results still pending are marked as pending.

| Output | Purpose |
|---|---|
| `final_presentation_results/report.md` | Main final results, separate screen-test ablations, Phase 1 comparison, parameters, taxonomy and rare/domain findings |
| `test_comparison.png` | Finalist mean F1 and sample SD, with seed counts and separate study labels |
| `screen_test_comparison.png` | Separate fixed-screen test comparison |
| `parameters.csv` | Full flags/configurations for every registered search and finalist run |
| `seed_results.csv` | Individual dev/test precision, recall and F1; screen/finalist type is explicit |
| `results.json` | Full saved metrics, diagnostics, configurations and completion status |
| `ablation_freezes/` | Immutable screen checkpoint/threshold lists committed before test inference |
| `ablation_test_results.csv` | Append-only screen evaluation ledger; reused results deduplicate |
| `quick_selection.json`, `null_selection.json` | Copies of the finalist freezes |

The combined report CSVs are refreshable exports. Original study development/epoch/test logs and the ablation evaluation ledger keep their append history.

For the presentation, keep the established matched-gold Phase 1 gain as the main historical finding. Describe the new NULL result against its matching NULL-off control, with explicit/NULL/combined F1, NULL precision/recall and paired seed differences. Different input budgets, precision and selection objectives make cross-study differences descriptive. A nonzero NULL F1 alone does not demonstrate a joint improvement. Historical test inspection makes the whole comparison exploratory.

## Background monitoring

The assistant-launched continuation records its PID, command, stdout and stderr paths in `artifacts/final_report_flow/latest.json`. Read its logs to see whether it is waiting, training, evaluating, complete or stopped with an error. The report's `results.json` also records finalist and screen-ablation completion independently. A launched background process is not itself evidence that the final evaluations have completed.

Focused validation covers waiting without changing the current queue, finalist-only continuation, deduplicated immutable ablation freezes, separate seed/budget cohorts, and actual CPU checkpoint inference that retains the frozen threshold without starting training.
