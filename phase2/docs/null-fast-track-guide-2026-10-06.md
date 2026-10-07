# NULL fast track: run tonight, report tomorrow

Techniques 1–5 from the [literature review](literature-review-null-and-finetuning-2026-10-06.md) are implemented in `run_null_fast_track.py` and the separate `null_experiments/` package. Existing training files, model checkpoints and final results are preserved. New improvements still need to be measured.

## Recommended overnight command

From `D:\UoA\bertfinetune`, run:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --first-seeds 2 --execute
```

This runs the staged development search, confirms the best NULL candidate and a NULL-off control on seeds 13 and 42, and freezes their checkpoints and thresholds. It stops before test evaluation. The default output is `artifacts/null_fast_track`.

The runner uses GPU mixed precision, tokenizes each input variant once, and reuses the cache across trials. Completed runs skip after provenance/checkpoint verification. An interruption resumes from the last completed epoch; the unfinished epoch is replayed. Run the same command again to continue. Use `--retry-failed` only for a recorded failure after resolving its cause.

## What the search does

| Stage | Candidates | Budget | Decision |
|---|---|---|---|
| 1: historical calibration | Saved NULL checkpoint, seed 13; fine dev threshold grid | Inference only | Diagnostic; does not change the old frozen study |
| 2–3: NULL loss | NULL off, weighted BCE, binary focal, ASL | 4 × 3 epochs, seed 13 | Best NULL loss by combined dev F1 |
| 4: implicit representation | Dedicated `[IA]`; `[IA]` plus category-conditioned attention | 2 × 3 epochs, seed 13 | Compare against the existing CLS candidates |
| 5: negative treatment | Random sampling control; adaptive hard-negative sampling | 2 × 3 epochs, seed 13 | Compare against all-cell candidates |
| Small parameter check | Winning method with a different NULL outer weight; one method-specific setting | 2 × 3 epochs, seed 13 | Select the short-budget winner |
| Confirmation | Best NULL candidate + NULL-off control | 2 methods × 2 seeds × 5 epochs | Freeze settings and per-seed dev-selected checkpoints/thresholds |

The default maximum is **50 model-epochs**, plus calibration/loading/checkpoint I/O. It is 14 training jobs: ten short candidates and four confirmation runs. Actual time depends on the GPU and architecture; the report records observed epoch timing. Historical 4.2-minute FP32 epochs would imply approximately 3.5 hours of epoch computation for this budget, before I/O. That is a reference calculation, not an AMP timing prediction.

This is an adaptive search: later stages use the best earlier setting. It covers all five techniques without a full Cartesian product. It can miss interactions or methods that improve later in training. To confirm two NULL candidates instead of one, add `--confirm-top 2` (60 model-epochs with two seeds). An optional `--tune-lr` adds one 3-epoch LR `2e-5` trial; the default retains the established `3e-5`.

For the shortest first pass that still covers techniques 1–5:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --first-seeds 1 --no-tune --output artifacts/null_fast_track_quick --execute
```

That is 34 model-epochs: eight 3-epoch screens plus two 5-epoch confirmations. Its single-seed result is preliminary. Use a different output folder when changing the search plan, precision, input budget or training settings.

## Extend the frozen shortlist to five seeds

After the default two-seed run finishes, continue only the confirmation stage:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step confirm --first-seeds 5 --execute
```

The screens do not run again. Verified seed 13/42 confirmation checkpoints skip; seeds 123, 2024 and 777 are added for both arms. This adds **30 model-epochs**. The previous two-seed freeze remains in `selections/`; `selection.json` points to the latest completed cohort. If using the shorter first-pass folder, include `--output artifacts/null_fast_track_quick` in subsequent commands.

## Report and visualization

The runner refreshes the report after every completed trial and stage. To refresh it manually, including while training:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step report
```

Open `artifacts/null_fast_track/report.md`. The small Git-ready copy is `null_fast_track_reports/null_fast_track/report.md`.

| File | Purpose |
|---|---|
| `report.md`, `comparison.png` | Technique completion counts, run statuses, explicit/NULL/combined F1, NULL precision/recall, parameter tables and comparison chart |
| `report_data.json` | Full configurations, development curves, slices, domain results and primary error taxonomy |
| `development_results.csv` | Appended completed-run rows; verified reruns do not duplicate them |
| `epoch_results.csv` | Appended epoch objectives, dev scores, thresholds, timing and skipped AMP updates |
| `events.jsonl` | Run/stage/completion/failure history |
| `search_plan.json`, `search_state.json` | Fixed search settings and stage-by-stage candidate selection |
| `selection.json`, `selections/` | Current freeze and immutable earlier seed-cohort freezes |
| `research/<trial_id>/seed_<seed>/` | Manifest, audit, history, selected checkpoint, dev predictions, probabilities and taxonomy |

Reports are refreshable summaries; CSV/event history appends. Research and smoke results are kept separate. The shareable folder contains report/chart/metadata only; model files and prediction/probability exports remain under Git-ignored `artifacts/`.

Read the **matched confirmation comparison** to decide whether NULL improves the joint result. A higher NULL F1 with a lower combined F1 is a trade-off, not an overall improvement. Do not pool the 3-epoch screens with the 5-epoch confirmations. The report shows means, sample SD and paired combined-F1 differences once matched confirmation seeds exist.

## Protocol fixed before the new search

- **Task:** in-domain M-ABSA target/category/sentiment detection; eligible explicit gold plus deduplicated NULL gold. Existing ambiguity/conflict/overlap exclusions for explicit annotations remain. All four sentiment labels, including Conflict, remain in the output space. This follow-up does not establish LODO improvement.
- **Selection:** combined development exact-triplet micro F1 for method, epoch and threshold selection. Threshold ties prefer NULL F1, then the higher threshold. Epoch ties prefer NULL F1, lower unweighted loss, then the earlier epoch. Candidate ties use explicit F1, unweighted loss and a deterministic config tie-break. Shortlist confirmation uses mean primary development F1 across the requested seeds.
- **NULL abstention:** threshold 1 explicitly emits no NULL predictions. It is always included. If it wins, report that the head did not improve the joint operating point; do not force nonzero output.
- **Training only:** vocabulary, class frequencies, positive weights, pair rarity and hard-negative targets come from training annotations. Unknown dev/test labels remain false negatives.
- **Same new-study input budget:** every arm reserves an extra slot after CLS, including the NULL-off and CLS controls. The inactive slot is masked; `[IA]` variants activate it. At max length 128, text has up to 125 token slots. All arms add the same embedding row. This differs from the old 126-text-token protocol, so use the new matched controls for causal comparisons.
- **Evaluation:** explicit and NULL exact F1, their combined count-based F1, projected metrics and term/category/sentiment taxonomy. NULL adds presence/category scores, NULL-only/mixed/no-NULL slices, pair rarity and per-domain diagnostics. Projected scores are diagnostics, not replacements for exact F1.
- **Test:** opened only by `--step final --execute`, after the entire cohort passes frozen-source/data/environment/checkpoint checks. It uses saved per-seed dev thresholds and does not recalibrate. Historical test inspection means the new study is exploratory.
- **Recovery/provenance:** settings, code, train/dev hashes, environment, checkpoint SHA and AMP settings are recorded. A changed training source/configuration requires a new study output rather than silently reusing a result.

The optional `--selection-metric null` prioritizes implicit extraction instead; declare it before a new search and use a separate output. The recommended combined objective answers whether adding NULL improves the joint system.

## Hyperparameters and professor recommendations

The explicit BIO/category/sentiment heads keep the previously selected **0.75 standard CE + 0.25 inverse-frequency CE**. The original inverse-frequency implementation is reused, including ignore-label handling and normalized mixture coefficients. Both CE training signals are retained. This follow-up adds focal loss specifically to the independent sigmoid NULL head; it does not replace the selected explicit mixture with focal CE.

| Setting | Default / small search |
|---|---|
| Encoder | `bert-base-uncased`, pinned revision `86b5e0934494bd15c9632b12f734a8a67f723594` |
| LR / optimization | `3e-5`, AdamW, weight decay `.01`, warmup `.10`, gradient clip `1.0` |
| Batch / max length | `16` / `128` |
| Precision | CUDA FP16 AMP; initial loss scale `1024`, adjusted dynamically; CPU uses FP32 |
| BCE NULL | Train negative/positive weight capped at `10` |
| Binary focal NULL | Weighted BCE × `(1 - unweighted target probability)^gamma`, initial gamma `2`; tune `1` if it wins |
| ASL NULL | Positive gamma `0`, negative gamma `4`, negative margin `.05`; tune negative gamma `2` if it wins; no extra BCE positive weight |
| NULL outer loss weight | Initial `.5`; tune `.1` on the winning method |
| BCE cap tuning | Cap `30` if BCE wins |
| Category attention | Learned category queries, attention dimension `128`; no additional BERT passes |
| Random/hard negatives | Always keep all positives; budget max(`16`, `8 × positive count`), limited by available negatives; hard fraction `.5` |
| Dev thresholds | `.02, .05, .10, .12, …, .38, .40, .45, .50, .60, .80`, plus abstention `1` |

All-cell losses use a cell mean. Random and hard samplers use equal positive/negative-group means per sentence, or only the negative mean for sentences with no positives. Thus sampled-versus-all also changes loss normalization; **hard-versus-random** is the comparison that isolates mining under the same budget and reduction. The implicit attention/mining implementations adapt literature ideas to TASD; they do not reproduce the complete iACOS four-element system.

## Individual experiments and checks

Custom configurations have their own identities and are not automatically included in the staged shortlist. For an ASL + implicit attention + hard-negative integration check:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step custom --recipe hard --null-loss asl --smoke --train-limit 128 --dev-limit 32 --output artifacts/null_fast_track_smoke --execute
```

Smoke metrics are integration checks, not evidence of research performance, and cannot be test-evaluated. The trainer refuses completion if no optimizer step succeeds.

Example separate focal experiment:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step custom --recipe focal --null-focal-gamma 1 --null-loss-weight .1 --seeds 13 --epochs 5 --output artifacts/null_focal_custom --execute
```

For all supported flags or a plan without training:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --help
.venv/Scripts/python.exe run_null_fast_track.py
```

## Final evaluation after freezing

After the selected seed cohort finishes and `selection.json` exists:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step final --execute
```

This evaluates the frozen NULL-on shortlist and NULL-off control, appends `test_results.csv`, and refreshes the report with individual test scores, cohort means/SD and matched differences. It does not train another model. Retain the established `final_results/` study alongside this follow-up; report any new gain or failure with the same exact metric and eligibility scope.

## Implementation validation: 6 October 2026

- **15 focused tests passed:** focal/ASL loss arithmetic and finite gradients, positive-preserving sampling, IA/span alignment, attention padding, unknown-label false negatives, abstention/objective selection, append-only CSV behavior, development-only stage selection, seed-cohort archives and deterministic interrupted-run recovery.
- **Real RTX 3050 GPU smoke passed:** ASL + `[IA]` + category attention + hard negatives, 128 training / 32 development examples, one epoch. Eight optimizer updates succeeded under FP16 AMP, with zero skipped updates. This checks integration only; its scores are not research results.
- **Preservation passed:** the historical 22-file training-source inventory still matches all 15 frozen runs. Large new artifacts are Git-ignored.
- **Technique 1 measured:** the historical seed-13 NULL checkpoint, rescored on development data only, selects abstention at threshold 1. Combined dev F1 moves from **0.2941 to 0.3234**; NULL F1 becomes zero. The finer grid does not establish an improved NULL extractor. This supports continuing with the new loss/representation experiments.

Initial calibration is already saved in `artifacts/null_fast_track/calibration/seed_13/`, and the recommended overnight command verifies/reuses it. New research training and test evaluation have not been started. Read the [initial shareable report](../results/null_fast_track/report.md); its charts populate as the new training trials complete.

**Subsequent update:** research training has since started. Use the [final report flow](final-report-flow-2026-10-06.md) to wait for that queue, extend the finalists, test the registered screens as separate ablations, and combine these results with completed quick tuning.
