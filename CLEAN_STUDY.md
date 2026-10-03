# Conflict Cleanup and Parameter Study

Version: `clean-study-v1`. Implemented after the discussion on 30 September 2026. Earlier test results had already been inspected, so this study is an exploratory follow-up.

## Experiment scope

- Cleanup removes the **five complete training examples** containing conflict labels: five conflict annotations and two additional positive annotations. Raw data are preserved; development and test files remain byte-for-byte identical. NULL handling and the original explicit-aspect filtering rules are retained.
- A: pooled in-domain, raw/clean data × Standard/Weighted × five seeds = 20 runs. Learning rate 2e-5, five epochs, no warmup, and no gradient clipping.
- B: pooled in-domain plus seven LODO folds × two losses × learning rates 2e-5/3e-5/5e-5 × seeds 13/42 = 96 runs. Thirty epochs, 10% warmup, and gradient clipping at 1.0. Search uses source-domain train/development data only.
- Within each condition, select a shared learning rate using mean best development exact-triplet micro-F1 across both losses and both search seeds; ties select the lower learning rate. Other folds and pooled in-domain results must not inform a LODO fold's parameter selection.
- C: reuse the 32 selected search checkpoints and train seeds 123/2024/777 in 48 additional runs, yielding 80 final results. Test evaluation uses development-selected checkpoints.
- The core budget is 164 full training runs, plus two GPU smoke checks using 64 training and 32 development examples. Aspect-only weighting, category-only weighting, 1e-4, and a 40-epoch extension are outside this study.
- No early stopping is used; all 30 epochs are logged. Epoch 8 of a 30-epoch linear schedule is not equivalent to a separate eight-epoch schedule. Any extension beyond the budget requires a separately versioned study.
- Cleanup changes both examples and the sentiment output vocabulary, so changes cannot be attributed solely to one large class weight. Report five-seed means, standard deviations, and paired differences, and disclose reuse of search seeds 13/42.

## Storage and downloads

Each run saves one `best_model.pt`. While training, it also maintains one rolling `resume.pt` containing current weights, AdamW state, scheduler state, the best model state, and random-number states. The recovery file is removed after training evidence is saved successfully. Interrupted runs resume after the last completed epoch; an unfinished epoch is repeated. Per-epoch metrics are stored in a small `history.json`, without separate model files for every epoch.

| Content | Cluster retention | Local download |
|---|---|---|
| Metrics, training curves, configuration, per-label scores | Keep all | Included |
| Development and test predictions | Store as gzip | Included |
| Cleanup audit, data hashes, raw-data snapshot | Keep one copy | Included |
| Executed code snapshot, model hashes, environment versions | Keep | Included |
| Pretrained BERT and tokenizer | Reuse shared assets; save one small tokenizer snapshot | Tokenizer/configuration only |
| Search checkpoints | Keep 96 until selection, then the selected 32 until evaluated | Excluded |
| Final checkpoints | After evaluation and successful export, retain one dev-selected representative per condition/loss, 16 total | Excluded |
| Optimizer and recovery files | Needed only for unfinished training | Excluded |

The 16 retained models occupy approximately 7 GB on the cluster. The result ZIP contains no model weights; its actual size is recorded at export. Saved predictions support rescoring, error analysis, and plotting. Inference on new text requires a retained checkpoint or retraining.

This is final retention, not peak storage. Candidate weights must remain available until selection. With two concurrent tasks, allow approximately 65–70 GB for the new study, including rolling recovery files and temporary atomic writes. Higher concurrency increases peak usage. A small download does not eliminate this temporary cluster storage requirement.

Pruning targets only explicitly named checkpoints inside the new directory identified by `study.json`. Existing study directories and older models are unaffected. Unfinished runs retain recovery files. Metrics and prediction hashes are checked before pruning. Completion markers remain after checkpoint removal so rerunning a completed task does not trigger unnecessary training.

## Upload and submission

Build `bertfinetune-clean-v1.tar.gz` locally without old results or pretrained model weights. Extract it into `/data/$USER` to create a separate `bertfinetune-clean-v1` directory. By default, the environment and BERT assets are reused from `/data/$USER/bertfinetune`. Set `CS760_SHARED_PROJECT` if that location differs, or set `CS760_PYTHON` and `BERT_MODEL_PATH` separately.

In the new cluster project directory, run:

```bash
sha256sum -c cluster/transfer_manifest.sha256
bash cluster/submit_clean_study.sh
```

The second command only prints the job graph. After checking GPU quotas and permitted wall times, submit:

```bash
CS760_CONCURRENCY=2 CS760_TRAIN_TIME=04:00:00 bash cluster/submit_clean_study.sh --submit
```

The script submits the following dependency chain:

1. `smoke`: two small training jobs. Failure prevents later stages from starting.
2. `main`: 116 interleaved jobs, comprising 20 replication runs and 96 search runs. The entire array uses at most two GPUs concurrently by default.
3. `select`: a CPU job selects parameters within each fold, verifies selected checkpoints, and removes the 64 unselected new search checkpoints.
4. `completion`: 80 jobs, comprising 32 checkpoint evaluations and 48 additional training runs.
5. `finalize`: verify 100 test results, summarize, export a ZIP, retain 16 representative checkpoints, and refresh the ZIP.

GPU jobs request one GPU, four CPUs, and 32 GB RAM. Four hours is an adjustable per-task wall-time request, not a guarantee of sufficient runtime or account-policy compliance. Measure a small run before full submission when needed. GPU type is assigned by the cluster; confirm any account-specific partition or A100 request settings. Training runs through Slurm; login nodes are used for preparation and submission.

To run smoke checks separately first:

```bash
sbatch --array=0-1%2 cluster/clean_study_gpu.sbatch smoke
```

Inspect the smoke logs before using the full submission script. The full workflow skips completed smoke tasks. Submitted job IDs are recorded; duplicate full submissions are refused. After investigating a failed array task, retry the relevant phase and index, for example:

```bash
sbatch --array=17 cluster/clean_study_gpu.sbatch main
```

A failure blocks dependent jobs. After retrying, inspect `submission_jobs.txt` and cancel/rebuild downstream jobs still waiting on the original failed dependency as needed. A retry does not automatically repair those dependencies.

## Download only the lightweight results

After completion, the new project contains:

```text
artifacts/clean_study_results_light.zip
artifacts/clean_study_results_light.zip.json
```

The second file records size and SHA256. The ZIP contains CSV and JSON summaries, training histories, compressed predictions, configuration/environment/data audits, and a code snapshot. These support local report preparation without downloading the entire `artifacts` or `model_cache` directory.

## Export lightweight evidence from older experiments

From the project containing this runner, specify the old results directory:

```bash
python run_clean_study.py export --legacy-root /path/to/old/artifacts --destination /path/to/old-results-light.zip
```

This reads old metrics and predictions without reading or deleting old checkpoints. Replace `python` with the actual environment's Python executable.

## Inspect the pruning plan manually

```bash
python run_clean_study.py prune --scope unselected
python run_clean_study.py prune --scope archive
```

The default is a dry run. `--apply` deletes only the listed new checkpoints. If representative models are no longer needed after downloading and verifying the result archive, use `--scope archive --keep-models none --apply`, then export again to refresh the retention records.

## Published results and frozen study metadata

The completed study is documented in [the English results report](results/clean-study-v1/REPORT.md). The publication directory also contains machine-readable summaries and the lightweight experiment evidence.

Study metadata hashes refer to the exact files used for training. This English documentation update changes a fingerprinted file; do not replace files inside a prepared cluster study to apply this translation. Use the original training version, commit `e3d110f`, when resuming or re-exporting that frozen study. This publication does not require any new training.
