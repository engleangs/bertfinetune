# LODO continuation

The background continuation waits for the user's seven seed-13 folds at the frozen `mixed_025`, LR `3e-5`, NULL-off configuration. It then trains seeds **42, 123, 2024 and 777**, freezes all **35 development-selected checkpoints**, evaluates their held-out official test sets, and writes `report.md`, `results.json`, `selection.json` and `comparison.png` here.

This folder contains small report artifacts only. Model weights remain under the Git-ignored `artifacts/experiments_v2/research/crossdomain/` directory.

Runtime status is saved in [status.json](../artifacts/lodo_continuation/status.json); the fixed plan and commands are in [plan.json](../artifacts/lodo_continuation/plan.json). The background launch metadata and stdout/stderr paths are recorded in [latest.json](../artifacts/lodo_continuation/latest.json).

Start the seed-13 command supplied in the conversation. The continuation does not launch those first seven training runs itself. Keep the machine awake; the helper waits up to 48 hours for this first cohort. If the first queue records a failed fold, the continuation stops with a recorded error rather than opening test data.

To resume the continuation after fixing a failure, use:

```powershell
.venv/Scripts/python.exe continue_lodo_finetuning.py --wait-for-first --execute
```

Do not run a second continuation while its `active.lock` is held by a live process. Recorded failed remaining runs can be retried with `--retry-failed`; completed matching runs are reused. The existing trainer restarts a failed run rather than resuming its unfinished epoch.

To regenerate the finished report from saved metrics:

```powershell
.venv/Scripts/python.exe continue_lodo_finetuning.py --step report
```

These are exploratory LODO transfer results: the original hyperparameters were selected using all-domain in-domain development data. Each new model nevertheless trains and selects its checkpoint using only its six source domains. Report exact-triplet, term and term+sentiment F1 alongside category coverage. The historical Phase 1 scores have an earlier eligibility protocol and require rescoring on corrected gold before claiming a numerical gain.

The primary in-domain/NULL report remains in [final_presentation_results](../final_presentation_results/report.md).

If Windows locks the central `development_results.csv` during final evaluation, the inference-only recovery wrapper preserves it and writes `artifacts/experiments_v2/development_results_lodo_recovery.csv`. Its provenance note is `artifacts/lodo_continuation/csv_recovery.json`. Saved checkpoint selection and test metrics are unchanged; completed evaluations are reused. A resume with all development folds complete skips training entirely.
