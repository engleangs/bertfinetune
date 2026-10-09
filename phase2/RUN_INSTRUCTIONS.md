# Phase 2: which file to run

**Main Python file: [compare_results.py](compare_results.py).**
It rebuilds the report comparisons from the bundled results and predictions:
explicit fine-tuning, NULL on/off, screening ablations, and all 35 Phase 2
LODO results with both Phase 1 baselines re-scored on matching gold.

| File | Purpose | Run directly? |
|---|---|---|
| [compare_results.py](compare_results.py) | Verify bundle hashes and regenerate tables, charts and the comparison report | Yes: this is the team entry point |
| [export_csv_reports.py](export_csv_reports.py) | Export Phase 1/Phase 2 development and test scores with the supplied CSV columns, plus matched-gold versus tables | Yes, for the CSV reports |
| [analysis.py](analysis.py) | Align historical gold and calculate exact/component scores and paired comparisons | Imported automatically by compare_results.py |
| build_phase2_bundle.py, at the repository root | Export a fresh snapshot from the original training-machine artifacts | For the snapshot owner when exporting updated results |

## Run from the repository root

```powershell
python phase2/compare_results.py
```

On the original training machine:

```powershell
.venv/Scripts/python.exe phase2/compare_results.py
```

## Run when you copied only this folder

```powershell
cd phase2
python compare_results.py
```

Python 3.10 or newer is sufficient. Keep the full phase2 folder, including
results/, evidence/ and bundle_manifest.json. The command verifies inputs
automatically, reads the compressed evidence directly and replaces only
generated comparison/ files. No GPU, PyTorch, original artifact folders,
training data or model weights are needed.

## Optional chart dependency

Ready-made charts are included. To regenerate them, install matplotlib once:

```powershell
python -m pip install -r phase2/requirements-analysis.txt
python phase2/compare_results.py
```

When inside phase2, omit the phase2/ prefix. To regenerate tables and the
report using only Python's standard library:

```powershell
python phase2/compare_results.py --no-plots
```

## Files to use for the report

- [comparison/report.md](comparison/report.md): main result comparison and interpretation.
- [comparison/in_domain_comparison.csv](comparison/in_domain_comparison.csv): five-seed explicit/NULL finalists.
- [comparison/development_screens.csv](comparison/development_screens.csv): development tuning evidence.
- [comparison/screen_ablations.csv](comparison/screen_ablations.csv): separate exploratory test ablations.
- [comparison/lodo_same_gold.csv](comparison/lodo_same_gold.csv): fair Phase 1 versus Phase 2 LODO comparison.
- [PARAMETERS.md](PARAMETERS.md) and [parameters.csv](parameters.csv): actual settings and selected checkpoints/thresholds.
- [REPORT_TASKS.md](REPORT_TASKS.md): suggested division of report work.

To generate the CSV reports:

    python phase2/export_csv_reports.py

Read [comparison/csv_reports/README.md](comparison/csv_reports/README.md). Each
phase has separate development_results.csv and test_results.csv files, plus
summary.csv and run_parameters.csv. The versus folder contains paired
five-seed differences on matching gold. Interrupted runs and the supplied
small smoke example stay separate from the completed full-data tables.
Use --no-plots for standard-library-only export. Inside phase2, run
python export_csv_reports.py.

The Phase 1 tables in reference/phase1/ are historical snapshots. The
comparison script reads their saved predictions from evidence/phase1_outputs.zip;
the complete independent Phase 1 analyzer is in the separate phase1_analysis
folder. Use the same-gold comparisons here when quoting an improvement,
and retain the exploratory label described in the report.
