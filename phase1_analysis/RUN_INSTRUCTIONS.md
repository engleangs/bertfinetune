# Phase 1: which file to run

**Main Python file: [analyze_phase1.py](analyze_phase1.py).**
It reproduces the historical standard-versus-weighted CE analysis from
70 LODO runs and 10 in-domain runs.

| File | Purpose | Output |
|---|---|---|
| [verify_bundle.py](verify_bundle.py) | Check saved input hashes, relative paths and the complete 80-run matrix | Verification message; no report changes |
| [analyze_phase1.py](analyze_phase1.py) | Recalculate metrics, error taxonomy, source-category rarity and NULL prevalence | result_analysis/current/ tables and detailed-analysis.md |
| [phase1_analysis.ipynb](phase1_analysis.ipynb) | Show the analysis interactively and regenerate notebook charts | result_analysis/current/figures/ |

## Run from the repository root

Use Python 3.10 or newer. Install the analysis dependencies once in your
chosen Python environment:

```powershell
python -m pip install -r phase1_analysis/requirements.txt
```

Then verify and regenerate the analysis:

```powershell
python phase1_analysis/verify_bundle.py
python phase1_analysis/analyze_phase1.py
```

On the original training machine, the existing environment can be used:

```powershell
.venv/Scripts/python.exe phase1_analysis/verify_bundle.py
.venv/Scripts/python.exe phase1_analysis/analyze_phase1.py
```

## Run when you copied only this folder

```powershell
cd phase1_analysis
python -m pip install -r requirements.txt
python verify_bundle.py
python analyze_phase1.py
```

Keep the entire folder, including analysis_helpers/, corpus/, lodo_runs/
and indomain_runs/. Analysis runs on CPU from saved predictions; it needs
no checkpoints or model downloads. It prints progress while validating the
80 runs and replaces the generated tables and detailed analysis.

## Files to use for the report

- [detailed-analysis.md](detailed-analysis.md): interpretation and methodology.
- [TEAM_FINDINGS.md](TEAM_FINDINGS.md): report questions and findings plan.
- [summary.csv](result_analysis/current/summary.csv): standard/weighted means and seed variation.
- [taxonomy_summary.csv](result_analysis/current/taxonomy_summary.csv): first-failed term/category/sentiment counts.
- [domain_metrics.csv](result_analysis/current/domain_metrics.csv), [rarity.csv](result_analysis/current/rarity.csv) and [null_prevalence.csv](result_analysis/current/null_prevalence.csv): domain, category support and implicit annotation audits.

For charts, open phase1_analysis.ipynb in VS Code or Jupyter and run all
cells using the same environment. Install ipykernel if that environment
is not available as a notebook kernel. The Python analyzer regenerates
tables and prose; the notebook regenerates the Phase 1 charts.

Phase 1 preserves the historical evaluation scope. For the comparison
against Phase 2 on corrected gold, use **phase2/compare_results.py** from
the repository root and read phase2/comparison/report.md.
