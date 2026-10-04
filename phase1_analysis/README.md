# Phase 1: shared LODO results and error analysis

This folder contains **70 completed LODO runs and 10 in-domain controls**:
seven domains, standard/weighted cross-entropy, and matched seeds
13, 42, 123, 2024 and 777. Restaurant's original cross-domain runs are included
once with the other six folds. All artifact paths are relative to this folder.

Start with [the team findings plan](TEAM_FINDINGS.md) and
[the detailed analysis](detailed-analysis.md). The primary research question is
which component fails first: term, category or sentiment.

| Location | Contents |
|---|---|
| `lodo_runs/<domain>/<loss>/seed_<seed>/` | All 70 LODO manifests, metrics, histories, dev/test predictions |
| `indomain_runs/<loss>/seed_<seed>/` | The 10 matched in-domain controls |
| `results_lodo.csv`, `results.csv` | Portable indexes: 70 LODO rows, 10 in-domain rows |
| `result_analysis/current/` | Current 80-run taxonomy, component F1, paired comparisons and corpus audits |
| `result_analysis/current/figures/` | Current notebook comparison charts, when rendered |
| `result_analysis/archive/initial_20_runs/` | Original 20-run presentation figures and reports |
| `result_analysis/archive/interim_48_runs/` | Original incomplete LODO snapshot; **not the final results** |
| `result_analysis/archive/team_review/` | Original notebook backup and historical validation records |
| `corpus/<domain>/en/` | The 21 small English train/dev/test files used to check gold scope and training frequencies |
| `phase1_analysis.ipynb` | Clean analysis notebook with outputs cleared |
| `analyze_phase1.py`, `analysis_helpers/` | Analysis-only snapshot of the reviewed code |
| `bundle_manifest.json` | File hashes, sizes, original code hashes and export provenance |

Model weights, checkpoints, tokenizers and ZIP archives are excluded. Existing
training artifacts stay in the original repository. Run manifests are preserved
byte-for-byte; their checkpoint/tokenizer references describe excluded training
files and are not needed for this analysis. The duplicate flat LODO export is
represented by the corresponding runs here rather than copied twice.

The approximately 50 MB `gold_error_records.jsonl` is also excluded from Git.
It can be regenerated locally with `--export-errors`.

## Reproduce after cloning

Python 3.10 or newer is sufficient. No GPU or model download is required.
From the repository root:

```powershell
python phase1_analysis/verify_bundle.py
python -m venv phase1_analysis/.venv
phase1_analysis/.venv/Scripts/python.exe -m pip install -r phase1_analysis/requirements.txt
phase1_analysis/.venv/Scripts/python.exe phase1_analysis/analyze_phase1.py
```

On macOS/Linux, use `phase1_analysis/.venv/bin/python` for the last two commands.
You can also run `python analyze_phase1.py` from inside this folder with an
existing environment that contains the analysis dependencies.

The analyzer recomputes metrics and taxonomy, verifies saved scores against both
indexes and metrics files, checks raw test sentences/gold and source-training
vocabularies/counts, and rejects duplicate or unmatched seeds. It updates
`result_analysis/current/` and `detailed-analysis.md`.

To inspect every gold error locally:

```powershell
phase1_analysis/.venv/Scripts/python.exe phase1_analysis/analyze_phase1.py --export-errors
```

For notebook charts, install `ipykernel` in the analysis environment and open
[phase1_analysis.ipynb](phase1_analysis.ipynb) with that kernel. Run cells in order.
The notebook works when launched from this folder or the repository root.

`verify_bundle.py` checks the immutable analysis inputs by default. Use
`--include-reports` to additionally check the exported report snapshot before
regenerating or editing reports. The local `.gitattributes` preserves file bytes
and hashes across Windows/Linux Git clones.

## Reporting rules and scope

Apply [reporting amendment 1.1](protocol-amendment-2026-10-04.md) to these historical
version-1 runs. The amendment takes precedence over older archive wording.
The archived interim report had only 48 of its planned 60 additional LODO runs;
`result_analysis/current/` includes the complete seven-fold, 70-run LODO matrix.

Keep exact-triplet micro-F1 as the primary performance metric. Component F1 and
the exclusive taxonomy explain failures; they do not replace the primary score.
Report rare versus unseen categories using each fold's source-training counts.
Historical NULL targets are unsupported by the explicit BIO architecture, so
NULL prevalence is a separate annotation audit. Keep the four sentiment classes
in this historical comparison; removing Conflict belongs to a separate study.

These tests have already been inspected. Findings are descriptive/exploratory.
The ongoing version-2 loss/NULL pilot is a separate development study in the
main repository and is not mixed into the Phase 1 tables. Its dated experiment
guide is included for context; commands in that guide refer to the main repo.

The corpus came from [the M-ABSA repository](https://github.com/swaggy66/M-ABSA),
using the paths in the main repository's `download_data.py`. Current corpus
hashes support consistency checks; they do not prove historical immutability.

## Refresh or share

From the original repository with its local artifacts:

```powershell
.venv/Scripts/python.exe consolidate_phase1.py --validate
```

Refreshing overwrites the exported run files, analysis code snapshot, notebook
and reports. Keep team-authored findings in new files. The README, team plan and
verifier remain editable. Original artifacts and training code are unchanged.

To stage just this package and its exporter:

```powershell
git add phase1_analysis consolidate_phase1.py
git diff --cached --stat
```

Review the staged changes, then commit and push on your chosen branch.
