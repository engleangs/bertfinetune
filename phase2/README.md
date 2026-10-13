# Phase 2 — final evaluation and code guide

This folder presents our completed BERT fine-tuning study and its evidence.
**Start with the notebook:** saved outputs can be read without a GPU, model
downloads or retraining.

## Start here

| What to review | File |
|---|---|
| Final evaluation, figures and errors | [Readable notebook](presentation/Final_Evaluation_and_Error_Analysis.ipynb) |
| Computation behind the notebook | [notebook_analysis.py](notebook_analysis.py): scoring, transitions, LODO, rarity and NULL |
| Parameters and metric definitions | [PARAMETERS.md](PARAMETERS.md) |
| Our work and external sources | [Reproduction and contributions](REPRODUCIBILITY_AND_CONTRIBUTIONS.md) |

Project code: [engleangs/bertfinetune](https://github.com/engleangs/bertfinetune).

## Findings

- **Explicit test F1:** 33.41% → **40.28%**, **+6.88 percentage points**,
  on the same reference answers across five seeds.
- **Selected recipe:** full BERT-base-uncased fine-tuning, **0.75 CE + 0.25
  inverse-frequency CE**, LR **3e-5**, NULL off.
- **Errors/transfer:** term recovery is the largest missed-triple group;
  source-category coverage constrains exact matching in held-out domains.
- **NULL:** the newer head abstained and tied its matched control. The earlier
  BCE head recovered some pairs but reduced combined F1. Recovery is unresolved.

These are exploratory **pipeline comparisons**, not loss-only causal effects.
Seed SD describes initialization variation on the same test examples.
The ongoing optional 120-model class-balanced extension is separate; its
partial development results are excluded from final test conclusions here.

## Reproduce the completed comparison

From the repository root, using only Python's standard library:

```powershell
python phase2/compare_results.py --no-plots
```

This verifies the snapshot and rebuilds tables/report without training.
[The comparison report](comparison/report.md) is already available.

<details>
<summary><strong>Rerun the notebook or its full analysis</strong></summary>

Use a **separate analysis environment**; do not change the active trainer.

```powershell
python -m venv .venv-analysis
.venv-analysis/Scripts/python.exe -m pip install -r phase2/requirements-notebook.txt
.venv-analysis/Scripts/python.exe phase2/notebook_analysis.py
```

On macOS/Linux use `.venv-analysis/bin/python`. The command writes tables,
four figures and verification to `comparison/notebook/`, using no GPU.
Use `--no-plots` to skip figures.

For interactive use, select that environment in VS Code/Jupyter and choose
**Run All**. Short notebook cells call the helper; long code stays separate.
With only a copied `phase2/` folder, run `python notebook_analysis.py` inside it.
Use `--bundle path/to/phase2.zip` to read a ZIP without extracting it.

[The notebook review](NOTEBOOK_REVIEW.md) records checks/corrections and links
the untouched original. Historical predictions are retained, and each model's
own source vocabulary is used for coverage.

</details>

<details>
<summary><strong>Inspect BERT, loss, NULL and metric implementations</strong></summary>

| Implementation | Code in the full repository |
|---|---|
| BERT/BIO/span-category/sentiment heads | [src/model.py](../src/model.py), [experiment model](../src/experiment_model.py) |
| CE mixture, inverse-frequency and explicit focal | [experiment_losses.py](../src/experiment_losses.py), [frequency weights](../src/losses.py) |
| Training, checkpoint selection and decoding | [experiment_train.py](../src/experiment_train.py) |
| Eligibility and explicit/NULL targets | [experiment_data.py](../src/experiment_data.py) |
| NULL representations and losses | [null model](../null_experiments/model.py), [NULL losses](../null_experiments/losses.py) |
| Development thresholds and NULL evaluation | [null evaluation](../null_experiments/evaluation.py) |
| Shared exact/component scoring | [analysis.py](analysis.py) |

The notebook needs only this folder. Inspecting training implementations needs
the full repository. [Contribution documentation](REPRODUCIBILITY_AND_CONTRIBUTIONS.md)
credits the external data, BERT and loss concepts and lists experiment scripts.

</details>

<details>
<summary><strong>Evidence, CSV exports and detailed reporting</strong></summary>

- [In-domain means/controls](comparison/in_domain_comparison.csv).
- [Development selections](comparison/development_screens.csv) and separate
  [exploratory test screens](comparison/screen_ablations.csv).
- [LODO exact/component comparison](comparison/lodo_same_gold.csv).
- [NULL findings](results/in_domain/null_findings.md).
- [Notebook-generated report](comparison/notebook/report.md).
- [Professor feedback implementation](PROFESSOR_RECOMMENDATIONS_IMPLEMENTED.md).
- [Run instructions](RUN_INSTRUCTIONS.md), [report task guide](REPORT_TASKS.md)
  and [parameter ledger](parameters.csv).

Separate Phase 1, Phase 2 and versus development/test CSV exports:

```powershell
python phase2/export_csv_reports.py --no-plots
```

Small metrics/predictions are in `evidence/phase1_outputs.zip` and
`evidence/phase2_outputs.zip`. Huge checkpoints and model caches are excluded.
CSV scores are fractions unless named `pct` or `pp`; figures/Markdown use
percentages. Rare means source count 1–5; unseen means zero. Partial groups
are not five-seed conclusions.

The notebook and comparison reports are generated views. The manifest protects
frozen inputs and helper code. Earlier presentations/protocol progress notes
are historical; do not alter evidence to hide a hash mismatch.

</details>

<details>
<summary><strong>TODO : Ongoing additional full study</strong></summary>

The standard / inverse-frequency / effective-number class-balanced extension
uses 3 losses × 5 seeds × 8 settings. All 120 development checkpoints must
finish and freeze before test evaluation. It is separate from the completed
presentation cohorts.


</details>
