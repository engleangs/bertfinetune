# BERT aspect–category–sentiment extraction — Group 07

**The main research study is complete.** Phase 1 has **80 completed runs**;
the completed Phase 2 studies contain **78 distinct test-evaluated models**.
The additional class-balanced study is ongoing separately.

## Review the project

| Start here | Purpose |
|---|---|
| [Phase 2 guide](phase2/README.md) | Short professor-facing guide, code links and expandable details |
| [Final evaluation notebook](phase2/presentation/Final_Evaluation_and_Error_Analysis.ipynb) | Executed results, prediction transitions, LODO coverage, rarity and NULL |
| [Current status](STATUS.md) | Verified completed cohorts and the one ongoing extension |
| [Task register](tasks.md) | Completed work and retired historical plans |
| [Final presentation](phase2/presentation/07_FinalResults.pptx) | Team's selected final deck |

## Completed findings

- **Explicit test F1:** **33.41% → 40.28%**, a **6.88 percentage-point**
  improvement on identical eligible reference answers across five seeds.
- **Selected fine-tuning:** BERT-base-uncased, **0.75 standard CE + 0.25
  inverse-frequency CE**, learning rate **3e-5**, NULL off.
- **Professor feedback:** standard CE, inverse-frequency CE, mixture-weight
  sweeps and focal loss were implemented and screened.
- **Error analysis:** term recovery is the largest missed-triple group;
  component metrics, rare/unseen labels and domains explain exact-match losses.
- **NULL study:** all selected evaluations completed. The newer adaptation
  abstained and tied its matched control; the earlier head recovered some
  pairs but reduced combined F1. This is an unresolved finding, not a pending run.
- **LODO:** all **35 selected Phase 2 tests** completed. Four zero-F1 domains
  reflect missing source-category support, not unfinished training.

The evaluated task has documented annotation-eligibility limits. Historical
test exposure makes the follow-up exploratory; the phase improvement compares
pipelines rather than isolating the loss alone. Research limitations are
described in [PARAMETERS.md](phase2/PARAMETERS.md), not labelled unfinished code.

## Reproduce the completed results

These commands read saved evidence and require no GPU or retraining:

```powershell
python phase2/compare_results.py --no-plots
python phase2/export_csv_reports.py --no-plots
```

For the reviewed notebook and figures, follow the separate analysis-environment
instructions in [phase2/README.md](phase2/README.md). Its reusable code is
[notebook_analysis.py](phase2/notebook_analysis.py).

| Code to inspect | Location |
|---|---|
| BERT encoder and prediction heads | [src/model.py](src/model.py), [experiment model](src/experiment_model.py) |
| Mixed CE, inverse-frequency weighting and focal | [experiment losses](src/experiment_losses.py), [frequency weights](src/losses.py) |
| Training, decoding and checkpoint selection | [experiment_train.py](src/experiment_train.py) |
| NULL adaptation and evaluation | [NULL model](null_experiments/model.py), [NULL evaluation](null_experiments/evaluation.py) |
| Source attribution and experiment entry points | [Reproducibility and contributions](phase2/REPRODUCIBILITY_AND_CONTRIBUTIONS.md) |

<details>
<summary><strong>The one ongoing study</strong></summary>

The user resumed the additional controlled class-balanced extension:
standard CE / inverse-frequency CE / effective-number CE, five seeds,
in-domain plus seven LODO settings (**120 models**). It completes and freezes
all development checkpoints before test evaluation. It is separate from
the completed presentation cohorts.

See [the full-study report](phase2/full_study/report.md). On the training machine:

```powershell
.venv/Scripts/python.exe run_class_balance_full_study.py --status
```

Its groups are correctly marked incomplete until all five seeds finish.
See [STATUS.md](STATUS.md) for the dated count. The existing queue is already
running; do not launch another queue to clear a documentation label.

</details>

<details>
<summary><strong>Phase 1 and archived planning</strong></summary>

[Phase 1](phase1_analysis/README.md) contains 70 completed LODO runs and
10 in-domain controls. Its native historical scores use an earlier eligibility
policy; use Phase 2's same-gold comparisons for the final improvement.

The former root README mixed early protocol gates and a “Partial” assessment
with completed findings. That text is preserved as a
[historical snapshot](phase2/reference/project_history/README_before_status_cleanup_2026_10_14.md).
It is not an active task list. [next-plan.md](next-plan.md) now points to current
status rather than asking the team to repeat completed work.

Unselected development-only variants and interrupted pilot attempts remain
archived and excluded. Their old statuses do not make the selected cohorts
unfinished. Future span/category/NULL improvements are optional research
directions, not outstanding tasks for the completed presentation.

</details>
