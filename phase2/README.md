# Phase 2 report handoff

**Run [compare_results.py](compare_results.py) to regenerate the Phase 2 comparisons.**
Read [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md) for the commands, dependencies
and report files.

Start with [the comparison report](comparison/report.md), then divide the work using
[REPORT_TASKS.md](REPORT_TASKS.md). All completed results are packaged here; your
teammates can copy this folder alone or clone it through Git.

The selected explicit model achieved **40.28% test F1**, compared with **33.41%**
for Phase 1 on the same corrected gold: **+6.88 percentage points**. NULL
extraction remains unresolved. LODO is complete for seven domains and five
seeds; the consolidated report also re-scores both historical Phase 1
baselines against the same Phase 2 gold.

## Run the comparison

Python 3.10 or newer is required. From the repository root:

    python phase2/compare_results.py

On the original machine, the existing environment also works:

    .venv/Scripts/python.exe phase2/compare_results.py

If you copied only the phase2 folder:

    cd phase2
    python compare_results.py

The command verifies the snapshot hashes and regenerates the tables, paired
seed differences, report and comparison charts in comparison/. It reads
saved results and predictions. It needs **no GPU, PyTorch, BERT downloads,
training data or checkpoints**. It does not choose parameters from test scores.
Running it again replaces only generated comparison files.

To regenerate charts, install the one optional plotting dependency:

    python -m pip install -r phase2/requirements-analysis.txt

Use requirements-analysis.txt without the phase2/ prefix when inside that
folder. Tables and the report need only Python's standard library:

    python phase2/compare_results.py --no-plots

The ready-made charts remain available with --no-plots. Analysis typically
finishes in seconds rather than training hours.

## Where to find the evidence

| Folder or file | Use |
|---|---|
| [comparison/report.md](comparison/report.md) | Main comparison; fair Phase 1 reference and interpretation |
| [comparison/in_domain_comparison.csv](comparison/in_domain_comparison.csv) | Five-seed finalist means, SD and selected settings |
| [comparison/development_screens.csv](comparison/development_screens.csv) | Development screening scores and parameters |
| [comparison/screen_ablations.csv](comparison/screen_ablations.csv) | Separate exploratory test ablations; do not select a winner here |
| [comparison/lodo_same_gold.csv](comparison/lodo_same_gold.csv) | Exact and component F1, category coverage and taxonomy for three LODO arms |
| comparison/*_by_seed.csv | Report-friendly taxonomy, domain, source-category rarity and triplet-by-sentiment rows |
| [results/in_domain/report.md](results/in_domain/report.md) | 25 completed finalist test rows and 21 screen test rows |
| [results/in_domain/null_findings.md](results/in_domain/null_findings.md) | NULL failure analysis and linked threshold evidence |
| [results/lodo/report.md](results/lodo/report.md) | All 35 completed Phase 2 LODO results and per-domain diagnostics |
| results/quick_tuning/ | Earlier quick-tuning report, taxonomy, domain and literature figures |
| results/null_fast_track/ | Completed NULL search, histories and frozen selection |
| reference/phase1/ | Original Phase 1 tables, figures, protocol notes and findings |
| [PARAMETERS.md](PARAMETERS.md), [parameters.csv](parameters.csv) | Settings and selected checkpoint/threshold per evaluated row |
| evidence/phase2_outputs.zip | Complete small metrics, audits, histories, manifests and saved predictions for research runs |
| evidence/phase1_outputs.zip | Small outputs and saved predictions for the 80 historical Phase 1 runs |
| [evidence/archive_inventory.json](evidence/archive_inventory.json) | Archive contents with original size and SHA-256 |
| docs/ | Historical protocol amendments, literature review and experiment guides |

Some screens also occur among the finalists. The 25 + 21 in-domain/NULL
test rows are **not 46 independent models**. The archive also retains older
development runs outside the frozen report; their presence does not make
them finalists or establish that every one has test results.

Raw evidence is compressed to keep the Git handoff small. For manual inspection,
open the ZIPs or optionally extract them:

    python phase2/compare_results.py --no-plots --extract-evidence

This adds roughly 600 MB under evidence/extracted/, which is Gitignored.
It is unnecessary for the comparison command. Checkpoints, optimizer states,
probability tensors, tokenizers, model caches and virtual environments are
excluded. The original output folders remain on the training machine.

## Reporting conventions

Use exact-triplet F1 as the main task metric and component F1 as diagnosis.
Report means and **sample SD across seeds 13, 42, 123, 2024 and 777**.
CSV F1 values are fractions from 0 to 1; Markdown and charts use percentages;
delta_pp columns use percentage points. Screening ablations have one seed
and varying budgets and remain separate from finalist means.

The LODO same-gold comparison retains every historical prediction, matches
example IDs and sentence text, and replaces only the historical gold. Original
Phase 1 scores remain in the evidence. The Phase 2 parameters came from
all-domain development tuning, so this LODO study is **exploratory transfer**.
Historical test results had already been inspected; describe the whole
follow-up as exploratory and do not claim an untouched confirmatory test.

Absolute paths inside copied JSON are provenance from the original machine.
The comparison never follows them. Snapshot inputs are hash-checked; edit
your report in a separate file. Generated comparison/ files are replaceable.
On the training machine, build_phase2_bundle.py refreshes the snapshot and
manifest, followed by the comparison command. Teammates only need phase2/.
