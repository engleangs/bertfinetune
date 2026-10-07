# Team starting point: Phase 1 findings

Read [detailed-analysis.md](detailed-analysis.md) before interpreting the tables.
Use `result_analysis/current/` for the complete results; archive folders preserve
earlier snapshots. Scores use a 0-1 scale. Means and sample SD are over the five
matched seeds; paired differences are weighted minus standard within domain/seed.

| Finding to investigate | Start with | Protocol check |
|---|---|---|
| Which component fails first? | `taxonomy_summary.csv`, `conditional.csv`, `run_diagnostics.json` | Correct + term + category + sentiment must partition retained gold; primary shares and conditional rates have different denominators |
| Are rare and unseen categories different? | `rarity.csv`, `summary.csv` | Source-training counts per fold: unseen=0, rare=1-5; do not reuse the in-domain vocabulary |
| Does category mismatch hide useful transfer? | `summary.csv`, `runs.csv` | Compare full-triplet, term, term+category and term+sentiment F1; show known-category coverage alongside filtered F1 |
| Where are implicit/NULL annotations concentrated? | `null_prevalence.csv`, `null_labels.csv` | Deduplicate within sentence; distinguish NULL-only and mixed sentences; historical explicit-only scores do not measure NULL prediction |
| Does sentiment imbalance explain failures? | `sentiment_counts.csv`, `sentiment_confusion.csv` | Distinguish raw, deduplicated and retained scope; conditional confusion excludes missing terms |
| Which domain is hardest? | `domain_metrics.csv`, `summary.csv` | Rank in-domain and LODO separately; zero-triplet ties require component F1 and coverage |
| Does weighting help consistently? | `paired.csv`, `runs.csv`, `summary.csv` | Compare matched seeds; discuss precision/recall and spurious predictions alongside F1 |

The current baseline shows in-domain exact-triplet F1 of approximately **0.3520
for standard CE and 0.2635 for weighted CE**. Weighting raises recall but reduces
precision. Standard's largest in-domain primary error is term; weighted's is
category. These are observed failure patterns, not causal head ablations.

Four LODO domains tie at zero full-triplet F1: Coursera, laptop, phone and sight.
Coverage and component scores distinguish these cases. Food is lowest in the
in-domain comparison and among LODO folds with nonzero category coverage.
Avoid claiming one unique hardest LODO domain from the zero-triplet ties.

For each assigned finding, add a short Markdown file under this folder with:

1. The question and exact table columns/denominators used.
2. A small comparison table or figure with actual seed counts.
3. One supported interpretation and its limitations.
4. A proposed development-only follow-up, linked to the dated protocol.

Optional per-gold examples can be regenerated with `analyze_phase1.py
--export-errors`. Select examples after reporting aggregate counts and avoid
treating repeated gold across seeds as independent samples.

The professor's mixed CE/weighted CE, focal loss and coefficient sweep are
implemented in the main repository's separate version-2 runner. Continue its
two-seed development pilot before a five-seed shortlist; this Phase 1 folder
provides the historical diagnosis and does not establish improvement from those
new losses. Do not tune the continuation on the inspected Phase 1 tests.
