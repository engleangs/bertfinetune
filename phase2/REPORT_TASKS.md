# Team report tasks

Read [comparison/report.md](comparison/report.md) first. Everyone can
regenerate the comparison using the command in [README.md](README.md).
There is no remaining training job required for this handoff.

| Report part | Evidence to use | Deliverable |
|---|---|---|
| Explicit fine-tuning and professor's recommendations | comparison/development_screens.csv; results/in_domain/parameters.csv; comparison/in_domain_comparison.csv; parameters.csv | Development search table, selected settings, five-seed explicit improvement and its limitations |
| Implicit / NULL analysis | results/in_domain/null_findings.md; results/null_fast_track/; evidence/null_diagnostics/ | Matched controls, NULL precision/recall, threshold behavior, rare/unseen support and unresolved extraction finding |
| Cross-domain / LODO | comparison/lodo_same_gold.csv; comparison/lodo_mean_deltas.csv; results/lodo/report.md | Seven-domain table, same-gold Phase 1 comparison, term+sentiment transfer and source-category coverage |
| Error taxonomy and hardest domains | comparison/taxonomy_by_seed.csv; comparison/domain_by_seed.csv; comparison/rarity_by_seed.csv; comparison/sentiment_by_seed.csv; original results.json and Phase 1 snapshots | First-failed term/category/sentiment chart, conditional denominators, rare versus unseen and domain ranking |

These tasks can be split among teammates. Keep draft prose or presentation
files outside the hashed snapshots, for example in your own report directory.

## Suggested report order

1. Explain the task and explicit versus implicit gold. Define exact-triplet
   scoring and the component projections.
2. Describe the locked seeds, source-only vocabulary/frequency counts,
   development selection and separate frozen test evaluation.
3. Show the professor-inspired development search, then the main explicit
   five-seed comparison: 33.41% to 40.28%, +6.88 percentage points.
4. Show NULL on versus its matching NULL-off control. Distinguish the older
   BCE/CLS result from the newer attention adaptation.
5. Show LODO exact F1 next to term+sentiment F1 and category coverage. Use
   the same-gold Phase 1 baselines from comparison/.
6. Explain the taxonomy and hardest domains, then conclude with the measured
   limitations and specific future work.

## Claims supported by these results

- The established explicit fine-tuning pipeline improved same-gold test
  F1 by 6.88 percentage points over Phase 1 standard CE.
- The teacher's standard, weighted, mixed and focal loss suggestions were
  implemented and screened. The selected explicit recipe is the mixture.
- NULL remains unresolved: the historical head recovered some pairs at
  low precision and a combined-score cost. The newer combined-dev-selected
  head abstained and effectively tied its matched control.
- All 35 Phase 2 LODO test evaluations are complete. Exact F1 and component
  transfer answer different questions, especially under category-label shift.

## Checks before submitting the report

Use means with sample SD across the five seeds for finalists. Keep
single-seed screens and their varying epoch budgets in a separate ablation
table. Do not count overlapping screening/finalist rows as independent
models or label the test-best screen as a newly selected winner.

Use comparison/lodo_same_gold.csv when quoting improvement over Phase 1;
the older raw Phase 1 tables have different gold eligibility. Retain the
original Phase 1 numbers as historical context, with their protocol stated.

Report rare and unseen separately using training counts. Include the
taxonomy denominator: error share among missed gold, failure rate among
all gold, or a conditional denominator. First failure does not establish
one causal source of error. Report prediction false positives separately.

Do not drop Conflict retrospectively or change test thresholds. Positive,
Negative, Neutral and Conflict remain in the task. The protocol notes and
the full parameter ledger document the actual configuration.

Describe the gain as a pipeline comparison. The LODO hyperparameters were
selected using all-domain development tuning, and historical tests were
inspected before the follow-up. Both findings need the exploratory label.
Source-category coverage can impose a zero exact-match ceiling; component
F1 provides an additional diagnosis rather than replacing the benchmark.
Sentiment-by-label F1 evaluates triples bearing that sentiment, rather
than accuracy of the sentiment classifier in isolation. An absent class
is not evidence that it was removed from the task.
