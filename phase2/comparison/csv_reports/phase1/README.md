# Phase 1 CSV report

The development_results.csv and test_results.csv headers match the supplied example exactly.
Scores are fractions in CSV and percentages below. Empty cells mean unavailable or undefined; they are not zeros.

| Split | Mode | Domain / fold | Recipe | Trial | Seeds | Explicit F1 (%) | NULL F1 (%) | Combined F1 (%) |
|---|---|---|---|---|---:|---:|---:|---:|
| dev | crossdomain | coursera | standard | phase1_standard | 5 | 31.30 | N/A | N/A |
| dev | crossdomain | coursera | weighted | phase1_weighted | 5 | 24.04 | N/A | N/A |
| dev | crossdomain | food | standard | phase1_standard | 5 | 33.79 | N/A | N/A |
| dev | crossdomain | food | weighted | phase1_weighted | 5 | 24.60 | N/A | N/A |
| dev | crossdomain | hotel | standard | phase1_standard | 5 | 30.74 | N/A | N/A |
| dev | crossdomain | hotel | weighted | phase1_weighted | 5 | 24.07 | N/A | N/A |
| dev | crossdomain | laptop | standard | phase1_standard | 5 | 35.33 | N/A | N/A |
| dev | crossdomain | laptop | weighted | phase1_weighted | 5 | 25.11 | N/A | N/A |
| dev | crossdomain | phone | standard | phase1_standard | 5 | 34.18 | N/A | N/A |
| dev | crossdomain | phone | weighted | phase1_weighted | 5 | 27.06 | N/A | N/A |
| dev | crossdomain | restaurant | standard | phase1_standard | 5 | 29.14 | N/A | N/A |
| dev | crossdomain | restaurant | weighted | phase1_weighted | 5 | 20.22 | N/A | N/A |
| dev | crossdomain | sight | standard | phase1_standard | 5 | 33.15 | N/A | N/A |
| dev | crossdomain | sight | weighted | phase1_weighted | 5 | 24.57 | N/A | N/A |
| dev | indomain | all_domains | standard | phase1_standard | 5 | 32.83 | N/A | N/A |
| dev | indomain | all_domains | weighted | phase1_weighted | 5 | 24.26 | N/A | N/A |
| test | crossdomain | coursera | standard | phase1_standard | 5 | 0.00 | N/A | N/A |
| test | crossdomain | coursera | weighted | phase1_weighted | 5 | 0.00 | N/A | N/A |
| test | crossdomain | food | standard | phase1_standard | 5 | 7.37 | N/A | N/A |
| test | crossdomain | food | weighted | phase1_weighted | 5 | 3.60 | N/A | N/A |
| test | crossdomain | hotel | standard | phase1_standard | 5 | 15.90 | N/A | N/A |
| test | crossdomain | hotel | weighted | phase1_weighted | 5 | 13.13 | N/A | N/A |
| test | crossdomain | laptop | standard | phase1_standard | 5 | 0.00 | N/A | N/A |
| test | crossdomain | laptop | weighted | phase1_weighted | 5 | 0.00 | N/A | N/A |
| test | crossdomain | phone | standard | phase1_standard | 5 | 0.00 | N/A | N/A |
| test | crossdomain | phone | weighted | phase1_weighted | 5 | 0.00 | N/A | N/A |
| test | crossdomain | restaurant | standard | phase1_standard | 5 | 12.36 | N/A | N/A |
| test | crossdomain | restaurant | weighted | phase1_weighted | 5 | 20.70 | N/A | N/A |
| test | crossdomain | sight | standard | phase1_standard | 5 | 0.00 | N/A | N/A |
| test | crossdomain | sight | weighted | phase1_weighted | 5 | 0.00 | N/A | N/A |
| test | indomain | all_domains | standard | phase1_standard | 5 | 35.20 | N/A | N/A |
| test | indomain | all_domains | weighted | phase1_weighted | 5 | 26.35 | N/A | N/A |

- [Development runs](development_results.csv), [test runs](test_results.csv).
- [Means and sample SD](summary.csv), [run settings and provenance](run_parameters.csv).
- LODO development scores evaluate the six source domains; the domain column identifies the held-out fold. Only the corresponding test scores evaluate the held-out domain.
- run_directory identifies an archive and member prefix relative to the phase2 handoff root; no original machine path is required.
- Historical NULL and offset-boundary F1 were not measured and remain blank. NULL prevalence is a separate audit.
- These native historical scores use older gold eligibility. Use the versus report for same-gold gains.
