# Interim LODO Result Analysis

Generated: 2026-09-18T21:10:12.896843+00:00

## Completion status

- Completed runs: **48/60 (80.0%)**
- Complete five-seed domains: coursera, hotel, laptop, phone
- Partial domains: sight, food
- Missing logical run keys: 12

## Paired exact-triplet F1

| Held-out domain | Status | Matched seeds | Standard F1 | Weighted F1 | Difference | 95% bootstrap CI | p-value |
|---|---|---:|---:|---:|---:|---:|---:|
| coursera | complete | 5/5 | 0.000 | 0.000 | 0.0000 | [0.000, 0.000] | — |
| hotel | complete | 5/5 | 0.159 | 0.131 | -0.0278 | [-0.044, -0.012] | 0.0376 |
| laptop | complete | 5/5 | 0.000 | 0.000 | 0.0000 | [0.000, 0.000] | — |
| phone | complete | 5/5 | 0.000 | 0.000 | 0.0000 | [0.000, 0.000] | — |
| sight | partial | 1/5 | 0.000 | 0.000 | 0.0000 | — | — |
| food | partial | 1/5 | 0.076 | 0.021 | -0.0549 | — | — |

Partial-domain statistics are descriptive only. Do not compare a single weighted seed with five standard seeds.

## Configuration-level metrics

| Domain | Config | Status | n | Precision | Recall | F1 | Aspect F1 | Aspect+sentiment F1 | Category coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| coursera | standard | complete | 5 | 0.000 | 0.000 | 0.000 | 0.559 | 0.531 | 0.2% |
| coursera | weighted | complete | 5 | 0.000 | 0.000 | 0.000 | 0.469 | 0.381 | 0.2% |
| hotel | standard | complete | 5 | 0.157 | 0.161 | 0.159 | 0.540 | 0.514 | 38.1% |
| hotel | weighted | complete | 5 | 0.092 | 0.227 | 0.131 | 0.492 | 0.458 | 38.1% |
| laptop | standard | complete | 5 | 0.000 | 0.000 | 0.000 | 0.393 | 0.369 | 0.0% |
| laptop | weighted | complete | 5 | 0.000 | 0.000 | 0.000 | 0.495 | 0.423 | 0.0% |
| phone | standard | complete | 5 | 0.000 | 0.000 | 0.000 | 0.219 | 0.209 | 0.0% |
| phone | weighted | complete | 5 | 0.000 | 0.000 | 0.000 | 0.338 | 0.303 | 0.0% |
| sight | standard | complete | 5 | 0.000 | 0.000 | 0.000 | 0.265 | 0.237 | 0.0% |
| sight | weighted | partial | 1 | 0.000 | 0.000 | 0.000 | 0.437 | 0.342 | 0.0% |
| food | standard | partial | 1 | 0.061 | 0.100 | 0.076 | 0.431 | 0.351 | 27.7% |
| food | weighted | partial | 1 | 0.013 | 0.060 | 0.021 | 0.289 | 0.228 | 27.7% |

## Interpretation limits

- This is a live interim analysis. Only complete five-seed domain pairs support the planned paired comparison.
- Error counts are the pipeline's existing preliminary counts; they are not yet the planned mutually exclusive gold-triplet taxonomy.
- Category coverage is the percentage of held-out gold triplets whose category exists in source training.
- These runs still use zero warmup and no gradient clipping, so they remain post-pilot evidence unless a protocol amendment accepts those settings.
- `NULL` aspects remain outside the explicit-span model and are not included in the restricted-task F1.
