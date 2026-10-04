# Presentation notes: completed experiment matrix

## Confirmatory in-domain result

Across five matched seeds, standard loss achieved 0.352 ± 0.008 exact-triplet micro-F1 and weighted loss achieved 0.264 ± 0.005. The paired difference was -0.0885 with a 95% paired-bootstrap interval of [-0.0979, -0.0798] and paired t-test p=0.0001. Class weighting therefore did not meet the pre-specified +0.02 improvement criterion.

## Secondary cross-domain result

On the held-out restaurant test split, standard loss achieved 0.124 ± 0.017, while weighted loss achieved 0.207 ± 0.023. The paired difference was +0.0834; this secondary result is exploratory. The source category vocabulary covers 71.3% of retained restaurant gold triplets.

## Interpretation

Weighting increases recall in both modes. In-domain, the precision loss is large enough to reduce F1. Cross-domain, the recall gain is large enough to increase F1. This is evidence of a precision–recall trade-off, not a universal improvement.

## Disclosure

These are observed saved-artifact results, not a fully protocol-conformant final analysis. See `PROTOCOL_REVIEW.md` before presenting conclusions.
