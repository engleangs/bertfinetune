# Presentation notes: pre-model text analysis

## Headline

All 21 configured English source files load successfully: 14,776 text records across 7 domains, with 0 parse failures.

## Defensible observations

- The corpus contains 21,017 raw aspect-category-sentiment annotations across 279 category labels and 4 normalized sentiment labels.
- 14,596 triplets (69.5%) fit the current explicit-aspect, aligned, single-label-pair baseline.
- 4,571 implicit/NULL triplets are outside the current BIO span architecture and must be disclosed as a scope limitation.
- Overall text length is a median of 13 words; 95% of records contain at most 43 words by the regex count used here.
- The label charts show why aggregate accuracy is insufficient: report per-label precision/recall and compare standard versus class-weighted loss as planned.

## Method note

These are pre-model descriptive statistics from raw local M-ABSA files. Sentiment/category plots include all raw annotations, including NULL aspects. “Model-ready” uses the exact same baseline selection logic as training. Word counts are not BERT WordPiece counts; token-level truncation/alignment remains covered by `data_audit.py`.
