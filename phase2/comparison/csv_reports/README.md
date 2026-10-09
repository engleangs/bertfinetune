# CSV reports matching development_results.csv

Generated from the frozen model-free handoff; no training or inference.

- [Phase 1 report](phase1/README.md): 80 historical runs, with separate development/test CSVs.
- [Phase 2 report](phase2/README.md): all unique bundled development/test models and their settings.
- [Phase 1 versus Phase 2](vs/README.md): matched five-seed development/test comparisons on corrected gold.
- [Supplied smoke example](supplied_smoke/README.md): kept separate from full-data results.

The phase CSVs preserve the original 25 columns and their order. The versus CSVs put Phase 1/Phase 2 metrics side by side, with percentage-point differences. Blank cells mean a metric was unavailable or had no denominator.

explicit_f1 is exact (term, category, sentiment) micro F1; null_f1 is implicit (category, sentiment) F1. Combined F1 uses combined TP/FP/FN, not an average of scores. term_f1, term_category_f1 and term_sentiment_f1 project and deduplicate within each sentence. boundary_f1 uses offsets and stays blank for historical runs without offset records. known_category_coverage uses each run's source vocabulary. Error counts assign the first failed component per gold triplet. Rare means source category count 1-5; unseen labels are separate. A saved NULL threshold is inactive when null_head is False in run_parameters.csv.

Regenerate from the repository root with:

    python phase2/export_csv_reports.py

Use --no-plots for the standard-library-only path. Inside a copied phase2 folder, run python export_csv_reports.py. Originals are preserved; these derived reports are replaceable.
