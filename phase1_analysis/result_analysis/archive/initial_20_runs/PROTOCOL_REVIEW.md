# Protocol review of the completed artifacts

## Overall status

The experiment matrix is complete, paired, and internally consistent, but it is **not fully compliant with protocol.md version 1.0**. Treat the current outputs as observed/post-pilot results unless a dated amendment explicitly accepts the deviations below.

## Confirmed alignment

- All 20 planned `(mode, configuration, seed)` runs completed exactly once.
- Both modes use seeds 13, 42, 123, 2024, and 777.
- The cross-domain test artifacts contain 544 restaurant records, matching the official restaurant test file only.
- Model, batch size, epochs, learning rate, weight decay, maximum length, loss-component weights, held-out domain, rare-label threshold, and minimum effect threshold match the protocol.
- Development exact-triplet micro-F1 is used for checkpoint selection, with lower evaluation loss breaking exact score ties.
- `python -m pip check` currently passes.

## Deviations and missing gates

1. **Warmup mismatch:** `src/train.py` uses `num_warmup_steps=0`; section 10 specifies 10% of total steps.
2. **Gradient clipping missing:** no `clip_grad_norm_` call is present; section 10 specifies maximum norm 1.0.
3. **Ambiguous alignment policy mismatch:** repeated aspect surface occurrences are assigned to the first match rather than excluded. The current data contain 486/114/225 ambiguous examples in the in-domain train/dev/test splits and 466/109/10 in the cross-domain train/dev/test splits.
4. **Standalone audits absent:** `artifacts/data_audit.json` and `artifacts/crossdomain_audit.json` were not present after the run.
5. **Environment not locked:** `requirements.txt` uses broad lower bounds or unpinned packages. Manifests record only Python, Torch, and Transformers versions.
6. **Experiment commit not recorded:** manifests do not include a git commit identifier.
7. **Macro-F1 universe mismatch:** the implemented macro-F1 averages over labels present in each run's gold/prediction union, rather than the protocol's fixed category universe.
8. **Original cross-domain metric gap:** `metrics.json` lacks known-category exact F1, category coverage, aspect-span F1, and aspect-plus-sentiment F1. `visualize_results.py` derives them post hoc from immutable saved predictions and labels them accordingly.

## Recommended handling

Do not silently rerun and replace these results. Either present them with this disclosure, or create a dated protocol amendment, fix the implementation, record a clean commit and locked environment, regenerate audits, and rerun the full matrix as a new experiment version.
