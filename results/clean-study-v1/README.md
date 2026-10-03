# Completed Clean Study Results

Start with the [English results report](REPORT.md). It explains the metrics and summarizes five-seed comparisons, conflict cleanup, cross-domain category coverage, and training duration.

## Files

| File | Contents |
|---|---|
| [REPORT.md](REPORT.md) | Human-readable English report with Precision, Recall, F1, means and standard deviations |
| [results.csv](results.csv) | 100 test-evaluation rows: 20 replication results and 80 final results |
| [summary.json](summary.json) | Grouped F1 summaries and paired differences |
| [selection.json](selection.json) | Per-condition learning-rate selection and development scores |
| [experiment_results.zip](experiment_results.zip) | Full lightweight evidence: predictions, histories, metrics, manifests, audits and reproducibility assets; no model weights |
| [experiment_results.zip.json](experiment_results.zip.json) | Published archive size and SHA256 |

The archive is 31.68 MB. It contains evidence for 164 full training runs plus two smoke checks. Exactly 100 runs have test evaluations; the other 64 full runs are unselected search candidates evaluated on development data only. The two smoke checks also use train/development data only. Sixteen representative model checkpoints remain on the cluster and are not included here.

## Verify the download

After downloading the ZIP, run:

```bash
sha256sum experiment_results.zip
```

Expected SHA256:

```text
929f58027f87de07a1e6be72f28bef3bd6c6ada23df1644c2fccd1a153065175
```

The companion JSON records the same digest. Within the ZIP, `EXPORT_INVENTORY.json` lists hashes for every included payload file.

## Archive provenance

Training used commit `e3d110f`. The original downloaded cluster export was verified against its receipt before publication. Its SHA256 was:

```text
a61431fcc37fefcfc488cb149e8bf68b9e42025ac9fcac8653ee59d6e6ca5d17
```

To keep shared documentation in English, the publication archive omits exactly one historical file: `reproducibility/code/CLEAN_STUDY.md`. All other original payload files, including all metrics, predictions, histories, manifests, configuration and data evidence, remain byte-for-byte unchanged. The ZIP inventory was refreshed and `PUBLICATION_NOTES.json` records the source archive and the omitted file hash. The published ZIP therefore has a different hash from the original cluster download. The original local download remains unchanged.

The omitted guide can be recovered from Git history at commit `e3d110f`; its English translation is now [CLEAN_STUDY.md](../../CLEAN_STUDY.md). Study metadata hashes describe the original training snapshot, not the translated guide. Use the original study code version when resuming or re-exporting frozen runs.

## Interpretation

The final in-domain exact-triplet F1 is 49.68 ± 0.36 for Standard and 48.98 ± 0.25 for Weighted, on a 0–100 scale. Standard deviations describe five-seed variation, not confidence intervals. Results are exploratory; no significance tests have been performed. Search seeds 13 and 42 were reused in the final comparisons. Several cross-domain exact-triplet scores are zero because of category-vocabulary coverage limitations; read the report before comparing those scores.

## Documentation language

Shared repository documentation and reports are maintained in English. Personal Chinese translations remain outside the repository. Original experimental annotations and evidence are not translated or altered.
