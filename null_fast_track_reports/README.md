# NULL fast-track reports

These folders contain small, shareable snapshots of the new techniques 1–5 study: report Markdown, comparison PNG, result/configuration JSON and the frozen shortlist when available. Models and full prediction/probability files stay in Git-ignored `artifacts/`.

- [Run guide, parameters and protocol](../null-fast-track-guide-2026-10-06.md).
- [Default study report](null_fast_track/report.md).
- [Historical final study](../final_results/README.md).

Run the default staged search and two-seed confirmation:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --first-seeds 2 --execute
```

Refresh the snapshot without training:

```powershell
.venv/Scripts/python.exe run_null_fast_track.py --step report
```

The current snapshot begins with development-only calibration. New training comparisons populate as runs finish. Smoke scores are excluded. Implementation status and a nonzero NULL F1 do not by themselves demonstrate a combined-F1 improvement.
