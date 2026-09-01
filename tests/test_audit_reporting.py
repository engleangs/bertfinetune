import csv
import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import analyze_results
from src.data import summarize_tokenized_targets


class TokenizedTargetSummaryTests(unittest.TestCase):
    def test_counts_alignment_misses_and_unknown_targets(self):
        dataset = [
            {
                "gold_triplets": [
                    ("battery", "BATTERY", "positive"),
                    ("screen", "DISPLAY", "negative"),
                    ("late term", "BATTERY", "negative"),
                ],
                "span_boundaries": [(1, 1), (3, 3)],
                "category_labels": torch.tensor([0, -100]),
                "sentiment_labels": torch.tensor([0, 1]),
            },
            {
                "gold_triplets": [
                    ("keyboard", "INPUT", "positive"),
                ],
                "span_boundaries": [(1, 1)],
                "category_labels": torch.tensor([-100]),
                "sentiment_labels": torch.tensor([-100]),
            },
        ]

        summary = summarize_tokenized_targets(dataset)

        self.assertEqual(
            summary,
            {
                "gold_triplets": 4,
                "token_aligned_span_targets": 3,
                "token_alignment_or_truncation_misses": 1,
                "unknown_category_targets": 2,
                "unknown_sentiment_targets": 1,
            },
        )


class AnalyzeResultsAuditTests(unittest.TestCase):
    def test_prints_paired_seed_rows_and_warns_about_missing_planned_seeds(self):
        rows = [
            {
                "mode": "indomain",
                "config": "standard",
                "seed": 1,
                "status": "complete",
                "test_micro_f1": 0.20,
            },
            {
                "mode": "indomain",
                "config": "weighted",
                "seed": 1,
                "status": "complete",
                "test_micro_f1": 0.25,
            },
            {
                "mode": "indomain",
                "config": "standard",
                "seed": 3,
                "status": "complete",
                "test_micro_f1": 0.40,
            },
            {
                "mode": "indomain",
                "config": "weighted",
                "seed": 3,
                "status": "complete",
                "test_micro_f1": 0.35,
            },
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "results.csv"
            with results_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            output = io.StringIO()
            with (
                patch.object(analyze_results.cfg, "MODES", ["indomain"]),
                patch.object(analyze_results.cfg, "SEEDS", [1, 2, 3, 4]),
                redirect_stdout(output),
            ):
                analyze_results.analyze(
                    metric_col="test_micro_f1", results_csv=results_path,
                )

        rendered = output.getvalue()
        self.assertRegex(
            rendered,
            re.compile(
                r"^.*seed\s*[=:]?\s*1\b.*standard.*0\.2000.*"
                r"weighted.*0\.2500.*(?:diff|difference).*\+?0\.0500.*$",
                re.IGNORECASE | re.MULTILINE,
            ),
        )
        self.assertRegex(
            rendered,
            re.compile(
                r"^.*seed\s*[=:]?\s*3\b.*standard.*0\.4000.*"
                r"weighted.*0\.3500.*(?:diff|difference).*-0\.0500.*$",
                re.IGNORECASE | re.MULTILINE,
            ),
        )

        warning_lines = [
            line for line in rendered.splitlines()
            if "warn" in line.casefold() and "planned" in line.casefold()
        ]
        self.assertTrue(
            warning_lines,
            msg=f"Expected a warning about missing planned seeds in:\n{rendered}",
        )
        warning = " ".join(warning_lines)
        self.assertRegex(warning, r"\b2\b")
        self.assertRegex(warning, r"\b4\b")


if __name__ == "__main__":
    unittest.main()
