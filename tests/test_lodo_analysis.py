import unittest

import pandas as pd

from analyze_lodo_results import (
    build_paired_summary,
    expected_and_missing,
)


class LodoAnalysisTests(unittest.TestCase):
    def test_missing_keys_and_complete_paired_domain(self):
        rows = [
            {
                "held_out_domain": "food",
                "config": config_name,
                "seed": seed,
                "test_micro_f1": score,
            }
            for config_name, scores in (
                ("standard", [0.20, 0.30]),
                ("weighted", [0.25, 0.36]),
            )
            for seed, score in zip((1, 2), scores)
        ]
        frame = pd.DataFrame(rows)
        expected, missing = expected_and_missing(
            frame, ["food"], ["standard", "weighted"], [1, 2],
        )
        self.assertEqual(len(expected), 4)
        self.assertEqual(missing, [])

        summary = build_paired_summary(frame, ["food"], [1, 2])[0]
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["matched_seeds"], 2)
        self.assertAlmostEqual(summary["mean_difference"], 0.055)

    def test_partial_domain_is_not_final_ready(self):
        frame = pd.DataFrame(
            [
                {
                    "held_out_domain": "sight",
                    "config": "standard",
                    "seed": 1,
                    "test_micro_f1": 0.1,
                },
                {
                    "held_out_domain": "sight",
                    "config": "weighted",
                    "seed": 1,
                    "test_micro_f1": 0.2,
                },
            ]
        )
        summary = build_paired_summary(frame, ["sight"], [1, 2])[0]
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["matched_seeds"], 1)
        self.assertIsNone(summary["p_value"])


if __name__ == "__main__":
    unittest.main()
