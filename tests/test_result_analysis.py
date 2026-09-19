import unittest

import pandas as pd

from src.result_analysis import derive_crossdomain_metrics, validate_complete_matrix


class CrossDomainMetricTests(unittest.TestCase):
    def test_derives_seen_coverage_and_projected_scores(self):
        gold = [
            [("screen", "KNOWN", "positive"), ("price", "UNSEEN", "negative")],
            [],
        ]
        predicted = [
            [("screen", "KNOWN", "positive"), ("price", "KNOWN", "negative")],
            [],
        ]
        result = derive_crossdomain_metrics(gold, predicted, {"KNOWN"})
        self.assertEqual(result["gold_category_coverage"], 0.5)
        self.assertEqual(result["unseen_categories"], ["UNSEEN"])
        self.assertAlmostEqual(result["full_exact_triplet"]["f1"], 0.5)
        self.assertAlmostEqual(result["source_known_exact_triplet"]["f1"], 2 / 3)
        self.assertEqual(result["aspect_span"]["f1"], 1.0)
        self.assertEqual(result["aspect_plus_sentiment"]["f1"], 1.0)


class MatrixValidationTests(unittest.TestCase):
    def test_accepts_complete_matrix_and_rejects_missing_run(self):
        rows = [
            {"mode": mode, "config": config, "seed": seed, "status": "complete"}
            for mode in ("in", "cross")
            for config in ("standard", "weighted")
            for seed in (1, 2)
        ]
        frame = pd.DataFrame(rows)
        validate_complete_matrix(frame, ("in", "cross"), ("standard", "weighted"), (1, 2))
        with self.assertRaises(ValueError):
            validate_complete_matrix(frame.iloc[:-1], ("in", "cross"), ("standard", "weighted"), (1, 2))


if __name__ == "__main__":
    unittest.main()

