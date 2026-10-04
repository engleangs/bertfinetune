import unittest

import pandas as pd

from src.team_taxonomy import summarize_taxonomy


class TaxonomySummaryTests(unittest.TestCase):
    @staticmethod
    def frame():
        return pd.DataFrame([
            {"setting": "In-domain", "held_out_domain": "all", "model": "standard",
             "seed": 13, "gold_triplets": 10, "correct": 2, "term": 5, "category": 2, "sentiment": 1},
            {"setting": "In-domain", "held_out_domain": "all", "model": "standard",
             "seed": 42, "gold_triplets": 10, "correct": 4, "term": 3, "category": 2, "sentiment": 1},
        ])

    def test_primary_shares_and_conditional_rates_have_different_denominators(self):
        row = summarize_taxonomy(self.frame()).iloc[0]
        self.assertEqual(row.dominant_primary_error, "term")
        self.assertEqual(row.term_mean, 4)
        self.assertAlmostEqual(row.term_error_share, 8 / 14)
        self.assertAlmostEqual(row.term_failure_rate, 8 / 20)
        self.assertAlmostEqual(row.category_failure_given_term, 4 / 12)
        self.assertAlmostEqual(row.sentiment_failure_given_term_category, 2 / 8)
        self.assertAlmostEqual(sum(row[f"{name}_error_share"] for name in ["term", "category", "sentiment"]), 1)

    def test_missing_predictions_leave_category_and_sentiment_rates_undefined(self):
        frame = self.frame().iloc[[0]].copy()
        frame[["correct", "term", "category", "sentiment"]] = [0, 10, 0, 0]
        row = summarize_taxonomy(frame).iloc[0]
        self.assertTrue(pd.isna(row.category_failure_given_term))
        self.assertTrue(pd.isna(row.sentiment_failure_given_term_category))

    def test_rejects_nonpartition_and_duplicate_run_keys(self):
        bad = self.frame()
        bad.loc[0, "sentiment"] = 2
        with self.assertRaisesRegex(ValueError, "partition"):
            summarize_taxonomy(bad)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_taxonomy(pd.concat([self.frame(), self.frame().iloc[[0]]]))


if __name__ == "__main__":
    unittest.main()
