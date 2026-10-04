import json
import unittest

from src.team_diagnostics import analyze_records, category_frequency_band


def record(gold, predicted, **metadata):
    return {"gold_triplets": gold, "predicted_triplets": predicted, **metadata}


class PrimaryErrorTests(unittest.TestCase):
    def test_hierarchy_keeps_independent_both_error_flags(self):
        records = [record(
            [("screen", "DISPLAY", "positive"), ("price", "PRICE", "negative")],
            [("screen", "BATTERY", "negative")],
            domain="laptop", example_id=7, sentence="The screen and price disappointed.",
        )]
        summary, rows = analyze_records(records, {"DISPLAY": 5, "PRICE": 6}, {"DISPLAY", "PRICE"})
        by_term = {row["gold"][0]: row for row in rows}
        screen = by_term["screen"]
        self.assertEqual(screen["outcome"], "category")
        self.assertFalse(screen["category_correct_on_term"])
        self.assertFalse(screen["sentiment_correct_on_term"])
        self.assertIsNone(screen["sentiment_correct_on_term_category"])
        self.assertEqual(by_term["price"]["outcome"], "term")
        self.assertEqual(summary["primary_outcome_counts"], {
            "correct": 0, "term": 1, "category": 1, "sentiment": 0,
        })
        self.assertEqual(summary["conditional_accuracy"]["sentiment_on_matched_term"]["denominator"], 1)
        self.assertEqual(screen["domain"], "laptop")
        self.assertEqual(screen["example_id"], 7)
        json.dumps(summary)
        json.dumps(rows)

    def test_correct_sentiment_is_preserved_when_category_masks_triplet(self):
        summary, rows = analyze_records(
            [record([("screen", "DISPLAY", "positive")], [("screen", "DEVICE", "positive")])],
            {"DEVICE": 101}, {"DEVICE"},
        )
        self.assertEqual(rows[0]["outcome"], "category")
        self.assertTrue(rows[0]["sentiment_correct_on_term"])
        self.assertFalse(rows[0]["source_category_seen"])
        self.assertEqual(rows[0]["rarity"], "unseen")
        self.assertEqual(summary["category_mismatch_with_correct_term_sentiment"], 1)
        self.assertEqual(summary["metrics"]["full_exact_triplet"]["f1"], 0.0)
        self.assertEqual(summary["metrics"]["aspect_plus_sentiment"]["f1"], 1.0)
        self.assertEqual(summary["metrics"]["term_plus_category"]["f1"], 0.0)

    def test_multiple_candidates_choose_exact_then_category_then_lexicographic(self):
        records = [record(
            [("x", "A", "positive"), ("y", "B", "positive"), ("z", "B", "positive")],
            [
                ("x", "A", "negative"), ("x", "A", "positive"),
                ("y", "A", "positive"), ("y", "B", "negative"),
                ("z", "D", "negative"), ("z", "A", "negative"),
            ],
        )]
        summary, rows = analyze_records(records, {"A": 1, "B": 1}, {"A", "B"})
        by_term = {row["gold"][0]: row for row in rows}
        self.assertEqual(by_term["x"]["outcome"], "correct")
        self.assertEqual(by_term["x"]["predicted"], ["x", "A", "positive"])
        self.assertEqual(by_term["y"]["outcome"], "sentiment")
        self.assertEqual(by_term["y"]["predicted"], ["y", "B", "negative"])
        self.assertTrue(by_term["y"]["sentiment_correct_on_term"])
        self.assertFalse(by_term["y"]["sentiment_correct_on_term_category"])
        self.assertEqual(by_term["z"]["predicted"], ["z", "A", "negative"])
        self.assertAlmostEqual(summary["conditional_accuracy"]["sentiment_on_matched_term"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["sentiment_confusion"]["chosen_prediction_accuracy"]["accuracy"], 1 / 3)
        self.assertEqual(summary["sentiment_confusion"]["labels"], ["negative", "positive"])
        self.assertEqual(summary["sentiment_confusion"]["matrix"], [[0, 0], [2, 1]])
        reversed_summary, reversed_rows = analyze_records(
            [record(records[0]["gold_triplets"][::-1], records[0]["predicted_triplets"][::-1])],
            {"A": 1, "B": 1}, {"A", "B"},
        )
        self.assertEqual(summary, reversed_summary)
        self.assertEqual(rows, reversed_rows)


class FrequencyAndCountingTests(unittest.TestCase):
    def test_bands_distinguish_rare_from_unseen_at_boundaries(self):
        counts = [0, 1, 5, 6, 20, 21, 100, 101]
        self.assertEqual([category_frequency_band(count) for count in counts], [
            "unseen", "rare", "rare", "low", "low", "medium", "medium", "frequent",
        ])
        summary, rows = analyze_records(
            [record([("a", "RARE", "positive"), ("b", "UNSEEN", "positive")], [])],
            {"RARE": 1}, {"RARE"},
        )
        self.assertEqual([row["rarity"] for row in rows], ["rare", "unseen"])
        self.assertEqual(summary["metrics"]["unseen_gold_triplets"], 1)
        self.assertEqual(summary["metrics"]["gold_category_coverage"], 0.5)

    def test_deduplicates_scores_and_counts_spurious_predictions_separately(self):
        correct = ("screen", "DISPLAY", "positive")
        wrong_label = ("screen", "DISPLAY", "negative")
        extra = ("battery", "BATTERY", "positive")
        records = [record([correct, correct], [correct, correct, wrong_label, extra, extra])]
        summary, rows = analyze_records(records, {"DISPLAY": 1}, {"DISPLAY"})
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["duplicate_counts"], {"gold": 1, "predicted": 2})
        self.assertEqual(summary["prediction_counts"], {
            "unique_triplets": 3,
            "incorrect_triplets": 2,
            "spurious_aspects": 1,
            "spurious_triplets": 1,
            "incorrect_label_triplets_on_gold_term": 1,
        })
        self.assertAlmostEqual(summary["metrics"]["full_exact_triplet"]["precision"], 1 / 3)
        self.assertEqual(summary["metrics"]["full_exact_triplet"]["recall"], 1.0)

    def test_empty_and_no_matching_terms_have_undefined_conditional_accuracy(self):
        summary, rows = analyze_records([], {}, set())
        self.assertEqual(rows, [])
        self.assertIsNone(summary["conditional_accuracy"]["sentiment_on_matched_term"]["accuracy"])
        self.assertEqual(summary["sentiment_confusion"]["matrix"], [])
        summary, _ = analyze_records([record([("x", "A", "positive")], [])], {"A": 1}, {"A"})
        self.assertEqual(summary["sentiment_confusion"]["unmatched_gold"], 1)
        self.assertIsNone(summary["conditional_accuracy"]["category_on_matched_term"]["accuracy"])

    def test_rejects_inconsistent_frequency_and_record_inputs(self):
        for invalid_count in (-1, 1.5, True):
            with self.subTest(invalid_count=invalid_count), self.assertRaises(ValueError):
                category_frequency_band(invalid_count)
        with self.assertRaisesRegex(ValueError, "positive-frequency"):
            analyze_records([], {"A": 1}, {"B"})
        with self.assertRaisesRegex(ValueError, "positive-frequency"):
            analyze_records([], {"A": 0}, {"A"})
        with self.assertRaisesRegex(ValueError, "missing"):
            analyze_records([{"gold_triplets": []}], {}, set())
        with self.assertRaisesRegex(ValueError, "Invalid"):
            analyze_records([record([("x", "A")], [])], {}, set())


if __name__ == "__main__":
    unittest.main()
