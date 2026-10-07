"""Common-gold comparisons preserve false positives and validate scope."""

import unittest

from experiment_reporting.phase_comparison import align, primary_outcomes, score, verify_metrics


class PhaseComparisonTests(unittest.TestCase):
    def records(self, gold, predicted, null_gold=(), null_predicted=()):
        return {("food", 0): {"domain": "food", "example_id": 0, "sentence": "good food",
                              "gold_triplets": gold, "predicted_triplets": predicted,
                              "gold_null_triplets": list(null_gold), "predicted_null_triplets": list(null_predicted)}}

    def test_old_predictions_are_not_filtered_to_current_gold(self):
        current = self.records([["food", "A", "positive"]], [])
        old = self.records([["food", "A", "positive"], ["good", "B", "positive"]],
                           [["food", "A", "positive"], ["good", "B", "positive"]])
        align(current, old)
        result = score(current, old)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_duplicate_triplets_and_null_scopes_are_scored_as_sets(self):
        triplet = ["food", "A", "positive"]
        null = ["NULL", "B", "negative"]
        records = self.records([triplet, triplet], [triplet, triplet], [null], [])
        self.assertEqual(score(records, records)["f1"], 1)
        self.assertEqual(score(records, records, "null")["fn"], 1)
        self.assertAlmostEqual(score(records, records, "combined")["f1"], 2 / 3)

    def test_different_sentences_and_current_gold_are_rejected(self):
        current = self.records([["food", "A", "positive"]], [])
        changed = self.records([["food", "B", "positive"]], [])
        with self.assertRaisesRegex(ValueError, "gold annotations"):
            align(current, changed, same_gold=True)
        changed[("food", 0)]["sentence"] = "other food"
        with self.assertRaisesRegex(ValueError, "Sentence contents"):
            align(current, changed)

    def test_first_failure_partitions_gold_and_saved_metric_mismatch_fails(self):
        gold = [["one", "A", "positive"], ["two", "A", "positive"],
                ["three", "A", "positive"], ["four", "A", "positive"]]
        predicted = [["one", "A", "positive"], ["three", "B", "positive"], ["four", "A", "negative"]]
        records = self.records(gold, predicted)
        outcomes = primary_outcomes(records, records)
        self.assertEqual(outcomes, {"correct": 1, "term": 1, "category": 1, "sentiment": 1})
        report = {"taxonomy": {"primary_outcome_counts": outcomes, "gold_triplets": 4}}
        for scope in ("explicit", "null", "combined"):
            values = score(records, records, scope)
            report[scope] = {"true_positives": values["tp"], "false_positives": values["fp"],
                             "false_negatives": values["fn"],
                             **{"micro_" + key: values[key] for key in ("f1", "precision", "recall")}}
        verify_metrics(records, report)
        report["combined"]["false_positives"] += 1
        with self.assertRaisesRegex(ValueError, "Saved combined false_positives"):
            verify_metrics(records, report)


if __name__ == "__main__":
    unittest.main()
