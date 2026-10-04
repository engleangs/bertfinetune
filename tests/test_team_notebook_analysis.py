import unittest

import pandas as pd

from analyze_team_notebook import audit_corpus, pair_runs
from src.data import Example


class PairedRunTests(unittest.TestCase):
    @staticmethod
    def sample_runs():
        rows = []
        for setting, domain in [("In-domain", "all"), ("LODO", "laptop")]:
            for seed, baseline, change in [(13, 0.25, -0.10), (42, 0.40, 0.05)]:
                for model in ("standard", "weighted"):
                    value = baseline + (change if model == "weighted" else 0)
                    rows.append({
                        "setting": setting,
                        "held_out_domain": domain,
                        "model": model,
                        "seed": seed,
                        "precision": value,
                        "recall": value,
                        "f1": value,
                        "aspect_f1": value,
                        "aspect_sentiment_f1": value,
                    })
        return pd.DataFrame(rows)

    def test_pairs_multiple_seeds_by_condition_and_seed(self):
        # Changing row order must not pair seed 13 with seed 42.
        frame = self.sample_runs().sample(frac=1, random_state=9)
        paired = pair_runs(frame)
        self.assertEqual(len(paired), 4)
        self.assertFalse(paired.duplicated(["setting", "held_out_domain", "seed"]).any())
        for row in paired.to_dict("records"):
            expected_standard, expected_change = {13: (0.25, -0.10), 42: (0.40, 0.05)}[row["seed"]]
            for metric in ("precision", "recall", "f1", "aspect_f1", "aspect_sentiment_f1"):
                self.assertAlmostEqual(row[f"{metric}_standard"], expected_standard)
                self.assertAlmostEqual(row[f"{metric}_weighted"], expected_standard + expected_change)
                self.assertAlmostEqual(row[f"{metric}_change"], expected_change)

    def test_rejects_missing_config_seed_instead_of_partial_pairing(self):
        frame = self.sample_runs()
        frame = frame[~((frame.model == "weighted") & (frame.seed == 42) & (frame.setting == "LODO"))]
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            pair_runs(frame)

    def test_rejects_duplicate_logical_config_seed(self):
        frame = self.sample_runs()
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pair_runs(frame)


class CorpusAuditTests(unittest.TestCase):
    def test_counts_raw_unique_retained_and_null_per_sentence(self):
        explicit = ("screen", "DISPLAY", "positive")
        implicit = ("NULL", "DEVICE", "negative")
        conflict = ("NULL", "DEVICE", "conflict")
        corpus = {("laptop", "test"): [
            Example("The screen is good.", [
                explicit, explicit, implicit, implicit,
                ("missing term", "DEVICE", "neutral"),
            ], "laptop"),
            # The same triplet in another sentence is another gold target.
            Example("A screen is useful.", [explicit], "laptop"),
            Example("It is complicated.", [conflict, conflict], "laptop"),
            Example("No annotations.", [], "laptop"),
        ]}
        sentiment, nulls, null_labels = audit_corpus(corpus)
        row = nulls.iloc[0].to_dict()
        self.assertEqual(row["examples"], 4)
        self.assertEqual(row["raw_triplets"], 8)
        self.assertEqual(row["unique_triplets"], 5)
        self.assertEqual(row["retained_explicit_triplets"], 2)
        self.assertEqual(row["null_triplets"], 2)
        self.assertAlmostEqual(row["null_percentage"], 40.0)
        self.assertAlmostEqual(row["scope_recall_upper_bound"], 2 / 5)
        self.assertEqual(row["mixed_examples"], 1)
        self.assertEqual(row["explicit_only_examples"], 1)
        self.assertEqual(row["null_only_examples"], 1)
        self.assertEqual(row["empty_examples"], 1)

        expected = {
            "raw": {"positive": 3, "negative": 2, "neutral": 1, "conflict": 2},
            "deduplicated": {"positive": 2, "negative": 1, "neutral": 1, "conflict": 1},
            "retained": {"positive": 2, "negative": 0, "neutral": 0, "conflict": 0},
        }
        for scope, counts in expected.items():
            rows = sentiment[sentiment.scope == scope].set_index("sentiment")
            self.assertEqual(rows["count"].to_dict(), counts)
            self.assertTrue((rows["total"] == sum(counts.values())).all())
            self.assertAlmostEqual(rows["percentage"].sum(), 100.0)
        self.assertEqual(
            {(r["category"], r["sentiment"]): r["count"] for r in null_labels.to_dict("records")},
            {("DEVICE", "negative"): 1, ("DEVICE", "conflict"): 1},
        )

    def test_empty_annotations_produce_zero_counts_without_division_error(self):
        corpus = {("laptop", "dev"): [Example("No annotations.", [], "laptop")]}
        sentiment, nulls, null_labels = audit_corpus(corpus)
        self.assertEqual(len(sentiment), 12)  # Four classes in each of three scopes.
        self.assertTrue((sentiment["count"] == 0).all())
        self.assertTrue((sentiment["total"] == 0).all())
        self.assertTrue((sentiment["percentage"] == 0).all())
        self.assertEqual(nulls.iloc[0]["empty_examples"], 1)
        self.assertEqual(nulls.iloc[0]["scope_recall_upper_bound"], 0.0)
        self.assertEqual(nulls.iloc[0]["null_percentage"], 0.0)
        self.assertTrue(null_labels.empty)


if __name__ == "__main__":
    unittest.main()
