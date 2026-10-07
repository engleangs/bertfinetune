"""Meaningful checks for matching gold, score projections and portable exports."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile

from phase2.analysis import align_gold, mean_sd, score_records
from phase2.compare_results import extract_evidence, in_domain_comparison, verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def record(gold, predicted, text="food and service"):
    return {"domain": "restaurant", "example_id": 7, "sentence": text,
            "gold_triplets": gold, "predicted_triplets": predicted}


class ComparisonTests(unittest.TestCase):
    def test_matching_gold_keeps_predictions_on_excluded_annotations_as_false_positives(self):
        old = record([["food", "quality", "positive"], ["service", "service", "negative"]],
                     [["food", "quality", "positive"], ["service", "service", "negative"]])
        new = record([["food", "quality", "positive"]], [])
        aligned = align_gold([old], [new])
        scored = score_records(aligned, ["quality", "service"])["explicit"]
        self.assertEqual((scored["tp"], scored["fp"], scored["fn"]), (1, 1, 0))
        self.assertEqual(old["gold_triplets"], [["food", "quality", "positive"], ["service", "service", "negative"]])

    def test_sentence_and_identity_must_match_before_rescoring(self):
        old = record([], [])
        for changed in (record([], [], text="changed"), {**old, "example_id": 9}):
            with self.assertRaises(ValueError):
                align_gold([old], [changed])
        with self.assertRaises(ValueError):
            align_gold([old, old], [old])

    def test_projections_deduplicate_per_sentence_without_hiding_category_errors(self):
        row = record([["food", "quality", "positive"], ["food", "style", "positive"]],
                     [["food", "price", "positive"], ["food", "price", "positive"]])
        scored = score_records([row], ["quality"])
        self.assertEqual(scored["explicit"]["tp"], 0)
        self.assertEqual(scored["explicit"]["fp"], 1)
        self.assertEqual(scored["term"]["f1"], 1)
        self.assertEqual(scored["term"]["gold"], 1)
        self.assertEqual(scored["term_sentiment"]["f1"], 1)
        self.assertEqual(scored["taxonomy"]["category"], 2)
        self.assertEqual(scored["coverage"], .5)

    def test_first_failure_counts_partition_gold(self):
        row = record([["a", "cat", "positive"], ["b", "cat", "positive"],
                      ["c", "cat", "positive"], ["d", "cat", "positive"]],
                     [["a", "cat", "positive"], ["c", "other", "negative"], ["d", "cat", "negative"]])
        self.assertEqual(score_records([row], ["cat"])["taxonomy"],
                         {"correct": 1, "term": 1, "category": 1, "sentiment": 1})

    def test_standard_deviation_is_sample_sd_and_unavailable_for_one_seed(self):
        self.assertEqual(mean_sd([1, 3]), {"mean": 2, "sd": 2**.5})
        self.assertIsNone(mean_sd([1])["sd"])
        with self.assertRaises(ValueError):
            mean_sd([])

    def test_finalist_duplicate_or_missing_seed_is_rejected(self):
        data = json.loads((ROOT / "final_presentation_results/results.json").read_text(encoding="utf-8"))
        target = next(row for row in data["runs"] if row["split"] == "test")
        changed = deepcopy(data)
        changed["runs"].append(deepcopy(target))
        with self.assertRaises(ValueError):
            in_domain_comparison(changed)
        changed = deepcopy(data)
        changed["runs"] = [row for row in changed["runs"]
                           if not (row["trial_id"] == target["trial_id"] and row["seed"] == target["seed"]
                                   and row["split"] == "test" and row["study"] == target["study"])]
        with self.assertRaises(ValueError):
            in_domain_comparison(changed)


class BundleSafetyTests(unittest.TestCase):
    def setUp(self):
        self.test_root = (ROOT / "artifacts/phase2_tests").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.fixture = (self.test_root / uuid.uuid4().hex).resolve()
        self.fixture.mkdir()

    def tearDown(self):
        if not self.fixture.is_relative_to(self.test_root):
            raise ValueError("Test cleanup escaped its workspace directory")
        shutil.rmtree(self.fixture)

    def test_manifest_rejects_changed_input(self):
        from phase2.compare_results import sha256
        path = self.fixture / "results.json"
        path.write_text("original", encoding="utf-8")
        (self.fixture / "bundle_manifest.json").write_text(json.dumps(
            {"files": [{"path": "results.json", "bytes": path.stat().st_size, "sha256": sha256(path)}]}),
            encoding="utf-8")
        self.assertEqual(verify_bundle(self.fixture), 1)
        path.write_text("modified", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_bundle(self.fixture)

    def test_extraction_rejects_path_traversal_and_model_files(self):
        evidence = self.fixture / "evidence"
        evidence.mkdir()
        for member in ("../../escape.json", "nested/best.pt"):
            with zipfile.ZipFile(evidence / "phase2_outputs.zip", "w") as archive:
                archive.writestr(member, "{}")
            with self.assertRaises(ValueError):
                extract_evidence(self.fixture)
        self.assertFalse((self.fixture / "escape.json").exists())


if __name__ == "__main__":
    unittest.main()
