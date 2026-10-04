"""Guard against misleading pilot comparisons and incomplete-run scores."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from experiment_reporting.analysis import build_analysis, cohort_identity, paired_comparisons, summaries
from experiment_reporting.report import headline, parameter_rows
from src.experiment_config import TrialConfig


def run(name, seed, f1, completed=True, cohort="explicit", null=False):
    cfg = asdict(TrialConfig(name, "mixed" if name.startswith("mixed") else "standard", null_head=null))
    return {"trial_id": name + "_trial", "recipe": name, "seed": seed, "cohort": cohort, "cohort_label": cohort,
            "vocabulary_scope": "explicit-null" if cohort == "null" else "explicit", "config": cfg,
            "completed": completed, "provisional": not completed, "status": "dev_complete" if completed else "running",
            "best_epoch": 2, "epochs_recorded": 2, "num_categories": 2, "sentiments": ["positive"], "training": {},
            "metrics": {"explicit_f1": f1, "precision": .5, "recall": .5, "macro_f1": .1, "null_f1": .2 if null else 0,
                        "combined_f1": .4 if null else .2, "boundary_f1": .6, "term_f1": .6,
                        "term_category_f1": .5, "term_sentiment_f1": .55, "coverage": 1, "known_f1": f1,
                        "rare_recall": None, "null_threshold": .7, "outcomes": {"correct": 4, "term": 4, "category": 1, "sentiment": 1},
                        "gold_triplets": 10}}


class ComparisonTests(unittest.TestCase):
    def test_completed_and_provisional_seeds_are_not_pooled(self):
        rows = summaries([run("standard", 13, .4), run("standard", 42, .9, completed=False)])
        self.assertEqual(rows[0]["explicit_f1_mean"], .4)
        self.assertEqual(rows[0]["scored_seeds"], [13])
        self.assertIsNone(rows[0]["explicit_f1_std"])

    def test_paired_delta_uses_same_seeds_and_requires_finished_pairs(self):
        rows = [run("standard", 13, .30), run("standard", 42, .40),
                run("mixed_033", 13, .35), run("mixed_033", 42, .45, completed=False)]
        comparison = next(r for r in paired_comparisons(rows, [13, 42], .02) if r["trial_id"] == "mixed_033_trial")
        self.assertEqual([p["seed"] for p in comparison["pairs"]], [13])
        self.assertFalse(comparison["ready"])
        self.assertEqual(comparison["verdict"], "Awaiting matched completed seeds")
        rows[-1]["completed"] = True
        comparison = next(r for r in paired_comparisons(rows, [13, 42], .02) if r["trial_id"] == "mixed_033_trial")
        self.assertAlmostEqual(comparison["delta_f1"], .05)
        self.assertEqual(comparison["verdict"], "Promising pilot; meets +2 pp target")

    def test_null_extension_uses_its_vocabulary_matched_control(self):
        rows = [run("standard", 13, .9), run("standard_null_vocab", 13, .3, cohort="null"),
                run("standard_null", 13, .32, cohort="null", null=True)]
        comparison = next(r for r in paired_comparisons(rows, [13], .02) if r["trial_id"] == "standard_null_trial")
        self.assertEqual(comparison["control"], "standard_null_vocab")
        self.assertAlmostEqual(comparison["delta_f1"], .02)
        self.assertAlmostEqual(comparison["delta_null_f1"], .2)

    def test_budget_task_and_data_changes_create_separate_cohorts(self):
        identity = {"config": asdict(TrialConfig("standard", "standard")), "mode": "indomain", "source_data_sha256": {"a": "hash"}}
        audit = {"categories": ["A"], "sentiments": ["positive"]}
        original = cohort_identity(identity, audit)
        for change in ("lr", "drop_conflict", "null_head"):
            altered = deepcopy(identity)
            altered["config"][change] = {"lr": 1e-5, "drop_conflict": True, "null_head": True}[change]
            self.assertNotEqual(original, cohort_identity(altered, audit))
        altered = deepcopy(identity)
        altered["source_data_sha256"]["a"] = "different"
        self.assertNotEqual(original, cohort_identity(altered, audit))
        altered = deepcopy(identity)
        altered["config"]["loss_type"] = "focal"
        self.assertEqual(original, cohort_identity(altered, audit))

    def test_inactive_hyperparameters_are_not_presented_as_applied(self):
        params = parameter_rows({"runs": [run("standard", 13, .3)]})[0]
        self.assertIsNone(params["focal_gamma"])
        self.assertIsNone(params["mixture_alpha"])
        self.assertIsNone(params["null_loss_weight"])
        self.assertEqual(params["lr"], 2e-5)

    def test_running_snapshot_selects_best_complete_history_epoch(self):
        manifest = {"status": "running", "fingerprint": "fixture", "identity": {
            "mode": "indomain", "held_out_domain": None, "seed": 13, "smoke": False,
            "config": asdict(TrialConfig("standard", "standard"))}}
        report = {"explicit": {"micro_f1": .3, "micro_precision": .4, "micro_recall": .2}, "unweighted_loss": 2}
        better = deepcopy(report)
        better["explicit"]["micro_f1"] = .4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "standard" / "seed_13"
            dest.mkdir(parents=True)
            (dest / "manifest.json").write_text(json.dumps(manifest))
            (dest / "history.json").write_text(json.dumps([
                {"epoch": 1, "development": better}, {"epoch": 2, "development": report}]))
            # A stale best-metrics file must not override the consistent history.
            (dest / "dev_metrics.json").write_text(json.dumps(report))
            analysis = build_analysis(root, "indomain", None, ["standard", "weighted"], [13, 42])
            self.assertEqual(analysis["runs"][0]["best_epoch"], 1)
            self.assertEqual(analysis["runs"][0]["metrics"]["explicit_f1"], .4)
            self.assertEqual(analysis["completed_runs"], 0)
            self.assertIn("no completed comparison", headline(analysis))
            self.assertEqual(analysis["grid"][-1]["status"], "not started")


if __name__ == "__main__":
    unittest.main()
