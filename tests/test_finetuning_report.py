"""Presentation export gates use synthetic saved results; no model is loaded."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import build_finetuning_report as report
from src.experiment_config import TrialConfig, candidate_recipes


class FineTuningReportTests(unittest.TestCase):
    def setUp(self):
        self.base = report.ROOT / "artifacts/report_tests"
        self.root = self.base / ("fixture-" + uuid.uuid4().hex)
        self.input = self.root / "input"
        self.output = self.root / "output"
        self.input.mkdir(parents=True)

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.base.resolve())
        shutil.rmtree(self.root)

    def add_result(self, recipe, seed, status="dev_complete"):
        settings = next(r for r in candidate_recipes("loss") if r["name"] == recipe)
        cfg = TrialConfig(**settings, lr=3e-5)
        directory = self.input / cfg.trial_id / f"seed_{seed}"
        directory.mkdir(parents=True)
        manifest = {"status": status, "fingerprint": f"fixture-{recipe}-{seed}",
                    "checkpoint_sha256": "synthetic-checkpoint-hash",
                    "training": {"best_epoch": 5},
                    "identity": {"config": asdict(cfg), "seed": seed, "mode": "indomain",
                                 "held_out_domain": None, "smoke": False,
                                 "source_sha256": {"fixture": "code"},
                                 "source_data_sha256": {"fixture": "data"},
                                 "environment": {"fixture": True}, "resolved_device": "cpu"}}
        metrics = {"explicit": {"micro_f1": .5, "micro_precision": .5, "micro_recall": .5},
                   "explicit_boundary": {"f1": .5},
                   "combined": {"micro_f1": .5},
                   "null": {"micro_f1": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 1},
                   "taxonomy": {"gold_triplets": 2,
                                "primary_outcome_counts": {"correct": 1, "term": 1, "category": 0, "sentiment": 0},
                                "metrics": {"aspect_span": {"f1": .5}, "term_plus_category": {"f1": .5},
                                            "aspect_plus_sentiment": {"f1": .5}, "gold_category_coverage": 1}}}
        for name, value in (("manifest.json", manifest), ("dev_metrics.json", metrics),
                            ("audit.json", {"categories": ["A"], "sentiments": ["positive"]})):
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
        if status == "complete":
            (directory / "test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return directory

    def export(self, require_complete=False):
        args = ["build_finetuning_report.py", "--input", str(self.input), "--output", str(self.output)]
        if require_complete:
            args.append("--require-core-complete")
        with (patch("sys.argv", args), patch.object(report, "make_figure", return_value=[]),
              redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
            report.main()

    def test_partial_run_is_excluded_and_incomplete_core_cannot_freeze(self):
        self.add_result("standard", 13, "failed")
        self.add_result("mixed_025", 13)
        rows, _ = report.load_results(self.input)
        self.assertEqual([(r["recipe"], r["split"]) for r in rows], [("mixed_025", "dev")])
        with self.assertRaises(SystemExit) as failure:
            self.export(require_complete=True)
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse((self.output / "frozen_core.json").exists())

    def test_core_excludes_changed_budget_or_loss_family(self):
        self.add_result("mixed_025", 13)
        rows, _ = report.load_results(self.input)
        self.assertTrue(report.core_result(rows[0]))
        for field, value in (("batch_size", 32), ("loss_type", "focal"), ("weighted_ce_weight", .5)):
            changed = {**rows[0], "config": {**rows[0]["config"], field: value}}
            self.assertFalse(report.core_result(changed))

    def test_complete_core_freezes_once_and_refuses_changed_checkpoints(self):
        for recipe in report.RECIPES:
            for seed in (13, 42, 123, 2024, 777):
                directory = self.add_result(recipe, seed)
        self.export(require_complete=True)
        freeze = self.output / "frozen_core.json"
        original = freeze.read_bytes()
        self.assertEqual(len(json.loads(original)), 20)
        self.export(require_complete=True)
        self.assertEqual(freeze.read_bytes(), original)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checkpoint_sha256"] = "changed-synthetic-checkpoint"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.export(require_complete=True)
        self.assertEqual(freeze.read_bytes(), original)

    def test_saved_test_scores_are_separate_and_pairs_require_matching_seeds(self):
        self.add_result("standard", 13, "complete")
        self.add_result("mixed_025", 13, "complete")
        self.add_result("mixed_025", 42)
        self.export()
        data = json.loads((self.output / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(r["split"] == "dev" for r in data["results"]), 3)
        self.assertEqual(sum(r["split"] == "test" for r in data["results"]), 2)
        self.assertEqual(len(data["paired_changes"]), 2)
        self.assertTrue(all([p["seed"] for p in pair["pairs"]] == [13] for pair in data["paired_changes"]))


if __name__ == "__main__":
    unittest.main()
