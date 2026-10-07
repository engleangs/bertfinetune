"""LODO waiting, provenance freezes and domain-aware reporting; no model training."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import continue_lodo_finetuning as flow
import evaluate_lodo_frozen as recovery
from experiment_reporting import lodo_report


class LodoContinuationTests(unittest.TestCase):
    def setUp(self):
        self.test_root = (flow.ROOT / "artifacts/lodo_continuation_tests").resolve()
        self.test_root.mkdir(parents=True, exist_ok=True)
        self.fixture = (self.test_root / uuid.uuid4().hex).resolve()
        self.fixture.mkdir()
        self.origin, self.variant = flow.selected_variant(
            flow.ROOT / "final_presentation_results/quick_selection.json")

    def tearDown(self):
        if not self.fixture.is_relative_to(self.test_root):
            raise ValueError("Test cleanup escaped its intended workspace directory")
        shutil.rmtree(self.fixture)

    def make_run(self, domain, seed, domains=("hotel", "food"), status="dev_complete"):
        folder = flow.directory(self.fixture, self.variant["trial_id"], domain, seed)
        folder.mkdir(parents=True)
        for name in flow.DEV_FILES:
            (folder / name).write_text("{}", encoding="utf-8")
        (folder / "best.pt").write_bytes(b"synthetic checkpoint, no model")
        metric = {
            "explicit": {"true_positives": 1, "false_positives": 1, "false_negatives": 1,
                         "micro_f1": .5, "micro_precision": .5, "micro_recall": .5, "macro_f1": .5},
            "null": {"micro_f1": 0.}, "combined": {"micro_f1": .4}, "null_threshold": .5,
            "taxonomy": {"gold_triplets": 2,
                         "primary_outcome_counts": {"correct": 1, "term": 1, "category": 0, "sentiment": 0},
                         "metrics": {"aspect_span": {"f1": .75}, "term_plus_category": {"f1": .5},
                                     "aspect_plus_sentiment": {"f1": .7}, "gold_category_coverage": .5},
                         "by_rarity": [{"rarity": "rare", "gold_triplets": 1,
                                        "primary_outcome_counts": {"correct": 1}, "error_rate": 0.},
                                       {"rarity": "unseen", "gold_triplets": 1,
                                        "primary_outcome_counts": {"correct": 0}, "error_rate": 1.}]}}
        flow.write_json(folder / "dev_metrics.json", metric)
        flow.write_json(folder / "history.json", [{"epoch": i} for i in range(1, 6)])
        manifest = {"status": status, "fingerprint": f"fixture_{domain}_{seed}",
                    "identity": {"config": deepcopy(self.variant["config"]), "seed": seed,
                                 "mode": "crossdomain", "held_out_domain": domain, "smoke": False,
                                 "source_domains": [d for d in domains if d != domain],
                                 "source_sha256": {}, "source_data_sha256": {}, "environment": {"fixture": True}},
                    "checkpoint_sha256": flow.file_hash(folder / "best.pt"),
                    "training": {"best_epoch": 5, "null_threshold": .5}}
        flow.write_json(folder / "manifest.json", manifest)
        # Invalid test JSON proves that completion/freezing reads development only.
        (folder / "test_metrics.json").write_text("do not read test during selection", encoding="utf-8")
        return folder, manifest, metric

    def test_commands_match_frozen_recipe_and_keep_remaining_seeds_separate(self):
        import run_experiments as trainer
        args = flow.parser().parse_args([])
        args.output = self.fixture
        train = flow.command(args, self.variant, "full", [42, 123, 2024, 777])
        parsed = trainer.parser().parse_args(train[3:])
        plan = trainer.make_plan(parsed)
        self.assertEqual(len(plan), 28)
        self.assertTrue(all(cfg.trial_id == self.variant["trial_id"] for cfg, _, _ in plan))
        self.assertEqual({seed for _, seed, _ in plan}, {42, 123, 2024, 777})
        self.assertEqual({domain for _, _, domain in plan}, set(flow.DOMAINS))
        test = flow.command(args, self.variant, "final", self.origin["seeds"])
        self.assertEqual(len(trainer.make_plan(trainer.parser().parse_args(test[3:]))), 35)

    def test_wait_requires_every_first_fold_and_failed_fold_stops_continuation(self):
        folder, manifest, _ = self.make_run("hotel", 13)
        with patch.object(flow, "DOMAINS", ("hotel", "food")):
            with self.assertRaisesRegex(ValueError, "first seven folds are incomplete"):
                flow.wait_for_first(self.fixture, self.variant, 13, wait=False, hours=1)
            manifest["status"] = "failed"
            manifest["error"] = "interrupted first queue"
            flow.write_json(folder / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "interrupted first queue"):
                flow.wait_for_first(self.fixture, self.variant, 13, wait=True, hours=1)

    def test_wait_polls_without_launching_a_training_process(self):
        pending = [{"domain": "hotel", "seed": 13, "status": "running"},
                   {"domain": "food", "seed": 13, "status": "pending"}]
        complete = [{**row, "status": "dev_complete"} for row in pending]
        updates = []
        with patch.object(flow, "DOMAINS", ("hotel", "food")), \
             patch.object(flow, "cohort_status", side_effect=[pending, complete]), \
             patch.object(flow.time, "sleep") as sleep, \
             patch.object(flow.subprocess, "run") as run:
            flow.wait_for_first(self.fixture, self.variant, 13, wait=True, hours=1, update=updates.append)
        self.assertEqual([u["completed_first_folds"] for u in updates], [0, 2])
        sleep.assert_called_once()
        run.assert_not_called()

    def test_development_freeze_is_immutable_and_verifies_checkpoint_contents(self):
        folders = [self.make_run(d, 13)[0] for d in ("hotel", "food")]
        path = self.fixture / "selection.json"
        with patch.object(flow, "DOMAINS", ("hotel", "food")):
            frozen = flow.freeze(self.fixture, self.variant, [13], path)
            original = path.read_bytes()
            flow.freeze(self.fixture, self.variant, [13], path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(len(frozen["runs"]), 2)
            (folders[0] / "best.pt").write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "Checkpoint contents changed"):
                flow.freeze(self.fixture, self.variant, [13], path)

    def test_freeze_rejects_held_out_domain_in_source_and_source_code_changes(self):
        folder, manifest, _ = self.make_run("hotel", 13, domains=("hotel",))
        manifest["identity"]["source_domains"] = ["hotel"]
        flow.write_json(folder / "manifest.json", manifest)
        with patch.object(flow, "DOMAINS", ("hotel",)):
            with self.assertRaisesRegex(ValueError, "Held-out domain appeared"):
                flow.freeze(self.fixture, self.variant, [13], self.fixture / "selection.json")
            manifest["identity"]["source_domains"] = []
            manifest["identity"]["source_sha256"] = {"config.py": "stale hash"}
            flow.write_json(folder / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Training code changed"):
                flow.freeze(self.fixture, self.variant, [13], self.fixture / "selection.json")

    def test_main_waits_then_trains_freezes_tests_and_reports_in_order(self):
        order = []
        args = ["--execute", "--wait-for-first", "--output", str(self.fixture),
                "--study-output", str(self.fixture / "study"), "--report-output", str(self.fixture / "report")]

        def run(command, **kwargs):
            order.append(command[command.index("--stage")+1])

        with patch.object(flow, "continuation_lock", return_value=nullcontext()), \
             patch.object(flow, "wait_for_first", side_effect=lambda *a, **k: order.append("wait")), \
             patch.object(flow.time, "sleep"), patch.object(flow.subprocess, "run", side_effect=run), \
             patch.object(flow, "freeze", side_effect=lambda *a, **k: order.append("freeze")), \
             patch.object(lodo_report, "write_report", side_effect=lambda *a: order.append("report")):
            flow.main(args)
        self.assertEqual(order, ["wait", "full", "freeze", "final", "report"])
        self.assertEqual(flow.read(self.fixture / "study/status.json")["status"], "complete")

    def test_incomplete_first_queue_prevents_training_and_test(self):
        args = ["--execute", "--wait-for-first", "--output", str(self.fixture),
                "--study-output", str(self.fixture / "study")]
        with patch.object(flow, "continuation_lock", return_value=nullcontext()), \
             patch.object(flow, "wait_for_first", side_effect=ValueError("first queue failed")), \
             patch.object(flow.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "first queue failed"):
                flow.main(args)
        run.assert_not_called()
        self.assertEqual(flow.read(self.fixture / "study/status.json")["status"], "failed")

    def test_recovery_of_completed_training_evaluates_without_retraining(self):
        commands = []
        args = ["--execute", "--wait-for-first", "--output", str(self.fixture),
                "--study-output", str(self.fixture / "study"), "--report-output", str(self.fixture / "report")]
        completed = [{"status": "complete"}, {"status": "dev_complete"}]
        with patch.object(flow, "continuation_lock", return_value=nullcontext()), \
             patch.object(flow, "wait_for_first"), patch.object(flow, "cohort_status", return_value=completed), \
             patch.object(flow.time, "sleep"), patch.object(flow.subprocess, "run", side_effect=lambda cmd, **kw: commands.append(cmd)), \
             patch.object(flow, "freeze"), patch.object(lodo_report, "write_report", return_value="report.md"):
            flow.main(args)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--stage")+1], "final")
        self.assertEqual(Path(commands[0][2]).name, "evaluate_lodo_frozen.py")
        self.assertIn("--freeze", commands[0])

    def test_locked_index_redirects_once_and_preserves_other_write_errors(self):
        calls = []
        metadata = self.fixture / "csv_recovery.json"

        def original(path, row):
            calls.append((path.name, row))
            if path.name == "development_results.csv":
                raise PermissionError("Windows sharing violation")
            return "saved"

        writer = recovery.resilient_csv_writer(original, metadata)
        master = self.fixture / "development_results.csv"
        self.assertEqual(writer(master, {"seed": 13}), "saved")
        self.assertEqual(writer(master, {"seed": 42}), "saved")
        self.assertEqual([p for p, _ in calls], ["development_results.csv",
                          "development_results_lodo_recovery.csv", "development_results_lodo_recovery.csv"])
        self.assertTrue(flow.read(metadata)["model_metrics_preserved"])
        with patch.object(recovery, "write_json") as write:
            denied = recovery.resilient_csv_writer(lambda *a: (_ for _ in ()).throw(PermissionError()), metadata)
            with self.assertRaises(PermissionError):
                denied(self.fixture / "test_metrics.json", {})
            write.assert_not_called()

    def test_recovery_rejects_training_and_a_different_frozen_matrix(self):
        import run_experiments as trainer
        args = flow.parser().parse_args([])
        args.output = self.fixture
        command = flow.command(args, self.variant, "final", self.origin["seeds"])
        parsed = trainer.parser().parse_args(command[3:])
        plan = trainer.make_plan(parsed)
        frozen = {"domains": list(flow.DOMAINS), "seeds": self.origin["seeds"],
                  "trial_id": self.variant["trial_id"]}
        recovery.validate_plan(parsed, plan, frozen)
        parsed.stage = "full"
        with self.assertRaisesRegex(ValueError, "training is disabled"):
            recovery.validate_plan(parsed, plan, frozen)
        parsed.stage = "final"
        with self.assertRaisesRegex(ValueError, "entire frozen"):
            recovery.validate_plan(parsed, plan[:-1], frozen)

    def test_report_keeps_fold_coverage_taxonomy_and_pooled_metrics(self):
        entries = [self.make_run(d, seed) for d in ("hotel", "food") for seed in (13, 42)]
        path = self.fixture / "selection.json"
        with patch.object(flow, "DOMAINS", ("hotel", "food")):
            flow.freeze(self.fixture, self.variant, [13, 42], path)
        for folder, manifest, metric in entries:
            manifest["status"] = "complete"
            flow.write_json(folder / "manifest.json", manifest)
            flow.write_json(folder / "test_metrics.json", metric)
        output = self.fixture / "report"
        report = lodo_report.write_report(path, output)
        result = flow.read(output / "results.json")
        self.assertEqual(result["test_runs"], 4)
        self.assertEqual(len(result["summaries"]), 2)
        self.assertEqual(result["overall"]["pooled_fold_micro_f1"], {"mean": .5, "sd": 0.})
        self.assertEqual(result["summaries"][0]["scores"]["coverage"]["mean"], .5)
        self.assertIn("not a strict source-only", report.read_text(encoding="utf-8"))
        self.assertTrue((output / "comparison.png").is_file())
        frozen = flow.read(path)
        frozen["runs"].append(frozen["runs"][0])
        flow.write_json(path, frozen)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            lodo_report.load_results(path)

    def test_zero_coverage_fold_keeps_projected_metrics_and_empty_rare_bucket(self):
        metric = self.make_run("hotel", 13)[2]
        metric["taxonomy"]["metrics"]["gold_category_coverage"] = 0.
        metric["taxonomy"]["by_rarity"] = [{"rarity": "unseen", "gold_triplets": 2,
                                             "primary_outcome_counts": {"correct": 0}, "error_rate": 1.}]
        metric["taxonomy"]["primary_outcome_counts"] = {"correct": 0, "term": 1, "category": 1, "sentiment": 0}
        metric["explicit"].update(true_positives=0, false_negatives=2, micro_f1=0., micro_recall=0., micro_precision=0.)
        from experiment_reporting.analysis import flatten_metrics
        frozen = {"domains": ["hotel"], "seeds": [13]}
        rows = [{"domain": "hotel", "seed": 13, "metrics": metric, "flat": flatten_metrics(metric)}]
        summary, overall = lodo_report.summarize(frozen, rows)
        self.assertEqual(summary[0]["scores"]["coverage"]["mean"], 0.)
        self.assertEqual(summary[0]["scores"]["term_f1"]["mean"], .75)
        self.assertEqual(summary[0]["rarity"]["rare"], {"mean_gold": 0., "recall": None})
        self.assertEqual(overall["pooled_fold_micro_f1"], {"mean": 0., "sd": None})


if __name__ == "__main__":
    unittest.main()
