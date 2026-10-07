"""No-training ablations, frozen thresholds, waiting and separate seed cohorts."""

from dataclasses import asdict
import json
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import finalize_finetuning_report as flow
from experiment_reporting.ablation_evaluation import evaluate_quick_screens, save_freeze, screen_variants
from experiment_reporting.presentation_report import snapshot, summarize
from null_experiments.config import NullConfig
from null_experiments.model import NullModel
from null_experiments.storage import digest
from src.artifacts import atomic_torch_save, atomic_write_json
from src.data import Example
from src.experiment_config import TrialConfig
from src.experiment_data import ExperimentDataset, training_vocabulary
from src.experiment_runtime import file_hash
from src.experiment_train import evaluate_trial
from tests.test_null_fast_track import FixtureMixin, ToyEncoder, ToyTokenizer


class FinalReportTests(FixtureMixin, unittest.TestCase):
    def test_wait_keeps_existing_queue_and_does_not_launch_training(self):
        path = self.fixture / "active.lock"
        path.write_text(json.dumps({"pid": 123, "host": socket.gethostname()}))
        with patch("null_experiments.storage.process_alive", side_effect=[True, False]), patch.object(flow.time, "sleep") as sleep:
            flow.wait_for_queue(self.fixture, wait=True)
        self.assertTrue(path.exists())
        sleep.assert_called_once()
        with patch("null_experiments.storage.process_alive", return_value=True):
            with self.assertRaisesRegex(ValueError, "training is active"):
                flow.wait_for_queue(self.fixture)

    def test_continuation_only_confirms_finalists_and_evaluates_tests(self):
        (self.fixture / "search_state.json").write_text(json.dumps({"selected_null": {}}))
        args = SimpleNamespace(null_study=self.fixture, device="auto", finish_null=True,
                               first_seeds=5, retry_failed=False)
        commands = flow.remaining_commands(args)
        self.assertEqual([c[c.index("--step")+1] for c in commands], ["confirm", "final"])
        self.assertNotIn("search", commands[0])
        self.assertEqual(commands[0][commands[0].index("--first-seeds")+1], "5")
        args.finish_null = False
        with self.assertRaisesRegex(ValueError, "No development freeze"):
            flow.remaining_commands(args)

    def test_reused_candidates_dedupe_and_freeze_cannot_change(self):
        directory = self.fixture / "trial/seed_13"
        directory.mkdir(parents=True)
        cfg = {"name": "focal", "lr": 2e-5, "epochs": 3}
        manifest = {"status": "dev_complete", "identity": {"config": cfg, "seed": 13, "smoke": False},
                    "fingerprint": "fingerprint", "checkpoint_sha256": "hash",
                    "training": {"best_epoch": 2, "null_threshold": .3}}
        atomic_write_json(directory / "manifest.json", manifest)
        atomic_write_json(directory / "dev_metrics.json", {"null_threshold": .3})
        (directory / "test_metrics.json").write_text("invalid test JSON")
        variants = screen_variants([{"config": cfg}, {"config": cfg}], lambda e: directory)
        self.assertEqual((len(variants), len(variants[0]["runs"])), (1, 1))
        frozen = {"created_at": "first", "variants": variants}
        path = self.fixture / "freeze.json"
        save_freeze(path, frozen)
        original = path.read_bytes()
        save_freeze(path, {**frozen, "created_at": "later"})
        self.assertEqual(path.read_bytes(), original)
        variants[0]["runs"][0]["null_threshold"] = .8
        with self.assertRaisesRegex(ValueError, "freeze changed"):
            save_freeze(path, frozen)

    def test_different_budgets_and_studies_do_not_pool(self):
        rows = []
        for study, trial, epoch, seed, score in [("Finalists", "e5", 5, 13, .4), ("Finalists", "e5", 5, 42, .6),
                                                 ("Screens", "e3", 3, 13, .9)]:
            metric = {"micro_f1": score, "micro_precision": score, "micro_recall": score}
            rows.append({"study": study, "trial_id": trial, "role": "NULL", "split": "test", "seed": seed,
                         "expected_seeds": [13, 42] if epoch == 5 else [13],
                         "metrics": {s: metric for s in ("explicit", "null", "combined")}})
        groups = summarize(rows)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["scores"]["combined"]["mean"], .5)
        self.assertIsNone(groups[1]["scores"]["combined"]["sd"])

    def test_old_screen_test_is_inference_only_and_uses_frozen_threshold(self):
        examples = [Example("screen is good", [("screen", "Screen", "positive"), ("NULL", "Battery", "negative")], "laptop")]
        cfg = TrialConfig(name="fixture", loss_type="standard", epochs=1, max_len=8, batch_size=1, null_head=True, vocabulary_scope="explicit-null")
        categories, sentiments, counts = training_vocabulary(examples, cfg)
        tokenizer = ToyTokenizer()

        def model_for(c, nc, ns):
            return NullModel(NullConfig(**asdict(c)), list(range(nc)), list(range(ns)), tokenizer, encoder=ToyEncoder())

        model = model_for(cfg, len(categories), len(sentiments))
        directory = self.fixture / cfg.trial_id / "seed_13"
        directory.mkdir(parents=True)
        atomic_torch_save(directory / "best.pt", {"state_dict": model.state_dict(), "config": asdict(cfg),
            "categories": categories, "sentiments": sentiments, "seed": 13, "best_epoch": 1, "null_threshold": 1.})
        metric, _, _ = evaluate_trial(model, ExperimentDataset(examples, tokenizer, categories, sentiments, cfg),
                                      categories, sentiments, counts, cfg, torch.device("cpu"), threshold=1.)
        atomic_write_json(directory / "dev_metrics.json", metric)
        identity = {"config": asdict(cfg), "seed": 13, "smoke": False}
        run = {"seed": 13, "directory": str(directory), "fingerprint": digest(identity),
               "checkpoint_sha256": file_hash(directory / "best.pt"), "best_epoch": 1, "null_threshold": 1.}
        atomic_write_json(directory / "manifest.json", {"identity": identity, "status": "dev_complete",
            "fingerprint": run["fingerprint"], "checkpoint_sha256": run["checkpoint_sha256"],
            "training": {"best_epoch": 1, "null_threshold": 1.}})
        frozen = {"seeds": [13], "variants": [{"trial_id": cfg.trial_id, "config": asdict(cfg), "roles": ["screen"], "runs": [run]}]}
        frozen = json.loads(json.dumps(frozen))
        with patch("src.experiment_data.load_source_splits", return_value=(examples, examples, ["laptop"])), \
             patch("src.experiment_data.load_test_split", return_value=examples) as load_test, \
             patch("src.experiment_runtime.data_inventory", return_value={"test": "hash"}), \
             patch("src.experiment_model.ExperimentModel", side_effect=model_for), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             patch("src.experiment_train.train_trial", side_effect=AssertionError("Training must never start")):
            evaluate_quick_screens(flow.ROOT, frozen, self.fixture, torch.device("cpu"))
            evaluate_quick_screens(flow.ROOT, frozen, self.fixture, torch.device("cpu"))
        self.assertEqual(load_test.call_count, 1)
        test = json.loads((directory / "test_metrics.json").read_text())
        self.assertEqual(test["null_threshold"], 1.)
        self.assertEqual(test["null"]["true_positives"] + test["null"]["false_positives"], 0)
        (self.fixture / "selection.json").write_text(json.dumps(frozen))
        _, rows = snapshot(self.fixture, "Fixture")
        self.assertEqual({r["split"] for r in rows}, {"dev", "test"})


if __name__ == "__main__":
    unittest.main()
