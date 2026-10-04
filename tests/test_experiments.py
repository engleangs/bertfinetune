"""Protocol, loss mathematics, NULL targets, and two-seed runner integration."""

from dataclasses import replace
import io
import json
import tempfile
from pathlib import Path
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import run_experiments as runner
from src.data import Example
from src.evaluate import complete_triplet_scores
from src.experiment_config import TrialConfig, candidate_recipes
from src.experiment_data import ExperimentDataset, select_targets, training_vocabulary
from src.experiment_losses import FocalCrossEntropy, MixedCrossEntropy, explicit_loss_functions, null_positive_weights
from src.experiment_model import ExperimentModel
import src.experiment_train as trainer
from tests.test_pipeline_smoke import TinyTokenizer


class TinyExperimentModel(ExperimentModel):
    def __init__(self, cfg, categories, sentiments):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(64, 8)
        self.bio_head = nn.Linear(8, 3)
        self.category_head = nn.Linear(8, categories)
        self.sentiment_head = nn.Linear(8, sentiments)
        self.null_head = nn.Linear(8, categories * sentiments) if cfg.null_head else None

    def encode_and_tag(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        return hidden, self.bio_head(hidden)


def examples():
    return [Example("Screen good", [("Screen", "DISPLAY", "positive"), ("NULL", "SERVICE", "negative")], "laptop"),
            Example("Battery bad", [("Battery", "BATTERY", "negative")], "laptop"),
            Example("Good overall", [("NULL", "OVERALL", "positive")], "hotel"),
            Example("No opinion", [], "hotel")]


class EligibilityTests(unittest.TestCase):
    def test_repeated_conflicting_and_all_overlapping_spans_are_excluded(self):
        ex = Example("Screen Screen; battery life good; sound nice", [
            ("Screen", "D", "positive"), ("battery life", "B", "positive"),
            ("battery", "B", "positive"), ("sound", "S", "positive"), ("sound", "S", "negative"),
            ("missing", "M", "neutral"), ("null", "O", "positive"), ("NULL", "O", "positive")])
        explicit, nulls, audit = select_targets(ex)
        self.assertEqual(explicit, [])
        self.assertEqual(nulls, [("NULL", "O", "positive")])
        self.assertEqual(audit["ambiguous_occurrence"], 1)
        self.assertEqual(audit["overlapping_span_annotations"], 2)
        self.assertEqual(audit["conflicting_span_annotations"], 2)
        reversed_result = select_targets(replace(ex, triplets=list(reversed(ex.triplets))))
        self.assertEqual((explicit, nulls, audit), reversed_result)

    def test_drop_conflict_preserves_other_targets_in_sentence(self):
        ex = Example("Screen good", [("Screen", "D", "positive"), ("Screen", "D", "conflict"), ("NULL", "O", "conflict")])
        self.assertEqual(select_targets(ex)[0], [])
        explicit, nulls, _ = select_targets(ex, drop_conflict=True)
        self.assertEqual(explicit, [("Screen", "D", "positive")])
        self.assertEqual(nulls, [])

    def test_null_targets_are_multilabel_with_negative_sentences_and_train_only_vocab(self):
        cfg = TrialConfig("null", "standard", null_head=True, max_len=10)
        categories, sentiments, _ = training_vocabulary(examples(), cfg)
        ds = ExperimentDataset(examples(), TinyTokenizer(), categories, sentiments, cfg)
        self.assertEqual(ds.null_targets.sum(1).tolist(), [1, 0, 1, 0])
        self.assertEqual(ds[0]["gold_triplets"], [("Screen", "DISPLAY", "positive")])
        self.assertEqual(ds.gold_spans[0][0]["start"], 0)
        self.assertEqual(null_positive_weights(ds, 2).max().item(), 2)
        unseen = Example("Fine", [("NULL", "UNSEEN", "positive")])
        unseen_ds = ExperimentDataset([unseen], TinyTokenizer(), categories, sentiments, cfg)
        self.assertEqual(unseen_ds.null_targets.sum().item(), 0)
        self.assertEqual(unseen_ds[0]["gold_null_triplets"], [("NULL", "UNSEEN", "positive")])
        gold = [unseen_ds[0]["gold_null_triplets"]]
        self.assertEqual(complete_triplet_scores(gold, [[]], categories)["false_negatives"], 1)

    def test_null_comparison_locks_the_same_vocabulary(self):
        vocabularies = [training_vocabulary(examples(), TrialConfig(**r)) for r in candidate_recipes("null")]
        self.assertEqual(vocabularies[0], vocabularies[1])


class LossTests(unittest.TestCase):
    def test_mixed_loss_matches_normalized_professor_formula(self):
        logits = torch.tensor([[2., -1.], [0., 1.], [9., 0.]], requires_grad=True)
        labels = torch.tensor([0, 1, -100])
        weights = torch.tensor([0.5, 3.])
        expected = (F.cross_entropy(logits[:2], labels[:2]) + 0.5 * F.cross_entropy(logits[:2], labels[:2], weight=weights)) / 1.5
        actual = MixedCrossEntropy(weights, 1, 0.5)(logits, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(2)))

    def test_focal_zero_equals_ce_and_focal_probability_is_unweighted(self):
        logits = torch.tensor([[2., -1.], [0., 1.]], requires_grad=True)
        labels = torch.tensor([1, 1])
        ce = F.cross_entropy(logits, labels, reduction="none")
        torch.testing.assert_close(FocalCrossEntropy(0)(logits, labels), ce.mean())
        expected = ((1 - torch.softmax(logits, -1)[range(2), labels]) ** 2 * ce).mean()
        torch.testing.assert_close(FocalCrossEntropy(2)(logits, labels), expected)
        ignored = FocalCrossEntropy(2)(logits, torch.tensor([-100, -100]))
        self.assertEqual(ignored.item(), 0)
        ignored.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_category_only_weighting_leaves_other_heads_standard(self):
        cfg = TrialConfig("category", "weighted", loss_heads=("category",))
        functions = explicit_loss_functions(cfg, ([0, 0, 1], [0, 0, 1], [0, 1]), (3, 2, 2), "cpu")
        self.assertIsNone(functions[0].weight)
        self.assertIsNotNone(functions[1].weight)
        self.assertIsNone(functions[2].weight)

    def test_fixed_macro_includes_absent_source_category_and_keeps_unseen_fn(self):
        gold = [[("a", "A", "positive"), ("u", "UNSEEN", "positive")]]
        pred = [[("a", "A", "positive")]]
        metrics = complete_triplet_scores(gold, pred, ["A", "B", "C"])
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertAlmostEqual(metrics["micro_f1"], 2 / 3)
        self.assertEqual(metrics["false_negatives"], 1)


class PlanTests(unittest.TestCase):
    def test_grid_defaults_to_two_matched_seeds(self):
        args = runner.parser().parse_args(["--suite", "all"])
        plan = runner.make_plan(args)
        self.assertEqual(len(plan), 26)
        self.assertEqual({seed for _, seed, _ in plan}, {13, 42})
        self.assertTrue(all(cfg.epochs == 5 for cfg, _, _ in plan))

    def test_custom_flags_and_smoke_override(self):
        args = runner.parser().parse_args(["--suite", "custom", "--loss", "mixed", "--loss-heads", "category",
                                         "--null-head", "--ce-weight", "1", "--weighted-ce-weight", ".5", "--smoke"])
        cfg = runner.make_plan(args)[0][0]
        self.assertTrue(cfg.null_head)
        self.assertEqual(cfg.loss_heads, ("category",))
        self.assertEqual(cfg.epochs, 1)
        self.assertAlmostEqual(cfg.weighted_ce_weight, .5)

    def test_invalid_configs_fail_before_training(self):
        for change in ({"lr": float("nan")}, {"max_grad_norm": 0}, {"null_head": True, "null_loss_weight": 0},
                       {"selection_metric": "combined"}, {"null_thresholds": (1.,)}, {"loss_heads": ("bio", "bio")},
                       {"bio_loss_weight": 0, "category_loss_weight": 0, "sentiment_loss_weight": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(TrialConfig("invalid", "standard"), **change).validate()
        with self.assertRaises(ValueError):
            runner.make_plan(runner.parser().parse_args(["--smoke", "--stage", "final"]))


class IntegrationTests(unittest.TestCase):
    def test_every_recipe_two_seeds_trains_saves_and_clips(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch.object(trainer.torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clipping:
                for recipe in candidate_recipes("all"):
                    for seed in (13, 42):
                        cfg = TrialConfig(**recipe, epochs=1, batch_size=2, max_len=10, lr=.001, warmup_ratio=.5)
                        categories, sentiments, frequencies = training_vocabulary(examples(), cfg)
                        ds = ExperimentDataset(examples(), TinyTokenizer(), categories, sentiments, cfg)
                        path = Path(directory) / cfg.name / str(seed)
                        _, result = trainer.train_trial(cfg, ds, ds, categories, sentiments, frequencies, seed,
                                                        torch.device("cpu"), path, model_factory=TinyExperimentModel)
                        self.assertEqual(result["warmup_steps"], 1)
                        self.assertEqual(result["optimizer_steps"], 2)
                        report = json.loads((path / "dev_metrics.json").read_text())
                        self.assertTrue(0 <= report["explicit"]["micro_f1"] <= 1)
                        self.assertEqual(report["taxonomy"]["gold_triplets"], 2)
                        self.assertEqual(report["null_gold"], 2)
                        self.assertTrue((path / "best.pt").exists())
                self.assertEqual(clipping.call_count, 52)

    def test_pilot_never_loads_test_and_final_reuses_frozen_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parser().parse_args(["--suite", "null", "--first-seeds", "1", "--epochs", "1",
                                              "--batch-size", "2", "--max-len", "10", "--output", directory])
            cfg = runner.make_plan(args)[1][0]
            with (patch.object(runner, "load_source_splits", return_value=(examples(), examples(), ["laptop"])),
                  patch.object(runner, "data_inventory", return_value={"synthetic": "fixture"}),
                  patch.object(runner.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer()),
                  patch.object(trainer, "ExperimentModel", TinyExperimentModel),
                  patch.object(runner, "ExperimentModel", TinyExperimentModel),
                  patch.object(runner, "load_test_split", return_value=examples()) as test_loader,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                runner.run_trial(args, cfg, 13, None, {}, {"source_sha256": {}})
                test_loader.assert_not_called()
                manifest_path = next(Path(directory).glob("**/manifest.json"))
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["status"], "dev_complete")
                runner.write_development_summary(Path(directory))
                summary = json.loads((Path(directory) / "development_summary.json").read_text())
                self.assertEqual(summary[0]["seeds"], [13])
                self.assertIsNone(summary[0]["explicit_f1_sample_std"])
                saved_threshold = manifest["training"]["null_threshold"]
                args.stage = "final"
                with patch.object(trainer, "choose_null_threshold", side_effect=AssertionError("Test threshold search forbidden")):
                    runner.run_trial(args, cfg, 13, None, {}, {"source_sha256": {}})
                self.assertEqual(test_loader.call_count, 1)
                report = json.loads((manifest_path.parent / "test_metrics.json").read_text())
                self.assertEqual(report["null_threshold"], saved_threshold)
                self.assertEqual(report["null_threshold_curve"], [])
                runner.run_trial(args, cfg, 13, None, {}, {"source_sha256": {}})
                self.assertEqual(test_loader.call_count, 1)
                with self.assertRaisesRegex(ValueError, "Provenance changed"):
                    runner.run_trial(args, cfg, 13, None, {"changed": True}, {"source_sha256": {}})


if __name__ == "__main__":
    unittest.main()
