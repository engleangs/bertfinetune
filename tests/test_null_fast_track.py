"""Loss mathematics, implicit routing, leakage guards and interruption recovery."""

from collections import Counter
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import re
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch
from torch import nn
from torch.nn import functional as F

import run_null_fast_track as runner
from src.data import Example
from src.experiment_runtime import file_hash
from null_experiments.config import NullConfig, VERSION, from_dict
from null_experiments.data import CachedDataset
from null_experiments.evaluation import choose_threshold, threshold_curve, count_scores
from null_experiments.losses import NullLoss, selected_cells
from null_experiments.model import NullModel
from null_experiments.storage import append_csv, digest
from null_experiments.study import search, confirm, check_freeze
from null_experiments.training import Runtime


class ToyTokenizer:
    pad_token_id = 0
    cls_token_id = 101

    def __len__(self):
        return 128

    def convert_tokens_to_ids(self, token):
        return 109

    def __call__(self, sentence, max_length, **kwargs):
        offsets = [(0, 0), *[m.span() for m in re.finditer(r"\S+", sentence)][:max_length-2], (0, 0)]
        ids = [101, *range(10, 10+len(offsets)-2), 102]
        length = len(ids)
        return {"input_ids": ids + [0] * (max_length-length),
                "attention_mask": [1] * length + [0] * (max_length-length),
                "offset_mapping": offsets + [(0, 0)] * (max_length-length)}

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "fixture.json").write_text("{}")


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(128, 12)
        self.dropout = nn.Dropout(.15)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        context = (hidden * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        return SimpleNamespace(last_hidden_state=self.dropout(hidden + context))


def toy_model(cfg, categories, sentiments, tokenizer):
    return NullModel(cfg, categories, sentiments, tokenizer, encoder=ToyEncoder())


class FixtureMixin:
    def setUp(self):
        self.base_dir = runner.ROOT / "artifacts/null_fast_track_tests"
        self.fixture = self.base_dir / uuid.uuid4().hex
        self.fixture.mkdir(parents=True)

    def tearDown(self):
        assert self.fixture.resolve().is_relative_to(self.base_dir.resolve())
        shutil.rmtree(self.fixture)


class LossAndRoutingTests(unittest.TestCase):
    def test_binary_focal_zero_gamma_equals_weighted_bce(self):
        cfg = NullConfig(null_loss="focal", null_focal_gamma=0)
        logits = torch.tensor([[1., -2., 0.]], requires_grad=True)
        targets = torch.tensor([[1., 0., 1.]])
        weights = torch.tensor([5., 7., 3.])
        self.assertTrue(torch.allclose(NullLoss(cfg, weights)(logits, targets),
                                      F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights)))

    def test_focal_uses_unweighted_probability_and_extreme_logits_stay_finite(self):
        cfg = NullConfig(null_loss="focal", null_focal_gamma=2)
        logits = torch.tensor([[2., -1., -1000., 1000.]], requires_grad=True)
        targets = torch.tensor([[1., 0., 1., 0.]])
        weights = torch.full((4,), 10.)
        result = NullLoss(cfg, weights)(logits, targets)
        p = logits.sigmoid()
        expected = (F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weights, reduction="none") *
                    (1-torch.where(targets.bool(), p, 1-p)).pow(2)).mean()
        self.assertTrue(torch.allclose(result, expected))
        result.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_asl_zero_focusing_no_margin_is_unweighted_bce(self):
        cfg = NullConfig(null_loss="asl", asl_gamma_positive=0, asl_gamma_negative=0, asl_margin=0)
        logits, targets = torch.tensor([[1., -2., 0.]]), torch.tensor([[1., 0., 1.]])
        self.assertTrue(torch.allclose(NullLoss(cfg, torch.full((3,), 50.))(logits, targets),
                                      F.binary_cross_entropy_with_logits(logits, targets), atol=1e-6))

    def test_asl_easy_negatives_are_suppressed_and_gradients_are_finite(self):
        cfg = NullConfig(null_loss="asl")
        logits = torch.tensor([[-10., 2., -1000., 1000.]], requires_grad=True)
        targets = torch.tensor([[0., 0., 1., 0.]])
        elements = NullLoss(cfg, torch.ones(4)).elementwise(logits, targets)
        self.assertEqual(float(elements[0, 0].detach()), 0)
        elements.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_hard_sampling_keeps_all_positives_and_highest_negatives(self):
        cfg = NullConfig(negative_sampling="hard", minimum_negatives=3, negative_ratio=1, hard_fraction=1)
        logits = torch.arange(10.).unsqueeze(0)
        targets = torch.zeros_like(logits)
        targets[0, 1] = 1
        mask = selected_cells(logits, targets, cfg)
        self.assertEqual(mask.nonzero().tolist(), [[0, 1], [0, 7], [0, 8], [0, 9]])
        no_positive = selected_cells(logits, torch.zeros_like(targets), cfg)
        self.assertEqual(int(no_positive.sum()), 3)

    def test_random_control_matches_negative_budget_and_handles_all_positive(self):
        cfg = NullConfig(negative_sampling="random", minimum_negatives=4, negative_ratio=2)
        targets = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]])
        self.assertEqual(int(selected_cells(torch.zeros_like(targets), targets, cfg).sum()), 6)
        all_positive = torch.ones_like(targets)
        self.assertTrue(selected_cells(targets, all_positive, cfg).all())
        self.assertTrue(torch.isfinite(NullLoss(cfg, torch.ones(8))(targets, all_positive)))

    def test_ia_and_control_have_same_spans_positions_and_text_budget(self):
        example = Example("screen is good", [("screen", "Screen", "positive"), ("NULL", "Battery", "negative")], "laptop")
        cfg = NullConfig(max_len=8)
        control = CachedDataset([example], ToyTokenizer(), ["Battery", "Screen"], ["negative", "positive"], cfg)[0]
        implicit = CachedDataset([example], ToyTokenizer(), ["Battery", "Screen"], ["negative", "positive"], replace(cfg, representation="ia"))[0]
        self.assertEqual(control["span_boundaries"], implicit["span_boundaries"])
        self.assertEqual(control["span_boundaries"], [(2, 2)])
        self.assertTrue(torch.equal(control["input_ids"][2:], implicit["input_ids"][2:]))
        self.assertEqual(int(control["attention_mask"][1]), 0)
        self.assertEqual(int(implicit["attention_mask"][1]), 1)
        self.assertEqual(int(implicit["bio_labels"][1]), -100)
        self.assertEqual(implicit["offset_mapping"][1].tolist(), [0, 0])
        self.assertEqual(len(implicit["input_ids"]), 8)
        self.assertEqual(implicit["gold_null_triplets"], [("NULL", "Battery", "negative")])

    def test_attention_ignores_padding_and_supports_multilabel_output(self):
        cfg = NullConfig(representation="ia_attention", attention_size=8)
        model = toy_model(cfg, ["Battery", "Screen"], ["negative", "positive"], ToyTokenizer())
        model._attention_mask = torch.tensor([[True, True, True, False]])
        hidden = torch.randn(1, 4, 12)
        expected = model.classify_null(hidden)
        changed = hidden.clone()
        changed[:, 3] = 100000
        self.assertEqual(expected.shape, (1, 4))
        self.assertTrue(torch.allclose(expected, model.classify_null(changed)))

    def test_unknown_gold_remains_false_negative_and_abstention_is_explicit(self):
        row = {"gold_triplets": [], "predicted_triplets": [], "gold_null_triplets": [["NULL", "Unknown", "positive"]]}
        curve = threshold_curve([row], torch.tensor([[1.]]), ["Known"], ["positive"], [.2])
        self.assertEqual(curve[0]["null"]["false_positives"], 1)
        self.assertEqual(curve[0]["null"]["false_negatives"], 1)
        self.assertEqual(curve[-1]["null"]["false_positives"], 0)
        self.assertEqual(choose_threshold(curve), 1)

    def test_combined_and_null_objectives_can_choose_different_thresholds(self):
        curve = [{"threshold": .2, "null": count_scores(1, 9, 9), "combined": count_scores(9, 13, 11)},
                 {"threshold": 1., "null": count_scores(0, 0, 10), "combined": count_scores(8, 4, 12)}]
        self.assertEqual(choose_threshold(curve, "combined"), 1)
        self.assertEqual(choose_threshold(curve, "null"), .2)


class LedgerAndPlannerTests(FixtureMixin, unittest.TestCase):
    def test_csv_appends_and_does_not_duplicate_completed_run(self):
        path = self.fixture / "results.csv"
        append_csv(path, {"id": 1, "f1": .3}, ("id",))
        previous = path.read_bytes()
        append_csv(path, {"id": 2, "f1": .4}, ("id",))
        self.assertTrue(path.read_bytes().startswith(previous))
        append_csv(path, {"id": 1, "f1": .9}, ("id",))
        with path.open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_config_identity_includes_all_null_parameters(self):
        cfg = NullConfig()
        self.assertEqual(from_dict(json.loads(json.dumps(asdict(cfg)))).trial_id, cfg.trial_id)
        for key, value in (("null_loss", "focal"), ("representation", "ia"), ("negative_sampling", "hard"),
                           ("precision", "fp32"), ("null_focal_gamma", 1.), ("epochs", 5)):
            self.assertNotEqual(replace(cfg, **{key: value}).trial_id, cfg.trial_id)

    def test_smoke_cannot_open_test_and_custom_flags_are_explicit(self):
        args = runner.parser().parse_args(["--smoke", "--step", "final"])
        with self.assertRaisesRegex(ValueError, "never evaluate test"):
            runner.validate_args(args)
        args = runner.parser().parse_args(["--step", "custom", "--recipe", "hard", "--null-loss", "asl"])
        runner.validate_args(args)
        cfg = runner.settings(args)
        self.assertEqual((cfg.null_loss, cfg.representation, cfg.negative_sampling), ("asl", "ia_attention", "hard"))

    def test_staged_search_and_freeze_use_dev_only_and_extend_seed_cohorts(self):
        runtime = SimulatedRuntime(self.fixture)
        args = SimpleNamespace(tune=True, tune_lr=False, retry_failed=False, first_seeds=2, epochs=5, confirm_top=1)
        state = search(runtime, NullConfig(), args)
        self.assertEqual(len(runtime.calls), 10)
        self.assertEqual({name for name, _, _ in runtime.calls},
                         {"2-3_loss_screen", "4_implicit_representation", "5_negative_sampling", "parameter_tuning"})
        self.assertEqual(state["selected_null"]["negative_sampling"], "hard")
        frozen = confirm(runtime, state, args)
        self.assertEqual(len(frozen["variants"]), 2)
        self.assertEqual(frozen["seeds"], [13, 42])
        first_archive = list((self.fixture / "selections").glob("*.json"))
        self.assertEqual(len(first_archive), 1)
        first_bytes = first_archive[0].read_bytes()
        args.first_seeds = 5
        confirm(runtime, state, args)
        self.assertEqual(first_archive[0].read_bytes(), first_bytes)
        self.assertEqual(len(list((self.fixture / "selections").glob("*.json"))), 2)
        # Selection routines never open these deliberately invalid test files.
        self.assertTrue(all((Path(r["directory"]) / "test_metrics.json").read_text() == "invalid test JSON"
                            for v in frozen["variants"] for r in v["runs"]))
        check_freeze(runtime, frozen)
        frozen["variants"][0]["runs"][0]["null_threshold"] = .1
        with self.assertRaisesRegex(ValueError, "threshold changed"):
            check_freeze(runtime, frozen)


class SimulatedRuntime:
    def __init__(self, output):
        self.output = output
        self.smoke = False
        self.code = {"source_sha256": {"fixture": "source"}}
        self.data_hashes = {"fixture": "train-dev"}
        self.environment = {"fixture": True}
        self.calls = []

    def destination(self, cfg, seed):
        return self.output / "research" / cfg.trial_id / f"seed_{seed}"

    def identity(self, cfg, seed):
        return {"config": asdict(cfg), "seed": seed, "smoke": False}

    def run(self, cfg, seed, stage, retry_failed):
        self.calls.append((stage, cfg, seed))
        directory = self.destination(cfg, seed)
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "manifest.json").exists():
            score = {"bce": .30, "focal": .4, "asl": .38}[cfg.null_loss]
            score += {"cls": 0, "ia": .01, "ia_attention": .02}[cfg.representation]
            score += {"all": 0, "random": .005, "hard": .025}[cfg.negative_sampling]
            score += .005 if cfg.null_loss_weight == .1 else 0
            score += .004 if cfg.null_focal_gamma == 1 else 0
            metric = {"micro_f1": score}
            metrics = {"combined": metric, "explicit": metric, "null": metric,
                       "unweighted_loss": 1., "null_threshold": .2}
            (directory / "dev_metrics.json").write_text(json.dumps(metrics))
            for name in ("best.pt", "dev_predictions.jsonl", "dev_probabilities.pt", "history.json", "audit.json"):
                (directory / name).write_text("fixture")
            manifest = {"status": "dev_complete", "identity": self.identity(cfg, seed),
                        "fingerprint": digest(self.identity(cfg, seed)), "checkpoint_sha256": file_hash(directory / "best.pt"),
                        "training": {"best_epoch": cfg.epochs, "null_threshold": .2}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            (directory / "test_metrics.json").write_text("invalid test JSON")
        return directory


class ToyRuntime(Runtime):
    def __init__(self, output):
        self.root, self.output, self.device = runner.ROOT, output, torch.device("cpu")
        self.smoke = True
        self.train_limit, self.dev_limit = 2, 2
        self.environment, self.code, self.data_hashes = {}, {"source_sha256": {}}, {}
        self.tokenizer = ToyTokenizer()

    def data(self, cfg):
        examples = [Example("screen is good", [("screen", "Screen", "positive"), ("NULL", "Battery", "negative")], "laptop"),
                    Example("screen is bad", [("screen", "Screen", "negative")], "laptop")]
        categories, sentiments = ["Battery", "Screen"], ["conflict", "negative", "neutral", "positive"]
        dataset = CachedDataset(examples, self.tokenizer, categories, sentiments, cfg)
        return dataset, dataset, categories, sentiments, {"Battery": 1, "Screen": 2}


class ResumeTests(FixtureMixin, unittest.TestCase):
    def test_interrupted_training_restores_optimizer_scheduler_rng_and_best(self):
        from null_experiments import training
        actual_evaluate = training.evaluate
        resumed = ToyRuntime(self.fixture / "resumed")
        clean = ToyRuntime(self.fixture / "clean")
        cfg = NullConfig(epochs=2, batch_size=1, max_len=8, precision="fp32", representation="ia_attention",
                         negative_sampling="hard", minimum_negatives=2, attention_size=8)
        calls = 0

        def interrupt_second_epoch(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("fixture interruption")
            return actual_evaluate(*args, **kwargs)

        with patch.object(training, "NullModel", side_effect=toy_model):
            with patch.object(training, "evaluate", side_effect=interrupt_second_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    resumed.run(cfg, 13, "custom")
            directory = resumed.run(cfg, 13, "custom")
            other = clean.run(cfg, 13, "custom")
        a = torch.load(directory / "best.pt", weights_only=True)
        b = torch.load(other / "best.pt", weights_only=True)
        for name in a["state_dict"]:
            self.assertTrue(torch.equal(a["state_dict"][name], b["state_dict"][name]), name)
        self.assertEqual(a["best_epoch"], b["best_epoch"])
        self.assertFalse((directory / "resume.pt").exists())
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["training"]["completed_epochs"], 2)
        self.assertEqual(manifest["training"]["optimizer_steps"], 4)
        with (resumed.output / "development_results.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
