import io
import csv
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import config as cfg
from analyze_category_ablation import load_scores, summarize
import run_category_ablation
import run_study
import src.train as train_module
from src.losses import get_loss_fns
from src.train import compute_batch_loss
from tests.test_pipeline_smoke import TinyABSAModel, TinyTokenizer, _synthetic_split


class CategoryWeightingTests(unittest.TestCase):
    def test_only_category_head_receives_inverse_frequency_weights(self):
        counts = ([0, 0, 1], [0, 0, 0, 1], [0, 0, 1])
        kwargs = {"num_categories": 3, "num_sentiments": 2}
        standard = get_loss_fns(SimpleNamespace(loss_type="standard"), *counts, **kwargs)
        all_weighted = get_loss_fns(SimpleNamespace(loss_type="weighted"), *counts, **kwargs)
        category_only = get_loss_fns(
            SimpleNamespace(loss_type="category_weighted"), *counts, **kwargs,
        )

        self.assertIsNone(standard[1].weight)
        self.assertIsNone(category_only[0].weight)
        self.assertIsNone(category_only[2].weight)
        self.assertTrue(torch.equal(category_only[1].weight, all_weighted[1].weight))
        self.assertAlmostEqual(category_only[1].weight[0].item(), 4 / 9)
        self.assertAlmostEqual(category_only[1].weight[1].item(), 4 / 3)
        self.assertEqual(category_only[1].weight[2].item(), 1.0)

    def test_category_only_changes_category_gradient_but_not_other_heads(self):
        counts = ([0, 0, 1], [0, 0, 0, 1], [0, 0, 1])
        kwargs = {"num_categories": 2, "num_sentiments": 2}
        batch = {
            "bio_labels": torch.tensor([[0, 1]]),
            "category_labels": [torch.tensor([0, 1])],
            "sentiment_labels": [torch.tensor([0, 1])],
        }
        gradients = {}
        for loss_type in ("standard", "category_weighted"):
            cfg_value = SimpleNamespace(
                loss_type=loss_type,
                category_loss_weight=1.0,
                sentiment_loss_weight=1.0,
            )
            bio_logits = torch.tensor(
                [[[0.3, 0.5, -0.1], [0.1, 0.2, 0.4]]], requires_grad=True,
            )
            category_logits = torch.tensor(
                [[0.4, 0.3], [0.1, 0.8]], requires_grad=True,
            )
            sentiment_logits = torch.tensor(
                [[0.2, 0.6], [0.7, 0.3]], requires_grad=True,
            )
            losses = get_loss_fns(cfg_value, *counts, **kwargs)
            total = compute_batch_loss(
                bio_logits, category_logits, sentiment_logits, batch,
                losses, cfg_value, torch.device("cpu"),
            )
            total.backward()
            gradients[loss_type] = (
                bio_logits.grad.clone(),
                category_logits.grad.clone(),
                sentiment_logits.grad.clone(),
            )

        standard = gradients["standard"]
        category_only = gradients["category_weighted"]
        self.assertTrue(torch.allclose(standard[0], category_only[0]))
        self.assertFalse(torch.allclose(standard[1], category_only[1]))
        self.assertTrue(torch.allclose(standard[2], category_only[2]))


class CategoryAblationRunnerTests(unittest.TestCase):
    def test_plan_accepts_existing_seeds_and_rejects_invalid_folds(self):
        self.assertEqual(
            run_category_ablation.build_plan("crossdomain", "hotel", [13, 42]),
            [
                ("crossdomain", "hotel", "category_weighted", 13),
                ("crossdomain", "hotel", "category_weighted", 42),
            ],
        )
        for mode, domain, seeds in (
            ("crossdomain", None, [13]),
            ("indomain", "hotel", [13]),
            ("indomain", None, [13, 13]),
            ("indomain", None, [999]),
        ):
            with self.subTest(mode=mode, domain=domain, seeds=seeds):
                with self.assertRaises(ValueError):
                    run_category_ablation.build_plan(mode, domain, seeds)

    def test_dry_run_never_starts_training_and_primary_grid_stays_two_configs(self):
        self.assertEqual([item.name for item in cfg.EXPERIMENTS], ["standard", "weighted"])
        with patch.object(run_category_ablation, "run") as train, redirect_stdout(io.StringIO()):
            run_category_ablation.main([
                "--mode", "crossdomain", "--held-out-domain", "hotel",
                "--seeds", "13", "--dry-run",
            ])
        train.assert_not_called()

    def test_category_ablation_completes_a_tiny_end_to_end_run(self):
        experiment = cfg.ExperimentConfig(
            name="category_weighted", loss_type="category_weighted",
            model_name="tiny-local-model", max_len=10, batch_size=2,
            epochs=1, lr=0.01, weight_decay=0.0,
        )
        split = _synthetic_split()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(run_study, "_split_examples", return_value=split),
                patch.object(
                    run_study.AutoTokenizer, "from_pretrained",
                    return_value=TinyTokenizer(),
                ),
                patch.object(cfg, "ABLATION_EXPERIMENTS", [experiment]),
                patch.object(
                    train_module, "build_model",
                    side_effect=lambda _cfg, categories, sentiments:
                        TinyABSAModel(categories, sentiments),
                ),
                redirect_stdout(io.StringIO()),
            ):
                summary = run_study.run(
                    mode="indomain", config_name="category_weighted", seed=13,
                    device="cpu", output_dir=root / "ablation",
                    results_csv=root / "results_category_ablation.csv",
                )
            self.assertEqual(summary["status"], "complete")
            run_dir = root / "ablation" / "indomain" / "category_weighted" / "seed_13"
            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["configuration"]["loss_type"], "category_weighted")
            self.assertTrue((run_dir / "test_predictions.jsonl").exists())


class CategoryAblationAnalysisTests(unittest.TestCase):
    def test_paired_comparison_marks_missing_seed_as_partial(self):
        score = lambda value: {
            "test_micro_precision": value,
            "test_micro_recall": value,
            "test_micro_f1": value,
        }
        reference = {
            "standard": {13: score(0.2), 42: score(0.3)},
            "weighted": {13: score(0.1), 42: score(0.2)},
        }
        ablation = {"category_weighted": {13: score(0.25)}}
        result = summarize(reference, ablation, [13, 42])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["matched_seeds"], [13])
        self.assertEqual(result["missing_seeds"]["category_weighted"], [42])
        self.assertAlmostEqual(
            result["metrics"]["test_micro_f1"]["category_minus_standard"],
            0.05,
        )

        ablation["category_weighted"][42] = score(0.35)
        result = summarize(reference, ablation, [13, 42])
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(
            result["metrics"]["test_micro_f1"]["category_minus_standard"],
            0.05,
        )

    def test_duplicate_completed_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "results.csv"
            fields = [
                "mode", "config", "seed", "status",
                "test_micro_precision", "test_micro_recall", "test_micro_f1",
            ]
            row = {
                "mode": "indomain", "config": "standard", "seed": "13",
                "status": "complete", "test_micro_precision": "0.5",
                "test_micro_recall": "0.5", "test_micro_f1": "0.5",
            }
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate completed key"):
                load_scores(path, "indomain", None, ("standard",))

    def test_legacy_restaurant_rows_with_blank_domain_can_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "results.csv"
            fields = [
                "mode", "held_out_domain", "config", "seed", "status",
                "test_micro_precision", "test_micro_recall", "test_micro_f1",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "mode": "crossdomain", "held_out_domain": "",
                    "config": "standard", "seed": "13", "status": "complete",
                    "test_micro_precision": "0.5", "test_micro_recall": "0.5",
                    "test_micro_f1": "0.5",
                })
            scores = load_scores(
                path, "crossdomain", "restaurant", ("standard",),
                legacy_holdout="restaurant",
            )
            self.assertIn(13, scores["standard"])


if __name__ == "__main__":
    unittest.main()
