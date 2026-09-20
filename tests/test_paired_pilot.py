import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import config as cfg
import run_paired_pilot as pilot
import src.data as data_module
import src.train as train_module
from src.data import ABSADataset
from tests.test_pipeline_smoke import TinyABSAModel, TinyTokenizer, _synthetic_split


class PairedPilotTests(unittest.TestCase):
    def test_crossdomain_tuning_never_reads_held_out_domain(self):
        paths = []

        def record(path, domain):
            paths.append((path, domain))
            return []

        with patch.object(data_module, "load_domain_file", side_effect=record):
            pilot._train_dev("restaurant", Path("unused-data"))
        self.assertEqual(len(paths), 2 * (len(cfg.DOMAINS) - 1))
        self.assertTrue(all(domain != "restaurant" for _, domain in paths))
        self.assertTrue(all("test.txt" not in path for path, _ in paths))

    def test_warmup_and_clipping_apply_to_each_optimizer_update(self):
        train, dev, _ = _synthetic_split()
        tokenizer = TinyTokenizer()
        categories = ["BATTERY", "DISPLAY"]
        sentiments = ["negative", "positive"]
        experiment = cfg.ExperimentConfig(
            name="standard", loss_type="standard", model_name="tiny-local-model",
            max_len=10, batch_size=1, epochs=2, lr=0.01,
            warmup_ratio=0.25, max_grad_norm=1.0,
        )
        train_ds = ABSADataset(train, tokenizer, categories, sentiments, 10)
        dev_ds = ABSADataset(dev, tokenizer, categories, sentiments, 10)
        real_scheduler = train_module.get_linear_schedule_with_warmup
        real_clip = torch.nn.utils.clip_grad_norm_
        with (
            patch.object(
                train_module, "build_model",
                side_effect=lambda _cfg, ncat, nsent: TinyABSAModel(ncat, nsent),
            ),
            patch.object(train_module, "get_linear_schedule_with_warmup", wraps=real_scheduler) as scheduler,
            patch.object(torch.nn.utils, "clip_grad_norm_", wraps=real_clip) as clip,
        ):
            result = train_module.train_one_config(
                experiment, train_ds, dev_ds, categories, sentiments,
                seed=13, device="cpu",
            )
        self.assertEqual(result["selection_metric"], "dev_complete_triplet_micro_f1")
        self.assertEqual(scheduler.call_args.kwargs["num_training_steps"], 4)
        self.assertEqual(scheduler.call_args.kwargs["num_warmup_steps"], 1)
        self.assertEqual(clip.call_count, 4)
        self.assertTrue(all(call.args[1] == 1.0 for call in clip.call_args_list))

    def test_common_recipe_selection_requires_all_matched_runs(self):
        recipes = [pilot.Recipe(2e-5, 5), pilot.Recipe(3e-5, 8)]
        plan = pilot.build_plan(["indomain"], recipes, [13])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                pilot.summarize(plan, root)
            for task in plan:
                path = task.output_dir(root) / "dev_summary.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                score = 0.2 if task.recipe == recipes[0] else 0.3
                payload = {
                    "pilot_version": pilot.PILOT_VERSION,
                    "fold": task.fold,
                    "seed": task.seed,
                    "config": task.config_name,
                    "recipe": asdict(task.recipe),
                    "warmup_ratio": pilot.WARMUP_RATIO,
                    "max_grad_norm": pilot.MAX_GRAD_NORM,
                    "selection_metric": "dev_complete_triplet_micro_f1",
                    "best_dev_triplet_micro_f1": score,
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
            report = pilot.summarize(plan, root)
            self.assertEqual(report["selected_recipe"], asdict(recipes[1]))
            self.assertEqual(report["task_count"], 6)

    def test_task_saves_development_result_without_test_artifacts(self):
        train, dev, _ = _synthetic_split()
        task = pilot.PilotTask("indomain", pilot.Recipe(3e-5, 1), 13, "category_weighted")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with (
                patch.object(pilot, "_train_dev", return_value=(train, dev)),
                patch.object(
                    pilot.AutoTokenizer, "from_pretrained", return_value=TinyTokenizer(),
                ),
                patch.object(
                    train_module, "build_model",
                    side_effect=lambda _cfg, ncat, nsent: TinyABSAModel(ncat, nsent),
                ),
            ):
                result = pilot.run_task(task, root / "unused-data", root, "cpu")
            run_dir = task.output_dir(root)
            self.assertEqual(result["selection_metric"], "dev_complete_triplet_micro_f1")
            self.assertTrue((run_dir / "history.json").is_file())
            self.assertTrue((run_dir / "dev_summary.json").is_file())
            self.assertFalse((run_dir / "test_predictions.jsonl").exists())
            with (run_dir / "manifest.json").open(encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
