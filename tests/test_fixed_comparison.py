import json
import tempfile
import unittest
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import config as cfg
import run_fixed_comparison as final
import run_study


class FixedComparisonTests(unittest.TestCase):
    def test_plan_has_120_unique_runs_and_four_complete_batches(self):
        plan = final.build_plan()
        batches = final.split_batches(plan)
        self.assertEqual(len(plan), 120)
        self.assertEqual(len(set(plan)), 120)
        self.assertEqual([len(batch) for batch in batches], [30] * 4)
        self.assertEqual([task for batch in batches for task in batch], plan)
        for batch in batches:
            conditions = {(task.mode, task.held_out_domain) for task in batch}
            self.assertEqual(len(conditions), 2)
            triples = Counter(
                (task.mode, task.held_out_domain, task.seed) for task in batch
            )
            self.assertEqual(set(triples.values()), {3})
            for mode, domain, seed in triples:
                self.assertEqual(
                    {task.config_name for task in batch
                     if (task.mode, task.held_out_domain, task.seed) == (mode, domain, seed)},
                    set(final.LOSSES),
                )

    def test_one_fixed_recipe_is_applied_to_all_three_losses(self):
        settings = []
        for name in final.LOSSES:
            experiment = final.fixed_config(name)
            self.assertEqual(experiment.lr, 3e-5)
            self.assertEqual(experiment.epochs, 8)
            self.assertEqual(experiment.warmup_ratio, 0.1)
            self.assertEqual(experiment.max_grad_norm, 1.0)
            settings.append({
                key: value for key, value in asdict(experiment).items()
                if key not in {"name", "loss_type"}
            })
        self.assertEqual(settings, [settings[0]] * 3)

    def test_each_batch_uses_its_own_results_csv(self):
        with patch.object(final.run_study, "run") as training:
            for index in range(4):
                final.run_batch(index, output_root=Path("fixed-results"))
                calls = training.call_args_list[index * 30:(index + 1) * 30]
                self.assertEqual(len(calls), 30)
                self.assertEqual(
                    {call.kwargs["results_csv"] for call in calls},
                    {Path("fixed-results") / f"batch_{index}.csv"},
                )
                self.assertTrue(all(
                    call.kwargs["experiment_override"].lr == 3e-5
                    and call.kwargs["experiment_override"].epochs == 8
                    for call in calls
                ))

    def test_completed_run_with_a_different_recipe_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            run_dir = output_root / "indomain" / "standard" / "seed_13"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({
                    "status": "complete",
                    "configuration": asdict(cfg.EXPERIMENTS[0]),
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "different training settings"):
                run_study.run(
                    "indomain", "standard", 13,
                    output_dir=output_root,
                    results_csv=output_root / "batch_0.csv",
                    experiment_override=final.fixed_config("standard"),
                )


if __name__ == "__main__":
    unittest.main()
