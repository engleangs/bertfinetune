"""Check staged budgets, compatibility with existing trials, and safe execution."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_fast_pilot as fast


class FastPlanTests(unittest.TestCase):
    def jobs(self, *flags):
        return fast.make_jobs(fast.parser().parse_args(list(flags)))[0]

    def test_default_screen_retains_professor_mixtures_and_one_focal(self):
        jobs = self.jobs()
        self.assertEqual([job.trial.name for job in jobs], list(fast.SCREEN))
        self.assertEqual({job.seed for job in jobs}, {13})
        self.assertEqual(sum(job.trial.epochs for job in jobs), 30)

    def test_small_screen_keeps_controls_without_duplicates(self):
        names = [job.trial.name for job in self.jobs("--recipes", "focal_2", "standard", "focal_2")]
        self.assertEqual(names, ["standard", "weighted", "focal_2"])

    def test_lr_step_changes_only_rate_and_reuses_the_reference_trial(self):
        screen = self.jobs()
        jobs = self.jobs("--step", "lr", "--winner", "mixed_033")
        self.assertEqual([job.trial.lr for job in jobs], [1e-5, 2e-5, 3e-5])
        reference = next(job for job in screen if job.trial.name == "mixed_033")
        self.assertEqual(jobs[1].directory, reference.directory)
        configurations = [{k: v for k, v in asdict(job.trial).items() if k != "lr"} for job in jobs]
        self.assertTrue(all(c == configurations[0] for c in configurations))

    def test_second_seed_check_matches_control_rates_and_budget(self):
        jobs = self.jobs("--step", "confirm", "--winner", "mixed_050", "--lr", "1e-5")
        self.assertEqual(len(jobs), 6)
        self.assertEqual({job.seed for job in jobs}, {13, 42})
        self.assertTrue(all(job.trial.lr == 1e-5 and job.trial.epochs == 5 for job in jobs))
        self.assertEqual({job.trial.name for job in jobs}, {"standard", "weighted", "mixed_050"})

    def test_ce_can_win_without_duplicate_controls(self):
        self.assertEqual(len(self.jobs("--step", "confirm", "--winner", "standard")), 4)

    def test_five_seed_expansion_preserves_registered_order(self):
        jobs = self.jobs("--step", "confirm", "--winner", "focal_2", "--first-seeds", "5")
        self.assertEqual(len(jobs), 15)
        self.assertEqual([job.seed for job in jobs[:5]], fast.config.SEEDS)

    def test_null_control_has_matched_scope(self):
        jobs = self.jobs("--step", "null")
        self.assertEqual(len(jobs), 2)
        self.assertEqual([job.trial.null_head for job in jobs], [False, True])
        self.assertEqual({job.trial.vocabulary_scope for job in jobs}, {"explicit-null"})

    def test_short_screen_is_isolated_and_cannot_be_confirmation(self):
        _, output = fast.make_jobs(fast.parser().parse_args(["--epochs", "3"]))
        self.assertEqual(output.name, "experiments_screen_e3")
        with self.assertRaises(ValueError):
            self.jobs("--step", "confirm", "--winner", "standard", "--epochs", "3")

    def test_invalid_selection_and_budgets_fail_before_execution(self):
        for flags in (("--step", "lr"), ("--first-seeds", "0"), ("--lr", "nan"),
                      ("--step", "lr", "--winner", "standard", "--learning-rates", "0.00002", "0.00002")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.jobs(*flags)

    def test_all_staged_commands_match_original_runner_configs_and_paths(self):
        import run_experiments as runner
        cases = [(), ("--step", "lr", "--winner", "mixed_033"),
                 ("--step", "confirm", "--winner", "weighted_category", "--lr", "1e-5"),
                 ("--step", "null"), ("--epochs", "3"),
                 ("--mode", "crossdomain", "--held-out-domain", "food")]
        for flags in cases:
            for job in self.jobs(*flags):
                with self.subTest(flags=flags, recipe=job.trial.name, seed=job.seed):
                    args = runner.parser().parse_args(job.command[2:])
                    plan = runner.make_plan(args)
                    self.assertEqual(len(plan), 1)
                    cfg, seed, domain = plan[0]
                    self.assertEqual(asdict(cfg), asdict(job.trial))
                    self.assertEqual(seed, job.seed)
                    expected = args.output / "research" / args.mode / (domain or "all_domains") / cfg.trial_id / f"seed_{seed}"
                    self.assertEqual(expected, job.directory)
                    self.assertNotEqual(args.stage, "final")


class FastExecutionTests(unittest.TestCase):
    def test_plan_only_never_launches_training(self):
        with patch.object(fast.subprocess, "run") as launch, patch.object(fast, "recorded_status", return_value="pending"), \
                patch.object(fast, "minutes_per_epoch", return_value=None), redirect_stdout(io.StringIO()):
            fast.main([])
        launch.assert_not_called()

    def test_running_queue_blocks_execution_even_with_shorter_output(self):
        with patch.object(fast.subprocess, "run") as launch, patch.object(fast, "recorded_status", return_value="pending"), \
                patch.object(fast, "minutes_per_epoch", return_value=None), \
                patch.object(fast, "running_manifest", return_value=Path("original/research/manifest.json")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                fast.main(["--execute", "--epochs", "3"])
        self.assertEqual(raised.exception.code, 2)
        launch.assert_not_called()

    def test_provenance_or_training_failure_stops_remaining_jobs(self):
        with patch.object(fast.subprocess, "run", return_value=SimpleNamespace(returncode=17)) as launch, \
                patch.object(fast, "recorded_status", return_value="pending"), \
                patch.object(fast, "minutes_per_epoch", return_value=None), \
                patch.object(fast, "running_manifest", return_value=None), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                fast.main(["--execute"])
        self.assertEqual(raised.exception.code, 17)
        self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
