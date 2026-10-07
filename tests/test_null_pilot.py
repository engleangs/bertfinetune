"""NULL search budgets, fair controls and trainer-compatible run identities."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io
import unittest
from unittest.mock import patch

import run_null_pilot as pilot


class NullPilotTests(unittest.TestCase):
    def jobs(self, *flags):
        return pilot.make_jobs(pilot.parser().parse_args(list(flags)))[0]

    def test_default_search_has_one_control_and_three_caps_in_first_seed(self):
        jobs = self.jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual({j.seed for j in jobs}, {13})
        self.assertEqual({j.trial.name for j in jobs}, {"mixed_025"})
        self.assertEqual([j.trial.null_head for j in jobs], [False, True, True, True])
        self.assertEqual([j.trial.null_pos_weight_cap for j in jobs[1:]], [1, 10, 30])
        self.assertTrue(all(j.trial.vocabulary_scope == "explicit-null" for j in jobs))
        self.assertTrue(all(j.trial.lr == 3e-5 and j.trial.epochs == 5 for j in jobs))
        self.assertTrue(all(j.trial.selection_metric == "explicit" for j in jobs))
        self.assertTrue(all(min(j.trial.null_thresholds) == .02 for j in jobs[1:]))

    def test_two_seed_small_pilot_changes_only_null_settings(self):
        jobs = self.jobs("--null-pos-weight-caps", "10", "--first-seeds", "2")
        self.assertEqual(len(jobs), 4)
        self.assertEqual([j.seed for j in jobs], [13, 42, 13, 42])
        unchanged = ["loss_type", "ce_weight", "weighted_ce_weight", "loss_heads", "lr", "epochs",
                     "batch_size", "max_len", "warmup_ratio", "max_grad_norm", "vocabulary_scope"]
        self.assertTrue(all(all(getattr(j.trial, k) == getattr(jobs[0].trial, k) for k in unchanged) for j in jobs))

    def test_control_does_not_change_when_null_search_parameters_change(self):
        first = self.jobs()[0]
        second = self.jobs("--null-pos-weight-caps", "20", "--null-loss-weights", "1", "--null-thresholds", ".01", ".1", ".5")[0]
        self.assertEqual(first.directory, second.directory)
        self.assertEqual(asdict(first.trial), asdict(second.trial))

    def test_invalid_parameters_fail_before_launching(self):
        flags = [("--first-seeds", "0"), ("--lr", "nan"), ("--epochs", "0"),
                 ("--null-pos-weight-caps", ".5"), ("--null-loss-weights", "0"),
                 ("--null-pos-weight-caps", "10", "10"), ("--null-thresholds", "0", ".5")]
        for values in flags:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.jobs(*values)

    def test_commands_match_original_trainer_trial_ids(self):
        import run_experiments as runner
        for flags in ((), ("--winner", "focal_2", "--first-seeds", "2", "--null-loss-weights", ".5", "1")):
            for job in self.jobs(*flags):
                args = runner.parser().parse_args(job.command[2:])
                [(cfg, seed, domain)] = runner.make_plan(args)
                self.assertEqual(asdict(cfg), asdict(job.trial))
                self.assertEqual(cfg.trial_id, job.trial.trial_id)
                self.assertEqual(seed, job.seed)
                self.assertIsNone(domain)
                self.assertNotEqual(args.stage, "final")

    def test_plan_only_and_dry_run_never_launch(self):
        for flags in ([], ["--execute", "--dry-run"]):
            with (patch.object(pilot.fast, "execute_jobs") as launch,
                  patch.object(pilot.fast, "recorded_status", return_value="pending"),
                  patch.object(pilot.fast, "minutes_per_epoch", return_value=None), redirect_stdout(io.StringIO())):
                pilot.main(flags)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
