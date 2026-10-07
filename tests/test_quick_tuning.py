"""Adaptive selection, frozen evaluation and report checks without training."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import run_quick_tuning as quick
from experiment_reporting import quick_report
from src.experiment_config import TrialConfig, candidate_recipes


class QuickTuningTests(unittest.TestCase):
    def setUp(self):
        self.base = quick.fast.ROOT / "artifacts/quick_tests"
        self.root = self.base / ("fixture-" + uuid.uuid4().hex)
        self.output = self.root / "models"
        self.study = self.root / "study"
        self.output.mkdir(parents=True)
        self.args = quick.parser().parse_args(["--output", str(self.output), "--study-output", str(self.study)])

    def tearDown(self):
        assert self.root.resolve().is_relative_to(self.base.resolve())
        shutil.rmtree(self.root)

    def add_result(self, cfg, seed, explicit=.4, combined=.35, null=.0, status="dev_complete"):
        directory = self.output / "research/indomain/all_domains" / cfg.trial_id / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"status": status, "fingerprint": f"fixture-{cfg.trial_id}-{seed}",
                    "checkpoint_sha256": "synthetic-checkpoint-hash", "training": {"best_epoch": 5},
                    "identity": {"config": asdict(cfg), "seed": seed, "mode": "indomain", "held_out_domain": None,
                                 "smoke": False, "source_sha256": {"fixture": "code"},
                                 "source_data_sha256": {"fixture": "data"}, "environment": {"fixture": True},
                                 "resolved_device": "cpu", "split_limits": None}}
        metrics = {"explicit": {"micro_f1": explicit, "micro_precision": explicit, "micro_recall": explicit},
                   "combined": {"micro_f1": combined}, "null": {"micro_f1": null, "true_positives": 0,
                   "false_positives": 0, "false_negatives": 1}, "unweighted_loss": 1.0, "null_threshold": .1,
                   "taxonomy": {"gold_triplets": 2,
                                "primary_outcome_counts": {"correct": 1, "term": 1, "category": 0, "sentiment": 0},
                                "metrics": {"aspect_span": {"f1": .5}, "term_plus_category": {"f1": .5},
                                            "aspect_plus_sentiment": {"f1": .5}}}}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "dev_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return directory

    def frozen(self, cfg=None):
        cfg = cfg or TrialConfig(name="standard", loss_type="standard", model_revision=quick.fast.pinned_revision())
        directory = self.add_result(cfg, 13)
        manifest = json.loads((directory / "manifest.json").read_text())
        return {"seeds": [13], "variants": [{"trial_id": cfg.trial_id, "config": asdict(cfg),
                 "roles": ["Best NULL off"], "runs": [{"seed": 13, "directory": str(directory),
                 "fingerprint": manifest["fingerprint"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                 "best_epoch": 5, "null_threshold": .1}]}]}

    def test_selection_never_opens_test_files_or_uses_failed_runs(self):
        directory = self.add_result(TrialConfig(name="standard", loss_type="standard"), 13, status="complete")
        (directory / "test_metrics.json").write_text("invalid JSON: must not be opened", encoding="utf-8")
        self.add_result(TrialConfig(name="failed", loss_type="standard"), 13, explicit=1, status="failed")
        rows = quick.development_rows(self.output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(quick.best_row(rows, "explicit")["metrics"]["explicit"]["micro_f1"], .4)

    def test_reconstructed_commands_keep_original_trial_id_and_final_stage(self):
        import run_experiments as runner
        cfgs = [TrialConfig(**recipe, model_revision=quick.fast.pinned_revision(), lr=3e-5)
                for recipe in candidate_recipes("loss") if recipe["name"] in quick.fast.SCREEN]
        cfgs.append(quick.null_jobs(self.args, {"config": {"name": "mixed_025", "lr": 3e-5}})[-1].trial)
        for cfg in cfgs:
            for stage in ("pilot", "final"):
                job = quick.job_for_config(cfg, 42, self.args, stage)
                args = runner.parser().parse_args(job.command[2:])
                [(actual, seed, domain)] = runner.make_plan(args)
                self.assertEqual(asdict(actual), asdict(cfg))
                self.assertEqual(actual.trial_id, cfg.trial_id)
                self.assertEqual(seed, 42)
                self.assertEqual(args.stage, stage)
                self.assertIsNone(domain)

    def test_freeze_rejects_changed_threshold_config_or_missing_seed(self):
        frozen = self.frozen()
        self.assertEqual(len(quick.final_jobs(frozen, self.args)), 1)
        run = frozen["variants"][0]["runs"][0]
        path = Path(run["directory"]) / "dev_metrics.json"
        original = path.read_text()
        raw = json.loads(original)
        raw["null_threshold"] = .2
        path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "threshold changed"):
            quick.final_jobs(frozen, self.args)
        path.write_text(original)
        frozen["variants"][0]["config"]["lr"] = 3e-5
        with self.assertRaisesRegex(ValueError, "identity"):
            quick.final_jobs(frozen, self.args)
        frozen = self.frozen()
        frozen["seeds"] = [13, 42]
        with self.assertRaisesRegex(ValueError, "declared seeds"):
            quick.final_jobs(frozen, self.args)

    def test_completed_configuration_must_match_requested_job(self):
        cfg = TrialConfig(name="standard", loss_type="standard", model_revision=quick.fast.pinned_revision())
        directory = self.add_result(cfg, 13)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["identity"]["config"]["ce_weight"] = 2
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "configuration"):
            quick.matching_rows(self.output, [quick.job_for_config(cfg, 13, self.args)])

    def test_all_step_selects_on_dev_freezes_then_tests_only_two_seeds(self):
        calls = []
        base = {"standard": .31, "weighted": .23, "mixed_025": .34,
                "mixed_033": .33, "mixed_050": .30, "focal_2": .29}

        def execute(jobs, output, parser):
            for job in jobs:
                stage = job.command[job.command.index("--stage") + 1]
                calls.append((stage, job.trial, job.seed))
                if stage == "final":
                    self.assertTrue((self.study / "selection.json").is_file())
                    manifest_path = job.directory / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["status"] = "complete"
                    manifest_path.write_text(json.dumps(manifest))
                    raw = json.loads((job.directory / "dev_metrics.json").read_text())
                    raw["explicit"]["micro_f1"] = .01 if job.trial.null_head else .99
                    (job.directory / "test_metrics.json").write_text(json.dumps(raw))
                    continue
                if (job.directory / "manifest.json").exists():
                    continue
                cfg = job.trial
                score = base[cfg.name]
                if cfg.lr == 3e-5:
                    score += .03
                elif cfg.lr == 1e-5:
                    score -= .08
                if cfg.vocabulary_scope == "explicit-null":
                    score -= .01
                combined, null = score - .04, 0
                if cfg.null_head:
                    score -= .01
                    null = .3 if cfg.null_pos_weight_cap == 10 else .4
                    combined = .35 if cfg.null_pos_weight_cap == 10 else .36
                if job.seed == 42:
                    score += .01
                    combined += .01
                self.add_result(cfg, job.seed, score, combined, null)

        flags = ["--output", str(self.output), "--study-output", str(self.study), "--execute"]
        with (patch.object(quick.fast, "execute_jobs", side_effect=execute),
              patch.object(quick_report, "figure"), redirect_stdout(io.StringIO())):
            quick.main(flags)
        frozen = json.loads((self.study / "selection.json").read_text())
        off = next(v for v in frozen["variants"] if "Best NULL off" in v["roles"])
        on = next(v for v in frozen["variants"] if "Best NULL on" in v["roles"])
        self.assertEqual((off["config"]["name"], off["config"]["lr"]), ("mixed_025", 3e-5))
        self.assertEqual(on["config"]["null_pos_weight_cap"], 30)
        finals = [call for call in calls if call[0] == "final"]
        self.assertEqual(len(finals), 6)
        self.assertEqual({seed for _, _, seed in calls}, {13, 42})
        self.assertEqual({cfg.name for stage, cfg, _ in calls if stage == "pilot" and cfg.lr == 1e-5}, {"mixed_025"})
        data = json.loads((self.study / "results.json").read_text())
        self.assertEqual(sum(r["split"] == "test" for r in data["runs"]), 6)
        # Even contradictory test scores never change the chosen settings.
        self.assertEqual(on["config"]["null_pos_weight_cap"], 30)

    def test_search_only_does_not_evaluate_tests_and_frozen_search_is_reused(self):
        frozen = self.frozen()
        quick.write_freeze(self.study / "selection.json", frozen)
        self.args.first_seeds = 1
        with patch.object(quick.fast, "execute_jobs") as execute, redirect_stdout(io.StringIO()):
            actual = quick.search(self.args, quick.parser())
        execute.assert_not_called()
        self.assertEqual(actual, json.loads(json.dumps(frozen)))
        self.args.first_seeds = 2
        with self.assertRaisesRegex(ValueError, "different seeds"):
            quick.search(self.args, quick.parser())

    def test_pending_report_is_honest_and_changed_freeze_is_rejected(self):
        frozen = self.frozen()
        with patch.object(quick_report, "figure"):
            quick_report.write_quick_report(frozen, self.study)
        self.assertIn("Test evaluation is pending", (self.study / "report.md").read_text())
        data = json.loads((self.study / "results.json").read_text())
        self.assertEqual([r["split"] for r in data["runs"]], ["dev"])
        frozen["variants"][0]["runs"][0]["checkpoint_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "changed"):
            quick_report.load_selected(frozen)

    def test_plan_dry_run_and_invalid_caps_do_not_execute(self):
        for flags in ([], ["--execute", "--dry-run"], ["--execute", "--null-pos-weight-caps", ".5"]):
            with (patch.object(quick.fast, "execute_jobs") as execute,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                if ".5" in flags:
                    with self.assertRaises(SystemExit):
                        quick.main(flags)
                else:
                    quick.main(flags)
            execute.assert_not_called()

    def test_null_contribution_uses_only_matched_seeds(self):
        frozen = {"variants": [{"trial_id": "on", "roles": ["Best NULL on"]},
                                {"trial_id": "off", "roles": ["Vocabulary control"]}]}
        rows = [{"trial_id": "on", "seed": 13, "split": "test",
                 "metrics": {"explicit_f1": .5, "combined_f1": .4, "null_f1": .2}},
                {"trial_id": "off", "seed": 13, "split": "test",
                 "metrics": {"explicit_f1": .4, "combined_f1": .3, "null_f1": 0}},
                {"trial_id": "off", "seed": 42, "split": "test",
                 "metrics": {"explicit_f1": .9, "combined_f1": .9, "null_f1": 0}}]
        table = "\n".join(quick_report.paired_null_table(frozen, rows))
        self.assertIn("| test | [13] | +10.00 | +10.00 | +20.00 |", table)


if __name__ == "__main__":
    unittest.main()
