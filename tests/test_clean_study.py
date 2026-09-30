import copy
import gzip
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

import config as cfg
import run_clean_study as study
import src.data as data_module
import src.model as model_module
import src.train as train_module
from src.artifacts import atomic_write_json, atomic_write_jsonl_gzip
from src.data import ABSADataset
from src.result_analysis import load_prediction_file
from src.study_data import file_digest, prepare_clean_data, split_paths, verify_source_splits
from src.study_plan import (FOLDS, LEARNING_RATES, baseline_tasks, search_tasks, main_tasks,
                            choose_recipes, final_tasks, completion_tasks, smoke_tasks)
from src.study_storage import lightweight_export, read_json, safe_checkpoint_path
from tests.test_pipeline_smoke import TinyABSAModel, TinyTokenizer, _synthetic_split


def search_summaries():
    return [{"task": t.payload(), "status": "trained", "best_dev_micro_f1":
             0.8 if t.lr == LEARNING_RATES[FOLDS.index(t.fold) % 3] else 0.2}
            for t in search_tasks()]


def write_evidence(root, task, score=0.4, evaluated=False):
    directory = study.run_dir(root, task)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory / "dev_summary.json", {
        "task": task.payload(), "status": "trained", "best_dev_micro_f1": score, "best_epoch": 2})
    atomic_write_json(directory / "history.json", [{"epoch": 1, "dev_micro_f1": score}])
    atomic_write_json(directory / "dev_metrics.json", {"loss": 1.0})
    atomic_write_jsonl_gzip(directory / "dev_predictions.jsonl.gz", [{"sentence": "example"}])
    names = ["dev_summary.json", "history.json", "dev_metrics.json", "dev_predictions.jsonl.gz"]
    if evaluated:
        atomic_write_json(directory / "metrics.json", {"test": {
            "complete_triplet": {"micro_f1": score, "micro_precision": score, "micro_recall": score},
            "rare_category_recall": 0.0, "projections": {"gold_category_coverage": 0.5,
                "aspect_span": {"f1": 0.7}, "aspect_plus_sentiment": {"f1": 0.6},
                "source_known_exact_triplet": {"f1": 0.5}}}})
        atomic_write_jsonl_gzip(directory / "test_predictions.jsonl.gz", [{"sentence": "test example"}])
        names += ["metrics.json", "test_predictions.jsonl.gz"]
    atomic_write_json(directory / "manifest.json", {
        "task": task.payload(), "training_complete": True, "evaluation_complete": evaluated,
        "evidence_sha256": {name: file_digest(directory / name) for name in names}})
    (directory / "best_model.pt").write_bytes(b"checkpoint placeholder")


class CleanStudyPlanTests(unittest.TestCase):
    def test_budget_and_fold_independent_selection(self):
        selection = choose_recipes(search_summaries())
        self.assertEqual(len(baseline_tasks()), 20)
        self.assertEqual(len(search_tasks()), 96)
        self.assertEqual(len(final_tasks(selection)), 48)
        self.assertEqual(len(completion_tasks(selection)), 80)
        self.assertEqual(len(set(t.key for t in main_tasks())), 116)
        self.assertEqual([t.stage for t in main_tasks()[:4]], ["baseline", "search"] * 2)
        for i, fold in enumerate(FOLDS):
            self.assertEqual(selection["folds"][fold]["lr"], LEARNING_RATES[i % 3])
        rows = search_summaries()
        for row in rows:
            if row["task"]["fold"] == "indomain":
                row["best_dev_micro_f1"] = 0.9
        changed = choose_recipes(rows)
        for fold in FOLDS[1:]:
            self.assertEqual(changed["folds"][fold], selection["folds"][fold])

    def test_incomplete_invalid_and_tied_search(self):
        rows = search_summaries()
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                choose_recipes(invalid)
        rows[0]["best_dev_micro_f1"] = float("nan")
        with self.assertRaises(ValueError):
            choose_recipes(rows)
        for row in rows:
            row["best_dev_micro_f1"] = 0.5
        self.assertTrue(all(row["lr"] == 2e-5 for row in choose_recipes(rows)["folds"].values()))

    def test_real_data_audit_and_source_only_hashing(self):
        raw = study.PROJECT / "data/m-absa"
        before = {p: file_digest(raw / p) for p in split_paths()}
        with tempfile.TemporaryDirectory() as temporary:
            clean = Path(temporary) / "clean"
            report = prepare_clean_data(raw, clean)
            self.assertEqual(report["removed_sentence_count"], 5)
            self.assertEqual(report["removed_conflict_triplets"], 5)
            self.assertEqual(report["removed_other_triplets"], 2)
            for p in split_paths(splits=("dev", "test")):
                self.assertEqual((raw / p).read_bytes(), (clean / p).read_bytes())
            self.assertEqual(report, prepare_clean_data(raw, clean))
            with patch("src.study_data.file_digest", wraps=file_digest) as digest:
                verify_source_splits(clean, report, "clean", "restaurant")
            paths = [str(c.args[0]).replace("\\", "/") for c in digest.call_args_list]
            self.assertEqual(len(paths), 12)
            self.assertTrue(all("restaurant/" not in p and "test.txt" not in p for p in paths))
            (clean / "hotel/en/train.txt").write_text("corruption", encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_clean_data(raw, clean)
        self.assertEqual(before, {p: file_digest(raw / p) for p in split_paths()})


class CleanStudyStorageTests(unittest.TestCase):
    def test_legacy_export_is_small_verified_and_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "old"
            root.mkdir()
            (root / "best_model.pt").write_bytes(b"do not read or delete")
            atomic_write_json(root / "metrics.json", {"f1": 0.4})
            atomic_write_jsonl_gzip(root / "test_predictions.jsonl.gz", [{"text": "hello"}])
            before = {p.name: file_digest(p) for p in root.iterdir()}
            output = Path(temporary) / "results.zip"
            receipt = lightweight_export(root, output, legacy=True)
            self.assertFalse(receipt["model_weights_included"])
            with zipfile.ZipFile(output) as archive:
                self.assertNotIn("best_model.pt", archive.namelist())
                inventory = json.loads(archive.read("EXPORT_INVENTORY.json"))
                for row in inventory["files"]:
                    self.assertEqual(row["sha256"], hashlib.sha256(archive.read(row["path"])).hexdigest())
            self.assertEqual(before, {p.name: file_digest(p) for p in root.iterdir()})

    def test_pruning_gates_selection_and_final_retention(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(study, "load_study"), redirect_stdout(io.StringIO()):
            root = Path(temporary) / "new"
            old = Path(temporary) / "old.pt"
            old.write_bytes(b"old checkpoint")
            for row in search_summaries():
                task = next(t for t in search_tasks() if t.key == row["task"]["key"])
                write_evidence(root, task, row["best_dev_micro_f1"])
            selection = study.select(root)
            selected = {t.key for t in completion_tasks(selection) if t.stage == "search"}
            self.assertEqual(len(selected), 32)
            report = study.prune(root)
            self.assertEqual(len(report["removals"]), 64)
            self.assertEqual(len(list(root.rglob("best_model.pt"))), 96)
            damaged = next(t for t in search_tasks() if t.key not in selected)
            evidence = study.run_dir(root, damaged) / "dev_predictions.jsonl.gz"
            original = evidence.read_bytes()
            evidence.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                study.prune(root, apply=True)
            self.assertEqual(len(list(root.rglob("best_model.pt"))), 96)
            evidence.write_bytes(original)
            study.prune(root, apply=True)
            self.assertEqual(len(list(root.rglob("best_model.pt"))), 32)
            for task in search_tasks():
                if task.key in selected:
                    self.assertTrue((study.run_dir(root, task) / "best_model.pt").exists())
            # A modified selection cannot redirect completion or checkpoint deletion.
            tampered = copy.deepcopy(selection)
            tampered["folds"]["indomain"]["lr"] = 5e-5
            atomic_write_json(root / "selection.json", tampered)
            with self.assertRaises(ValueError):
                study.load_selection(root)
            atomic_write_json(root / "selection.json", selection)
            with self.assertRaises((FileNotFoundError, ValueError)):
                study.prune(root, scope="archive", apply=True)
            for task in [*baseline_tasks(), *completion_tasks(selection)]:
                score = (read_json(study.run_dir(root, task) / "dev_summary.json")["best_dev_micro_f1"]
                         if task.stage == "search" else 0.3)
                write_evidence(root, task, score, evaluated=True)
            for task in smoke_tasks():
                write_evidence(root, task)
            self.assertEqual(study.summarize(root)["result_count"], 100)
            lightweight_export(root, Path(temporary) / "final.zip")
            report = study.prune(root, scope="archive", apply=True)
            self.assertEqual(len(report["retained_model_keys"]), 16)
            self.assertEqual(len(list(root.rglob("best_model.pt"))), 16)
            self.assertEqual(study.summarize(root)["result_count"], 100)
            self.assertEqual(old.read_bytes(), b"old checkpoint")
            with self.assertRaises(ValueError):
                safe_checkpoint_path(root, "../best_model.pt")
            with self.assertRaises(ValueError):
                safe_checkpoint_path(root, "study.json")


class DropoutTinyModel(TinyABSAModel):
    def __init__(self, categories, sentiments):
        super().__init__(categories, sentiments)
        self.dropout = torch.nn.Dropout(0.4)
        with torch.no_grad():
            self.embedding.weight.normal_(0, 0.1)
            self.bio_head.weight.normal_(0, 0.1)

    def encode_and_tag(self, input_ids, attention_mask):
        hidden = self.dropout(self.embedding(input_ids))
        return hidden, self.bio_head(hidden)


class CleanStudyTrainingTests(unittest.TestCase):
    def test_resume_preserves_dropout_shuffle_optimizer_and_schedule(self):
        train, dev, _ = _synthetic_split()
        categories, sentiments = ["BATTERY", "DISPLAY"], ["negative", "positive"]
        train_ds = ABSADataset(train * 2, TinyTokenizer(), categories, sentiments, 10)
        dev_ds = ABSADataset(dev, TinyTokenizer(), categories, sentiments, 10)
        experiment = cfg.ExperimentConfig(name="weighted", loss_type="weighted", model_name="tiny",
            max_len=10, batch_size=1, epochs=4, lr=0.01, warmup_ratio=0.25, max_grad_norm=1)
        build = lambda config, nc, ns: DropoutTinyModel(nc, ns)
        with tempfile.TemporaryDirectory() as temporary, patch.object(train_module, "build_model", side_effect=build), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            complete = train_module.train_one_config(experiment, train_ds, dev_ds, categories, sentiments,
                13, device="cpu", resume_path=root / "complete.pt", run_identity="fixture")
            real_save = train_module.atomic_torch_save

            def interrupt_after_epoch_two(path, payload):
                real_save(path, payload)
                if payload.get("next_epoch") == 2:
                    raise RuntimeError("simulated interruption")

            with patch.object(train_module, "atomic_torch_save", side_effect=interrupt_after_epoch_two):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_module.train_one_config(experiment, train_ds, dev_ds, categories, sentiments,
                        13, device="cpu", resume_path=root / "resume.pt", run_identity="fixture")
            with self.assertRaisesRegex(ValueError, "identity"):
                train_module.train_one_config(experiment, train_ds, dev_ds, categories, sentiments,
                    13, device="cpu", resume_path=root / "resume.pt", resume=True, run_identity="wrong")
            resumed = train_module.train_one_config(experiment, train_ds, dev_ds, categories, sentiments,
                13, device="cpu", resume_path=root / "resume.pt", resume=True, run_identity="fixture")
            self.assertEqual(complete["history"], resumed["history"])
            self.assertEqual(complete["best_epoch"], resumed["best_epoch"])
            for name, value in complete["model"].state_dict().items():
                self.assertTrue(torch.equal(value, resumed["model"].state_dict()[name]), name)
            full = torch.load(root / "complete.pt", weights_only=False)
            restart = torch.load(root / "resume.pt", weights_only=False)
            self.assertEqual(full["scheduler_state_dict"], restart["scheduler_state_dict"])
            for name, value in full["model_state_dict"].items():
                self.assertTrue(torch.equal(value, restart["model_state_dict"][name]))

    def test_search_then_evaluation_and_skip_without_checkpoint(self):
        splits = dict(zip(("train", "dev", "test"), _synthetic_split()))
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            project = Path(temporary)
            (project / "cluster").mkdir()
            model_root = project / "model"
            model_root.mkdir()
            for name in study.MODEL_FILES:
                (model_root / name).write_bytes(b"fixture")
            for index, domain in enumerate(cfg.DOMAINS):
                for split, examples in splits.items():
                    path = project / "data/m-absa" / domain / "en" / f"{split}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    text = "".join(f"{ex.sentence}####{ex.triplets!r}\n" for ex in examples)
                    if split == "train" and index < 5:
                        text += "Mixed####[('Mixed', 'DISPLAY', 'conflict')]\n"
                    path.write_text(text, encoding="utf-8")
            output = project / "study"
            build = lambda config, nc, ns: TinyABSAModel(nc, ns)
            with patch.object(study, "PROJECT", project), patch.object(study, "model_directory", return_value=model_root), \
                    patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                    patch.object(train_module, "build_model", side_effect=build), \
                    patch.object(model_module, "build_model", side_effect=build):
                study.prepare(output)
                task = next(t for t in search_tasks() if t.fold == "restaurant" and t.loss == "weighted")
                with patch.object(data_module, "load_domain_file", wraps=data_module.load_domain_file) as loader:
                    study.run_task(task, output, device="cpu")
                paths = [str(c.args[0]).replace("\\", "/") for c in loader.call_args_list]
                self.assertTrue(all("restaurant/" not in p and "test.txt" not in p for p in paths))
                directory = study.run_dir(output, task)
                self.assertFalse((directory / "resume.pt").exists())
                self.assertFalse((directory / "metrics.json").exists())
                self.assertEqual(len(read_json(directory / "history.json")), 30)
                with patch.object(train_module, "train_one_config", side_effect=AssertionError("must not retrain")):
                    study.run_task(task, output, device="cpu", evaluate_only=True)
                    self.assertTrue(read_json(directory / "manifest.json")["evaluation_complete"])
                    with gzip.open(directory / "test_predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
                        self.assertEqual(len(list(handle)), 3)
                    gold, predicted = load_prediction_file(directory / "test_predictions.jsonl.gz")
                    self.assertEqual(len(gold), len(predicted))
                    self.assertEqual(len(gold), 3)
                    (directory / "best_model.pt").unlink()
                    study.run_task(task, output, device="cpu", evaluate_only=True)
                metrics = read_json(directory / "metrics.json")["test"]
                self.assertIn("projections", metrics)
                self.assertIn("loss_components", metrics)


if __name__ == "__main__":
    unittest.main()
