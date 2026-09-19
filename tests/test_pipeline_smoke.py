import csv
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

import config as cfg
import run_study
import src.train as train_module
from src.data import Example


class TinyTokenizer:
    """Whitespace tokenizer with offsets and a bounded, local token id space."""

    is_fast = True

    def __call__(
        self,
        sentence,
        truncation,
        max_length,
        padding,
        return_offsets_mapping,
    ):
        del truncation, padding, return_offsets_mapping
        matches = list(re.finditer(r"\S+", sentence))[: max_length - 2]
        token_ids = [3 + sum(match.group().encode("utf-8")) % 57 for match in matches]
        input_ids = [1, *token_ids, 2]
        offsets = [(0, 0), *((match.start(), match.end()) for match in matches), (0, 0)]
        attention_mask = [1] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids.extend([0] * padding_length)
        attention_mask.extend([0] * padding_length)
        offsets.extend([(0, 0)] * padding_length)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
        }

    def save_pretrained(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "tokenizer_config.json").write_text(
            '{"tokenizer_class": "TinyTokenizer"}\n', encoding="utf-8",
        )
        return (str(destination),)


class TinyABSAModel(nn.Module):
    """Small trainable substitute that implements the production model API."""

    def __init__(self, num_categories, num_sentiments):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.bio_head = nn.Linear(8, 3)
        self.category_head = nn.Linear(8, num_categories)
        self.sentiment_head = nn.Linear(8, num_sentiments)
        with torch.no_grad():
            self.embedding.weight.fill_(0.05)
            self.bio_head.weight.zero_()
            self.bio_head.bias.copy_(torch.tensor([0.2, 0.1, 0.0]))
            self.category_head.weight.zero_()
            self.category_head.bias.zero_()
            self.sentiment_head.weight.zero_()
            self.sentiment_head.bias.zero_()

    def encode_and_tag(self, input_ids, attention_mask):
        del attention_mask
        hidden_states = self.embedding(input_ids)
        return hidden_states, self.bio_head(hidden_states)

    def classify_spans(self, hidden_states, span_boundaries):
        span_vectors = []
        for batch_index, spans in enumerate(span_boundaries):
            for start, end in spans:
                span_vectors.append(
                    hidden_states[batch_index, start : end + 1].mean(dim=0),
                )
        if not span_vectors:
            return None, None
        span_tensor = torch.stack(span_vectors)
        return self.category_head(span_tensor), self.sentiment_head(span_tensor)


def _synthetic_split():
    train = [
        Example(
            "Battery life is good",
            [("Battery life", "BATTERY", "positive")],
            "laptop",
        ),
        Example(
            "Screen is bad",
            [("Screen", "DISPLAY", "negative")],
            "laptop",
        ),
    ]
    dev = [
        Example(
            "Battery is good",
            [("Battery", "BATTERY", "positive")],
            "laptop",
        ),
        Example(
            "Screen is dull",
            [("Screen", "DISPLAY", "negative")],
            "laptop",
        ),
    ]
    test = [
        Example(
            "Screen is bright",
            [("Screen", "DISPLAY", "positive")],
            "laptop",
        ),
        Example(
            "Battery drains fast",
            [("Battery", "BATTERY", "negative")],
            "laptop",
        ),
        Example(
            "Keyboard is fine",
            [("Keyboard", "INPUT", "positive")],
            "laptop",
        ),
    ]
    return train, dev, test


class PipelineSmokeTests(unittest.TestCase):
    def test_standard_and_weighted_runs_write_complete_unique_artifacts(self):
        train_examples, dev_examples, test_examples = _synthetic_split()
        experiments = [
            cfg.ExperimentConfig(
                name=name,
                loss_type=name,
                model_name="tiny-local-model",
                max_len=10,
                batch_size=2,
                epochs=1,
                lr=0.01,
                weight_decay=0.0,
            )
            for name in ("standard", "weighted")
        ]

        def build_tiny_model(_experiment_cfg, num_categories, num_sentiments):
            return TinyABSAModel(num_categories, num_sentiments)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "runs"
            results_csv = temporary_root / "results.csv"

            with (
                patch.object(
                    run_study,
                    "_split_examples",
                    return_value=(train_examples, dev_examples, test_examples),
                ),
                patch.object(
                    run_study.AutoTokenizer,
                    "from_pretrained",
                    side_effect=lambda _name: TinyTokenizer(),
                ),
                patch.object(run_study.cfg, "EXPERIMENTS", experiments),
                patch.object(train_module, "build_model", side_effect=build_tiny_model),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                summaries = {
                    name: run_study.run(
                        mode="indomain",
                        config_name=name,
                        seed=7,
                        device="cpu",
                        output_dir=output_root,
                        results_csv=results_csv,
                    )
                    for name in ("standard", "weighted")
                }
                standard_checkpoint = (
                    output_root / "indomain" / "standard" / "seed_7"
                    / "best_model.pt"
                )
                standard_checkpoint.unlink()
                with self.assertRaisesRegex(RuntimeError, "missing artifacts"):
                    run_study.run(
                        mode="indomain",
                        config_name="standard",
                        seed=7,
                        device="cpu",
                        output_dir=output_root,
                        results_csv=results_csv,
                    )
                # Exercise atomic replacement and logical-key CSV upsert.
                run_study.run(
                    mode="indomain",
                    config_name="standard",
                    seed=7,
                    device="cpu",
                    output_dir=output_root,
                    results_csv=results_csv,
                    overwrite=True,
                )

            for name in ("standard", "weighted"):
                with self.subTest(config=name):
                    run_dir = output_root / "indomain" / name / "seed_7"
                    expected_files = {
                        "manifest.json",
                        "history.json",
                        "best_model.pt",
                        "metrics.json",
                        "dev_predictions.jsonl",
                        "test_predictions.jsonl",
                    }
                    self.assertTrue(
                        expected_files.issubset(
                            {path.name for path in run_dir.iterdir() if path.is_file()},
                        ),
                    )

                    with (run_dir / "manifest.json").open(encoding="utf-8") as file:
                        manifest = json.load(file)
                    with (run_dir / "history.json").open(encoding="utf-8") as file:
                        history = json.load(file)
                    with (run_dir / "metrics.json").open(encoding="utf-8") as file:
                        metrics = json.load(file)
                    with (run_dir / "dev_predictions.jsonl").open(encoding="utf-8") as file:
                        dev_records = [json.loads(line) for line in file]
                    with (run_dir / "test_predictions.jsonl").open(encoding="utf-8") as file:
                        test_records = [json.loads(line) for line in file]
                    checkpoint = torch.load(
                        run_dir / "best_model.pt",
                        map_location="cpu",
                        weights_only=True,
                    )

                    self.assertEqual(manifest["status"], "complete")
                    self.assertEqual(
                        manifest["split_sizes"], {"train": 2, "dev": 2, "test": 3},
                    )
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]["epoch"], 1)
                    self.assertEqual(metrics["summary"]["status"], "complete")
                    self.assertEqual(metrics["summary"]["config"], name)
                    self.assertEqual(summaries[name]["test_size"], 3)
                    self.assertIn("model_state_dict", checkpoint)
                    self.assertEqual(checkpoint["config"]["epochs"], 1)
                    self.assertEqual(len(dev_records), len(dev_examples))
                    self.assertEqual(
                        [record["sentence"] for record in test_records],
                        [example.sentence for example in test_examples],
                    )
                    self.assertEqual(
                        [record["example_id"] for record in test_records], [0, 1, 2],
                    )

            with results_csv.open(encoding="utf-8", newline="") as file:
                result_rows = list(csv.DictReader(file))
            result_keys = {
                (row["mode"], row["config"], row["seed"]) for row in result_rows
            }
            self.assertEqual(len(result_rows), 2)
            self.assertEqual(len(result_keys), 2)
            self.assertEqual(
                result_keys,
                {("indomain", "standard", "7"), ("indomain", "weighted", "7")},
            )

    def test_explicit_holdout_is_part_of_artifact_path_and_result_key(self):
        train_examples, dev_examples, test_examples = _synthetic_split()
        experiment = cfg.ExperimentConfig(
            name="standard",
            loss_type="standard",
            model_name="tiny-local-model",
            max_len=10,
            batch_size=2,
            epochs=1,
            lr=0.01,
            weight_decay=0.0,
        )

        def build_tiny_model(_experiment_cfg, num_categories, num_sentiments):
            return TinyABSAModel(num_categories, num_sentiments)

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "lodo_runs"
            results_csv = temporary_root / "results_lodo.csv"

            with (
                patch.object(
                    run_study,
                    "_split_examples",
                    return_value=(train_examples, dev_examples, test_examples),
                ),
                patch.object(
                    run_study.AutoTokenizer,
                    "from_pretrained",
                    side_effect=lambda _name: TinyTokenizer(),
                ),
                patch.object(run_study.cfg, "EXPERIMENTS", [experiment]),
                patch.object(train_module, "build_model", side_effect=build_tiny_model),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                for held_out_domain in ("laptop", "food"):
                    run_study.run(
                        mode="crossdomain",
                        held_out_domain=held_out_domain,
                        config_name="standard",
                        seed=7,
                        device="cpu",
                        output_dir=output_root,
                        results_csv=results_csv,
                    )

            for held_out_domain in ("laptop", "food"):
                run_dir = (
                    output_root / held_out_domain / "standard" / "seed_7"
                )
                self.assertTrue((run_dir / "manifest.json").exists())
                with (run_dir / "manifest.json").open(encoding="utf-8") as file:
                    manifest = json.load(file)
                self.assertEqual(manifest["held_out_domain"], held_out_domain)

            with results_csv.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            keys = {
                (
                    row["mode"],
                    row["held_out_domain"],
                    row["config"],
                    row["seed"],
                )
                for row in rows
            }
            self.assertEqual(
                keys,
                {
                    ("crossdomain", "laptop", "standard", "7"),
                    ("crossdomain", "food", "standard", "7"),
                },
            )


if __name__ == "__main__":
    unittest.main()
