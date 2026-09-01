import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.artifacts import atomic_write_json, atomic_write_jsonl, upsert_csv_row
from src.data import (
    Example,
    aligned_explicit_triplets,
    find_span,
    parse_line,
    summarize_task_scope,
)
from src.evaluate import complete_triplet_scores, decode_bio_to_spans
from src.losses import get_loss_fns
from src.train import compute_batch_loss


class ParseLineTests(unittest.TestCase):
    def test_tuple_annotations_are_trimmed_and_sentiments_normalized(self):
        example = parse_line(
            "Battery life is fine####"
            "((' Battery life ', ' BATTERY#QUALITY ', ' PoS '), "
            "('screen', 'DISPLAY#QUALITY', 'NEGATIVE'), "
            "('price', 'PRICE#GENERAL', 'Neu'))",
            domain="laptop",
        )

        self.assertEqual(example.sentence, "Battery life is fine")
        self.assertEqual(example.domain, "laptop")
        self.assertEqual(
            example.triplets,
            [
                ("Battery life", "BATTERY#QUALITY", "positive"),
                ("screen", "DISPLAY#QUALITY", "negative"),
                ("price", "PRICE#GENERAL", "neutral"),
            ],
        )

    def test_malformed_rows_are_rejected(self):
        malformed_rows = [
            "sentence without an annotation delimiter",
            "sentence####{'not': 'a sequence of triplets'}",
            "sentence####[('aspect', 'category')]",
            "sentence####[('aspect', 'category', 3)]",
            "sentence####this is not a Python literal",
        ]

        for row in malformed_rows:
            with self.subTest(row=row):
                with self.assertRaises((ValueError, SyntaxError)):
                    parse_line(row)


class ExplicitAspectFilteringTests(unittest.TestCase):
    def test_case_insensitive_span_uses_sentence_casing(self):
        sentence = "The Battery LIFE is excellent."

        self.assertEqual(find_span(sentence, "battery life"), (4, 16))
        example = Example(
            sentence=sentence,
            triplets=[
                ("battery life", "BATTERY#QUALITY", "positive"),
                ("NULL", "LAPTOP#GENERAL", "positive"),
                ("charger", "ACCESSORIES#GENERAL", "negative"),
            ],
            domain="laptop",
        )

        self.assertEqual(
            aligned_explicit_triplets(example),
            [("Battery LIFE", "BATTERY#QUALITY", "positive")],
        )

    def test_single_pair_scope_deduplicates_and_reports_exclusions(self):
        example = Example(
            sentence="battery and screen",
            triplets=[
                ("battery", "BATTERY", "positive"),
                ("battery", "BATTERY", "positive"),
                ("battery", "POWER", "negative"),
                ("battery and screen", "DEVICE", "neutral"),
                ("screen", "DISPLAY", "positive"),
                ("NULL", "DEVICE", "positive"),
                ("keyboard", "INPUT", "negative"),
            ],
        )

        self.assertEqual(
            aligned_explicit_triplets(example),
            [
                ("battery", "BATTERY", "positive"),
                ("screen", "DISPLAY", "positive"),
            ],
        )
        summary = summarize_task_scope([example])
        self.assertEqual(summary["included_explicit_triplets"], 2)
        self.assertEqual(summary["duplicate_triplets_excluded"], 1)
        self.assertEqual(summary["additional_label_pairs_excluded"], 1)
        self.assertEqual(summary["overlapping_triplets_excluded"], 1)
        self.assertEqual(summary["implicit_triplets_excluded"], 1)
        self.assertEqual(summary["unaligned_triplets_excluded"], 1)


class BioDecodingTests(unittest.TestCase):
    def test_valid_mask_closes_spans_and_orphan_i_starts_a_span(self):
        spans = decode_bio_to_spans(
            [0, 1, 2, 0, 2, 2, 1],
            [False, True, True, True, True, False, True],
        )

        self.assertEqual(spans, [(1, 2), (4, 4), (6, 6)])

    def test_invalid_decoder_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            decode_bio_to_spans([0, 1], [True])
        with self.assertRaises(ValueError):
            decode_bio_to_spans([0, 7, 0])


class CompleteTripletMetricTests(unittest.TestCase):
    def test_list_triplets_are_accepted_with_exact_set_semantics(self):
        gold = [
            [
                ["battery", "BATTERY", "positive"],
                ["screen", "DISPLAY", "negative"],
            ]
        ]
        pred = [
            [
                ["battery", "BATTERY", "positive"],
                ["screen", "DISPLAY", "positive"],
            ]
        ]

        scores = complete_triplet_scores(gold, pred)

        self.assertEqual(scores["true_positives"], 1)
        self.assertEqual(scores["false_positives"], 1)
        self.assertEqual(scores["false_negatives"], 1)
        self.assertAlmostEqual(scores["micro_f1"], 0.5)

    def test_unequal_example_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            complete_triplet_scores([[['a', 'c', 'positive']]], [])


class WeightedLossTests(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            loss_type="weighted",
            category_loss_weight=0.5,
            sentiment_loss_weight=2.0,
        )

    def test_explicit_vocab_sizes_include_classes_absent_from_training(self):
        bio_fn, category_fn, sentiment_fn = get_loss_fns(
            self._config(),
            bio_label_counts=[0, 0, 1, 2],
            category_label_counts=[0, 0],
            sentiment_label_counts=[1],
            num_categories=3,
            num_sentiments=4,
        )

        self.assertEqual(bio_fn.weight.numel(), 3)
        self.assertEqual(category_fn.weight.numel(), 3)
        self.assertEqual(sentiment_fn.weight.numel(), 4)
        self.assertEqual(category_fn.weight[1:].tolist(), [1.0, 1.0])
        self.assertEqual(
            [sentiment_fn.weight[index].item() for index in (0, 2, 3)],
            [1.0, 1.0, 1.0],
        )

    def test_unknown_targets_are_ignored_without_poisoning_batch_loss(self):
        cfg = self._config()
        loss_fns = get_loss_fns(
            cfg,
            bio_label_counts=[0, 1],
            category_label_counts=[0],
            sentiment_label_counts=[0],
            num_categories=3,
            num_sentiments=4,
        )
        bio_logits = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]], requires_grad=True,
        )
        category_logits = torch.tensor(
            [[0.0, 2.0, -1.0], [3.0, 0.0, -2.0]], requires_grad=True,
        )
        sentiment_logits = torch.tensor(
            [
                [3.0, 0.0, -1.0, -2.0],
                [0.0, -1.0, 2.0, -2.0],
            ],
            requires_grad=True,
        )
        batch = {
            "bio_labels": torch.tensor([[0, 1]]),
            "category_labels": [torch.tensor([1, -100])],
            "sentiment_labels": [torch.tensor([-100, 2])],
        }

        actual = compute_batch_loss(
            bio_logits,
            category_logits,
            sentiment_logits,
            batch,
            loss_fns,
            cfg,
            torch.device("cpu"),
        )
        expected = (
            loss_fns[0](bio_logits.reshape(-1, 3), batch["bio_labels"].reshape(-1))
            + cfg.category_loss_weight
            * loss_fns[1](category_logits[:1], torch.tensor([1]))
            + cfg.sentiment_loss_weight
            * loss_fns[2](sentiment_logits[1:], torch.tensor([2]))
        )

        self.assertTrue(math.isfinite(actual.item()))
        self.assertTrue(torch.allclose(actual, expected))

    def test_all_unknown_span_targets_leave_only_the_bio_loss(self):
        cfg = self._config()
        loss_fns = get_loss_fns(
            cfg,
            bio_label_counts=[0],
            category_label_counts=[],
            sentiment_label_counts=[],
            num_categories=2,
            num_sentiments=3,
        )
        bio_logits = torch.tensor([[[2.0, 0.0, -1.0]]])
        category_logits = torch.zeros((1, 2))
        sentiment_logits = torch.zeros((1, 3))
        batch = {
            "bio_labels": torch.tensor([[0]]),
            "category_labels": [torch.tensor([-100])],
            "sentiment_labels": [torch.tensor([-100])],
        }

        actual = compute_batch_loss(
            bio_logits,
            category_logits,
            sentiment_logits,
            batch,
            loss_fns,
            cfg,
            torch.device("cpu"),
        )
        expected = loss_fns[0](bio_logits.reshape(-1, 3), torch.tensor([0]))

        self.assertTrue(math.isfinite(actual.item()))
        self.assertTrue(torch.allclose(actual, expected))


class ArtifactPersistenceTests(unittest.TestCase):
    def test_atomic_json_and_jsonl_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            json_path = root / "nested" / "metrics.json"
            jsonl_path = root / "predictions.jsonl"

            atomic_write_json(json_path, {"score": 0.5, "labels": ["café"]})
            atomic_write_json(json_path, {"score": 0.75})
            atomic_write_jsonl(
                jsonl_path,
                [{"id": 1, "term": "café"}, {"id": 2, "term": "screen"}],
            )

            with json_path.open(encoding="utf-8") as file:
                self.assertEqual(json.load(file), {"score": 0.75})
            with jsonl_path.open(encoding="utf-8") as file:
                rows = [json.loads(line) for line in file if line.strip()]
            self.assertEqual(
                rows,
                [{"id": 1, "term": "café"}, {"id": 2, "term": "screen"}],
            )
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_csv_upsert_keeps_one_row_per_logical_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "results.csv"
            fieldnames = ["mode", "config", "seed", "micro_f1"]
            key_fields = ["mode", "config", "seed"]

            first_replaced = upsert_csv_row(
                destination,
                {
                    "mode": "indomain",
                    "config": "standard",
                    "seed": 42,
                    "micro_f1": 0.1,
                },
                fieldnames,
                key_fields,
            )
            second_replaced = upsert_csv_row(
                destination,
                {
                    "mode": "indomain",
                    "config": "standard",
                    "seed": "42",
                    "micro_f1": 0.8,
                },
                fieldnames,
                key_fields,
            )
            distinct_replaced = upsert_csv_row(
                destination,
                {
                    "mode": "indomain",
                    "config": "weighted",
                    "seed": 42,
                    "micro_f1": 0.9,
                },
                fieldnames,
                key_fields,
            )

            with destination.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual((first_replaced, second_replaced, distinct_replaced), (0, 1, 0))
            self.assertEqual(len(rows), 2)
            keys = {(row["mode"], row["config"], row["seed"]) for row in rows}
            self.assertEqual(len(keys), 2)
            standard = next(row for row in rows if row["config"] == "standard")
            self.assertEqual(standard["micro_f1"], "0.8")
            self.assertEqual(list(destination.parent.glob(".results.csv.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
