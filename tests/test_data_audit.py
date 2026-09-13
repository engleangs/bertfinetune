import re
import unittest

from src.data import Example
from src.dataset_audit import (
    aggregate_audits,
    audit_split,
    find_all_spans,
    repeated_term_summary,
    split_overlap_summary,
)


class TinyTokenizer:
    is_fast = True

    def __call__(self, sentence, truncation, max_length, padding, return_offsets_mapping):
        del truncation, padding, return_offsets_mapping
        matches = list(re.finditer(r"\S+", sentence))[: max_length - 2]
        offsets = [(0, 0), *((match.start(), match.end()) for match in matches), (0, 0)]
        attention = [1] * len(offsets)
        input_ids = list(range(1, len(offsets) + 1))
        pad = max_length - len(offsets)
        return {
            "input_ids": input_ids + [0] * pad,
            "attention_mask": attention + [0] * pad,
            "offset_mapping": offsets + [(0, 0)] * pad,
        }


class RepeatedTermAuditTests(unittest.TestCase):
    def test_finds_all_case_insensitive_occurrences(self):
        self.assertEqual(
            find_all_spans("Service was slow; SERVICE improved", "service"),
            [(0, 7), (18, 25)],
        )
        self.assertEqual(find_all_spans("Nothing explicit", "NULL"), [])

    def test_distinguishes_repeated_annotations_from_ambiguous_text(self):
        examples = [
            Example(
                "service improved and service smiled",
                [
                    ("service", "QUALITY", "positive"),
                    ("service", "STAFF", "positive"),
                ],
            )
        ]
        self.assertEqual(
            repeated_term_summary(examples),
            {
                "repeated_annotated_aspect_text_examples": 1,
                "ambiguous_repeated_occurrence_examples": 1,
                "ambiguous_repeated_occurrence_triplets": 2,
            },
        )


class SplitAuditTests(unittest.TestCase):
    def test_reports_exclusions_unknown_labels_and_truncation(self):
        examples = [
            Example(
                "battery is good but late term fails",
                [
                    ("battery", "BATTERY", "positive"),
                    ("late term", "UNSEEN", "negative"),
                    ("NULL", "DEVICE", "positive"),
                ],
                "laptop",
            ),
            Example("empty", [], "laptop"),
        ]
        result = audit_split(
            "laptop", "test", examples, TinyTokenizer(),
            ["BATTERY", "DEVICE"], ["positive", "negative"], max_len=5,
        )
        self.assertEqual(result["examples"], 2)
        self.assertEqual(result["empty_annotation_examples"], 1)
        self.assertEqual(result["implicit_triplets_excluded"], 1)
        self.assertEqual(result["included_explicit_triplets"], 2)
        self.assertEqual(result["token_alignment_or_truncation_misses"], 1)
        self.assertEqual(result["unseen_categories"], ["UNSEEN"])
        self.assertEqual(result["unknown_category_targets"], 0)

    def test_aggregates_counts_and_unions_label_names(self):
        rows = [
            {"domain": "a", "split": "train", "examples": 2, "unseen_categories": ["X"], "unseen_sentiments": []},
            {"domain": "b", "split": "test", "examples": 3, "unseen_categories": ["X", "Y"], "unseen_sentiments": ["mixed"]},
        ]
        self.assertEqual(
            aggregate_audits(rows),
            {"examples": 5, "unseen_categories": ["X", "Y"], "unseen_sentiments": ["mixed"]},
        )

    def test_counts_sentence_and_labeled_overlap_separately(self):
        train = [Example("same", [("same", "A", "positive")])]
        dev = [Example("same", [("same", "B", "negative")])]
        test = [Example("same", [("same", "A", "positive")])]
        result = split_overlap_summary(train, dev, test)
        self.assertEqual(result["train_dev_repeated_sentences"], 1)
        self.assertEqual(result["train_test_identical_labeled_examples"], 1)
        self.assertEqual(result["train_dev_identical_labeled_examples"], 0)


if __name__ == "__main__":
    unittest.main()
