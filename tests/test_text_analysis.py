import unittest

from src.data import Example
from src.text_analysis import CorpusRecord, count_words, records_to_summary_rows, summarize_corpus


class TextAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            CorpusRecord(
                "laptop",
                "train",
                Example(
                    "Battery life isn't bad.",
                    [
                        ("Battery life", "BATTERY", "positive"),
                        ("NULL", "PRICE", "negative"),
                    ],
                    "laptop",
                ),
            ),
            CorpusRecord("hotel", "test", Example("Clean room", [], "hotel")),
        ]

    def test_count_words_handles_apostrophes(self):
        self.assertEqual(count_words("Battery life isn't bad."), 4)

    def test_summary_reports_raw_and_model_ready_counts(self):
        result = summarize_corpus(self.records, expected_files=2)
        self.assertEqual(result["validation"]["loaded_source_files"], 2)
        self.assertEqual(result["overall"]["examples"], 2)
        self.assertEqual(result["overall"]["total_triplets"], 2)
        self.assertEqual(result["overall"]["model_ready_triplets"], 1)
        self.assertEqual(result["overall"]["model_ready_percent"], 50.0)
        self.assertEqual(result["sentiment_counts"]["overall"]["negative"], 1)
        self.assertEqual(result["baseline_scope"]["implicit_triplets_excluded"], 1)

    def test_csv_rows_include_domain_length_statistics(self):
        result = summarize_corpus(self.records)
        rows = records_to_summary_rows(result)
        self.assertEqual(len(rows), 4)
        laptop_train = next(row for row in rows if row["domain"] == "laptop" and row["split"] == "train")
        self.assertEqual(laptop_train["examples"], 1)
        self.assertEqual(laptop_train["median_words_domain"], 4.0)


if __name__ == "__main__":
    unittest.main()

