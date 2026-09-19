import unittest

import config as cfg
from run_lodo_study import build_run_matrix, default_remaining_domains


class LodoRunnerTests(unittest.TestCase):
    def test_default_matrix_covers_six_domains_without_restaurant(self):
        domains = default_remaining_domains()
        matrix = build_run_matrix(
            domains,
            ["standard", "weighted"],
            cfg.SEEDS,
        )
        self.assertEqual(len(domains), 6)
        self.assertNotIn(cfg.HOLD_OUT_DOMAIN, domains)
        self.assertEqual(len(matrix), 60)
        self.assertEqual(len(matrix), len(set(matrix)))

    def test_custom_smoke_matrix_and_invalid_duplicates(self):
        matrix = build_run_matrix(
            ["coursera", "food"],
            ["standard", "weighted"],
            [13],
        )
        self.assertEqual(
            matrix,
            [
                ("coursera", "standard", 13),
                ("coursera", "weighted", 13),
                ("food", "standard", 13),
                ("food", "weighted", 13),
            ],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_run_matrix(
                ["coursera", "coursera"], ["standard"], [13],
            )


if __name__ == "__main__":
    unittest.main()
