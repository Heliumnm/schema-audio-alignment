from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import (group_cluster_ci, metadata_matrix, pair_cluster_ci,
                      probe_validation_folds)  # noqa: E402


class EvaluateHelperTest(unittest.TestCase):
    def test_metadata_vocabulary_is_fit_on_train_only(self):
        table = pd.DataFrame({"age_decade": ["20-29", "30-39", "90-99"],
                              "sex": ["F", "M", "X"]})
        matrix, names = metadata_matrix(
            table, ["age_decade", "sex"], np.asarray([True, True, False]))
        self.assertEqual(matrix.shape, (3, 6))
        self.assertNotIn("age_decade=90-99", names)
        self.assertNotIn("sex=X", names)
        self.assertEqual(matrix[2, names.index("age_decade=[UNSEEN_IN_TRAIN]")], 1)
        self.assertEqual(matrix[2, names.index("sex=[UNSEEN_IN_TRAIN]")], 1)

    def test_pair_cluster_bootstrap_keeps_matched_pairs_together(self):
        y = np.asarray([0, 1, 0, 1])
        pair = np.asarray(["a", "a", "b", "b"])
        strong = np.asarray([[0.1, 0.9, 0.2, 0.8], [0.1, 0.9, 0.2, 0.8]])
        weak = np.asarray([[0.4, 0.6, 0.6, 0.4], [0.4, 0.6, 0.6, 0.4]])

        def accuracy(labels, probability):
            return float(np.mean((probability >= 0.5) == labels))

        result = pair_cluster_ci(strong, weak, y, pair, accuracy, 100, 0)
        self.assertEqual(result["n_pairs"], 2)
        self.assertEqual(result["observed"], 0.5)

    def test_stratum_bootstrap_keeps_all_rows_in_group(self):
        y = np.asarray([0, 1, 0, 1, 0, 1])
        group = np.asarray(["a", "a", "a", "a", "b", "b"])
        strong = np.tile(np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]), (2, 1))
        weak = np.tile(np.asarray([0.4, 0.6, 0.6, 0.4, 0.55, 0.45]), (2, 1))

        def accuracy(labels, probability):
            return float(np.mean((probability >= 0.5) == labels))

        result = group_cluster_ci(strong, weak, y, group, accuracy, 100, 0)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result["cluster"], "exact_matching_stratum")
        self.assertGreater(result["observed"], 0)

    def test_probe_folds_follow_probe_target_and_exclude_missing(self):
        target = pd.Series(([False] * 10 + [True] * 10 + [None] * 5), dtype="boolean")
        mask = np.ones(len(target), dtype=bool)
        folds = probe_validation_folds(target, mask)
        self.assertTrue(np.all(folds[target.isna()] == -1))
        for fold in range(5):
            self.assertEqual(
                set(target.iloc[np.where(folds == fold)[0]].astype(bool)),
                {False, True},
            )


if __name__ == "__main__":
    unittest.main()
