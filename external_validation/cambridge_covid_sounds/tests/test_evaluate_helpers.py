from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import metadata_matrix, pair_cluster_ci  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
