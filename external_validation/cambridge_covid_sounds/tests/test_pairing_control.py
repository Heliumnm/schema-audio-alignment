import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external_validation" / "cambridge_covid_sounds" / "src"))

from projector import build_stratified_pairing  # noqa: E402
from evaluate import retrieval_summary  # noqa: E402
from train_alignment import ARMS, audit_alignment  # noqa: E402


class WithinLabelSexPairingTest(unittest.TestCase):
    def test_joint_strata_pairing_is_deterministic_bijective_and_no_self(self):
        label = np.asarray([0, 0, 0, 1, 1, 1, 1])
        sex = np.asarray(["F", "F", "F", "M", "M", "M", "M"])
        first = build_stratified_pairing(label, sex, seed=3)
        second = build_stratified_pairing(label, sex, seed=3)
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.array_equal(np.sort(first), np.arange(len(first))))
        self.assertFalse(np.any(first == np.arange(len(first))))
        self.assertTrue(np.all(label[first] == label))
        self.assertTrue(np.all(sex[first] == sex))

    def test_singleton_joint_stratum_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "singleton joint stratum"):
            build_stratified_pairing(
                np.asarray([0, 0, 1]), np.asarray(["F", "F", "M"]), seed=0)

    def test_retrieval_reports_locked_primary_and_diagnostic_contrasts(self):
        with tempfile.TemporaryDirectory() as directory:
            text_file = Path(directory) / "text.npz"
            bank = np.eye(4, dtype=np.float32)
            np.savez(text_file, text_id=np.arange(4), unique_embeddings=bank)
            table = pd.DataFrame({"splits": ["validation"] * 4})
            correct = np.repeat(bank[None], 2, axis=0)
            reversed_bank = np.repeat(bank[::-1][None], 2, axis=0)
            representations = {
                "correct": correct,
                "within_label": reversed_bank,
                "within_label_sex": reversed_bank,
                "global": reversed_bank,
            }
            result = retrieval_summary(
                representations, table, text_file, seeds=[0, 1], boot=20)
            self.assertIn("correct_minus_within_label_sex", result)
            self.assertIn("within_label_sex_minus_within", result)
            self.assertGreater(result["correct_minus_within_label_sex"]["observed"], 0)

    def test_four_arm_audit_uses_one_common_reduced_training_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cohort = root / "cohort.csv"
            table = pd.DataFrame({
                "participant_identifier": [f"p{i}" for i in range(5)],
                "splits": ["train"] * 5,
                "y": [0, 0, 1, 1, 1],
                "sex": ["F", "F", "M", "M", "OTHER"],
            })
            table.to_csv(cohort, index=False)
            pairing = np.asarray([1, 0, 3, 2])
            identity = np.arange(4)
            runs = {}
            for arm in ARMS:
                chosen = identity if arm == "correct" else pairing
                np.savez(root / f"repr_{arm}_seed0.npz",
                         participants=table.participant_identifier.to_numpy(), pairing=chosen)
                runs[f"{arm}_seed0"] = {"init_hash": "same", "batch_order_hash": "same"}
            (root / "manifest.json").write_text(json.dumps({
                "runs": runs, "seeds": [0], "epochs": 1, "n_train": 4,
            }))
            audit_alignment(root, cohort, [0], 1)


if __name__ == "__main__":
    unittest.main()
