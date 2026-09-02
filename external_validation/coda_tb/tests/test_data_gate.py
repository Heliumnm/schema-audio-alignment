from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "src/data_gate.py"
SPEC = importlib.util.spec_from_file_location("coda_tb_data_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def participant(identifier: str, label: int, country: str = "UG", sex: str = "Male"):
    return {
        "participant": identifier,
        "label": label,
        "country": country,
        "sex": sex,
        "age": 40.0,
        "height": 170.0,
        "weight": 70.0,
        "reported_cough_dur": 10.0,
        "log_cough_days": 2.3978952728,
        "tb_prior": "No",
        "tb_prior_pul": "No",
        "tb_prior_extrapul": "No",
        "tb_prior_unknown": "No",
        "hemoptysis": "No",
        "heart_rate": 75.0,
        "temperature": 37.0,
        "weight_loss": "No",
        "smoke_lweek": "No",
        "fever": "No",
        "night_sweats": "No",
        "hiv_status": "Negative",
    }


class DataGateTests(unittest.TestCase):
    def test_hungarian_known_optimum(self):
        assignment = gate.hungarian([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        self.assertEqual(sum([[4, 1, 3], [2, 0, 5], [3, 2, 2]][i][j] for i, j in assignment), 5)

    def test_exact_balanced_pairs_pass_at_power_floor(self):
        pairs = []
        for index in range(gate.MIN_MATCHED_PAIRS):
            negative = participant(f"n{index:03d}", 0)
            positive = participant(f"p{index:03d}", 1)
            pairs.append({"negative": negative, "positive": positive})
        report = gate.balance(pairs)
        self.assertTrue(report["passed"])
        self.assertEqual(report["max_abs_smd"], 0.0)
        self.assertEqual(report["max_level_difference"], 0.0)

    def test_country_mismatch_breaks_exact_balance(self):
        pairs = []
        for index in range(gate.MIN_MATCHED_PAIRS):
            negative = participant(f"n{index:03d}", 0, country="UG")
            positive = participant(f"p{index:03d}", 1, country="VN")
            pairs.append({"negative": negative, "positive": positive})
        self.assertFalse(gate.balance(pairs)["passed"])

    def test_outer_split_is_deterministic_and_disjoint(self):
        rows_a = [participant(f"x{index:03d}", index % 2) for index in range(40)]
        rows_b = list(reversed([dict(row) for row in rows_a]))
        gate.deterministic_outer_split(rows_a)
        gate.deterministic_outer_split(rows_b)
        a = {row["participant"]: row["outer_split"] for row in rows_a}
        b = {row["participant"]: row["outer_split"] for row in rows_b}
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {"development", "target_candidate"})

    def test_pairing_never_crosses_country_or_sex(self):
        rows = []
        for country in ("UG", "VN"):
            for sex in ("Male", "Female"):
                for index in range(5):
                    neg = participant(f"n-{country}-{sex}-{index}", 0, country, sex)
                    pos = participant(f"p-{country}-{sex}-{index}", 1, country, sex)
                    neg["outer_split"] = pos["outer_split"] = "target_candidate"
                    rows.extend((neg, pos))
        pairs = gate.initial_pairs(rows)
        self.assertEqual(len(pairs), 20)
        for pair in pairs:
            self.assertEqual(pair["negative"]["country"], pair["positive"]["country"])
            self.assertEqual(pair["negative"]["sex"], pair["positive"]["sex"])

    def test_match_first_v2_freezes_exactly_one_hundred_pairs_before_split(self):
        rows = []
        for index in range(130):
            rows.append(participant(f"n{index:03d}", 0))
            rows.append(participant(f"p{index:03d}", 1))
        before, pairs, removed = gate.assign_splits_and_pairs(
            rows, gate.PROTOCOL_MATCH_FIRST_V2
        )
        self.assertEqual(len(before), 130)
        self.assertEqual(len(pairs), gate.MIN_MATCHED_PAIRS)
        self.assertEqual(len(removed), 30)
        self.assertEqual(sum(row["split"] == "matched_target" for row in rows), 200)
        self.assertEqual(sum(row["outer_split"] == "matched_target" for row in rows), 200)
        self.assertFalse(
            {row["participant"] for row in rows if row["split"] == "matched_target"}
            & {row["participant"] for row in rows if row["split"] == "train"}
        )

    def test_v1_remains_the_default_protocol(self):
        rows = [participant(f"x{index:03d}", index % 2) for index in range(240)]
        before, pairs, _ = gate.assign_splits_and_pairs(
            rows, gate.PROTOCOL_SPLIT_FIRST_V1
        )
        self.assertLess(len(before), 120)
        self.assertLessEqual(len(pairs), len(before))
        self.assertEqual(set(row["outer_split"] for row in rows),
                         {"development", "target_candidate"})


if __name__ == "__main__":
    unittest.main()
