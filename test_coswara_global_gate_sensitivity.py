from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from coswara_global_gate_sensitivity import (  # noqa: E402
    allowed_binary_smd_difference,
    solve_selection,
)


def row(participant: str, label: int, age: int) -> dict[str, object]:
    return {
        "participant_id": participant,
        "label": label,
        "age": age,
        "age_decade": "20",
        "sex": "female",
        "cough": "NO",
        "fever": "NO",
        "fatigue": "NO",
        "sore_throat": "NO",
        "breathing_difficulty": "NO",
        "asthma": "NO",
        "other_respiratory": "NO",
        "smoker": "NO",
        "diarrhoea": "NO",
        "loss_of_smell": "NO",
        "country_group": "India",
        "province_group": "Karnataka",
        "halfyear": "2021-H1",
        "vaccination": "n",
        "mask_use": "n",
        "quality_available": "YES",
        "manual_quality": "2",
    }


def test_allowed_smd_difference_is_exact_at_fixed_n():
    bound = allowed_binary_smd_difference(100, 100)
    assert bound == 4
    assert allowed_binary_smd_difference(1, 2) is None


def test_global_solver_finds_balanced_selection():
    rows = [
        row("n0", 0, 22), row("n1", 0, 24),
        row("p0", 1, 22), row("p1", 1, 24),
    ]
    selected, report = solve_selection(rows, target_pairs=2, time_limit_s=10)
    assert report["success"]
    assert len(selected) == 4
    assert sum(int(value["age"]) for value in selected if value["label"] == 0) == 46
    assert sum(int(value["age"]) for value in selected if value["label"] == 1) == 46
