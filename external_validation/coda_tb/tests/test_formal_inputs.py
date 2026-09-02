import importlib.util
from pathlib import Path

import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "src" / "prepare_formal_inputs.py"
SPEC = importlib.util.spec_from_file_location("prepare_formal_inputs", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_schema_order_and_numeric_canonicalisation():
    row = {field: "No" for field in MODULE.PROFILE_FIELDS}
    for field in MODULE.NUMERIC_FIELDS:
        row[field] = "30.0000"
    row["sex"] = "Female"
    text = MODULE.schema_text(pd.Series(row))
    assert text.startswith("[AGE=30] [SEX=Female] [HEIGHT=30]")
    assert text.endswith("[NIGHT_SWEATS=No]")
    assert len(text.split(" ")) == len(MODULE.PROFILE_FIELDS)


def test_missing_is_explicit_and_not_no():
    assert MODULE.canonical("fever", float("nan")) == "[MISSING]"
    assert MODULE.canonical("age", None) == "[MISSING]"


def test_source_test_is_not_a_target_alias():
    assert set(MODULE.EXPECTED_COUNTS) == {
        "train", "validation", "source_test", "matched_target"
    }
    assert MODULE.EXPECTED_COUNTS["matched_target"] == {0: 100, 1: 100}
