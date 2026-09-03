from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from prepare_formal_inputs import FIELDS, safe, schema


def test_schema_is_fixed_and_missing_is_explicit():
    row = {field: "NO" for field in FIELDS}
    row["age"] = 37
    row["smoker"] = "[MISSING]"
    text = schema(row)
    assert text.startswith("[AGE=37] [SEX=NO]")
    assert "[SMOKER=[MISSING]]" in text
    assert len(text.split()) == len(FIELDS)


def test_unsafe_value_fails():
    try:
        safe("bad[value]")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe schema value was accepted")


def test_legacy_lowercase_missing_is_canonicalised():
    assert safe("[missing]") == "[MISSING]"
