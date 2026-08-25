from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_gate import execute  # noqa: E402


def write_wave(path: Path, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, count = 16_000, 9_600
    samples = [int(8_000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(count)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * count, *samples))


def config(root: Path, imbalanced_platform: bool = False) -> Path:
    rows = []
    splits = [("train", 40), ("validation", 12), ("test", 20)]
    for split, per_class in splits:
        for label in (0, 1):
            for index in range(per_class):
                pid = f"{split}_{label}_{index:03d}"
                platform = ("WEB" if label else "ANDROID") if imbalanced_platform else \
                    ("WEB" if index % 2 else "ANDROID")
                rows.append({
                    "uid": pid, "label": label, "fold": split,
                    "age": 30 + index % 10, "gender": "female" if index % 2 else "male",
                    "smoker": "never", "cough": 1, "fever": index % 2,
                    "sore_throat": index % 2, "shortness_of_breath": 0,
                    "asthma": 0, "other_respiratory": 0, "platform": platform,
                })
                write_wave(root / "audio" / pid / "session" / f"{pid}_cough.wav",
                           220 + {"train": 0, "validation": 100, "test": 200}[split]
                           + label * 40 + index)
    pd.DataFrame(rows).to_csv(root / "participants.csv", index=False)
    payload = {
        "format_version": "cambridge-external-v1", "dataset": "synthetic-test",
        "output_root": str(root / "out"),
        "inputs": {"participant_csv": str(root / "participants.csv"),
                   "metadata_csv": None, "audio_root": str(root / "audio"),
                   "audio_manifest_csv": None},
        "columns": {"participant_id": "uid", "label": "label", "split": "fold",
                    "canonical": {
                        "age": "age", "sex": "gender", "smoker": "smoker",
                        "cough": "cough", "fever": "fever",
                        "sore_throat": "sore_throat",
                        "shortness_of_breath": "shortness_of_breath",
                        "asthma": "asthma", "other_respiratory": "other_respiratory",
                        "platform": "platform",
                    }},
        "labels": {"positive_values": [1, "1"], "negative_values": [0, "0"]},
        "splits": {"train_values": ["train"], "validation_values": ["validation"],
                   "test_values": ["test"]},
        "field_rules": {
            "age": {"kind": "number", "min": 18, "max": 100, "allow_missing": False},
            "sex": {"kind": "category", "map": {"female": "FEMALE", "male": "MALE"},
                    "allow_missing": False},
            "smoker": {"kind": "category", "identity": True},
            **{field: {"kind": "boolean", "true_values": [1, "1"],
                       "false_values": [0, "0"]} for field in
               ("cough", "fever", "sore_throat", "shortness_of_breath", "asthma",
                "other_respiratory")},
            "platform": {"kind": "category", "identity": True},
        },
        "audio": {"primary_modalities": ["cough"], "min_duration_s": 0.5,
                  "scan_layout": "cambridge_task2",
                  "manifest_columns": {"participant_id": "uid", "path": "path",
                                       "modality": "modality"}},
        "matching": {"min_pairs": 10, "max_abs_smd": 0.12,
                     "max_fine_balance_difference": 0.08,
                     "exact_fields": ["age_decade", "sex", "cough", "fever",
                                      "sore_throat", "shortness_of_breath", "asthma",
                                      "other_respiratory"],
                     "cost_fields": [{"field": "age", "kind": "continuous", "scale": 10},
                                     {"field": "smoker", "kind": "categorical"},
                                     {"field": "platform", "kind": "categorical", "weight": 2}],
                     "smd_fields": ["age", "sex", "smoker", "cough", "fever",
                                    "sore_throat", "shortness_of_breath", "asthma",
                                    "other_respiratory"],
                     "continuous_smd_fields": ["age"],
                     "fine_balance_fields": ["platform", "smoker"]},
        "protocol": {"schema_fields": ["age_decade", "sex", "smoker", "cough", "fever",
                                              "sore_throat", "shortness_of_breath", "asthma",
                                              "other_respiratory"],
                     "cohort_field": "platform", "min_train_n": 60,
                     "min_train_per_class": 30, "min_validation_per_class": 10,
                     "min_test_per_class": 15},
        "models": {},
    }
    path = root / "config.json"
    path.write_text(json.dumps(payload))
    return path


class DataGateTest(unittest.TestCase):
    def test_optional_metadata_table_is_joined_before_canonical_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = config(root)
            participants = pd.read_csv(root / "participants.csv")
            metadata_columns = [column for column in participants.columns
                                if column not in ("label", "fold")]
            participants[metadata_columns].to_csv(root / "metadata.csv", index=False)
            participants[["uid", "label", "fold"]].to_csv(
                root / "participants.csv", index=False)
            payload = json.loads(config_path.read_text())
            payload["inputs"]["metadata_csv"] = str(root / "metadata.csv")
            payload["inputs"]["metadata_participant_id"] = "uid"
            config_path.write_text(json.dumps(payload))

            result = execute(config_path)
            self.assertEqual(result["verdict"], "GO")
            self.assertEqual(result["n_audio_eligible"], 144)

    def test_balanced_official_test_goes_and_public_output_has_no_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = execute(config(root))
            self.assertEqual(result["verdict"], "GO")
            self.assertEqual(result["matching"]["n_final_pairs"], 20)
            public_text = (root / "out" / "public" / "data_gate.json").read_text()
            self.assertNotIn("test_1_000", public_text)
            private = pd.read_csv(root / "out" / "private" / "participant_manifest.csv")
            self.assertEqual(int(private.in_matched_test.sum()), 40)
            self.assertEqual(private.participant_identifier.nunique(), len(private))

    def test_frozen_fine_balance_returns_no_go_without_model_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = execute(config(root, imbalanced_platform=True))
            self.assertEqual(result["verdict"], "NO_GO")
            self.assertFalse(result["representations_generated"])
            self.assertFalse(result["model_outputs_read"])
            self.assertFalse(result["gate_checks"]["fine_balance"])


if __name__ == "__main__":
    unittest.main()
