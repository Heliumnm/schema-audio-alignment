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
from prepare_reconstructed_cohort import build, write_config  # noqa: E402


def write_wave(path: Path, frequency: float = 220.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, count = 16_000, 9_600
    samples = [int(8_000 * math.sin(2 * math.pi * frequency * i / rate)) for i in range(count)]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * count, *samples))


class ReconstructedCohortTest(unittest.TestCase):
    def test_web_folder_is_the_released_subject_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            audio_root = root / "covid19"
            metadata_root.mkdir()
            rows = []
            folders = []
            for label in (0, 1):
                for index in range(3):
                    folder = f"2021-05-{label * 3 + index + 1:02d}_12_00_00_000000"
                    folders.append(folder)
                    rows.append({
                        "Uid": "form-app-users", "Folder Name": folder, "Language": "en",
                        "Covid-Tested": "positiveLast14" if label else "negativeNever",
                        "Age": "20-29", "Sex": "female" if index % 2 else "male",
                        "Smoking": "never", "Symptoms": "drycough",
                        "Medhistory": "none", "Cough filename": "audio_file_cough.wav",
                    })
                    write_wave(audio_root / "form-app-users" / folder /
                               "audio_file_cough.wav", 300 + label * 30 + index)
            pd.DataFrame(rows).to_csv(metadata_root / "web.csv", sep=";", index=True)

            report = build(metadata_root, audio_root, root / "out")
            people = pd.read_csv(root / "out/private/reconstructed_participants.csv")
            audio = pd.read_csv(root / "out/private/reconstructed_audio_manifest.csv")

            self.assertEqual(report["format_version"], "cambridge-reconstruction-v2")
            self.assertEqual(report["n_reconstructed_participants"], 6)
            self.assertEqual(set(people.uid), set(folders))
            self.assertEqual(audio.uid.nunique(), 6)
            self.assertEqual(set(people.platform), {"WEB"})
            self.assertNotIn("form-app-users", set(people.uid))

    def test_session_linkage_strict_labels_conflicts_and_deterministic_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_root = root / "metadata"
            audio_root = root / "covid19"
            metadata_root.mkdir()
            rows = []
            for label in (0, 1):
                for platform in ("android", "ios"):
                    for index in range(10):
                        uid = f"{platform}_{label}_{index:02d}"
                        folder = f"2021-03-{index + 1:02d}_12_00_00_000000"
                        rows.append({
                            "Uid": uid, "Folder Name": folder, "Language": "en",
                            "Covid-Tested": "positiveLast14" if label else "negativeNever",
                            "Age": "20-29", "Sex": "female" if index % 2 else "male",
                            "Smoking": "never", "Symptoms": "drycough, fever",
                            "Medhistory": "none", "Cough filename": "audio_file_cough.wav",
                            "_platform": platform,
                        })
                        write_wave(audio_root / uid / folder / "audio_file_cough.wav",
                                   200 + label * 30 + index)

            # A participant observed under both strict labels must be excluded as a whole.
            conflict_uid = "android_conflict"
            for index, status in enumerate(("negativeNever", "positiveLast14")):
                folder = f"2021-04-0{index + 1}_12_00_00_000000"
                rows.append({
                    "Uid": conflict_uid, "Folder Name": folder, "Language": "en",
                    "Covid-Tested": status, "Age": "30-39", "Sex": "male",
                    "Smoking": "never", "Symptoms": "none", "Medhistory": "none",
                    "Cough filename": "audio_file_cough.wav", "_platform": "android",
                })
                write_wave(audio_root / conflict_uid / folder / "audio_file_cough.wav",
                           500 + index)

            # Non-English and non-strict historical status are data-gate exclusions.
            rows.extend([
                {"Uid": "android_nonenglish", "Folder Name": "s1", "Language": "it",
                 "Covid-Tested": "positiveLast14", "Age": "20-29", "Sex": "male",
                 "Smoking": "never", "Symptoms": "none", "Medhistory": "none",
                 "Cough filename": "audio_file_cough.wav", "_platform": "android"},
                {"Uid": "android_old", "Folder Name": "s1", "Language": "en",
                 "Covid-Tested": "positiveOver14", "Age": "20-29", "Sex": "male",
                 "Smoking": "never", "Symptoms": "none", "Medhistory": "none",
                 "Cough filename": "audio_file_cough.wav", "_platform": "android"},
            ])

            table = pd.DataFrame(rows)
            for platform in ("android", "ios"):
                subset = table[table._platform == platform].drop(columns="_platform")
                subset.to_csv(metadata_root / f"{platform}.csv", sep=";", index=True)

            first = build(metadata_root, audio_root, root / "out1")
            second = build(metadata_root, audio_root, root / "out2")
            people1 = pd.read_csv(root / "out1/private/reconstructed_participants.csv")
            people2 = pd.read_csv(root / "out2/private/reconstructed_participants.csv")
            audio = pd.read_csv(root / "out1/private/reconstructed_audio_manifest.csv")

            self.assertEqual(first["n_reconstructed_participants"], 40)
            self.assertEqual(first["split_counts"], second["split_counts"])
            self.assertTrue(people1.equals(people2))
            self.assertEqual(people1.fold.value_counts().to_dict(),
                             {"train": 28, "test": 8, "validation": 4})
            self.assertEqual(audio.uid.nunique(), 40)
            self.assertNotIn(conflict_uid, set(people1.uid))
            self.assertEqual(first["exclusions"]["participant_observed_under_both_labels"], 1)
            self.assertFalse(first["official_split_reproduced"])

            config_path = root / "out1/config.local.json"
            write_config(audio_root, root / "out1", config_path)
            gate = execute(config_path)
            self.assertEqual(gate["verdict"], "NO_GO")  # frozen 100-pair floor
            self.assertEqual(gate["split_origin"],
                             "reconstructed model-blind 70/10/20 participant split")
            self.assertFalse(gate["model_outputs_read"])

            v2_config_path = root / "out1/config.match_first_v2.json"
            write_config(audio_root, root / "out1", v2_config_path, "match_first_v2")
            v2_config = json.loads(v2_config_path.read_text())
            self.assertEqual(v2_config["format_version"], "cambridge-external-v1")
            self.assertEqual(v2_config["protocol"]["split_strategy"], "match_first_v2")
            self.assertEqual(v2_config["protocol"]["development_split_ratios"],
                             [0.70, 0.15, 0.15])


if __name__ == "__main__":
    unittest.main()
