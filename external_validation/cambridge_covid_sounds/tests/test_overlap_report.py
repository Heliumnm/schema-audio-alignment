from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from overlap_report import execute  # noqa: E402


def add_submission(root: Path, uid: str, sample: str) -> None:
    folder = root / uid / sample
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("audio_file_cough.wav", "audio_file_breathe.wav", "audio_file_read.wav"):
        (folder / name).write_bytes(b"inventory-only")


class OverlapReportTest(unittest.TestCase):
    def test_identifier_free_cross_task_split_and_hash_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task1, task2, output = root / "task1", root / "task2", root / "output"
            uids = [f"A00000000{index}" for index in range(4)]
            for index, uid in enumerate(uids):
                add_submission(task2, uid, f"sample_{index}")
            add_submission(task1, uids[0], "sample_0")
            add_submission(task1, "B000000009", "other_sample")

            public, private = output / "public", output / "private"
            public.mkdir(parents=True)
            private.mkdir()
            (public / "reconstruction_report.json").write_text(json.dumps({
                "format_version": "cambridge-reconstruction-v2",
                "longitudinal_label_diagnostics": {
                    "uids_with_multiple_raw_covid_statuses": 1,
                    "uids_with_both_strict_endpoint_labels_before_audio_linkage": 0,
                    "strict_conflicting_audio_linked_uids_excluded": 0,
                },
                "exclusions": {"participant_observed_under_both_labels": 0},
            }))
            (public / "data_gate.json").write_text(json.dumps({
                "identity_namespace_version": "cambridge-task2-official-loader-v1",
            }))
            pd.DataFrame({
                "participant_identifier": uids,
                "splits": ["train", "validation", "test", "matched_target"],
            }).to_csv(private / "participant_manifest.csv", index=False)
            pd.DataFrame({
                "participant_identifier": uids,
                "objective_qc_pass": [True] * 4,
                "duplicate_participant_excluded": [False] * 4,
                "pcm_sha256": [f"pcm-{index}" for index in range(4)],
                "raw_sha256": [f"raw-{index}" for index in range(4)],
            }).to_csv(private / "audio_qc_manifest.csv", index=False)

            report = execute(task1, task2, output, expected_task2_uids=4,
                             expected_task2_samples=4)

            self.assertEqual(report["task2"]["unique_uid_count"], 4)
            self.assertEqual(report["task2"]["unique_sample_submission_count"], 4)
            self.assertEqual(report["task2"]["recordings_per_uid"]["histogram"], {"3": 4})
            self.assertEqual(report["task2"]["platform_uid_counts"]["ANDROID"], 4)
            self.assertEqual(report["cross_task"]["task1_intersection_task2_uid_count"], 1)
            self.assertEqual(
                report["cross_task"]["task1_intersection_task2_sample_id_count"], 1)
            self.assertTrue(report["formal_training_permitted_by_overlap_audit"])
            self.assertNotIn(uids[0], (public / "overlap_report.json").read_text())


if __name__ == "__main__":
    unittest.main()
