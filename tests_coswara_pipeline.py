"""Synthetic safety tests for the pinned Coswara streaming extractor."""

import gzip
import io
import json
import math
import os
import struct
import sys
import tarfile
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent / "src"))
import extract_coswara_cough_heavy as extract  # noqa: E402
import audit_coswara_audio as audit  # noqa: E402
import freeze_coswara_external_split as freeze  # noqa: E402


DATE = "20200413"


def gzip_tar(entries):
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as tar:
            for name, payload, kind in entries:
                info = tarfile.TarInfo(name)
                if kind == "file":
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
                elif kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = payload.decode()
                    tar.addfile(info)
                else:
                    raise AssertionError(kind)
    return raw.getvalue()


def split_zip(path, payload):
    cut = len(payload) // 2
    members = [
        f"{extract.ARCHIVE_ROOT}/{DATE}/{DATE}.tar.gz.aa",
        f"{extract.ARCHIVE_ROOT}/{DATE}/{DATE}.tar.gz.ab",
    ]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(members[0], payload[:cut])
        archive.writestr(members[1], payload[cut:])
    return members


def build_completed_date(root, entries):
    archive_path = root / extract.ARCHIVE_NAME
    members = split_zip(archive_path, gzip_tar(entries))
    output = root / "output"
    output.mkdir()
    script_sha256 = extract.digest(Path(extract.__file__).resolve(), "sha256")
    with zipfile.ZipFile(archive_path) as archive:
        extract.extract_date(
            archive,
            archive_path,
            output,
            DATE,
            members,
            script_sha256,
        )
    return output, members, script_sha256


class CoswaraExtractorTest(unittest.TestCase):
    def test_streams_only_target_and_resumes_verified_date(self):
        entries = [
            (f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file"),
            (f"{DATE}/idA/breathing-deep.wav", b"IGNORE", "file"),
            (f"{DATE}/idB/cough-heavy.wav", b"RIFF_B", "file"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                n = extract.extract_date(
                    archive, archive_path, output, DATE, members, "script-hash"
                )
                self.assertEqual(n, 2)
                self.assertEqual(
                    extract.extract_date(
                        archive, archive_path, output, DATE, members, "script-hash"
                    ),
                    2,
                )
            self.assertEqual((output / DATE / "idA" / extract.TARGET_NAME).read_bytes(), b"RIFF_A")
            self.assertEqual((output / DATE / "idB" / extract.TARGET_NAME).read_bytes(), b"RIFF_B")
            self.assertFalse((output / DATE / "idA" / "breathing-deep.wav").exists())
            marker = json.loads((output / DATE / ".complete.json").read_text())
            self.assertEqual(marker["n_cough_heavy"], 2)

    def test_refuses_unsafe_tar_path_without_publishing_date(self):
        entries = [("../idA/cough-heavy.wav", b"BAD", "file")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaisesRegex(ValueError, "unsafe tar path"):
                    extract.extract_date(
                        archive, archive_path, output, DATE, members, "script-hash"
                    )
            self.assertFalse((output / DATE).exists())

    def test_refuses_links_and_duplicate_targets(self):
        cases = [
            [(f"{DATE}/idA/cough-heavy.wav", b"elsewhere", "symlink")],
            [
                (f"{DATE}/idA/cough-heavy.wav", b"A", "file"),
                (f"{DATE}/idA/cough-heavy.wav", b"B", "file"),
            ],
        ]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                archive_path = root / extract.ARCHIVE_NAME
                members = split_zip(archive_path, gzip_tar(entries))
                output = root / "output"
                output.mkdir()
                with zipfile.ZipFile(archive_path) as archive:
                    with self.assertRaises(ValueError):
                        extract.extract_date(
                            archive, archive_path, output, DATE, members, "script-hash"
                        )
                self.assertFalse((output / DATE).exists())

    def test_target_path_is_bound_to_the_current_date(self):
        invalid_names = [
            "idA/cough-heavy.wav",
            f"wrongdate/idA/cough-heavy.wav",
            f"prefix/{DATE}/idA/cough-heavy.wav",
        ]
        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "target path must be"):
                    extract.validate_tar_path(name, DATE)

    def test_completion_rejects_participant_symlink(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                extract.extract_date(
                    archive, archive_path, output, DATE, members, "script-hash"
                )
            participant = output / DATE / "idA"
            outside = root / "outside-idA"
            participant.rename(outside)
            os.symlink(outside, participant)
            self.assertFalse(
                extract.completion_matches(
                    output / DATE, DATE, members, "script-hash"
                )
            )

    def test_completion_rejects_content_and_marker_tampering(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                extract.extract_date(
                    archive, archive_path, output, DATE, members, "script-hash"
                )
            target = output / DATE / "idA" / extract.TARGET_NAME
            target.write_bytes(b"RIFF_B")  # same byte count, different content
            self.assertFalse(
                extract.completion_matches(output / DATE, DATE, members, "script-hash")
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                extract.extract_date(
                    archive, archive_path, output, DATE, members, "script-hash"
                )
            marker_path = output / DATE / ".complete.json"
            marker = json.loads(marker_path.read_text())
            marker["files_sha256"] = "0" * 64
            marker_path.write_text(json.dumps(marker))
            self.assertFalse(
                extract.completion_matches(output / DATE, DATE, members, "script-hash")
            )


class CoswaraAudioAuditTest(unittest.TestCase):
    def test_pcm_waveform_audit_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tone.wav"
            samples = [int(12000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            first = audit.waveform_audit(path)
            second = audit.waveform_audit(path)
            self.assertTrue(first["decode_ok"])
            self.assertEqual(first["decoded_frames"], 16000)
            self.assertEqual(first["header_frames"], 16000)
            self.assertAlmostEqual(first["duration_s"], 1.0)
            self.assertFalse(first["all_zero"])
            self.assertEqual(first["pcm_sha256"], second["pcm_sha256"])
            self.assertEqual(first["raw_sha256"], second["raw_sha256"])

    def test_quality_header_with_leading_space_is_parsed_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "quality.csv"
            path.write_text("FILENAME, QUALITY\nidA_cough-heavy,2\nidB_cough-heavy,0\n")
            self.assertEqual(audit.read_quality(path), {"idA": 2, "idB": 0})

    def test_refuses_tampered_completed_directory(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / extract.ARCHIVE_NAME
            members = split_zip(archive_path, gzip_tar(entries))
            output = root / "output"
            output.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                extract.extract_date(
                    archive, archive_path, output, DATE, members, "script-hash"
                )
                (output / DATE / "idA" / extract.TARGET_NAME).unlink()
                with self.assertRaises(FileExistsError):
                    extract.extract_date(
                        archive, archive_path, output, DATE, members, "script-hash"
                    )

    def test_accepts_only_complete_extractor_v2_provenance(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        with tempfile.TemporaryDirectory() as temp:
            output, _, _ = build_completed_date(Path(temp), entries)
            files = audit.extracted_files(output, allow_partial=True)
            self.assertEqual(
                [(date, participant) for date, participant, _ in files],
                [(DATE, "idA")],
            )

    def test_rejects_every_tampered_extractor_v2_provenance_field(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        tampered_values = {
            "format_version": 1,
            "archive_name": "not-the-pinned-archive.zip",
            "archive_bytes": 1,
            "archive_md5": "bad-md5",
            "archive_root": "wrong-root",
            "zip_members": [
                f"{extract.ARCHIVE_ROOT}/{DATE}/{DATE}.tar.gz.ab",
                f"{extract.ARCHIVE_ROOT}/{DATE}/{DATE}.tar.gz.aa",
            ],
            "script_sha256": "bad-script-hash",
            "participants_sha256": "bad-participant-hash",
            "files_sha256": "bad-files-hash",
        }
        for key, value in tampered_values.items():
            with self.subTest(field=key), tempfile.TemporaryDirectory() as temp:
                output, _, _ = build_completed_date(Path(temp), entries)
                marker = output / DATE / ".complete.json"
                payload = json.loads(marker.read_text())
                payload[key] = value
                marker.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    audit.extracted_files(output, allow_partial=True)

    def test_rejects_symlinks_extra_directories_and_incomplete_date_set(self):
        entries = [(f"{DATE}/idA/cough-heavy.wav", b"RIFF_A", "file")]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, _, _ = build_completed_date(root, entries)
            (output / "unexpected").mkdir()
            with self.assertRaisesRegex(ValueError, "unexpected extraction-root entry"):
                audit.extracted_files(output, allow_partial=True)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, _, _ = build_completed_date(root, entries)
            linked_root = root / "linked-output"
            os.symlink(output, linked_root)
            with self.assertRaisesRegex(ValueError, "not a real directory"):
                audit.extracted_files(linked_root, allow_partial=True)

        with tempfile.TemporaryDirectory() as temp:
            output, _, _ = build_completed_date(Path(temp), entries)
            (output / DATE / "extra-participant").mkdir()
            with self.assertRaisesRegex(ValueError, "completion verification failed"):
                audit.extracted_files(output, allow_partial=True)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, _, _ = build_completed_date(root, entries)
            participant = output / DATE / "idA"
            outside = root / "outside-idA"
            participant.rename(outside)
            os.symlink(outside, participant)
            with self.assertRaisesRegex(ValueError, "completion verification failed"):
                audit.extracted_files(output, allow_partial=True)

        with tempfile.TemporaryDirectory() as temp:
            output, _, _ = build_completed_date(Path(temp), entries)
            with self.assertRaisesRegex(ValueError, "pinned 43-date set"):
                audit.extracted_files(output, allow_partial=False)

    def test_duplicate_flags_are_row_hash_specific(self):
        rows = [
            {
                "participant_id": "idA",
                "relative_path": "d1/idA/cough-heavy.wav",
                "raw_sha256": "raw-shared",
                "pcm_sha256": "pcm-a",
            },
            {
                "participant_id": "idA",
                "relative_path": "d2/idA/cough-heavy.wav",
                "raw_sha256": "raw-a-only",
                "pcm_sha256": "pcm-shared",
            },
            {
                "participant_id": "idB",
                "relative_path": "d1/idB/cough-heavy.wav",
                "raw_sha256": "raw-shared",
                "pcm_sha256": "pcm-b",
            },
            {
                "participant_id": "idC",
                "relative_path": "d1/idC/cough-heavy.wav",
                "raw_sha256": "raw-c",
                "pcm_sha256": "pcm-shared",
            },
        ]
        raw_groups, pcm_groups, counts = audit.annotate_duplicate_rows(rows)
        self.assertEqual(counts["idA"], 2)
        self.assertEqual(
            [row["cross_id_raw_duplicate"] for row in rows],
            [True, False, True, False],
        )
        self.assertEqual(
            [row["cross_id_pcm_duplicate"] for row in rows],
            [False, True, False, True],
        )
        self.assertEqual(
            [member["relative_path"] for member in raw_groups["raw-shared"]],
            ["d1/idA/cough-heavy.wav", "d1/idB/cough-heavy.wav"],
        )
        self.assertIn("pcm-shared", pcm_groups)

    def test_outputs_are_unique_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "manifest.csv"
            audit.atomic_csv(output, [{"value": "first"}], ["value"])
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                audit.atomic_csv(output, [{"value": "second"}], ["value"])
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(list(root.glob(".manifest.csv.*.tmp")), [])

    def test_output_lock_rejects_a_second_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs = (root / "manifest.csv", root / "report.json")
            with audit.output_locks(outputs):
                with self.assertRaises(RuntimeError):
                    with audit.output_locks(outputs):
                        self.fail("second writer unexpectedly acquired the audit lock")


class CoswaraSplitTest(unittest.TestCase):
    def test_hungarian_finds_rectangular_minimum(self):
        cost = [[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]]
        assignment = freeze.hungarian(cost)
        self.assertEqual(len(assignment), 2)
        self.assertEqual(sum(cost[i][j] for i, j in assignment), 3.0)

    def test_exact_matching_and_balance(self):
        def row(participant, label, age):
            base = {
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
                "smoker": "n",
                "asthma": "NO",
                "other_respiratory": "NO",
                "diarrhoea": "NO",
                "loss_of_smell": "NO",
                "country_group": "India",
                "province_group": "Karnataka",
                "halfyear": "2020-H1",
                "vaccination": "n",
                "mask_use": "n",
                "manual_quality": "2",
                "quality_available": "YES",
            }
            return base

        rows = [row("n1", 0, 25), row("n2", 0, 27), row("p1", 1, 25), row("p2", 1, 27)]
        pairs = freeze.match_rows(rows, "test")
        self.assertEqual(len(pairs), 2)
        report = freeze.balance_report(pairs, {r["participant_id"]: r for r in rows})
        self.assertTrue(report["passes"])

    def test_official_checkbox_semantics_and_respiratory_union(self):
        blank = {
            "meta_asthma": "",
            "meta_others_resp": "",
            "meta_cld": "",
            "meta_pneumonia": "",
        }
        self.assertEqual(freeze.symptom(blank["meta_asthma"]), "NO")
        self.assertEqual(
            freeze.any_checkbox(
                blank, ("meta_others_resp", "meta_cld", "meta_pneumonia")
            ),
            "NO",
        )
        chronic_lung = dict(blank, meta_cld="True")
        pneumonia = dict(blank, meta_pneumonia="1")
        for row in (chronic_lung, pneumonia):
            self.assertEqual(
                freeze.any_checkbox(
                    row, ("meta_others_resp", "meta_cld", "meta_pneumonia")
                ),
                "YES",
            )


if __name__ == "__main__":
    unittest.main()
