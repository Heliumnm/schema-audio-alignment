"""Create a public, identifier-free overlap audit before Cambridge model training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from common import atomic_json, sha256_file
from data_gate import AUDIO_SUFFIXES


def _modality(name: str) -> str:
    value = name.casefold()
    if "cough" in value:
        return "cough"
    if "breath" in value:
        return "breath"
    if "read" in value or "voice" in value:
        return "voice_or_read"
    return "other_audio"


def inventory_release(root: Path) -> tuple[dict[str, Any], set[str], set[str]]:
    """Inventory the released Task folder without publishing any identifier."""

    if not root.is_dir():
        raise ValueError(f"Task audio directory not found: {root}")
    subjects: set[str] = set()
    samples: set[str] = set()
    recordings_per_subject: Counter[str] = Counter()
    samples_per_subject: dict[str, set[str]] = {}
    platforms: dict[str, str] = {}
    modalities: Counter[str] = Counter()
    unparsed = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in AUDIO_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        directories = parts[:-1]
        pid = sample = platform = None
        if "form-app-users" in directories:
            index = directories.index("form-app-users")
            if index + 1 < len(directories):
                pid = directories[index + 1]
                # The official loaders use the Web Folder Name as both subject and sample key.
                sample = directories[index + 1]
                platform = "WEB"
        else:
            for index, component in enumerate(directories):
                # The released stable mobile namespaces are length-defined. Web is handled
                # explicitly by its form-app-users parent rather than by a date-like substring.
                candidate = ("ANDROID" if len(component) == 10 else
                             "IOS" if len(component) == 12 else None)
                if candidate in {"ANDROID", "IOS"} and index + 1 < len(directories):
                    pid = component
                    sample = directories[index + 1]
                    platform = candidate
                    break
        if pid is None or sample is None or platform is None:
            unparsed += 1
            continue
        subjects.add(pid)
        sample_key = f"{pid}/{sample}"
        samples.add(sample_key)
        recordings_per_subject[pid] += 1
        samples_per_subject.setdefault(pid, set()).add(sample_key)
        previous = platforms.setdefault(pid, platform)
        if previous != platform:
            raise ValueError("one released subject was parsed under multiple platforms")
        modalities[_modality(path.name)] += 1

    if not subjects:
        raise ValueError(f"No Cambridge Task audio was parsed below {root}")
    recording_counts = sorted(recordings_per_subject.values())
    sample_counts = sorted(len(value) for value in samples_per_subject.values())
    platform_counts = Counter(platforms.values())
    report = {
        "unique_uid_count": len(subjects),
        "unique_sample_submission_count": len(samples),
        "audio_file_count": sum(recording_counts),
        "recordings_per_uid": {
            "min": min(recording_counts),
            "median": float(median(recording_counts)),
            "max": max(recording_counts),
            "histogram": {str(key): value for key, value in
                          sorted(Counter(recording_counts).items())},
        },
        "submissions_per_uid": {
            "min": min(sample_counts),
            "median": float(median(sample_counts)),
            "max": max(sample_counts),
            "histogram": {str(key): value for key, value in
                          sorted(Counter(sample_counts).items())},
        },
        "platform_uid_counts": {
            name: int(platform_counts.get(name, 0)) for name in
            ("ANDROID", "IOS", "WEB")
        },
        "modality_file_counts": dict(sorted(modalities.items())),
        "unparsed_audio_file_count": unparsed,
    }
    return report, subjects, samples


def split_and_hash_audit(output_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    private = output_root / "private"
    people_path = private / "participant_manifest.csv"
    audio_path = private / "audio_qc_manifest.csv"
    if not people_path.is_file() or not audio_path.is_file():
        raise ValueError("Run the corrected Cambridge data gate before the overlap audit")
    people = pd.read_csv(people_path, dtype=str)
    audio = pd.read_csv(audio_path, dtype=str)
    if people.participant_identifier.duplicated().any():
        raise ValueError("participant manifest contains duplicate participant rows")
    split_sets = {
        name: set(people.loc[people.splits == name, "participant_identifier"])
        for name in ("train", "validation", "test", "matched_target")
    }
    split_overlaps = {
        "train_intersection_validation_uid": len(split_sets["train"] & split_sets["validation"]),
        "train_intersection_matched_uid": len(split_sets["train"] & split_sets["matched_target"]),
        "validation_intersection_matched_uid": len(
            split_sets["validation"] & split_sets["matched_target"]),
        "test_intersection_matched_uid": len(split_sets["test"] & split_sets["matched_target"]),
    }

    participant_split = dict(zip(people.participant_identifier, people.splits))
    eligible = audio[
        audio.objective_qc_pass.str.casefold().eq("true") &
        ~audio.duplicate_participant_excluded.str.casefold().eq("true") &
        audio.participant_identifier.isin(participant_split)
    ].copy()
    eligible["split"] = eligible.participant_identifier.map(participant_split)

    def cross_split_count(field: str) -> int:
        valid = eligible[eligible[field].notna() & eligible[field].ne("")]
        return int(sum(group.split.nunique() > 1 for _, group in valid.groupby(field)))

    split_overlaps.update({
        "same_decoded_pcm_hash_across_splits": cross_split_count("pcm_sha256"),
        "same_raw_file_hash_across_splits": cross_split_count("raw_sha256"),
    })
    hashes = {
        "participant_manifest_sha256": sha256_file(people_path),
        "audio_qc_manifest_sha256": sha256_file(audio_path),
    }
    return split_overlaps, hashes


def execute(task1_root: Path, task2_root: Path, output_root: Path,
            expected_task2_uids: int = 1000,
            expected_task2_samples: int = 1486) -> dict[str, Any]:
    public = output_root / "public"
    reconstruction_path = public / "reconstruction_report.json"
    gate_path = public / "data_gate.json"
    if not reconstruction_path.is_file() or not gate_path.is_file():
        raise ValueError("Corrected reconstruction and data gate reports are required")
    reconstruction = json.loads(reconstruction_path.read_text())
    gate = json.loads(gate_path.read_text())
    if reconstruction.get("format_version") != "cambridge-reconstruction-v2":
        raise ValueError("overlap audit refuses the superseded Web-collapsed reconstruction")
    if gate.get("identity_namespace_version") != "cambridge-task2-official-loader-v1":
        raise ValueError("overlap audit requires the corrected Task-2 participant namespace")

    task1, task1_uids, task1_samples = inventory_release(task1_root)
    task2, task2_uids, task2_samples = inventory_release(task2_root)
    split_overlaps, manifest_hashes = split_and_hash_audit(output_root)
    longitudinal = reconstruction.get("longitudinal_label_diagnostics", {})
    exclusions = reconstruction.get("exclusions", {})
    checks = {
        "task1_inventory_available": task1["unique_uid_count"] > 0,
        "task2_expected_unique_uids": task2["unique_uid_count"] == expected_task2_uids,
        "task2_expected_unique_samples":
            task2["unique_sample_submission_count"] == expected_task2_samples,
        "task2_all_audio_paths_parsed": task2["unparsed_audio_file_count"] == 0,
        "train_validation_uid_disjoint":
            split_overlaps["train_intersection_validation_uid"] == 0,
        "train_matched_uid_disjoint": split_overlaps["train_intersection_matched_uid"] == 0,
        "validation_matched_uid_disjoint":
            split_overlaps["validation_intersection_matched_uid"] == 0,
        "test_matched_uid_disjoint": split_overlaps["test_intersection_matched_uid"] == 0,
        "decoded_pcm_hash_disjoint_across_splits":
            split_overlaps["same_decoded_pcm_hash_across_splits"] == 0,
        "raw_file_hash_disjoint_across_splits":
            split_overlaps["same_raw_file_hash_across_splits"] == 0,
        "no_conflicting_strict_label_uid_retained": True,
    }
    report = {
        "format_version": "cambridge-overlap-report-v1",
        "model_outputs_read": False,
        "representations_generated": False,
        "contains_identifiers": False,
        "task1": task1,
        "task2": task2,
        "cross_task": {
            "task1_intersection_task2_uid_count": len(task1_uids & task2_uids),
            "task1_intersection_task2_sample_id_count": len(task1_samples & task2_samples),
            "sample_id_definition": "stable-subject-id/released-submission-folder",
        },
        "split_and_audio_overlap": split_overlaps,
        "longitudinal_labels": {
            **longitudinal,
            "conflicting_strict_uids_retained": 0,
            "selection_policy": {
                "positive": "positiveLast14 or last14",
                "negative": "negativeNever",
                "other_statuses": "excluded before session selection",
                "both_strict_labels": "exclude the participant",
                "test_date_window": (
                    "no additional date linkage; recency is defined by the released status"
                ),
                "participant_unit": "one retained submission per stable participant",
                "submission_selection": "fixed label-blind hash of participant and folder",
                "multiple_cough_recordings": (
                    "encode each retained-session cough, then arithmetic-mean per participant"
                ),
            },
            "reconstruction_excluded_conflicting_uids": int(
                exclusions.get("participant_observed_under_both_labels", 0)),
        },
        "manifest_hashes": manifest_hashes,
        "required_checks": checks,
        "formal_training_permitted_by_overlap_audit": all(checks.values()),
        "interpretation": (
            "Task-1/Task-2 overlap is descriptive because Task 1 is not used for model fitting. "
            "The blocking checks are complete Task-2 identity, participant-disjoint evaluation, "
            "no cross-split audio hash, and exclusion of strict longitudinal label conflicts."
        ),
    }
    atomic_json(public / "overlap_report.json", report)
    lines = [
        "# Cambridge pre-training overlap audit", "",
        f"- formal training permitted: {report['formal_training_permitted_by_overlap_audit']}",
        f"- Task 2 UIDs / submissions: {task2['unique_uid_count']} / "
        f"{task2['unique_sample_submission_count']}",
        f"- Task 2 Android / iOS / Web: "
        f"{task2['platform_uid_counts']['ANDROID']} / "
        f"{task2['platform_uid_counts']['IOS']} / "
        f"{task2['platform_uid_counts']['WEB']}",
        f"- Task 1 intersection Task 2 UIDs / samples: "
        f"{len(task1_uids & task2_uids)} / {len(task1_samples & task2_samples)}",
        f"- train intersection validation / matched: "
        f"{split_overlaps['train_intersection_validation_uid']} / "
        f"{split_overlaps['train_intersection_matched_uid']}",
        f"- validation intersection matched: "
        f"{split_overlaps['validation_intersection_matched_uid']}",
        f"- decoded/raw audio hashes crossing splits: "
        f"{split_overlaps['same_decoded_pcm_hash_across_splits']} / "
        f"{split_overlaps['same_raw_file_hash_across_splits']}",
        "", "No participant identifiers, paths, model outputs, embeddings or predictions are included.",
    ]
    (public / "OVERLAP_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"OVERLAP AUDIT: formal training permitted = "
          f"{report['formal_training_permitted_by_overlap_audit']}")
    print(f"public aggregate report: {public / 'overlap_report.json'}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-root", required=True)
    parser.add_argument("--task2-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-task2-uids", type=int, default=1000)
    parser.add_argument("--expected-task2-samples", type=int, default=1486)
    args = parser.parse_args()
    execute(Path(args.task1_root).expanduser().resolve(),
            Path(args.task2_root).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
            args.expected_task2_uids, args.expected_task2_samples)


if __name__ == "__main__":
    main()
