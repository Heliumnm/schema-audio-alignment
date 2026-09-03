"""Build a model-blind Cambridge cohort when the released Task-2 split CSV is absent.

The reconstruction is deliberately conservative. It links each cough to the metadata row from
the same participant and collection folder, keeps only English recent-positive or never-positive
negative sessions, excludes participants observed under both labels, chooses one session per
participant by a fixed label-blind hash, and creates a deterministic participant-level 70/10/20
split. Row-level manifests remain below ``output/private``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common import atomic_json, canonical_string, sha256_file, sha256_text
from data_gate import (AUDIO_SUFFIXES, CAMBRIDGE_FIELD_ALIASES, MISSING, _find_column,
                       _multiselect_state, _normalise_age_group,
                       _normalise_sex, _normalise_smoking, _platform_from_path,
                       read_table)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SALT = "cambridge-reconstructed-participant-split-v1"
SESSION_SALT = "cambridge-reconstructed-index-session-v1"
POSITIVE_VALUES = {"positiveLast14", "last14"}
NEGATIVE_VALUES = {"negativeNever"}


def _column(frame: pd.DataFrame, name: str, required: bool = True) -> str | None:
    aliases = {
        "folder": ("Folder Name", "folder_name", "folder"),
        "label": ("Covid-Tested", "covid_tested", "covid-tested"),
        "language": ("Language", "language"),
        "cough_filename": ("Cough filename", "cough_filename"),
    }
    candidates = aliases[name] if name in aliases else CAMBRIDGE_FIELD_ALIASES[name]
    return _find_column(frame, candidates, required)


def _strict_label(value: Any) -> int | None:
    text = canonical_string(value)
    if text in POSITIVE_VALUES:
        return 1
    if text in NEGATIVE_VALUES:
        return 0
    return None


def _is_english(value: Any) -> bool:
    return canonical_string(value).casefold() in {"en", "english"}


def _session_audio(audio_root: Path, uid: str, folder: str,
                   cough_filename: str, platform: str) -> list[Path]:
    # The official Task-2 loader uses the collection folder as the Web sample key and reads
    # form-app-users/<folder> directly.  In the all-metadata export every Web row instead has
    # the constant Uid "form-app-users".  Treating that constant as a participant would merge
    # every Web submission into one person, so reconstruction must use Folder Name as the
    # only released Web subject namespace.
    if platform.casefold() == "web":
        session_root = audio_root / "form-app-users" / folder
        session_roots = [session_root] if session_root.is_dir() else []
    else:
        participant_root = audio_root / uid
        session_root = participant_root / folder
        session_roots = [session_root] if session_root.is_dir() else []

    found: list[Path] = []
    # Exact collection-folder linkage is mandatory. A flattened Task-2 directory without
    # collection identity is insufficient for a session-level label reconstruction.
    for session_root in session_roots:
        if cough_filename != MISSING:
            named = session_root / Path(cough_filename).name
            if named.is_file() and "cough" in named.name.casefold():
                found.append(named.resolve())
        for path in session_root.glob("*"):
            if (path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES and
                    "cough" in path.name.casefold()):
                found.append(path.resolve())
    return sorted(set(found))


def _single(values: Iterable[Any], normaliser, field: str) -> str:
    observed_values = [normaliser(value) for value in values]
    observed = sorted({value for value in observed_values if value != MISSING})
    if len(observed) > 1:
        raise ValueError(f"conflicting {field}: {observed}")
    return observed[0] if observed else MISSING


def _split(records: pd.DataFrame) -> pd.Series:
    assignments: dict[str, str] = {}
    for _, group in records.groupby(["label", "platform"], dropna=False, sort=True):
        ordered = sorted(group.uid, key=lambda uid: sha256_text(f"{SPLIT_SALT}|{uid}"))
        n = len(ordered)
        n_train = int(0.70 * n)
        n_validation = int(0.10 * n)
        for index, uid in enumerate(ordered):
            assignments[uid] = ("train" if index < n_train else
                                "validation" if index < n_train + n_validation else
                                "test")
    return records.uid.map(assignments)


def build(metadata_root: Path, audio_root: Path, output_root: Path) -> dict[str, Any]:
    if not metadata_root.is_dir():
        raise ValueError(f"metadata directory not found: {metadata_root}")
    if not audio_root.is_dir():
        raise ValueError(f"audio directory not found: {audio_root}")
    private = output_root / "private"
    public = output_root / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)

    frames, hashes = [], []
    exclusions = Counter()
    for path in sorted(metadata_root.glob("**/*.csv")):
        frame = read_table(path)
        try:
            columns = {name: _column(frame, name) for name in
                       ("participant_id", "folder", "label", "language", "age", "sex",
                        "smoker", "medical_history", "symptoms")}
        except ValueError:
            exclusions["metadata_file_without_required_columns"] += 1
            continue
        columns["cough_filename"] = _column(frame, "cough_filename", required=False)
        renamed = {column: name for name, column in columns.items() if column is not None}
        local = frame.rename(columns=renamed)[list(renamed.values())].copy()
        if "cough_filename" not in local:
            local["cough_filename"] = MISSING
        local["platform"] = _platform_from_path(path)
        frames.append(local)
        hashes.append((str(path.relative_to(metadata_root)), sha256_file(path)))
    if not frames:
        raise ValueError("no Cambridge metadata CSV had the required columns")
    metadata = pd.concat(frames, ignore_index=True)
    for field in ("participant_id", "folder", "label", "language", "cough_filename"):
        metadata[field] = metadata[field].map(canonical_string)
    web = metadata.platform.map(canonical_string).str.casefold() == "web"
    metadata.loc[web, "participant_id"] = metadata.loc[web, "folder"]

    exclusions["non_english_session"] = int((~metadata.language.map(_is_english)).sum())
    metadata = metadata[metadata.language.map(_is_english)].copy()
    metadata["y"] = metadata.label.map(_strict_label)
    exclusions["non_strict_covid_status"] = int(metadata.y.isna().sum())
    metadata = metadata[metadata.y.notna()].copy()
    metadata["y"] = metadata.y.astype(int)
    exclusions["missing_uid_or_folder"] = int(((metadata.participant_id == MISSING) |
                                                (metadata.folder == MISSING)).sum())
    metadata = metadata[(metadata.participant_id != MISSING) &
                        (metadata.folder != MISSING)].copy()

    sessions = []
    grouped = metadata.groupby(["participant_id", "folder"], sort=False, dropna=False)
    for (uid, folder), group in grouped:
        if group.y.nunique() != 1:
            exclusions["conflicting_label_within_session"] += 1
            continue
        platform = canonical_string(group.platform.iloc[0])
        audio_files = _session_audio(audio_root, uid, folder,
                                     canonical_string(group.cough_filename.iloc[0]),
                                     platform)
        if not audio_files:
            exclusions["session_without_linked_cough"] += 1
            continue
        sessions.append({"uid": uid, "folder": folder, "label": int(group.y.iloc[0]),
                         "group": group, "audio_files": audio_files,
                         "platform": platform})

    by_uid: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        by_uid.setdefault(session["uid"], []).append(session)
    participant_rows, audio_rows = [], []
    symptom_codes = {
        "cough": {"drycough", "wetcough", "cough"},
        "fever": {"fever"},
        "sore_throat": {"sorethroat"},
        "shortness_of_breath": {"shortbreath", "shortnessofbreath"},
    }
    history_codes = {
        "asthma": {"asthma"},
        "other_respiratory": {"copd", "cystic", "cysticfibrosis", "long",
                              "longtermlungdisease", "lung", "lungdisease",
                              "pulmonary", "pulmonaryfibrosis", "otherrespiratory",
                              "otherrespiratorycondition"},
    }
    for uid, participant_sessions in sorted(by_uid.items()):
        labels = {session["label"] for session in participant_sessions}
        if len(labels) != 1:
            exclusions["participant_observed_under_both_labels"] += 1
            continue
        selected = min(participant_sessions,
                       key=lambda session: sha256_text(
                           f"{SESSION_SALT}|{uid}|{session['folder']}"))
        group = selected["group"]
        try:
            row = {
                "uid": uid, "label": selected["label"],
                "age_band": _single(group.age, _normalise_age_group, "age"),
                "sex": _single(group.sex, _normalise_sex, "sex"),
                "smoker": _single(group.smoker, _normalise_smoking, "smoker"),
                "platform": selected["platform"],
            }
        except ValueError:
            exclusions["invalid_or_conflicting_session_metadata"] += 1
            continue
        for field, codes in symptom_codes.items():
            row[field] = _multiselect_state(group.symptoms, codes)
        for field, codes in history_codes.items():
            row[field] = _multiselect_state(group.medical_history, codes)
        participant_rows.append(row)
        for path in selected["audio_files"]:
            audio_rows.append({"uid": uid, "path": str(path), "modality": "cough"})

    participants = pd.DataFrame(participant_rows)
    if participants.empty:
        raise ValueError("no participant had strict metadata and a linked cough recording")
    participants["fold"] = _split(participants)
    participants = participants.sort_values("uid").reset_index(drop=True)
    audio_manifest = pd.DataFrame(audio_rows)
    audio_manifest = audio_manifest[audio_manifest.uid.isin(set(participants.uid))]
    audio_manifest = audio_manifest.sort_values(["uid", "path"]).reset_index(drop=True)

    participant_path = private / "reconstructed_participants.csv"
    audio_path = private / "reconstructed_audio_manifest.csv"
    participants.to_csv(participant_path, index=False)
    audio_manifest.to_csv(audio_path, index=False)
    split_table = pd.crosstab(participants.fold, participants.label).reindex(
        ["train", "validation", "test"], fill_value=0)
    report = {
        "format_version": "cambridge-reconstruction-v2",
        "model_outputs_read": False,
        "representations_generated": False,
        "cohort_definition": {
            "language": "English only",
            "positive": sorted(POSITIVE_VALUES),
            "negative": sorted(NEGATIVE_VALUES),
            "participant_conflict": "exclude participant if both labels are observed",
            "participant_namespace": (
                "Android/iOS Uid; Web Folder Name, matching the official Task-2 loader"
            ),
            "index_session": "one audio-linked eligible session per participant by fixed hash",
            "split": "participant-level 70/10/20, stratified by label and platform",
            "split_salt_sha256": sha256_text(SPLIT_SALT),
        },
        "n_metadata_rows": int(sum(len(frame) for frame in frames)),
        "n_reconstructed_participants": int(len(participants)),
        "n_reconstructed_audio_files": int(len(audio_manifest)),
        "split_counts": {
            split: {"negative": int(split_table.loc[split].get(0, 0)),
                    "positive": int(split_table.loc[split].get(1, 0))}
            for split in ("train", "validation", "test")
        },
        "exclusions": dict(sorted(exclusions.items())),
        "metadata_bundle_sha256": sha256_text("\n".join(
            f"{name}\t{digest}" for name, digest in hashes)),
        "private_manifest_hashes": {
            "participants": sha256_file(participant_path),
            "audio": sha256_file(audio_path),
        },
        "official_split_reproduced": False,
        "interpretation": ("This is a prespecified reconstruction from session-level metadata, "
                           "not the missing official Task-2 split."),
    }
    atomic_json(public / "reconstruction_report.json", report)
    lines = [
        "# Cambridge reconstructed cohort", "",
        "This cohort does not reproduce the missing official Task-2 split.", "",
        f"- participants: {len(participants)}",
        f"- linked cough files: {len(audio_manifest)}",
        f"- train/validation/test: "
        f"{int((participants.fold == 'train').sum())} / "
        f"{int((participants.fold == 'validation').sum())} / "
        f"{int((participants.fold == 'test').sum())}",
        "- split: participant-level 70/10/20, stratified by label and platform",
        "- no representation, prediction or model score was read",
    ]
    (public / "RECONSTRUCTION_REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def write_config(audio_root: Path, output_root: Path, destination: Path,
                 split_strategy: str = "provided_split_v1") -> None:
    template = json.loads((PACKAGE_ROOT / "config.reconstructed.example.json").read_text())
    template["output_root"] = str(output_root.resolve())
    template["inputs"]["participant_csv"] = str(
        (output_root / "private" / "reconstructed_participants.csv").resolve())
    template["inputs"]["audio_manifest_csv"] = str(
        (output_root / "private" / "reconstructed_audio_manifest.csv").resolve())
    template["inputs"]["audio_root"] = str(audio_root.resolve())
    if split_strategy == "match_first_v2":
        template["protocol"].update({
            "split_strategy": "match_first_v2",
            "identity_namespace_version": "cambridge-task2-official-loader-v1",
            "split_origin": ("model-blind target-first matching; remaining participants split "
                             "70/15/15 by label x platform"),
            "development_split_salt": "cambridge-match-first-v2-development",
            "development_split_ratios": [0.70, 0.15, 0.15],
            "min_validation_n": 75,
            "min_test_n": 75,
            "min_validation_per_class": 20,
            "min_test_per_class": 20,
            "min_train_unique_profiles": 100,
            "max_train_modal_profile_share": 0.10,
        })
    elif split_strategy != "provided_split_v1":
        raise ValueError(f"unknown split strategy: {split_strategy}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write-config", required=True)
    parser.add_argument("--split-strategy", choices=("provided_split_v1", "match_first_v2"),
                        default="provided_split_v1")
    args = parser.parse_args()
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report = build(metadata_root, audio_root, output_root)
    write_config(audio_root, output_root, Path(args.write_config).expanduser().resolve(),
                 args.split_strategy)
    print(f"RECONSTRUCTED: {report['n_reconstructed_participants']} participants")
    print(f"aggregate report: {output_root / 'public' / 'reconstruction_report.json'}")


if __name__ == "__main__":
    main()
