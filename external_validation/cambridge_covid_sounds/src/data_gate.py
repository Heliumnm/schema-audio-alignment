"""Data-only GO/NO-GO gate for Cambridge COVID-19 Sounds.

This program never imports torch or opens a representation/prediction file. It standardises
the restricted release, audits cough waveforms, freezes a participant-level split, and
constructs a deterministic matched subset inside the frozen test population.

Private row-level manifests are written below ``output/private``. Only aggregate JSON and
Markdown are written below ``output/public``. The latter is the only directory intended to
be returned by a controlled-access collaborator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from common import (atomic_json, canonical_string, load_config, output_paths,
                    public_config, resolve_path, safe_identifier, sha256_file,
                    sha256_text)


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"}
MISSING = "[MISSING]"


CAMBRIDGE_FIELD_ALIASES = {
    "participant_id": ("Uid", "uid", "UID", "participant_id"),
    "age": ("Age", "age"),
    "sex": ("Sex", "sex", "gender", "Gender"),
    "smoker": ("Smoking", "smoking", "smoker", "Smoker"),
    "medical_history": ("Medhistory", "medhistory", "medical_history"),
    "symptoms": ("Symptoms", "symptoms"),
}


def _find_column(table: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in table.columns}
    for alias in aliases:
        if str(alias).strip().casefold() in lookup:
            return lookup[str(alias).strip().casefold()]
    if required:
        raise ValueError(f"none of the expected columns {list(aliases)!r} were found")
    return None


def _normalise_code(value: Any) -> str:
    return re.sub(r"[^a-z0-9+]", "", canonical_string(value).casefold())


def _normalise_age_group(value: Any) -> str:
    code = _normalise_code(value)
    mapping = {
        "018": "00-19", "019": "00-19", "1619": "00-19", "1819": "00-19",
        "unter20": "00-19",
        "2029": "20-29", "3029": "30-39", "3039": "30-39",
        "4049": "40-49", "5059": "50-59", "6069": "60-69",
        "7079": "70-79", "8089": "80-89", "90": "90+", "90+": "90+",
    }
    if code in {"", "missing", "pnts", "ptns", "prefernottosay", "nan", "none"}:
        return MISSING
    if code in mapping:
        return mapping[code]
    # Some exports contain exact numeric ages even though the released data dictionary
    # specifies bands. Convert them deterministically without pretending the band is exact.
    try:
        number = int(float(canonical_string(value)))
    except ValueError as exc:
        raise ValueError(f"unmapped Cambridge age band {value!r}") from exc
    if not 0 <= number <= 120:
        raise ValueError(f"Cambridge age {number} outside [0, 120]")
    if number < 20:
        return "00-19"
    if number >= 90:
        return "90+"
    return f"{10 * (number // 10):02d}-{10 * (number // 10) + 9:02d}"


def _normalise_sex(value: Any) -> str:
    code = _normalise_code(value)
    mapping = {"female": "FEMALE", "f": "FEMALE", "male": "MALE", "m": "MALE",
               "other": "OTHER"}
    if code in {"", "missing", "pnts", "ptns", "prefernottosay", "nan", "none"}:
        return MISSING
    if code not in mapping:
        raise ValueError(f"unmapped Cambridge sex value {value!r}")
    return mapping[code]


def _normalise_smoking(value: Any) -> str:
    code = _normalise_code(value)
    mapping = {
        "ex": "EX_SMOKER", "exsmoker": "EX_SMOKER", "never": "NEVER",
        "ltonce": "LESS_THAN_ONCE", "1to10": "1_TO_10", "11to20": "11_TO_20",
        "21+": "21_PLUS", "21plus": "21_PLUS", "ecig": "E_CIGARETTE",
    }
    if code in {"", "missing", "pnts", "ptns", "prefernottosay", "nan", "none"}:
        return MISSING
    if code not in mapping:
        raise ValueError(f"unmapped Cambridge smoking value {value!r}")
    return mapping[code]


def _codes(value: Any) -> set[str]:
    text = canonical_string(value)
    if text == MISSING:
        return set()
    localised_keys = re.findall(r"key\s*:\s*[\"']([^\"']+)[\"']", text,
                                flags=re.IGNORECASE)
    if localised_keys:
        return {_normalise_code(key) for key in localised_keys if _normalise_code(key)}
    return {_normalise_code(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9_+ -]*", text)
            if _normalise_code(token)}


def _multiselect_state(values: Iterable[Any], positive_codes: set[str]) -> str:
    observed: set[str] = set()
    for value in values:
        observed.update(_codes(value))
    if observed & positive_codes:
        return "YES"
    prefer = {"pnts", "ptns", "prefernottosay", "localizedstringkey"}
    informative = {code for code in observed if code not in prefer}
    return "NO" if informative else MISSING


def _platform_from_path(path: Path) -> str:
    text = str(path).casefold()
    if "android" in text:
        return "ANDROID"
    if "ios" in text:
        return "IOS"
    if "web" in text:
        return "WEB"
    return MISSING


def _platform_from_uid(uid: str) -> str:
    # This follows the official Task-2 loader and the released data dictionary.
    if "202" in uid:
        return "WEB"
    if len(uid) == 10:
        return "ANDROID"
    if len(uid) == 12:
        return "IOS"
    return MISSING


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the Cambridge release without requiring a manual Excel-to-CSV conversion."""

    if path.suffix.casefold() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, nrows=nrows)
    if path.suffix.casefold() == ".csv":
        # The official Task-2 table is comma-separated, whereas the three released
        # android/ios/web metadata files are semicolon-separated and start with an empty
        # exported-index column. Detect this from the header rather than requiring the
        # collaborator to rewrite controlled-access data.
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            header = handle.readline()
        separator = ";" if header.count(";") > header.count(",") else ","
        return pd.read_csv(path, nrows=nrows, low_memory=False, sep=separator,
                           encoding="utf-8-sig")
    raise ValueError(f"unsupported table format {path.suffix!r}; use CSV, XLSX or XLSM")


def _single_static_value(values: Iterable[Any], normaliser, field: str) -> Any:
    normalised = []
    for value in values:
        candidate = normaliser(value)
        if candidate != MISSING:
            normalised.append(candidate)
    unique = sorted(set(normalised))
    if len(unique) > 1:
        raise ValueError(f"conflicting static Cambridge field {field}: {unique}")
    return unique[0] if unique else MISSING


def _metadata_files(adapter: dict[str, Any], config_path: Path) -> tuple[Path, list[Path]]:
    root = resolve_path(config_path, adapter.get("metadata_root"))
    if root is None or not root.is_dir():
        raise ValueError("source_adapter.metadata_root must point to the three-platform metadata directory")
    pattern = str(adapter.get("metadata_glob", "**/*.csv"))
    files = sorted(path for path in root.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"no Cambridge metadata files matched {pattern!r} below metadata_root")
    return root, files


def adapt_cambridge_task2_raw(source: pd.DataFrame, config: dict[str, Any],
                              config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the official uid/label/fold file to raw three-platform metadata.

    Static demographics must be internally consistent. Daily symptom and medical-history
    multi-select fields are collapsed with a prespecified participant-level ``ever`` rule:
    any positive code gives YES; observed non-target responses give NO; only missing/prefer-
    not-to-say gives [MISSING]. This adapter never reads audio or model output.
    """

    adapter = config["inputs"].get("source_adapter", {})
    if adapter.get("name") != "cambridge_task2_raw":
        return source, {}
    source = source.copy()
    source_id = _find_column(source, CAMBRIDGE_FIELD_ALIASES["participant_id"])
    configured_id = config["columns"]["participant_id"]
    if source_id != configured_id:
        source = source.rename(columns={source_id: configured_id})
    source[configured_id] = source[configured_id].map(canonical_string)

    metadata_root, files = _metadata_files(adapter, config_path)
    frames, skipped = [], Counter()
    hashes = []
    for path in files:
        frame = read_table(path)
        uid_column = _find_column(frame, CAMBRIDGE_FIELD_ALIASES["participant_id"], required=False)
        if uid_column is None:
            skipped["no_participant_id_column"] += 1
            continue
        frame = frame.copy()
        frame["__cam_uid"] = frame[uid_column].map(canonical_string)
        frame["__cam_source_platform"] = _platform_from_path(path)
        renamed = {}
        for field, aliases in CAMBRIDGE_FIELD_ALIASES.items():
            if field == "participant_id":
                continue
            column = _find_column(frame, aliases, required=False)
            if column is not None:
                renamed[column] = f"__raw_{field}"
        frame = frame.rename(columns=renamed)
        for field in ("age", "sex", "smoker", "medical_history", "symptoms"):
            column = f"__raw_{field}"
            if column not in frame:
                frame[column] = None
        frames.append(frame[["__cam_uid", "__cam_source_platform", "__raw_age", "__raw_sex",
                             "__raw_smoker", "__raw_medical_history", "__raw_symptoms"]])
        hashes.append((str(path.relative_to(metadata_root)), sha256_file(path)))
    if not frames:
        raise ValueError("none of the metadata files contained a Cambridge Uid column")
    metadata = pd.concat(frames, ignore_index=True)
    metadata = metadata[metadata.__cam_uid != MISSING]

    symptom_codes = {
        "cough": {"drycough", "wetcough", "cough"},
        "fever": {"fever"},
        "sore_throat": {"sorethroat"},
        "shortness_of_breath": {"shortbreath", "shortnessofbreath"},
    }
    history_codes = {
        "asthma": {"asthma"},
        "other_respiratory": {
            "copd", "cystic", "cysticfibrosis", "long", "longtermlungdisease",
            "lung", "lungdisease", "pulmonary", "pulmonaryfibrosis",
            "otherrespiratory", "otherrespiratorycondition",
        },
    }
    rows, exclusions = [], Counter()
    for uid, group in metadata.groupby("__cam_uid", sort=False):
        try:
            age = _single_static_value(group.__raw_age, _normalise_age_group, "age")
            sex = _single_static_value(group.__raw_sex, _normalise_sex, "sex")
            smoker = _single_static_value(group.__raw_smoker, _normalise_smoking, "smoking")
            platforms = {value for value in group.__cam_source_platform if value != MISSING}
            if len(platforms) > 1:
                raise ValueError("participant occurs in multiple platform metadata files")
        except ValueError:
            exclusions["conflicting_or_invalid_static_metadata"] += 1
            continue
        platform = next(iter(platforms), _platform_from_uid(uid))
        record = {
            "__cam_uid": uid, "__cam_age_band": age, "__cam_sex": sex,
            "__cam_smoker": smoker, "__cam_platform": platform,
        }
        for field, codes in symptom_codes.items():
            record[f"__cam_{field}"] = _multiselect_state(group.__raw_symptoms, codes)
        for field, codes in history_codes.items():
            record[f"__cam_{field}"] = _multiselect_state(group.__raw_medical_history, codes)
        rows.append(record)
    canonical = pd.DataFrame(rows)
    if canonical.empty:
        raise ValueError("no valid participant metadata remained after Cambridge adaptation")
    before = int(source[configured_id].nunique())
    source = source.merge(canonical, left_on=configured_id, right_on="__cam_uid",
                          how="inner", validate="many_to_one")
    after = int(source[configured_id].nunique())
    digest_payload = "\n".join(f"{name}\t{digest}" for name, digest in hashes)
    report = {
        "name": "cambridge_task2_raw", "n_metadata_files": len(hashes),
        "n_metadata_rows": int(len(metadata)), "n_task2_participants_before_join": before,
        "n_task2_participants_after_join": after,
        "n_task2_participants_without_valid_metadata": before - after,
        "metadata_bundle_sha256": sha256_text(digest_payload),
        "metadata_files_skipped": dict(skipped), "adapter_exclusions": dict(exclusions),
        "aggregation_rule": "static-consistent; multiselect-ever-positive",
    }
    return source, report


def _normalised_lookup(values: Iterable[Any]) -> dict[str, str]:
    return {str(value).strip().casefold(): str(value) for value in values}


def map_label(value: Any, config: dict[str, Any]) -> int | None:
    text = canonical_string(value).casefold()
    positive = {str(v).strip().casefold() for v in config["positive_values"]}
    negative = {str(v).strip().casefold() for v in config["negative_values"]}
    if text in positive:
        return 1
    if text in negative:
        return 0
    return None


def map_split(value: Any, config: dict[str, Any]) -> str | None:
    text = canonical_string(value).casefold()
    for canonical in ("train", "validation", "test"):
        allowed = {str(v).strip().casefold() for v in config[f"{canonical}_values"]}
        if text in allowed:
            return canonical
    return None


def transform(value: Any, rule: dict[str, Any], field: str) -> Any:
    if canonical_string(value) == MISSING:
        if not rule.get("allow_missing", True):
            raise ValueError(f"{field}: missing value is forbidden")
        return MISSING
    kind = rule["kind"]
    if kind == "number":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field}: non-finite numeric value")
        lo, hi = rule.get("min"), rule.get("max")
        if lo is not None and number < float(lo) or hi is not None and number > float(hi):
            raise ValueError(f"{field}: value {number} outside [{lo}, {hi}]")
        return number
    if kind == "boolean":
        text = canonical_string(value).casefold()
        yes = {str(v).strip().casefold() for v in rule["true_values"]}
        no = {str(v).strip().casefold() for v in rule["false_values"]}
        if text in yes:
            return "YES"
        if text in no:
            return "NO"
        raise ValueError(f"{field}: unmapped boolean value {value!r}")
    if kind == "category":
        text = canonical_string(value)
        lookup = {str(k).strip().casefold(): str(v) for k, v in rule.get("map", {}).items()}
        if text.casefold() in lookup:
            return lookup[text.casefold()]
        if rule.get("identity", False):
            return text
        raise ValueError(f"{field}: unmapped category {value!r}")
    raise ValueError(f"{field}: unsupported rule kind {kind!r}")


def age_band(age: Any) -> str:
    if age == MISSING:
        return MISSING
    value = int(float(age))
    return f"{10 * (value // 10):02d}-{10 * (value // 10) + 9:02d}"


def render_schema(row: dict[str, Any], config: dict[str, Any]) -> str:
    fields = config["protocol"]["schema_fields"]
    parts = []
    for field in fields:
        # Derived, prespecified fields (currently age_decade) need not have a raw-value rule.
        # This keeps exact age out of the text while retaining raw age for matching/probing.
        tag = config["field_rules"].get(field, {}).get("tag", field.upper())
        value = row[field]
        if isinstance(value, float):
            value = f"{value:.3g}"
        parts.append(f"[{tag}={value}]")
    return " ".join(parts)


def load_rows(config: dict[str, Any], config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = config["inputs"]
    participant_path = resolve_path(config_path, inputs["participant_csv"])
    assert participant_path is not None and participant_path.is_file()
    source = read_table(participant_path)
    source, adapter_report = adapt_cambridge_task2_raw(source, config, config_path)
    columns = config["columns"]
    primary_required = {columns["participant_id"], columns["label"], columns["split"]}
    primary_missing = sorted(primary_required - set(source.columns))
    if primary_missing:
        raise ValueError(
            f"participant CSV is missing configured columns {primary_missing}; available columns "
            f"are written to the column inventory")

    extra_path = resolve_path(config_path, inputs.get("metadata_csv"))
    if adapter_report and extra_path is not None:
        raise ValueError("use source_adapter.metadata_root or metadata_csv, not both")
    if extra_path is not None:
        extra = read_table(extra_path)
        extra_id = inputs.get("metadata_participant_id", columns["participant_id"])
        if extra_id not in extra.columns:
            raise ValueError(f"metadata CSV lacks participant key {extra_id!r}")
        extra = extra.rename(columns={extra_id: "__extra_pid"})
        if extra["__extra_pid"].duplicated().any():
            raise ValueError("metadata CSV contains repeated participant IDs")
        source = source.merge(
            extra, left_on=columns["participant_id"], right_on="__extra_pid",
            how="left", validate="many_to_one", suffixes=("", "__extra"))

    # Canonical metadata may live in the optional DTA metadata table, so validate it only
    # after the participant and metadata tables have been joined.
    required_merged = primary_required | set(columns["canonical"].values())
    missing = sorted(required_merged - set(source.columns))
    if missing:
        raise ValueError(
            f"merged participant/metadata table is missing configured columns {missing}; "
            f"available columns are written to the column inventory")

    inventory = {
        "participant_csv_sha256": sha256_file(participant_path),
        "metadata_csv_sha256": (adapter_report.get("metadata_bundle_sha256") if adapter_report
                                else sha256_file(extra_path) if extra_path else None),
        "n_input_rows": int(len(source)),
        "columns": sorted(map(str, source.columns)),
        "source_adapter": adapter_report or None,
    }
    records, exclusions = [], Counter()
    field_rules = config["field_rules"]
    for raw_pid, group in source.groupby(columns["participant_id"], sort=False, dropna=False):
        try:
            pid = safe_identifier(raw_pid)
        except ValueError:
            exclusions["unsafe_participant_id"] += 1
            continue
        record: dict[str, Any] = {"participant_identifier": pid}
        conflict = False
        for canonical, raw_column in {
            "__label": columns["label"], "__split": columns["split"],
            **columns["canonical"],
        }.items():
            values = [v for v in group[raw_column].tolist() if canonical_string(v) != MISSING]
            unique = {canonical_string(v) for v in values}
            if len(unique) > 1:
                exclusions[f"conflicting_{canonical}"] += 1
                conflict = True
                break
            record[canonical] = values[0] if values else None
        if conflict:
            continue
        label = map_label(record.pop("__label"), config["labels"])
        split = map_split(record.pop("__split"), config["splits"])
        if label is None:
            exclusions["nonbinary_or_unknown_label"] += 1
            continue
        if split is None:
            exclusions["unknown_split"] += 1
            continue
        record["y"], record["splits"] = label, split
        failed = False
        for field, rule in field_rules.items():
            try:
                record[field] = transform(record.get(field), rule, field)
            except ValueError:
                exclusions[f"invalid_{field}"] += 1
                failed = True
                break
        if failed:
            continue
        if "age" in record:
            record["age_decade"] = age_band(record.get("age", MISSING))
        records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("no participants remain after label/split/metadata standardisation")
    if table.participant_identifier.duplicated().any():
        raise AssertionError("participant standardisation did not produce one row per patient")
    table["text"] = [render_schema(row, config) for row in table.to_dict("records")]
    inventory["standardisation_exclusions"] = dict(exclusions)
    inventory["n_standardised_participants"] = int(len(table))
    return table, inventory


def scan_audio(config: dict[str, Any], config_path: Path,
               allowed_participants: set[str] | None = None) -> pd.DataFrame:
    inputs, audio = config["inputs"], config["audio"]
    audio_root = resolve_path(config_path, inputs["audio_root"])
    assert audio_root is not None and audio_root.is_dir()
    manifest_path = resolve_path(config_path, inputs.get("audio_manifest_csv"))
    rows = []
    if manifest_path is not None:
        manifest = read_table(manifest_path)
        mapping = audio["manifest_columns"]
        for column in (mapping["participant_id"], mapping["path"]):
            if column not in manifest.columns:
                raise ValueError(f"audio manifest lacks configured column {column!r}")
        modality_column = mapping.get("modality")
        for row in manifest.itertuples(index=False):
            values = row._asdict()
            modality = canonical_string(values.get(modality_column)) if modality_column else "cough"
            if modality.casefold() not in {str(v).casefold() for v in audio["primary_modalities"]}:
                continue
            pid = safe_identifier(values[mapping["participant_id"]])
            if allowed_participants is not None and pid not in allowed_participants:
                continue
            path = Path(str(values[mapping["path"]]))
            path = path if path.is_absolute() else audio_root / path
            rows.append({"participant_identifier": pid,
                         "audio_path": str(path.resolve()), "modality": modality})
    else:
        if audio.get("scan_layout") != "cambridge_task2":
            raise ValueError("without audio_manifest_csv, audio.scan_layout must be cambridge_task2")
        # The DTA/full release is covid19/<participant>/<collection-time>/<three wavs>.
        # Web recordings may instead live under form-app-users/<participant>. Enumerate only
        # the Task-2 participant directories so a 1,000-person audit does not decode the full
        # restricted corpus by accident.
        participant_roots: list[tuple[str, Path]] = []
        allowed = allowed_participants
        for child in sorted(audio_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "form-app-users":
                for web_child in sorted(child.iterdir()):
                    if web_child.is_dir() and (allowed is None or web_child.name in allowed):
                        participant_roots.append((safe_identifier(web_child.name), web_child))
            elif allowed is None or child.name in allowed:
                participant_roots.append((safe_identifier(child.name), child))
        for pid, participant_root in participant_roots:
            for path in sorted(participant_root.rglob("*")):
                if (path.is_file() and path.suffix.casefold() in AUDIO_SUFFIXES and
                        not path.name.startswith("._") and
                        "cough" in path.name.casefold()):
                    rows.append({"participant_identifier": pid,
                                 "audio_path": str(path.resolve()), "modality": "cough"})
    if not rows:
        raise ValueError("no primary cough audio files were discovered")
    return pd.DataFrame(rows).drop_duplicates(["participant_identifier", "audio_path"])


def audit_waveform(path: Path, min_duration_s: float) -> dict[str, Any]:
    try:
        try:
            import soundfile as sf

            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except ModuleNotFoundError:
            # A dependency-free fallback keeps the data gate/test usable for ordinary
            # PCM WAV releases. Formal environments install soundfile for other codecs.
            import wave

            with wave.open(str(path), "rb") as handle:
                channels, width = handle.getnchannels(), handle.getsampwidth()
                sample_rate, frames = handle.getframerate(), handle.getnframes()
                raw = handle.readframes(frames)
            dtype = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}.get(width)
            if dtype is None:
                raise ValueError(f"unsupported PCM sample width {width}")
            decoded = np.frombuffer(raw, dtype=dtype)
            if width == 1:
                decoded = decoded.astype(np.float32) - 128.0
                scale = 128.0
            else:
                decoded = decoded.astype(np.float32)
                scale = float(2 ** (8 * width - 1))
            data = (decoded / scale).reshape(-1, channels)
        mono = data.mean(axis=1)
        finite = bool(np.isfinite(mono).all())
        duration = len(mono) / float(sample_rate) if sample_rate else 0.0
        nonzero = bool(np.any(mono != 0))
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) if len(mono) else 0.0
        clip_fraction = float(np.mean(np.abs(mono) >= 0.999)) if len(mono) else 0.0
        canonical = np.ascontiguousarray(mono, dtype=np.float32)
        pcm_hash = hashlib.sha256(
            f"{sample_rate}|{len(canonical)}|".encode() + canonical.tobytes()).hexdigest()
        passed = finite and nonzero and duration >= min_duration_s
        error = None
    except Exception as exc:  # row-level diagnostic; the type is public, the path is private
        sample_rate, duration, rms, clip_fraction, pcm_hash = 0, 0.0, 0.0, 0.0, ""
        finite, nonzero, passed, error = False, False, False, type(exc).__name__
    return {
        "sample_rate": int(sample_rate), "duration_s": float(duration),
        "finite": finite, "nonzero": nonzero, "rms": rms,
        "clip_fraction": clip_fraction, "pcm_sha256": pcm_hash,
        "raw_sha256": sha256_file(path) if path.is_file() else "",
        "objective_qc_pass": passed, "error_type": error,
    }


def audit_audio(audio: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    for i, row in enumerate(audio.itertuples(index=False)):
        path = Path(row.audio_path)
        result = audit_waveform(path, float(config["audio"]["min_duration_s"]))
        records.append({**row._asdict(), **result})
        if (i + 1) % 1000 == 0:
            print(f"audio QC {i+1}/{len(audio)}", flush=True)
    audited = pd.DataFrame(records)
    passing = audited[audited.objective_qc_pass].copy()

    # Keep one participant from every cross-ID identical-PCM group using a label-blind hash.
    duplicate_excluded: set[str] = set()
    for _, group in passing.groupby("pcm_sha256"):
        participants = sorted(group.participant_identifier.unique())
        if len(participants) <= 1:
            continue
        keep = min(participants, key=lambda pid: sha256_text(f"cambridge-dedup-v1|{pid}"))
        duplicate_excluded.update(pid for pid in participants if pid != keep)
    audited["duplicate_participant_excluded"] = audited.participant_identifier.isin(
        duplicate_excluded)
    passing = audited[audited.objective_qc_pass & ~audited.duplicate_participant_excluded]
    report = {
        "n_discovered_files": int(len(audited)),
        "n_qc_pass_files": int(audited.objective_qc_pass.sum()),
        "n_qc_fail_files": int((~audited.objective_qc_pass).sum()),
        "n_participants_with_passing_audio": int(passing.participant_identifier.nunique()),
        "n_cross_id_duplicate_participants_excluded": len(duplicate_excluded),
        "error_types": audited.error_type.dropna().value_counts().to_dict(),
        "duration_s": {
            "min": float(passing.duration_s.min()) if len(passing) else None,
            "median": float(passing.duration_s.median()) if len(passing) else None,
            "max": float(passing.duration_s.max()) if len(passing) else None,
        },
    }
    return audited, report


def smd_continuous(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = math.sqrt((float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))) / 2)
    difference = float(np.mean(a) - np.mean(b))
    if pooled == 0:
        return 0.0 if difference == 0 else math.inf
    return difference / pooled


def smd_binary(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = float(np.mean(a)), float(np.mean(b))
    scale = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    if scale == 0:
        return 0.0 if pa == pb else math.inf
    return (pa - pb) / scale


def balance_report(pairs: pd.DataFrame, people: pd.DataFrame,
                   config: dict[str, Any]) -> dict[str, Any]:
    if pairs.empty:
        return {"n_pairs": 0, "max_abs_smd": math.inf,
                "max_fine_difference": math.inf, "fields": {}}
    indexed = people.set_index("participant_identifier")
    negative = indexed.loc[pairs.negative_id].reset_index()
    positive = indexed.loc[pairs.positive_id].reset_index()
    fields: dict[str, Any] = {}
    max_smd, max_fine = 0.0, 0.0
    continuous = set(config["matching"].get("continuous_smd_fields", []))
    for field in config["matching"]["smd_fields"]:
        if field in continuous:
            observed = (negative[field] != MISSING) & (positive[field] != MISSING)
            value = smd_continuous(negative.loc[observed, field], positive.loc[observed, field]) \
                if observed.any() else math.inf
            fields[field] = {"kind": "continuous", "smd": float(value),
                             "n_complete_pairs": int(observed.sum())}
            max_smd = max(max_smd, abs(value))
        else:
            levels = sorted(set(negative[field].astype(str)) | set(positive[field].astype(str)))
            level_values = {}
            for level in levels:
                value = smd_binary((negative[field].astype(str) == level).to_numpy(),
                                   (positive[field].astype(str) == level).to_numpy())
                level_values[level] = float(value)
                max_smd = max(max_smd, abs(value))
            fields[field] = {"kind": "categorical", "level_smd": level_values}
    fine = {}
    for field in config["matching"].get("fine_balance_fields", []):
        levels = sorted(set(negative[field].astype(str)) | set(positive[field].astype(str)))
        fine[field] = {}
        for level in levels:
            difference = float(np.mean(negative[field].astype(str) == level) -
                               np.mean(positive[field].astype(str) == level))
            fine[field][level] = difference
            max_fine = max(max_fine, abs(difference))
    exact_mismatches = {
        field: int(np.sum(negative[field].astype(str).to_numpy() !=
                          positive[field].astype(str).to_numpy()))
        for field in config["matching"]["exact_fields"]
    }
    return {"n_pairs": int(len(pairs)), "max_abs_smd": float(max_smd),
            "max_fine_difference": float(max_fine), "fields": fields,
            "fine_balance": fine, "exact_mismatches": exact_mismatches}


def make_pairs(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    from scipy.optimize import linear_sum_assignment

    exact = config["matching"]["exact_fields"]
    cost_specs = config["matching"].get("cost_fields", [])
    pairs = []
    grouped = candidates.groupby(exact, dropna=False, sort=True) if exact else [((), candidates)]
    for stratum, group in grouped:
        negative = group[group.y == 0].sort_values("participant_identifier").reset_index(drop=True)
        positive = group[group.y == 1].sort_values("participant_identifier").reset_index(drop=True)
        if negative.empty or positive.empty:
            continue
        cost = np.zeros((len(negative), len(positive)), dtype=float)
        for spec in cost_specs:
            field, weight = spec["field"], float(spec.get("weight", 1.0))
            if spec.get("kind") == "continuous":
                scale = float(spec.get("scale", 1.0))
                a = pd.to_numeric(negative[field], errors="coerce").to_numpy()
                b = pd.to_numeric(positive[field], errors="coerce").to_numpy()
                delta = np.abs(a[:, None] - b[None, :]) / scale
                delta[~np.isfinite(delta)] = float(spec.get("missing_penalty", 2.0))
                cost += weight * delta
            else:
                a = negative[field].astype(str).to_numpy()
                b = positive[field].astype(str).to_numpy()
                cost += weight * (a[:, None] != b[None, :])
        # Stable microscopic tie break; too small to change a genuine cost comparison.
        for i, neg in enumerate(negative.participant_identifier):
            for j, pos in enumerate(positive.participant_identifier):
                tie = int(sha256_text(f"cambridge-match-v1|{neg}|{pos}")[:12], 16)
                cost[i, j] += tie / (16 ** 12) * 1e-9
        rows, cols = linear_sum_assignment(cost)
        stratum_text = "|".join(map(str, stratum if isinstance(stratum, tuple) else (stratum,)))
        for i, j in zip(rows, cols):
            neg, pos = negative.iloc[i], positive.iloc[j]
            pair_hash = sha256_text(
                f"cambridge-pair-v1|{neg.participant_identifier}|{pos.participant_identifier}")
            pairs.append({"pair_id": pair_hash[:16],
                          "negative_id": neg.participant_identifier,
                          "positive_id": pos.participant_identifier,
                          "stratum": stratum_text, "cost": float(cost[i, j])})
    return pd.DataFrame(pairs).sort_values("pair_id").reset_index(drop=True) if pairs else \
        pd.DataFrame(columns=["pair_id", "negative_id", "positive_id", "stratum", "cost"])


def trim_pairs(pairs: pd.DataFrame, people: pd.DataFrame,
               config: dict[str, Any], *,
               force_minimum: bool = False) -> tuple[pd.DataFrame, dict[str, Any], int]:
    threshold_smd = float(config["matching"]["max_abs_smd"])
    threshold_fine = float(config["matching"]["max_fine_balance_difference"])
    minimum = int(config["matching"]["min_pairs"])

    def objective(report: dict[str, Any]) -> float:
        return max(report["max_abs_smd"] / threshold_smd,
                   report["max_fine_difference"] / threshold_fine)

    def removal_objectives(current_pairs: pd.DataFrame) -> np.ndarray:
        """Compute every one-pair deletion objective without rebuilding data frames.

        This is algebraically the same balance objective used by ``balance_report``.  The
        vectorised form matters for match-first v2: Cambridge starts with roughly 277 pairs
        and follows the frozen greedy path all the way to exactly 100 pairs.
        """
        n = len(current_pairs)
        if n <= 1:
            return np.full(n, math.inf)
        indexed = people.set_index("participant_identifier")
        negative = indexed.loc[current_pairs.negative_id].reset_index()
        positive = indexed.loc[current_pairs.positive_id].reset_index()
        after_n = float(n - 1)
        max_smd = np.zeros(n, dtype=float)
        continuous = set(config["matching"].get("continuous_smd_fields", []))

        for field in config["matching"]["smd_fields"]:
            if field in continuous:
                a = pd.to_numeric(negative[field], errors="coerce").to_numpy(float)
                b = pd.to_numeric(positive[field], errors="coerce").to_numpy(float)
                observed = ((negative[field].astype(str).to_numpy() != MISSING) &
                            (positive[field].astype(str).to_numpy() != MISSING) &
                            np.isfinite(a) & np.isfinite(b))
                weight = observed.astype(float)
                count = weight.sum() - weight
                sum_a = np.sum(np.where(observed, a, 0.0)) - np.where(observed, a, 0.0)
                sum_b = np.sum(np.where(observed, b, 0.0)) - np.where(observed, b, 0.0)
                square_a = np.sum(np.where(observed, a * a, 0.0)) - \
                    np.where(observed, a * a, 0.0)
                square_b = np.sum(np.where(observed, b * b, 0.0)) - \
                    np.where(observed, b * b, 0.0)
                valid = count >= 2
                value = np.full(n, math.inf)
                if np.any(valid):
                    mean_a = sum_a[valid] / count[valid]
                    mean_b = sum_b[valid] / count[valid]
                    var_a = np.maximum(
                        (square_a[valid] - sum_a[valid] ** 2 / count[valid]) /
                        (count[valid] - 1), 0.0)
                    var_b = np.maximum(
                        (square_b[valid] - sum_b[valid] ** 2 / count[valid]) /
                        (count[valid] - 1), 0.0)
                    pooled = np.sqrt((var_a + var_b) / 2)
                    difference = mean_a - mean_b
                    local = np.divide(
                        difference, pooled, out=np.full_like(difference, math.inf),
                        where=pooled != 0)
                    local[(pooled == 0) & (difference == 0)] = 0.0
                    value[valid] = local
                max_smd = np.maximum(max_smd, np.abs(value))
            else:
                left = negative[field].astype(str).to_numpy()
                right = positive[field].astype(str).to_numpy()
                for level in sorted(set(left) | set(right)):
                    a = (left == level).astype(float)
                    b = (right == level).astype(float)
                    pa = (a.sum() - a) / after_n
                    pb = (b.sum() - b) / after_n
                    scale = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
                    difference = pa - pb
                    value = np.divide(
                        difference, scale, out=np.full_like(difference, math.inf),
                        where=scale != 0)
                    value[(scale == 0) & (difference == 0)] = 0.0
                    max_smd = np.maximum(max_smd, np.abs(value))

        max_fine = np.zeros(n, dtype=float)
        for field in config["matching"].get("fine_balance_fields", []):
            left = negative[field].astype(str).to_numpy()
            right = positive[field].astype(str).to_numpy()
            for level in sorted(set(left) | set(right)):
                a = (left == level).astype(float)
                b = (right == level).astype(float)
                difference = ((a.sum() - a) - (b.sum() - b)) / after_n
                max_fine = np.maximum(max_fine, np.abs(difference))
        return np.maximum(max_smd / threshold_smd, max_fine / threshold_fine)

    current, removed = pairs.copy().reset_index(drop=True), 0
    report = balance_report(current, people, config)
    while (len(current) > minimum and
           (force_minimum or objective(report) > 1)):
        objectives = removal_objectives(current)
        index = min(range(len(current)),
                    key=lambda i: (float(objectives[i]), str(current.loc[i, "pair_id"])))
        current = current.drop(index).reset_index(drop=True)
        removed += 1
        # v1 stops at the first passing cohort.  v2 follows the same objective to the frozen
        # 100-pair endpoint, so only its final report is needed.
        if not force_minimum:
            report = balance_report(current, people, config)
    if force_minimum:
        report = balance_report(current, people, config)
    return current.reset_index(drop=True), report, removed


def assign_match_first_development_splits(eligible: pd.DataFrame, matched_ids: set[str],
                                          config: dict[str, Any]) -> pd.Series:
    """Freeze target first, then split only the remaining participants label x cohort."""
    protocol = config["protocol"]
    cohort = str(protocol["cohort_field"])
    salt = str(protocol.get(
        "development_split_salt", "cambridge-match-first-v2-development"))
    ratios = protocol.get("development_split_ratios", [0.70, 0.15, 0.15])
    if len(ratios) != 3 or not math.isclose(sum(map(float, ratios)), 1.0, abs_tol=1e-12):
        raise ValueError("development_split_ratios must contain three values summing to one")
    train_ratio, validation_ratio, _ = map(float, ratios)
    assignments = {pid: "matched_target" for pid in matched_ids}
    remaining = eligible[~eligible.participant_identifier.isin(matched_ids)]
    for _, group in remaining.groupby(["y", cohort], dropna=False, sort=True):
        ordered = sorted(
            group.participant_identifier.astype(str),
            key=lambda pid: sha256_text(f"{salt}|{pid}"))
        n = len(ordered)
        n_train = int(train_ratio * n)
        n_validation = int(validation_ratio * n)
        for index, pid in enumerate(ordered):
            assignments[pid] = ("train" if index < n_train else
                                "validation" if index < n_train + n_validation else
                                "test")
    result = eligible.participant_identifier.map(assignments)
    if result.isna().any():
        raise AssertionError("match-first split did not assign every eligible participant")
    return result


def execute(config_file: str | Path) -> dict[str, Any]:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    atomic_json(paths["public"] / "frozen_config_public.json", public_config(config))

    participant_csv = resolve_path(config_path, config["inputs"]["participant_csv"])
    assert participant_csv is not None
    raw_columns = list(read_table(participant_csv, nrows=0).columns)
    atomic_json(paths["public"] / "column_inventory.json", {
        "columns": sorted(raw_columns), "n_columns": len(raw_columns),
        "contains_no_row_values": True,
    })

    people, inventory = load_rows(config, config_path)
    audio = scan_audio(config, config_path, set(people.participant_identifier))
    audited, audio_report = audit_audio(audio, config)
    valid_audio = audited[audited.objective_qc_pass & ~audited.duplicate_participant_excluded]
    files = valid_audio.groupby("participant_identifier").audio_path.apply(list)
    people["audio_files_json"] = people.participant_identifier.map(
        lambda pid: json.dumps(files.get(pid, []), separators=(",", ":")))
    people["audio_eligible"] = people.participant_identifier.isin(files.index)
    eligible = people[people.audio_eligible].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("no participant has both standardised metadata and passing cough audio")

    strategy = str(config["protocol"].get("split_strategy", "provided_split_v1"))
    if strategy not in {"provided_split_v1", "match_first_v2"}:
        raise ValueError(f"unknown protocol.split_strategy: {strategy}")
    identity_namespace_version = config["protocol"].get("identity_namespace_version")
    if (strategy == "match_first_v2" and
            identity_namespace_version != "cambridge-task2-official-loader-v1"):
        raise ValueError(
            "match_first_v2 requires the corrected Cambridge Web subject namespace; "
            "rebuild the cohort with the current prepare_reconstructed_cohort.py"
        )
    # v1 preserves the supplied membership and matches test only.  The separately frozen v2
    # sensitivity endpoint matches all eligible people first, then splits only the remainder.
    candidates = (eligible.copy() if strategy == "match_first_v2" else
                  eligible[eligible.splits == "test"].copy())
    pairs_initial = make_pairs(candidates, config)
    pairs, balance, removed = trim_pairs(
        pairs_initial, eligible, config, force_minimum=(strategy == "match_first_v2"))
    matched_ids = set(pairs.negative_id) | set(pairs.positive_id)
    if strategy == "match_first_v2":
        eligible["splits"] = assign_match_first_development_splits(
            eligible, matched_ids, config)
    eligible["in_matched_test"] = eligible.participant_identifier.isin(matched_ids)
    eligible["pair_id"] = MISSING
    pair_of = {row.negative_id: row.pair_id for row in pairs.itertuples()}
    pair_of.update({row.positive_id: row.pair_id for row in pairs.itertuples()})
    eligible["pair_id"] = eligible.participant_identifier.map(pair_of).fillna(MISSING)

    split_counts = {}
    split_names = (["train", "validation", "test", "matched_target"]
                   if strategy == "match_first_v2" else ["train", "validation", "test"])
    for split in split_names:
        subset = eligible[eligible.splits == split]
        split_counts[split] = {"n": int(len(subset)), "negative": int((subset.y == 0).sum()),
                               "positive": int((subset.y == 1).sum())}
    profile_counts = eligible[eligible.splits == "train"].text.value_counts()
    profile_report = {
        "train_unique_profiles": int(len(profile_counts)),
        "train_n": int(profile_counts.sum()),
        "modal_profile_share": float(profile_counts.iloc[0] / profile_counts.sum())
        if len(profile_counts) else None,
        "effective_profiles_exp_entropy": None,
    }
    if len(profile_counts):
        probability = profile_counts.to_numpy() / profile_counts.sum()
        profile_report["effective_profiles_exp_entropy"] = float(
            np.exp(-np.sum(probability * np.log(probability))))

    gate_checks = {
        "participant_linkage": len(eligible) > 0,
        "train_size": split_counts["train"]["n"] >= int(config["protocol"]["min_train_n"]),
        "train_class_size": min(split_counts["train"]["negative"],
                                split_counts["train"]["positive"]) >=
                            int(config["protocol"]["min_train_per_class"]),
        "validation_class_size": min(split_counts["validation"]["negative"],
                                     split_counts["validation"]["positive"]) >=
                                 int(config["protocol"]["min_validation_per_class"]),
        "test_class_size": min(split_counts["test"]["negative"],
                               split_counts["test"]["positive"]) >=
                           int(config["protocol"]["min_test_per_class"]),
        "matching_size": len(pairs) >= int(config["matching"]["min_pairs"]),
        "smd_balance": balance["max_abs_smd"] <= float(config["matching"]["max_abs_smd"]),
        "fine_balance": balance["max_fine_difference"] <=
                        float(config["matching"]["max_fine_balance_difference"]),
        "exact_balance": all(value == 0 for value in balance["exact_mismatches"].values()),
        "split_disjoint": True,  # one canonical row per participant makes this structural
    }
    if strategy == "match_first_v2":
        gate_checks.update({
            "matching_exact_size": len(pairs) == int(config["matching"]["min_pairs"]),
            "validation_size": split_counts["validation"]["n"] >=
                               int(config["protocol"].get("min_validation_n", 75)),
            "source_test_size": split_counts["test"]["n"] >=
                                int(config["protocol"].get("min_test_n", 75)),
            "matched_target_class_size": min(
                split_counts["matched_target"]["negative"],
                split_counts["matched_target"]["positive"]) ==
                int(config["matching"]["min_pairs"]),
            "train_profile_diversity": len(profile_counts) >=
                                       int(config["protocol"].get(
                                           "min_train_unique_profiles", 100)),
            "train_modal_profile_share": (
                len(profile_counts) > 0 and
                float(profile_counts.iloc[0] / profile_counts.sum()) <=
                float(config["protocol"].get("max_train_modal_profile_share", 0.10))),
        })
    verdict = "GO" if all(gate_checks.values()) else "NO_GO"
    split_origin = str(config["protocol"].get("split_origin", "provided participant CSV"))
    public = {
        "format_version": ("cambridge-external-gate-v2" if strategy == "match_first_v2"
                           else "cambridge-external-gate-v1"),
        "verdict": verdict,
        "model_outputs_read": False,
        "representations_generated": False,
        "n_standardised": int(len(people)), "n_audio_eligible": int(len(eligible)),
        "split_strategy": strategy, "split_origin": split_origin,
        "identity_namespace_version": identity_namespace_version,
        "split_counts": split_counts, "audio_qc": audio_report,
        "matching": {"n_initial_pairs": int(len(pairs_initial)),
                     "n_final_pairs": int(len(pairs)), "n_trimmed": removed,
                     "balance": balance},
        "profiles": profile_report, "gate_checks": gate_checks,
        "input_hashes": {"participant_csv": inventory["participant_csv_sha256"],
                         "metadata_csv": inventory["metadata_csv_sha256"]},
        "source_adapter": inventory.get("source_adapter"),
        "standardisation_exclusions": inventory["standardisation_exclusions"],
        "protocol_note": ("NO_GO is a data-feasibility decision, not a model result."
                          if verdict == "NO_GO" else
                          "GO permits the frozen formal model stage; it is not a result."),
    }
    atomic_json(paths["public"] / "data_gate.json", public)

    private_people = paths["private"] / "participant_manifest.csv"
    private_pairs = paths["private"] / "matched_pairs.csv"
    private_audio = paths["private"] / "audio_qc_manifest.csv"
    eligible.to_csv(private_people, index=False)
    pairs.to_csv(private_pairs, index=False)
    audited.to_csv(private_audio, index=False)
    private_manifest = {
        "participant_manifest_sha256": sha256_file(private_people),
        "matched_pairs_sha256": sha256_file(private_pairs),
        "audio_qc_manifest_sha256": sha256_file(private_audio),
        "config_sha256": sha256_file(config_path),
        "verdict": verdict,
    }
    atomic_json(paths["private"] / "manifest_hashes.json", private_manifest)

    report_lines = [
        "# Cambridge COVID-19 Sounds external data gate",
        "", f"**Verdict: {verdict}**", "",
        f"- audio-eligible participants: {len(eligible)}",
        f"- frozen train/validation/source-test ({split_origin}): "
        f"{split_counts['train']['n']} / {split_counts['validation']['n']} / "
        f"{split_counts['test']['n']}",
    ]
    if strategy == "match_first_v2":
        report_lines.append(
            f"- matched target participants: {split_counts['matched_target']['n']}")
    report_lines.extend([
        f"- matched pairs: {len(pairs)} (initial {len(pairs_initial)})",
        f"- max |SMD|: {balance['max_abs_smd']:.4f}",
        f"- max fine-balance difference: {balance['max_fine_difference']:.4f}",
        "", "No representation, prediction, AUROC or NLL was read by this stage.",
    ])
    (paths["public"] / "DATA_GATE_REPORT.md").write_text("\n".join(report_lines) + "\n")
    print(f"DATA GATE {verdict}: {len(eligible)} participants, {len(pairs)} matched pairs")
    print(f"public aggregate output: {paths['public']}")
    return public


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
