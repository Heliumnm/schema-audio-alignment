"""Data-only GO/NO-GO gate for Cambridge COVID-19 Sounds.

This program never imports torch or opens a representation/prediction file. It standardises
the restricted release, audits cough waveforms, freezes a participant-level official split,
and constructs a deterministic matched subset inside the official test population.

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


def read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the Cambridge release without requiring a manual Excel-to-CSV conversion."""

    if path.suffix.casefold() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, nrows=nrows)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    raise ValueError(f"unsupported table format {path.suffix!r}; use CSV, XLSX or XLSM")


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
    columns = config["columns"]
    primary_required = {columns["participant_id"], columns["label"], columns["split"]}
    primary_missing = sorted(primary_required - set(source.columns))
    if primary_missing:
        raise ValueError(
            f"participant CSV is missing configured columns {primary_missing}; available columns "
            f"are written to the column inventory")

    extra_path = resolve_path(config_path, inputs.get("metadata_csv"))
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
        "metadata_csv_sha256": sha256_file(extra_path) if extra_path else None,
        "n_input_rows": int(len(source)),
        "columns": sorted(map(str, source.columns)),
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


def scan_audio(config: dict[str, Any], config_path: Path) -> pd.DataFrame:
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
            path = Path(str(values[mapping["path"]]))
            path = path if path.is_absolute() else audio_root / path
            rows.append({"participant_identifier": safe_identifier(values[mapping["participant_id"]]),
                         "audio_path": str(path.resolve()), "modality": modality})
    else:
        if audio.get("scan_layout") != "cambridge_task2":
            raise ValueError("without audio_manifest_csv, audio.scan_layout must be cambridge_task2")
        for path in sorted(audio_root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in AUDIO_SUFFIXES:
                continue
            name = path.name.casefold()
            if "cough" not in name:
                continue
            relative = path.relative_to(audio_root)
            parts = relative.parts
            if len(parts) < 2:
                continue
            pid = parts[1] if parts[0] == "form-app-users" and len(parts) >= 3 else parts[0]
            rows.append({"participant_identifier": safe_identifier(pid),
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
               config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], int]:
    threshold_smd = float(config["matching"]["max_abs_smd"])
    threshold_fine = float(config["matching"]["max_fine_balance_difference"])
    minimum = int(config["matching"]["min_pairs"])

    def objective(report: dict[str, Any]) -> float:
        return max(report["max_abs_smd"] / threshold_smd,
                   report["max_fine_difference"] / threshold_fine)

    current, removed = pairs.copy(), 0
    report = balance_report(current, people, config)
    while len(current) > minimum and objective(report) > 1:
        # Deterministic greedy path. It stops at the first passing cohort, never below floor.
        best = None
        for index in current.index:
            trial = current.drop(index).reset_index(drop=True)
            trial_report = balance_report(trial, people, config)
            key = (objective(trial_report), current.loc[index, "pair_id"])
            if best is None or key < best[0]:
                best = (key, trial, trial_report)
        assert best is not None
        _, current, report = best
        removed += 1
    return current.reset_index(drop=True), report, removed


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
    audio = scan_audio(config, config_path)
    audited, audio_report = audit_audio(audio, config)
    valid_audio = audited[audited.objective_qc_pass & ~audited.duplicate_participant_excluded]
    files = valid_audio.groupby("participant_identifier").audio_path.apply(list)
    people["audio_files_json"] = people.participant_identifier.map(
        lambda pid: json.dumps(files.get(pid, []), separators=(",", ":")))
    people["audio_eligible"] = people.participant_identifier.isin(files.index)
    eligible = people[people.audio_eligible].copy().reset_index(drop=True)
    if eligible.empty:
        raise ValueError("no participant has both standardised metadata and passing cough audio")

    # Official train/validation/test membership is immutable; target matching uses test only.
    candidates = eligible[eligible.splits == "test"].copy()
    pairs_initial = make_pairs(candidates, config)
    pairs, balance, removed = trim_pairs(pairs_initial, eligible, config)
    matched_ids = set(pairs.negative_id) | set(pairs.positive_id)
    eligible["in_matched_test"] = eligible.participant_identifier.isin(matched_ids)
    eligible["pair_id"] = MISSING
    pair_of = {row.negative_id: row.pair_id for row in pairs.itertuples()}
    pair_of.update({row.positive_id: row.pair_id for row in pairs.itertuples()})
    eligible["pair_id"] = eligible.participant_identifier.map(pair_of).fillna(MISSING)

    split_counts = {}
    for split in ("train", "validation", "test"):
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
    verdict = "GO" if all(gate_checks.values()) else "NO_GO"
    public = {
        "format_version": "cambridge-external-gate-v1",
        "verdict": verdict,
        "model_outputs_read": False,
        "representations_generated": False,
        "n_standardised": int(len(people)), "n_audio_eligible": int(len(eligible)),
        "split_counts": split_counts, "audio_qc": audio_report,
        "matching": {"n_initial_pairs": int(len(pairs_initial)),
                     "n_final_pairs": int(len(pairs)), "n_trimmed": removed,
                     "balance": balance},
        "profiles": profile_report, "gate_checks": gate_checks,
        "input_hashes": {"participant_csv": inventory["participant_csv_sha256"],
                         "metadata_csv": inventory["metadata_csv_sha256"]},
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
        f"- official train/validation/test: {split_counts['train']['n']} / "
        f"{split_counts['validation']['n']} / {split_counts['test']['n']}",
        f"- matched pairs: {len(pairs)} (initial {len(pairs_initial)})",
        f"- max |SMD|: {balance['max_abs_smd']:.4f}",
        f"- max fine-balance difference: {balance['max_fine_difference']:.4f}",
        "", "No representation, prediction, AUROC or NLL was read by this stage.",
    ]
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
