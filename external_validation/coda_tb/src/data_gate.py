#!/usr/bin/env python3
"""Model-blind CODA TB linkage, waveform-QC, split, and matching gate.

This program intentionally has no dependency on torch, audio encoders, representation
files, predictions, or disease-model scores.  Exit code 3 is the preregistered scientific
NO-GO after all controlled manifests and the aggregate audit have been written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import struct
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_CLINICAL_ROWS = 1105
EXPECTED_SOLICITED_ROWS = 9772
MIN_DURATION_SECONDS = 0.20
MIN_ELIGIBLE_PARTICIPANTS = 1000
MIN_ELIGIBLE_POSITIVE = 250
MIN_ELIGIBLE_NEGATIVE = 700
MIN_MATCHED_PAIRS = 100
MAX_ABS_SMD = 0.12
MAX_LEVEL_DIFFERENCE = 0.08
MIN_TRAIN = 400
MIN_TRAIN_PER_CLASS = 100
MIN_VAL_OR_SOURCE = 75
MIN_VAL_OR_SOURCE_PER_CLASS = 15
MIN_UNIQUE_PROFILES = 100
MAX_PROFILE_FRACTION = 0.10
PROTOCOL_SPLIT_FIRST_V1 = "split-first-v1"
PROTOCOL_MATCH_FIRST_V2 = "match-first-v2"

EXACT_MATCH_FIELDS = ("country", "sex")
CONTINUOUS_BALANCE_FIELDS = (
    "age",
    "height",
    "weight",
    "log_cough_days",
    "heart_rate",
    "temperature",
)
CATEGORICAL_BALANCE_FIELDS = (
    "country",
    "sex",
    "tb_prior",
    "tb_prior_pul",
    "tb_prior_extrapul",
    "tb_prior_unknown",
    "hemoptysis",
    "weight_loss",
    "smoke_lweek",
    "fever",
    "night_sweats",
    "hiv_status",
)
FINE_BALANCE_FIELDS = CATEGORICAL_BALANCE_FIELDS
PROFILE_FIELDS = (
    "age",
    "sex",
    "height",
    "weight",
    "reported_cough_dur",
    "tb_prior",
    "tb_prior_pul",
    "tb_prior_extrapul",
    "tb_prior_unknown",
    "hemoptysis",
    "heart_rate",
    "temperature",
    "weight_loss",
    "smoke_lweek",
    "fever",
    "night_sweats",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_unique_csv(path: Path, key: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    if key not in rows[0]:
        raise ValueError(f"missing key {key!r} in {path}")
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        identifier = row[key].strip()
        if not identifier:
            raise ValueError(f"blank {key} in {path}")
        if identifier in mapping:
            raise ValueError(f"duplicate {key}={identifier!r} in {path}")
        mapping[identifier] = row
    return rows, mapping


def category(value: object) -> str:
    text = str(value).strip()
    return text if text else "[MISSING]"


def number(value: object, field: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"missing numeric field {field}")
    parsed = float(text)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite numeric field {field}: {text!r}")
    return parsed


def label_value(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "positive", "tb positive"}:
        return 1
    if text in {"0", "negative", "tb negative"}:
        return 0
    raise ValueError(f"unexpected tb_status: {value!r}")


def pcm_has_signal(frames: bytes, sample_width: int) -> bool:
    if sample_width == 1:
        return any(value != 128 for value in frames)
    if sample_width == 2:
        return any(value != 0 for (value,) in struct.iter_unpack("<h", frames))
    if sample_width == 4:
        return any(value != 0 for (value,) in struct.iter_unpack("<i", frames))
    if sample_width == 3:
        for offset in range(0, len(frames), 3):
            chunk = frames[offset : offset + 3]
            if len(chunk) != 3:
                return False
            value = int.from_bytes(chunk, "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            if value != 0:
                return True
        return False
    return False


def audit_wav(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "filename": path.name,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "qc_pass": False,
        "failure": "",
        "raw_sha256": "",
        "pcm_sha256": "",
        "sample_rate": 0,
        "channels": 0,
        "sample_width": 0,
        "frames": 0,
        "duration_seconds": 0.0,
    }
    if not path.is_file():
        result["failure"] = "missing"
        return result
    if path.stat().st_size <= 44:
        result["failure"] = "empty_or_header_only"
        return result
    result["raw_sha256"] = sha256_file(path)
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            n_frames = handle.getnframes()
            compression = handle.getcomptype()
            frames = handle.readframes(n_frames)
            trailing = handle.readframes(1)
    except (wave.Error, EOFError, OSError) as exc:
        result["failure"] = f"decode:{type(exc).__name__}"
        return result
    result.update(
        {
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width": sample_width,
            "frames": n_frames,
            "duration_seconds": n_frames / sample_rate if sample_rate else 0.0,
        }
    )
    if compression != "NONE":
        result["failure"] = f"compressed:{compression}"
        return result
    if channels < 1 or sample_rate < 1 or sample_width not in {1, 2, 3, 4}:
        result["failure"] = "invalid_format"
        return result
    expected_bytes = n_frames * channels * sample_width
    if len(frames) != expected_bytes or trailing:
        result["failure"] = "frame_count_mismatch"
        return result
    if result["duration_seconds"] < MIN_DURATION_SECONDS:
        result["failure"] = "too_short"
        return result
    if not pcm_has_signal(frames, sample_width):
        result["failure"] = "all_zero"
        return result
    pcm_digest = hashlib.sha256()
    pcm_digest.update(f"{channels}|{sample_width}|{sample_rate}|".encode("ascii"))
    pcm_digest.update(frames)
    result["pcm_sha256"] = pcm_digest.hexdigest()
    result["qc_pass"] = True
    return result


def missingness(rows: Sequence[dict[str, str]], fields: Sequence[str]) -> dict[str, float]:
    denominator = len(rows)
    return {
        field: (sum(not row.get(field, "").strip() for row in rows) / denominator if denominator else math.nan)
        for field in fields
    }


def prepare_participants(
    clinical: dict[str, dict[str, str]],
    additional: dict[str, dict[str, str]],
    valid_files: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    excluded = Counter()
    participants: list[dict[str, object]] = []
    for participant in sorted(clinical):
        if participant not in additional:
            excluded["additional_missing"] += 1
            continue
        files = valid_files.get(participant, [])
        if not files:
            excluded["no_valid_solicited_cough"] += 1
            continue
        row = clinical[participant]
        domain = additional[participant]
        try:
            age = number(row["age"], "age")
            height = number(row["height"], "height")
            weight = number(row["weight"], "weight")
            cough_days = number(row["reported_cough_dur"], "reported_cough_dur")
            heart_rate = number(row["heart_rate"], "heart_rate")
            temperature = number(row["temperature"], "temperature")
            label = label_value(row["tb_status"])
        except (KeyError, ValueError):
            excluded["invalid_required_metadata"] += 1
            continue
        prepared: dict[str, object] = {
            "participant": participant,
            "label": label,
            "country": category(domain.get("Country", "")),
            "sex": category(row.get("sex", "")),
            "age": age,
            "height": height,
            "weight": weight,
            "reported_cough_dur": cough_days,
            "log_cough_days": math.log1p(max(cough_days, 0.0)),
            "tb_prior": category(row.get("tb_prior", "")),
            "tb_prior_pul": category(row.get("tb_prior_Pul", "")),
            "tb_prior_extrapul": category(row.get("tb_prior_Extrapul", "")),
            "tb_prior_unknown": category(row.get("tb_prior_Unknown", "")),
            "hemoptysis": category(row.get("hemoptysis", "")),
            "heart_rate": heart_rate,
            "temperature": temperature,
            "weight_loss": category(row.get("weight_loss", "")),
            "smoke_lweek": category(row.get("smoke_lweek", "")),
            "fever": category(row.get("fever", "")),
            "night_sweats": category(row.get("night_sweats", "")),
            "hiv_status": category(domain.get("HIVstatus", "")),
            "audio_count": len(files),
            "audio_files": "|".join(str(item["filename"]) for item in sorted(files, key=lambda x: str(x["filename"]))),
            "pcm_hashes": "|".join(str(item["pcm_sha256"]) for item in sorted(files, key=lambda x: str(x["filename"]))),
        }
        profile = "|".join(f"{field}={prepared[field]}" for field in PROFILE_FIELDS)
        prepared["profile_sha256"] = sha256_text(profile)
        participants.append(prepared)
    return participants, dict(sorted(excluded.items()))


def stratum(row: dict[str, object]) -> tuple[object, ...]:
    return row["label"], row["country"], row["sex"]


def deterministic_outer_split(rows: Sequence[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: sha256_text(f"coda-outer-v1|{row['participant']}"))
        n_development = math.floor(0.60 * len(ordered))
        if len(ordered) >= 2:
            n_development = min(max(n_development, 1), len(ordered) - 1)
        for index, row in enumerate(ordered):
            row["outer_split"] = "development" if index < n_development else "target_candidate"


def deterministic_development_split(rows: Sequence[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["outer_split"] == "development":
            groups[stratum(row)].append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: sha256_text(f"coda-dev-v1|{row['participant']}"))
        n_train = math.floor(0.70 * len(ordered))
        n_validation = math.floor(0.15 * len(ordered))
        for index, row in enumerate(ordered):
            if index < n_train:
                row["split"] = "train"
            elif index < n_train + n_validation:
                row["split"] = "validation"
            else:
                row["split"] = "source_test"
    for row in rows:
        if row["outer_split"] == "target_candidate":
            row["split"] = "target_unmatched"


def pair_cost(negative: dict[str, object], positive: dict[str, object]) -> float:
    cost = abs(float(negative["age"]) - float(positive["age"])) / 10.0
    cost += abs(float(negative["height"]) - float(positive["height"])) / 10.0
    cost += abs(float(negative["weight"]) - float(positive["weight"])) / 10.0
    cost += abs(float(negative["log_cough_days"]) - float(positive["log_cough_days"]))
    cost += abs(float(negative["heart_rate"]) - float(positive["heart_rate"])) / 10.0
    cost += abs(float(negative["temperature"]) - float(positive["temperature"]))
    for field in CATEGORICAL_BALANCE_FIELDS:
        if field in EXACT_MATCH_FIELDS:
            continue
        cost += 1.0 if negative[field] != positive[field] else 0.0
    tie = int(
        sha256_text(f"coda-match-v1|{negative['participant']}|{positive['participant']}"), 16
    ) / (2**256)
    return cost + tie * 1e-9


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment for n rows to distinct columns, n <= m."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m or any(len(row) != m for row in cost):
        raise ValueError("Hungarian matrix must be rectangular with rows <= columns")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


def initial_pairs(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[int, list[dict[str, object]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in rows:
        if row["outer_split"] != "target_candidate":
            continue
        key = (str(row["country"]), str(row["sex"]))
        groups[key][int(row["label"])].append(row)
    pairs: list[dict[str, object]] = []
    for key in sorted(groups):
        negatives = sorted(groups[key][0], key=lambda row: str(row["participant"]))
        positives = sorted(groups[key][1], key=lambda row: str(row["participant"]))
        if not negatives or not positives:
            continue
        if len(negatives) <= len(positives):
            matrix = [[pair_cost(negative, positive) for positive in positives] for negative in negatives]
            assignments = [(negatives[i], positives[j]) for i, j in hungarian(matrix)]
        else:
            matrix = [[pair_cost(negative, positive) for negative in negatives] for positive in positives]
            assignments = [(negatives[j], positives[i]) for i, j in hungarian(matrix)]
        for negative, positive in assignments:
            pair_id = sha256_text(
                f"coda-pair-v1|{negative['participant']}|{positive['participant']}"
            )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "country": key[0],
                    "sex": key[1],
                    "negative_id": negative["participant"],
                    "positive_id": positive["participant"],
                    "negative": negative,
                    "positive": positive,
                    "cost": pair_cost(negative, positive),
                }
            )
    return sorted(pairs, key=lambda pair: str(pair["pair_id"]))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def smd(negative: Sequence[float], positive: Sequence[float]) -> float:
    pooled = math.sqrt((variance(negative) + variance(positive)) / 2.0)
    difference = mean(positive) - mean(negative)
    if pooled == 0:
        return 0.0 if difference == 0 else math.copysign(math.inf, difference)
    return difference / pooled


def balance(pairs: Sequence[dict[str, object]]) -> dict[str, object]:
    negatives = [pair["negative"] for pair in pairs]
    positives = [pair["positive"] for pair in pairs]
    field_smd: dict[str, float] = {}
    level_difference: dict[str, float] = {}
    for field in CONTINUOUS_BALANCE_FIELDS:
        field_smd[field] = smd(
            [float(row[field]) for row in negatives],
            [float(row[field]) for row in positives],
        )
    for field in CATEGORICAL_BALANCE_FIELDS:
        levels = sorted({str(row[field]) for row in negatives + positives})
        for level in levels:
            key = f"{field}={level}"
            neg = [1.0 if str(row[field]) == level else 0.0 for row in negatives]
            pos = [1.0 if str(row[field]) == level else 0.0 for row in positives]
            field_smd[key] = smd(neg, pos)
            level_difference[key] = abs(mean(pos) - mean(neg))
    finite_smd = [abs(value) for value in field_smd.values() if math.isfinite(value)]
    infinite = any(not math.isfinite(value) for value in field_smd.values())
    maximum_smd = math.inf if infinite else max(finite_smd, default=0.0)
    maximum_level = max(level_difference.values(), default=0.0)
    exact_ok = all(
        all(pair["negative"][field] == pair["positive"][field] for pair in pairs)
        for field in EXACT_MATCH_FIELDS
    )
    passed = (
        len(pairs) >= MIN_MATCHED_PAIRS
        and exact_ok
        and maximum_smd <= MAX_ABS_SMD
        and maximum_level <= MAX_LEVEL_DIFFERENCE
    )
    return {
        "n_pairs": len(pairs),
        "exact_balance_ok": exact_ok,
        "max_abs_smd": maximum_smd,
        "max_level_difference": maximum_level,
        "field_smd": dict(sorted(field_smd.items())),
        "level_difference": dict(sorted(level_difference.items())),
        "passed": passed,
    }


def balance_objective(report: dict[str, object]) -> float:
    return max(
        float(report["max_abs_smd"]) / MAX_ABS_SMD,
        float(report["max_level_difference"]) / MAX_LEVEL_DIFFERENCE,
    )


def trim_pairs(pairs: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    current = list(pairs)
    removed: list[str] = []
    while len(current) >= MIN_MATCHED_PAIRS:
        report = balance(current)
        if report["passed"]:
            return current, removed
        if len(current) == MIN_MATCHED_PAIRS:
            break
        candidates: list[tuple[float, str, int]] = []
        for index, pair in enumerate(current):
            remaining = current[:index] + current[index + 1 :]
            objective = balance_objective(balance(remaining))
            candidates.append((objective, str(pair["pair_id"]), index))
        _, pair_id, index = min(candidates)
        removed.append(pair_id)
        del current[index]
    return current, removed


def trim_pairs_to_floor(
    pairs: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[str]]:
    """Follow the v1 greedy path to exactly the frozen 100-pair target size.

    Unlike v1, v2 does not stop when a larger set first passes the balance limits.  The
    target size is fixed at the preregistered power floor so that target selection cannot
    consume a result-dependent amount of the development cohort.
    """
    current = list(pairs)
    removed: list[str] = []
    while len(current) > MIN_MATCHED_PAIRS:
        candidates: list[tuple[float, str, int]] = []
        for index, pair in enumerate(current):
            remaining = current[:index] + current[index + 1 :]
            objective = balance_objective(balance(remaining))
            candidates.append((objective, str(pair["pair_id"]), index))
        _, pair_id, index = min(candidates)
        removed.append(pair_id)
        del current[index]
    return current, removed


def assign_splits_and_pairs(
    participants: list[dict[str, object]], protocol: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    """Apply one frozen split/matching protocol and mutate participant split fields."""
    if protocol == PROTOCOL_SPLIT_FIRST_V1:
        deterministic_outer_split(participants)
        deterministic_development_split(participants)
        pairs_before = initial_pairs(participants)
        pairs, removed_pair_ids = trim_pairs(pairs_before)
    elif protocol == PROTOCOL_MATCH_FIRST_V2:
        # Every eligible participant is allowed to compete for the matched endpoint.  No
        # development assignment exists until the endpoint is frozen.
        for row in participants:
            row["outer_split"] = "target_candidate"
            row["split"] = "target_unmatched"
        pairs_before = initial_pairs(participants)
        pairs, removed_pair_ids = trim_pairs_to_floor(pairs_before)

        matched_ids = {
            str(identifier)
            for pair in pairs
            for identifier in (pair["negative_id"], pair["positive_id"])
        }
        for row in participants:
            if str(row["participant"]) in matched_ids:
                row["outer_split"] = "matched_target"
                row["split"] = "matched_target"
            else:
                row["outer_split"] = "development"
                row["split"] = ""
        # Reuse the exact v1 development salt and 70/15/15 rule.  Matched rows have an
        # outer_split value ignored by this function and therefore cannot flow back.
        deterministic_development_split(participants)
        return pairs_before, pairs, removed_pair_ids
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    matched_ids = {
        str(identifier)
        for pair in pairs
        for identifier in (pair["negative_id"], pair["positive_id"])
    }
    for row in participants:
        if str(row["participant"]) in matched_ids:
            row["split"] = "matched_target"
    return pairs_before, pairs, removed_pair_ids


def class_counts(rows: Sequence[dict[str, object]], split: str) -> dict[str, int]:
    selected = [row for row in rows if row.get("split") == split]
    counts = Counter(int(row["label"]) for row in selected)
    return {"n": len(selected), "negative": counts[0], "positive": counts[1]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=(PROTOCOL_SPLIT_FIRST_V1, PROTOCOL_MATCH_FIRST_V2),
        default=PROTOCOL_SPLIT_FIRST_V1,
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    output = args.output_dir.resolve()
    clinical_path = root / "meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv"
    additional_path = root / "meta_data/Clinical/CODA_TB_additional_variables_train.csv"
    solicited_map_path = root / "meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv"
    audio_root = root / "raw_data/solicited_data"
    for path in (clinical_path, additional_path, solicited_map_path, audio_root):
        if not path.exists():
            raise FileNotFoundError(path)

    clinical_rows, clinical = read_unique_csv(clinical_path, "participant")
    additional_rows, additional = read_unique_csv(additional_path, "participant")
    with solicited_map_path.open(newline="", encoding="utf-8-sig") as handle:
        solicited_rows = list(csv.DictReader(handle))
    filenames = [row["filename"].strip() for row in solicited_rows]
    if any(not name or Path(name).name != name for name in filenames):
        raise ValueError("solicited metadata contains blank or unsafe filename")
    if len(set(filenames)) != len(filenames):
        raise ValueError("solicited metadata filenames are not unique")
    mapped_participants = {row["participant"].strip() for row in solicited_rows}
    unknown_mapped = mapped_participants - set(clinical)

    actual_files = {path.name for path in audio_root.glob("*.wav") if path.is_file()}
    expected_files = set(filenames)
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files)

    participant_for_file = {row["filename"].strip(): row["participant"].strip() for row in solicited_rows}
    qc_rows: list[dict[str, object]] = []
    for index, filename in enumerate(sorted(expected_files), start=1):
        audited = audit_wav(audio_root / filename)
        audited["participant"] = participant_for_file[filename]
        qc_rows.append(audited)
        if index % 1000 == 0:
            print(f"audio_qc {index}/{len(expected_files)}", flush=True)

    passing = [row for row in qc_rows if row["qc_pass"]]
    pcm_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in passing:
        pcm_groups[str(row["pcm_sha256"])].append(row)
    retained_by_participant: dict[str, list[dict[str, object]]] = defaultdict(list)
    cross_participant_duplicate_groups = 0
    cross_participant_duplicate_files_removed = 0
    within_participant_duplicate_files_removed = 0
    for group in pcm_groups.values():
        by_participant: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            by_participant[str(row["participant"])].append(row)
        participants = sorted(by_participant)
        if len(participants) > 1:
            cross_participant_duplicate_groups += 1
            owner = min(participants, key=lambda pid: sha256_text(f"coda-dedup-v1|{pid}"))
            cross_participant_duplicate_files_removed += sum(
                len(items) for pid, items in by_participant.items() if pid != owner
            )
            participants = [owner]
        for participant in participants:
            items = sorted(by_participant[participant], key=lambda row: str(row["filename"]))
            retained_by_participant[participant].append(items[0])
            within_participant_duplicate_files_removed += len(items) - 1

    participants, excluded = prepare_participants(clinical, additional, retained_by_participant)
    pairs_before, pairs, removed_pair_ids = assign_splits_and_pairs(
        participants, args.protocol
    )
    pair_report = balance(pairs)

    profile_counts = Counter(str(row["profile_sha256"]) for row in participants)
    eligible_counts = Counter(int(row["label"]) for row in participants)
    split_counts = {
        split: class_counts(participants, split)
        for split in ("train", "validation", "source_test", "matched_target", "target_unmatched")
    }
    split_sets = {
        split: {str(row["participant"]) for row in participants if row["split"] == split}
        for split in split_counts
    }
    split_overlap = {
        f"{left}__{right}": len(split_sets[left] & split_sets[right])
        for index, left in enumerate(split_sets)
        for right in list(split_sets)[index + 1 :]
    }
    pcm_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in participants:
        for digest in str(row["pcm_hashes"]).split("|"):
            if digest:
                pcm_to_splits[digest].add(str(row["split"]))
    cross_split_pcm = sum(len(splits) > 1 for splits in pcm_to_splits.values())

    metadata_fields = list(clinical_rows[0])
    country_missingness = {
        country: missingness(
            [clinical[pid] for pid, row in additional.items() if row.get("Country", "").strip() == country],
            metadata_fields,
        )
        for country in sorted({row.get("Country", "").strip() for row in additional_rows})
    }
    label_missingness = {
        str(label): missingness(
            [row for row in clinical_rows if label_value(row["tb_status"]) == label], metadata_fields
        )
        for label in (0, 1)
    }

    gates = {
        "clinical_row_count": len(clinical_rows) == EXPECTED_CLINICAL_ROWS,
        "additional_row_count": len(additional_rows) == EXPECTED_CLINICAL_ROWS,
        "clinical_additional_participant_sets_equal": set(clinical) == set(additional),
        "solicited_row_count": len(solicited_rows) == EXPECTED_SOLICITED_ROWS,
        "solicited_filenames_unique": len(set(filenames)) == len(filenames),
        "mapped_participants_known": not unknown_mapped,
        "download_file_set_exact": not missing_files and not extra_files,
        "eligible_participant_count": len(participants) >= MIN_ELIGIBLE_PARTICIPANTS,
        "eligible_positive_count": eligible_counts[1] >= MIN_ELIGIBLE_POSITIVE,
        "eligible_negative_count": eligible_counts[0] >= MIN_ELIGIBLE_NEGATIVE,
        "matched_pair_count": len(pairs) >= MIN_MATCHED_PAIRS,
        "matched_balance": bool(pair_report["passed"]),
        "train_size": split_counts["train"]["n"] >= MIN_TRAIN,
        "train_class_size": min(split_counts["train"]["negative"], split_counts["train"]["positive"]) >= MIN_TRAIN_PER_CLASS,
        "validation_size": split_counts["validation"]["n"] >= MIN_VAL_OR_SOURCE,
        "validation_class_size": min(split_counts["validation"]["negative"], split_counts["validation"]["positive"]) >= MIN_VAL_OR_SOURCE_PER_CLASS,
        "source_test_size": split_counts["source_test"]["n"] >= MIN_VAL_OR_SOURCE,
        "source_test_class_size": min(split_counts["source_test"]["negative"], split_counts["source_test"]["positive"]) >= MIN_VAL_OR_SOURCE_PER_CLASS,
        "profile_unique_count": len(profile_counts) >= MIN_UNIQUE_PROFILES,
        "profile_largest_fraction": (max(profile_counts.values(), default=0) / len(participants)) <= MAX_PROFILE_FRACTION,
        "participant_split_overlap": max(split_overlap.values(), default=0) == 0,
        "pcm_split_overlap": cross_split_pcm == 0,
        "model_blind_program": True,
    }
    go = all(gates.values())

    qc_fields = [
        "participant",
        "filename",
        "exists",
        "size_bytes",
        "qc_pass",
        "failure",
        "raw_sha256",
        "pcm_sha256",
        "sample_rate",
        "channels",
        "sample_width",
        "frames",
        "duration_seconds",
    ]
    atomic_csv(output / "coda_tb_audio_qc.csv", qc_rows, qc_fields)
    participant_fields = [
        "participant",
        "label",
        "country",
        "sex",
        "split",
        "outer_split",
        "age",
        "height",
        "weight",
        "reported_cough_dur",
        "log_cough_days",
        "tb_prior",
        "tb_prior_pul",
        "tb_prior_extrapul",
        "tb_prior_unknown",
        "hemoptysis",
        "heart_rate",
        "temperature",
        "weight_loss",
        "smoke_lweek",
        "fever",
        "night_sweats",
        "hiv_status",
        "audio_count",
        "profile_sha256",
        "audio_files",
        "pcm_hashes",
    ]
    atomic_csv(output / "coda_tb_participant_manifest.csv", participants, participant_fields)
    pair_rows = [
        {
            "pair_id": pair["pair_id"],
            "country": pair["country"],
            "sex": pair["sex"],
            "negative_id": pair["negative_id"],
            "positive_id": pair["positive_id"],
            "cost": pair["cost"],
        }
        for pair in pairs
    ]
    atomic_csv(
        output / "coda_tb_matched_pairs.csv",
        pair_rows,
        ("pair_id", "country", "sex", "negative_id", "positive_id", "cost"),
    )

    audit: dict[str, object] = {
        "schema_version": (
            "coda-tb-data-gate-match-first-v2"
            if args.protocol == PROTOCOL_MATCH_FIRST_V2
            else "coda-tb-data-gate-v1"
        ),
        "protocol": args.protocol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "source": {"train_synapse_id": "syn39711065", "solicited_synapse_id": "syn40358494"},
        "input_hashes": {
            "clinical_sha256": sha256_file(clinical_path),
            "additional_sha256": sha256_file(additional_path),
            "solicited_map_sha256": sha256_file(solicited_map_path),
        },
        "linkage": {
            "clinical_rows": len(clinical_rows),
            "additional_rows": len(additional_rows),
            "solicited_rows": len(solicited_rows),
            "mapped_participants": len(mapped_participants),
            "unknown_mapped_participants": len(unknown_mapped),
            "missing_wav": len(missing_files),
            "extra_wav": len(extra_files),
        },
        "audio_qc": {
            "passing_files": len(passing),
            "failure_counts": dict(sorted(Counter(str(row["failure"]) for row in qc_rows if not row["qc_pass"]).items())),
            "format_counts": {
                "sample_rate": dict(sorted(Counter(str(row["sample_rate"]) for row in passing).items())),
                "channels": dict(sorted(Counter(str(row["channels"]) for row in passing).items())),
                "sample_width": dict(sorted(Counter(str(row["sample_width"]) for row in passing).items())),
            },
            "cross_participant_duplicate_groups": cross_participant_duplicate_groups,
            "cross_participant_duplicate_files_removed": cross_participant_duplicate_files_removed,
            "within_participant_duplicate_files_removed": within_participant_duplicate_files_removed,
        },
        "eligible": {
            "participants": len(participants),
            "negative": eligible_counts[0],
            "positive": eligible_counts[1],
            "excluded": excluded,
            "countries": dict(sorted(Counter(str(row["country"]) for row in participants).items())),
            "country_label": {
                country: {
                    "negative": sum(row["country"] == country and row["label"] == 0 for row in participants),
                    "positive": sum(row["country"] == country and row["label"] == 1 for row in participants),
                }
                for country in sorted({str(row["country"]) for row in participants})
            },
        },
        "missingness": {
            "overall": missingness(clinical_rows, metadata_fields),
            "by_label": label_missingness,
            "by_country": country_missingness,
        },
        "profiles": {
            "unique": len(profile_counts),
            "largest_count": max(profile_counts.values(), default=0),
            "largest_fraction": max(profile_counts.values(), default=0) / len(participants) if participants else math.nan,
            "shared_fraction": sum(count for count in profile_counts.values() if count > 1) / len(participants) if participants else math.nan,
        },
        "splits": split_counts,
        "split_overlap": split_overlap,
        "cross_split_pcm_groups": cross_split_pcm,
        "matching": {
            "candidate_pool": (
                "all eligible participants"
                if args.protocol == PROTOCOL_MATCH_FIRST_V2
                else "frozen 40% target candidate"
            ),
            "target_size_rule": (
                f"exactly {MIN_MATCHED_PAIRS} pairs along the v1 greedy path"
                if args.protocol == PROTOCOL_MATCH_FIRST_V2
                else "first balanced set along the v1 greedy path, never below floor"
            ),
            "initial_pairs": len(pairs_before),
            "removed_pairs": len(removed_pair_ids),
            "final": pair_report,
        },
        "gates": gates,
        "decision": "GO" if go else "NO-GO",
        "controlled_output_hashes": {},
    }
    for name in ("coda_tb_audio_qc.csv", "coda_tb_participant_manifest.csv", "coda_tb_matched_pairs.csv"):
        audit["controlled_output_hashes"][name] = sha256_file(output / name)  # type: ignore[index]
    atomic_json(output / "coda_tb_data_gate.json", audit)
    print(json.dumps({"decision": audit["decision"], "gates": gates, "matching": audit["matching"]}, indent=2))
    return 0 if go else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
