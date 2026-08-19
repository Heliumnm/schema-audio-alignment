"""Freeze Coswara external evaluation pairs and participant-level development splits.

Input is the label-blind audio-QC manifest produced by ``audit_coswara_audio.py``.  This
script uses metadata labels only to construct the preregistered paired evaluation sets and
the development split.  It never imports model predictions or representation files.

The primary output is a participant manifest whose hash must be checked by every later
training/evaluation program.  A nonzero exit after writing outputs means the frozen data
or power gate failed; changing rules after that failure is prohibited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


POSITIVE_STATUSES = {"positive_mild", "positive_moderate", "positive_asymp"}
NEGATIVE_STATUSES = {
    "healthy",
    "no_resp_illness_exposed",
    "resp_illness_not_identified",
}
EXACT_FIELDS = (
    "age_decade",
    "sex",
    "cough",
    "fever",
    "fatigue",
    "sore_throat",
    "breathing_difficulty",
    "asthma",
    "other_respiratory",
)
COST_FIELDS = (
    ("smoker", 1.0),
    ("diarrhoea", 1.0),
    ("loss_of_smell", 1.0),
    ("country_group", 2.0),
    ("province_group", 2.0),
    ("halfyear", 2.0),
    ("vaccination", 1.0),
    ("mask_use", 1.0),
    ("manual_quality", 1.0),
)
SMD_CATEGORICAL_FIELDS = (
    "smoker",
    "asthma",
    "other_respiratory",
    "diarrhoea",
    "loss_of_smell",
)
PROPORTION_FIELDS = (
    "country_group",
    "province_group",
    "halfyear",
    "vaccination",
    "mask_use",
    "quality_available",
    "manual_quality",
)
MIN_PRIMARY_PAIRS = 100
MIN_SENSITIVITY_PAIRS = 60
MAX_ABS_SMD = 0.12
MAX_LEVEL_PROPORTION_DIFFERENCE = 0.08
BALANCE_OBJECTIVE_SMD_SCALE = 0.10
BALANCE_OBJECTIVE_PROPORTION_SCALE = 0.05


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_bytes: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def is_true(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"unexpected boolean value: {value!r}")


def symptom(value: str) -> str:
    return "YES" if is_true(value) else "NO"


def any_checkbox(row: dict[str, str], fields: Sequence[str]) -> str:
    return "YES" if any(is_true(row.get(field, "")) for field in fields) else "NO"


def category(value: str) -> str:
    stripped = value.strip()
    return stripped if stripped else "[MISSING]"


def smoker_category(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"y", "true"}:
        return "YES"
    if normalized in {"n", "false"}:
        return "NO"
    if normalized == "":
        return "[MISSING]"
    raise ValueError(f"unexpected smoker value: {value!r}")


def parse_age(value: str) -> int:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"non-integer age: {value!r}")
    return int(numeric)


def broad_label(status: str) -> int | None:
    if status in POSITIVE_STATUSES:
        return 1
    if status in NEGATIVE_STATUSES:
        return 0
    return None


def halfyear(record_date: str) -> str:
    try:
        year_text, month_text, _ = record_date.split("-")
        year, month = int(year_text), int(month_text)
        if not (1 <= month <= 12):
            raise ValueError
    except ValueError:
        return "[MISSING]"
    return f"{year}-H{1 if month <= 6 else 2}"


def province_group(country: str, province: str) -> str:
    if country.strip().lower() != "india":
        return "non-India"
    if province.strip() == "Tamil Nadu":
        return "Tamil Nadu"
    if province.strip() == "Karnataka":
        return "Karnataka"
    return "other India"


def prepare_rows(qc_path: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    with qc_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError("audio-QC manifest is empty")

    excluded = Counter()
    participant_counts = Counter(row["participant_id"] for row in raw)
    multiple_ids = {participant for participant, count in participant_counts.items() if count != 1}
    excluded["multiple_recordings"] = len(multiple_ids)
    unique_rows = []
    for row in raw:
        participant = row["participant_id"]
        if participant in multiple_ids:
            continue
        if not is_true(row["metadata_present"]):
            excluded["metadata_absent"] += 1
            continue
        if not is_true(row["objective_qc_pass"]):
            excluded["objective_audio_qc"] += 1
            continue
        try:
            age = parse_age(row["meta_a"])
        except (ValueError, KeyError):
            excluded["invalid_age"] += 1
            continue
        if not (15 <= age <= 90):
            excluded["age_outside_15_90"] += 1
            continue
        if row.get("meta_rU", "") != "n":
            excluded["not_known_nonreturning"] += 1
            continue
        label = broad_label(row.get("meta_covid_status", ""))
        if label is None:
            excluded["nonbinary_status"] += 1
            continue

        quality = category(row.get("manual_quality", ""))
        prepared: dict[str, object] = dict(row)
        prepared.update(
            {
                "label": label,
                "age": age,
                "age_decade": f"{(age // 10) * 10:02d}",
                "sex": category(row.get("meta_g", "")).lower(),
                "cough": symptom(row.get("meta_cough", "")),
                "fever": symptom(row.get("meta_fever", "")),
                "fatigue": symptom(row.get("meta_ftg", "")),
                "sore_throat": symptom(row.get("meta_st", "")),
                "breathing_difficulty": symptom(row.get("meta_bd", "")),
                "smoker": smoker_category(row.get("meta_smoker", "")),
                "asthma": symptom(row.get("meta_asthma", "")),
                "other_respiratory": any_checkbox(
                    row, ("meta_others_resp", "meta_cld", "meta_pneumonia")
                ),
                "diarrhoea": symptom(row.get("meta_diarrhoea", "")),
                "loss_of_smell": symptom(row.get("meta_loss_of_smell", "")),
                "country": category(row.get("meta_l_c", "")),
                "province": category(row.get("meta_l_s", "")),
                "halfyear": halfyear(row.get("meta_record_date", "")),
                "vaccination": category(row.get("meta_vacc", "")).lower(),
                "mask_use": category(row.get("meta_um", "")).lower(),
                "manual_quality": quality,
                "quality_available": "NO" if quality == "[MISSING]" else "YES",
                "test_status": row.get("meta_test_status", "").strip().lower(),
                "test_type": row.get("meta_testType", "").strip().lower(),
            }
        )
        prepared["province_group"] = province_group(
            str(prepared["country"]), str(prepared["province"])
        )
        unique_rows.append(prepared)

    # Remove cross-ID decoded-PCM duplicates without consulting labels.  A participant can
    # survive at most one group because this stage has exactly one recording per ID.
    pcm_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in unique_rows:
        pcm_groups[str(row["pcm_sha256"])].append(row)
    kept: list[dict[str, object]] = []
    for group in pcm_groups.values():
        if len(group) == 1:
            kept.extend(group)
            continue
        ordered = sorted(
            group,
            key=lambda row: hash_text(f"coswara-dedup-v1|{row['participant_id']}"),
        )
        kept.append(ordered[0])
        excluded["cross_id_pcm_duplicate"] += len(ordered) - 1
    return kept, {key: value for key, value in sorted(excluded.items()) if value}


def exact_stratum(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in EXACT_FIELDS)


def country_groups(rows: Sequence[dict[str, object]]) -> dict[str, str]:
    counts = Counter(str(row["country"]) for row in rows)
    return {country: country if count >= 10 else "OTHER" for country, count in counts.items()}


def pair_cost(negative: dict[str, object], positive: dict[str, object]) -> float:
    cost = abs(int(negative["age"]) - int(positive["age"])) / 10.0
    for field, weight in COST_FIELDS:
        if negative[field] != positive[field]:
            cost += weight
    tie = int(
        hash_text(
            f"coswara-match-v1|{negative['participant_id']}|{positive['participant_id']}"
        ),
        16,
    ) / (2**256)
    return cost + tie * 1e-9


def hungarian(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Minimum-cost assignment for n rows to distinct columns, with n <= m."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m or any(len(row) != m for row in cost):
        raise ValueError("Hungarian input must be rectangular with rows <= columns")
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
    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def match_rows(
    rows: Sequence[dict[str, object]], name: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], dict[int, list[dict[str, object]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for row in rows:
        groups[exact_stratum(row)][int(row["label"])].append(row)

    pairs: list[dict[str, object]] = []
    for stratum, classes in sorted(groups.items()):
        negatives = sorted(classes[0], key=lambda row: str(row["participant_id"]))
        positives = sorted(classes[1], key=lambda row: str(row["participant_id"]))
        if not negatives or not positives:
            continue
        if len(negatives) <= len(positives):
            assignment = hungarian(
                [[pair_cost(negative, positive) for positive in positives] for negative in negatives]
            )
            selected = [(negatives[i], positives[j]) for i, j in assignment]
        else:
            assignment = hungarian(
                [[pair_cost(negative, positive) for negative in negatives] for positive in positives]
            )
            selected = [(negatives[j], positives[i]) for i, j in assignment]
        for negative, positive in selected:
            pair_id = hash_text(
                f"coswara-pair-v1|{name}|{negative['participant_id']}|{positive['participant_id']}"
            )[:20]
            pairs.append(
                {
                    "set": name,
                    "pair_id": pair_id,
                    "stratum": "|".join(stratum),
                    "negative_id": negative["participant_id"],
                    "positive_id": positive["participant_id"],
                    "cost": pair_cost(negative, positive),
                }
            )
    return sorted(pairs, key=lambda row: (str(row["stratum"]), str(row["pair_id"])))


def rows_for_pairs(
    pairs: Sequence[dict[str, object]], by_id: dict[str, dict[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    negative = [by_id[str(pair["negative_id"])] for pair in pairs]
    positive = [by_id[str(pair["positive_id"])] for pair in pairs]
    return negative, positive


def binary_smd(negative: Sequence[bool], positive: Sequence[bool]) -> float:
    p0 = sum(negative) / len(negative)
    p1 = sum(positive) / len(positive)
    denominator = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
    if denominator == 0:
        return 0.0 if p0 == p1 else math.copysign(1.0e12, p1 - p0)
    return (p1 - p0) / denominator


def balance_report(
    pairs: Sequence[dict[str, object]], by_id: dict[str, dict[str, object]]
) -> dict[str, object]:
    if not pairs:
        return {"n_pairs": 0, "passes": False}
    negative, positive = rows_for_pairs(pairs, by_id)
    exact_mismatches = {
        field: sum(a[field] != b[field] for a, b in zip(negative, positive))
        for field in EXACT_FIELDS
    }
    age0 = [float(row["age"]) for row in negative]
    age1 = [float(row["age"]) for row in positive]
    mean0, mean1 = sum(age0) / len(age0), sum(age1) / len(age1)
    var0 = sum((value - mean0) ** 2 for value in age0) / max(len(age0) - 1, 1)
    var1 = sum((value - mean1) ** 2 for value in age1) / max(len(age1) - 1, 1)
    pooled = math.sqrt((var0 + var1) / 2)
    if pooled == 0:
        age_smd = 0.0 if mean0 == mean1 else math.copysign(1.0e12, mean1 - mean0)
    else:
        age_smd = (mean1 - mean0) / pooled

    categorical_smd: dict[str, dict[str, float]] = {}
    for field in SMD_CATEGORICAL_FIELDS:
        levels = sorted({str(row[field]) for row in [*negative, *positive]})
        categorical_smd[field] = {
            level: binary_smd(
                [row[field] == level for row in negative],
                [row[field] == level for row in positive],
            )
            for level in levels
        }
    proportion_differences: dict[str, dict[str, float]] = {}
    for field in PROPORTION_FIELDS:
        levels = sorted({str(row[field]) for row in [*negative, *positive]})
        proportion_differences[field] = {
            level: (
                sum(row[field] == level for row in positive) / len(positive)
                - sum(row[field] == level for row in negative) / len(negative)
            )
            for level in levels
        }

    max_categorical_smd = max(
        (abs(value) for fields in categorical_smd.values() for value in fields.values()),
        default=0.0,
    )
    max_proportion_difference = max(
        (
            abs(value)
            for fields in proportion_differences.values()
            for value in fields.values()
        ),
        default=0.0,
    )
    passes = (
        not any(exact_mismatches.values())
        and abs(age_smd) <= MAX_ABS_SMD
        and max_categorical_smd <= MAX_ABS_SMD
        and max_proportion_difference <= MAX_LEVEL_PROPORTION_DIFFERENCE
    )
    return {
        "n_pairs": len(pairs),
        "exact_mismatches": exact_mismatches,
        "age_smd": age_smd,
        "categorical_smd": categorical_smd,
        "proportion_differences": proportion_differences,
        "max_abs_categorical_smd": max_categorical_smd,
        "max_abs_proportion_difference": max_proportion_difference,
        "passes": passes,
    }


def balance_objective(report: dict[str, object]) -> float:
    return max(
        abs(float(report["age_smd"])) / BALANCE_OBJECTIVE_SMD_SCALE,
        float(report["max_abs_categorical_smd"]) / BALANCE_OBJECTIVE_SMD_SCALE,
        float(report["max_abs_proportion_difference"])
        / BALANCE_OBJECTIVE_PROPORTION_SCALE,
    )


def trim_to_balance(
    pairs: list[dict[str, object]],
    by_id: dict[str, dict[str, object]],
    minimum_pairs: int,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Greedily remove pairs by a frozen balance objective, never below the power floor."""
    current = list(pairs)
    trace: list[dict[str, object]] = []
    while True:
        report = balance_report(current, by_id)
        trace.append(
            {
                "n_pairs": len(current),
                "objective": balance_objective(report) if current else None,
                "passes": report["passes"],
            }
        )
        if bool(report["passes"]) or len(current) <= minimum_pairs:
            return current, report, trace
        candidates = []
        for index, pair in enumerate(current):
            proposed = current[:index] + current[index + 1 :]
            proposed_report = balance_report(proposed, by_id)
            candidates.append(
                (
                    balance_objective(proposed_report),
                    str(pair["pair_id"]),
                    index,
                )
            )
        _, _, remove_index = min(candidates)
        current = current[:remove_index] + current[remove_index + 1 :]


def development_split(rows: Sequence[dict[str, object]]) -> dict[str, str]:
    strata: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        strata[(int(row["label"]), str(row["sex"]), str(row["halfyear"]))].append(row)
    assignment: dict[str, str] = {}
    for stratum_rows in strata.values():
        ordered = sorted(
            stratum_rows,
            key=lambda row: hash_text(f"coswara-split-v1|{row['participant_id']}"),
        )
        n = len(ordered)
        n_train = math.floor(0.70 * n)
        n_validation = math.floor(0.15 * n)
        for index, row in enumerate(ordered):
            if index < n_train:
                split = "train"
            elif index < n_train + n_validation:
                split = "validation"
            else:
                split = "source_test"
            assignment[str(row["participant_id"])] = split
    return assignment


def metadata_hash(row: dict[str, object]) -> str:
    fields = [*EXACT_FIELDS, "age", *(field for field, _ in COST_FIELDS)]
    payload = {field: row[field] for field in fields}
    return hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc", type=Path, default=Path("results/coswara_audio_qc.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/coswara_external_manifest.csv")
    )
    parser.add_argument(
        "--pairs", type=Path, default=Path("results/coswara_external_pairs.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("results/coswara_external_split_audit.json")
    )
    args = parser.parse_args()

    rows, excluded = prepare_rows(args.qc)
    if not rows:
        raise ValueError("no participants remain after objective QC and eligibility filters")
    by_id = {str(row["participant_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise AssertionError("participant IDs are not unique after QC")

    primary_candidates = [
        row
        for row in rows
        if (int(row["label"]) == 1 and row["test_status"] == "p")
        or (int(row["label"]) == 0 and row["test_status"] == "n")
    ]
    mapping = country_groups(primary_candidates)
    for row in rows:
        row["country_group"] = mapping.get(str(row["country"]), "OTHER")

    maximum_pair_sets = {
        "primary": match_rows(primary_candidates, "primary"),
        "manual_quality_1_2": match_rows(
            [row for row in primary_candidates if row["manual_quality"] in {"1", "2"}],
            "manual_quality_1_2",
        ),
        "known_test_type": match_rows(
            [row for row in primary_candidates if row["test_type"] in {"rtpcr", "rat"}],
            "known_test_type",
        ),
        "manual_quality_2": match_rows(
            [row for row in primary_candidates if row["manual_quality"] == "2"],
            "manual_quality_2",
        ),
    }
    pair_sets: dict[str, list[dict[str, object]]] = {}
    balances: dict[str, dict[str, object]] = {}
    balance_traces: dict[str, list[dict[str, object]]] = {}
    for name, maximum_pairs in maximum_pair_sets.items():
        minimum = MIN_PRIMARY_PAIRS if name == "primary" else MIN_SENSITIVITY_PAIRS
        selected, balance, trace = trim_to_balance(maximum_pairs, by_id, minimum)
        pair_sets[name] = selected
        balances[name] = balance
        balance_traces[name] = trace

    primary_pairs = pair_sets["primary"]
    quality_pairs = pair_sets["manual_quality_1_2"]
    testtype_pairs = pair_sets["known_test_type"]

    evaluation_ids = {
        str(pair[field])
        for pairs in pair_sets.values()
        for pair in pairs
        for field in ("negative_id", "positive_id")
    }
    development = [row for row in rows if str(row["participant_id"]) not in evaluation_ids]
    split_assignment = development_split(development)

    membership: dict[str, dict[str, str]] = defaultdict(dict)
    for set_name, pairs in pair_sets.items():
        for pair in pairs:
            membership[str(pair["negative_id"])][set_name] = str(pair["pair_id"])
            membership[str(pair["positive_id"])][set_name] = str(pair["pair_id"])

    manifest_rows: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: str(item["participant_id"])):
        participant = str(row["participant_id"])
        eval_membership = membership.get(participant, {})
        split = "external_evaluation" if eval_membership else split_assignment[participant]
        manifest_rows.append(
            {
                "participant_id": participant,
                "split": split,
                "label": row["label"],
                "relative_path": row["relative_path"],
                "raw_sha256": row["raw_sha256"],
                "pcm_sha256": row["pcm_sha256"],
                "metadata_sha256": metadata_hash(row),
                "primary_pair_id": eval_membership.get("primary", ""),
                "quality_pair_id": eval_membership.get("manual_quality_1_2", ""),
                "testtype_pair_id": eval_membership.get("known_test_type", ""),
                "excellent_pair_id": eval_membership.get("manual_quality_2", ""),
                "exact_stratum": "|".join(exact_stratum(row)),
                "manual_quality": row["manual_quality"],
                "test_status": row["test_status"],
                "test_type": row["test_type"],
            }
        )
    manifest_fields = list(manifest_rows[0])
    atomic_csv(args.manifest, manifest_rows, manifest_fields)

    all_pairs = [pair for pairs in pair_sets.values() for pair in pairs]
    pair_fields = ["set", "pair_id", "stratum", "negative_id", "positive_id", "cost"]
    atomic_csv(args.pairs, all_pairs, pair_fields)

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "source_test", "external_evaluation"):
        selected = [row for row in manifest_rows if row["split"] == split]
        split_counts[split] = {
            "n": len(selected),
            "negative": sum(int(row["label"]) == 0 for row in selected),
            "positive": sum(int(row["label"]) == 1 for row in selected),
        }

    primary_gate = len(primary_pairs) >= MIN_PRIMARY_PAIRS and bool(balances["primary"]["passes"])
    development_gate = (
        split_counts["train"]["n"] >= 900
        and min(split_counts["train"]["negative"], split_counts["train"]["positive"]) >= 250
        and split_counts["validation"]["n"] >= 200
        and min(
            split_counts["validation"]["negative"], split_counts["validation"]["positive"]
        )
        >= 50
        and split_counts["source_test"]["n"] >= 200
        and min(
            split_counts["source_test"]["negative"], split_counts["source_test"]["positive"]
        )
        >= 50
    )
    report: dict[str, object] = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "qc_manifest": str(args.qc),
        "qc_manifest_sha256": sha256_file(args.qc),
        "n_eligible_after_audio_qc_and_dedup": len(rows),
        "excluded": excluded,
        "n_strict_confirmation_candidates": len(primary_candidates),
        "candidate_label_counts": dict(sorted(Counter(int(row["label"]) for row in primary_candidates).items())),
        "pair_counts": {name: len(pairs) for name, pairs in pair_sets.items()},
        "maximum_pair_counts_before_balance_trimming": {
            name: len(pairs) for name, pairs in maximum_pair_sets.items()
        },
        "balances": balances,
        "balance_traces": balance_traces,
        "split_counts": split_counts,
        "primary_pair_gate": primary_gate,
        "development_size_gate": development_gate,
        "manual_quality_sensitivity_powered": (
            len(quality_pairs) >= MIN_SENSITIVITY_PAIRS
            and bool(balances["manual_quality_1_2"]["passes"])
        ),
        "known_test_type_sensitivity_powered": (
            len(testtype_pairs) >= MIN_SENSITIVITY_PAIRS
            and bool(balances["known_test_type"]["passes"])
        ),
        "formal_external_gate": primary_gate and development_gate,
        "manifest_sha256": sha256_file(args.manifest),
        "pairs_sha256": sha256_file(args.pairs),
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["formal_external_gate"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
