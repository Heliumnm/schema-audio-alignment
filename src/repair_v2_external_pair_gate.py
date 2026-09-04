"""Aggregate-only pair-support gates for Transfer Repair v2 external datasets.

The gate reads only source-train labels, domains, and predefined nuisance fields. It never
opens audio, embeddings, checkpoints, predictions, retrieval metrics, or target scores.
Participant identifiers are used only to assert uniqueness and are never written to output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


UNIT = "participant_identifier"
LABEL = "y"
SPLIT = "splits"
MIN_ELIGIBLE_FRACTION = 0.80
MIN_CROSS_DOMAIN_POSITIVES = 2
MIN_MATCHED_NEGATIVES = 1
SCRIPT_PATH = Path(__file__).resolve()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domain: str
    required_fields: tuple[str, ...]
    strict_negative: Callable[[pd.Series, pd.DataFrame], np.ndarray]
    distance_report: Callable[[pd.Series, pd.DataFrame], dict[str, float]]
    rule: dict


def _value(value: object) -> str:
    return "[MISSING]" if pd.isna(value) else str(value).strip()


def _number(value: object) -> float:
    try:
        answer = float(value)
    except (TypeError, ValueError):
        return np.nan
    return answer if np.isfinite(answer) else np.nan


CODA_CLINICAL = (
    "tb_prior", "hemoptysis", "weight_loss", "smoke_lweek", "fever",
    "night_sweats", "hiv_status",
)
CAMBRIDGE_EXACT = (
    "age_band", "sex", "cough", "fever", "sore_throat",
    "shortness_of_breath", "asthma", "other_respiratory",
)


def _base_opposite_same_domain(row: pd.Series, pool: pd.DataFrame, domain: str) -> np.ndarray:
    return ((pool[LABEL].to_numpy(dtype=int) != int(row[LABEL])) &
            (pool[domain].to_numpy(dtype=object) == row[domain]))


def coda_strict_negative(row: pd.Series, pool: pd.DataFrame) -> np.ndarray:
    mask = _base_opposite_same_domain(row, pool, "country")
    mask &= pool["sex"].to_numpy(dtype=object) == row["sex"]
    anchor_age = _number(row["age"])
    ages = pd.to_numeric(pool["age"], errors="coerce").to_numpy(dtype=float)
    mask &= np.isfinite(anchor_age) & np.isfinite(ages) & (np.abs(ages - anchor_age) <= 10.0)
    hamming = np.zeros(len(pool), dtype=np.int64)
    for field in CODA_CLINICAL:
        hamming += pool[field].to_numpy(dtype=object) != row[field]
    mask &= hamming <= 2
    return mask


def coda_distance_report(row: pd.Series, pool: pd.DataFrame) -> dict[str, float]:
    base = _base_opposite_same_domain(row, pool, "country")
    base &= pool["sex"].to_numpy(dtype=object) == row["sex"]
    if not base.any():
        return {"nearest_age_gap_years": np.nan, "nearest_clinical_hamming": np.nan,
                "nearest_total_distance": np.nan}
    anchor_age = _number(row["age"])
    ages = pd.to_numeric(pool["age"], errors="coerce").to_numpy(dtype=float)
    age_gap = np.abs(ages - anchor_age)
    hamming = np.zeros(len(pool), dtype=np.int64)
    for field in CODA_CLINICAL:
        hamming += pool[field].to_numpy(dtype=object) != row[field]
    total = age_gap / 10.0 + hamming
    total[~base | ~np.isfinite(total)] = np.inf
    if not np.isfinite(total).any():
        return {"nearest_age_gap_years": np.nan, "nearest_clinical_hamming": np.nan,
                "nearest_total_distance": np.nan}
    index = int(np.argmin(total))
    return {"nearest_age_gap_years": float(age_gap[index]),
            "nearest_clinical_hamming": float(hamming[index]),
            "nearest_total_distance": float(total[index])}


def cambridge_strict_negative(row: pd.Series, pool: pd.DataFrame) -> np.ndarray:
    mask = _base_opposite_same_domain(row, pool, "platform")
    for field in CAMBRIDGE_EXACT:
        mask &= pool[field].to_numpy(dtype=object) == row[field]
    return mask


def cambridge_distance_report(row: pd.Series, pool: pd.DataFrame) -> dict[str, float]:
    base = _base_opposite_same_domain(row, pool, "platform")
    if not base.any():
        return {"nearest_exact_field_mismatches": np.nan,
                "strict_negative_same_smoker_fraction": np.nan}
    mismatch = np.zeros(len(pool), dtype=np.int64)
    for field in CAMBRIDGE_EXACT:
        mismatch += pool[field].to_numpy(dtype=object) != row[field]
    mismatch[~base] = len(CAMBRIDGE_EXACT) + 1
    strict = base & (mismatch == 0)
    smoker_fraction = np.nan
    if strict.any():
        smoker_fraction = float(
            (pool.loc[strict, "smoker"].to_numpy(dtype=object) == row["smoker"]).mean())
    return {"nearest_exact_field_mismatches": float(mismatch.min()),
            "strict_negative_same_smoker_fraction": smoker_fraction}


SPECS = {
    "coda_tb": DatasetSpec(
        name="CODA TB", domain="country",
        required_fields=("country", "sex", "age", *CODA_CLINICAL),
        strict_negative=coda_strict_negative,
        distance_report=coda_distance_report,
        rule={
            "positive": "same TB label, different country, different participant",
            "negative_exact": ["country", "sex"],
            "negative_age_caliper_years": 10.0,
            "negative_clinical_hamming_fields": list(CODA_CLINICAL),
            "negative_max_clinical_hamming": 2,
        }),
    "cambridge": DatasetSpec(
        name="Cambridge Task 2", domain="platform",
        required_fields=("platform", *CAMBRIDGE_EXACT, "smoker"),
        strict_negative=cambridge_strict_negative,
        distance_report=cambridge_distance_report,
        rule={
            "positive": "same COVID label, different platform, different participant",
            "negative_exact": ["platform", *CAMBRIDGE_EXACT],
            "smoker": "reported diagnostically; not an exact eligibility field",
        }),
}


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(block)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {"n_finite": 0, "min": None, "q25": None, "median": None,
                "q75": None, "max": None}
    q = np.quantile(finite, [0, .25, .5, .75, 1])
    return {"n_finite": int(len(finite)), "min": float(q[0]), "q25": float(q[1]),
            "median": float(q[2]), "q75": float(q[3]), "max": float(q[4])}


def analyse(frame: pd.DataFrame, dataset: str) -> dict:
    spec = SPECS[dataset]
    required = [UNIT, LABEL, SPLIT, spec.domain, *spec.required_fields]
    required = list(dict.fromkeys(required))
    missing = [field for field in required if field not in frame]
    if missing:
        raise ValueError(f"{spec.name} manifest missing fields: {missing}")
    if frame[UNIT].duplicated().any():
        raise ValueError("manifest contains duplicate participant identifiers")

    train = frame.loc[frame[SPLIT].astype(str) == "train", required].copy()
    if not len(train):
        raise ValueError("source train is empty")
    train[LABEL] = pd.to_numeric(train[LABEL], errors="raise").astype(int)
    if set(train[LABEL].unique()) != {0, 1}:
        raise ValueError("source train must contain labels 0 and 1")
    for field in [spec.domain, *spec.required_fields]:
        train[field] = train[field].map(_value)
    # Age must remain numeric for CODA after missing values have been made explicit.
    if dataset == "coda_tb":
        train["age"] = pd.to_numeric(train["age"].replace("[MISSING]", np.nan), errors="coerce")

    domains = sorted(train[spec.domain].unique(), key=str)
    if len(domains) < 2:
        raise ValueError("source train must contain at least two environments")

    positive_counts: list[int] = []
    negative_counts: list[int] = []
    distance_values: dict[str, list[float]] = {}
    for _, row in train.iterrows():
        positive = ((train[LABEL].to_numpy(dtype=int) == int(row[LABEL])) &
                    (train[spec.domain].to_numpy(dtype=object) != row[spec.domain]) &
                    (train[UNIT].to_numpy(dtype=object) != row[UNIT]))
        strict_negative = spec.strict_negative(row, train)
        positive_counts.append(int(positive.sum()))
        negative_counts.append(int(strict_negative.sum()))
        for name, value in spec.distance_report(row, train).items():
            distance_values.setdefault(name, []).append(value)

    train["n_positive"] = positive_counts
    train["n_strict_negative"] = negative_counts
    train["positive_ok"] = train["n_positive"] >= MIN_CROSS_DOMAIN_POSITIVES
    train["negative_ok"] = train["n_strict_negative"] >= MIN_MATCHED_NEGATIVES
    train["eligible"] = train["positive_ok"] & train["negative_ok"]

    def coverage(part: pd.DataFrame) -> dict:
        return {
            "n": int(len(part)),
            "n_positive_ok": int(part["positive_ok"].sum()),
            "fraction_positive_ok": float(part["positive_ok"].mean()),
            "n_negative_ok": int(part["negative_ok"].sum()),
            "fraction_negative_ok": float(part["negative_ok"].mean()),
            "n_joint_eligible": int(part["eligible"].sum()),
            "fraction_joint_eligible": float(part["eligible"].mean()),
            "positive_count_quantiles": _quantiles(part["n_positive"].astype(float).tolist()),
            "strict_negative_count_quantiles": _quantiles(
                part["n_strict_negative"].astype(float).tolist()),
        }

    overall = coverage(train)
    by_label = {str(label): coverage(train.loc[train[LABEL] == label]) for label in (0, 1)}
    table = (train.groupby([spec.domain, LABEL], dropna=False).size().rename("n")
             .reset_index())
    contingency = [
        {"domain": str(row[spec.domain]), "disease_label": int(row[LABEL]), "n": int(row["n"])}
        for _, row in table.iterrows()
    ]
    domain_has_both = {
        str(domain): all(
            int(((train[spec.domain] == domain) & (train[LABEL] == label)).sum()) > 0
            for label in (0, 1))
        for domain in domains
    }
    checks = {
        "overall_joint_coverage_ge_0_80": (
            overall["fraction_joint_eligible"] >= MIN_ELIGIBLE_FRACTION),
        "each_label_joint_coverage_ge_0_80": all(
            by_label[str(label)]["fraction_joint_eligible"] >= MIN_ELIGIBLE_FRACTION
            for label in (0, 1)),
        "each_domain_contains_both_labels": all(domain_has_both.values()),
        "candidate_pool_source_train_only": True,
        "target_participants_used": 0,
        "self_pairing_count": 0,
        "participant_identifiers_unique": True,
    }
    go = (checks["overall_joint_coverage_ge_0_80"] and
          checks["each_label_joint_coverage_ge_0_80"] and
          checks["each_domain_contains_both_labels"] and
          checks["candidate_pool_source_train_only"] and
          checks["target_participants_used"] == 0 and
          checks["self_pairing_count"] == 0 and
          checks["participant_identifiers_unique"])
    return {
        "decision": "GO" if go else "NO_GO",
        "dataset": spec.name,
        "source_train_n": int(len(train)),
        "domain_field": spec.domain,
        "candidate_rule": spec.rule,
        "thresholds": {
            "minimum_cross_domain_same_label_positives": MIN_CROSS_DOMAIN_POSITIVES,
            "minimum_strict_same_domain_opposite_label_negatives": MIN_MATCHED_NEGATIVES,
            "minimum_joint_coverage_overall_and_per_label": MIN_ELIGIBLE_FRACTION,
        },
        "domain_label_contingency": contingency,
        "domain_has_both_labels": domain_has_both,
        "coverage_overall": overall,
        "coverage_by_label": by_label,
        "matching_quality": {name: _quantiles(values)
                             for name, values in distance_values.items()},
        "gate_checks": checks,
        "model_blindness": (
            "Only source-train participant IDs, labels, domains, and frozen nuisance "
            "fields were opened. No audio, representation, checkpoint, prediction, "
            "retrieval metric, target model score, or target participant was used."),
    }


def load_and_analyse(manifest: Path, dataset: str) -> dict:
    frame = pd.read_csv(manifest, low_memory=False)
    result = analyse(frame, dataset)
    result["provenance"] = {
        "manifest_filename": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "preregistration": "docs/TRANSFER_REPAIR_V2_EXTERNAL_GATE_PREREG_ZH.md",
    }
    return result


def _synthetic(dataset: str, coupled: bool = False) -> pd.DataFrame:
    rows = []
    domains = ("A", "B", "C")
    for domain in domains:
        for label in (0, 1):
            if coupled and ((domain == "A" and label == 1) or
                            (domain != "A" and label == 0)):
                continue
            for index in range(24):
                common = {UNIT: f"{dataset}-{domain}-{label}-{index}", LABEL: label,
                          SPLIT: "train"}
                if dataset == "coda_tb":
                    common.update({"country": domain, "sex": "F" if index % 2 else "M",
                                   "age": 30 + index % 4, "tb_prior": index % 2,
                                   "hemoptysis": 0, "weight_loss": label,
                                   "smoke_lweek": index % 2, "fever": 0,
                                   "night_sweats": label, "hiv_status": 0})
                else:
                    common.update({"platform": domain, "age_band": f"age-{index % 2}",
                                   "sex": "F" if index % 2 else "M", "smoker": "NO",
                                   "cough": index % 2, "fever": 0, "sore_throat": 0,
                                   "shortness_of_breath": index % 3 == 0, "asthma": 0,
                                   "other_respiratory": 0})
                rows.append(common)
    return pd.DataFrame(rows)


def self_test() -> None:
    for dataset in SPECS:
        good = analyse(_synthetic(dataset), dataset)
        if good["decision"] != "GO":
            raise AssertionError(f"balanced {dataset} fixture should GO")
        bad = analyse(_synthetic(dataset, coupled=True), dataset)
        if bad["decision"] != "NO_GO":
            raise AssertionError(f"coupled {dataset} fixture should NO_GO")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            atomic_json(bad, path)
            assert json.load(open(path))["decision"] == "NO_GO"
    print("SELF-TEST PASS: balanced fixtures GO and domain-label-coupled fixtures NO_GO")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS))
    parser.add_argument("--manifest")
    parser.add_argument("--out")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or not args.manifest or not args.out:
        parser.error("--dataset, --manifest, and --out are required unless --self-test is used")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    result = load_and_analyse(Path(args.manifest), args.dataset)
    atomic_json(result, output)
    print(json.dumps({
        "decision": result["decision"], "dataset": result["dataset"],
        "source_train_n": result["source_train_n"],
        "domain_label_contingency": result["domain_label_contingency"],
        "coverage_overall": result["coverage_overall"],
        "coverage_by_label": result["coverage_by_label"],
        "matching_quality": result["matching_quality"],
        "gate_checks": result["gate_checks"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
