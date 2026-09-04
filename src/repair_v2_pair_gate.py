"""Model-blind pair-feasibility gate for Transfer Repair v2.

This script reads only Standard-train participant labels and measured nuisance fields.  It
does not import or open audio embeddings, projector checkpoints, predictions, or test
metrics.  Public output is aggregate-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


UNIT = "participant_identifier"
MIN_ELIGIBLE_FRACTION = 0.80
MIN_CROSS_SOURCE_POSITIVES = 2
MIN_MATCHED_NEGATIVES = 1
MATCH_KEYS = (
    "recruitment_source",
    "gender",
    "age",
    "symptom_cough_any",
    "symptom_none",
)
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(block)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _counts(frame: pd.DataFrame, keys: list[str]) -> dict[tuple, int]:
    return {tuple(k if isinstance(k, tuple) else (k,)): int(v)
            for k, v in frame.groupby(keys, dropna=False).size().items()}


def _lookup(table: dict[tuple, int], *key) -> int:
    return int(table.get(tuple(key), 0))


def analyse(train: pd.DataFrame) -> dict:
    required = [UNIT, "y", *MATCH_KEYS]
    missing = [column for column in required if column not in train]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if train[UNIT].duplicated().any():
        raise ValueError("Standard train contains duplicate participant identifiers")
    if set(train["y"].astype(int).unique()) != {0, 1}:
        raise ValueError("Standard train must contain both disease labels")

    d = train[required].copy()
    d["y"] = d["y"].astype(int)
    for column in MATCH_KEYS:
        d[column] = d[column].astype("object").where(d[column].notna(), "[MISSING]")

    by_y = _counts(d, ["y"])
    by_y_source = _counts(d, ["y", "recruitment_source"])
    by_y_source_sex = _counts(d, ["y", "recruitment_source", "gender"])
    by_y_source_age = _counts(d, ["y", "recruitment_source", "age"])
    by_y_source_sex_age = _counts(
        d, ["y", "recruitment_source", "gender", "age"])
    by_stratum_y = _counts(d, [*MATCH_KEYS, "y"])
    by_source_y = by_y_source
    sources = sorted(d["recruitment_source"].unique(), key=str)

    p_total, p_best, n_strict, n_source_only, p_same_disease = [], [], [], [], []
    for row in d.itertuples(index=False):
        y = int(row.y)
        source = row.recruitment_source
        sex = row.gender
        age = row.age
        cough = row.symptom_cough_any
        none = row.symptom_none

        total_other_source = (_lookup(by_y, y) - _lookup(by_y_source, y, source))
        same_sex_other_source = (
            sum(_lookup(by_y_source_sex, y, s, sex) for s in sources if s != source))
        same_age_other_source = (
            sum(_lookup(by_y_source_age, y, s, age) for s in sources if s != source))
        same_both_other_source = sum(
            _lookup(by_y_source_sex_age, y, s, sex, age)
            for s in sources if s != source)
        score2 = total_other_source - same_sex_other_source - same_age_other_source + \
            same_both_other_source
        score0 = same_both_other_source
        score1 = total_other_source - score2 - score0
        assert min(score0, score1, score2) >= 0
        best = next((count for count in (score2, score1, score0)
                     if count >= MIN_CROSS_SOURCE_POSITIVES), 0)

        opposite = 1 - y
        strict = _lookup(
            by_stratum_y, source, sex, age, cough, none, opposite)
        source_only = _lookup(by_source_y, opposite, source)
        p_total.append(total_other_source)
        p_best.append(best)
        n_strict.append(strict)
        n_source_only.append(source_only)
        p_same_disease.append(_lookup(by_y, y) - 1)

    d["n_cross_source_same_disease"] = np.asarray(p_total, dtype=np.int64)
    d["n_preferred_cross_source_same_disease"] = np.asarray(p_best, dtype=np.int64)
    d["n_strict_matched_opposite_disease"] = np.asarray(n_strict, dtype=np.int64)
    d["n_same_source_opposite_disease"] = np.asarray(n_source_only, dtype=np.int64)
    d["n_same_disease_any_source"] = np.asarray(p_same_disease, dtype=np.int64)
    d["positive_ok"] = (
        d["n_preferred_cross_source_same_disease"] >= MIN_CROSS_SOURCE_POSITIVES)
    d["negative_ok"] = (
        d["n_strict_matched_opposite_disease"] >= MIN_MATCHED_NEGATIVES)
    d["eligible"] = d["positive_ok"] & d["negative_ok"]

    def coverage(mask: np.ndarray) -> dict:
        q = d.loc[mask]
        return {
            "n": int(len(q)),
            "n_positive_ok": int(q["positive_ok"].sum()),
            "fraction_positive_ok": float(q["positive_ok"].mean()),
            "n_negative_ok": int(q["negative_ok"].sum()),
            "fraction_negative_ok": float(q["negative_ok"].mean()),
            "n_eligible_both": int(q["eligible"].sum()),
            "fraction_eligible_both": float(q["eligible"].mean()),
            "cross_positive_count_quantiles": np.quantile(
                q["n_cross_source_same_disease"], [0, .25, .5, .75, 1]).tolist(),
            "strict_negative_count_quantiles": np.quantile(
                q["n_strict_matched_opposite_disease"], [0, .25, .5, .75, 1]).tolist(),
        }

    contingency = (d.groupby(["recruitment_source", "y"], dropna=False).size()
                   .rename("n").reset_index())
    contingency_records = [
        {"recruitment_source": str(row.recruitment_source),
         "disease_label": int(row.y), "n": int(row.n)}
        for row in contingency.itertuples(index=False)
    ]
    source_has_both = {
        source: all(_lookup(by_source_y, label, source) > 0 for label in (0, 1))
        for source in sources
    }
    all_coverage = coverage(np.ones(len(d), dtype=bool))
    class_coverage = {
        str(label): coverage((d["y"] == label).to_numpy()) for label in (0, 1)}
    gate_checks = {
        "overall_eligible_fraction_ge_0_80": (
            all_coverage["fraction_eligible_both"] >= MIN_ELIGIBLE_FRACTION),
        "each_label_eligible_fraction_ge_0_80": all(
            class_coverage[str(label)]["fraction_eligible_both"] >= MIN_ELIGIBLE_FRACTION
            for label in (0, 1)),
        "each_source_contains_both_labels": all(source_has_both.values()),
        "candidate_pool_is_standard_train_only": True,
        "self_pairing_is_impossible_by_definition": True,
    }
    go = all(gate_checks.values())
    diagnostics = {
        "fraction_with_any_same_disease_other_participant": float(
            (d["n_same_disease_any_source"] >= 1).mean()),
        "fraction_with_any_same_source_opposite_label": float(
            (d["n_same_source_opposite_disease"] >= 1).mean()),
        "note": (
            "These relaxed counts are diagnostic only and cannot substitute for the "
            "frozen cross-source-positive plus strict-matched-negative gate."),
    }
    return {
        "decision": "GO" if go else "NO_GO",
        "gate_checks": gate_checks,
        "frozen_thresholds": {
            "minimum_eligible_fraction_overall_and_per_label": MIN_ELIGIBLE_FRACTION,
            "minimum_cross_source_positives_per_anchor": MIN_CROSS_SOURCE_POSITIVES,
            "minimum_strict_matched_negatives_per_anchor": MIN_MATCHED_NEGATIVES,
            "strict_negative_match_keys": list(MATCH_KEYS),
        },
        "n_standard_train": int(len(d)),
        "source_label_contingency": contingency_records,
        "source_has_both_labels": source_has_both,
        "coverage_overall": all_coverage,
        "coverage_by_disease_label": class_coverage,
        "relaxed_diagnostics_not_used_for_gate": diagnostics,
        "model_blindness": (
            "No audio embedding, projector, prediction, retrieval metric, validation "
            "metric, matched metric, or matched-long metric was opened."),
    }


def load_standard_train(cohort_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    cohort = pd.read_csv(cohort_path, low_memory=False)
    metadata = pd.read_csv(metadata_path, low_memory=False)
    cohort_columns = [UNIT, "splits", "y"]
    metadata_columns = [UNIT, *MATCH_KEYS]
    for frame, columns, name in ((cohort, cohort_columns, "cohort"),
                                 (metadata, metadata_columns, "metadata")):
        missing = [column for column in columns if column not in frame]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame[UNIT].duplicated().any():
            raise ValueError(f"{name} contains duplicate participant identifiers")
    d = cohort[cohort_columns].merge(
        metadata[metadata_columns], on=UNIT, validate="one_to_one")
    train = d.loc[d["splits"] == "train"].drop(columns="splits").reset_index(drop=True)
    if len(train) != int((cohort["splits"] == "train").sum()):
        raise ValueError("not every Standard-train participant linked to metadata")
    provenance = {
        "cohort_path": str(cohort_path),
        "cohort_sha256": sha256_file(cohort_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }
    return train, provenance


def self_test() -> None:
    rows = []
    for source in ("A", "B"):
        for y in (0, 1):
            for i in range(24):
                rows.append({
                    UNIT: f"{source}{y}{i}", "y": y,
                    "recruitment_source": source,
                    "gender": "F" if i % 2 else "M",
                    "age": "old" if (i // 2) % 2 else "young",
                    "symptom_cough_any": i % 2,
                    "symptom_none": (i + 1) % 2,
                })
    good = analyse(pd.DataFrame(rows))
    assert good["decision"] == "GO"
    bad_frame = pd.DataFrame(rows)
    bad_frame["recruitment_source"] = np.where(bad_frame["y"] == 1, "A", "B")
    bad = analyse(bad_frame)
    assert bad["decision"] == "NO_GO"
    assert bad["coverage_overall"]["fraction_eligible_both"] == 0.0
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gate.json"
        atomic_json(bad, out)
        assert json.load(open(out))["decision"] == "NO_GO"
    print("SELF-TEST PASS: balanced source-label data GO; deterministic source-label "
          "coupling NO_GO", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument(
        "--metadata",
        default="/mnt/hd/data_heliu/resp_datasets/ukcovid/participant_metadata.csv")
    parser.add_argument("--out", default="results/transfer_repair_v2_pair_gate.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    train, provenance = load_standard_train(Path(args.cohort), Path(args.metadata))
    result = analyse(train)
    result.update({
        "standing": "model-blind Transfer Repair v2 pair-feasibility gate",
        "provenance": provenance,
    })
    atomic_json(result, output)
    print(json.dumps({
        "decision": result["decision"],
        "n_standard_train": result["n_standard_train"],
        "source_label_contingency": result["source_label_contingency"],
        "coverage_overall": result["coverage_overall"],
        "coverage_by_disease_label": result["coverage_by_disease_label"],
        "gate_checks": result["gate_checks"],
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
