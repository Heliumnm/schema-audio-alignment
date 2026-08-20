"""Exploratory UKCOVID matched direct-fusion attribution control.

The official test sets have informed earlier protocol work, so this script cannot produce
confirmation.  Its purpose is narrower: determine whether the observed transfer failure is
specific to metadata alignment or whether raw frozen audio also adds no detectable disease
information once the same metadata is available.

No data path runs without the explicit ``--evaluate-tests`` switch.  ``--self-test`` uses
only synthetic data and does not open repository results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_metadata_alignment import (
    PROJECTOR_ARMS,
    TEST_SPECS,
    atomic_json,
    atomic_npz,
    fit_disease_head,
    hierarchical_ci,
    load_cohort,
    load_representation,
    make_validation_folds,
    paired_hierarchical_ci,
    sha16,
    verify_training_manifest,
)
from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from metadata_text import BINARY_MAP, FIELDS, MISSING, symptom_columns


ARMS = (
    "metadata_only",
    "metadata_raw_ast",
    "metadata_correct",
    "metadata_within_label",
    "metadata_global",
)
ALIGNMENT_BY_FUSION_ARM = {
    "metadata_raw_ast": "raw_ast",
    "metadata_correct": "correct",
    "metadata_within_label": "within_label",
    "metadata_global": "global",
}
COMPARISONS = (
    ("metadata_correct", "metadata_within_label",
     "individual_correspondence_primary"),
    ("metadata_raw_ast", "metadata_only", "direct_audio_increment"),
    ("metadata_correct", "metadata_raw_ast", "alignment_versus_raw_fusion"),
    ("metadata_within_label", "metadata_global", "label_level_cooccurrence"),
    ("metadata_correct", "metadata_only", "total_correct_alignment_increment"),
)


def stable_unique(values) -> list[str]:
    out: list[str] = []
    for value in values:
        value = str(value)
        if value not in out:
            out.append(value)
    return out


def mapped_one_hot(series: pd.Series, mapping: dict, tag: str) -> tuple[np.ndarray, list[str]]:
    observed = series.dropna()
    unknown = observed[~observed.isin(list(mapping))]
    if len(unknown):
        raise ValueError(f"{tag}: unknown values {sorted(set(map(str, unknown)))[:8]}")
    mapped = series.map(mapping).fillna(MISSING).astype(str)
    levels = stable_unique(list(mapping.values()) + [MISSING])
    X = np.stack([(mapped == level).to_numpy(np.float32) for level in levels], axis=1)
    assert np.allclose(X.sum(axis=1), 1.0)
    return X, [f"{tag}={level}" for level in levels]


def build_metadata_matrix(data_dir: Path, cohort_path: Path,
                          participants: np.ndarray) -> tuple[np.ndarray, list[str], dict]:
    cohort = pd.read_csv(cohort_path)
    metadata = pd.read_csv(data_dir / "participant_metadata.csv", low_memory=False)
    symptoms = symptom_columns(metadata)
    specs = list(FIELDS) + [
        (column.replace("symptom_", "").upper(), column, BINARY_MAP)
        for column in symptoms
    ]
    columns = [UNIT] + [column for _, column, _ in specs]
    d = cohort[[UNIT]].merge(metadata[columns], on=UNIT, validate="one_to_one")
    order = {pid: i for i, pid in enumerate(cohort[UNIT])}
    d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
    assert np.array_equal(d[UNIT].to_numpy(), participants)

    matrices, names = [], []
    for tag, column, mapping in specs:
        X, feature_names = mapped_one_hot(d[column], mapping, tag)
        matrices.append(X)
        names.extend(feature_names)
    M = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
    assert M.shape == (len(participants), len(names)) and np.isfinite(M).all()
    audit = {
        "shape": list(M.shape),
        "feature_names": names,
        "feature_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")).hexdigest(),
        "matrix_sha16": sha16(M),
        "symptom_columns": symptoms,
        "excluded_symptoms": ["symptom_onset", "symptom_prefer_not_to_say"],
        "semantics": "fixed metadata_text.py mappings with explicit [MISSING] one-hot",
    }
    return M, names, audit


def metrics(predictions: dict[str, np.ndarray], d: pd.DataFrame,
            tests: dict[str, np.ndarray], boot: int, seed: int) -> dict:
    y_all = d["y"].to_numpy()
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    out = {"arms": {}, "comparisons": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        P = predictions[arm]
        for test_name, mask in tests.items():
            y, Pt = y_all[mask], P[:, mask]
            auc, auc_ci = hierarchical_ci(Pt, y, auroc, boot, seed)
            nnll, nnll_ci = hierarchical_ci(Pt, y, neg_nll, boot, seed + 1)
            out["arms"][arm][test_name] = {
                "n": int(mask.sum()),
                "auroc": auc,
                "auroc_ci": auc_ci,
                "neg_nll": nnll,
                "neg_nll_ci": nnll_ci,
                "per_seed_auroc": [auroc(y, p) for p in Pt],
                "per_seed_nll": [float(logloss(y, p).mean()) for p in Pt],
                "calibration_per_seed": [calib_diag(y, p) for p in Pt],
            }
    for a, b, label in COMPARISONS:
        key = f"{a}_minus_{b}"
        out["comparisons"][key] = {"interpretation": label}
        for test_name, mask in tests.items():
            y = y_all[mask]
            A, B = predictions[a][:, mask], predictions[b][:, mask]
            out["comparisons"][key][test_name] = {
                "delta_neg_nll": paired_hierarchical_ci(
                    A, B, y, neg_nll, boot, seed + 2),
                "delta_auroc": paired_hierarchical_ci(
                    A, B, y, auroc, boot, seed + 3),
            }
    out["primary_result_path"] = (
        "comparisons.metadata_correct_minus_metadata_within_label.matched.delta_neg_nll")
    return out


def self_test() -> None:
    s = pd.Series(["A", "B", None, "A"])
    X, names = mapped_one_hot(s, {"A": "ALPHA", "B": "BETA"}, "TEST")
    assert names == ["TEST=ALPHA", "TEST=BETA", f"TEST={MISSING}"]
    assert np.array_equal(X.argmax(1), np.asarray([0, 1, 2, 0]))
    try:
        mapped_one_hot(pd.Series(["C"]), {"A": "ALPHA"}, "TEST")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown metadata value did not raise")
    print("SELF-TEST PASS: fixed one-hot mappings and fail-closed unknown values")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--ast-emb", default="results/ast_embeddings.npz")
    ap.add_argument("--alignment-dir", default="results/alignment")
    ap.add_argument("--out-dir", default="results/direct_fusion_matched")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--evaluate-tests", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.evaluate_tests:
        ap.error("test evaluation is locked; pass --evaluate-tests explicitly")
    seeds = list(args.seeds)
    assert len(seeds) == len(set(seeds)) == 5

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.npz", "head_fits.json", "results.json"):
        assert not (out_dir / name).exists(), f"refusing to overwrite {out_dir / name}"

    manifest = verify_training_manifest(
        Path(args.alignment_dir) / "manifest.json", seeds, args.expected_epochs)
    d, train, val, tests = load_cohort(args)
    participants = d[UNIT].to_numpy()
    y = d["y"].to_numpy()
    folds = make_validation_folds(d, val)
    M, _, metadata_audit = build_metadata_matrix(
        Path(args.data), Path(args.cohort), participants)

    raw_ast_cache: dict[str, np.ndarray] = {}
    predictions: dict[str, np.ndarray] = {}
    selection: dict[str, list[dict]] = {}
    arrays: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }

    for fusion_arm in ARMS:
        aligned_arm = ALIGNMENT_BY_FUSION_ARM.get(fusion_arm)
        fit_seeds = [seeds[0]] if fusion_arm in ("metadata_only", "metadata_raw_ast") \
            else seeds
        arm_heads = []
        for seed in fit_seeds:
            if aligned_arm is None:
                X = M
                representation_hash = metadata_audit["matrix_sha16"]
            else:
                R = load_representation(args, participants, aligned_arm, seed, "raw",
                                        raw_ast_cache, manifest)
                X = np.concatenate([M, R], axis=1)
                representation_hash = sha16(R)
            print(f"{fusion_arm} seed {seed}: {X.shape[1]} features", flush=True)
            head = fit_disease_head(X, y, train, val, folds, seed)
            arm_heads.append(head)
            selection.setdefault(fusion_arm, []).append({
                "seed": seed,
                "C": head.C,
                "calibrator_coef": head.calibrator_coef,
                "calibrator_intercept": head.calibrator_intercept,
                "selection": head.selection,
                "audio_representation_hash": representation_hash,
            })
            del X

        if len(arm_heads) == 1:
            arm_heads = arm_heads * len(seeds)
            base = selection[fusion_arm][0]
            selection[fusion_arm] = [
                {**base, "seed": seed, "replicated_deterministic_reference": True}
                for seed in seeds
            ]
        P = np.stack([head.calibrated for head in arm_heads])
        L = np.stack([head.logits for head in arm_heads])
        O = np.stack([head.val_oof_calibrated for head in arm_heads])
        predictions[fusion_arm] = P
        arrays[f"{fusion_arm}__calibrated_prob"] = P
        arrays[f"{fusion_arm}__raw_logits"] = L
        arrays[f"{fusion_arm}__val_oof_calibrated_prob"] = O

    # Freeze predictions and fitted parameters before computing any test statistic.
    atomic_npz(out_dir / "predictions.npz", **arrays)
    atomic_json({
        "standing": "exploratory; official UKCOVID tests informed earlier protocols",
        "training_manifest": manifest,
        "metadata_audit": metadata_audit,
        "validation_folds_hash": sha16(folds),
        "selection": selection,
        "test_status": "explicitly_unlocked_after_predictions_were_saved",
    }, out_dir / "head_fits.json")

    result = {
        "standing": "exploratory attribution control",
        "primary_comparison": (
            "metadata_correct minus metadata_within_label on matched delta_neg_nll"),
        "calibration": "single final Standard-validation Platt calibrator per arm/seed",
        "metrics": metrics(predictions, d, tests, args.bootstrap, args.bootstrap_seed),
    }
    atomic_json(result, out_dir / "results.json")
    print(f"wrote frozen predictions, fits and results under {out_dir}")


if __name__ == "__main__":
    main()
