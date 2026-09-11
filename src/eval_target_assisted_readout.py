"""Preregistered E3 target-assisted readout diagnostic for UKCOVID.

For each frozen representation, the scaler, regularisation strength, logistic disease
readout and Platt calibrator are learned using participant-disjoint matched-long only.
The resulting model is then applied unchanged to matched.  This is a post-hoc diagnostic
using target-distribution labels, never a source-only or deployable performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audio_baselines_v2 import (C_GRID, FOLD_SEED, N_FOLDS, UNIT, auroc,
                                calib_diag, logloss)
from audit_information_channels import sha256_file
from eval_metadata_alignment import (atomic_json, atomic_npz, load_cohort,
                                     load_representation, paired_hierarchical_ci,
                                     sha16, verify_training_manifest)
from eval_within_label_sex_control import (NEW_ARM, load_e2_representation,
                                          verify_e2_manifest)


ARMS = ("raw", "correct", "within_label", "global", NEW_ARM)
ALIGNED_ARMS = ("correct", "within_label", "global", NEW_ARM)
COMPARISONS = (
    ("correct", "within_label", "original exact-pair increment"),
    ("correct", NEW_ARM, "exact-pair increment after preserving label and recorded sex"),
    (NEW_ARM, "within_label", "increment from preserving sex in shuffled pairs"),
    ("correct", "raw", "correct alignment versus frozen raw audio"),
)
SCRIPT_PATH = Path(__file__).resolve()


@dataclass
class TargetHead:
    C: float
    logits: np.ndarray
    calibrated: np.ndarray
    oof_logits: np.ndarray
    selection: dict
    calibrator_coef: float
    calibrator_intercept: float


def make_target_folds(y: np.ndarray, fit: np.ndarray) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    index = np.flatnonzero(fit)
    folds = np.full(len(y), -1, dtype=np.int16)
    splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for fold, (_, held) in enumerate(splitter.split(index, y[index])):
        folds[index[held]] = fold
    assert np.all(folds[fit] >= 0) and np.all(folds[~fit] == -1)
    assert all(set(np.unique(y[folds == k])) == {0, 1} for k in range(N_FOLDS))
    return folds


def fold_logits(X: np.ndarray, y: np.ndarray, fit: np.ndarray, folds: np.ndarray,
                C: float, seed: int) -> tuple[np.ndarray, list[float]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan, dtype=np.float64)
    scores = []
    for fold in range(N_FOLDS):
        inner = fit & (folds != fold)
        held = folds == fold
        scaler = StandardScaler().fit(X[inner])
        model = LogisticRegression(C=C, max_iter=5000, random_state=seed)
        model.fit(scaler.transform(X[inner]), y[inner])
        oof[held] = model.decision_function(scaler.transform(X[held]))
        scores.append(float(auroc(y[held], oof[held])))
    assert np.isfinite(oof[fit]).all() and np.isnan(oof[~fit]).all()
    return oof, scores


def fit_target_head(X: np.ndarray, y: np.ndarray, fit: np.ndarray,
                    folds: np.ndarray, seed: int) -> TargetHead:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    candidate: dict[float, tuple[np.ndarray, list[float]]] = {}
    means, ses = {}, {}
    for C in C_GRID:
        oof, scores = fold_logits(X, y, fit, folds, C, seed)
        candidate[C] = (oof, scores)
        means[C] = float(np.mean(scores))
        ses[C] = float(np.std(scores, ddof=1) / np.sqrt(N_FOLDS))
    best = max(C_GRID, key=lambda value: means[value])
    threshold = means[best] - ses[best]
    chosen = min(value for value in C_GRID if means[value] >= threshold)
    oof = candidate[chosen][0]

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(oof[fit, None], y[fit])
    scaler = StandardScaler().fit(X[fit])
    final = LogisticRegression(C=chosen, max_iter=5000, random_state=seed)
    final.fit(scaler.transform(X[fit]), y[fit])
    logits = final.decision_function(scaler.transform(X)).astype(np.float64)
    calibrated = calibrator.predict_proba(logits[:, None])[:, 1].astype(np.float64)
    assert np.isfinite(logits).all() and np.isfinite(calibrated).all()
    return TargetHead(
        C=float(chosen), logits=logits, calibrated=calibrated, oof_logits=oof,
        selection={
            "mean_auroc": {str(C): means[C] for C in C_GRID},
            "se": {str(C): ses[C] for C in C_GRID},
            "per_fold": {str(C): candidate[C][1] for C in C_GRID},
            "best_C": float(best), "one_se_threshold": float(threshold),
            "chosen_C": float(chosen),
        },
        calibrator_coef=float(calibrator.coef_[0, 0]),
        calibrator_intercept=float(calibrator.intercept_[0]),
    )


def fit_all(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    if predictions_path.exists() or fits_path.exists():
        raise FileExistsError("refusing to overwrite E3 fit output")
    seeds = list(args.seeds)
    original_manifest_path = Path(args.alignment_dir) / "manifest.json"
    original_manifest = verify_training_manifest(
        original_manifest_path, seeds, args.expected_epochs)
    e2_manifest_path = Path(args.e2_alignment_dir) / "manifest.json"
    e2_manifest = verify_e2_manifest(e2_manifest_path, seeds, args.expected_epochs)
    data, _, _, tests = load_cohort(args)
    participants = data[UNIT].to_numpy()
    y = data["y"].to_numpy()
    fit = tests["matched_long"]
    score = tests["matched"]
    assert not np.any(fit & score)
    folds = make_target_folds(y, fit)

    arrays = {
        "participants": participants,
        "seeds": np.asarray(seeds),
        "arms": np.asarray(ARMS),
        "label": y,
        "mask__matched_long": fit,
        "mask__matched": score,
        "folds__matched_long": folds,
    }
    fits: dict[str, dict] = {arm: {} for arm in ARMS}
    raw_cache: dict[str, np.ndarray] = {}
    for arm in ARMS:
        arm_heads = []
        fit_seeds = [seeds[0]] if arm == "raw" else seeds
        for seed in fit_seeds:
            if arm == NEW_ARM:
                representation = load_e2_representation(
                    Path(args.e2_alignment_dir), e2_manifest, participants, seed)
            else:
                loader_arm = "raw_ast" if arm == "raw" else arm
                representation = load_representation(
                    args, participants, loader_arm, seed, "raw", raw_cache,
                    original_manifest)
            head = fit_target_head(representation, y, fit, folds, seed)
            arm_heads.append(head)
            fits[arm][str(seed)] = {
                "representation_sha16": sha16(representation),
                "C": head.C,
                "calibrator_coef": head.calibrator_coef,
                "calibrator_intercept": head.calibrator_intercept,
                "selection": head.selection,
            }
            print(f"{args.backbone} {arm} seed {seed}: target-assisted head fitted", flush=True)
        if arm == "raw":
            arm_heads = arm_heads * len(seeds)
            base = fits[arm][str(seeds[0])]
            fits[arm] = {
                str(seed): {**base, "replicated_deterministic_reference": True}
                for seed in seeds}
        arrays[f"logits__{arm}"] = np.stack([head.logits for head in arm_heads])
        arrays[f"calibrated_prob__{arm}"] = np.stack(
            [head.calibrated for head in arm_heads])
        arrays[f"oof_logits__{arm}"] = np.stack([head.oof_logits for head in arm_heads])

    atomic_npz(predictions_path, **arrays)
    atomic_json({
        "standing": "post-hoc target-assisted diagnostic; not source-only performance",
        "backbone": args.backbone,
        "train_select_calibrate_population": "matched_long only",
        "evaluation_population": "participant-disjoint matched only",
        "n_matched_long": int(fit.sum()),
        "n_matched": int(score.sum()),
        "fold_sha16": sha16(folds),
        "original_manifest_sha256": sha256_file(original_manifest_path),
        "e2_manifest_sha256": sha256_file(e2_manifest_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "predictions_sha256": sha256_file(predictions_path),
        "fits": fits,
        "matched_metrics_computed": False,
    }, fits_path)
    print(f"wrote target-assisted predictions under {out_dir}; matched not scored")


def score_matched(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    record = json.load(open(fits_path))
    assert sha256_file(SCRIPT_PATH) == record["script_sha256"]
    assert sha256_file(predictions_path) == record["predictions_sha256"]
    with np.load(predictions_path, allow_pickle=True) as z:
        assert tuple(map(str, z["arms"])) == ARMS
        y = np.asarray(z["label"], dtype=int)
        matched = np.asarray(z["mask__matched"], dtype=bool)
        assert not np.any(matched & np.asarray(z["mask__matched_long"], dtype=bool))
        probability = {arm: np.asarray(z[f"calibrated_prob__{arm}"], dtype=float)
                       for arm in ARMS}
    yt = y[matched]
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    arms = {}
    for arm in ARMS:
        P = probability[arm][:, matched]
        arms[arm] = {
            "n": int(matched.sum()),
            "mean_auroc": float(np.mean([auroc(yt, row) for row in P])),
            "per_seed_auroc": [auroc(yt, row) for row in P],
            "mean_nll": float(np.mean([logloss(yt, row).mean() for row in P])),
            "per_seed_nll": [float(logloss(yt, row).mean()) for row in P],
            "calibration_per_seed": [calib_diag(yt, row) for row in P],
        }
    comparisons = {}
    for index, (a, b, interpretation) in enumerate(COMPARISONS):
        A, B = probability[a][:, matched], probability[b][:, matched]
        comparisons[f"{a}_minus_{b}"] = {
            "interpretation": interpretation,
            "delta_auroc": paired_hierarchical_ci(
                A, B, yt, auroc, args.bootstrap,
                args.bootstrap_seed + 100 * index),
            "delta_neg_nll": paired_hierarchical_ci(
                A, B, yt, neg_nll, args.bootstrap,
                args.bootstrap_seed + 100 * index + 1),
        }
    atomic_json({
        "standing": record["standing"],
        "backbone": record["backbone"],
        "arms": arms,
        "comparisons": comparisons,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "predictions_sha256": record["predictions_sha256"],
    }, metrics_path)
    print(f"wrote target-assisted matched metrics to {metrics_path}")


def self_test() -> None:
    rng = np.random.RandomState(5)
    n = 240
    y = np.tile([0, 1], n // 2)
    fit = np.zeros(n, bool); fit[:200] = True
    X = rng.normal(size=(n, 12)).astype(np.float32)
    X[:, 0] += y * 0.5
    folds = make_target_folds(y, fit)
    head = fit_target_head(X, y, fit, folds, seed=0)
    assert head.C in C_GRID and np.isfinite(head.calibrated).all()
    assert np.isfinite(head.oof_logits[fit]).all() and np.isnan(head.oof_logits[~fit]).all()
    print("SELF-TEST PASS: target-only folds, one-SE selection, OOF Platt and final readout")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--ast-emb", default="results/ast_embeddings.npz")
    ap.add_argument("--alignment-dir", default="results/alignment")
    ap.add_argument("--e2-alignment-dir", default="results/pairing_followup/e2_alignment_ast")
    ap.add_argument("--backbone", default="AST-6L")
    ap.add_argument("--out-dir", default="results/pairing_followup/e3_target_ast")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--score-tests", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fit:
        fit_all(args, out_dir)
    else:
        score_matched(args, out_dir)


if __name__ == "__main__":
    main()
