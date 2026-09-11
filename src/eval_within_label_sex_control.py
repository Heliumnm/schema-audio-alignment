"""Evaluate the preregistered UKCOVID W_{y,s} pairing control.

The fit stage reuses the already-frozen Raw/Correct/Within/Global prediction archive and
fits downstream heads only for the new ``within_label_sex`` representation.  It writes all
per-participant predictions before the explicitly unlocked score stage reads matched
endpoints.  The score stage reports the two preregistered new contrasts on matched only:

    Correct - W_{y,s}
    W_{y,s} - Within-label

Existing C-W and C-Raw results are referenced by hash rather than silently recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from audit_information_channels import (TARGET_GROUPS, TARGETS, load_probe_table,
                                        sha256_file)
from eval_metadata_alignment import (atomic_json, atomic_npz, fit_disease_head,
                                     fit_probe, hierarchical_ci,
                                     make_validation_folds, paired_hierarchical_ci,
                                     sha16)


NEW_ARM = "within_label_sex"
OLD_ARMS = ("raw", "correct", "within_label", "global")
ARMS = (*OLD_ARMS, NEW_ARM)
COMPARISONS = (
    ("correct", NEW_ARM, "exact-pair increment after preserving label and recorded sex"),
    (NEW_ARM, "within_label", "increment from preserving recorded sex in shuffled pairs"),
)
SCRIPT_PATH = Path(__file__).resolve()


def verify_e2_manifest(path: Path, seeds: list[int], epochs: int) -> dict:
    with open(path) as stream:
        manifest = json.load(stream)
    assert manifest["arm"] == NEW_ARM
    assert manifest["n_train"] == 20_714 and manifest["batch"] == 64
    assert manifest["epochs"] == epochs
    assert manifest["matched_or_test_labels_read"] is False
    assert set(seeds).issubset(set(map(int, manifest["seeds"])))
    assert set(manifest["runs"]) == {f"{NEW_ARM}_seed{s}" for s in manifest["seeds"]}
    for seed in seeds:
        run = manifest["runs"][f"{NEW_ARM}_seed{seed}"]
        reference = run["reference"]
        assert reference["reference_run"] == f"within_label_seed{seed}"
        assert reference["reference_init_hash"] == run["init_hash"]
        assert reference["reference_batch_order_hash"] == run["batch_order_hash"]
    return manifest


def load_e2_representation(directory: Path, manifest: dict, participants: np.ndarray,
                           seed: int) -> np.ndarray:
    path = directory / f"repr_{NEW_ARM}_seed{seed}.npz"
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants)
        assert str(np.asarray(z["arm"]).item()) == NEW_ARM
        assert int(np.asarray(z["seed"]).item()) == seed
        assert int(np.asarray(z["epochs"]).item()) == manifest["epochs"]
        assert json.loads(str(np.asarray(z["input_hashes"]).item())) == manifest["input_hashes"]
        assert str(np.asarray(z["source_commit"]).item()) == manifest["source_commit"]
        raw = np.asarray(z["raw"], dtype=np.float32)
        normalized = np.asarray(z["normalized"], dtype=np.float32)
    expected = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    assert np.allclose(expected, normalized, rtol=2e-5, atol=2e-6)
    run = manifest["runs"][f"{NEW_ARM}_seed{seed}"]
    assert sha16(raw) == run["repr_raw_hash"]
    assert sha16(normalized) == run["repr_norm_hash"]
    assert np.isfinite(raw).all() and np.all(np.linalg.norm(raw, axis=1) > 0)
    return raw


def load_old_archive(path: Path, participants: np.ndarray, seeds: list[int],
                     state: dict) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants)
        assert np.array_equal(z["seeds"], np.asarray(seeds))
        assert set(map(str, z["arms"])) == set(OLD_ARMS)
        assert tuple(map(str, z["targets"])) == TARGETS
        arrays = {key: np.asarray(z[key]) for key in z.files}
    for split in ("standard", "matched", "matched_long"):
        assert np.array_equal(arrays[f"mask__{split}"], state["tests"][split])
    for target, values in state["targets"].items():
        assert np.array_equal(arrays[f"observed__{target}"], values.notna().to_numpy())
        assert np.array_equal(arrays[f"label__{target}"],
                              values.fillna(0).astype(int).to_numpy())
    return arrays


def fit_new_arm(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    if predictions_path.exists() or fits_path.exists():
        raise FileExistsError("refusing to overwrite an E2 fit archive")

    seeds = list(args.seeds)
    manifest_path = Path(args.e2_alignment_dir) / "manifest.json"
    manifest = verify_e2_manifest(manifest_path, seeds, args.expected_epochs)
    data, train, val, tests, targets, target_audit = load_probe_table(args)
    participants = data[UNIT].to_numpy()
    folds = make_validation_folds(data, val)
    old_path = Path(args.original_predictions)
    arrays = load_old_archive(old_path, participants, seeds, {
        "tests": tests, "targets": targets})
    arrays["arms"] = np.asarray(ARMS)

    scores: dict[str, list[np.ndarray]] = {target: [] for target in TARGETS}
    disease_prob: list[np.ndarray] = []
    disease_oof: list[np.ndarray] = []
    fits: dict[str, dict] = {}
    for seed in seeds:
        representation = load_e2_representation(
            Path(args.e2_alignment_dir), manifest, participants, seed)
        fits[str(seed)] = {"representation_sha16": sha16(representation), "targets": {}}
        for target_name in TARGETS:
            values = targets[target_name]
            if target_name == "covid":
                head = fit_disease_head(
                    representation, data["y"].to_numpy(), train, val, folds, seed)
                scores[target_name].append(head.logits.astype(np.float32))
                disease_prob.append(head.calibrated.astype(np.float32))
                disease_oof.append(head.val_oof_calibrated.astype(np.float32))
                fits[str(seed)]["targets"][target_name] = {
                    "C": head.C,
                    "calibrator_coef": head.calibrator_coef,
                    "calibrator_intercept": head.calibrator_intercept,
                    "selection": head.selection,
                }
            else:
                logits, c_value, selection, observed = fit_probe(
                    representation, values, train, folds, seed)
                assert np.array_equal(observed, values.notna().to_numpy())
                scores[target_name].append(logits.astype(np.float32))
                fits[str(seed)]["targets"][target_name] = {
                    "C": c_value, "selection": selection}
            print(f"{args.backbone} {NEW_ARM} seed {seed}: fitted {target_name}", flush=True)

    for target_name in TARGETS:
        arrays[f"score__{NEW_ARM}__{target_name}"] = np.stack(scores[target_name])
    arrays[f"calibrated_prob__{NEW_ARM}__covid"] = np.stack(disease_prob)
    arrays[f"val_oof_prob__{NEW_ARM}__covid"] = np.stack(disease_oof)
    atomic_npz(predictions_path, **arrays)
    record = {
        "standing": "post-hoc preregistered W_{y,s} diagnostic",
        "backbone": args.backbone,
        "variant": "raw",
        "seeds": seeds,
        "target_groups": TARGET_GROUPS,
        "target_audit": target_audit,
        "validation_fold_sha16": sha16(folds),
        "e2_manifest_sha256": sha256_file(manifest_path),
        "original_predictions": str(old_path),
        "original_predictions_sha256": sha256_file(old_path),
        "predictions_sha256": sha256_file(predictions_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "fits": fits,
        "matched_metrics_computed": False,
    }
    atomic_json(record, fits_path)
    print(f"wrote E2 predictions and fits to {out_dir}; matched metrics not read")


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
        matched = np.asarray(z["mask__matched"], dtype=bool)
        labels = {target: np.asarray(z[f"label__{target}"], dtype=int)
                  for target in TARGETS}
        observed = {target: np.asarray(z[f"observed__{target}"], dtype=bool)
                    for target in TARGETS}
        score = {(arm, target): np.asarray(z[f"score__{arm}__{target}"], dtype=float)
                 for arm in ARMS for target in TARGETS}
        covid_prob = {
            arm: np.asarray(z[f"calibrated_prob__{arm}__covid"], dtype=float)
            for arm in ARMS}

    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    result = {"arms": {}, "comparisons": {}}
    for target_index, target_name in enumerate(TARGETS):
        mask = matched & observed[target_name]
        y = labels[target_name][mask]
        value = covid_prob[NEW_ARM] if target_name == "covid" else score[(NEW_ARM, target_name)]
        auc, ci = hierarchical_ci(
            value[:, mask], y, auroc, args.bootstrap,
            args.bootstrap_seed + 1000 * target_index)
        arm_entry = {"n": int(mask.sum()), "auroc": auc, "auroc_ci": ci,
                     "per_seed_auroc": [auroc(y, row[mask]) for row in value]}
        if target_name == "covid":
            nnll, nnll_ci = hierarchical_ci(
                value[:, mask], y, neg_nll, args.bootstrap,
                args.bootstrap_seed + 1000 * target_index + 1)
            arm_entry.update({
                "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                "per_seed_nll": [float(logloss(y, row[mask]).mean()) for row in value],
                "calibration_per_seed": [calib_diag(y, row[mask]) for row in value],
            })
        result["arms"][target_name] = {NEW_ARM: arm_entry}
        result["comparisons"][target_name] = {}
        for comparison_index, (a, b, interpretation) in enumerate(COMPARISONS):
            av = covid_prob[a] if target_name == "covid" else score[(a, target_name)]
            bv = covid_prob[b] if target_name == "covid" else score[(b, target_name)]
            entry = {
                "interpretation": interpretation,
                "matched": {
                    "delta_auroc": paired_hierarchical_ci(
                        av[:, mask], bv[:, mask], y, auroc, args.bootstrap,
                        args.bootstrap_seed + 20_000 + 1000 * target_index +
                        100 * comparison_index)}}
            if target_name == "covid":
                entry["matched"]["delta_neg_nll"] = paired_hierarchical_ci(
                    av[:, mask], bv[:, mask], y, neg_nll, args.bootstrap,
                    args.bootstrap_seed + 30_000 + 1000 * target_index +
                    100 * comparison_index)
            result["comparisons"][target_name][f"{a}_minus_{b}"] = entry

    atomic_json({
        "standing": record["standing"],
        "backbone": record["backbone"],
        "endpoint": "matched only; explicitly unlocked after durable predictions",
        "comparisons": [f"{a}_minus_{b}" for a, b, _ in COMPARISONS],
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "original_results_are_referenced_not_recomputed": True,
        "original_predictions_sha256": record["original_predictions_sha256"],
        "predictions_sha256": record["predictions_sha256"],
        "metrics": result,
    }, metrics_path)
    print(f"wrote explicitly unlocked E2 matched metrics to {metrics_path}")


def self_test() -> None:
    rng = np.random.RandomState(41)
    y = np.tile([0, 1], 80)
    base = rng.normal(size=(5, len(y)))
    better = base + 0.2 * (2 * y - 1)[None, :]
    out = paired_hierarchical_ci(better, base, y, auroc, 100, 9)
    assert out["observed"] > 0 and len(out["per_seed"]) == 5
    assert set(a for pair in COMPARISONS for a in pair[:2]).issubset(ARMS)
    print("SELF-TEST PASS: E2 arm/comparison definitions and paired bootstrap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--artefacts", default="results/artefact_features.csv")
    ap.add_argument("--e2-alignment-dir", required=False,
                    default="results/pairing_followup/e2_alignment_ast")
    ap.add_argument("--original-predictions", required=False,
                    default="results/information_channels_ast/predictions.npz")
    ap.add_argument("--backbone", default="AST-6L")
    ap.add_argument("--out-dir", default="results/pairing_followup/e2_eval_ast")
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
        fit_new_arm(args, out_dir)
    else:
        score_matched(args, out_dir)


if __name__ == "__main__":
    main()
