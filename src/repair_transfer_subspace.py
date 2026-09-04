"""Exploratory source-fitted transfer repair for frozen metadata alignment.

The repair removes a small, preregistered linear nuisance subspace from each frozen
``correct`` representation and concatenates the result with the frozen raw audio
embedding.  Fitting and test scoring are deliberately separated::

    python src/repair_transfer_subspace.py --self-test
    python src/repair_transfer_subspace.py --preflight [... paths ...]
    python src/repair_transfer_subspace.py --fit-source [... paths ...]
    python src/repair_transfer_subspace.py --score-tests --out-dir ...

``--fit-source`` may create predictions for all fixed cohort rows, but reports only
Standard-validation diagnostics.  Matched/test metrics exist only after the explicit
``--score-tests`` action.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from audit_information_channels import load_probe_table, sha256_file
from eval_metadata_alignment import (
    atomic_json,
    atomic_npz,
    fit_disease_head,
    fit_probe,
    hierarchical_ci,
    load_representation,
    make_validation_folds,
    paired_hierarchical_ci,
    sha16,
    verify_training_manifest,
)


DISEASE_ARMS = (
    "raw",
    "correct",
    "within_label",
    "raw_correct",
    "correct_erased",
    "raw_correct_erased",
)
PROBE_ARMS = ("correct", "correct_erased")
NUISANCE_TARGETS = (
    "recruitment_source",
    "sex_female",
    "age_65plus",
    "long_recording",
    "large_file",
    "loud_recording",
    "clipped",
)
COMPARISONS = (
    ("raw_correct_erased", "raw_correct", "primary repair effect"),
    ("raw_correct_erased", "raw", "repaired fusion versus frozen raw audio"),
    ("correct_erased", "correct", "erased versus original aligned branch"),
)
ERASER_C = 1.0
ERASER_MAX_ITER = 5000
SVD_RELATIVE_TOLERANCE = 1e-8
BOOTSTRAP_SEED = 20260904
SCRIPT_PATH = Path(__file__).resolve()


def _standardize_train(X: np.ndarray, train: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.asarray(X)[train])
    transformed = scaler.transform(np.asarray(X)).astype(np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("non-finite values after train-only nuisance standardisation")
    return transformed, scaler


def fit_linear_eraser(X: np.ndarray, targets: dict, train: np.ndarray,
                      seed: int) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    """Fit seven fixed linear nuisance directions using Standard train only."""
    from sklearn.linear_model import LogisticRegression

    Xs, scaler = _standardize_train(X, train)
    directions, target_audit = [], {}
    for name in NUISANCE_TARGETS:
        values = targets[name]
        observed = values.notna().to_numpy()
        fit = train & observed
        y = values.fillna(0).astype(int).to_numpy()
        levels, counts = np.unique(y[fit], return_counts=True)
        if not np.array_equal(levels, [0, 1]):
            raise ValueError(f"eraser target {name} lacks both classes in source train")
        model = LogisticRegression(
            C=ERASER_C, max_iter=ERASER_MAX_ITER, random_state=seed)
        model.fit(Xs[fit], y[fit])
        w = np.asarray(model.coef_[0], dtype=np.float64)
        norm = float(np.linalg.norm(w))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"invalid nuisance direction for {name}")
        directions.append(w / norm)
        target_audit[name] = {
            "n_train_observed": int(fit.sum()),
            "class_counts": {str(int(k)): int(v) for k, v in zip(levels, counts)},
            "coefficient_norm_before_unit_scaling": norm,
        }

    W = np.stack(directions)
    _, singular_values, vh = np.linalg.svd(W, full_matrices=False)
    tolerance = SVD_RELATIVE_TOLERANCE * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    if rank < 1:
        raise ValueError("nuisance direction matrix has zero numerical rank")
    Q = np.asarray(vh[:rank], dtype=np.float64)
    erased = np.asarray(Xs, dtype=np.float64) - (Xs @ Q.T) @ Q
    max_residual = float(np.max(np.abs(erased @ Q.T)))
    erased = erased.astype(np.float32)
    if not np.isfinite(erased).all() or not np.any(np.linalg.norm(erased, axis=1) > 0):
        raise ValueError("erased representation is non-finite or fully zero")

    audit = {
        "seed": int(seed),
        "nuisance_targets": list(NUISANCE_TARGETS),
        "eraser_C": ERASER_C,
        "eraser_max_iter": ERASER_MAX_ITER,
        "svd_relative_tolerance": SVD_RELATIVE_TOLERANCE,
        "svd_absolute_tolerance": tolerance,
        "singular_values": singular_values.tolist(),
        "rank": rank,
        "max_absolute_post_projection_component": max_residual,
        "input_sha16": sha16(np.asarray(X, dtype=np.float32)),
        "standardized_sha16": sha16(Xs),
        "erased_sha16": sha16(erased),
        "targets": target_audit,
    }
    state = {
        "mean": np.asarray(scaler.mean_, dtype=np.float64),
        "scale": np.asarray(scaler.scale_, dtype=np.float64),
        "basis": Q,
    }
    return erased, audit, state


def _load_state(args):
    seeds = list(args.seeds)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("formal repair requires five unique projector seeds")
    manifest_path = Path(args.alignment_dir) / "manifest.json"
    manifest = verify_training_manifest(manifest_path, seeds, args.expected_epochs)
    data, train, val, tests, targets, target_audit = load_probe_table(args)
    participants = data[UNIT].to_numpy()
    folds = make_validation_folds(data, val)
    state = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "data": data,
        "train": train,
        "val": val,
        "tests": tests,
        "targets": targets,
        "target_audit": target_audit,
        "participants": participants,
        "folds": folds,
    }
    return state


def preflight(args, out_dir: Path) -> None:
    path = out_dir / "preflight.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    state = _load_state(args)
    raw_cache: dict[str, np.ndarray] = {}
    reps = {}
    for arm in ("raw_ast", "correct", "within_label"):
        reps[arm] = {}
        seeds = [args.seeds[0]] if arm == "raw_ast" else args.seeds
        for seed in seeds:
            X = load_representation(
                args, state["participants"], arm, seed, args.variant, raw_cache,
                state["manifest"])
            reps[arm][str(seed)] = {
                "shape": list(X.shape), "sha16": sha16(X),
                "finite": bool(np.isfinite(X).all()),
                "nonzero_rows": int(np.sum(np.linalg.norm(X, axis=1) > 0)),
            }
    atomic_json({
        "standing": "exploratory preregistered Transfer Repair v1",
        "backbone": args.backbone,
        "variant": args.variant,
        "n_cohort": int(len(state["data"])),
        "n_train": int(state["train"].sum()),
        "n_validation": int(state["val"].sum()),
        "n_matched_not_scored": int(state["tests"]["matched"].sum()),
        "n_matched_long_not_scored": int(state["tests"]["matched_long"].sum()),
        "representations": reps,
        "nuisance_targets": list(NUISANCE_TARGETS),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "alignment_manifest_sha256": sha256_file(state["manifest_path"]),
        "audio_embedding_sha256": sha256_file(Path(args.ast_emb)),
        "preflight_pass": True,
        "test_metrics_computed": False,
    }, path)
    print(f"PREFLIGHT PASS: {path}; matched/test metrics not scored", flush=True)


def _validation_source_report(arrays: dict[str, np.ndarray], state: dict) -> dict:
    val, folds = state["val"], state["folds"]
    y = state["data"]["y"].to_numpy()
    disease = {}
    for arm in DISEASE_ARMS:
        P = arrays[f"val_oof_prob__{arm}__covid"][:, val]
        disease[arm] = {
            "n": int(val.sum()),
            "auroc_mean": float(np.mean([auroc(y[val], row) for row in P])),
            "per_seed": [auroc(y[val], row) for row in P],
        }

    nuisance, deltas = {}, {}
    for name in NUISANCE_TARGETS:
        values = state["targets"][name]
        observed = values.notna().to_numpy()
        mask = val & observed & (folds >= 0)
        yt = values.fillna(0).astype(int).to_numpy()[mask]
        nuisance[name] = {}
        for arm in PROBE_ARMS:
            S = arrays[f"score__{arm}__{name}"][:, mask]
            nuisance[name][arm] = {
                "n": int(mask.sum()),
                "auroc_mean": float(np.mean([auroc(yt, row) for row in S])),
                "per_seed": [auroc(yt, row) for row in S],
            }
        deltas[name] = (nuisance[name]["correct_erased"]["auroc_mean"] -
                        nuisance[name]["correct"]["auroc_mean"])
    macro_delta = float(np.mean(list(deltas.values())))
    n_not_increased = int(sum(v <= 0.01 for v in deltas.values()))
    return {
        "standing": "source-only technical smoke; no matched/test metric was read",
        "disease_validation_diagnostic": disease,
        "nuisance_validation": nuisance,
        "nuisance_delta_erased_minus_correct": deltas,
        "macro_nuisance_delta": macro_delta,
        "n_targets_not_increased_over_0_01": n_not_increased,
        "technical_gate": {
            "macro_delta_below_zero": bool(macro_delta < 0),
            "at_least_5_of_7_not_increased_over_0_01": bool(n_not_increased >= 5),
            "pass": bool(macro_delta < 0 and n_not_increased >= 5),
        },
    }


def fit_source(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    erasers_path = out_dir / "erasers.npz"
    fits_path = out_dir / "fits.json"
    source_path = out_dir / "source_smoke.json"
    for path in (predictions_path, erasers_path, fits_path, source_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    state = _load_state(args)
    data, train, val, folds = (state["data"], state["train"], state["val"],
                               state["folds"])
    participants = state["participants"]
    seeds = list(args.seeds)
    raw_cache: dict[str, np.ndarray] = {}
    raw = load_representation(
        args, participants, "raw_ast", seeds[0], args.variant, raw_cache,
        state["manifest"])

    disease_prob = {arm: [] for arm in DISEASE_ARMS}
    disease_oof = {arm: [] for arm in DISEASE_ARMS}
    probe_score = {
        arm: {name: [] for name in NUISANCE_TARGETS} for arm in PROBE_ARMS}
    fits: dict[str, dict] = {arm: {} for arm in DISEASE_ARMS}
    eraser_arrays: dict[str, np.ndarray] = {}
    eraser_audits = {}

    raw_head = fit_disease_head(raw, data["y"].to_numpy(), train, val, folds, seeds[0])
    for _ in seeds:
        disease_prob["raw"].append(raw_head.calibrated.astype(np.float32))
        disease_oof["raw"].append(raw_head.val_oof_calibrated.astype(np.float32))
    for seed in seeds:
        fits["raw"][str(seed)] = {
            "replicated_deterministic_reference": True,
            "C": raw_head.C,
            "selection": raw_head.selection,
            "calibrator_coef": raw_head.calibrator_coef,
            "calibrator_intercept": raw_head.calibrator_intercept,
        }

    for seed in seeds:
        correct = load_representation(
            args, participants, "correct", seed, args.variant, raw_cache,
            state["manifest"])
        within = load_representation(
            args, participants, "within_label", seed, args.variant, raw_cache,
            state["manifest"])
        erased, eraser_audit, eraser_state = fit_linear_eraser(
            correct, state["targets"], train, seed)
        eraser_audits[str(seed)] = eraser_audit
        for key, value in eraser_state.items():
            eraser_arrays[f"seed{seed}__{key}"] = value

        representations = {
            "correct": correct,
            "within_label": within,
            "raw_correct": np.concatenate([raw, correct], axis=1),
            "correct_erased": erased,
            "raw_correct_erased": np.concatenate([raw, erased], axis=1),
        }
        for arm, representation in representations.items():
            head = fit_disease_head(
                representation, data["y"].to_numpy(), train, val, folds, seed)
            disease_prob[arm].append(head.calibrated.astype(np.float32))
            disease_oof[arm].append(head.val_oof_calibrated.astype(np.float32))
            fits[arm][str(seed)] = {
                "representation_sha16": sha16(representation),
                "C": head.C,
                "selection": head.selection,
                "calibrator_coef": head.calibrator_coef,
                "calibrator_intercept": head.calibrator_intercept,
            }
            print(f"{args.backbone} seed {seed}: fitted disease head {arm}", flush=True)

        for arm, representation in (("correct", correct),
                                    ("correct_erased", erased)):
            fits[arm][str(seed)]["nuisance_probes"] = {}
            for name in NUISANCE_TARGETS:
                logits, C, selection, observed = fit_probe(
                    representation, state["targets"][name], train, folds, seed)
                if not np.array_equal(observed, state["targets"][name].notna().to_numpy()):
                    raise AssertionError(f"observed mask mismatch for {name}")
                probe_score[arm][name].append(logits.astype(np.float32))
                fits[arm][str(seed)]["nuisance_probes"][name] = {
                    "C": C, "selection": selection}
            print(f"{args.backbone} seed {seed}: fitted seven probes for {arm}", flush=True)

        del correct, within, erased, representations

    arrays: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "disease_arms": np.asarray(DISEASE_ARMS),
        "probe_arms": np.asarray(PROBE_ARMS),
        "nuisance_targets": np.asarray(NUISANCE_TARGETS),
        "label__covid": data["y"].to_numpy(dtype=np.int8),
        "mask__standard": state["tests"]["standard"],
        "mask__matched": state["tests"]["matched"],
        "mask__matched_long": state["tests"]["matched_long"],
    }
    for name in NUISANCE_TARGETS:
        values = state["targets"][name]
        arrays[f"label__{name}"] = values.fillna(0).astype(int).to_numpy(dtype=np.int8)
        arrays[f"observed__{name}"] = values.notna().to_numpy()
    for arm in DISEASE_ARMS:
        arrays[f"calibrated_prob__{arm}__covid"] = np.stack(disease_prob[arm])
        arrays[f"val_oof_prob__{arm}__covid"] = np.stack(disease_oof[arm])
    for arm in PROBE_ARMS:
        for name in NUISANCE_TARGETS:
            arrays[f"score__{arm}__{name}"] = np.stack(probe_score[arm][name])

    source_report = _validation_source_report(arrays, state)
    atomic_npz(erasers_path, **eraser_arrays)
    atomic_npz(predictions_path, **arrays)
    fit_record = {
        "standing": "exploratory preregistered Transfer Repair v1",
        "backbone": args.backbone,
        "variant": args.variant,
        "seeds": seeds,
        "nuisance_targets": list(NUISANCE_TARGETS),
        "eraser_audits": eraser_audits,
        "fits": fits,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "alignment_manifest_sha256": sha256_file(state["manifest_path"]),
        "predictions_sha256": sha256_file(predictions_path),
        "erasers_sha256": sha256_file(erasers_path),
        "source_smoke_pass": source_report["technical_gate"]["pass"],
        "test_metrics_computed": False,
    }
    atomic_json(source_report, source_path)
    atomic_json(fit_record, fits_path)
    print(f"wrote source-only repair fit under {out_dir}", flush=True)
    print(json.dumps(source_report["technical_gate"], indent=2), flush=True)


def score_tests(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    if not predictions_path.is_file() or not fits_path.is_file():
        raise FileNotFoundError("--fit-source must finish before --score-tests")
    with open(fits_path) as stream:
        record = json.load(stream)
    if not record.get("source_smoke_pass"):
        raise RuntimeError("source-only technical gate did not pass")
    if sha256_file(SCRIPT_PATH) != record["script_sha256"]:
        raise RuntimeError("script changed between source fitting and test scoring")
    if sha256_file(predictions_path) != record["predictions_sha256"]:
        raise RuntimeError("prediction archive changed after fitting")

    with np.load(predictions_path, allow_pickle=True) as z:
        y = np.asarray(z["label__covid"], dtype=int)
        masks = {name: np.asarray(z[f"mask__{name}"], dtype=bool)
                 for name in ("standard", "matched", "matched_long")}
        P = {arm: np.asarray(z[f"calibrated_prob__{arm}__covid"], dtype=float)
             for arm in DISEASE_ARMS}
        nuisance_y = {name: np.asarray(z[f"label__{name}"], dtype=int)
                      for name in NUISANCE_TARGETS}
        nuisance_obs = {name: np.asarray(z[f"observed__{name}"], dtype=bool)
                        for name in NUISANCE_TARGETS}
        nuisance_score = {
            (arm, name): np.asarray(z[f"score__{arm}__{name}"], dtype=float)
            for arm in PROBE_ARMS for name in NUISANCE_TARGETS}

    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    brier = lambda yy, pp: -float(np.mean((np.asarray(pp) - np.asarray(yy)) ** 2))
    result = {"arms": {}, "comparisons": {}, "nuisance_probes": {}}
    for arm in DISEASE_ARMS:
        result["arms"][arm] = {}
        for i, (name, mask) in enumerate(masks.items()):
            pred, yt = P[arm][:, mask], y[mask]
            auc, auc_ci = hierarchical_ci(
                pred, yt, auroc, args.bootstrap, args.bootstrap_seed + 100 * i)
            nnll, nnll_ci = hierarchical_ci(
                pred, yt, neg_nll, args.bootstrap, args.bootstrap_seed + 100 * i + 1)
            nbrier, nbrier_ci = hierarchical_ci(
                pred, yt, brier, args.bootstrap, args.bootstrap_seed + 100 * i + 2)
            result["arms"][arm][name] = {
                "n": int(mask.sum()), "auroc": auc, "auroc_ci": auc_ci,
                "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                "negative_brier": nbrier, "negative_brier_ci": nbrier_ci,
                "per_seed_auroc": [auroc(yt, row) for row in pred],
                "per_seed_calibration": [calib_diag(yt, row) for row in pred],
            }
    for comparison_index, (a, b, meaning) in enumerate(COMPARISONS):
        key = f"{a}_minus_{b}"
        result["comparisons"][key] = {"interpretation": meaning}
        for i, (name, mask) in enumerate(masks.items()):
            offset = args.bootstrap_seed + 10000 + 1000 * comparison_index + 10 * i
            result["comparisons"][key][name] = {
                "delta_auroc": paired_hierarchical_ci(
                    P[a][:, mask], P[b][:, mask], y[mask], auroc,
                    args.bootstrap, offset),
                "delta_neg_nll": paired_hierarchical_ci(
                    P[a][:, mask], P[b][:, mask], y[mask], neg_nll,
                    args.bootstrap, offset + 1),
                "delta_negative_brier": paired_hierarchical_ci(
                    P[a][:, mask], P[b][:, mask], y[mask], brier,
                    args.bootstrap, offset + 2),
            }

    for target_index, target in enumerate(NUISANCE_TARGETS):
        result["nuisance_probes"][target] = {}
        for i, (name, base_mask) in enumerate(masks.items()):
            mask = base_mask & nuisance_obs[target]
            yt = nuisance_y[target][mask]
            a = nuisance_score[("correct_erased", target)][:, mask]
            b = nuisance_score[("correct", target)][:, mask]
            result["nuisance_probes"][target][name] = {
                "correct_erased_minus_correct_delta_auroc": paired_hierarchical_ci(
                    a, b, yt, auroc, args.bootstrap,
                    args.bootstrap_seed + 50000 + 100 * target_index + i)
            }

    output = {
        "standing": record["standing"],
        "backbone": record["backbone"],
        "variant": record["variant"],
        "primary_result_path": (
            "metrics.comparisons.raw_correct_erased_minus_raw_correct."
            "matched.delta_auroc"),
        "success_threshold": (
            "paired matched delta AUROC >= +0.01 with 95% CI lower > 0; also "
            "requires preservation and nuisance/calibration conditions in prereg"),
        "predictions_sha256": record["predictions_sha256"],
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metrics": result,
    }
    atomic_json(output, metrics_path)
    print(f"wrote explicitly unlocked repair metrics to {metrics_path}", flush=True)


def self_test() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(91)
    n, d = 800, 24
    train = np.arange(n) < 500
    latent = rng.normal(size=(n, len(NUISANCE_TARGETS)))
    X = rng.normal(scale=0.25, size=(n, d))
    X[:, :len(NUISANCE_TARGETS)] += latent
    targets = {}
    for k, name in enumerate(NUISANCE_TARGETS):
        import pandas as pd
        targets[name] = pd.Series((latent[:, k] > 0).astype(int))

    erased1, audit1, state1 = fit_linear_eraser(X, targets, train, seed=0)
    erased2, audit2, state2 = fit_linear_eraser(X, targets, train, seed=0)
    assert np.array_equal(erased1, erased2)
    assert audit1["erased_sha16"] == audit2["erased_sha16"]
    assert np.array_equal(state1["basis"], state2["basis"])
    assert np.max(np.abs(erased1 @ state1["basis"].T)) < 2e-5
    held = ~train
    before, after = [], []
    Xs, _ = _standardize_train(X, train)
    for name in NUISANCE_TARGETS:
        y = targets[name].to_numpy()
        m0 = LogisticRegression(C=1.0, max_iter=5000).fit(Xs[train], y[train])
        m1 = LogisticRegression(C=1.0, max_iter=5000).fit(erased1[train], y[train])
        before.append(roc_auc_score(y[held], m0.decision_function(Xs[held])))
        after.append(roc_auc_score(y[held], m1.decision_function(erased1[held])))
    assert np.mean(after) < np.mean(before) - 0.2
    assert audit1["rank"] <= len(NUISANCE_TARGETS)

    # Verify durable output helpers and no test dependency on eraser fitting.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "eraser.npz"
        atomic_npz(path, basis=state1["basis"], mean=state1["mean"], scale=state1["scale"])
        with np.load(path) as z:
            assert np.array_equal(z["basis"], state1["basis"])
    print("SELF-TEST PASS: deterministic train-only subspace fit, orthogonal erasure, "
          "and held-out nuisance reduction", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--artefacts", default="results/artefact_features.csv")
    parser.add_argument("--audio-emb", dest="ast_emb", default="results/ast_embeddings.npz")
    parser.add_argument("--alignment-dir", default="results/alignment")
    parser.add_argument("--backbone", default="AST-6L")
    parser.add_argument("--variant", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--out-dir", default="results/transfer_repair_v1_ast")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected-epochs", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--fit-source", action="store_true")
    mode.add_argument("--score-tests", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(args, out_dir)
    elif args.fit_source:
        fit_source(args, out_dir)
    else:
        score_tests(args, out_dir)


if __name__ == "__main__":
    main()
