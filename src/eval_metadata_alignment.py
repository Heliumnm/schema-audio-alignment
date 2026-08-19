"""Frozen downstream evaluation for the formal metadata-alignment experiment.

This script is intentionally separate from ``train_metadata_alignment.py``.  Projector
training never sees a disease-validation or test score; this file consumes only completed
epoch-500 representations and fits the downstream heads afterwards.

Protocol
--------
* One disease head is fitted independently for every projector arm and projector seed.
* C is selected with the frozen one-standard-error rule on the full Standard validation.
* A five-fold out-of-fold Platt score is retained for validation diagnostics; one final
  Platt calibrator is fitted on all of Standard val and is the only calibrator applied to
  Standard, matched and matched_long tests.
* ``raw_ast`` is a deterministic reference.  Its predictions are copied across the five
  projector seeds so every paired contrast has the same seed axis.
* Raw post-ReLU projector output is primary.  L2-normalised output is a declared
  sensitivity analysis.
* Confidence intervals use a paired participant x seed hierarchical bootstrap.  The
  observed statistic and every individual seed's signed difference are also saved.

The test sets require an explicit ``--evaluate-tests`` switch.  There is deliberately no
implicit data-running mode: without that switch the only allowed action is ``--self-test``,
which uses synthetic arrays and never opens UKCOVID data.

Example (only after all 15 epoch-500 representation files are complete)::

    python src/eval_metadata_alignment.py --evaluate-tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import (C_GRID, FOLD_SEED, N_FOLDS, UNIT, auroc,
                                calib_diag, logloss)


ARMS = ("raw_ast", "correct", "within_label", "global")
PROJECTOR_ARMS = ("correct", "within_label", "global")
VARIANTS = ("raw", "normalized")
TEST_SPECS = {
    "standard": ("splits", "test"),
    "matched": ("in_matched_rebalanced_test", True),
    "matched_long": ("in_matched_rebalanced_long_test", True),
}
COMPARISONS = (
    ("correct", "within_label", "individual_correspondence_primary"),
    ("correct", "raw_ast", "vs_frozen_ast_key_secondary"),
    ("within_label", "global", "label_level_cooccurrence"),
    ("correct", "global", "total_alignment_effect"),
)
PROBE_TARGETS = (
    "recruitment_source",
    "gender",
    "age_65plus",
    "symptom_cough_any",
    "symptom_none",
)


def sha16(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"cannot JSON-encode {type(x)}")


def mask_from_spec(d: pd.DataFrame, spec: tuple[str, object]) -> np.ndarray:
    col, val = spec
    return (d[col] == val).to_numpy()


def make_validation_folds(d: pd.DataFrame, val: np.ndarray) -> np.ndarray:
    """Frozen five folds of full Standard val, stratified by label x source."""
    from sklearn.model_selection import StratifiedKFold

    idx = np.where(val)[0]
    strat = (d.loc[idx, "y"].astype(str) + "|" +
             d.loc[idx, "recruitment_source"].fillna("NA").astype(str))
    counts = strat.value_counts()
    strat = strat.map(lambda s: s if counts[s] >= N_FOLDS
                      else s.split("|")[0] + "|RARE")
    folds = np.full(len(d), -1, dtype=np.int16)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    for k, (_, held) in enumerate(skf.split(idx, strat)):
        folds[idx[held]] = k
    assert np.all(folds[val] >= 0) and np.all(folds[~val] == -1)
    return folds


@dataclass
class FittedHead:
    C: float
    logits: np.ndarray
    calibrated: np.ndarray
    val_oof_calibrated: np.ndarray
    calibrator_coef: float
    calibrator_intercept: float
    selection: dict


def _standardize(X: np.ndarray, train: np.ndarray):
    """Fit feature scaling on Standard train only and transform the fixed cohort."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X[train])
    return scaler.transform(X)


def _fit_logistic_logits(Xs: np.ndarray, y: np.ndarray, train: np.ndarray,
                         C: float, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    model.fit(Xs[train], y[train])
    out = model.decision_function(Xs)
    assert out.shape == (len(y),) and np.isfinite(out).all()
    return out.astype(np.float64, copy=False)


def select_C_and_logits(X: np.ndarray, y: np.ndarray, train: np.ndarray,
                        folds: np.ndarray, seed: int) -> tuple[float, dict, np.ndarray]:
    """One-SE C selection, matching the frozen v3 validation protocol.

    Every candidate is trained on Standard train.  The full Standard validation is split
    into fixed strata-preserving folds only to estimate the fold-to-fold SE; test data are
    not involved.  Scaling is fitted once on train and shared by all candidates.
    """
    Xs = _standardize(np.asarray(X, dtype=np.float32), train)
    means, ses, per_fold, candidate_logits = {}, {}, {}, {}
    for C in C_GRID:
        logits = _fit_logistic_logits(Xs, y, train, C, seed)
        candidate_logits[C] = logits
        scores = [auroc(y[folds == k], logits[folds == k]) for k in range(N_FOLDS)]
        means[C] = float(np.mean(scores))
        ses[C] = float(np.std(scores, ddof=1) / np.sqrt(N_FOLDS))
        per_fold[C] = [float(v) for v in scores]
    best = max(C_GRID, key=lambda c: means[c])
    threshold = means[best] - ses[best]
    chosen = min(c for c in C_GRID if means[c] >= threshold)
    info = {
        "mean_auroc": {str(c): means[c] for c in C_GRID},
        "se": {str(c): ses[c] for c in C_GRID},
        "per_fold": {str(c): per_fold[c] for c in C_GRID},
        "best_C": float(best),
        "one_se_threshold": float(threshold),
        "chosen_C": float(chosen),
    }
    return float(chosen), info, candidate_logits[chosen]


def _fit_platt_from_logits(logits: np.ndarray, y: np.ndarray, fit: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(logits[fit, None], y[fit])
    return model


def _apply_platt_logits(model, logits: np.ndarray) -> np.ndarray:
    return model.predict_proba(logits[:, None])[:, 1].astype(np.float64, copy=False)


def fit_disease_head(X: np.ndarray, y: np.ndarray, train: np.ndarray,
                     val: np.ndarray, folds: np.ndarray, seed: int) -> FittedHead:
    C, selection, logits = select_C_and_logits(X, y, train, folds, seed)

    # Out-of-fold calibration is a validation diagnostic only.  The final calibrator below
    # is refitted once on all Standard val before any test score is read.
    oof = np.full(len(y), np.nan, dtype=np.float64)
    for k in range(N_FOLDS):
        fit = val & (folds != k)
        held = folds == k
        cal_k = _fit_platt_from_logits(logits, y, fit)
        oof[held] = _apply_platt_logits(cal_k, logits[held])
    assert np.isfinite(oof[val]).all() and np.isnan(oof[~val]).all()

    final_cal = _fit_platt_from_logits(logits, y, val)
    calibrated = _apply_platt_logits(final_cal, logits)
    return FittedHead(
        C=C,
        logits=logits,
        calibrated=calibrated,
        val_oof_calibrated=oof,
        calibrator_coef=float(final_cal.coef_[0, 0]),
        calibrator_intercept=float(final_cal.intercept_[0]),
        selection=selection,
    )


def probe_labels(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Frozen binary probe definitions; missing labels remain missing and are dropped."""
    return {
        "recruitment_source": (d["recruitment_source"] == "Test and Trace").where(
            d["recruitment_source"].notna()),
        "gender": (d["gender"] == "Female").where(d["gender"].notna()),
        "age_65plus": (d["age"] == "65+").where(d["age"].notna()),
        "symptom_cough_any": d["symptom_cough_any"].where(
            d["symptom_cough_any"].notna()),
        "symptom_none": d["symptom_none"].where(d["symptom_none"].notna()),
    }


def fit_probe(X: np.ndarray, target: pd.Series, train: np.ndarray,
              folds: np.ndarray, seed: int) -> tuple[np.ndarray, float, dict, np.ndarray]:
    """Same standardised logistic head and one-SE rule; missing targets are excluded."""
    observed = target.notna().to_numpy()
    yt = target.fillna(0).astype(int).to_numpy()
    train_t = train & observed
    folds_t = np.where(observed, folds, -1)
    # select_C expects arrays whose masks share X's row axis; unlabelled rows simply never
    # enter train or validation folds.
    C, info, logits = select_C_and_logits(X, yt, train_t, folds_t, seed)
    return logits, C, info, observed


def metric_mean(P: np.ndarray, y: np.ndarray, fn: Callable) -> float:
    return float(np.mean([fn(y, P[k]) for k in range(P.shape[0])]))


def hierarchical_ci(P: np.ndarray, y: np.ndarray, fn: Callable,
                    boot: int, seed: int) -> tuple[float, list[float]]:
    """Participant x seed bootstrap for one arm."""
    observed = metric_mean(P, y, fn)
    rng, values = np.random.RandomState(seed), []
    n_seeds, n = P.shape
    for _ in range(boot):
        ii = rng.choice(n, n, replace=True)
        if np.unique(y[ii]).size < 2:
            continue
        ss = rng.choice(n_seeds, n_seeds, replace=True)
        values.append(float(np.mean([fn(y[ii], P[s, ii]) for s in ss])))
    assert values, "bootstrap produced no valid replicate"
    lo, hi = np.percentile(values, [2.5, 97.5])
    return observed, [float(lo), float(hi)]


def paired_hierarchical_ci(A: np.ndarray, B: np.ndarray, y: np.ndarray, fn: Callable,
                           boot: int, seed: int) -> dict:
    """Paired participant x seed bootstrap, preserving arm and seed pairing."""
    assert A.shape == B.shape and A.shape[1] == len(y)
    seed_delta = np.asarray([fn(y, A[k]) - fn(y, B[k])
                             for k in range(A.shape[0])], dtype=float)
    observed = float(seed_delta.mean())
    rng, values = np.random.RandomState(seed), []
    n_seeds, n = A.shape
    for _ in range(boot):
        ii = rng.choice(n, n, replace=True)
        if np.unique(y[ii]).size < 2:
            continue
        ss = rng.choice(n_seeds, n_seeds, replace=True)
        values.append(float(np.mean([
            fn(y[ii], A[s, ii]) - fn(y[ii], B[s, ii]) for s in ss
        ])))
    assert values, "bootstrap produced no valid replicate"
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "observed": observed,
        "ci": [float(lo), float(hi)],
        "per_seed": seed_delta.tolist(),
        "seed_signs": np.sign(seed_delta).astype(int).tolist(),
        "n_seeds": int(n_seeds),
        "n_participants": int(n),
    }


def load_cohort(args) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cohort = pd.read_csv(args.cohort)
    meta = pd.read_csv(Path(args.data) / "participant_metadata.csv", low_memory=False)
    splits = pd.read_csv(Path(args.data) / "train_test_splits.csv", low_memory=False)
    keep_meta = [UNIT, "covid_test_result", "recruitment_source", "age", "gender",
                 "symptom_cough_any", "symptom_none"]
    keep_splits = [UNIT, "splits", "in_matched_rebalanced_test",
                   "in_matched_rebalanced_long_test"]
    d = (cohort[[UNIT]].merge(meta[keep_meta], on=UNIT, validate="one_to_one")
         .merge(splits[keep_splits], on=UNIT, validate="one_to_one"))
    order = {pid: i for i, pid in enumerate(cohort[UNIT])}
    d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
    assert np.array_equal(d[UNIT].to_numpy(), cohort[UNIT].to_numpy())
    d["y"] = (d["covid_test_result"] == "Positive").astype(int)

    train = (d["splits"] == "train").to_numpy()
    val = (d["splits"] == "val").to_numpy()
    tests = {name: mask_from_spec(d, spec) for name, spec in TEST_SPECS.items()}
    assert not np.any(train & val)
    for name, m in tests.items():
        assert not np.any(m & train) and not np.any(m & val), \
            f"split leak: {name} overlaps Standard train/val"
    assert not np.any(tests["matched"] & tests["matched_long"]), \
        "matched and matched_long must be participant-disjoint"
    return d, train, val, tests


def verify_training_manifest(path: Path, seeds: list[int], expected_epochs: int) -> dict:
    with open(path) as f:
        manifest = json.load(f)
    assert int(manifest["epochs"]) == expected_epochs
    available = set(map(int, manifest["seeds"]))
    assert set(seeds).issubset(available), \
        f"requested seeds {seeds} are not a subset of formal seeds {sorted(available)}"
    for seed in seeds:
        for arm in PROJECTOR_ARMS:
            key = f"{arm}_seed{seed}"
            run = manifest.get("runs", {}).get(key)
            assert run, f"formal run missing from manifest: {key}"
            # The uninterrupted formal run predates the resumable trainer.  Its manifest
            # has n_repr and final-state hashes, but not the later `complete` and
            # `epochs_completed` fields.  Accept both schemas; the representation file's
            # own epoch scalar and hashes are checked again when it is opened.
            if "complete" in run:
                assert run["complete"], f"formal run incomplete: {key}"
            if "epochs_completed" in run:
                assert int(run["epochs_completed"]) == expected_epochs
            assert int(run.get("n_repr", 0)) > 0, f"formal run has no representations: {key}"
    return manifest


def load_representation(args, participants: np.ndarray, arm: str, seed: int,
                        variant: str, raw_ast_cache: dict,
                        training_manifest: dict) -> np.ndarray:
    if arm == "raw_ast":
        if "raw" not in raw_ast_cache:
            with np.load(args.ast_emb, allow_pickle=True) as z:
                assert np.array_equal(z["participants"], participants)
                raw = np.asarray(z["embeddings"], dtype=np.float32)
            expected = training_manifest.get("input_hashes", {}).get("audio")
            assert expected is None or sha16(raw) == expected, \
                "raw AST cache does not match the cache used for formal alignment"
            norm = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
            raw_ast_cache["raw"], raw_ast_cache["normalized"] = raw, norm
        return raw_ast_cache[variant]

    path = Path(args.alignment_dir) / f"repr_{arm}_seed{seed}.npz"
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants), f"index mismatch: {path}"
        assert str(np.asarray(z["arm"]).item()) == arm
        assert int(np.asarray(z["seed"]).item()) == seed
        assert int(np.asarray(z["epochs"]).item()) == args.expected_epochs
        file_hashes = json.loads(str(np.asarray(z["input_hashes"]).item()))
        assert file_hashes == training_manifest.get("input_hashes"), \
            f"input-cache hashes do not match manifest: {path}"
        assert str(np.asarray(z["source_commit"]).item()) == training_manifest["source_commit"]
        raw = np.asarray(z["raw"], dtype=np.float32)
        normalized = np.asarray(z["normalized"], dtype=np.float32)
    # The two stored views must be the same representation up to L2 normalisation.
    recomputed = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    assert np.allclose(recomputed, normalized, rtol=2e-5, atol=2e-6), \
        f"stored normalised representation is inconsistent: {path}"
    run = training_manifest["runs"][f"{arm}_seed{seed}"]
    if "repr_raw_hash" in run:
        assert sha16(raw) == run["repr_raw_hash"], f"raw representation hash mismatch: {path}"
    if "repr_norm_hash" in run:
        assert sha16(normalized) == run["repr_norm_hash"], \
            f"normalised representation hash mismatch: {path}"
    X = raw if variant == "raw" else normalized
    assert X.shape[0] == len(participants) and np.isfinite(X).all()
    return X


def _prediction_key(kind: str, variant: str, arm: str, suffix: str) -> str:
    return f"{kind}__{variant}__{arm}__{suffix}"


def evaluate_metrics(preds: dict, d: pd.DataFrame, tests: dict[str, np.ndarray],
                     variants: list[str], boot: int, bootstrap_seed: int) -> dict:
    y_all = d["y"].to_numpy()
    out = {"arms": {}, "comparisons": {}}
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    for variant in variants:
        out["arms"][variant], out["comparisons"][variant] = {}, {}
        for arm in ARMS:
            P = preds[(variant, arm)]["calibrated"]
            out["arms"][variant][arm] = {}
            for test_name, m in tests.items():
                y, Pt = y_all[m], P[:, m]
                auc, auc_ci = hierarchical_ci(Pt, y, auroc, boot, bootstrap_seed)
                nnll, nnll_ci = hierarchical_ci(Pt, y, neg_nll, boot,
                                                 bootstrap_seed + 1)
                per_seed_auc = [auroc(y, p) for p in Pt]
                per_seed_nll = [float(logloss(y, p).mean()) for p in Pt]
                out["arms"][variant][arm][test_name] = {
                    "n": int(m.sum()), "auroc": auc, "auroc_ci": auc_ci,
                    "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                    "per_seed_auroc": per_seed_auc,
                    "per_seed_nll": per_seed_nll,
                    "calibration_per_seed": [calib_diag(y, p) for p in Pt],
                }
        for a, b, label in COMPARISONS:
            key = f"{a}_minus_{b}"
            out["comparisons"][variant][key] = {"interpretation": label}
            for test_name, m in tests.items():
                y = y_all[m]
                A = preds[(variant, a)]["calibrated"][:, m]
                B = preds[(variant, b)]["calibrated"][:, m]
                out["comparisons"][variant][key][test_name] = {
                    "delta_neg_nll": paired_hierarchical_ci(
                        A, B, y, neg_nll, boot, bootstrap_seed + 2),
                    "delta_auroc": paired_hierarchical_ci(
                        A, B, y, auroc, boot, bootstrap_seed + 3),
                }
    if "raw" in variants:
        out["primary_result_path"] = (
            "comparisons.raw.correct_minus_within_label.matched.delta_neg_nll")
    if "normalized" in variants:
        out["sensitivity_result_path"] = (
            "comparisons.normalized.correct_minus_within_label.matched.delta_neg_nll")
    return out


def evaluate_probes(probe_preds: dict, probe_observed: dict, d: pd.DataFrame,
                    tests: dict[str, np.ndarray], variants: list[str], boot: int,
                    bootstrap_seed: int) -> dict:
    targets = probe_labels(d)
    out = {"arms": {}, "comparisons": {}}
    for variant in variants:
        out["arms"][variant], out["comparisons"][variant] = {}, {}
        for target_name in PROBE_TARGETS:
            yt = targets[target_name].fillna(0).astype(int).to_numpy()
            obs = probe_observed[target_name]
            out["arms"][variant][target_name] = {}
            for arm in ARMS:
                P = probe_preds[(variant, arm, target_name)]
                out["arms"][variant][target_name][arm] = {}
                for test_name, m0 in tests.items():
                    m = m0 & obs
                    score, ci = hierarchical_ci(P[:, m], yt[m], auroc, boot,
                                                bootstrap_seed)
                    out["arms"][variant][target_name][arm][test_name] = {
                        "n": int(m.sum()), "auroc": score, "ci": ci,
                        "per_seed": [auroc(yt[m], p[m]) for p in P],
                    }
            out["comparisons"][variant][target_name] = {}
            for a, b, label in COMPARISONS[:3]:
                key = f"{a}_minus_{b}"
                out["comparisons"][variant][target_name][key] = {
                    "interpretation": label}
                for test_name, m0 in tests.items():
                    m = m0 & obs
                    A = probe_preds[(variant, a, target_name)][:, m]
                    B = probe_preds[(variant, b, target_name)][:, m]
                    out["comparisons"][variant][target_name][key][test_name] = \
                        paired_hierarchical_ci(A, B, yt[m], auroc, boot,
                                               bootstrap_seed + 11)
    return out


def self_test() -> None:
    """Synthetic-only static/logic check; never opens a repository result or test set."""
    rng = np.random.RandomState(17)
    n = 240
    d = pd.DataFrame({
        "y": np.tile([0, 1], n // 2),
        "recruitment_source": np.tile(["R", "T", "R", "T"], n // 4),
    })
    train = np.zeros(n, bool); train[:120] = True
    val = np.zeros(n, bool); val[120:200] = True
    folds = make_validation_folds(d, val)
    X = rng.normal(size=(n, 8)).astype(np.float32)
    X[:, 0] += d.y.to_numpy() * 0.4
    head = fit_disease_head(X, d.y.to_numpy(), train, val, folds, seed=0)
    assert head.C in C_GRID
    assert np.isfinite(head.logits).all() and np.isfinite(head.calibrated).all()
    assert np.isfinite(head.val_oof_calibrated[val]).all()
    assert np.isnan(head.val_oof_calibrated[~val]).all()

    y = np.tile([0, 1], 50)
    B = rng.uniform(0.1, 0.9, size=(3, len(y)))
    A = np.clip(B + 0.02 * (2 * y - 1), 1e-4, 1 - 1e-4)
    r = paired_hierarchical_ci(A, B, y, auroc, boot=100, seed=3)
    expected = np.mean([auroc(y, A[k]) - auroc(y, B[k]) for k in range(3)])
    assert abs(r["observed"] - expected) < 1e-12 and len(r["per_seed"]) == 3
    print("SELF-TEST PASS: C selection, OOF/final Platt, and paired hierarchical bootstrap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--ast-emb", default="results/ast_embeddings.npz")
    ap.add_argument("--alignment-dir", default="results/alignment")
    ap.add_argument("--out-dir", default="results/alignment_eval")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                    help="safe staged runs, e.g. --variants raw")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--skip-probes", action="store_true",
                       help="fit and score disease heads only; probe fits can run separately")
    scope.add_argument("--probes-only", action="store_true",
                       help="fit and score probes without repeating disease-head fits")
    ap.add_argument("--run-tag", default=None,
                    help="output suffix; defaults to variants plus disease/probe scope")
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260819)
    ap.add_argument("--evaluate-tests", action="store_true",
                    help="explicitly unlock Standard/matched/matched_long evaluation")
    ap.add_argument("--self-test", action="store_true",
                    help="run synthetic checks only; do not read repository data")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.evaluate_tests:
        ap.error("formal data evaluation is locked; pass --evaluate-tests explicitly")

    seeds = list(args.seeds)
    assert len(seeds) == len(set(seeds)), "seed list contains duplicates"
    variants = list(dict.fromkeys(args.variants))
    out_dir = Path(args.out_dir)
    assert out_dir.resolve() != Path(args.alignment_dir).resolve(), \
        "evaluation output directory must not overwrite the formal training directory"
    out_dir.mkdir(parents=True, exist_ok=True)
    scope_tag = ("probes-only" if args.probes_only else
                 "disease-only" if args.skip_probes else "disease-probes")
    auto_tag = f"{'-'.join(variants)}__{scope_tag}"
    run_tag = args.run_tag or auto_tag
    assert run_tag and all(c.isalnum() or c in "-_." for c in run_tag), \
        "run tag may contain only letters, numbers, dash, underscore and dot"
    manifest = verify_training_manifest(Path(args.alignment_dir) / "manifest.json",
                                        seeds, args.expected_epochs)
    d, train, val, tests = load_cohort(args)
    participants = d[UNIT].to_numpy()
    y = d["y"].to_numpy()
    folds = make_validation_folds(d, val)
    print(f"cohort {len(d)}  train {train.sum()}  Standard val {val.sum()}  "
          f"folds {np.bincount(folds[folds >= 0]).tolist()}")
    predictions, probe_predictions, probe_obs = {}, {}, {}
    selection = {variant: {} for variant in variants}
    probe_selection = {variant: {} for variant in variants}
    raw_ast_cache: dict[str, np.ndarray] = {}
    targets = {} if args.skip_probes else probe_labels(d)
    npz_arrays: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    probe_npz: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }

    for variant in variants:
        for arm in ARMS:
            arm_heads, arm_probe = [], {t: [] for t in PROBE_TARGETS}
            fit_seeds = [seeds[0]] if arm == "raw_ast" else seeds
            for seed in fit_seeds:
                print(f"[{variant}] {arm} seed {seed}: selecting downstream C", flush=True)
                X = load_representation(args, participants, arm, seed, variant,
                                        raw_ast_cache, manifest)
                if not args.probes_only:
                    head = fit_disease_head(X, y, train, val, folds, seed)
                    arm_heads.append(head)
                    selection[variant].setdefault(arm, []).append({
                        "seed": seed, "C": head.C,
                        "calibrator_coef": head.calibrator_coef,
                        "calibrator_intercept": head.calibrator_intercept,
                        "selection": head.selection,
                        "representation_hash": sha16(X),
                    })
                if not args.skip_probes:
                    for target_name in PROBE_TARGETS:
                        logits, C, info, observed = fit_probe(
                            X, targets[target_name], train, folds, seed)
                        arm_probe[target_name].append(logits)
                        probe_obs[target_name] = observed
                        probe_selection[variant].setdefault(arm, {}).setdefault(
                            target_name, []).append({"seed": seed, "C": C,
                                                   "selection": info})

            # Replicate the deterministic raw-AST solution along the formal projector-seed
            # axis.  This is deliberate pairing, not five fictitious optimiser runs.
            if arm == "raw_ast":
                if not args.probes_only:
                    arm_heads = arm_heads * len(seeds)
                    selection[variant][arm] = [
                        {**selection[variant][arm][0], "seed": seed,
                         "replicated_deterministic_reference": True} for seed in seeds]
                if not args.skip_probes:
                    for target_name in PROBE_TARGETS:
                        arm_probe[target_name] = arm_probe[target_name] * len(seeds)
                        base = probe_selection[variant][arm][target_name][0]
                        probe_selection[variant][arm][target_name] = [
                            {**base, "seed": seed,
                             "replicated_deterministic_reference": True} for seed in seeds]

            if not args.probes_only:
                raw_logits = np.stack([h.logits for h in arm_heads])
                calibrated = np.stack([h.calibrated for h in arm_heads])
                oof = np.stack([h.val_oof_calibrated for h in arm_heads])
                Cs = np.asarray([h.C for h in arm_heads])
                cal_coef = np.asarray([h.calibrator_coef for h in arm_heads])
                cal_intercept = np.asarray([h.calibrator_intercept for h in arm_heads])
                predictions[(variant, arm)] = {
                    "raw_logits": raw_logits, "calibrated": calibrated,
                    "val_oof_calibrated": oof,
                }
                base = f"disease__{variant}__{arm}"
                npz_arrays[f"{base}__raw_logits"] = raw_logits
                npz_arrays[f"{base}__calibrated_prob"] = calibrated
                npz_arrays[f"{base}__val_oof_calibrated_prob"] = oof
                npz_arrays[f"{base}__C"] = Cs
                npz_arrays[f"{base}__calibrator_coef"] = cal_coef
                npz_arrays[f"{base}__calibrator_intercept"] = cal_intercept

            if not args.skip_probes:
                for target_name in PROBE_TARGETS:
                    PP = np.stack(arm_probe[target_name])
                    probe_predictions[(variant, arm, target_name)] = PP
                    probe_npz[_prediction_key(
                        "probe", variant, arm, f"{target_name}__raw_logits")] = PP

    # Selection/calibration parameters and all per-participant predictions are written
    # before scoring.  A test result can therefore always be traced back to one frozen head.
    if not args.probes_only:
        atomic_npz(out_dir / f"metadata_alignment_predictions__{run_tag}.npz", **npz_arrays)
    if not args.skip_probes:
        atomic_npz(out_dir / f"metadata_alignment_probe_predictions__{run_tag}.npz",
                   **probe_npz)
    fit_record = {
        "training_manifest": manifest,
        "seeds": seeds,
        "variants": variants,
        "disease_included": not args.probes_only,
        "probes_included": not args.skip_probes,
        "selection": selection,
        "probe_selection": probe_selection,
        "validation_folds_hash": sha16(folds),
        "test_status": "explicitly_unlocked",
    }
    atomic_json(fit_record, out_dir / f"metadata_alignment_head_fits__{run_tag}.json")

    results = {
        "standing": "exploratory; official tests informed earlier protocol versions",
        "primary_representation": "raw post-ReLU projector output",
        "normalized_representation": "pre-declared sensitivity",
        "calibration": "one final Standard-val Platt calibrator per arm and seed",
    }
    if not args.probes_only:
        results["disease"] = evaluate_metrics(
            predictions, d, tests, variants, args.bootstrap, args.bootstrap_seed)
    if not args.skip_probes:
        results["probes"] = evaluate_probes(
            probe_predictions, probe_obs, d, tests, variants,
            args.bootstrap, args.bootstrap_seed)
    atomic_json(results, out_dir / f"metadata_alignment_results__{run_tag}.json")
    print(f"wrote formal predictions, fits and exploratory test results under {out_dir} "
          f"with tag {run_tag}")


if __name__ == "__main__":
    main()
