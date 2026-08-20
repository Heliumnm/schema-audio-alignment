"""Frozen E1 direct- and raw-preserving-fusion follow-up.

This evaluator implements Section 2 of ``ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md``.
It never selects a hyperparameter on Standard test, matched, or matched-long.  The
existing metadata-only predictions are reused bit-for-bit.  For the AST run, all
previously completed direct-fusion arms are also reused; only the three missing
raw-preserving arms are fitted.

Recommended order on the server::

    # OPERA-CT: creates the shared M+A deterministic reference.
    python src/eval_fusion_followup.py --backbone opera ... --evaluate-tests

    # AST: reuses M+A from the OPERA archive and all old AST direct-fusion arms.
    python src/eval_fusion_followup.py --backbone ast ... \
        --common-preds results/fusion_followup/opera/predictions.npz \
        --reuse-existing-backbone-arms --evaluate-tests

``--self-test`` is synthetic-only and opens no repository result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Prediction archives produced by NumPy 2 pickle object/string participant arrays under
# ``numpy._core``.  NumPy 1.x exposes the identical implementation as ``numpy.core``.
# Registering the compatibility alias makes frozen archives portable without rewriting.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

from artefact_baseline import FEATS
from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from eval_matched_direct_fusion import build_metadata_matrix
from eval_metadata_alignment import (
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


SEEDS = (0, 1, 2, 3, 4)
ALIGNMENT = ("correct", "within_label", "global")
ALL_ARMS = (
    "M", "M+A", "M+R", "M+C", "M+W", "M+G",
    "M+R+C", "M+R+W", "M+R+G",
)
ARM_SLUG = {
    "M": "m",
    "M+A": "m_a",
    "M+R": "m_r",
    "M+C": "m_c",
    "M+W": "m_w",
    "M+G": "m_g",
    "M+R+C": "m_r_c",
    "M+R+W": "m_r_w",
    "M+R+G": "m_r_g",
}
OLD_AST_KEYS = {
    "M": "metadata_only",
    "M+R": "metadata_raw_ast",
    "M+C": "metadata_correct",
    "M+W": "metadata_within_label",
    "M+G": "metadata_global",
}
ALIGNMENT_BY_LETTER = {"C": "correct", "W": "within_label", "G": "global"}
COMPARISONS = (
    ("M+R", "M", "direct_audio_increment_primary"),
    ("M+C", "M+W", "individual_pairing_conditional_on_metadata"),
    ("M+C", "M+R", "alignment_versus_raw_fusion"),
    ("M+R+C", "M+R", "correct_alignment_increment_with_raw_preserved"),
    ("M+R+C", "M+R+W", "individual_pairing_with_raw_preserved"),
    ("M+R", "M+A", "audio_increment_over_recording_artefacts"),
    ("M+W", "M+G", "label_level_cooccurrence_conditional_on_metadata"),
)


def file_sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _as_seed_matrix(x: np.ndarray, seeds: list[int]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] == 1:
        x = np.repeat(x, len(seeds), axis=0)
    assert x.shape[0] == len(seeds)
    return x


def _read_triplet(z, prefix: str, seeds: list[int]) -> dict[str, np.ndarray]:
    keys = {
        "calibrated": prefix + "__calibrated_prob",
        "logits": prefix + "__raw_logits",
        "oof": prefix + "__val_oof_calibrated_prob",
    }
    missing = [key for key in keys.values() if key not in z.files]
    assert not missing, f"prediction archive lacks {missing}"
    return {name: _as_seed_matrix(z[key], seeds) for name, key in keys.items()}


def load_old_direct_fusion(path: Path, participants: np.ndarray,
                           seeds: list[int]) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(np.asarray(z["participants"]), participants), \
            "old direct-fusion participant order mismatch"
        return {
            arm: _read_triplet(z, old, seeds)
            for arm, old in OLD_AST_KEYS.items()
        }


def load_common_ma(path: Path, participants: np.ndarray,
                   seeds: list[int]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(np.asarray(z["participants"]), participants), \
            "common M+A participant order mismatch"
        return _read_triplet(z, f"fusion__{ARM_SLUG['M+A']}", seeds)


def load_artefacts(path: Path, participants: np.ndarray) -> tuple[np.ndarray, dict]:
    f = pd.read_csv(path)
    assert f[UNIT].is_unique, "artefact table has duplicate participants"
    indexed = f.set_index(UNIT)
    missing = pd.Index(participants).difference(indexed.index)
    assert len(missing) == 0, f"artefact table is missing {len(missing)} participants"
    x = indexed.loc[participants, FEATS].to_numpy(np.float32)
    assert x.shape == (len(participants), len(FEATS)) and np.isfinite(x).all()
    return x, {"features": FEATS, "shape": list(x.shape), "matrix_sha16": sha16(x)}


def heads_to_triplet(heads: list, seeds: list[int]) -> tuple[dict[str, np.ndarray], list[dict]]:
    replicated = len(heads) == 1 and len(seeds) > 1
    if replicated:
        heads = heads * len(seeds)
    assert len(heads) == len(seeds)
    triplet = {
        "calibrated": np.stack([h.calibrated for h in heads]),
        "logits": np.stack([h.logits for h in heads]),
        "oof": np.stack([h.val_oof_calibrated for h in heads]),
    }
    records = []
    for seed, head in zip(seeds, heads):
        records.append({
            "seed": seed,
            "C": head.C,
            "calibrator_coef": head.calibrator_coef,
            "calibrator_intercept": head.calibrator_intercept,
            "selection": head.selection,
            "replicated_deterministic_reference": replicated,
        })
    return triplet, records


def fit_one_arm(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
                folds: np.ndarray, seeds: list[int], deterministic: bool) \
        -> tuple[dict[str, np.ndarray], list[dict]]:
    fit_seeds = [seeds[0]] if deterministic else seeds
    heads = [fit_disease_head(x, y, train, val, folds, seed) for seed in fit_seeds]
    return heads_to_triplet(heads, seeds)


def make_metrics(pred: dict[str, dict[str, np.ndarray]], d: pd.DataFrame,
                 tests: dict[str, np.ndarray], boot: int, seed: int) -> dict:
    y_all = d["y"].to_numpy()
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    brier = lambda yy, pp: float(np.mean((np.asarray(pp) - np.asarray(yy)) ** 2))
    out: dict = {"arms": {}, "comparisons": {}}
    for arm in ALL_ARMS:
        out["arms"][arm] = {}
        for split, mask in tests.items():
            y, p = y_all[mask], pred[arm]["calibrated"][:, mask]
            auc, auc_ci = hierarchical_ci(p, y, auroc, boot, seed)
            nnll, nnll_ci = hierarchical_ci(p, y, neg_nll, boot, seed + 1)
            br, br_ci = hierarchical_ci(p, y, brier, boot, seed + 2)
            out["arms"][arm][split] = {
                "n": int(mask.sum()), "auroc": auc, "auroc_ci": auc_ci,
                "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                "brier": br, "brier_ci": br_ci,
                "per_seed_auroc": [auroc(y, row) for row in p],
                "per_seed_nll": [float(logloss(y, row).mean()) for row in p],
                "calibration_per_seed": [calib_diag(y, row) for row in p],
            }
    for a, b, label in COMPARISONS:
        key = f"{ARM_SLUG[a]}_minus_{ARM_SLUG[b]}"
        out["comparisons"][key] = {"arms": [a, b], "interpretation": label}
        for split, mask in tests.items():
            y = y_all[mask]
            aa = pred[a]["calibrated"][:, mask]
            bb = pred[b]["calibrated"][:, mask]
            out["comparisons"][key][split] = {
                "delta_auroc": paired_hierarchical_ci(
                    aa, bb, y, auroc, boot, seed + 10),
                "delta_neg_nll": paired_hierarchical_ci(
                    aa, bb, y, neg_nll, boot, seed + 11),
                "delta_brier": paired_hierarchical_ci(
                    aa, bb, y, brier, boot, seed + 12),
            }
    out["primary_result_path"] = "comparisons.m_r_minus_m.matched.delta_auroc"
    return out


def _store(arrays: dict[str, np.ndarray], arm: str,
           triplet: dict[str, np.ndarray]) -> None:
    prefix = f"fusion__{ARM_SLUG[arm]}"
    arrays[prefix + "__calibrated_prob"] = triplet["calibrated"]
    arrays[prefix + "__raw_logits"] = triplet["logits"]
    arrays[prefix + "__val_oof_calibrated_prob"] = triplet["oof"]


def self_test() -> None:
    rng = np.random.RandomState(41)
    participants = np.asarray([f"p{i}" for i in range(40)])
    seeds = list(SEEDS)
    # Test deterministic seed replication and dynamic metrics without repository data.
    y = np.tile([0, 1], 20)
    base = np.clip(.45 + .1 * y + rng.normal(0, .02, len(y)), .01, .99)
    pred = {}
    for i, arm in enumerate(ALL_ARMS):
        p = np.clip(base + i * .001, .001, .999)
        pred[arm] = {
            "calibrated": np.repeat(p[None, :], 5, axis=0),
            "logits": np.repeat(np.log(p / (1 - p))[None, :], 5, axis=0),
            "oof": np.full((5, len(y)), np.nan),
        }
    d = pd.DataFrame({"y": y})
    tests = {"standard": np.ones(len(y), bool),
             "matched": np.ones(len(y), bool),
             "matched_long": np.ones(len(y), bool)}
    result = make_metrics(pred, d, tests, boot=30, seed=7)
    assert result["primary_result_path"].endswith("delta_auroc")
    assert set(result["arms"]) == set(ALL_ARMS)
    assert len(participants) == pred["M"]["calibrated"].shape[1]
    print("SELF-TEST PASS: arm mapping, deterministic seed axis, metrics and comparisons")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=("opera", "ast"), default=None)
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--audio-emb", required=False,
                    help="raw AST or OPERA embedding archive")
    ap.add_argument("--alignment-dir", required=False)
    ap.add_argument("--alignment-manifest", required=False)
    ap.add_argument("--old-direct-preds", default="results/direct_fusion_matched/predictions.npz")
    ap.add_argument("--common-preds", default=None,
                    help="prior E1 archive providing the one shared M+A fit")
    ap.add_argument("--artefacts", default="results/artefact_features.csv")
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--reuse-existing-backbone-arms", action="store_true",
                    help="required for AST: reuse old M+R/M+C/M+W/M+G bit-for-bit")
    ap.add_argument("--evaluate-tests", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.backbone is None:
        ap.error("--backbone is required for a formal run")
    if not args.evaluate_tests:
        ap.error("test evaluation is locked; pass --evaluate-tests explicitly")
    seeds = list(args.seeds)
    assert seeds == list(SEEDS), "formal run requires the frozen five seeds 0..4"
    if not args.audio_emb or not args.alignment_dir or not args.alignment_manifest:
        ap.error("--audio-emb, --alignment-dir and --alignment-manifest are required")
    if args.backbone == "ast":
        assert args.reuse_existing_backbone_arms, \
            "AST protocol must reuse completed direct-fusion arms"
        assert args.common_preds, "AST protocol must reuse the shared M+A fit from OPERA"

    out_dir = Path(args.out_dir or f"results/fusion_followup/{args.backbone}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / name for name in ("predictions.npz", "config.json", "results.json")}
    for path in paths.values():
        assert not path.exists(), f"refusing to overwrite {path}"

    # load_cohort expects these attributes and independently enforces split disjointness.
    cohort_args = SimpleNamespace(data=args.data, cohort=args.cohort)
    d, train, val, tests = load_cohort(cohort_args)
    participants = d[UNIT].to_numpy()
    y = d["y"].to_numpy()
    folds = make_validation_folds(d, val)
    metadata, _, metadata_audit = build_metadata_matrix(
        Path(args.data), Path(args.cohort), participants)
    artefacts, artefact_audit = load_artefacts(Path(args.artefacts), participants)

    old = load_old_direct_fusion(Path(args.old_direct_preds), participants, seeds)
    pred: dict[str, dict[str, np.ndarray]] = {"M": old["M"]}
    selection: dict[str, list[dict] | dict] = {
        "M": {"reused_bitwise": True, "source": str(args.old_direct_preds)}
    }

    if args.common_preds:
        pred["M+A"] = load_common_ma(Path(args.common_preds), participants, seeds)
        selection["M+A"] = {"reused_bitwise": True, "source": str(args.common_preds)}
    else:
        pred["M+A"], selection["M+A"] = fit_one_arm(
            np.concatenate([metadata, artefacts], axis=1), y, train, val, folds,
            seeds, deterministic=True)

    manifest = verify_training_manifest(Path(args.alignment_manifest), seeds,
                                        args.expected_epochs)
    loader_args = SimpleNamespace(
        ast_emb=args.audio_emb,
        alignment_dir=args.alignment_dir,
        expected_epochs=args.expected_epochs,
    )
    cache: dict[str, np.ndarray] = {}
    raw = load_representation(loader_args, participants, "raw_ast", seeds[0], "raw",
                              cache, manifest)

    if args.reuse_existing_backbone_arms:
        for arm in ("M+R", "M+C", "M+W", "M+G"):
            pred[arm] = old[arm]
            selection[arm] = {"reused_bitwise": True, "source": str(args.old_direct_preds)}
    else:
        pred["M+R"], selection["M+R"] = fit_one_arm(
            np.concatenate([metadata, raw], axis=1), y, train, val, folds,
            seeds, deterministic=True)
        for letter in ("C", "W", "G"):
            arm = "M+" + letter
            aligned_name = ALIGNMENT_BY_LETTER[letter]
            heads = []
            records = []
            for seed in seeds:
                aligned = load_representation(loader_args, participants, aligned_name,
                                              seed, "raw", cache, manifest)
                triplet, record = fit_one_arm(
                    np.concatenate([metadata, aligned], axis=1), y, train, val,
                    folds, [seed], deterministic=False)
                heads.append({k: v[0] for k, v in triplet.items()})
                records.extend(record)
            pred[arm] = {k: np.stack([head[k] for head in heads])
                         for k in ("calibrated", "logits", "oof")}
            selection[arm] = records

    # The new raw-preserving arms are the only AST fits and are also required for OPERA.
    for letter in ("C", "W", "G"):
        arm = "M+R+" + letter
        aligned_name = ALIGNMENT_BY_LETTER[letter]
        heads = []
        records = []
        for seed in seeds:
            aligned = load_representation(loader_args, participants, aligned_name,
                                          seed, "raw", cache, manifest)
            triplet, record = fit_one_arm(
                np.concatenate([metadata, raw, aligned], axis=1), y, train, val,
                folds, [seed], deterministic=False)
            heads.append({k: v[0] for k, v in triplet.items()})
            records.extend(record)
        pred[arm] = {k: np.stack([head[k] for head in heads])
                     for k in ("calibrated", "logits", "oof")}
        selection[arm] = records

    assert set(pred) == set(ALL_ARMS)
    input_paths = {
        "cohort": Path(args.cohort), "audio_embeddings": Path(args.audio_emb),
        "alignment_manifest": Path(args.alignment_manifest),
        "old_direct_predictions": Path(args.old_direct_preds),
        "artefact_features": Path(args.artefacts),
        "participant_metadata": Path(args.data) / "participant_metadata.csv",
        "train_test_splits": Path(args.data) / "train_test_splits.csv",
    }
    if args.common_preds:
        input_paths["common_predictions"] = Path(args.common_preds)
    config = {
        "protocol": "ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md/E1",
        "script_sha256": file_sha256(Path(__file__)),
        "standing": "exploratory discovery-cohort robustness analysis",
        "backbone": args.backbone,
        "seeds": seeds,
        "expected_epochs": args.expected_epochs,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "selection": "Standard train heads; complete Standard-val one-SE and Platt only",
        "target_tuning": False,
        "metadata_audit": metadata_audit,
        "artefact_audit": artefact_audit,
        "validation_folds_sha16": sha16(folds),
        "raw_representation_sha16": sha16(raw),
        "input_file_sha256": {name: file_sha256(path) for name, path in input_paths.items()},
        "selection_records": selection,
    }
    config["config_sha256"] = canonical_hash(config)

    arrays: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "config_sha256": np.asarray(config["config_sha256"]),
    }
    for arm in ALL_ARMS:
        _store(arrays, arm, pred[arm])
    # Predictions and configuration are frozen before any test metric is calculated.
    atomic_npz(paths["predictions.npz"], **arrays)
    atomic_json(config, paths["config.json"])
    assert file_sha256(Path(__file__)) == config["script_sha256"], \
        "evaluator changed after its configuration was frozen"
    results = {
        "protocol": "E1 cross-backbone direct/raw-preserving fusion",
        "backbone": args.backbone,
        "config_sha256": config["config_sha256"],
        "predictions_sha256": file_sha256(paths["predictions.npz"]),
        "metrics": make_metrics(pred, d, tests, args.bootstrap, args.bootstrap_seed),
    }
    atomic_json(results, paths["results.json"])
    print(f"wrote E1 {args.backbone} predictions/config/results to {out_dir}")


if __name__ == "__main__":
    main()
