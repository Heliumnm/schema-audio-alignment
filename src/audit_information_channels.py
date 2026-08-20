"""Fit and score the frozen information-channel probe matrix.

The matrix operationalises the taxonomy in
``docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md`` for one audio backbone at a time:

* acquisition: long recording, large file, loud recording, clipping;
* participant: sex and age 65+;
* clinical context: cough and no symptoms;
* cohort/protocol: recruitment source;
* disease: COVID.

Representations are raw, correct metadata alignment, within-label shuffled alignment, and
global-shuffled alignment.  The fitted probe protocol is reused from
``eval_metadata_alignment.py``.  Fitting and test scoring are separate commands so all
per-participant/per-seed predictions, labels, configuration, and hashes exist before a
formal test statistic is read::

    python src/audit_information_channels.py --self-test
    python src/audit_information_channels.py --preflight [paths ...]
    python src/audit_information_channels.py --fit [paths ...]
    python src/audit_information_channels.py --score-tests --out-dir ...

The script is generic in the backbone.  Run it once with the AST caches and once with the
OPERA-CT caches.  Outputs are no-clobber.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from eval_metadata_alignment import (
    atomic_json,
    atomic_npz,
    fit_disease_head,
    fit_probe,
    hierarchical_ci,
    load_cohort,
    load_representation,
    make_validation_folds,
    paired_hierarchical_ci,
    sha16,
    verify_training_manifest,
)


ARMS = ("raw", "correct", "within_label", "global")
ALIGNED_ARMS = ("correct", "within_label", "global")
TARGET_GROUPS = {
    "acquisition": ("long_recording", "large_file", "loud_recording", "clipped"),
    "participant": ("sex_female", "age_65plus"),
    "clinical_context": ("cough_any", "no_symptoms"),
    "cohort_protocol": ("recruitment_source",),
    "disease": ("covid",),
}
TARGETS = tuple(t for group in TARGET_GROUPS.values() for t in group)
COMPARISONS = (
    ("correct", "within_label", "C-W individual metadata correspondence"),
    ("within_label", "global", "W-G label/population co-occurrence"),
    ("correct", "raw", "C-R change relative to frozen raw audio"),
)
BOOTSTRAP_SEED = 20260820
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                return h.hexdigest()
            h.update(block)


def load_probe_table(args) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                          dict[str, np.ndarray], dict[str, pd.Series], dict]:
    data, train, val, tests = load_cohort(args)
    features = pd.read_csv(args.artefacts)
    needed = [UNIT, "duration_s", "size_bytes", "rms", "clip_frac"]
    missing = [column for column in needed if column not in features]
    if missing:
        raise ValueError(f"artefact table missing required columns: {missing}")
    features = features[needed]
    if features[UNIT].duplicated().any():
        raise ValueError("artefact table has duplicate participant identifiers")
    joined = data[[UNIT]].merge(features, on=UNIT, how="left", validate="one_to_one")
    assert np.array_equal(joined[UNIT].to_numpy(), data[UNIT].to_numpy())
    if joined[needed[1:]].isna().any().any():
        raise ValueError("artefact target values are missing for the frozen cohort")
    for column in needed[1:]:
        data[column] = joined[column].to_numpy()

    thresholds = {
        "duration_s_median": float(data["duration_s"].median()),
        "size_bytes_median": float(data["size_bytes"].median()),
        "rms_median": float(data["rms"].median()),
        "clipped_rule": "clip_frac > 0",
    }
    target = {
        "long_recording": (data["duration_s"] > thresholds["duration_s_median"]),
        "large_file": (data["size_bytes"] > thresholds["size_bytes_median"]),
        "loud_recording": (data["rms"] > thresholds["rms_median"]),
        "clipped": (data["clip_frac"] > 0),
        "sex_female": (data["gender"] == "Female").where(data["gender"].notna()),
        "age_65plus": (data["age"] == "65+").where(data["age"].notna()),
        "cough_any": data["symptom_cough_any"].where(
            data["symptom_cough_any"].notna()),
        "no_symptoms": data["symptom_none"].where(data["symptom_none"].notna()),
        "recruitment_source": (
            data["recruitment_source"] == "Test and Trace").where(
                data["recruitment_source"].notna()),
        "covid": data["y"].astype(int),
    }
    assert tuple(target) == TARGETS
    for name, values in target.items():
        observed = values.notna()
        levels = set(values[observed].astype(int).unique())
        if levels != {0, 1}:
            raise ValueError(f"target {name} is not binary over observed rows: {levels}")
    audit = {
        "thresholds": thresholds,
        "threshold_definition": (
            "reuses audio_baselines_v2.py frozen full-cohort medians for duration_s, "
            "size_bytes, and rms; clipped is clip_frac > 0"),
        "target_counts": {
            name: {
                "n_observed": int(values.notna().sum()),
                "n_positive": int(values.fillna(0).astype(int).sum()),
            } for name, values in target.items()
        },
        "artefact_file_sha256": sha256_file(Path(args.artefacts)),
    }
    return data, train, val, tests, target, audit


def prepare(args) -> tuple[dict, dict]:
    seeds = list(args.seeds)
    assert len(seeds) == len(set(seeds)) == 5, "formal matrix requires five unique seeds"
    manifest_path = Path(args.alignment_dir) / "manifest.json"
    manifest = verify_training_manifest(manifest_path, seeds, args.expected_epochs)
    data, train, val, tests, targets, target_audit = load_probe_table(args)
    participants = data[UNIT].to_numpy()
    folds = make_validation_folds(data, val)
    raw_cache: dict[str, np.ndarray] = {}
    representation_audit: dict[str, dict] = {}
    for arm in ARMS:
        representation_audit[arm] = {}
        fit_seeds = [seeds[0]] if arm == "raw" else seeds
        loader_arm = "raw_ast" if arm == "raw" else arm
        for seed in fit_seeds:
            x = load_representation(
                args, participants, loader_arm, seed, args.variant, raw_cache, manifest)
            representation_audit[arm][str(seed)] = {
                "shape": list(x.shape),
                "sha16": sha16(x),
                "finite": bool(np.isfinite(x).all()),
                "nonzero_rows": int(np.sum(np.linalg.norm(x, axis=1) > 0)),
            }
    audit = {
        "standing": "exploratory information-channel audit on frozen UKCOVID cohort",
        "backbone": args.backbone,
        "variant": args.variant,
        "n_cohort": int(len(data)),
        "n_train": int(train.sum()),
        "n_standard_validation": int(val.sum()),
        "n_standard_test": int(tests["standard"].sum()),
        "n_matched": int(tests["matched"].sum()),
        "n_matched_long": int(tests["matched_long"].sum()),
        "seeds": seeds,
        "expected_epochs": int(args.expected_epochs),
        "alignment_manifest_sha256": sha256_file(manifest_path),
        "training_input_hashes": manifest.get("input_hashes", {}),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "audio_embedding_file_sha256": sha256_file(Path(args.ast_emb)),
        "validation_fold_sha16": sha16(folds),
        "representations": representation_audit,
        **target_audit,
    }
    state = {
        "manifest": manifest,
        "data": data,
        "train": train,
        "val": val,
        "tests": tests,
        "targets": targets,
        "folds": folds,
        "participants": participants,
    }
    return audit, state


def fit_matrix(args, audit: dict, state: dict, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    for path in (predictions_path, fits_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    data = state["data"]
    participants = state["participants"]
    train, val, folds = state["train"], state["val"], state["folds"]
    seeds = list(args.seeds)
    raw_cache: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {
        "participants": participants,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "arms": np.asarray(ARMS),
        "targets": np.asarray(TARGETS),
        "mask__standard": state["tests"]["standard"],
        "mask__matched": state["tests"]["matched"],
        "mask__matched_long": state["tests"]["matched_long"],
    }
    for target_name, values in state["targets"].items():
        observed = values.notna().to_numpy()
        label = values.fillna(0).astype(int).to_numpy()
        arrays[f"observed__{target_name}"] = observed
        arrays[f"label__{target_name}"] = label

    fits: dict[str, dict] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        loader_arm = "raw_ast" if arm == "raw" else arm
        fit_seeds = [seeds[0]] if arm == "raw" else seeds
        target_scores: dict[str, list[np.ndarray]] = {target: [] for target in TARGETS}
        disease_prob: list[np.ndarray] = []
        disease_oof: list[np.ndarray] = []
        for seed in fit_seeds:
            representation = load_representation(
                args, participants, loader_arm, seed, args.variant, raw_cache,
                state["manifest"])
            fits[arm][str(seed)] = {
                "representation_sha16": sha16(representation), "targets": {}}
            for target_name in TARGETS:
                values = state["targets"][target_name]
                if target_name == "covid":
                    head = fit_disease_head(
                        representation, data["y"].to_numpy(), train, val, folds, seed)
                    target_scores[target_name].append(head.logits.astype(np.float32))
                    disease_prob.append(head.calibrated.astype(np.float32))
                    disease_oof.append(head.val_oof_calibrated.astype(np.float32))
                    fits[arm][str(seed)]["targets"][target_name] = {
                        "C": head.C,
                        "calibrator_coef": head.calibrator_coef,
                        "calibrator_intercept": head.calibrator_intercept,
                        "selection": head.selection,
                    }
                else:
                    logits, c_value, selection, observed = fit_probe(
                        representation, values, train, folds, seed)
                    assert np.array_equal(observed, values.notna().to_numpy())
                    target_scores[target_name].append(logits.astype(np.float32))
                    fits[arm][str(seed)]["targets"][target_name] = {
                        "C": c_value, "selection": selection,
                    }
                print(f"{args.backbone} {arm} seed {seed}: fitted {target_name}", flush=True)

        # Raw audio is deterministic.  Replication aligns its predictions with the five
        # true alignment seeds; it does not create five independent raw fits.
        if arm == "raw":
            base = fits[arm][str(fit_seeds[0])]
            fits[arm] = {
                str(seed): {**base, "replicated_deterministic_reference": True}
                for seed in seeds
            }
            for target_name in TARGETS:
                target_scores[target_name] = target_scores[target_name] * len(seeds)
            disease_prob = disease_prob * len(seeds)
            disease_oof = disease_oof * len(seeds)

        for target_name in TARGETS:
            arrays[f"score__{arm}__{target_name}"] = np.stack(
                target_scores[target_name]).astype(np.float32)
        arrays[f"calibrated_prob__{arm}__covid"] = np.stack(disease_prob).astype(np.float32)
        arrays[f"val_oof_prob__{arm}__covid"] = np.stack(disease_oof).astype(np.float32)

    # Per-participant predictions are always durable before any formal test score exists.
    atomic_npz(predictions_path, **arrays)
    atomic_json({
        **audit,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "fits": fits,
        "test_metrics_computed": False,
        "raw_seed_axis_note": (
            "raw audio has one deterministic head replicated across five alignment seeds"),
    }, fits_path)
    print(f"wrote frozen probe predictions and fit record under {out_dir}; tests not scored")


def score_matrix(args, out_dir: Path) -> None:
    predictions_path = out_dir / "predictions.npz"
    fits_path = out_dir / "fits.json"
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    if not predictions_path.is_file() or not fits_path.is_file():
        raise FileNotFoundError("fit stage must finish before test scoring")
    with open(fits_path) as stream:
        fit_record = json.load(stream)
    if sha256_file(SCRIPT_PATH) != fit_record["script_sha256"]:
        raise RuntimeError("probe script changed between frozen fitting and test scoring")
    if sha256_file(predictions_path) != fit_record["predictions_sha256"]:
        raise ValueError("prediction archive changed after the fit record was frozen")

    with np.load(predictions_path, allow_pickle=True) as z:
        files = set(z.files)
        seeds = np.asarray(z["seeds"])
        assert len(seeds) == 5
        tests = {name: np.asarray(z[f"mask__{name}"], dtype=bool)
                 for name in ("standard", "matched", "matched_long")}
        labels = {target: np.asarray(z[f"label__{target}"], dtype=int)
                  for target in TARGETS}
        observed = {target: np.asarray(z[f"observed__{target}"], dtype=bool)
                    for target in TARGETS}
        score = {(arm, target): np.asarray(z[f"score__{arm}__{target}"], dtype=float)
                 for arm in ARMS for target in TARGETS}
        covid_prob = {
            arm: np.asarray(z[f"calibrated_prob__{arm}__covid"], dtype=float)
            for arm in ARMS}
        expected = {
            f"score__{arm}__{target}" for arm in ARMS for target in TARGETS}
        assert expected.issubset(files)

    out: dict[str, dict] = {"arms": {}, "comparisons": {}}
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    for target_index, target_name in enumerate(TARGETS):
        out["arms"][target_name] = {}
        out["comparisons"][target_name] = {}
        y_all = labels[target_name]
        for arm in ARMS:
            out["arms"][target_name][arm] = {}
            values = covid_prob[arm] if target_name == "covid" else score[(arm, target_name)]
            for test_index, (test_name, base_mask) in enumerate(tests.items()):
                mask = base_mask & observed[target_name]
                y, prediction = y_all[mask], values[:, mask]
                auc, auc_ci = hierarchical_ci(
                    prediction, y, auroc, args.bootstrap,
                    args.bootstrap_seed + 1000 * target_index + 10 * test_index)
                entry = {
                    "n": int(mask.sum()), "auroc": auc, "auroc_ci": auc_ci,
                    "per_seed_auroc": [auroc(y, row) for row in prediction],
                }
                if target_name == "covid":
                    nnll, nnll_ci = hierarchical_ci(
                        prediction, y, neg_nll, args.bootstrap,
                        args.bootstrap_seed + 1000 * target_index + 10 * test_index + 1)
                    entry.update({
                        "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                        "per_seed_nll": [float(logloss(y, row).mean())
                                         for row in prediction],
                        "calibration_per_seed": [calib_diag(y, row)
                                                 for row in prediction],
                    })
                out["arms"][target_name][arm][test_name] = entry

        for comparison_index, (a, b, label) in enumerate(COMPARISONS):
            key = f"{a}_minus_{b}"
            out["comparisons"][target_name][key] = {"interpretation": label}
            a_values = covid_prob[a] if target_name == "covid" else score[(a, target_name)]
            b_values = covid_prob[b] if target_name == "covid" else score[(b, target_name)]
            for test_index, (test_name, base_mask) in enumerate(tests.items()):
                mask = base_mask & observed[target_name]
                y = y_all[mask]
                entry = {
                    "delta_auroc": paired_hierarchical_ci(
                        a_values[:, mask], b_values[:, mask], y, auroc,
                        args.bootstrap,
                        args.bootstrap_seed + 20_000 + 1000 * target_index +
                        100 * comparison_index + 10 * test_index)
                }
                if target_name == "covid":
                    entry["delta_neg_nll"] = paired_hierarchical_ci(
                        a_values[:, mask], b_values[:, mask], y, neg_nll,
                        args.bootstrap,
                        args.bootstrap_seed + 30_000 + 1000 * target_index +
                        100 * comparison_index + 10 * test_index)
                out["comparisons"][target_name][key][test_name] = entry

    result = {
        "standing": fit_record["standing"],
        "backbone": fit_record["backbone"],
        "variant": fit_record["variant"],
        "target_groups": TARGET_GROUPS,
        "comparison_order": [f"{a}_minus_{b}" for a, b, _ in COMPARISONS],
        "primary_reporting": (
            "report each C-W target separately; do not average probes into one score"),
        "predictions_sha256": fit_record["predictions_sha256"],
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "metrics": out,
    }
    atomic_json(result, metrics_path)
    print(f"wrote explicitly unlocked formal test metrics to {metrics_path}")


def self_test() -> None:
    rng = np.random.RandomState(12)
    n = 300
    y = np.tile([0, 1], n // 2)
    base = rng.normal(size=(5, n))
    better = base + 0.5 * (2 * y - 1)[None, :]
    result = paired_hierarchical_ci(
        better, base, y, auroc, boot=100, seed=17)
    result2 = paired_hierarchical_ci(
        better, base, y, auroc, boot=100, seed=17)
    assert result == result2 and result["observed"] > 0
    p = 1 / (1 + np.exp(-better))
    assert np.isfinite(logloss(y, p[0])).all()
    assert tuple(t for group in TARGET_GROUPS.values() for t in group) == TARGETS

    # Exercise the durable-predictions -> explicitly unlocked scorer boundary with a
    # complete synthetic archive.  No repository data are opened.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        arrays: dict[str, np.ndarray] = {
            "participants": np.asarray([f"p{i:03d}" for i in range(n)]),
            "seeds": np.arange(5),
            "arms": np.asarray(ARMS),
            "targets": np.asarray(TARGETS),
            "mask__standard": np.arange(n) < 100,
            "mask__matched": (np.arange(n) >= 100) & (np.arange(n) < 200),
            "mask__matched_long": np.arange(n) >= 200,
        }
        for target_name in TARGETS:
            arrays[f"label__{target_name}"] = y
            arrays[f"observed__{target_name}"] = np.ones(n, dtype=bool)
            for arm_index, arm in enumerate(ARMS):
                arm_score = base + 0.05 * arm_index * (2 * y - 1)[None, :]
                arrays[f"score__{arm}__{target_name}"] = arm_score.astype(np.float32)
                if target_name == "covid":
                    arrays[f"calibrated_prob__{arm}__covid"] = (
                        1 / (1 + np.exp(-arm_score))).astype(np.float32)
                    arrays[f"val_oof_prob__{arm}__covid"] = (
                        1 / (1 + np.exp(-arm_score))).astype(np.float32)
        predictions_path = out_dir / "predictions.npz"
        atomic_npz(predictions_path, **arrays)
        atomic_json({
            "standing": "synthetic self-test", "backbone": "synthetic",
            "variant": "raw", "predictions_sha256": sha256_file(predictions_path),
            "script_sha256": sha256_file(SCRIPT_PATH),
        }, out_dir / "fits.json")
        score_args = argparse.Namespace(bootstrap=20, bootstrap_seed=19)
        score_matrix(score_args, out_dir)
        scored = json.load(open(out_dir / "metrics.json"))
        assert scored["metrics"]["arms"]["covid"]["correct"]["matched"]["n"] == 100
        assert scored["predictions_sha256"] == sha256_file(predictions_path)
    print("SELF-TEST PASS: taxonomy, paired hierarchy, fit/score archive boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--artefacts", default="results/artefact_features.csv")
    parser.add_argument("--audio-emb", dest="ast_emb", default="results/ast_embeddings.npz")
    parser.add_argument("--alignment-dir", default="results/alignment")
    parser.add_argument("--backbone", default="AST-6L")
    parser.add_argument("--variant", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--out-dir", default="results/information_channels_ast")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected-epochs", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--score-tests", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.score_tests:
        score_matrix(args, out_dir)
        return

    preflight_path = out_dir / "preflight.json"
    if args.preflight and preflight_path.exists():
        raise FileExistsError(f"refusing to overwrite {preflight_path}")
    audit, state = prepare(args)
    if args.preflight:
        atomic_json({**audit, "preflight_pass": True,
                     "formal_test_metrics_computed": False}, preflight_path)
        print(f"PREFLIGHT PASS: wrote {preflight_path}; tests not scored")
        return
    fit_matrix(args, audit, state, out_dir)


if __name__ == "__main__":
    main()
