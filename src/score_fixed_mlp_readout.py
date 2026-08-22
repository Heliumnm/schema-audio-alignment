#!/usr/bin/env python3
"""Score frozen fixed-MLP predictions after a non-portable object-hash guard failed.

The trainer used ``sha16(participants)`` where ``participants`` is an object-dtype
NumPy array.  ``object_array.tobytes()`` serializes process-local object pointers, so
that guard cannot survive a new Python process.  This scorer leaves the frozen trainer,
manifests and prediction archives untouched.  It verifies the trainer hash, compares
every participant array value-for-value, records stable length-prefixed UTF-8 hashes,
then performs the already frozen scoring protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import eval_fixed_mlp_readout as frozen
from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from eval_metadata_alignment import (
    atomic_json,
    hierarchical_ci,
    load_cohort,
    paired_hierarchical_ci,
)


def stable_string_hash(values: np.ndarray) -> str:
    """Hash strings without object pointers or delimiter ambiguity."""
    h = hashlib.sha256()
    flat = np.asarray(values).reshape(-1)
    h.update(len(flat).to_bytes(8, "little"))
    for value in flat:
        payload = str(value).encode("utf-8")
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
    return h.hexdigest()


def score(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    manifest_path = out / "manifest.json"
    manifest = json.load(open(manifest_path))
    trainer_path = Path(frozen.__file__)
    assert manifest["status"] == "complete_predictions_frozen_before_test_scoring"
    assert manifest["script_sha256"] == frozen.file_sha256(trainer_path), \
        "frozen trainer no longer matches the completed fit manifest"

    d, _, _, tests = load_cohort(args)
    participants = d[UNIT].to_numpy()
    y_all = d["y"].to_numpy(np.int64)
    arms = tuple(manifest["arms"])
    seeds = tuple(manifest["seeds"])
    assert seeds == frozen.SEEDS

    fit_hashes: dict[str, str] = {}
    for arm in arms:
        for seed in seeds:
            path = out / f"fit_{arm}_seed{seed}.npz"
            assert frozen.completed_fit(
                path, participants, arm, str(manifest["backbone"]), seed)
            fit_hashes[path.name] = frozen.file_sha256(path)

    config_path = out / "score_config.json"
    result_path = out / "results.json"
    assert not config_path.exists() and not result_path.exists(), \
        "refusing to overwrite an existing score configuration or result"
    config = {
        "protocol": "frozen E2 fixed nonlinear readout scoring",
        "standing": "exploratory fixed-MLP sensitivity",
        "technical_correction": (
            "trainer manifest cohort_hash used object_array.tobytes and is process-local; "
            "all fit archives are instead checked value-for-value and with stable UTF-8 hash"),
        "legacy_manifest_cohort_hash": manifest["cohort_hash"],
        "stable_participant_hash": stable_string_hash(participants),
        "participant_count": int(len(participants)),
        "trainer_script_sha256": manifest["script_sha256"],
        "scorer_script_sha256": frozen.file_sha256(Path(__file__)),
        "manifest_sha256": frozen.file_sha256(manifest_path),
        "fit_file_sha256": fit_hashes,
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": frozen.BOOTSTRAP_SEED,
        "arms": list(arms),
        "seeds": list(seeds),
    }
    atomic_json(config, config_path)

    probs = {
        arm: frozen.load_fit_matrix(out, participants, arm, seeds)[1]
        for arm in arms
    }
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    result = {
        "standing": "exploratory fixed-MLP sensitivity",
        "technical_correction": config["technical_correction"],
        "score_config_sha256": frozen.file_sha256(config_path),
        "arms": {},
        "comparisons": {},
    }
    for arm, prediction in probs.items():
        result["arms"][arm] = {}
        for name, mask in tests.items():
            y, current = y_all[mask], prediction[:, mask]
            auc, auc_ci = hierarchical_ci(
                current, y, auroc, args.bootstrap, frozen.BOOTSTRAP_SEED)
            nnll, nnll_ci = hierarchical_ci(
                current, y, neg_nll, args.bootstrap, frozen.BOOTSTRAP_SEED + 1)
            result["arms"][arm][name] = {
                "n": int(mask.sum()),
                "auroc": auc,
                "auroc_ci": auc_ci,
                "neg_nll": nnll,
                "neg_nll_ci": nnll_ci,
                "per_seed_auroc": [auroc(y, p) for p in current],
                "per_seed_nll": [float(logloss(y, p).mean()) for p in current],
                "calibration": [calib_diag(y, p) for p in current],
            }
    for arm_a, arm_b, label in frozen.COMPARISONS:
        if arm_a not in probs or arm_b not in probs:
            continue
        key = f"{arm_a}_minus_{arm_b}"
        result["comparisons"][key] = {"interpretation": label}
        for name, mask in tests.items():
            y = y_all[mask]
            result["comparisons"][key][name] = {
                "delta_neg_nll": paired_hierarchical_ci(
                    probs[arm_a][:, mask], probs[arm_b][:, mask], y, neg_nll,
                    args.bootstrap, frozen.BOOTSTRAP_SEED + 2),
                "delta_auroc": paired_hierarchical_ci(
                    probs[arm_a][:, mask], probs[arm_b][:, mask], y, auroc,
                    args.bootstrap, frozen.BOOTSTRAP_SEED + 3),
            }
    result["primary_result_path"] = (
        "comparisons.audio_correct_minus_audio_within_label.matched.delta_neg_nll")
    assert frozen.file_sha256(Path(__file__)) == config["scorer_script_sha256"], \
        "scorer changed after its configuration was frozen"
    atomic_json(result, result_path)
    print(f"SCORED frozen predictions -> {result_path}")


def self_test() -> None:
    a = np.asarray(["x", "", "中文", "a|b"], dtype=object)
    b = np.asarray([str(x) for x in a], dtype=object)
    assert stable_string_hash(a) == stable_string_hash(b)
    assert stable_string_hash(a) != stable_string_hash(a[::-1])
    assert stable_string_hash(np.asarray(["ab", "c"], object)) != \
        stable_string_hash(np.asarray(["a", "bc"], object))
    print("SELF-TEST PASS: stable participant hashing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--out-dir", default="results/mlp_ast")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        score(args)


if __name__ == "__main__":
    main()
