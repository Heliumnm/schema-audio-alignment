"""Validate the two 500-update train-only S1 runs without computing scientific scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("correct", "within_label", "within_label_sex", "global")
BACKBONES = ("ast", "opera_ct")


def effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    singular = np.sqrt(np.maximum(np.linalg.eigvalsh(gram), 0))[::-1]
    probability = singular / max(float(singular.sum()), 1e-12)
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def sha16(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args(); config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text()); root = Path(config["output_root"])
    if not root.is_absolute(): root = (config_path.parent / root).resolve()
    table = pd.read_csv(root / "private" / "participant_manifest.csv")
    train = (table.splits == "train").to_numpy(); output = {"technical_pass": True,
        "scientific_scores_computed": False, "n_train": int(train.sum()), "backbones": {}}
    failures = []
    for backbone in BACKBONES:
        directory = root / "private" / "s1" / backbone
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["epochs"] != 50 or manifest["seeds"] != [0]:
            failures.append(f"{backbone}:wrong_schedule")
        initial = {manifest["runs"][f"{arm}_seed0"]["init_hash"] for arm in ARMS}
        order = {manifest["runs"][f"{arm}_seed0"]["batch_order_hash"] for arm in ARMS}
        if len(initial) != 1: failures.append(f"{backbone}:different_initialisation")
        if len(order) != 1: failures.append(f"{backbone}:different_batch_order")
        arms = {}
        for arm in ARMS:
            run = manifest["runs"][f"{arm}_seed0"]
            archive = np.load(directory / f"repr_{arm}_seed0.npz", allow_pickle=True)
            values = np.asarray(archive["raw"], dtype=np.float32)
            legal = (values.shape == (len(table), 2560) and np.isfinite(values).all() and
                     np.all(np.linalg.norm(values, axis=1) > 0))
            rank = effective_rank(values[train])
            decreased = float(run["loss_last"]) < float(run["loss_first"])
            if not legal: failures.append(f"{backbone}:{arm}:invalid_output")
            if rank < 2: failures.append(f"{backbone}:{arm}:collapse")
            if not decreased: failures.append(f"{backbone}:{arm}:loss_not_decreased")
            arms[arm] = {"loss_first": float(run["loss_first"]),
                         "loss_last": float(run["loss_last"]),
                         "loss_decreased": decreased, "effective_rank_train": rank,
                         "effective_rank_ceiling": int(min(train.sum() - 1, 2560)),
                         "finite_nonzero": bool(legal), "representation_sha16": sha16(values),
                         "pairing_hash": run["pairing_hash"]}
        output["backbones"][backbone] = {"shared_initialisation": len(initial) == 1,
            "shared_batch_order": len(order) == 1, "arms": arms}
    output["technical_pass"] = not failures; output["failures"] = failures
    destination = root / "public" / "s1_train_only_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(f"S1 {'PASS' if not failures else 'FAIL'}: {destination}")
    if failures: raise SystemExit(3)


if __name__ == "__main__":
    main()
