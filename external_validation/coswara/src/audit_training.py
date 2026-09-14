"""Technical-only audit of all Coswara epoch-500 projector artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("correct", "within_label", "within_label_sex", "global")
BACKBONES = ("ast", "opera_ct", "hear")
SEEDS = (0, 1, 2, 3, 4)


def sha16(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def execute(config_file: str) -> Path:
    config_path = Path(config_file).resolve(); config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    table = pd.read_csv(root / "private" / "participant_manifest.csv")
    participants = table.participant_identifier.astype(str).to_numpy()
    train = np.where(table.splits.eq("train").to_numpy())[0]
    y = table.y.astype(int).to_numpy()[train]
    sex = table.sex.fillna("[MISSING]").astype(str).to_numpy()[train]
    expected = {f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS}
    failures, output, pairing = [], {}, {}
    for backbone in BACKBONES:
        directory = root / "models" / backbone / "alignment"
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            failures.append(f"{backbone}:manifest_missing"); continue
        manifest = json.loads(manifest_path.read_text()); runs = manifest.get("runs", {})
        if set(runs) != expected or manifest.get("seeds") != list(SEEDS) or \
                int(manifest.get("epochs", -1)) != 500:
            failures.append(f"{backbone}:wrong_run_set_or_schedule")
        pairing[backbone], records = {}, {}
        for seed in SEEDS:
            init = {runs.get(f"{arm}_seed{seed}", {}).get("init_hash") for arm in ARMS}
            order = {runs.get(f"{arm}_seed{seed}", {}).get("batch_order_hash") for arm in ARMS}
            if len(init) != 1 or None in init: failures.append(f"{backbone}:seed{seed}:init")
            if len(order) != 1 or None in order: failures.append(f"{backbone}:seed{seed}:order")
            for arm in ARMS:
                key = f"{arm}_seed{seed}"; run = runs.get(key, {})
                archive_path = directory / f"repr_{arm}_seed{seed}.npz"
                checkpoint = directory / f"{arm}_seed{seed}_epoch500.pt"
                if not archive_path.is_file() or not checkpoint.is_file():
                    failures.append(f"{backbone}:{key}:missing"); continue
                archive = np.load(archive_path, allow_pickle=True)
                raw = np.asarray(archive["raw"], np.float32)
                normal = np.asarray(archive["normalized"], np.float32)
                pair = np.asarray(archive["pairing"], np.int64)
                legal = (raw.shape == (len(table), 2560) and normal.shape == raw.shape and
                         np.array_equal(archive["participants"].astype(str), participants) and
                         pair.shape == (len(train),) and
                         np.array_equal(np.sort(pair), np.arange(len(train))) and
                         np.isfinite(raw).all() and np.isfinite(normal).all() and
                         np.allclose(np.linalg.norm(normal, axis=1), 1, atol=2e-5))
                if legal:
                    fixed = pair == np.arange(len(train))
                    legal = bool((np.all(fixed) if arm == "correct" else not np.any(fixed)) and
                                 (arm not in ("within_label", "within_label_sex") or
                                  np.all(y[pair] == y)) and
                                 (arm != "within_label_sex" or np.all(sex[pair] == sex)))
                if not legal: failures.append(f"{backbone}:{key}:illegal")
                pair_hash = sha16(pair); pairing[backbone][key] = pair_hash
                if pair_hash != run.get("pairing_hash") or sha16(raw) != run.get("repr_raw_hash"):
                    failures.append(f"{backbone}:{key}:hash")
                if not float(run.get("loss_last", np.inf)) < float(run.get("loss_first", -np.inf)):
                    failures.append(f"{backbone}:{key}:loss")
                records[key] = {"pairing_hash": pair_hash,
                                "loss_first": float(run.get("loss_first", np.nan)),
                                "loss_last": float(run.get("loss_last", np.nan))}
        output[backbone] = records
    if set(pairing) == set(BACKBONES):
        for key in expected:
            if len({pairing[name].get(key) for name in BACKBONES}) != 1:
                failures.append(f"cross_backbone:{key}:pairing")
    payload = {"format_version": "coswara-training-audit-v1",
               "standing": config["protocol"]["standing"],
               "scientific_scores_computed": False, "target_read": False,
               "technical_pass": not failures, "failures": failures,
               "n_participants": len(table), "n_train": len(train), "runs": output}
    destination = root / "public" / "formal_training_audit.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(f"COSWARA TRAINING {'PASS' if not failures else 'FAIL'}: {destination}")
    if failures: raise SystemExit(3)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    execute(parser.parse_args().config)
