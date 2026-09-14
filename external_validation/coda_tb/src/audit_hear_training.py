"""Technical audit for the post-hoc CODA HeAR backbone extension.

No validation/test score is computed here.  In addition to the normal formal-training
invariants, every HeAR pairing hash must equal the already-completed AST and OPERA-CT
run with the same arm and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("correct", "within_label", "within_label_sex", "global")
SEEDS = (0, 1, 2, 3, 4)


def sha16(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()[:16]


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def execute(config_file: str) -> Path:
    config_path = Path(config_file).expanduser().resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"]).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    table = pd.read_csv(root / "private" / "participant_manifest.csv")
    participants = table.participant_identifier.astype(str).to_numpy()
    train = np.where((table.splits == "train").to_numpy())[0]
    y = table.y.astype(int).to_numpy()[train]
    sex = table.sex.fillna("[MISSING]").astype(str).to_numpy()[train]
    expected = {f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS}
    failures: list[str] = []
    directory = root / "models" / "hear" / "alignment"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    runs = manifest.get("runs", {})
    if set(runs) != expected:
        failures.append("hear:wrong_run_set")
    if manifest.get("seeds") != list(SEEDS) or int(manifest.get("epochs", -1)) != 500:
        failures.append("hear:wrong_schedule")
    if int(manifest.get("n_train", -1)) != len(train) or int(manifest.get("batch", -1)) != 64:
        failures.append("hear:wrong_train_or_batch_count")
    if int(manifest.get("audio_input_dim", -1)) != 512:
        failures.append("hear:wrong_audio_input_dim")

    reference = {}
    for backbone in ("ast", "opera_ct"):
        result = root / "public" / f"formal_results_{backbone}.json"
        other = root / "models" / backbone / "alignment" / "manifest.json"
        if not result.is_file() or not other.is_file():
            failures.append(f"{backbone}:primary_result_or_manifest_missing")
            continue
        reference[backbone] = json.loads(other.read_text()).get("runs", {})

    records = {}
    for seed in SEEDS:
        initial = {runs.get(f"{arm}_seed{seed}", {}).get("init_hash") for arm in ARMS}
        order = {runs.get(f"{arm}_seed{seed}", {}).get("batch_order_hash") for arm in ARMS}
        if len(initial) != 1 or None in initial:
            failures.append(f"hear:seed{seed}:initialisation_not_shared")
        if len(order) != 1 or None in order:
            failures.append(f"hear:seed{seed}:batch_order_not_shared")
        for arm in ARMS:
            key = f"{arm}_seed{seed}"
            run = runs.get(key, {})
            archive_path = directory / f"repr_{arm}_seed{seed}.npz"
            checkpoint_path = directory / f"{arm}_seed{seed}_epoch500.pt"
            if not archive_path.is_file() or not checkpoint_path.is_file():
                failures.append(f"hear:{key}:missing_file")
                continue
            archive = np.load(archive_path, allow_pickle=True)
            raw = np.asarray(archive["raw"], dtype=np.float32)
            normalized = np.asarray(archive["normalized"], dtype=np.float32)
            pairing = np.asarray(archive["pairing"], dtype=np.int64)
            legal = (
                raw.shape == (len(table), 2560)
                and normalized.shape == raw.shape
                and np.array_equal(archive["participants"].astype(str), participants)
                and int(archive["seed"]) == seed
                and str(archive["arm"]) == arm
                and int(archive["epochs"]) == 500
                and np.isfinite(raw).all() and np.isfinite(normalized).all()
                and np.all(np.linalg.norm(raw, axis=1) > 0)
                and np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=2e-5)
            )
            if not legal:
                failures.append(f"hear:{key}:invalid_representation")
            if pairing.shape != (len(train),) or not np.array_equal(
                    np.sort(pairing), np.arange(len(train))):
                failures.append(f"hear:{key}:pairing_not_bijection")
            else:
                fixed = pairing == np.arange(len(train))
                if arm == "correct" and not np.all(fixed):
                    failures.append(f"hear:{key}:correct_not_identity")
                if arm != "correct" and np.any(fixed):
                    failures.append(f"hear:{key}:shuffle_fixed_point")
                if arm in ("within_label", "within_label_sex") and not np.all(
                        y[pairing] == y):
                    failures.append(f"hear:{key}:within_crosses_label")
                if arm == "within_label_sex" and not np.all(sex[pairing] == sex):
                    failures.append(f"hear:{key}:within_sex_crosses_sex")
            raw_hash, norm_hash, pair_hash = sha16(raw), sha16(normalized), sha16(pairing)
            if raw_hash != run.get("repr_raw_hash") or norm_hash != run.get("repr_norm_hash"):
                failures.append(f"hear:{key}:representation_hash")
            if pair_hash != run.get("pairing_hash"):
                failures.append(f"hear:{key}:pairing_hash")
            for backbone, other_runs in reference.items():
                if pair_hash != other_runs.get(key, {}).get("pairing_hash"):
                    failures.append(f"hear_vs_{backbone}:{key}:pairing_differs")
            if not float(run.get("loss_last", np.inf)) < float(run.get("loss_first", -np.inf)):
                failures.append(f"hear:{key}:loss_not_decreased")
            records[key] = {"representation_sha16": raw_hash,
                            "pairing_hash": pair_hash,
                            "loss_first": float(run.get("loss_first", np.nan)),
                            "loss_last": float(run.get("loss_last", np.nan)),
                            "finite_nonzero_unit_normalized": bool(legal),
                            "checkpoint_bytes": int(checkpoint_path.stat().st_size)}

    payload = {
        "format_version": "coda-hear-training-audit-v1",
        "standing": "post-hoc third-backbone robustness extension",
        "scientific_scores_computed": False,
        "target_read": False,
        "n_participants": int(len(table)),
        "n_train": int(len(train)),
        "audio_input_dim": int(manifest.get("audio_input_dim", -1)),
        "schedule": {"arms": list(ARMS), "seeds": list(SEEDS), "epochs": 500},
        "runs": records,
        "pairings_equal_existing_ast_and_opera": not any(
            "pairing_differs" in failure for failure in failures),
        "technical_pass": not failures,
        "failures": failures,
    }
    destination = root / "public" / "formal_training_audit_hear.json"
    atomic_json(destination, payload)
    print(f"HEAR FORMAL TRAINING {'PASS' if not failures else 'FAIL'}: {destination}")
    if failures:
        raise SystemExit(3)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
