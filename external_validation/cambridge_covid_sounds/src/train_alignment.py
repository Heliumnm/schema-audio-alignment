"""Run the locked four-pairing projector training after a GO data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import REPO_ROOT, load_config, output_paths


UNLOCK = "FROZEN_PROTOCOL_AND_DATA_GATE_GO"
ARMS = ("correct", "within_label", "within_label_sex", "global")


def array_sha16(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def run(command: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def audit_alignment(output: Path, cohort: Path, seeds: list[int], epochs: int) -> None:
    """Fail closed unless the four frozen pairing trajectories are complete and legal."""
    table = pd.read_csv(cohort)
    participants = table.participant_identifier.astype(str).to_numpy()
    train = np.where(table.splits.astype(str).eq("train").to_numpy())[0]
    labels = table.y.to_numpy()[train]
    sex = table.sex.fillna("[MISSING]").astype(str).to_numpy()[train]
    keys = np.asarray(list(zip(labels.astype(str), sex)), dtype=object)
    counts = {}
    for key in map(tuple, keys.tolist()):
        counts[key] = counts.get(key, 0) + 1
    singleton = {key for key, count in counts.items() if count < 2}
    keep = np.asarray([tuple(key) not in singleton for key in keys.tolist()], dtype=bool)
    train, labels, sex = train[keep], labels[keep], sex[keep]

    manifest = json.loads((output / "manifest.json").read_text())
    expected = {f"{arm}_seed{seed}" for arm in ARMS for seed in seeds}
    if (set(manifest.get("runs", {})) != expected or manifest.get("seeds") != seeds or
            int(manifest.get("epochs", -1)) != epochs or
            int(manifest.get("n_train", -1)) != len(train)):
        raise RuntimeError(f"incomplete or incompatible four-arm manifest: {output}")
    for seed in seeds:
        runs = manifest["runs"]
        if len({runs[f"{arm}_seed{seed}"]["init_hash"] for arm in ARMS}) != 1:
            raise RuntimeError(f"seed {seed}: arm initialisations differ")
        if len({runs[f"{arm}_seed{seed}"]["batch_order_hash"] for arm in ARMS}) != 1:
            raise RuntimeError(f"seed {seed}: arm batch orders differ")
        for arm in ARMS:
            archive = np.load(output / f"repr_{arm}_seed{seed}.npz", allow_pickle=True)
            if not np.array_equal(archive["participants"].astype(str), participants):
                raise RuntimeError(f"{arm} seed {seed}: participant order mismatch")
            pairing = np.asarray(archive["pairing"], dtype=np.int64)
            if (pairing.shape != (len(train),) or
                    not np.array_equal(np.sort(pairing), np.arange(len(train)))):
                raise RuntimeError(f"{arm} seed {seed}: pairing is not a bijection")
            fixed = pairing == np.arange(len(train))
            if arm == "correct" and not np.all(fixed):
                raise RuntimeError(f"{arm} seed {seed}: not the identity pairing")
            if arm != "correct" and np.any(fixed):
                raise RuntimeError(f"{arm} seed {seed}: self-pairing found")
            if arm in ("within_label", "within_label_sex") and not np.all(
                    labels[pairing] == labels):
                raise RuntimeError(f"{arm} seed {seed}: disease label crossed")
            if arm == "within_label_sex" and not np.all(sex[pairing] == sex):
                raise RuntimeError(f"{arm} seed {seed}: recorded sex crossed")
    print(f"{output.name}: four-arm pairing audit PASS", flush=True)


def execute(config_file: str, backbone: str) -> Path:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    gate = json.loads((paths["public"] / "data_gate.json").read_text())
    if gate["verdict"] != "GO":
        raise RuntimeError("data gate is not GO; projector training is forbidden")
    if config["protocol"].get("formal_unlock") != UNLOCK:
        raise RuntimeError(
            f"set protocol.formal_unlock exactly to {UNLOCK!r} only after reviewing the gate")
    cohort = paths["private"] / "participant_manifest.csv"
    text = paths["models"] / "text_embeddings.npz"
    audio = paths["models"] / backbone / "raw_embeddings.npz"
    for path in (cohort, text, audio):
        if not path.is_file():
            raise FileNotFoundError(path)
    text_cache = np.load(text, allow_pickle=True)
    if text_cache["unique_embeddings"].shape[1] != 2560:
        raise RuntimeError("the frozen RespiraMFM projector requires 2560-D Phi-2 targets")
    output = paths["models"] / backbone / "alignment"
    output.mkdir(parents=True, exist_ok=True)
    settings = config["models"]["alignment"]
    seeds = list(map(int, settings.get("seeds", [0, 1, 2, 3, 4])))
    epochs = int(settings.get("epochs", 500))
    if seeds != [0, 1, 2, 3, 4] or epochs != 500:
        raise RuntimeError("formal external protocol is frozen to seeds 0..4 and 500 epochs")
    manifest = output / "manifest.json"
    if manifest.is_file():
        existing = json.loads(manifest.read_text())
        expected = {f"{arm}_seed{seed}" for arm in ARMS for seed in seeds}
        audio_cache = np.load(audio, allow_pickle=True)
        current_hashes = {
            "audio": array_sha16(audio_cache["embeddings"]),
            "text_unique": array_sha16(text_cache["unique_embeddings"]),
            "text_id": array_sha16(text_cache["text_id"]),
        }
        if (int(existing.get("epochs", -1)) == epochs and
                expected <= set(existing.get("runs", {})) and
                existing.get("input_hashes") == current_hashes):
            print(f"{backbone}: verified existing formal alignment {manifest}")
            audit_alignment(output, cohort, seeds, epochs)
            return output
        raise RuntimeError(f"refusing to overwrite incompatible training directory {output}")

    script = REPO_ROOT / "src" / "train_metadata_alignment.py"
    scratch = paths["private"] / "rehearsal" / backbone
    scratch.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(script), "--cohort", str(cohort), "--texts", str(cohort),
            "--emb", str(audio), "--text_emb", str(text),
            "--include-within-label-sex", "--sex-column", "sex",
            "--within-label-sex-support", "drop-singleton-strata"]
    run(base + ["--rehearsal", "--device", settings.get("device", "cuda")], scratch)
    run(base + ["--out_dir", str(output), "--seeds", *map(str, seeds),
                "--epochs", str(epochs), "--device", settings.get("device", "cuda")],
        REPO_ROOT)
    audit_alignment(output, cohort, seeds, epochs)
    print(f"{backbone}: formal alignment complete")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ast", "opera_ct", "hear"), required=True)
    args = parser.parse_args()
    execute(args.config, args.backbone)


if __name__ == "__main__":
    main()
