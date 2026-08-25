"""Run the frozen three-pairing projector training after a GO data gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import REPO_ROOT, load_config, output_paths


UNLOCK = "FROZEN_PROTOCOL_AND_DATA_GATE_GO"


def array_sha16(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def run(command: list[str], cwd: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


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
        expected = {f"{arm}_seed{seed}" for arm in ("correct", "within_label", "global")
                    for seed in seeds}
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
            return output
        raise RuntimeError(f"refusing to overwrite incompatible training directory {output}")

    script = REPO_ROOT / "src" / "train_metadata_alignment.py"
    scratch = paths["private"] / "rehearsal" / backbone
    scratch.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(script), "--cohort", str(cohort), "--texts", str(cohort),
            "--emb", str(audio), "--text_emb", str(text)]
    run(base + ["--rehearsal", "--device", settings.get("device", "cuda")], scratch)
    run(base + ["--out_dir", str(output), "--seeds", *map(str, seeds),
                "--epochs", str(epochs), "--device", settings.get("device", "cuda")],
        REPO_ROOT)
    print(f"{backbone}: formal alignment complete")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ast", "opera_ct"), required=True)
    args = parser.parse_args()
    execute(args.config, args.backbone)


if __name__ == "__main__":
    main()
