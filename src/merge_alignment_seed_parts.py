"""Merge independently scheduled metadata-alignment seed groups without recomputation.

Each seed is mathematically independent: ``train_metadata_alignment.py`` resets model,
batch and dropout RNGs from that seed.  This utility verifies that all part manifests use
the same frozen inputs/protocol, validates every representation, then hard-links the
immutable files into one evaluator-compatible directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ARMS = ("correct", "within_label", "global")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()

    expected_seeds = list(args.seeds)
    assert len(expected_seeds) == len(set(expected_seeds))
    parts = [Path(path) for path in args.parts]
    manifests = [json.load(open(path / "manifest.json")) for path in parts]
    reference = manifests[0]
    for manifest in manifests:
        for key in ("source_commit", "n_train", "batch", "epochs", "input_hashes"):
            assert manifest[key] == reference[key], f"part mismatch: {key}"
        assert manifest["epochs"] == args.epochs

    found_seeds = [seed for manifest in manifests for seed in manifest["seeds"]]
    assert sorted(found_seeds) == sorted(expected_seeds), (found_seeds, expected_seeds)
    assert len(found_seeds) == len(set(found_seeds)), "a seed appears in multiple parts"

    out = Path(args.out_dir)
    assert not out.exists(), f"refusing to overwrite {out}"
    out.mkdir(parents=True)
    merged_runs: dict[str, dict] = {}
    file_records: dict[str, dict] = {}
    try:
        for part, manifest in zip(parts, manifests):
            part_seeds = list(manifest["seeds"])
            assert set(manifest["runs"]) == {
                f"{arm}_seed{seed}" for seed in part_seeds for arm in ARMS
            }
            for seed in part_seeds:
                for arm in ARMS:
                    run_key = f"{arm}_seed{seed}"
                    assert run_key not in merged_runs
                    repr_name = f"repr_{arm}_seed{seed}.npz"
                    ckpt_name = f"{arm}_seed{seed}_epoch{args.epochs}.pt"
                    source_repr, source_ckpt = part / repr_name, part / ckpt_name
                    assert source_repr.is_file() and source_ckpt.is_file()
                    with np.load(source_repr, allow_pickle=True) as z:
                        assert int(np.asarray(z["seed"]).item()) == seed
                        assert str(np.asarray(z["arm"]).item()) == arm
                        assert int(np.asarray(z["epochs"]).item()) == args.epochs
                        assert str(np.asarray(z["source_commit"]).item()) == reference["source_commit"]
                        assert z["raw"].shape[0] == z["participants"].shape[0]
                        assert z["raw"].shape == z["normalized"].shape
                        assert np.isfinite(z["raw"]).all() and np.isfinite(z["normalized"]).all()
                    os.link(source_repr, out / repr_name)
                    os.link(source_ckpt, out / ckpt_name)
                    file_records[run_key] = {
                        "part": str(part),
                        "repr_sha256": sha256_file(source_repr),
                        "checkpoint_sha256": sha256_file(source_ckpt),
                    }
                    merged_runs[run_key] = manifest["runs"][run_key]

        merged = {
            "source_commit": reference["source_commit"],
            "n_train": reference["n_train"],
            "batch": reference["batch"],
            "epochs": reference["epochs"],
            "seeds": expected_seeds,
            "input_hashes": reference["input_hashes"],
            "runs": merged_runs,
            "distributed_execution": {
                "seed_parts": [str(path) for path in parts],
                "file_records": file_records,
                "scientific_protocol_changed": False,
            },
        }
        tmp = out / "manifest.json.tmp"
        with open(tmp, "w") as stream:
            json.dump(merged, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, out / "manifest.json")
    except Exception:
        # Keep any partial directory visible for audit; never silently retry into it.
        raise
    print(f"MERGE PASS: {len(expected_seeds)} seeds x {len(ARMS)} arms -> {out}")


if __name__ == "__main__":
    main()
