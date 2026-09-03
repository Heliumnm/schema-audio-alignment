"""Resumable, model-score-blind HeAR extraction for the frozen UKCOVID cohort.

The script has four disjoint modes: synthetic window self-test, a stratified repeated
preflight, one participant shard, and a no-model merge.  It never computes a disease,
metadata, retrieval or probe score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SR = 16_000
WINDOW_SAMPLES = 32_000
EMBED_DIM = 512
REVISION = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
SPEC = (f"HeAR-1.0.0@{REVISION}|16k-mono|2s|ceil-full-coverage|right-zero-pad|"
        "window-mean|one-recording-per-participant")


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha16_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()[:16]


def sha16_strings(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()[:16]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def model_hash(path: Path) -> str:
    files = [path / "fingerprint.pb", path / "saved_model.pb",
             path / "variables" / "variables.index",
             path / "variables" / "variables.data-00000-of-00001"]
    if not all(item.is_file() for item in files):
        raise FileNotFoundError(f"incomplete HeAR SavedModel at {path}")
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def window_starts(n_samples: int) -> list[int]:
    if n_samples <= WINDOW_SAMPLES:
        return [0]
    count = int(math.ceil(n_samples / WINDOW_SAMPLES))
    last = n_samples - WINDOW_SAMPLES
    starts = [int(round(index * last / (count - 1))) for index in range(count)]
    starts[0], starts[-1] = 0, last
    assert starts == sorted(starts) and len(starts) == len(set(starts))
    return starts


def fixed_windows(waveform: np.ndarray) -> np.ndarray:
    windows = []
    for start in window_starts(len(waveform)):
        segment = waveform[start:start + WINDOW_SAMPLES]
        if len(segment) < WINDOW_SAMPLES:
            segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)))
        windows.append(np.asarray(segment, dtype=np.float32))
    return np.stack(windows)


def load_audio(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != SR:
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=SR)
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid audio: {path}")
    return waveform


def configure_model(path: Path):
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    tf.random.set_seed(20260903)
    model = tf.saved_model.load(str(path))
    signature = model.signatures.get("serving_default")
    if signature is None:
        raise RuntimeError("HeAR SavedModel lacks serving_default signature")
    return tf, model, signature, model_hash(path)


def infer_windows(windows: np.ndarray, signature, tf, batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(windows), batch_size):
        batch = tf.convert_to_tensor(windows[start:start + batch_size], dtype=tf.float32)
        output = signature(x=batch)
        if "output_0" not in output:
            raise RuntimeError(f"HeAR output_0 absent; available={sorted(output)}")
        values.append(np.asarray(output["output_0"].numpy(), dtype=np.float32))
    matrix = np.concatenate(values)
    if matrix.shape != (len(windows), EMBED_DIM):
        raise RuntimeError(f"unexpected HeAR output shape {matrix.shape}")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise RuntimeError("invalid HeAR window embedding")
    return matrix


def encode_paths(paths: list[Path], signature, tf, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    grouped = [fixed_windows(load_audio(path)) for path in paths]
    counts = np.asarray([len(value) for value in grouped], dtype=np.int32)
    matrix = infer_windows(np.concatenate(grouped), signature, tf, batch_size)
    cursor, output = 0, []
    for count in counts:
        output.append(matrix[cursor:cursor + count].mean(axis=0))
        cursor += int(count)
    assert cursor == len(matrix)
    return np.stack(output).astype(np.float32), counts


def cohort_hash(table: pd.DataFrame) -> str:
    return sha16_strings(table.participant_identifier.astype(str))


def check_indices(table: pd.DataFrame, n: int = 100) -> list[int]:
    selected: list[int] = []

    def add(indices) -> None:
        for index in indices:
            index = int(index)
            if index not in selected:
                selected.append(index)

    for split in ("train", "val", "test"):
        for label in (0, 1):
            add(table.index[(table.splits == split) & (table.y == label)][:5])
    for field in ("in_matched_rebalanced_test", "in_matched_rebalanced_long_test"):
        for label in (0, 1):
            add(table.index[table[field].astype(bool) & (table.y == label)][:5])
    duration = table.cough_length.astype(float)
    add(duration.nsmallest(10).index)
    add(duration.nlargest(10).index)
    add((duration - 2.0).abs().nsmallest(10).index)
    add((duration - 4.0).abs().nsmallest(10).index)
    add(table.sort_values("participant_identifier").index)
    return selected[:n]


def self_test() -> None:
    short = np.linspace(-0.25, 0.5, 16_000, dtype=np.float32)
    short_windows = fixed_windows(short)
    assert short_windows.shape == (1, WINDOW_SAMPLES)
    assert np.array_equal(short_windows[0, :len(short)], short)
    assert np.all(short_windows[0, len(short):] == 0)
    long = np.arange(72_000, dtype=np.float32)
    assert window_starts(len(long)) == [0, 20_000, 40_000]
    long_windows = fixed_windows(long)
    assert long_windows.shape == (3, WINDOW_SAMPLES)
    assert long_windows[0, 0] == 0 and long_windows[-1, -1] == 71_999
    print("HeAR window self-test PASS")


def preflight(args, table: pd.DataFrame, paths: list[Path]) -> None:
    tf, model, signature, checkpoint = configure_model(Path(args.model))
    indices = check_indices(table)
    first, counts = encode_paths([paths[index] for index in indices], signature, tf,
                                 args.batch_size)
    second, second_counts = encode_paths([paths[index] for index in indices], signature, tf,
                                         args.batch_size)
    failures = []
    if first.shape != (len(indices), EMBED_DIM): failures.append("shape")
    if not np.array_equal(first, second): failures.append("repeat_not_bitwise_identical")
    if not np.array_equal(counts, second_counts): failures.append("window_count_changed")
    if not np.isfinite(first).all(): failures.append("nonfinite")
    if np.any(np.linalg.norm(first, axis=1) == 0): failures.append("zero_embedding")
    expected = np.asarray([len(window_starts(max(1, int(round(
        float(table.loc[index, "cough_length"]) * SR))))) for index in indices])
    if not np.array_equal(counts, expected): failures.append("window_rule")
    payload = {"format_version": "hear-ukcovid-preflight-v1", "spec": SPEC,
               "revision": REVISION, "checkpoint_sha256": checkpoint,
               "cohort_sha16": cohort_hash(table), "n": len(indices),
               "shape": list(first.shape), "window_counts": {
                   "min": int(counts.min()), "median": float(np.median(counts)),
                   "max": int(counts.max())},
               "embedding_sha16": sha16_array(first), "bitwise_repeat": not failures,
               "failures": failures, "passed": not failures}
    atomic_json(Path(args.preflight_out), payload)
    if failures:
        raise SystemExit(f"HeAR preflight FAIL: {failures}")
    print(f"HeAR preflight PASS: {args.preflight_out}")


def extract_shard(args, table: pd.DataFrame, paths: list[Path]) -> None:
    report = json.loads(Path(args.preflight_out).read_text())
    checkpoint = model_hash(Path(args.model))
    if not (report.get("passed") and report.get("spec") == SPEC and
            report.get("checkpoint_sha256") == checkpoint and
            report.get("cohort_sha16") == cohort_hash(table)):
        raise RuntimeError("matching HeAR preflight is absent")
    indices = np.array_split(np.arange(len(table)), args.num_shards)[args.shard_index]
    output_dir = Path(args.shard_dir); output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    expected_participants = table.iloc[indices].participant_identifier.astype(str).to_numpy()
    if output.is_file():
        old = np.load(output, allow_pickle=True)
        if (np.array_equal(old["participants"].astype(str), expected_participants) and
                str(old["spec"]) == SPEC and str(old["checkpoint_sha256"]) == checkpoint and
                str(old["cohort_sha16"]) == cohort_hash(table)):
            print(f"verified existing {output}")
            return
        raise RuntimeError(f"refusing to overwrite incompatible {output}")

    tf, model, signature, loaded_checkpoint = configure_model(Path(args.model))
    assert loaded_checkpoint == checkpoint
    embeddings, counts = [], []
    for start in range(0, len(indices), args.participant_batch_size):
        batch_indices = indices[start:start + args.participant_batch_size]
        value, n_windows = encode_paths([paths[index] for index in batch_indices],
                                        signature, tf, args.batch_size)
        embeddings.append(value); counts.append(n_windows)
        done = min(start + len(batch_indices), len(indices))
        if done % 500 < len(batch_indices) or done == len(indices):
            print(f"shard {args.shard_index}: {done}/{len(indices)}", flush=True)
    matrix = np.concatenate(embeddings)
    n_windows = np.concatenate(counts)
    if matrix.shape != (len(indices), EMBED_DIM):
        raise RuntimeError(f"bad shard shape {matrix.shape}")
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(temporary, participants=expected_participants,
                        embeddings=matrix, n_windows=n_windows, indices=indices,
                        spec=SPEC, revision=REVISION, checkpoint_sha256=checkpoint,
                        cohort_sha16=cohort_hash(table), embedding_sha16=sha16_array(matrix))
    os.replace(temporary, output)
    print(f"wrote {output} {matrix.shape} {sha16_array(matrix)}")


def merge(args, table: pd.DataFrame) -> None:
    checkpoint = model_hash(Path(args.model)); parts = []
    for index in range(args.num_shards):
        path = Path(args.shard_dir) / f"shard_{index:02d}_of_{args.num_shards:02d}.npz"
        if not path.is_file(): raise FileNotFoundError(path)
        part = np.load(path, allow_pickle=True)
        if (str(part["spec"]) != SPEC or str(part["checkpoint_sha256"]) != checkpoint or
                str(part["cohort_sha16"]) != cohort_hash(table)):
            raise RuntimeError(f"incompatible shard {path}")
        if str(part["embedding_sha16"]) != sha16_array(part["embeddings"]):
            raise RuntimeError(f"corrupt shard {path}")
        parts.append({key: part[key] for key in ("participants", "embeddings",
                                                 "n_windows", "indices")})
    indices = np.concatenate([part["indices"] for part in parts])
    participants = np.concatenate([part["participants"].astype(str) for part in parts])
    matrix = np.concatenate([part["embeddings"] for part in parts])
    counts = np.concatenate([part["n_windows"] for part in parts])
    if not np.array_equal(indices, np.arange(len(table))):
        raise RuntimeError("shards do not cover cohort exactly once and in order")
    expected = table.participant_identifier.astype(str).to_numpy()
    if not np.array_equal(participants, expected):
        raise RuntimeError("participant order mismatch at merge")
    if matrix.shape != (len(table), EMBED_DIM) or not np.isfinite(matrix).all():
        raise RuntimeError("invalid merged embeddings")
    output = Path(args.merge_out); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old = np.load(output, allow_pickle=True)
        if (np.array_equal(old["participants"].astype(str), expected) and
                sha16_array(old["embeddings"]) == sha16_array(matrix) and
                str(old["checkpoint_sha256"]) == checkpoint):
            print(f"verified existing {output}"); return
        raise RuntimeError(f"refusing to overwrite incompatible {output}")
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(temporary, participants=participants, embeddings=matrix,
                        n_windows=counts, spec=SPEC, revision=REVISION,
                        checkpoint_sha256=checkpoint, cohort_sha16=cohort_hash(table),
                        embedding_sha16=sha16_array(matrix))
    os.replace(temporary, output)
    atomic_json(output.with_name(output.stem + "_audit.json"), {
        "format_version": "hear-ukcovid-extraction-v1", "spec": SPEC,
        "revision": REVISION, "checkpoint_sha256": checkpoint,
        "cohort_sha16": cohort_hash(table), "n_participants": len(table),
        "shape": list(matrix.shape), "embedding_sha16": sha16_array(matrix),
        "finite": True, "nonzero": int(np.sum(np.linalg.norm(matrix, axis=1) > 0)),
        "windows": {"min": int(counts.min()), "median": float(np.median(counts)),
                    "max": int(counts.max()), "total": int(counts.sum())},
        "n_shards": args.num_shards})
    print(f"wrote {output} {matrix.shape} {sha16_array(matrix)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--audio-root", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("--preflight-out", default="results/hear_preflight.json")
    parser.add_argument("--shard-dir", default="results/hear_shards")
    parser.add_argument("--merge-out", default="results/hear_embeddings.npz")
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--participant-batch-size", type=int, default=32)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--extract-shard", action="store_true")
    modes.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not args.model:
        parser.error("--model is required outside --self-test")
    table = pd.read_csv(args.cohort)
    required = {"participant_identifier", "cough_file_name", "cough_length", "splits", "y",
                "in_matched_rebalanced_test", "in_matched_rebalanced_long_test"}
    if not required <= set(table):
        raise ValueError(f"cohort columns missing: {sorted(required - set(table))}")
    if args.merge:
        merge(args, table); return
    if not args.audio_root:
        parser.error("--audio-root is required for preflight/extraction")
    paths = [Path(args.audio_root) / str(value) for value in table.cough_file_name]
    if args.preflight:
        preflight(args, table, paths)
    elif args.extract_shard:
        if not 0 <= args.shard_index < args.num_shards:
            parser.error("--shard-index must be in [0, num-shards)")
        extract_shard(args, table, paths)


if __name__ == "__main__":
    main()

