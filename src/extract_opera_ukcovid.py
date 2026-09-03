"""Frozen OPERA-CT extraction for the UKCOVID audio cohort.

Modes are deliberately separated:

* ``--self-test`` checks pure window logic without loading data or a model;
* ``--preflight`` runs a stratified 100-recording technical gate;
* ``--extract-shard`` writes one resumable GPU shard after a passing preflight;
* ``--merge`` verifies and merges all completed shards in cohort order.

No disease metric is computed anywhere in this file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SPEC_VERSION = "opera-ct-ukcovid-v1-2026-08-20"
EXPECTED_CHECKPOINT_SHA256 = (
    "83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae")
SR = 16_000
WINDOW_S = 8.0
WINDOW_SAMPLES = int(SR * WINDOW_S)
EMBED_DIM = 768
EXPECTED_MEL_SHAPE = (251, 64)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def sha16_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def sha16_strings(values) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as stream:
        json.dump(obj, stream, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def window_starts(n_samples: int) -> list[int]:
    """Uniform fixed windows covering the complete post-trim signal."""
    if n_samples <= WINDOW_SAMPLES:
        return [0]
    n_windows = int(math.ceil(n_samples / WINDOW_SAMPLES))
    last = n_samples - WINDOW_SAMPLES
    starts = [int(round(i * last / (n_windows - 1))) for i in range(n_windows)]
    starts[0], starts[-1] = 0, last
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert starts[0] == 0 and starts[-1] + WINDOW_SAMPLES == n_samples
    return starts


def activate_opera_namespace(opera_root: Path) -> None:
    """Make ``src`` resolve to OPERA, even when another project owns that name.

    OPERA uses the generic top-level package name ``src``.  The external audit also has
    a directory named ``src`` and Slurm launches from inside that directory, so changing
    ``PYTHONPATH`` alone is insufficient once the wrong package has entered
    ``sys.modules``.  Clear only that ambiguous namespace, put the verified OPERA root
    first, then assert the imported package actually came from OPERA.
    """

    import sys

    root = Path(opera_root).expanduser().resolve()
    sentinel = root / "src" / "model" / "models_cola.py"
    utility = root / "src" / "util.py"
    if not sentinel.is_file() or not utility.is_file():
        raise FileNotFoundError(
            f"OPERA source layout is incomplete: expected {sentinel} and {utility}")
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    root_string = str(root)
    sys.path[:] = [root_string] + [entry for entry in sys.path if entry != root_string]
    importlib.invalidate_caches()
    package = importlib.import_module("src")
    locations = [Path(item).resolve() for item in getattr(package, "__path__", [])]
    expected = (root / "src").resolve()
    if expected not in locations:
        raise ImportError(
            f"src resolved outside OPERA: expected {expected}, got {locations}")


def configure_opera(opera_root: Path, checkpoint: Path, device: str):
    import torch

    activate_opera_namespace(opera_root)
    from src.model.models_cola import Cola
    from src.util import get_entire_signal_librosa

    actual = sha256_file(checkpoint)
    assert actual == EXPECTED_CHECKPOINT_SHA256, (actual, EXPECTED_CHECKPOINT_SHA256)
    model = Cola(encoder="htsat")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, model, get_entire_signal_librosa, actual


def load_native_windows(path: Path, get_entire_signal_librosa) -> tuple[np.ndarray, dict]:
    import librosa

    audio, _ = librosa.load(path, sr=SR, mono=True)
    if not np.isfinite(audio).all() or len(audio) == 0:
        raise ValueError(f"invalid decoded waveform: {path}")
    trimmed, interval = librosa.effects.trim(
        audio, frame_length=SR // 10, hop_length=SR // 20)
    if len(trimmed) == 0:
        raise ValueError(f"silence trim produced an empty waveform: {path}")
    starts = window_starts(len(trimmed))
    mels = []
    for start in starts:
        segment = trimmed[start:start + WINDOW_SAMPLES]
        mel = get_entire_signal_librosa(
            "", "", input_sec=WINDOW_S, sample_rate=SR, spectrogram=True,
            pad=True, from_cycle=True, yt=segment, types="repeat")
        mel = np.asarray(mel, dtype=np.float32)
        if mel.shape != EXPECTED_MEL_SHAPE or not np.isfinite(mel).all():
            raise ValueError(f"unexpected mel tensor {mel.shape} for {path}")
        mels.append(mel)
    return np.stack(mels), {
        "decoded_samples": int(len(audio)),
        "trimmed_samples": int(len(trimmed)),
        "trim_interval": [int(interval[0]), int(interval[1])],
        "starts": starts,
        "n_windows": len(starts),
    }


def embed_participant_batch(rows: list, audio_root: Path, torch, model,
                            get_entire_signal_librosa, device: str):
    all_mels, counts, metadata = [], [], []
    for row in rows:
        path = audio_root / str(row.cough_file_name)
        mels, info = load_native_windows(path, get_entire_signal_librosa)
        all_mels.append(mels)
        counts.append(len(mels))
        metadata.append(info)
    tensor = torch.tensor(np.concatenate(all_mels), dtype=torch.float32, device=device)
    with torch.no_grad():
        features = model.extract_feature(tensor, EMBED_DIM).float().cpu().numpy()
    assert features.shape == (sum(counts), EMBED_DIM) and np.isfinite(features).all()
    output, cursor = [], 0
    for count in counts:
        output.append(features[cursor:cursor + count].mean(axis=0))
        cursor += count
    out = np.stack(output).astype(np.float32)
    assert out.shape == (len(rows), EMBED_DIM) and np.isfinite(out).all()
    return out, metadata


def build_check_set(cohort: pd.DataFrame, artefacts: pd.DataFrame,
                    data_dir: Path, n: int = 100) -> pd.DataFrame:
    meta = pd.read_csv(data_dir / "participant_metadata.csv", low_memory=False)
    keep_meta = ["participant_identifier", "recruitment_source"]
    d = (cohort.merge(artefacts, on="participant_identifier", validate="one_to_one")
         .merge(meta[keep_meta], on="participant_identifier", validate="one_to_one"))
    d["clipped"] = d["clip_frac"] > 0
    rng, chosen = np.random.RandomState(20260820), []

    def take(mask, count, cell):
        available = d[mask & ~d.participant_identifier.isin([pid for pid, _ in chosen])]
        if len(available) == 0:
            return
        selected = rng.choice(len(available), min(count, len(available)), replace=False)
        chosen.extend((pid, cell) for pid in available.iloc[selected].participant_identifier)

    take(d.duration_s <= d.duration_s.quantile(0.001), 8, "shortest")
    take((d.duration_s > 7.5) & (d.duration_s <= 8.0), 8, "below_8s")
    take((d.duration_s > 8.0) & (d.duration_s < 9.0), 8, "above_8s")
    take(d.duration_s >= d.duration_s.quantile(0.999), 8, "longest")
    take(d.clipped, 10, "clipped")
    take(~d.clipped, 10, "unclipped")
    take(d.recruitment_source == "REACT", 8, "REACT")
    take(d.recruitment_source == "Test and Trace", 8, "TestAndTrace")
    take(d.splits == "test", 6, "standard_test")
    take(d.in_matched_rebalanced_test == True, 6, "matched")  # noqa: E712
    take(d.in_matched_rebalanced_long_test == True, 6, "matched_long")  # noqa: E712
    take(d.participant_identifier.notna(), n - len(chosen), "filler")
    return pd.DataFrame(chosen[:n], columns=["participant_identifier", "cell"])


def run_preflight(args) -> None:
    cohort = pd.read_csv(args.cohort)
    artefacts = pd.read_csv(args.artefacts)
    lookup = cohort.set_index("participant_identifier")
    check = build_check_set(cohort, artefacts, Path(args.data))
    torch, model, preprocess, ckpt_hash = configure_opera(
        Path(args.opera_root), Path(args.checkpoint), args.device)
    records, failures = [], []
    for i, item in enumerate(check.itertuples()):
        row = lookup.loc[item.participant_identifier]
        f1, m1 = embed_participant_batch([row], Path(args.audio_root), torch, model,
                                         preprocess, args.device)
        f2, m2 = embed_participant_batch([row], Path(args.audio_root), torch, model,
                                         preprocess, args.device)
        info = m1[0]
        expected = max(1, int(math.ceil(info["trimmed_samples"] / WINDOW_SAMPLES)))
        record = {
            "participant_identifier": item.participant_identifier,
            "cell": item.cell,
            "n_windows": info["n_windows"],
            "expected_windows": expected,
            "trimmed_duration_s": info["trimmed_samples"] / SR,
            "covers_head": info["starts"][0] == 0,
            "covers_tail": (info["starts"][-1] + WINDOW_SAMPLES >=
                             info["trimmed_samples"]),
            "finite": bool(np.isfinite(f1).all()),
            "nonzero": bool(np.linalg.norm(f1) > 0),
            "bitwise_repeat": bool(np.array_equal(f1, f2) and m1 == m2),
        }
        if not all([record["n_windows"] == expected, record["covers_head"],
                    record["covers_tail"], record["finite"], record["nonzero"],
                    record["bitwise_repeat"]]):
            failures.append(record)
        records.append(record)
        if (i + 1) % 20 == 0:
            print(f"preflight {i+1}/{len(check)}", flush=True)
    out = {
        "spec": SPEC_VERSION,
        "passed": not failures,
        "n": len(records),
        "checkpoint_sha256": ckpt_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "cohort_sha256": sha256_file(Path(args.cohort)),
        "failures": failures,
        "records": records,
    }
    atomic_json(out, Path(args.preflight_out))
    print(f"PREFLIGHT {'PASS' if out['passed'] else 'FAIL'}: {len(records)} records")
    if failures:
        raise SystemExit(3)


def shard_paths(out_dir: Path, shard: int) -> dict[str, Path]:
    stem = out_dir / f"shard{shard:02d}"
    return {
        "embedding": Path(str(stem) + "_embeddings.npy"),
        "windows": Path(str(stem) + "_n_windows.npy"),
        "duration": Path(str(stem) + "_trimmed_duration_s.npy"),
        "progress": Path(str(stem) + "_progress.json"),
        "lock": Path(str(stem) + ".lock"),
    }


def run_shard(args) -> None:
    preflight = json.load(open(args.preflight_out))
    assert preflight["passed"] and preflight["spec"] == SPEC_VERSION
    assert preflight["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert preflight["script_sha256"] == sha256_file(Path(__file__))
    assert preflight["cohort_sha256"] == sha256_file(Path(args.cohort))
    assert 0 <= args.shard < args.num_shards

    cohort = pd.read_csv(args.cohort)
    indices = np.arange(args.shard, len(cohort), args.num_shards, dtype=np.int64)
    rows = [cohort.iloc[i] for i in indices]
    participant_ids_hash = sha16_strings(
        cohort.iloc[indices].participant_identifier.astype(str))
    paths = shard_paths(Path(args.out_dir), args.shard)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    lock = open(paths["lock"], "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    if paths["progress"].exists():
        progress = json.load(open(paths["progress"]))
        assert progress["spec"] == SPEC_VERSION
        assert progress["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        assert progress["script_sha256"] == sha256_file(Path(__file__))
        assert progress["cohort_sha256"] == sha256_file(Path(args.cohort))
        assert progress["shard"] == args.shard and progress["num_shards"] == args.num_shards
        assert progress["participant_ids_sha16"] == participant_ids_hash
        completed = int(progress["completed"])
        embeddings = np.lib.format.open_memmap(paths["embedding"], mode="r+")
        windows = np.lib.format.open_memmap(paths["windows"], mode="r+")
        durations = np.lib.format.open_memmap(paths["duration"], mode="r+")
    else:
        completed = 0
        embeddings = np.lib.format.open_memmap(
            paths["embedding"], mode="w+", dtype=np.float32,
            shape=(len(indices), EMBED_DIM))
        windows = np.lib.format.open_memmap(
            paths["windows"], mode="w+", dtype=np.int16, shape=(len(indices),))
        durations = np.lib.format.open_memmap(
            paths["duration"], mode="w+", dtype=np.float32, shape=(len(indices),))

    torch, model, preprocess, ckpt_hash = configure_opera(
        Path(args.opera_root), Path(args.checkpoint), args.device)
    script_hash = sha256_file(Path(__file__))
    cohort_hash = sha256_file(Path(args.cohort))
    for start in range(completed, len(rows), args.batch_participants):
        end = min(start + args.batch_participants, len(rows))
        features, info = embed_participant_batch(
            rows[start:end], Path(args.audio_root), torch, model, preprocess, args.device)
        embeddings[start:end] = features
        windows[start:end] = [item["n_windows"] for item in info]
        durations[start:end] = [item["trimmed_samples"] / SR for item in info]
        embeddings.flush(); windows.flush(); durations.flush()
        progress = {
            "spec": SPEC_VERSION,
            "complete": end == len(rows),
            "completed": end,
            "n_rows": len(rows),
            "shard": args.shard,
            "num_shards": args.num_shards,
            "checkpoint_sha256": ckpt_hash,
            "script_sha256": script_hash,
            "cohort_sha256": cohort_hash,
            "participant_ids_sha16": participant_ids_hash,
        }
        atomic_json(progress, paths["progress"])
        if end % 500 < args.batch_participants or end == len(rows):
            print(f"shard {args.shard}: {end}/{len(rows)}", flush=True)
    assert np.isfinite(np.asarray(embeddings)).all()
    assert np.all(np.asarray(windows) >= 1) and np.all(np.asarray(durations) > 0)
    progress.update({
        "complete": True,
        "embedding_file_sha256": sha256_file(paths["embedding"]),
        "windows_file_sha256": sha256_file(paths["windows"]),
        "duration_file_sha256": sha256_file(paths["duration"]),
        "embedding_values_sha16": sha16_array(np.asarray(embeddings)),
    })
    atomic_json(progress, paths["progress"])
    print(f"SHARD {args.shard} COMPLETE", flush=True)


def run_merge(args) -> None:
    out = Path(args.merge_out)
    assert not out.exists(), f"refusing to overwrite {out}"
    cohort = pd.read_csv(args.cohort)
    n = len(cohort)
    embeddings = np.empty((n, EMBED_DIM), dtype=np.float32)
    windows = np.empty(n, dtype=np.int16)
    durations = np.empty(n, dtype=np.float32)
    seen = np.zeros(n, dtype=bool)
    shard_records = []
    for shard in range(args.num_shards):
        paths = shard_paths(Path(args.out_dir), shard)
        progress = json.load(open(paths["progress"]))
        assert progress["complete"] and progress["completed"] == progress["n_rows"]
        assert progress["spec"] == SPEC_VERSION
        assert progress["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
        assert progress["script_sha256"] == sha256_file(Path(__file__))
        assert progress["cohort_sha256"] == sha256_file(Path(args.cohort))
        assert progress["shard"] == shard and progress["num_shards"] == args.num_shards
        for key, file_key in (("embedding", "embedding_file_sha256"),
                              ("windows", "windows_file_sha256"),
                              ("duration", "duration_file_sha256")):
            assert sha256_file(paths[key]) == progress[file_key]
        idx = np.arange(shard, n, args.num_shards, dtype=np.int64)
        assert progress["participant_ids_sha16"] == sha16_strings(
            cohort.iloc[idx].participant_identifier.astype(str))
        E = np.load(paths["embedding"], mmap_mode="r")
        W = np.load(paths["windows"], mmap_mode="r")
        D = np.load(paths["duration"], mmap_mode="r")
        assert E.shape == (len(idx), EMBED_DIM) and len(W) == len(D) == len(idx)
        embeddings[idx], windows[idx], durations[idx], seen[idx] = E, W, D, True
        shard_records.append(progress)
    assert seen.all() and np.isfinite(embeddings).all() and np.all(windows >= 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp.npz")
    np.savez_compressed(
        tmp,
        participants=cohort.participant_identifier.astype(str).to_numpy(),
        embeddings=embeddings,
        n_windows=windows,
        trimmed_duration_s=durations,
        spec=SPEC_VERSION,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        script_sha256=sha256_file(Path(__file__)),
        cohort_sha256=sha256_file(Path(args.cohort)),
        num_shards=args.num_shards,
    )
    os.replace(tmp, out)
    audit = {
        "spec": SPEC_VERSION,
        "n_participants": n,
        "embedding_shape": list(embeddings.shape),
        "finite": bool(np.isfinite(embeddings).all()),
        "nonzero": int(np.sum(np.linalg.norm(embeddings, axis=1) > 0)),
        "embedding_norm": {
            "min": float(np.linalg.norm(embeddings, axis=1).min()),
            "median": float(np.median(np.linalg.norm(embeddings, axis=1))),
            "max": float(np.linalg.norm(embeddings, axis=1).max()),
        },
        "window_counts": {str(k): int(v) for k, v in zip(
            *np.unique(windows, return_counts=True))},
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "script_sha256": sha256_file(Path(__file__)),
        "cohort_sha256": sha256_file(Path(args.cohort)),
        "output_sha256": sha256_file(out),
        "embedding_values_sha16": sha16_array(embeddings),
        "shards": shard_records,
        "disease_metric_computed": False,
    }
    atomic_json(audit, Path(args.merge_audit))
    print(f"MERGE COMPLETE: {embeddings.shape} -> {out}")


def self_test() -> None:
    for length, expected in [
        (1, [0]),
        (WINDOW_SAMPLES, [0]),
        (WINDOW_SAMPLES + 1, [0, 1]),
        (2 * WINDOW_SAMPLES, [0, WINDOW_SAMPLES]),
        (3 * WINDOW_SAMPLES + 17, [0, 85339, 170678, 256017]),
    ]:
        actual = window_starts(length)
        assert actual == expected, (length, actual, expected)
        if length > WINDOW_SAMPLES:
            assert actual[-1] + WINDOW_SAMPLES == length
    print("SELF-TEST PASS: OPERA full-coverage window logic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--artefacts", default="results/artefact_features.csv")
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument(
        "--audio-root",
        default="/mnt/hd/data_heliu/resp_datasets/ukcovid/audio/audio")
    ap.add_argument("--opera-root",
                    default="/mnt/hd/data_heliu/icbhi_pathology_fidelity/OPERA")
    ap.add_argument("--checkpoint", default=(
        "/mnt/hd/data_heliu/icbhi_pathology_fidelity/OPERA/cks/model/"
        "encoder-operaCT.ckpt"))
    ap.add_argument("--preflight-out", default="results/opera_ct_preflight.json")
    ap.add_argument("--out-dir", default="results/opera_ct_shards")
    ap.add_argument("--merge-out", default="results/opera_ct_embeddings.npz")
    ap.add_argument("--merge-audit", default="results/opera_ct_embeddings_audit.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--batch-participants", type=int, default=24)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--extract-shard", action="store_true")
    mode.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.preflight:
        run_preflight(args)
    elif args.extract_shard:
        run_shard(args)
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
