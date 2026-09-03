"""Participant-level frozen AST-6L / OPERA-CT extraction for the external audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (REPO_ROOT, atomic_json, load_config, output_paths, resolve_path,
                    sha16_array, sha256_file)


SR = 16_000
AST_WINDOW_S = 10.24
AST_WINDOW_SAMPLES = int(SR * AST_WINDOW_S)
AST_PATCH, AST_STRIDE, AST_FREQ, AST_TIME = 16, 10, 12, 101
HEAR_WINDOW_S = 2.0
HEAR_WINDOW_SAMPLES = int(SR * HEAR_WINDOW_S)
HEAR_OUTPUT_DIM = 512


def window_starts(n_samples: int, window: int) -> list[int]:
    if n_samples <= window:
        return [0]
    count = int(math.ceil(n_samples / window))
    last = n_samples - window
    starts = [int(round(i * last / (count - 1))) for i in range(count)]
    starts[0], starts[-1] = 0, last
    return starts


def load_audio(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if rate != SR:
        waveform = librosa.resample(waveform, orig_sr=rate, target_sr=SR)
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform {path}")
    return waveform


def fixed_windows(waveform: np.ndarray, window_samples: int) -> np.ndarray:
    """Return deterministic, full-coverage windows with right-zero padding.

    Long recordings use ``ceil(duration/window)`` evenly spaced starts including
    both the first and last possible start.  This is deliberately content blind:
    no cough detector, energy selection or peak normalisation is applied.
    """
    windows = []
    for start in window_starts(len(waveform), window_samples):
        segment = waveform[start:start + window_samples]
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))
        windows.append(np.asarray(segment, dtype=np.float32))
    return np.stack(windows)


def ast_recording(path: Path, model, feature_extractor, torch, device: str) -> np.ndarray:
    waveform = load_audio(path)
    outputs = []
    for start in window_starts(len(waveform), AST_WINDOW_SAMPLES):
        segment = waveform[start:start + AST_WINDOW_SAMPLES]
        real_samples = len(segment)
        if real_samples < AST_WINDOW_SAMPLES:
            segment = np.pad(segment, (0, AST_WINDOW_SAMPLES - real_samples))
        valid_frames = min(1024, int(round(real_samples / SR * 100)))
        valid_time = max(0, (valid_frames - AST_PATCH) // AST_STRIDE + 1)
        if valid_time < 1:
            raise ValueError(f"recording too short for one complete AST patch: {path}")
        inputs = feature_extractor([segment], sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state[:, 2:, :]
        grid = hidden.reshape(1, AST_FREQ, AST_TIME, -1)[0, :, :valid_time]
        outputs.append(grid.reshape(-1, grid.shape[-1]).mean(0).float().cpu().numpy())
    return np.mean(outputs, axis=0).astype(np.float32)


def configure_ast(config: dict, config_path: Path, device: str):
    import torch
    from transformers import ASTModel, AutoFeatureExtractor

    path = resolve_path(config_path, config["models"]["ast"]["model_path"])
    if path is None:
        raise ValueError("models.ast.model_path is required")
    extractor = AutoFeatureExtractor.from_pretrained(path)
    model = ASTModel.from_pretrained(path)
    layers = int(config["models"]["ast"].get("layers", 6))
    model.encoder.layer = torch.nn.ModuleList(list(model.encoder.layer)[:layers])
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint = hashlib.sha256()
    for candidate in sorted(path.glob("*")) if path.is_dir() else [path]:
        if candidate.suffix in {".bin", ".safetensors"}:
            checkpoint.update(sha256_file(candidate).encode())
    return model, extractor, torch, checkpoint.hexdigest(), layers


def configure_opera(config: dict, config_path: Path, device: str):
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from extract_opera_ukcovid import configure_opera as upstream_configure

    root = resolve_path(config_path, config["models"]["opera_ct"]["opera_root"])
    checkpoint = resolve_path(config_path, config["models"]["opera_ct"]["checkpoint_path"])
    if root is None or checkpoint is None:
        raise ValueError("OPERA root and checkpoint paths are required")
    return (*upstream_configure(root, checkpoint, device),)


def opera_recording(path: Path, model, preprocess, torch, device: str) -> np.ndarray:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from extract_opera_ukcovid import EMBED_DIM, load_native_windows

    mels, _ = load_native_windows(path, preprocess)
    tensor = torch.tensor(mels, dtype=torch.float32, device=device)
    with torch.no_grad():
        features = model.extract_feature(tensor, EMBED_DIM).float().cpu().numpy()
    return features.mean(axis=0).astype(np.float32)


def _model_file_hash(path: Path) -> str:
    """Hash the SavedModel graph and variables, independent of cache metadata."""
    candidates = [path / "fingerprint.pb", path / "saved_model.pb",
                  path / "variables" / "variables.index",
                  path / "variables" / "variables.data-00000-of-00001"]
    missing = [str(candidate) for candidate in candidates if not candidate.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete HeAR SavedModel: {missing}")
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(str(candidate.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(candidate).encode("ascii"))
    return digest.hexdigest()


def configure_hear(config: dict, config_path: Path):
    import tensorflow as tf

    settings = config["models"]["hear"]
    path = resolve_path(config_path, settings["model_path"])
    if path is None:
        raise ValueError("models.hear.model_path is required")
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
    if "serving_default" not in model.signatures:
        raise RuntimeError("HeAR SavedModel lacks serving_default signature")
    signature = model.signatures["serving_default"]
    return tf, model, signature, _model_file_hash(path)


def hear_recording(path: Path, signature, tf, batch_size: int = 64,
                   input_key: str = "x", output_key: str = "output_0") -> np.ndarray:
    windows = fixed_windows(load_audio(path), HEAR_WINDOW_SAMPLES)
    outputs = []
    for start in range(0, len(windows), batch_size):
        batch = tf.convert_to_tensor(windows[start:start + batch_size], dtype=tf.float32)
        result = signature(**{input_key: batch})
        if output_key not in result:
            raise RuntimeError(
                f"HeAR output {output_key!r} absent; available={sorted(result)}")
        outputs.append(np.asarray(result[output_key].numpy(), dtype=np.float32))
    matrix = np.concatenate(outputs, axis=0)
    if matrix.shape != (len(windows), HEAR_OUTPUT_DIM):
        raise RuntimeError(f"unexpected HeAR output shape {matrix.shape}")
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise RuntimeError(f"invalid HeAR embedding for {path}")
    return matrix.mean(axis=0).astype(np.float32)


def check_indices(table: pd.DataFrame, n: int = 28) -> list[int]:
    """Deterministic split/label coverage plus CODA recording-count/duration extremes."""
    chosen, seen = [], set()

    def add(indices) -> None:
        for index in indices:
            index = int(index)
            if index not in seen:
                chosen.append(index); seen.add(index)

    splits = ["train", "validation", "test"]
    if "matched_target" in set(table.splits.astype(str)):
        splits.append("matched_target")
    per_cell = 2 if n >= 2 * len(splits) * 2 else 1
    for split in splits:
        for label in (0, 1):
            indices = table.index[(table.splits == split) & (table.y == label)].tolist()
            add(indices[:per_cell])

    for field in ("audio_count", "audio_total_duration_s", "audio_max_duration_s"):
        if field in table:
            add(table.sort_values([field, "participant_identifier"],
                                  ascending=[True, True]).index[:2])
            add(table.sort_values([field, "participant_identifier"],
                                  ascending=[False, True]).index[:2])
    add(table.sort_values("participant_identifier").index)
    return chosen[:n]


def execute(config_file: str, backbone: str) -> Path:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    gate = json.loads((paths["public"] / "data_gate.json").read_text())
    if gate["verdict"] != "GO":
        raise RuntimeError("data gate is not GO; representation extraction is forbidden")
    manifest_path = paths["private"] / "participant_manifest.csv"
    table = pd.read_csv(manifest_path)
    files = [list(map(Path, json.loads(value))) for value in table.audio_files_json]
    device = config["models"].get("device", "cuda")

    if backbone == "ast":
        model, preprocessing, torch, checkpoint_hash, layers = configure_ast(
            config, config_path, device)
        encode = lambda path: ast_recording(path, model, preprocessing, torch, device)
        spec = f"AST-first-{layers}-layers|participant-equal-recording-mean"
    elif backbone == "opera_ct":
        torch, model, preprocessing, checkpoint_hash = configure_opera(
            config, config_path, device)
        encode = lambda path: opera_recording(path, model, preprocessing, torch, device)
        spec = "OPERA-CT|participant-equal-recording-mean"
    elif backbone == "hear":
        tf, model, signature, checkpoint_hash = configure_hear(config, config_path)
        settings = config["models"]["hear"]
        encode = lambda path: hear_recording(
            path, signature, tf, int(settings.get("batch_size", 64)),
            str(settings.get("input_key", "x")),
            str(settings.get("output_key", "output_0")))
        revision = str(settings.get("revision", "unknown"))
        spec = (f"HeAR-1.0.0@{revision}|16k-mono|2s|ceil-full-coverage|"
                "right-zero-pad|window-mean|participant-equal-recording-mean")
    else:
        raise ValueError(backbone)

    failures = []
    for index in check_indices(table):
        first = np.mean([encode(path) for path in files[index]], axis=0)
        second = np.mean([encode(path) for path in files[index]], axis=0)
        if not np.array_equal(first, second) or not np.isfinite(first).all() or not np.any(first):
            failures.append({"split": table.loc[index, "splits"],
                             "label": int(table.loc[index, "y"])})
    if failures:
        raise RuntimeError(f"{backbone} deterministic preflight failed: {failures}")

    output_dir = paths["models"] / backbone
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "raw_embeddings.npz"
    if output.exists():
        existing = np.load(output, allow_pickle=True)
        if (np.array_equal(existing["participants"].astype(str),
                           table.participant_identifier.astype(str).to_numpy()) and
                str(existing["cohort_sha256"]) == sha256_file(manifest_path) and
                str(existing["checkpoint_sha256"]) == checkpoint_hash and
                str(existing["spec"]) == spec):
            print(f"{backbone}: verified existing extraction {output}")
            return output
        raise RuntimeError(f"refusing to overwrite incompatible {output}")

    embeddings, counts = [], []
    for index, participant_files in enumerate(files):
        representation = np.mean([encode(path) for path in participant_files], axis=0)
        embeddings.append(representation)
        counts.append(len(participant_files))
        if (index + 1) % 250 == 0:
            print(f"{backbone}: {index+1}/{len(table)} participants", flush=True)
    matrix = np.stack(embeddings).astype(np.float32)
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise RuntimeError("non-finite or zero external representation")
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(
        temporary, participants=table.participant_identifier.astype(str).to_numpy(),
        embeddings=matrix, n_recordings=np.asarray(counts), spec=spec,
        checkpoint_sha256=checkpoint_hash, cohort_sha256=sha256_file(manifest_path))
    os.replace(temporary, output)
    atomic_json(paths["public"] / f"{backbone}_extraction_audit.json", {
        "backbone": backbone, "n_participants": len(table), "shape": list(matrix.shape),
        "finite": True, "nonzero": len(table), "spec": spec,
        "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": sha256_file(manifest_path),
        "embedding_values_sha16": sha16_array(matrix),
        "recordings_per_participant": {"min": int(min(counts)),
                                       "median": float(np.median(counts)),
                                       "max": int(max(counts))},
        "preflight_n": len(check_indices(table)), "preflight_passed": True,
    })
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ast", "opera_ct", "hear"), required=True)
    args = parser.parse_args()
    execute(args.config, args.backbone)


if __name__ == "__main__":
    main()
