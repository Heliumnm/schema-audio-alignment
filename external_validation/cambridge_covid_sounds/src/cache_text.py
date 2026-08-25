"""Cache each unique Cambridge metadata schema once in the frozen Phi-2 space."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import (atomic_json, load_config, output_paths, resolve_path, sha16_array,
                    sha256_file)


PLACEHOLDER = "[AGE=[MISSING]] [SEX=[MISSING]]"


def execute(config_file: str) -> Path:
    import torch
    from transformers import AutoModel, AutoTokenizer

    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    gate = json.loads((paths["public"] / "data_gate.json").read_text())
    if gate["verdict"] != "GO":
        raise RuntimeError("data gate is not GO; text embedding is forbidden")
    manifest_path = paths["private"] / "participant_manifest.csv"
    table = pd.read_csv(manifest_path)
    unique = sorted(table.text.unique())
    text_id = {text: index for index, text in enumerate(unique)}
    settings = config["models"]["text"]
    revision = str(settings.get("revision", "UNSPECIFIED"))
    maximum = int(settings.get("max_length", 256))
    output = paths["models"] / "text_embeddings.npz"
    if output.exists():
        existing = np.load(output, allow_pickle=True)
        if (np.array_equal(existing["participants"].astype(str),
                           table.participant_identifier.astype(str).to_numpy()) and
                str(existing["cohort_sha256"]) == sha256_file(manifest_path) and
                str(existing["model_revision"]) == revision and
                int(existing["max_length"]) == maximum and
                str(existing["pooling"]) == "mask_aware_mean"):
            print(f"verified existing text cache {output}")
            return output
        raise RuntimeError(f"refusing to overwrite incompatible {output}")

    model_path = resolve_path(config_path, settings["model_path"])
    if model_path is None:
        raise ValueError("models.text.model_path is required")
    device = config["models"].get("device", "cuda")
    batch_size = int(settings.get("batch_size", 16))
    dtype_name = settings.get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = [len(tokenizer(text)["input_ids"]) for text in unique]
    if max(lengths) > maximum:
        raise RuntimeError(
            f"max schema length {max(lengths)} exceeds frozen max_length {maximum}; "
            "do not truncate or change this after model execution")
    model = AutoModel.from_pretrained(model_path, dtype=dtype).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(texts: list[str]) -> np.ndarray:
        pad_count = (-len(texts)) % batch_size
        values = list(texts) + [PLACEHOLDER] * pad_count
        outputs = []
        for start in range(0, len(values), batch_size):
            batch = tokenizer(
                values[start:start + batch_size], return_tensors="pt",
                padding="max_length", truncation=True, max_length=maximum).to(device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            outputs.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(outputs)[:len(texts)].astype(np.float32)

    embeddings = encode(unique)
    permutation = np.random.RandomState(0).permutation(len(unique))
    shuffled = encode([unique[index] for index in permutation])
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    if not np.array_equal(embeddings, restored):
        raise RuntimeError("Phi-2 cache is not bit-identical under shuffled encode order")
    participant_text_id = np.asarray([text_id[text] for text in table.text], dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".tmp.npz")
    np.savez_compressed(
        temporary, participants=table.participant_identifier.astype(str).to_numpy(),
        text_id=participant_text_id, unique_texts=np.asarray(unique),
        unique_embeddings=embeddings, pooling="mask_aware_mean",
        max_length=maximum, model_revision=revision,
        cohort_sha256=sha256_file(manifest_path))
    os.replace(temporary, output)
    atomic_json(paths["public"] / "text_axis_audit.json", {
        "n_participants": len(table), "n_unique_profiles": len(unique),
        "token_length": {"min": min(lengths), "max": max(lengths),
                         "frozen_max_length": maximum},
        "shape": list(embeddings.shape), "pooling": "mask_aware_mean",
        "padding": "right-fixed-length", "dtype": dtype_name,
        "model_revision": revision,
        "cache_bit_identical_under_shuffle": True,
        "embedding_values_sha16": sha16_array(embeddings),
        "cohort_sha256": sha256_file(manifest_path),
    })
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
