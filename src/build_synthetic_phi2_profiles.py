"""Build the frozen 81-profile Phi-2 cache for the E4 language sensitivity.

The strings use the same bracketed field/value schema syntax as the UKCOVID metadata
axis, while keeping the synthetic factors deliberately abstract.  Each unique profile is
encoded exactly once with the already-frozen Phi-2 mask-aware-mean protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


LEVELS = ("LOW", "MID", "HIGH")
BATCH = 16
MAX_LEN = 200
PLACEHOLDER = "[PAD_TEXT]"
REVISION = "810d367871c1d460086d9f82db8696f2e0a0fcd0"


def profiles() -> tuple[np.ndarray, list[str]]:
    codes = np.asarray(np.meshgrid(*([np.arange(3)] * 4))).reshape(4, -1).T.astype(np.int8)
    texts = [" ".join(f"[CONTEXT_{j + 1}={LEVELS[int(v)]}]"
                      for j, v in enumerate(code)) for code in codes]
    assert codes.shape == (81, 4) and len(set(texts)) == 81
    return codes, texts


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/hd/data_heliu/hf_models/phi-2")
    ap.add_argument("--out", default="results/synthetic_phi2_profiles.npz")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    codes, texts = profiles()
    if args.self_test:
        assert texts[0] == "[CONTEXT_1=LOW] [CONTEXT_2=LOW] [CONTEXT_3=LOW] [CONTEXT_4=LOW]"
        print("SELF-TEST PASS: 81 deterministic bracketed synthetic profiles")
        return

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
    assert max(lengths) <= MAX_LEN
    model = AutoModel.from_pretrained(args.model, dtype=torch.bfloat16).eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(items: list[str]) -> np.ndarray:
        n = len(items)
        padded = list(items) + [PLACEHOLDER] * ((-n) % BATCH)
        rows = []
        for start in range(0, len(padded), BATCH):
            batch = tokenizer(
                padded[start:start + BATCH], return_tensors="pt",
                padding="max_length", truncation=True, max_length=MAX_LEN).to("cuda")
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            rows.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(rows)[:n].astype(np.float32)

    embeddings = encode(texts)
    permutation = np.random.RandomState(0).permutation(len(texts))
    shuffled = encode([texts[i] for i in permutation])
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    assert np.array_equal(embeddings, restored), "Phi-2 profile cache is not bit-stable"
    cache_hash = hashlib.sha256(embeddings.tobytes()).hexdigest()
    provenance = {
        "schema": "four abstract context fields in UKCOVID bracketed field=value syntax",
        "pooling": "last_hidden_state mask-aware mean",
        "padding": "right, fixed max_length",
        "max_length": MAX_LEN,
        "batch": BATCH,
        "dtype": "torch.bfloat16",
        "revision": REVISION,
        "model_path": str(Path(args.model)),
        "n_profiles": 81,
        "embedding_sha256": cache_hash,
        "script_sha256": sha256_file(Path(__file__)),
        "bit_stable_under_shuffle": True,
    }
    atomic_npz(
        out, profile_codes=codes, embeddings=embeddings,
        texts=np.asarray(texts), provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(f"wrote frozen 81-profile Phi-2 cache {out} sha256={cache_hash[:16]}")


if __name__ == "__main__":
    main()
