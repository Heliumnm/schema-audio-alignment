"""Train the preregistered W_{y,s} pairing control for UKCOVID.

The new arm pairs each audio recording with metadata from a *different* participant
having the same disease label and recorded sex.  It reuses the formal alignment
training loop so that optimiser, minibatches, dropout, final-checkpoint selection and
representation export remain identical to the existing Correct/Within/Global runs.

Pair construction has an RNG isolated from model and minibatch RNGs.  Within each
(label, sex) stratum, participants are shuffled once and mapped around a one-position
cycle.  The pairing is held fixed for all 500 epochs and saved in full.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from projector import ContrastiveProjectionHead, SOURCE_COMMIT
from train_metadata_alignment import (BATCH, epoch_batches, load_all, sha,
                                      train_one)


ARM = "within_label_sex"
PAIRING_SEED_BASE = 70_000
EXPECTED_STRATA = {
    (0, "FEMALE"): 7_378,
    (0, "MALE"): 5_866,
    (1, "FEMALE"): 4_690,
    (1, "MALE"): 2_775,
    (1, "MISSING"): 5,
}


def parse_recorded_sex(texts: np.ndarray) -> np.ndarray:
    out = []
    for text in texts:
        match = re.search(r"(?:^|\s)\[SEX=(\[MISSING\]|[^\]]+)\]", str(text))
        if match is None:
            raise ValueError(f"metadata schema has no [SEX=...] field: {text!r}")
        value = match.group(1)
        out.append("MISSING" if value == "[MISSING]" else value)
    return np.asarray(out, dtype=str)


def stratum_counts(labels: np.ndarray, sex: np.ndarray) -> dict[tuple[int, str], int]:
    assert len(labels) == len(sex)
    return {
        (int(label), str(level)): int(np.sum((labels == label) & (sex == level)))
        for label in np.unique(labels)
        for level in np.unique(sex[labels == label])
    }


def build_within_label_sex_pairing(labels: np.ndarray, sex: np.ndarray,
                                   seed: int) -> np.ndarray:
    """Return a deterministic, bijective, no-self pairing within (label, sex)."""
    labels = np.asarray(labels)
    sex = np.asarray(sex)
    counts = stratum_counts(labels, sex)
    if any(size < 2 for size in counts.values()):
        raise ValueError(f"singleton (label, sex) stratum: {counts}")

    rng = np.random.RandomState(PAIRING_SEED_BASE + int(seed))
    pairing = np.full(len(labels), -1, dtype=np.int64)
    for label, level in sorted(counts):
        group = np.flatnonzero((labels == label) & (sex == level))
        cycle = rng.permutation(group)
        pairing[cycle] = np.roll(cycle, -1)

    assert np.all(pairing >= 0)
    assert np.array_equal(np.sort(pairing), np.arange(len(labels)))
    assert not np.any(pairing == np.arange(len(labels)))
    assert np.all(labels[pairing] == labels)
    assert np.all(sex[pairing] == sex)
    return pairing


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def check_reference(reference_dir: Path, hashes: dict, seed: int,
                    init_hash: str, batch_hash: str) -> dict:
    manifest_path = reference_dir / "manifest.json"
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    assert manifest["n_train"] == 20_714
    assert manifest["batch"] == BATCH
    assert manifest["epochs"] == 500
    assert manifest["input_hashes"] == hashes
    key = f"within_label_seed{seed}"
    reference = manifest["runs"][key]
    assert reference["init_hash"] == init_hash, (reference["init_hash"], init_hash)
    assert reference["batch_order_hash"] == batch_hash, (
        reference["batch_order_hash"], batch_hash)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reference_run": key,
        "reference_init_hash": reference["init_hash"],
        "reference_batch_order_hash": reference["batch_order_hash"],
    }


def self_test() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1, 1])
    sex = np.asarray(["F", "F", "F", "M", "M", "M", "M"])
    p1 = build_within_label_sex_pairing(labels, sex, 3)
    p2 = build_within_label_sex_pairing(labels, sex, 3)
    assert np.array_equal(p1, p2)
    assert not np.any(p1 == np.arange(len(labels)))
    assert np.all(labels[p1] == labels) and np.all(sex[p1] == sex)
    try:
        build_within_label_sex_pairing(np.asarray([0, 0, 1]),
                                       np.asarray(["F", "F", "M"]), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("singleton stratum was not rejected")
    parsed = parse_recorded_sex(np.asarray([
        "[AGE=18TO24] [SEX=FEMALE] [SMOKER=NO]",
        "[AGE=[MISSING]] [SEX=[MISSING]] [SMOKER=[MISSING]]",
    ]))
    assert parsed.tolist() == ["FEMALE", "MISSING"]
    print("SELF-TEST PASS: sex parsing, stratified cycle, bijection and no-self checks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--texts", default="results/metadata_texts.csv")
    ap.add_argument("--emb", default="results/ast_embeddings.npz")
    ap.add_argument("--text-emb", default="results/metadata_text_embeddings.npz")
    ap.add_argument("--reference-alignment-dir", default="results/alignment")
    ap.add_argument("--out-dir", default="results/pairing_followup/e2_alignment_ast")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="validate real-data strata, pairings and reference RNG hashes only")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.rehearsal:
        args.seeds, args.epochs, args.log_every = [0], 1, 1
        args.out_dir = str(Path(args.out_dir).with_name(Path(args.out_dir).name + "_rehearsal"))

    out_dir = Path(args.out_dir)
    C, T, A, tid, U, tr, hashes = load_all(argparse.Namespace(
        emb=args.emb, text_emb=args.text_emb, cohort=args.cohort, texts=args.texts))
    labels = C.y.to_numpy()[tr]
    sex = parse_recorded_sex(T.text.to_numpy()[tr])
    counts = stratum_counts(labels, sex)
    if not args.rehearsal:
        assert counts == EXPECTED_STRATA, (counts, EXPECTED_STRATA)
        assert args.epochs == 500, "formal E2 uses only the preregistered epoch-500 state"
    assert len(tr) == 20_714
    text_train = U[tid[tr]]
    audio_input_dim = int(A.shape[1])

    if args.preflight:
        for seed in args.seeds:
            pairing = build_within_label_sex_pairing(labels, sex, seed)
            torch.manual_seed(seed)
            initial = ContrastiveProjectionHead(in_dim=audio_input_dim)
            init_hash = sha(np.concatenate([
                p.detach().cpu().numpy().ravel() for p in initial.parameters()
            ]))
            batch_hash = sha(np.concatenate(epoch_batches(
                len(tr), np.random.RandomState(50_000 + seed))))
            reference = check_reference(Path(args.reference_alignment_dir), hashes,
                                        seed, init_hash, batch_hash)
            print(f"PREFLIGHT seed {seed}: pairing={sha(pairing)} init={init_hash} "
                  f"batch={batch_hash} reference={reference['reference_run']}")
        print(f"PREFLIGHT PASS: strata={counts}; no training or representation export")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "standing": "post-hoc preregistered W_{y,s} pairing diagnostic",
        "source_commit": SOURCE_COMMIT,
        "arm": ARM,
        "n_train": int(len(tr)),
        "batch": BATCH,
        "epochs": int(args.epochs),
        "seeds": list(args.seeds),
        "input_hashes": hashes,
        "audio_input_dim": audio_input_dim,
        "pairing_seed_base": PAIRING_SEED_BASE,
        "stratum_counts": {f"y={y}|sex={s}": n for (y, s), n in counts.items()},
        "runs": {},
        "matched_or_test_labels_read": False,
    }

    for seed in args.seeds:
        pairing = build_within_label_sex_pairing(labels, sex, seed)
        torch.manual_seed(seed)
        initial = ContrastiveProjectionHead(in_dim=audio_input_dim)
        init_hash = sha(np.concatenate([
            p.detach().cpu().numpy().ravel() for p in initial.parameters()
        ]))
        batch_hash = sha(np.concatenate(epoch_batches(
            len(tr), np.random.RandomState(50_000 + seed))))
        reference = None
        if not args.rehearsal:
            reference = check_reference(Path(args.reference_alignment_dir), hashes,
                                        seed, init_hash, batch_hash)
        print(f"[{ARM} seed {seed}] pairing {sha(pairing)} init {init_hash} "
              f"batch-order {batch_hash} same-schema collisions "
              f"{np.mean(tid[tr] == tid[tr][pairing])*100:.2f}%", flush=True)

        model, history, _ = train_one(
            A, text_train, pairing, tr, seed, args.epochs, hashes,
            args.log_every, str(out_dir), ARM, save_ckpt=not args.rehearsal,
            text_ids=tid[tr], device=args.device)
        model.eval()
        with torch.no_grad():
            projected = np.concatenate([
                model(torch.tensor(A[i:i + 4096], dtype=torch.float32,
                                   device=args.device)).cpu().numpy()
                for i in range(0, len(A), 4096)
            ])
        normalized = projected / np.maximum(
            np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
        assert len(projected) == len(C)
        repr_path = out_dir / f"repr_{ARM}_seed{seed}.npz"
        np.savez_compressed(
            repr_path,
            participants=C.participant_identifier.to_numpy(),
            raw=projected.astype(np.float32),
            normalized=normalized.astype(np.float32),
            pairing=pairing,
            train_participants=C.participant_identifier.to_numpy()[tr],
            train_recorded_sex=sex,
            seed=seed,
            arm=ARM,
            epochs=args.epochs,
            input_hashes=json.dumps(hashes),
            source_commit=SOURCE_COMMIT,
        )
        manifest["runs"][f"{ARM}_seed{seed}"] = {
            "pairing_hash": sha(pairing),
            "init_hash": init_hash,
            "batch_order_hash": batch_hash,
            "reference": reference,
            "repr_raw_hash": sha(projected),
            "repr_norm_hash": sha(normalized),
            "repr_file_sha256": sha256_file(repr_path),
            "loss_first": history[0]["loss"],
            "loss_last": history[-1]["loss"],
            "history": history if args.epochs <= 5 else history[::max(1, len(history) // 50)],
        }
        print(f"saved {repr_path} raw {sha(projected)} norm {sha(normalized)}", flush=True)

    manifest_path = out_dir / "manifest.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    with open(tmp, "w") as stream:
        json.dump(manifest, stream, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, manifest_path)
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
