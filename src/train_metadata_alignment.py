"""Formal metadata-alignment training with an optional locked W_{y,s} arm.

Deliberately **not** a copy of S1's loop. S1 drew batches with `pos + BATCH > N` as the
refill condition, which silently drops the tail of every epoch — harmless at 3,072 = 48x64,
but on the real 20,714 it would discard 42 participants per epoch, every epoch, for 500
epochs. Here an epoch is a permutation of all 20,714 cut into **323 batches of 64 plus one
of 42**, matching `DataLoader(..., batch_size=64, shuffle=True)` with `drop_last=False`,
and every participant appears exactly once per epoch. That invariant is asserted, not hoped
for.

Within a seed the three arms share the initialisation, the epoch permutations and the
dropout stream, so the pairing is the only difference. Pairings are drawn once per seed,
held fixed for all 500 epochs, and saved **in full**, not only as a hash.

Checkpoints carry the model, the optimiser, the epoch, every RNG state, the input cache
hashes and the pairing, so a run can be resumed and shown to be the same run.

Only the epoch-500 state is used. Intermediate checkpoints may be written; picking a
better-looking one is not permitted.

    python src/train_metadata_alignment.py --rehearsal      # 1 seed, 3 arms, 1 epoch
    python src/train_metadata_alignment.py --include-within-label-sex --rehearsal
    python src/train_metadata_alignment.py --seeds 0 1 2 3 4 --epochs 500
"""
import os, json, argparse, hashlib
from collections import Counter
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projector import (ContrastiveProjectionHead, contrastive_loss, make_optimizer,
                       build_pairing, build_stratified_pairing, SOURCE_COMMIT)

BATCH, BASE_ARMS = 64, ("correct", "within_label", "global")
MODE = {"correct": "correct", "within_label": "within", "global": "global"}


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def epoch_batches(n, rng):
    """A permutation of all n, cut into ceil(n/BATCH) batches with drop_last=False."""
    order = rng.permutation(n)
    return [order[i:i + BATCH] for i in range(0, n, BATCH)]


def rng_state(seed_rng):
    return {"torch": torch.get_rng_state(), "numpy": seed_rng.get_state()}


def load_all(args):
    z = np.load(args.emb, allow_pickle=True)
    tz = np.load(args.text_emb, allow_pickle=True)
    C = pd.read_csv(args.cohort)
    T = pd.read_csv(args.texts)
    assert np.array_equal(z["participants"], C.participant_identifier.to_numpy())
    assert np.array_equal(tz["participants"], T.participant_identifier.to_numpy())
    A = z["embeddings"]; tid = tz["text_id"]; U = tz["unique_embeddings"]
    tr = np.where((T.splits == "train").to_numpy())[0]
    return C, T, A, tid, U, tr, {"audio": sha(A), "text_unique": sha(U),
                                 "text_id": sha(tid)}


def train_one(A, X_all, pair, tr, seed, epochs, hashes, log_every, out_dir, arm,
              save_ckpt, text_ids=None, resume_from=None, device="cpu"):
    n = len(tr)
    A_tr = torch.tensor(A[tr], dtype=torch.float32, device=device)
    X_tr = torch.tensor(X_all[pair], dtype=torch.float32, device=device)
    torch.manual_seed(seed)                       # seeds CPU and all CUDA generators
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # AST and OPERA-CT are 768-D, while HeAR is 512-D.  The projector source
    # architecture is unchanged apart from the input width required by the frozen
    # backbone.  Inferring it from the audited cache avoids a silent hard-coded
    # adapter or zero-padding step.
    audio_input_dim = int(A.shape[1])
    model = ContrastiveProjectionHead(in_dim=audio_input_dim).to(device)
    opt = make_optimizer(model)
    model.train()
    brng = np.random.RandomState(50_000 + seed)   # shared epoch permutations across arms
    start_ep = 0
    if resume_from is not None:                   # exact resume, RNG states included
        model.load_state_dict({k: v.to(device) for k, v in resume_from["model"].items()})
        opt.load_state_dict(resume_from["optimizer"])
        torch.set_rng_state(resume_from["torch_rng"])
        # Dropout on CUDA draws from the CUDA generator, not the CPU one. Restoring only
        # torch_rng passes on CPU and silently diverges on GPU, which is what the
        # rehearsal caught: continuous and resumed runs gave different final weights.
        if resume_from.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_from["cuda_rng"])
        brng.set_state(resume_from["batch_rng"])
        start_ep = resume_from["epoch"]
    hist, seen_check = [], None
    for ep in range(start_ep, epochs):
        batches = epoch_batches(n, brng)
        assert len(batches) == int(np.ceil(n / BATCH))
        assert sum(len(b) for b in batches) == n
        if ep == start_ep:
            seen_check = np.sort(np.concatenate(batches))
            assert np.array_equal(seen_check, np.arange(n)), \
                "an epoch does not cover every participant exactly once"
            assert len(batches[-1]) == n % BATCH or n % BATCH == 0
        tot, logk = 0.0, []
        for b in batches:
            # K counts identical TEXTS in the batch, not identical participant indices.
            # Indexing `pair` alone gives participant ids, which are unique by
            # construction, so the floor would read a constant 0.
            bt = text_ids[pair[b]]
            inv = dict(zip(*np.unique(bt, return_counts=True)))
            logk.append(float(np.mean(np.log([inv[t] for t in bt]))))
            loss = contrastive_loss(model(A_tr[b]), X_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        hist.append({"epoch": ep, "loss": tot / n, "mean_log_K": float(np.mean(logk)),
                     "n_batches": len(batches), "last_batch": len(batches[-1])})
        if (ep + 1) % log_every == 0 or ep == 0:
            print(f"    [{arm} s{seed}] epoch {ep+1}/{epochs} loss {hist[-1]['loss']:.4f} "
                  f"floor {hist[-1]['mean_log_K']:.4f}", flush=True)
    ck = {"model": {k: v.cpu() for k, v in model.state_dict().items()},
          "optimizer": opt.state_dict(), "epoch": epochs,
          "torch_rng": torch.get_rng_state(),
          "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available()
                       else None),
          "batch_rng": brng.get_state(),
          "pairing": pair, "input_hashes": hashes, "source_commit": SOURCE_COMMIT,
          "seed": seed, "arm": arm, "audio_input_dim": audio_input_dim}
    if save_ckpt:
        torch.save(ck, os.path.join(out_dir, f"{arm}_seed{seed}_epoch{epochs}.pt"))
    return model, hist, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--texts", default="results/metadata_texts.csv")
    ap.add_argument("--emb", default="results/ast_embeddings.npz")
    ap.add_argument("--text_emb", default="results/metadata_text_embeddings.npz")
    ap.add_argument("--out_dir", default="results/alignment")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-within-label-sex", action="store_true",
                    help="add the locked W_{y,s} arm using disease label and recorded sex")
    ap.add_argument("--sex-column", default="sex",
                    help="canonical cohort column used by W_{y,s}; default: sex")
    ap.add_argument("--within-label-sex-support", choices=("fail", "drop-singleton-strata"),
                    default="fail", help="locked common-cohort policy for W_{y,s}")
    args = ap.parse_args()
    if args.rehearsal:
        args.seeds, args.epochs, args.log_every = [0], 1, 1
        args.out_dir = "results/alignment_rehearsal"
    os.makedirs(args.out_dir, exist_ok=True)

    C, T, A, tid, U, tr, hashes = load_all(args)
    arms = (("correct", "within_label", "within_label_sex", "global")
            if args.include_within_label_sex else BASE_ARMS)
    sex = None
    support = {"policy": None, "n_before": int(len(tr)), "n_after": int(len(tr)),
               "dropped": 0, "dropped_strata": {}}
    if args.include_within_label_sex:
        if args.sex_column not in C.columns:
            raise ValueError(f"W_y,s requires cohort column {args.sex_column!r}")
        labels_all = C.y.to_numpy()[tr]
        sex_all = C[args.sex_column].fillna("[MISSING]").astype(str).to_numpy()[tr]
        keys = list(zip(labels_all.astype(str), sex_all.astype(str)))
        counts = dict(sorted(Counter(keys).items()))
        singleton_keys = {key: count for key, count in counts.items() if count < 2}
        support.update({
            "policy": args.within_label_sex_support,
            "stratum_counts_before": {"|".join(key): count for key, count in counts.items()},
            "dropped_strata": {"|".join(key): count for key, count in singleton_keys.items()},
        })
        if singleton_keys:
            if args.within_label_sex_support == "fail":
                raise ValueError(f"singleton (label, sex) strata: {singleton_keys}")
            keep = np.asarray([key not in singleton_keys for key in keys], dtype=bool)
            tr = tr[keep]
        support["n_after"] = int(len(tr))
        support["dropped"] = int(support["n_before"] - support["n_after"])
        sex = C[args.sex_column].fillna("[MISSING]").astype(str).to_numpy()[tr]
        # Build once before any optimisation so a singleton stratum is a data/protocol
        # failure, not a partially completed training run.
    y = C.y.to_numpy()[tr]
    if args.include_within_label_sex:
        build_stratified_pairing(y, sex, seed=args.seeds[0])
    X_all = U[tid[tr]]
    n = len(tr)
    print(f"{n} Standard-train participants -> {n // BATCH} batches of {BATCH} "
          f"plus one of {n % BATCH}")
    print(f"input hashes {hashes}   source {SOURCE_COMMIT[:12]}   device {args.device}")

    # every seed's pairings must be legal bijections, checked before any training
    for s_ in args.seeds:
        for arm in arms:
            j = (build_stratified_pairing(y, sex, seed=s_) if arm == "within_label_sex"
                 else build_pairing(y, MODE[arm], s_))
            assert np.array_equal(np.sort(j), np.arange(n)), f"{arm} s{s_} not a bijection"
            if arm in ("within_label", "within_label_sex"):
                assert np.all(y[j] == y), f"{arm} s{s_} crosses a label"
            if arm == "within_label_sex":
                assert np.all(sex[j] == sex), f"{arm} s{s_} crosses recorded sex"
            if arm != "correct":
                assert not np.any(j == np.arange(n)), f"{arm} s{s_} has a fixed point"
    print(f"pairing check: {len(args.seeds)} seeds x {len(arms)} arms are legal bijections")

    audio_input_dim = int(A.shape[1])
    manifest = {"source_commit": SOURCE_COMMIT, "n_train": int(n), "batch": BATCH,
                "epochs": args.epochs, "seeds": args.seeds, "input_hashes": hashes,
                "audio_input_dim": audio_input_dim, "arms": list(arms),
                "within_label_sex_column": args.sex_column if args.include_within_label_sex else None,
                "within_label_sex_support": support,
                "runs": {}}
    for s_ in args.seeds:
        init_hash = None
        for arm in arms:
            j = (build_stratified_pairing(y, sex, seed=s_) if arm == "within_label_sex"
                 else build_pairing(y, MODE[arm], s_))
            torch.manual_seed(s_)
            ih = sha(np.concatenate([p.detach().numpy().ravel()
                                     for p in ContrastiveProjectionHead(
                                         in_dim=audio_input_dim).parameters()]))
            if init_hash is None:
                init_hash = ih
            assert ih == init_hash, "arms do not share initialisation within a seed"
            bh = sha(np.concatenate(epoch_batches(n, np.random.RandomState(50_000 + s_))))
            print(f"\n  [{arm} seed {s_}] pairing {sha(j)}  init {ih}  batch-order {bh}  "
                  f"collisions "
                  f"{'n/a (identity)' if arm == 'correct' else f'{np.mean(tid[tr] == tid[tr][j])*100:.2f}%'}")
            model, hist, ck = train_one(A, X_all, j, tr, s_, args.epochs, hashes,
                                        args.log_every, args.out_dir, arm,
                                        save_ckpt=not args.rehearsal,
                                        text_ids=tid[tr], device=args.device)
            model.eval()
            with torch.no_grad():
                P = np.concatenate([
                    model(torch.tensor(A[i:i+4096], dtype=torch.float32,
                                       device=args.device)).cpu().numpy()
                    for i in range(0, len(A), 4096)])
            Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
            assert len(P) == len(C), "representation does not cover the full cohort"
            np.savez_compressed(
                os.path.join(args.out_dir, f"repr_{arm}_seed{s_}.npz"),
                participants=C.participant_identifier.to_numpy(),
                raw=P.astype(np.float32), normalized=Pn.astype(np.float32),
                pairing=j, seed=s_, arm=arm, epochs=args.epochs,
                input_hashes=json.dumps(hashes), source_commit=SOURCE_COMMIT)
            manifest["runs"][f"{arm}_seed{s_}"] = {
                "pairing_hash": sha(j), "init_hash": ih, "batch_order_hash": bh,
                "repr_raw_hash": sha(P), "repr_norm_hash": sha(Pn),
                "n_repr": int(len(P)), "loss_first": hist[0]["loss"],
                "loss_last": hist[-1]["loss"], "history": hist if args.epochs <= 5 else
                                                          hist[::max(1, len(hist)//50)]}
            print(f"    saved representation for {len(P)} participants  "
                  f"raw {sha(P)}  norm {sha(Pn)}")

    if args.rehearsal:
        print("\n--- resume equivalence: 2 epochs straight vs 1 + restore + 1 ---")
        j = build_pairing(y, MODE["correct"], 0)
        w = lambda m: sha(np.concatenate([p.detach().cpu().numpy().ravel()
                                          for p in m.parameters()]))
        m2, _, _ = train_one(A, X_all, j, tr, 0, 2, hashes, 99, args.out_dir, "correct",
                             False, text_ids=tid[tr], device=args.device)
        m1, _, ck1 = train_one(A, X_all, j, tr, 0, 1, hashes, 99, args.out_dir, "correct",
                               False, text_ids=tid[tr], device=args.device)
        m1b, _, _ = train_one(A, X_all, j, tr, 0, 2, hashes, 99, args.out_dir, "correct",
                              False, text_ids=tid[tr], resume_from=ck1, device=args.device)
        ok = w(m2) == w(m1b)
        manifest["resume_equivalent"] = ok
        print(f"  continuous {w(m2)}   resumed {w(m1b)}   identical: {ok}")
        assert ok, "resuming from a checkpoint does not reproduce continuous training"

    out = os.path.join(args.out_dir, "manifest.json")
    json.dump(manifest, open(out, "w"), indent=1, default=str)
    print(f"\nwrote {out}")
    if args.rehearsal:
        print("REHEARSAL ONLY — no checkpoints written, no test set read.")


if __name__ == "__main__":
    main()
