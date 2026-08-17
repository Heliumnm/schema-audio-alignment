"""S1 — short run on real data. Technical checks only.

A fixed 4,096 participants from Standard train: 3,072 to train on, 1,024 held back to
monitor. Three arms, seed 0, batch 64, 500 updates. **Standard val, matched and
matched_long are not read.**

Recorded: loss against the duplicate-induced floor, gradient norms, effective rank of the
projected outputs, and profile-aware retrieval — MRR and Recall@1/10 with candidates
collapsed to unique profiles, because with 5,437 texts over 20,714 people an
exact-participant metric is measuring ambiguity, not retrieval.

**The ordering of `correct` / `within_label` / `global` is an observation, not a gate**, and
after the fact it turned out not even to be a usable observation. **These retrieval numbers
do not enter the paper.** Four reasons, all found after the run and all recorded rather
than repaired, because S1's job — technical plumbing — was done either way:

1. **The real baseline beats every arm.** Ranking the candidate profiles by their training
   frequency, with no audio at all and the same ranking for every anchor, gives
   MRR 0.1702, R@1 0.0732, R@10 0.4385 — above `correct`'s 0.0787 / 0.0400 / 0.1367.
   `1/469 = 0.0021` was never the relevant baseline, and `correct > within > global` is
   therefore no evidence of individual-level correspondence.
2. **The monitor set is not a random holdout.** The subsample is sorted before it is cut,
   so `mon` is the last 1,024 participants in identifier order, not a random 1,024.
3. **The effective-rank denominator is wrong when written as /2560.** With 1,024 monitor
   rows the ceiling after mean-centring is min(n−1, dim) = 1,023, so 381 is 381/1023.
4. **A first and a last gradient norm are two points**, and cannot establish that one arm
   is more learnable than another.

What S1 does gate is technical, and that part stands: the loss moves, the outputs do not
collapse, the hashes are stable, and the pairings are what they claim to be.

    python src/smoke_s1.py
"""
import os, json, hashlib
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projector import (ContrastiveProjectionHead, contrastive_loss, make_optimizer,
                       build_pairing, SOURCE_COMMIT)

N_SUB, N_TRAIN, BATCH, UPDATES, SEED = 4096, 3072, 64, 500, 0


def effective_rank(P):
    """exp of the entropy of the normalised singular spectrum. A collapsed representation
    has an effective rank near 1 whatever its norm."""
    s = np.linalg.svd(P - P.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def profile_retrieval(Pn, Xn_unique, gold_uid):
    """Candidates are unique profiles; the gold answer is the anchor's own profile."""
    S = Pn @ Xn_unique.T
    order = np.argsort(-S, axis=1)
    rank = np.array([np.where(order[i] == gold_uid[i])[0][0] for i in range(len(Pn))])
    return {"mrr": float(np.mean(1.0 / (rank + 1))),
            "recall_at_1": float(np.mean(rank == 0)),
            "recall_at_10": float(np.mean(rank < 10)),
            "n_candidates": int(Xn_unique.shape[0])}


def main():
    z = np.load("results/ast_embeddings.npz", allow_pickle=True)
    tz = np.load("results/metadata_text_embeddings.npz", allow_pickle=True)
    C = pd.read_csv("results/ukcovid_audio_cohort.csv")
    T = pd.read_csv("results/metadata_texts.csv")
    assert np.array_equal(z["participants"], C.participant_identifier.to_numpy())
    assert np.array_equal(tz["participants"], T.participant_identifier.to_numpy())

    tr_all = np.where((T.splits == "train").to_numpy())[0]
    rs = np.random.RandomState(SEED)
    sub = np.sort(rs.choice(tr_all, N_SUB, replace=False))
    fit, mon = sub[:N_TRAIN], sub[N_TRAIN:]
    A_all = z["embeddings"]; tid = tz["text_id"]; U = tz["unique_embeddings"]
    y = C.y.to_numpy()

    A_fit = torch.tensor(A_all[fit], dtype=torch.float32)
    X_fit_all = torch.tensor(U[tid[fit]], dtype=torch.float32)
    A_mon = torch.tensor(A_all[mon], dtype=torch.float32)
    mon_uids = np.unique(tid[mon])
    uid_pos = {u: i for i, u in enumerate(mon_uids)}
    gold = np.array([uid_pos[u] for u in tid[mon]])
    Xu = U[mon_uids]
    Xun = Xu / np.linalg.norm(Xu, axis=1, keepdims=True)
    print(f"S1: {N_SUB} participants ({N_TRAIN} fit / {len(mon)} monitor), "
          f"{len(mon_uids)} distinct profiles among the monitor set")
    print(f"    batch {BATCH}, {UPDATES} updates, seed {SEED}, source {SOURCE_COMMIT[:12]}")
    print("    Standard val, matched and matched_long are NOT read")

    res = {"source_commit": SOURCE_COMMIT, "n_sub": N_SUB, "n_fit": N_TRAIN,
           "n_monitor": int(len(mon)), "batch": BATCH, "updates": UPDATES, "seed": SEED,
           "arms": {}}
    for mode in ("correct", "within_label", "global"):
        j = build_pairing(y[fit], {"within_label": "within"}.get(mode, mode), SEED)
        # Only meaningful for the shuffled arms. In `correct` the pairing is the identity,
        # so this ratio is 1.0 by definition and would masquerade as a comparable quantity.
        collide = (None if mode == "correct"
                   else float(np.mean(tid[fit] == tid[fit][j])))
        Xp = X_fit_all[torch.tensor(j)]
        pair_hash = hashlib.sha256(j.tobytes()).hexdigest()[:16]

        torch.manual_seed(SEED)
        m = ContrastiveProjectionHead(); opt = make_optimizer(m)
        m.train()
        rs2 = np.random.RandomState(SEED)
        losses, gnorms, logK = [], [], []
        step, order, pos = 0, rs2.permutation(N_TRAIN), 0
        while step < UPDATES:
            if pos + BATCH > N_TRAIN:
                order, pos = rs2.permutation(N_TRAIN), 0
            b = order[pos:pos + BATCH]; pos += BATCH
            bt = tid[fit][j][b]
            _, cnt = np.unique(bt, return_counts=True)
            inv = dict(zip(*np.unique(bt, return_counts=True)))
            logK.append(float(np.mean(np.log([inv[t] for t in bt]))))
            loss = contrastive_loss(m(A_fit[b]), Xp[b])
            opt.zero_grad(); loss.backward()
            gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))
            opt.step()
            losses.append(float(loss.detach())); gnorms.append(gn); step += 1

        m.eval()
        with torch.no_grad():
            P = m(A_mon).numpy()
        Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        ret = profile_retrieval(Pn, Xun, gold)
        e = {"pairing_hash": pair_hash, "identical_profile_collision": collide,
             "loss_first": losses[0], "loss_last50_mean": float(np.mean(losses[-50:])),
             "loss_floor_logK": float(np.mean(logK)),
             "loss_decreased": bool(np.mean(losses[-50:]) < losses[0]),
             "grad_norm_first": gnorms[0], "grad_norm_last": gnorms[-1],
             "effective_rank": effective_rank(P), "output_dim": int(P.shape[1]),
             "output_hash": hashlib.sha256(P.tobytes()).hexdigest()[:16],
             "finite": bool(np.isfinite(P).all()), **ret}
        res["arms"][mode] = e
        cstr = ("n/a (identity pairing)" if collide is None else f"{collide*100:.2f}%")
        print(f"\n  [{mode}]  pairing {pair_hash}  identical-profile collisions {cstr}")
        print(f"    loss {losses[0]:.4f} -> {e['loss_last50_mean']:.4f}   "
              f"floor(mean log K) {e['loss_floor_logK']:.4f}   "
              f"grad norm {gnorms[0]:.3f} -> {gnorms[-1]:.3f}")
        print(f"    effective rank {e['effective_rank']:.2f} of {P.shape[1]}   "
              f"MRR {ret['mrr']:.4f}  R@1 {ret['recall_at_1']:.4f}  "
              f"R@10 {ret['recall_at_10']:.4f}  over {ret['n_candidates']} profiles")

    fails = [f"{k}:{c}" for k, v in res["arms"].items()
             for c in ("loss_decreased", "finite") if not v[c]]
    fails += [f"{k}:collapse" for k, v in res["arms"].items() if v["effective_rank"] < 2]
    res["technical_pass"] = not fails
    res["failures"] = fails
    json.dump(res, open("results/smoke_s1.json", "w"), indent=1)
    print(f"\nS1 technical checks {'PASSED' if not fails else 'FAILED: ' + str(fails)}")
    o = {k: res["arms"][k]["recall_at_1"] for k in res["arms"]}
    print(f"arm ordering by R@1: {o}  — recorded as an observation, not a gate. "
          f"Three arms coming out alike is a possible true null.")


if __name__ == "__main__":
    main()
