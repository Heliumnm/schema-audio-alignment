"""Day 1: do false negatives explain the global-alignment negative result?

Every alignment run so far used single-positive InfoNCE — one text per recording,
everything else a negative. With schema text that is wrong: 3,552 distinct strings
over 4,142 training cycles means a sampled "negative" is often the anchor's own text,
or one differing in a single tier. Penalising the model for matching text it should
match is the failure mode the debiased-contrastive literature describes, and it is a
live explanation for "more negatives made it worse" (§2.92).

Going straight from single to soft-multi would change two things at once — whether
multiple positives are allowed, and whether near-neighbours become soft positives.
Three conditions separate them:

  A  single      diagonal only                     the original result
  B  exact       all samples matching on all 4     exact-duplicate false negatives
                 tier fields
  C  soft        target ∝ fraction of tier fields  near-neighbour false negatives
                 matched

Identical optimiser, batch, seeds, epochs across conditions; the only change is the
target distribution. Positive counts and target entropy are printed BEFORE the
results, because high-frequency values (wheeze_type=none, intensity=medium) could
make most of the batch a strong positive and collapse the objective — that would look
like a loss improvement while destroying the representation.

Significance is a patient-cluster bootstrap, not a seed t-test: segments from one
patient are not independent.

    python src/multipositive.py --schema_text results/schema_text.json \
        --audio_emb results/ast_feats_clean.npy \
        --audio_index results/ast_feats_clean_index.json \
        --text_emb results/text_emb_bio/all.npz \
        --split_map results/split_official.json --target wheeze
"""
import os, json, argparse
import numpy as np

TAU, EPOCHS, LR, HIDDEN, DROPOUT = 0.07, 500, 1e-3, 1024, 0.1
FIELDS = ["crackle_tier", "wheeze_tier", "wheeze_type", "intensity_tier"]


def tiers(schema):
    """The four categorical fields the similarity is defined over. Missing values
    return None and are excluded from the denominator rather than counted as a match."""
    a = schema["adventitious_sounds"]
    return {
        "crackle_tier": a["crackle"]["likelihood"]["level"],
        "wheeze_tier": a["wheeze"]["likelihood"]["level"],
        "wheeze_type": a["wheeze"]["type"]["value"],
        "intensity_tier": schema["breath_sound"]["intensity"]["level"],
    }


def similarity(T):
    """S[i,j] = fraction of jointly-present tier fields on which i and j agree."""
    n = len(T)
    S = np.zeros((n, n), dtype=np.float32)
    cols = {f: np.array([t.get(f) for t in T], dtype=object) for f in FIELDS}
    for f in FIELDS:
        c = cols[f]
        present = np.array([v is not None for v in c])
        eq = (c[:, None] == c[None, :]) & present[:, None] & present[None, :]
        S += eq.astype(np.float32)
        # denominator counts only fields present in BOTH
        if f == FIELDS[0]:
            D = (present[:, None] & present[None, :]).astype(np.float32)
        else:
            D += (present[:, None] & present[None, :]).astype(np.float32)
    return np.divide(S, np.maximum(D, 1e-8), out=np.zeros_like(S), where=D > 0)


def make_target(S_b, mode, torch):
    """Row-normalised target distribution over the batch for each anchor."""
    n = len(S_b)
    if mode == "single":
        return torch.eye(n, device=S_b.device)
    if mode == "exact":
        P = (S_b >= 1.0 - 1e-6).float()
    elif mode == "soft":
        P = S_b.clone()
    else:                                   # soft-sharp
        # The plain soft target is degenerate here: with four categorical fields it
        # marks 218 of 256 batch entries as positive and its entropy (5.28) sits
        # against the uniform ceiling (5.55), which is not a contrastive objective
        # at all. Thresholding at 0.75 (>=3 of 4 fields agreeing) and sharpening the
        # remainder keeps near-neighbours as positives without flattening the target.
        P = torch.where(S_b >= 0.75, S_b ** 4, torch.zeros_like(S_b))
    # S_b[i,i] is already 1.0 — a record matches itself on every field — so adding an
    # identity here double-weighted the anchor's own positive. Clamp instead.
    P = torch.maximum(P, torch.eye(n, device=S_b.device))
    return P / P.sum(1, keepdim=True).clamp(min=1e-8)


def main():
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_text", required=True)
    ap.add_argument("--audio_emb", required=True); ap.add_argument("--audio_index", required=True)
    ap.add_argument("--text_emb", required=True); ap.add_argument("--split_map", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target", default="wheeze", choices=["wheeze", "crackle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = json.load(open(args.schema_text))
    smap = json.load(open(args.split_map))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(args.manifest))}
    A = np.load(args.audio_emb); aidx = json.load(open(args.audio_index))
    tz = np.load(args.text_emb, allow_pickle=True)
    tpos = {str(i): k for k, i in enumerate(tz["ids"])}; T = tz["emb"]

    ids = [i for i in D if i in aidx and i in tpos and i in smap and i in meta]
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
    Xt = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)
    split = np.array([smap[i] for i in ids])
    pid = np.array([i.split("_")[0] for i in ids])
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK"
    print(f"{len(ids)} segments | train {tr.sum()} test {te.sum()} | "
          f"target={args.target} pos={y[te].mean():.3f}")

    S_tr = similarity([tiers(D[i]["schema"]) for i in np.array(ids)[tr]])
    print(f"\nsimilarity matrix over {S_tr.shape[0]} train segments")
    for mode in ["single", "exact", "soft", "soft-sharp"]:
        St = torch.tensor(S_tr[:args.batch, :args.batch])
        Q = make_target(St, mode, torch)
        npos = (Q > 1e-6).sum(1).float()
        ent = -(Q.clamp(min=1e-12) * Q.clamp(min=1e-12).log()).sum(1)
        print(f"  {mode:<7} positives/anchor {npos.mean():6.1f} "
              f"(max {int(npos.max()):4d})   target entropy {ent.mean():.3f} "
              f"/ {np.log(args.batch):.3f} max")
    for mode in ["exact", "soft", "soft-sharp"]:
        Q = make_target(torch.tensor(S_tr[:args.batch, :args.batch]), mode, torch)
        e = -(Q.clamp(min=1e-12) * Q.clamp(min=1e-12).log()).sum(1).mean()
        # a target within 20% of uniform carries almost no contrastive signal; the
        # first run's plain `soft` hit 5.28/5.55 and its MCC fell while AUROC held
        if e > 0.8 * np.log(args.batch):
            print(f"  WARNING: {mode} target entropy {e:.3f} is near-uniform "
                  f"({np.log(args.batch):.3f}) — treat its result as degenerate")

    res = {}
    for mode in ["single", "exact", "soft", "soft-sharp"]:
        aucs, mccs, scores = [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            head = nn.Sequential(nn.Linear(Xa.shape[1], HIDDEN), nn.LayerNorm(HIDDEN),
                                 nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, Xt.shape[1])).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.1)
            at = torch.tensor(Xa[tr], device=dev); tt = torch.tensor(Xt[tr], device=dev)
            Sall = torch.tensor(S_tr, device=dev)
            for _ in range(EPOCHS):
                perm = torch.randperm(len(at), device=dev)
                for s in range(0, len(perm), args.batch):
                    b = perm[s:s + args.batch]
                    if len(b) < 2: continue
                    za = head(at[b]); zt = tt[b]
                    za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    zt = zt / zt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    lg = za @ zt.T / TAU
                    Sb = Sall[b][:, b]
                    Q = make_target(Sb, mode, torch)
                    # the reverse direction needs its own normalisation: Q is
                    # row-normalised, and P/rowsum is not symmetric even when S is,
                    # so reusing Q against lg.T targets the wrong distribution
                    Qt = make_target(Sb.T, mode, torch)
                    loss = 0.5 * (-(Q * lg.log_softmax(1)).sum(1).mean()
                                  - (Qt * lg.T.log_softmax(1)).sum(1).mean())
                    opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                P = head(torch.tensor(Xa, device=dev)).cpu().numpy()
            clf = LogisticRegression(max_iter=20000, class_weight="balanced").fit(P[tr], y[tr])
            sc = clf.predict_proba(P[te])[:, 1]
            aucs.append(roc_auc_score(y[te], sc))
            mccs.append(matthews_corrcoef(y[te], clf.predict(P[te])))
            scores.append(sc)
        res[mode] = {"auroc": aucs, "mcc": mccs, "score": np.mean(scores, 0)}
        print(f"\n{mode:<7} AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
              f"MCC {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")

    # patient-cluster bootstrap: segments within a patient are not independent
    print(f"\nΔ vs single, patient-cluster bootstrap ({args.boot} resamples)")
    pte, yte = pid[te], y[te]
    pats = np.array(sorted(set(pte)))
    idx_by_pat = {p: np.where(pte == p)[0] for p in pats}
    rng = np.random.RandomState(0)
    for mode in ["exact", "soft", "soft-sharp"]:
        deltas = []
        for _ in range(args.boot):
            take = np.concatenate([idx_by_pat[p] for p in rng.choice(pats, len(pats))])
            if len(set(yte[take])) < 2: continue
            deltas.append(roc_auc_score(yte[take], res[mode]["score"][take])
                          - roc_auc_score(yte[take], res["single"]["score"][take]))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        same = all(np.sign(np.array(res[mode]["auroc"]) - np.array(res["single"]["auroc"])) ==
                   np.sign(np.mean(res[mode]["auroc"]) - np.mean(res["single"]["auroc"])))
        verdict = "IMPROVES" if lo > 0 and same else ("HURTS" if hi < 0 and same else "no effect")
        print(f"  {mode:<7} ΔAUROC {np.mean(deltas):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"3/3 seeds same sign: {same}  -> {verdict}")

    if args.out:
        json.dump({m: {"auroc": v["auroc"], "mcc": v["mcc"]} for m, v in res.items()},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
