"""Compare two text arms under the same criterion used to reject the loss variants.

`matched_serialized` scored 0.696 against `matched_template`'s 0.675 — a +0.021 gap
between arms carrying identical information, which would be the first evidence that
structure matters. But +0.021 is the same magnitude as the multi-positive effects
already declared "no effect" (+0.022, CI crossing zero), so it gets the same test
rather than a friendlier one: patient-cluster bootstrap plus a same-sign requirement
across seeds.

Also supports the field-ablation controls, since the v2 arms include continuous
`duration_s` and duration alone reaches AUROC 0.644 on crackle — a gain could be that
confound arriving through the text rather than anything structural.

    python src/compare_arms.py --arms matched_template matched_serialized \
        --text_dir results/text_emb_v2 --target wheeze
"""
import os, json, argparse
import numpy as np

TAU, EPOCHS, LR, HIDDEN, DROPOUT = 0.07, 500, 1e-3, 1024, 0.1


def train_arm(Xa, Xt, tr, te, y, seeds, batch, dev, torch, nn):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    aucs, mccs, scores = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        head = nn.Sequential(nn.Linear(Xa.shape[1], HIDDEN), nn.LayerNorm(HIDDEN),
                             nn.ReLU(), nn.Dropout(DROPOUT),
                             nn.Linear(HIDDEN, Xt.shape[1])).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.1)
        at = torch.tensor(Xa[tr], device=dev); tt = torch.tensor(Xt[tr], device=dev)
        for _ in range(EPOCHS):
            perm = torch.randperm(len(at), device=dev)
            for s in range(0, len(perm), batch):
                b = perm[s:s + batch]
                if len(b) < 2: continue
                za = head(at[b])
                za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                zt = tt[b] / tt[b].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                lg = za @ zt.T / TAU
                t = torch.arange(len(b), device=dev)
                loss = 0.5 * (nn.functional.cross_entropy(lg, t) +
                              nn.functional.cross_entropy(lg.T, t))
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            P = head(torch.tensor(Xa, device=dev)).cpu().numpy()
        clf = LogisticRegression(max_iter=20000, class_weight="balanced").fit(P[tr], y[tr])
        sc = clf.predict_proba(P[te])[:, 1]
        aucs.append(roc_auc_score(y[te], sc))
        mccs.append(matthews_corrcoef(y[te], clf.predict(P[te])))
        scores.append(sc)
    return {"auroc": aucs, "mcc": mccs, "score": np.mean(scores, 0)}


def main():
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--audio_emb", default="results/ast_feats_clean.npy")
    ap.add_argument("--audio_index", default="results/ast_feats_clean_index.json")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target", default="wheeze", choices=["wheeze", "crackle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smap = json.load(open(args.split_map))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(args.manifest))}
    A = np.load(args.audio_emb); aidx = json.load(open(args.audio_index))

    embs = {}
    for a in args.arms:
        z = np.load(os.path.join(args.text_dir, f"{a}.npz"), allow_pickle=True)
        embs[a] = ({str(i): k for k, i in enumerate(z["ids"])}, z["emb"])

    ids = [i for i in aidx if i in smap and i in meta
           and all(i in embs[a][0] for a in args.arms)]
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
    split = np.array([smap[i] for i in ids]); pid = np.array([i.split("_")[0] for i in ids])
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK"
    print(f"{len(ids)} segments | train {tr.sum()} test {te.sum()} | target={args.target}")

    res = {}
    for a in args.arms:
        tpos, T = embs[a]
        Xt = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)
        res[a] = train_arm(Xa, Xt, tr, te, y, args.seeds, args.batch, dev, torch, nn)
        print(f"  {a:<22} AUROC {np.mean(res[a]['auroc']):.3f} ± {np.std(res[a]['auroc']):.3f}"
              f"   MCC {np.mean(res[a]['mcc']):.3f}")

    ref = args.arms[0]
    pte, yte = pid[te], y[te]
    pats = np.array(sorted(set(pte)))
    by = {p: np.where(pte == p)[0] for p in pats}
    rng = np.random.RandomState(0)
    print(f"\nΔ vs {ref}, patient-cluster bootstrap ({args.boot} resamples, "
          f"{len(pats)} test patients)")
    for a in args.arms[1:]:
        d = []
        for _ in range(args.boot):
            take = np.concatenate([by[p] for p in rng.choice(pats, len(pats))])
            if len(set(yte[take])) < 2: continue
            d.append(roc_auc_score(yte[take], res[a]["score"][take])
                     - roc_auc_score(yte[take], res[ref]["score"][take]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs = np.array(res[a]["auroc"]) - np.array(res[ref]["auroc"])
        same = bool(np.all(np.sign(diffs) == np.sign(np.mean(diffs))))
        verdict = "IMPROVES" if lo > 0 and same else ("HURTS" if hi < 0 and same else "no effect")
        print(f"  {a:<22} ΔAUROC {np.mean(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"3/3 same sign: {same}  -> {verdict}")

    if args.out:
        json.dump({a: {"auroc": v["auroc"], "mcc": v["mcc"]} for a, v in res.items()},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
