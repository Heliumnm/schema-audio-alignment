"""Is typed_schema's gain structure, or just a trainable text tower?

typed_schema scored 0.697 against matched_template_acoustic's 0.659 — the largest
effect in Phase 2. But the comparison is confounded: typed trains its text encoder
from scratch while the string arms consume *frozen* Bio_ClinicalBERT embeddings, so
typed can reshape its text side to the task and the string arms cannot.

Three arms, all with the same trainable audio head, isolate it:

  string_frozen     frozen text embeddings, audio head only     (what was measured)
  string_trainable  frozen embeddings + a trainable text MLP    capacity-matched
  typed             typed set encoder, trained                  structure + capacity

  typed vs string_trainable  ->  the structure effect
  string_trainable vs string_frozen  ->  the capacity effect

Same bootstrap criterion as everywhere else: patient-cluster resampling plus a
same-sign requirement across seeds. +0.038 is only three times the effects already
rejected, so it does not get a friendlier test.

    python src/typed_vs_string.py --schema_v2 results/schema_v2.json \
        --text_emb results/text_emb_v2/matched_template_acoustic.npz \
        --manifest .../manifest.json --target wheeze
"""
import os, json, argparse
import numpy as np

TAU, EPOCHS, LR, DIM, HIDDEN, DROPOUT = 0.07, 500, 1e-3, 128, 1024, 0.1


def main():
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, matthews_corrcoef
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from typed_encoder import build_vocab, encode_nodes

    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_v2", required=True); ap.add_argument("--text_emb", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_emb", default="results/ast_feats_clean.npy")
    ap.add_argument("--audio_index", default="results/ast_feats_clean_index.json")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--target", default="wheeze"); ap.add_argument("--blocks", nargs="+",
                                                                  default=["acoustic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--shuffle", default="none",
                    choices=["none", "all", "values", "fields"],
                    help="control: destroy part of the schema and see if the gain "
                         "survives. 'all' permutes whole records across segments, "
                         "'values' permutes each field's values independently "
                         "(keeping marginals), 'fields' strips field identity.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = json.load(open(args.schema_v2)); smap = json.load(open(args.split_map))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(args.manifest))}
    A = np.load(args.audio_emb); aidx = json.load(open(args.audio_index))
    tz = np.load(args.text_emb, allow_pickle=True)
    tpos = {str(i): k for k, i in enumerate(tz["ids"])}; T = tz["emb"]

    ids = [i for i in D if i in aidx and i in smap and i in meta and i in tpos]
    split = np.array([smap[i] for i in ids]); pid = np.array([i.split("_")[0] for i in ids])
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK"

    blocks = tuple(args.blocks)
    fv, vv, pv, edges = build_vocab(D, list(np.array(ids)[tr]), blocks)
    X, M = encode_nodes(D, ids, fv, vv, pv, edges, blocks)
    if args.shuffle != "none":
        rs = np.random.RandomState(1234)
        if args.shuffle == "all":
            # whole schema records permuted across segments: destroys audio-schema
            # correspondence entirely while keeping the schema distribution intact
            X = X[rs.permutation(len(X))]
        elif args.shuffle == "values":
            # each field's value column permuted independently: field marginals and
            # co-occurrence with provenance survive, joint structure does not
            for j in range(X.shape[1]):
                X[:, j, 1] = X[rs.permutation(len(X)), j, 1]
        else:                                    # fields
            # strip field identity, keep values: tests whether typing per se matters
            X[..., 0] = 0
        print(f"CONTROL: schema shuffled ({args.shuffle}) — a surviving gain would "
              f"mean typed is not learning schema content")
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
    Xs = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)
    print(f"{len(ids)} segments | blocks={blocks} | {len(fv)} fields, {len(vv)} bins")

    At = torch.tensor(Xa, device=dev); St = torch.tensor(Xs, device=dev)
    Xt = torch.tensor(X, device=dev); Mt = torch.tensor(M, device=dev)

    class Typed(nn.Module):
        def __init__(s, nf, nv, np_, d=DIM, out=768):
            super().__init__()
            s.f = nn.Embedding(nf, d); s.v = nn.Embedding(nv + 1, d)
            s.p = nn.Embedding(np_, d); s.m = nn.Embedding(2, d)
            s.mlp = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN),
                                  nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, out))
        def forward(s, x, mask):
            n = s.f(x[..., 0]) + s.v(x[..., 1]) + s.p(x[..., 2]) + s.m(x[..., 3])
            return s.mlp((n * mask.unsqueeze(-1)).sum(1) /
                         mask.sum(1, keepdim=True).clamp(min=1))

    def mlp(din, dout):
        return nn.Sequential(nn.Linear(din, HIDDEN), nn.LayerNorm(HIDDEN),
                             nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, dout))

    res = {}
    for arm in ["string_frozen", "string_trainable", "typed"]:
        aucs, mccs, scores = [], [], []
        for seed in args.seeds:
            torch.manual_seed(seed); np.random.seed(seed)
            head = mlp(Xa.shape[1], 768).to(dev)
            if arm == "typed":
                txt = Typed(len(fv), len(vv), len(pv)).to(dev)
            elif arm == "string_trainable":
                txt = mlp(Xs.shape[1], 768).to(dev)   # same shape as the audio head
            else:
                txt = None
            params = list(head.parameters()) + (list(txt.parameters()) if txt else [])
            opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.1)
            tri = np.where(tr)[0]
            for _ in range(EPOCHS):
                for s in range(0, len(tri), args.batch):
                    b = np.random.permutation(tri)[s:s + args.batch]
                    if len(b) < 2: continue
                    bi = torch.tensor(b, device=dev)
                    za = head(At[bi])
                    zt = (txt(Xt[bi], Mt[bi]) if arm == "typed"
                          else (txt(St[bi]) if txt else St[bi]))
                    za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    zt = zt / zt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    lg = za @ zt.T / TAU
                    t = torch.arange(len(b), device=dev)
                    loss = 0.5 * (nn.functional.cross_entropy(lg, t) +
                                  nn.functional.cross_entropy(lg.T, t))
                    opt.zero_grad(); loss.backward(); opt.step()
            head.eval()
            with torch.no_grad():
                P = head(At).cpu().numpy()
            clf = LogisticRegression(max_iter=20000, class_weight="balanced").fit(P[tr], y[tr])
            sc = clf.predict_proba(P[te])[:, 1]
            aucs.append(roc_auc_score(y[te], sc))
            mccs.append(matthews_corrcoef(y[te], clf.predict(P[te])))
            scores.append(sc)
        res[arm] = {"auroc": aucs, "mcc": mccs, "score": np.mean(scores, 0)}
        print(f"  {arm:<18} AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
              f"MCC {np.mean(mccs):.3f}")

    pte, yte = pid[te], y[te]
    pats = np.array(sorted(set(pte))); by = {p: np.where(pte == p)[0] for p in pats}
    rng = np.random.RandomState(0)
    print(f"\npatient-cluster bootstrap ({args.boot} resamples, {len(pats)} patients)")
    for a, b in [("string_trainable", "string_frozen"), ("typed", "string_trainable"),
                 ("typed", "string_frozen")]:
        d = []
        for _ in range(args.boot):
            take = np.concatenate([by[p] for p in rng.choice(pats, len(pats))])
            if len(set(yte[take])) < 2: continue
            d.append(roc_auc_score(yte[take], res[a]["score"][take])
                     - roc_auc_score(yte[take], res[b]["score"][take]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs = np.array(res[a]["auroc"]) - np.array(res[b]["auroc"])
        same = bool(np.all(np.sign(diffs) == np.sign(np.mean(diffs))))
        v = "IMPROVES" if lo > 0 and same else ("HURTS" if hi < 0 and same else "no effect")
        print(f"  {a:<17} vs {b:<17} Δ {np.mean(d):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"same sign {same}  -> {v}")

    if args.out:
        json.dump({a: {"auroc": v["auroc"], "mcc": v["mcc"]} for a, v in res.items()},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
