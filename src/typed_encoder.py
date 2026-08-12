"""The last untested representation: a typed, permutation-invariant set encoder.

`matched_template` and `matched_serialized` carry identical information and differ
only in string format; the gap between them flipped sign once continuous `duration_s`
was removed, so serialisation format does nothing at matched content. A set encoder
is the one representation left that a string cannot express:

  - field identity is an explicit embedding, not a token the tokeniser may split;
  - order carries no information (masked mean over nodes), so nothing can be learned
    from field sequence;
  - missing values get their own embedding rather than the words "not available";
  - provenance is a separate channel, not prose.

Each field becomes one node:

    node = E_field[f] + E_valuebin[bin(v)] + E_provenance[p] + E_missing[m]
    z    = MLP(masked_mean(nodes))

Both towers stay frozen elsewhere in this project; here the *text* tower is this
encoder and it trains, while the audio tower stays frozen — so the comparison against
the string arms is not confounded by trainability on the audio side.

Value binning is fit on TRAIN ONLY. Continuous fields (duration_s) are quantised to
deciles; categorical fields keep their own vocabulary. Fitting bins on all data would
leak test distribution into the representation.

    python src/typed_encoder.py --schema_v2 results/schema_v2.json \
        --manifest .../manifest.json --target wheeze --blocks acoustic
"""
import os, json, argparse
import numpy as np

TAU, EPOCHS, LR, DIM, HIDDEN, DROPOUT = 0.07, 500, 1e-3, 128, 1024, 0.1


def build_vocab(records, ids_train, blocks):
    """Field/value/provenance vocabularies, fit on train only."""
    fieldv, valv, provv = {}, {}, {}
    cont = {}
    for i in ids_train:
        for f, d in records[i]["fields"].items():
            if d["block"] not in blocks:
                continue
            fieldv.setdefault(f, len(fieldv))
            provv.setdefault(d["provenance"], len(provv))
            if d["missing"]:
                continue
            v = d["value"]
            if isinstance(v, (int, float)):
                cont.setdefault(f, []).append(float(v))
    edges = {f: np.quantile(vs, np.linspace(0, 1, 11)[1:-1]) for f, vs in cont.items()}
    for i in ids_train:
        for f, d in records[i]["fields"].items():
            if d["block"] not in blocks or d["missing"]:
                continue
            valv.setdefault(bin_of(f, d["value"], edges), len(valv))
    return fieldv, valv, provv, edges


def bin_of(f, v, edges):
    if isinstance(v, (int, float)) and f in edges:
        return f"{f}#q{int(np.searchsorted(edges[f], float(v)))}"
    return f"{f}#{v}"


def encode_nodes(records, ids, fieldv, valv, provv, edges, blocks):
    """-> (N, F, 4) int index tensor + (N, F) mask, one row per field slot."""
    F = len(fieldv)
    X = np.zeros((len(ids), F, 4), dtype=np.int64)
    M = np.zeros((len(ids), F), dtype=np.float32)
    for n, i in enumerate(ids):
        for f, d in records[i]["fields"].items():
            if d["block"] not in blocks or f not in fieldv:
                continue
            j = fieldv[f]
            miss = int(d["missing"])
            vb = 0 if miss else valv.get(bin_of(f, d["value"], edges), 0)
            X[n, j] = [j, vb, provv.get(d["provenance"], 0), miss]
            M[n, j] = 1.0
    return X, M


def main():
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_v2", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_emb", default="results/ast_feats_clean.npy")
    ap.add_argument("--audio_index", default="results/ast_feats_clean_index.json")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--target", default="wheeze", choices=["wheeze", "crackle"])
    ap.add_argument("--blocks", nargs="+", default=["acoustic"],
                    choices=["recording", "acoustic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = json.load(open(args.schema_v2))
    smap = json.load(open(args.split_map))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(args.manifest))}
    A = np.load(args.audio_emb); aidx = json.load(open(args.audio_index))

    ids = [i for i in D if i in aidx and i in smap and i in meta]
    split = np.array([smap[i] for i in ids]); pid = np.array([i.split("_")[0] for i in ids])
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK"

    blocks = tuple(args.blocks)
    fieldv, valv, provv, edges = build_vocab(D, list(np.array(ids)[tr]), blocks)
    print(f"{len(ids)} segments | blocks={blocks} | "
          f"{len(fieldv)} fields, {len(valv)} value bins, {len(provv)} provenances")
    X, M = encode_nodes(D, ids, fieldv, valv, provv, edges, blocks)
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)

    class Typed(nn.Module):
        def __init__(s, nf, nv, np_, d=DIM, out=768):
            super().__init__()
            s.f = nn.Embedding(nf, d); s.v = nn.Embedding(nv + 1, d)
            s.p = nn.Embedding(np_, d); s.m = nn.Embedding(2, d)
            s.mlp = nn.Sequential(nn.Linear(d, HIDDEN), nn.LayerNorm(HIDDEN),
                                  nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, out))

        def forward(s, x, mask):
            n = s.f(x[..., 0]) + s.v(x[..., 1]) + s.p(x[..., 2]) + s.m(x[..., 3])
            pooled = (n * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return s.mlp(pooled)

    Xt = torch.tensor(X, device=dev); Mt = torch.tensor(M, device=dev)
    At = torch.tensor(Xa, device=dev)
    aucs, mccs, scores = [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        txt = Typed(len(fieldv), len(valv), len(provv)).to(dev)
        head = nn.Sequential(nn.Linear(Xa.shape[1], HIDDEN), nn.LayerNorm(HIDDEN),
                             nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 768)).to(dev)
        opt = torch.optim.AdamW(list(txt.parameters()) + list(head.parameters()),
                                lr=LR, weight_decay=0.1)
        tri = np.where(tr)[0]
        for _ in range(EPOCHS):
            perm = np.random.permutation(tri)
            for s in range(0, len(perm), args.batch):
                b = perm[s:s + args.batch]
                if len(b) < 2: continue
                bi = torch.tensor(b, device=dev)
                za, zt = head(At[bi]), txt(Xt[bi], Mt[bi])
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
        print(f"  seed {seed}: AUROC {aucs[-1]:.3f}  MCC {mccs[-1]:.3f}", flush=True)

    print(f"\ntyped_schema  AUROC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}   "
          f"MCC {np.mean(mccs):.3f} ± {np.std(mccs):.3f}")
    if args.out:
        json.dump({"auroc": aucs, "mcc": mccs, "blocks": list(blocks),
                   "score": np.mean(scores, 0).tolist(),
                   "test_ids": [i for i, k in zip(ids, te) if k]},
                  open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
