"""Statistical audit of the Phase-2 typed-schema result.

The Day-3 conclusion — "shuffling the schema keeps ~3/4 of the gain, so typed is not
learning schema content" — was reached against the wrong comparison axis. It measured
each shuffled arm against `string_trainable`; the question is `typed_intact` vs
`typed_shuffled`, and on that axis the drop is 0.706 → 0.651, i.e. **0.055**, which
points the opposite way. Four further defects made the number unusable:

  - the permutation was applied to train *and* test, so the test schema was scrambled
    too and the arm was not "trained on noise, evaluated honestly";
  - a single fixed permutation, so the shuffled score carries no sampling variance;
  - the `fields` control zeroed the field-id channel while the value vocabulary is
    keyed `field#value`, leaking field identity straight back in;
  - the bootstrap ran on seed-averaged predictions while the table reported
    seed-averaged AUROC — two different estimands.

This rebuilds the test properly. Pre-registered criterion, fixed before running:

    typed_intact beats record-shuffled, AND all seeds agree in sign, AND the paired
    patient-cluster bootstrap CI on (intact − shuffled) excludes zero.

Controls:
  record      schema records permuted across TRAIN segments only, N permutations
  constant    every segment gets the same schema — removes all per-sample signal
  nofield     field identity removed from both the id channel and the value keys

    python src/audit.py --schema_v2 results/schema_v2.json \
        --manifest .../manifest.json --target wheeze --perms 10
"""
import os, json, argparse
import numpy as np

TAU, EPOCHS, LR, DIM, HIDDEN, DROPOUT = 0.07, 500, 1e-3, 128, 1024, 0.1


def bin_of(f, v, edges, keep_field=True):
    """Value key. With keep_field=False the field name is stripped from the key too,
    otherwise `field#value` smuggles field identity past the id-channel ablation."""
    if isinstance(v, (int, float)) and f in edges:
        q = int(np.searchsorted(edges[f], float(v)))
        return f"{f}#q{q}" if keep_field else f"q{q}"
    return f"{f}#{v}" if keep_field else f"{v}"


def build(records, ids, ids_train, blocks, keep_field=True):
    fieldv, provv, cont = {}, {}, {}
    for i in ids_train:
        for f, d in records[i]["fields"].items():
            if d["block"] not in blocks:
                continue
            fieldv.setdefault(f, len(fieldv)); provv.setdefault(d["provenance"], len(provv))
            if not d["missing"] and isinstance(d["value"], (int, float)):
                cont.setdefault(f, []).append(float(d["value"]))
    edges = {f: np.quantile(v, np.linspace(0, 1, 11)[1:-1]) for f, v in cont.items()}
    valv = {}
    for i in ids_train:
        for f, d in records[i]["fields"].items():
            if d["block"] in blocks and not d["missing"]:
                valv.setdefault(bin_of(f, d["value"], edges, keep_field), len(valv))
    F = len(fieldv)
    X = np.zeros((len(ids), F, 4), dtype=np.int64); M = np.zeros((len(ids), F), np.float32)
    for n, i in enumerate(ids):
        for f, d in records[i]["fields"].items():
            if d["block"] not in blocks or f not in fieldv:
                continue
            j = fieldv[f]
            miss = int(d["missing"])
            vb = 0 if miss else valv.get(bin_of(f, d["value"], edges, keep_field), 0)
            X[n, j] = [j if keep_field else 0, vb, provv.get(d["provenance"], 0), miss]
            M[n, j] = 1.0
    return X, M, len(fieldv), len(valv), len(provv)


def run_seed(X, M, Xa, tr, te, y, seed, batch, dev, torch, nn, nf, nv, npv):
    from sklearn.linear_model import LogisticRegression
    torch.manual_seed(seed); np.random.seed(seed)

    class Typed(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Embedding(max(nf, 1), DIM); s.v = nn.Embedding(nv + 1, DIM)
            s.p = nn.Embedding(max(npv, 1), DIM); s.m = nn.Embedding(2, DIM)
            s.mlp = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.LayerNorm(HIDDEN),
                                  nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 768))
        def forward(s, x, mask):
            n = s.f(x[..., 0]) + s.v(x[..., 1]) + s.p(x[..., 2]) + s.m(x[..., 3])
            return s.mlp((n * mask.unsqueeze(-1)).sum(1) /
                         mask.sum(1, keepdim=True).clamp(min=1))

    head = nn.Sequential(nn.Linear(Xa.shape[1], HIDDEN), nn.LayerNorm(HIDDEN),
                         nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 768)).to(dev)
    txt = Typed().to(dev)
    opt = torch.optim.AdamW(list(head.parameters()) + list(txt.parameters()),
                            lr=LR, weight_decay=0.1)
    At = torch.tensor(Xa, device=dev)
    Xt = torch.tensor(X, device=dev); Mt = torch.tensor(M, device=dev)
    tri = np.where(tr)[0]
    for _ in range(EPOCHS):
        perm = np.random.permutation(tri)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
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
    return clf.predict_proba(P[te])[:, 1]


def main():
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--schema_v2", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_emb", default="results/ast_feats_clean.npy")
    ap.add_argument("--audio_index", default="results/ast_feats_clean_index.json")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--target", default="wheeze"); ap.add_argument("--blocks", nargs="+",
                                                                  default=["acoustic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--perms", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="results/audit.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = json.load(open(args.schema_v2)); smap = json.load(open(args.split_map))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(args.manifest))}
    A = np.load(args.audio_emb); aidx = json.load(open(args.audio_index))
    ids = [i for i in D if i in aidx and i in smap and i in meta]
    split = np.array([smap[i] for i in ids]); pid = np.array([i.split("_")[0] for i in ids])
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK"
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
    blocks = tuple(args.blocks)
    idtr = list(np.array(ids)[tr])

    X, M, nf, nv, npv = build(D, ids, idtr, blocks, keep_field=True)
    Xnf, Mnf, nf2, nv2, npv2 = build(D, ids, idtr, blocks, keep_field=False)
    print(f"{len(ids)} segments | intact {nf} fields/{nv} bins | "
          f"nofield {nf2} fields/{nv2} bins (field id stripped from value keys too)")
    print(f"criterion: intact > record-shuffled AND all {len(args.seeds)} seeds same "
          f"sign AND paired patient-bootstrap CI excludes 0\n")

    tri = np.where(tr)[0]
    def shuffled(seedp):
        """Permute schema rows among TRAIN segments only; test schema untouched."""
        Xs = X.copy()
        rs = np.random.RandomState(1000 + seedp)
        Xs[tri] = Xs[tri][rs.permutation(len(tri))]
        return Xs

    preds = {}
    for name, Xv, Mv, dims in [("intact", X, M, (nf, nv, npv)),
                               ("constant", np.repeat(X[:1], len(X), 0), M, (nf, nv, npv)),
                               ("nofield", Xnf, Mnf, (nf2, nv2, npv2))]:
        preds[name] = [run_seed(Xv, Mv, Xa, tr, te, y, s, args.batch, dev, torch, nn, *dims)
                       for s in args.seeds]
        aur = [roc_auc_score(y[te], p) for p in preds[name]]
        print(f"  {name:<20} AUROC {np.mean(aur):.3f} ± {np.std(aur):.3f}")

    shuf = []
    for k in range(args.perms):
        Xs = shuffled(k)
        p = run_seed(Xs, M, Xa, tr, te, y, args.seeds[k % len(args.seeds)],
                     args.batch, dev, torch, nn, nf, nv, npv)
        shuf.append(p)
        if k == 0 or (k + 1) % 5 == 0:
            print(f"  record-shuffle perm {k+1}/{args.perms} "
                  f"AUROC {roc_auc_score(y[te], p):.3f}", flush=True)
    aur_s = [roc_auc_score(y[te], p) for p in shuf]
    print(f"  {'record-shuffled':<20} AUROC {np.mean(aur_s):.3f} ± {np.std(aur_s):.3f} "
          f"over {args.perms} permutations")

    # paired bootstrap per seed, then aggregate — matching the estimand the table reports
    yte, pte = y[te], pid[te]
    pats = np.array(sorted(set(pte))); by = {p: np.where(pte == p)[0] for p in pats}
    rng = np.random.RandomState(0)
    print(f"\npaired patient-cluster bootstrap ({args.boot} resamples, {len(pats)} patients)")
    out = {"target": args.target, "auroc": {}}
    for name, ref in [("record-shuffled", shuf), ("constant", preds["constant"]),
                      ("nofield", preds["nofield"])]:
        per_seed = []
        for i_s, pi in enumerate(preds["intact"]):
            pr = ref[i_s % len(ref)]
            d = []
            for _ in range(args.boot):
                take = np.concatenate([by[p] for p in rng.choice(pats, len(pats))])
                if len(set(yte[take])) < 2: continue
                d.append(roc_auc_score(yte[take], pi[take]) - roc_auc_score(yte[take], pr[take]))
            per_seed.append(np.array(d))
        allb = np.concatenate(per_seed)
        lo, hi = np.percentile(allb, [2.5, 97.5])
        signs = [np.mean(x) for x in per_seed]
        same = bool(np.all(np.sign(signs) == np.sign(np.mean(signs))))
        v = "SIGNAL" if lo > 0 and same else ("REVERSED" if hi < 0 and same else "inconclusive")
        print(f"  intact − {name:<17} Δ {np.mean(allb):+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{len(signs)}/{len(signs)} same sign {same}  -> {v}")
        out["auroc"][name] = {"delta": float(np.mean(allb)), "ci": [float(lo), float(hi)],
                              "same_sign": same, "verdict": v}

    out["auroc"]["intact"] = float(np.mean([roc_auc_score(y[te], p) for p in preds["intact"]]))
    out["test_ids"] = [i for i, k in zip(ids, te) if k]
    out["predictions"] = {k: [p.tolist() for p in v] for k, v in preds.items()}
    out["predictions"]["record-shuffled"] = [p.tolist() for p in shuf]
    json.dump(out, open(args.out, "w"))
    print(f"\nwrote {args.out} (per-test-id predictions included)")


if __name__ == "__main__":
    main()
