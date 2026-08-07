"""
Collect the alignment sweep into the paper's main table (and Figure 1).

Always prints the RAW frozen-feature baseline in the same row set, computed on the
identical split and seeds. Without it the alignment numbers are uninterpretable —
the question is not "which text condition wins" but "does any of them beat doing
nothing at all".

    python src/summarize.py --results results --target wheeze \
        --audio_emb .../opera_feats_clean.npy --audio_index .../opera_index_clean.json \
        --manifest .../manifest.json --schema_text results/schema_text.json
"""
import os, json, glob, argparse
import numpy as np

CONDS = ["dataset", "signal", "model", "all"]


def raw_baseline(audio_emb, audio_index, manifest, target, seeds=(0, 1, 2)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    A = np.load(audio_emb); aidx = json.load(open(audio_index))
    meta = {os.path.basename(e["path"]): e for e in json.load(open(manifest))}
    ids = [i for i in aidx if i in meta]
    X = np.stack([A[aidx[i]] for i in ids])
    split = np.array([meta[i]["split"] for i in ids])
    pid = np.array([i.split("_")[0] for i in ids])
    tr, te = split == "train", split == "test"
    assert not (set(pid[tr]) & set(pid[te])), "PATIENT LEAK in raw baseline"
    y = np.array([int(meta[i]["label"] in (target, "both")) for i in ids])
    ms, au = [], []
    for s in seeds:
        c = LogisticRegression(max_iter=3000, class_weight="balanced",
                               random_state=s).fit(X[tr], y[tr])
        ms.append(matthews_corrcoef(y[te], c.predict(X[te])))
        au.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
    return {"mcc": (float(np.mean(ms)), float(np.std(ms))),
            "auroc": (float(np.mean(au)), float(np.std(au)))}


def text_resolution(schema_text):
    if not schema_text or not os.path.exists(schema_text):
        return {}
    D = json.load(open(schema_text))
    return {c: len({v["text"][c] for v in D.values() if c in v["text"]}) for c in CONDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--target", default="wheeze")
    ap.add_argument("--audio_emb"); ap.add_argument("--audio_index")
    ap.add_argument("--manifest"); ap.add_argument("--schema_text", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    uniq = text_resolution(args.schema_text)
    base = None
    if args.audio_emb and args.audio_index and args.manifest:
        base = raw_baseline(args.audio_emb, args.audio_index, args.manifest, args.target)

    rows = []
    if base:
        rows.append(("RAW features (no alignment)", None,
                     base["mcc"], base["auroc"], None))
    for c in CONDS:
        f = os.path.join(args.results, f"align_{args.target}_{c}.json")
        if not os.path.exists(f):
            continue
        a = json.load(open(f))["aggregate"]
        rows.append((f"align / {c}", uniq.get(c),
                     (a["mcc"]["mean"], a["mcc"]["sd"]),
                     (a["auroc"]["mean"], a["auroc"]["sd"]),
                     a["retrieval_r1"]["mean"]))

    w = max(len(r[0]) for r in rows)
    print(f"\n=== {args.target} · patient-independent ICBHI test · 3 seeds ===\n")
    print(f"{'condition'.ljust(w)}  {'uniq':>6}  {'MCC':<15} {'AUROC':<15} {'R@1':>7}  {'dAUROC':>7}")
    ref = base["auroc"][0] if base else None
    for name, u, mcc, auroc, r1 in rows:
        d = "" if (ref is None or name.startswith("RAW")) else f"{auroc[0]-ref:+.3f}"
        print(f"{name.ljust(w)}  {('-' if u is None else u):>6}  "
              f"{mcc[0]:.3f} ± {mcc[1]:.3f}  {auroc[0]:.3f} ± {auroc[1]:.3f}  "
              f"{('-' if r1 is None else f'{r1:.4f}'):>7}  {d:>7}")

    if ref is not None:
        best = max((r for r in rows if not r[0].startswith("RAW")),
                   key=lambda r: r[3][0], default=None)
        if best:
            print()
            if best[3][0] < ref:
                print(f"NEGATIVE RESULT: every alignment condition is below the frozen-feature "
                      f"baseline.\n  best = {best[0]} at AUROC {best[3][0]:.3f} vs raw "
                      f"{ref:.3f} ({best[3][0]-ref:+.3f}).\n  Contrastive alignment on frozen "
                      f"features is discarding task information, not adding it.")
            else:
                print(f"Best condition {best[0]} beats raw features "
                      f"({best[3][0]:.3f} vs {ref:.3f}).")

    if args.out:
        json.dump({"target": args.target, "raw_baseline": base, "resolution": uniq,
                   "rows": [{"condition": n, "uniq": u, "mcc": m, "auroc": a, "r1": r}
                            for n, u, m, a, r in rows]},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
