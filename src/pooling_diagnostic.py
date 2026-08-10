"""How much pathology information does mean-pooling destroy?

Every alignment run in this project compared a single mean-pooled vector per cycle
against a single text embedding. That is the wrong operation for crackles: they are
5-20 ms transients inside a 2-3 s cycle, so averaging over the whole spectrogram
averages them away. Local (patch-level) alignment — GLoRIA/BioViL's fix for chest
X-ray — is the one untested variant with a mechanism behind it rather than a "scale
it up" argument.

Whether it is worth 3-4 weeks depends on a number nobody has measured: does the
label information actually live in the time axis, or is it already in the global
mean? This settles that in an hour.

Four readouts on the same frozen AST features, same official split, same probe:

  mean       global average over time            (what every run so far used)
  max        max over time                       (transient-sensitive)
  mean+max   concatenated                        (both)
  segments   pooled within K time segments,      (coarse time resolution)
             concatenated

If `mean` is already as good as the rest, the pooling operation is not the
bottleneck and local alignment cannot help. If the time-resolved readouts are
clearly better, that gap is the headroom local alignment would be competing for.

    python src/pooling_diagnostic.py --segments 4
"""
import os, json, argparse
import numpy as np
import torch
import soundfile as sf
import librosa

SR = 16000
R = "/mnt/hd/data_heliu/icbhi_pathology_fidelity"
AST_PATH = "/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593"


def load(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def probe(X, ids, smap, meta, tgt, seeds=(0, 1, 2)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    keep = [j for j, i in enumerate(ids) if i in smap and i in meta]
    X, kid = X[keep], [ids[j] for j in keep]
    sp = np.array([smap[i] for i in kid])
    tr, te = sp == "train", sp == "test"
    y = np.array([int(meta[i]["label"] in (tgt, "both")) for i in kid])
    return float(np.mean([
        roc_auc_score(y[te], LogisticRegression(max_iter=20000, class_weight="balanced",
                                                random_state=s).fit(X[tr], y[tr])
                      .predict_proba(X[te])[:, 1]) for s in seeds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6,
                    help="AST's first 6 layers beat all 12 on this task (§2.98)")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from transformers import ASTModel, AutoFeatureExtractor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(AST_PATH)
    m = ASTModel.from_pretrained(AST_PATH)
    if args.layers < 12:
        m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)

    seg = json.load(open(f"{R}/data/segments/manifest.json"))
    meta = {os.path.basename(e["path"]): e for e in seg}
    smap = json.load(open("results/split_official.json"))

    pools = {"mean": [], "max": [], "segments": []}
    ids = []
    for i in range(0, len(seg), args.batch):
        chunk = seg[i:i + args.batch]
        ys = [load(os.path.join(R, e["path"])) for e in chunk]
        inp = fe(ys, sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = m(**inp).last_hidden_state          # (B, tokens, D)
        pools["mean"].append(h.mean(1).cpu().numpy())
        pools["max"].append(h.max(1).values.cpu().numpy())
        # split the token axis into K contiguous blocks and mean-pool each, so the
        # readout keeps coarse temporal structure the global mean discards
        k = args.segments
        b = torch.stack([c.mean(1) for c in torch.chunk(h, k, dim=1)], 1)
        pools["segments"].append(b.flatten(1).cpu().numpy())
        ids += [os.path.basename(e["path"]) for e in chunk]
        if (i // args.batch) % 50 == 0:
            print(f"  {i}/{len(seg)}", flush=True)

    F = {k: np.concatenate(v) for k, v in pools.items()}
    F["mean+max"] = np.concatenate([F["mean"], F["max"]], 1)

    print(f"\nAST first {args.layers} layers, official split, 3 seeds\n")
    print("%-12s %6s %9s %9s" % ("readout", "dim", "wheeze", "crackle"))
    order = ["mean", "max", "mean+max", "segments"]
    res = {}
    for k in order:
        w = probe(F[k], ids, smap, meta, "wheeze")
        c = probe(F[k], ids, smap, meta, "crackle")
        res[k] = (w, c)
        print("%-12s %6d %9.3f %9.3f" % (k, F[k].shape[1], w, c))

    bw = max(res, key=lambda k: res[k][0])
    bc = max(res, key=lambda k: res[k][1])
    print(f"\nheadroom over global mean:")
    print(f"  wheeze  best={bw} {res[bw][0]:.3f} vs mean {res['mean'][0]:.3f} "
          f"({res[bw][0] - res['mean'][0]:+.3f})")
    print(f"  crackle best={bc} {res[bc][1]:.3f} vs mean {res['mean'][1]:.3f} "
          f"({res[bc][1] - res['mean'][1]:+.3f})")
    print("\nA gap here is what local alignment would be competing for. No gap means "
          "the pooling operation is not the bottleneck and that direction is closed too.")


if __name__ == "__main__":
    main()
