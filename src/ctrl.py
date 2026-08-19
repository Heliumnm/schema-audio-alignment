"""Three controls on the AST-beats-respiratory-encoders finding.

The headline gap (AST 0.796 vs OPERA-CE 0.531 on ICBHI wheeze) is confounded three
ways: AST is 17x larger (86.2 M vs 5.0 M), sees 128 mel bins to 8 kHz rather than 64
to 2 kHz, and was compared on one task of one corpus. These isolate bandwidth and
depth; the KAUH replication is run separately.
"""
import os, json, argparse
import numpy as np, torch, soundfile as sf, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SR, HOP = 16000, 512
R = "/mnt/hd/data_heliu/icbhi_pathology_fidelity"


def mel(a, n_mels, fmax, nfft=1024):
    S = librosa.feature.melspectrogram(y=a, sr=SR, n_mels=n_mels, fmin=50, fmax=fmax,
                                       n_fft=nfft, hop_length=HOP)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
    return S.T.astype(np.float32)


def load(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1: y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def cola_encoder(ckpt, prefix, device):
    from efficientnet_pytorch import EfficientNet
    class E(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.cnn1 = torch.nn.Conv2d(1, 3, 3)
            s.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False,
                                                    drop_connect_rate=0.1)
        def forward(s, x):
            return s.efficientnet(s.cnn1(x.unsqueeze(1))).squeeze(3).squeeze(2)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    w = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    m = E(); miss, unexp = m.load_state_dict(w, strict=False)
    assert not miss and not unexp, (list(miss)[:3], list(unexp)[:3])
    return m.eval().to(device)


def extract_cola(enc, seg, n_mels, fmax, device, sec=8.0, bs=8):
    nf = int(sec * SR / HOP) + 1
    ids, out = [], []
    for i in range(0, len(seg), bs):
        ms, ks = [], []
        for e in seg[i:i + bs]:
            m = mel(load(os.path.join(R, e["path"])), n_mels, fmax)
            m = np.pad(m, ((0, nf - len(m)), (0, 0))) if len(m) < nf else m[(len(m)-nf)//2:(len(m)-nf)//2+nf]
            ms.append(m); ks.append(os.path.basename(e["path"]))
        with torch.no_grad():
            out.append(enc(torch.tensor(np.stack(ms), device=device)).cpu().numpy())
        ids += ks
    return np.concatenate(out), ids


def probe(X, ids, smap, meta, tgt):
    keep = [j for j, i in enumerate(ids) if i in smap and i in meta]
    X = X[keep]; kid = [ids[j] for j in keep]
    sp = np.array([smap[i] for i in kid]); tr, te = sp == "train", sp == "test"
    y = np.array([int(meta[i]["label"] in (tgt, "both")) for i in kid])
    return float(np.mean([roc_auc_score(y[te], LogisticRegression(
        max_iter=20000, class_weight="balanced", random_state=s).fit(X[tr], y[tr])
        .predict_proba(X[te])[:, 1]) for s in [0, 1, 2]]))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = json.load(open(f"{R}/data/segments/manifest.json"))
    meta = {os.path.basename(e["path"]): e for e in seg}
    smap = json.load(open("results/split_official.json"))

    print("=== control 1: bandwidth — does OPERA-CE improve with AST's mel settings? ===")
    enc = cola_encoder(f"{R}/OPERA/cks/model/encoder-operaCE.ckpt", "encoder.", dev)
    for tag, nm, fx in [("64 mel / 2 kHz (COLA default)", 64, 2000),
                        ("128 mel / 8 kHz (AST-like)", 128, 8000)]:
        X, ids = extract_cola(enc, seg, nm, fx, dev)
        print("  OPERA-CE  %-30s wheeze %.3f  crackle %.3f"
              % (tag, probe(X, ids, smap, meta, "wheeze"), probe(X, ids, smap, meta, "crackle")))

    print("\n=== control 2: depth — AST truncated to its first 6 of 12 layers ===")
    from transformers import ASTModel
    A = np.load("results/ast_feats_clean.npy"); aidx = json.load(open("results/ast_feats_clean_index.json"))
    ids_full = list(aidx)
    print("  AST full (12 layers, 86.2 M)          wheeze %.3f  crackle %.3f"
          % (probe(np.stack([A[aidx[i]] for i in ids_full]), ids_full, smap, meta, "wheeze"),
             probe(np.stack([A[aidx[i]] for i in ids_full]), ids_full, smap, meta, "crackle")))
    m = ASTModel.from_pretrained("/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:6])
    m = m.eval().to(dev)
    print("  truncated to %.1f M params" % (sum(p.numel() for p in m.parameters())/1e6))
    from transformers import AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained("/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ids, out = [], []
    for i in range(0, len(seg), 8):
        ys = [load(os.path.join(R, e["path"])) for e in seg[i:i+8]]
        inp = fe(ys, sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.append(m(**inp).last_hidden_state.mean(1).cpu().numpy())
        ids += [os.path.basename(e["path"]) for e in seg[i:i+8]]
    X6 = np.concatenate(out)
    print("  AST 6 layers                          wheeze %.3f  crackle %.3f"
          % (probe(X6, ids, smap, meta, "wheeze"), probe(X6, ids, smap, meta, "crackle")))


if __name__ == "__main__":
    main()
