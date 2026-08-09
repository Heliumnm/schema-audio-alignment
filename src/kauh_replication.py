"""Control 3: does the AST advantage replicate on a different corpus AND a different task?

The ICBHI result measured cycle-level adventitious sounds. KAUH labels are
patient-level disease, which is closer to what OPERA's own benchmark scores — so this
partly answers whether the earlier comparison even touched the task OPERA claims to
win 16 of 19 of.

KAUH is 112 patients x 3 export filters (Bell/Diaphragm/Extended) = 336 files, so
splits are patient-level; a random file split would put the same auscultation on both
sides. Diagnosis strings are case-inconsistent in the dataset and are normalised.

    python src/kauh_replication.py
"""
import os, re, collections
import numpy as np, torch, soundfile as sf, librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SR, HOP = 16000, 512
KAUH = "/mnt/hd/data_heliu/resp_datasets/KAUH/AudioFiles"
AST_PATH = "/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593"
OP = "/mnt/hd/data_heliu/icbhi_pathology_fidelity/OPERA/cks/model"
STE = "/mnt/hd/data_heliu/hf_models/stetholm_adapter.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def parse(fn):
    stem = re.sub(r"\s+", " ", fn[:-4]).strip()
    m = re.match(r"^([BDE])(P\d+)_(.+)$", stem)
    if not m:
        return None
    d = m.group(3).split(",")[0].strip().lower()
    return m.group(2), {"n": "normal", "bron": "bronchiectasis"}.get(d, d)


def load(p):
    y, sr = sf.read(p, dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def mel(a, n_mels=64, fmax=2000):
    S = librosa.feature.melspectrogram(y=a, sr=SR, n_mels=n_mels, fmin=50, fmax=fmax,
                                       n_fft=1024, hop_length=HOP)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
    return S.T.astype(np.float32)


def cola(ckpt, prefix):
    from efficientnet_pytorch import EfficientNet

    class E(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.cnn1 = torch.nn.Conv2d(1, 3, 3)
            s.efficientnet = EfficientNet.from_name(
                "efficientnet-b0", include_top=False, drop_connect_rate=0.1)

        def forward(s, x):
            return s.efficientnet(s.cnn1(x.unsqueeze(1))).squeeze(3).squeeze(2)

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    w = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    m = E()
    miss, unexp = m.load_state_dict(w, strict=False)
    assert not miss and not unexp, (list(miss)[:3], list(unexp)[:3])
    return m.eval().to(dev)


def main():
    files = sorted(f for f in os.listdir(KAUH)
                   if f.endswith(".wav") and not f.startswith("._"))
    recs = [(f, *parse(f)) for f in files if parse(f)]
    print(f"{len(recs)} files, {len(set(r[1] for r in recs))} patients")
    print("diagnoses:", dict(collections.Counter(r[2] for r in recs).most_common(6)))

    feats, nf = {}, int(8.0 * SR / HOP) + 1
    for name, ck, pref in [("OPERA-CE", f"{OP}/encoder-operaCE.ckpt", "encoder."),
                           ("StethoLM", STE, "audio_encoder.")]:
        enc, out = cola(ck, pref), []
        for i in range(0, len(recs), 8):
            ms = []
            for f, _, _ in recs[i:i + 8]:
                m = mel(load(os.path.join(KAUH, f)))
                m = (np.pad(m, ((0, nf - len(m)), (0, 0))) if len(m) < nf
                     else m[(len(m) - nf) // 2:(len(m) - nf) // 2 + nf])
                ms.append(m)
            with torch.no_grad():
                out.append(enc(torch.tensor(np.stack(ms), device=dev)).cpu().numpy())
        feats[name] = np.concatenate(out)
        del enc
        torch.cuda.empty_cache()

    from transformers import ASTModel, AutoFeatureExtractor
    fe = AutoFeatureExtractor.from_pretrained(AST_PATH)
    for tag, nlayer in [("AST", 12), ("AST-6L", 6)]:
        m = ASTModel.from_pretrained(AST_PATH)
        if nlayer < 12:
            m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:nlayer])
        m, out = m.eval().to(dev), []
        for i in range(0, len(recs), 8):
            ys = [load(os.path.join(KAUH, f)) for f, _, _ in recs[i:i + 8]]
            inp = fe(ys, sampling_rate=SR, return_tensors="pt").to(dev)
            with torch.no_grad():
                out.append(m(**inp).last_hidden_state.mean(1).cpu().numpy())
        feats[tag] = np.concatenate(out)
        del m
        torch.cuda.empty_cache()

    pid = np.array([r[1] for r in recs])
    diag = np.array([r[2] for r in recs])
    keys = ["AST", "AST-6L", "OPERA-CE", "StethoLM"]
    print("\npatient-level 70/30, 5 repeats, disease vs normal\n")
    print("%-14s %8s %8s %9s %9s" % ("task", *keys))
    for dname in ["asthma", "copd", "heart failure", "pneumonia"]:
        sel = np.isin(diag, [dname, "normal"])
        if sel.sum() < 40:
            continue
        y, P = (diag[sel] == dname).astype(int), pid[sel]
        row = []
        for k in keys:
            X, aucs = feats[k][sel], []
            for s in range(5):
                rng = np.random.RandomState(s)
                ps = sorted(set(P)); rng.shuffle(ps)
                te = np.isin(P, list(ps[:max(2, int(len(ps) * 0.3))]))
                tr = ~te
                if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                    continue
                c = LogisticRegression(max_iter=20000, class_weight="balanced").fit(X[tr], y[tr])
                aucs.append(roc_auc_score(y[te], c.predict_proba(X[te])[:, 1]))
            row.append(np.mean(aucs) if aucs else float("nan"))
        print("%-14s %8.3f %8.3f %9.3f %9.3f" % (dname, *row),
              f"  (n={sel.sum()}, pos={y.mean():.2f})")


if __name__ == "__main__":
    main()
