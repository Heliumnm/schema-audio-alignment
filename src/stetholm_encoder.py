"""
Probe StethoLM's audio encoder without assembling the full model.

The question 师兄's proposal turns on is whether StethoLM's generated descriptions
carry label information — Qwen2-Audio's do not (AUROC 0.509, see EXPERIMENT_PLAN
§2.96). Generating those descriptions needs the gated MedGemma-4B backbone, an 8.6 GB
download running at ~170 KB/s.

But a necessary condition can be tested now. If the *encoder* cannot separate
wheeze/crackle, the descriptions built on top of it cannot either. The adapter ships
the encoder complete (4.05 M params), so it runs standalone.

Architecture, read off the checkpoint and matched against OPERA's `Encoder`
(`OPERA/src/model/models_cola.py:10`), which it reproduces exactly:

    mel (1ch) -> Conv2d(1, 3, k=3) -> EfficientNet-B0 (include_top=False) -> 1280-d

Mel settings are OPERA's `pre_process_audio_mel_t` (`OPERA/src/util.py:406`):
16 kHz, n_mels=64, f_min=50, f_max=2000, n_fft=1024, hop=512, power_to_db then
min-max normalised, transposed to (time, mel). Getting these wrong produces garbage
features and a *false* negative, which is why they were taken from source rather
than assumed.

    pip install efficientnet_pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple
    python src/stetholm_encoder.py \
        --adapter /mnt/hd/data_heliu/hf_models/stetholm_adapter.pt \
        --manifest .../manifest.json --audio_root .../icbhi_pathology_fidelity \
        --out results/stetholm_feats --batch 32
"""
import os, json, argparse
import numpy as np
import soundfile as sf
import librosa

SR = 16000
N_MELS, F_MIN, F_MAX, NFFT, HOP = 64, 50, 2000, 1024, 512


def mel(audio):
    """OPERA's pre_process_audio_mel_t, reproduced exactly -> (time, n_mels)."""
    S = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, fmin=F_MIN,
                                       fmax=F_MAX, n_fft=NFFT, hop_length=HOP)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
    return S.T.astype(np.float32)


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def build_encoder(adapter_path, device, prefix="audio_encoder."):
    import torch
    from efficientnet_pytorch import EfficientNet

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn1 = torch.nn.Conv2d(1, 3, kernel_size=3)
            self.efficientnet = EfficientNet.from_name(
                "efficientnet-b0", include_top=False, drop_connect_rate=0.1)

        def forward(self, x):                    # x: (B, T, n_mels)
            x = x.unsqueeze(1)
            x = self.cnn1(x)
            x = self.efficientnet(x)
            return x.squeeze(3).squeeze(2)       # (B, 1280)

    sd = torch.load(adapter_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)          # OPERA ships a Lightning checkpoint
    # StethoLM prefixes the encoder "audio_encoder.", OPERA-CE uses "encoder.".
    # Both hold the same 360 tensors, so the two are directly comparable once the
    # prefix is stripped — same architecture, same mel settings, same probe.
    ae = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    assert ae, f"no tensors under prefix {prefix!r}; keys look like {list(sd)[:3]}"
    enc = Encoder()
    missing, unexpected = enc.load_state_dict(ae, strict=False)
    # a silent shape/name mismatch here would look exactly like "the encoder is
    # uninformative", so refuse to continue on anything but a clean load
    assert not missing, f"missing keys in encoder: {list(missing)[:5]}"
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    print(f"loaded {len(ae)} tensors, {sum(v.numel() for v in ae.values())/1e6:.2f} M params")
    return enc.eval().to(device)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prefix", default="audio_encoder.",
                    help="'audio_encoder.' for StethoLM, 'encoder.' for OPERA-CE")
    ap.add_argument("--input_sec", type=float, default=8.0,
                    help="OPERA-CE uses 8 s windows; pad or centre-crop to match")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    enc = build_encoder(args.adapter, dev, args.prefix)
    seg = json.load(open(args.manifest))
    n_frames = int(args.input_sec * SR / HOP) + 1

    def resolve(e):
        p = e.get("path", e["filename"])
        return p if (os.path.isabs(p) or not args.audio_root) else os.path.join(args.audio_root, p)

    ids, embs, n_err, first_err = [], [], 0, None
    for s0 in range(0, len(seg), args.batch):
        ms, keys = [], []
        for e in seg[s0:s0 + args.batch]:
            p = resolve(e)
            try:
                m = mel(load_audio(p))
            except Exception as ex:
                n_err += 1
                first_err = first_err or f"{type(ex).__name__}: {ex} ({p})"
                continue
            if len(m) < n_frames:                      # pad short cycles
                m = np.pad(m, ((0, n_frames - len(m)), (0, 0)))
            else:                                      # centre-crop long ones
                o = (len(m) - n_frames) // 2
                m = m[o:o + n_frames]
            ms.append(m); keys.append(os.path.basename(p))
        if not ms:
            if s0 == 0:
                raise SystemExit(f"First batch read 0 files.\n  {first_err}\n"
                                 f"  Manifest paths are relative — pass --audio_root.")
            continue
        with torch.no_grad():
            v = enc(torch.tensor(np.stack(ms), device=dev)).cpu().numpy()
        embs.append(v); ids.extend(keys)
        if (s0 // args.batch) % 20 == 0:
            print(f"  {s0}/{len(seg)}", flush=True)

    E = np.concatenate(embs).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + ".npy", E)
    json.dump({k: i for i, k in enumerate(ids)}, open(args.out + "_index.json", "w"))
    print(f"\nwrote {E.shape} -> {args.out}.npy ({n_err} read failures)")


if __name__ == "__main__":
    main()
