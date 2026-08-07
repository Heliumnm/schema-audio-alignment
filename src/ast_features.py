"""
Extract AST (Audio Spectrogram Transformer) embeddings for the ICBHI cycles.

This is the audio tower EXPERIMENT_PLAN §3 specifies for the main runs. The first
sweep used OPERA-CT, whose COLA-family pretraining is the same family as StethoLM's
encoder — aligning it to text derived from that family maximises circularity, which
is exactly the effect that dominated the first results. AST is ImageNet/AudioSet
pretrained and shares no lineage with the text-generating models, so it isolates
whether the negative result is about contrastive alignment or about tower choice.

Output matches the OPERA feature format so contrastive_align.py needs no changes:
  <out>.npy    float32 (N, D)
  <out>_index.json   {segment_id: row}

    python src/ast_features.py \
        --manifest .../manifest.json --audio_root .../icbhi_pathology_fidelity \
        --model /path/to/ast-finetuned-audioset-10-10-0.4593 \
        --out results/ast_feats_clean --batch 16 --device cuda
"""
import os, json, argparse
import numpy as np
import soundfile as sf
import librosa

SR = 16000


def load_audio(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y


def main():
    import torch
    from transformers import AutoFeatureExtractor, ASTModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio_root", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default="path")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--pool", default="mean", choices=["mean", "cls"])
    args = ap.parse_args()

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    mdl = ASTModel.from_pretrained(args.model).eval().to(dev)
    print(f"AST on {dev}, hidden={mdl.config.hidden_size}, pool={args.pool}")

    seg = json.load(open(args.manifest))

    def resolve(e):
        p = e.get(args.key, e["path"])
        return p if (os.path.isabs(p) or not args.audio_root) else os.path.join(args.audio_root, p)

    ids, embs, n_err, first_err = [], [], 0, None
    for s0 in range(0, len(seg), args.batch):
        chunk = seg[s0:s0 + args.batch]
        ys, keys = [], []
        for e in chunk:
            p = resolve(e)
            try:
                ys.append(load_audio(p)); keys.append(os.path.basename(p))
            except Exception as ex:
                n_err += 1
                if first_err is None:
                    first_err = f"{type(ex).__name__}: {ex}  (path={p})"
        if not ys:
            # a wrong --audio_root fails on every file; say so instead of writing an
            # empty array that only breaks much later
            if s0 == 0:
                raise SystemExit(f"First batch read 0 files.\n  {first_err}\n"
                                 f"  Manifest paths are relative — pass --audio_root.")
            continue
        inp = fe(ys, sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = mdl(**inp).last_hidden_state          # (B, T, D)
        v = h.mean(1) if args.pool == "mean" else h[:, 0]
        embs.append(v.float().cpu().numpy()); ids.extend(keys)
        if (s0 // args.batch) % 20 == 0:
            print(f"  {s0}/{len(seg)}", flush=True)

    E = np.concatenate(embs).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out + ".npy", E)
    json.dump({k: i for i, k in enumerate(ids)}, open(args.out + "_index.json", "w"))
    print(f"\nwrote {E.shape} -> {args.out}.npy  ({n_err} read failures)")
    if n_err:
        print(f"  first failure: {first_err}")


if __name__ == "__main__":
    main()
