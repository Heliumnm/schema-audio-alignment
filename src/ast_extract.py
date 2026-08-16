"""Extract the frozen AST layer-6 representation, exactly as AUDIO_PREPROCESSING_SPEC says.

Run with `--check` first: it builds a **stratified** 100-recording set rather than a random
one and verifies every property the spec promises, including bit-identical repeats. Full
extraction refuses to start unless the check passed.

    python src/ast_extract.py --check
    python src/ast_extract.py --full
"""
import os, json, argparse, hashlib
import numpy as np
import pandas as pd

SPEC_VERSION = "v1-2026-08-17"
SR, WIN_S = 16000, 10.24
PATCH, STRIDE, N_FREQ, N_TIME, MAX_FRAMES = 16, 10, 12, 101, 1024
FPS = 100.0
LAYERS = 6


def window_starts(dur_s):
    """Uniformly spread windows that cover the whole recording; window 0 begins at 0 and
    the last ends exactly at the end."""
    n = max(1, int(np.ceil(dur_s / WIN_S)))
    if n == 1:
        return [0.0], n
    step = (dur_s - WIN_S) / (n - 1)
    return [i * step for i in range(n)], n


def n_valid_patches(n_valid_frames):
    """Time patch t spans mel frames [10t, 10t+16). Keep only fully covered patches, so no
    averaged patch contains padding. Never duration/101."""
    return max(0, int(np.floor((min(n_valid_frames, MAX_FRAMES) - PATCH) / STRIDE)) + 1)


def load_window(path, start_s):
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y[:, 0]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)   # no peak normalisation
    a = int(round(start_s * SR))
    seg = y[a:a + int(WIN_S * SR)]
    n_real = len(seg)
    if n_real < int(WIN_S * SR):                            # right-pad only
        seg = np.concatenate([seg, np.zeros(int(WIN_S * SR) - n_real, np.float32)])
    return seg, n_real / SR


def embed(model, fe, torch, path, dur_s, dev):
    starts, n_win = window_starts(dur_s)
    embs, meta = [], []
    for s in starts:
        seg, real_s = load_window(path, s)
        nvf = int(round(real_s * FPS))
        npv = n_valid_patches(nvf)
        assert npv > 0, f"{path} window at {s:.2f}s yields no valid patch"
        inp = fe([seg], sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = model(**inp).last_hidden_state[:, 2:, :]     # drop the two special tokens
        g = h.reshape(1, N_FREQ, N_TIME, -1)[0, :, :npv, :]  # valid time patches only
        embs.append(g.reshape(-1, g.shape[-1]).mean(0).float().cpu().numpy())
        meta.append({"start_s": float(s), "n_valid_frames": nvf, "n_valid_patches": npv})
    return np.mean(embs, 0), meta, n_win


def build_check_set(C, F, n=100, seed=0):
    """Stratified, not random: the spec names the cells that must appear."""
    d = C.merge(F, on="participant_identifier")
    d["clipped"] = d.clip_frac > 0
    d["n_win"] = np.ceil(d.duration_s / WIN_S).astype(int)
    rs = np.random.RandomState(seed)
    picks = []

    def take(mask, k, why):
        sub = d[mask & ~d.participant_identifier.isin([p for p, _ in picks])]
        if len(sub) == 0:
            return
        idx = rs.choice(len(sub), min(k, len(sub)), replace=False)
        for pid in sub.iloc[idx].participant_identifier:
            picks.append((pid, why))

    take(d.duration_s <= d.duration_s.quantile(0.001), 8, "shortest")
    take((d.duration_s > 9.5) & (d.duration_s <= WIN_S), 8, "just_below_10.24")
    take((d.duration_s > WIN_S) & (d.duration_s < 11.5), 8, "just_above_10.24")
    take(d.duration_s >= d.duration_s.quantile(0.9999), 6, "longest")
    take(d.n_win == 1, 10, "single_window")
    take(d.n_win >= 3, 10, "multi_window_3plus")
    take(d.clipped, 10, "clipped")
    take(~d.clipped, 10, "unclipped")
    take(d.recruitment_source == "REACT", 6, "REACT")
    take(d.recruitment_source == "Test and Trace", 6, "TestAndTrace")
    take(d.splits == "test", 6, "standard_test")
    take(d.in_matched_rebalanced_test == True, 6, "matched")          # noqa: E712
    take(d.in_matched_rebalanced_long_test == True, 6, "matched_long")  # noqa: E712
    while len(picks) < n:
        take(d.participant_identifier.notna(), n - len(picks), "filler")
        break
    return pd.DataFrame(picks, columns=["participant_identifier", "cell"]).head(n)


def main():
    import torch
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--feats", default="results/artefact_features.csv")
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default="results/ast_embeddings.npz")
    ap.add_argument("--check_out", default="results/ast_preflight.json")
    args = ap.parse_args()

    C = pd.read_csv(args.cohort)
    F = pd.read_csv(args.feats)
    p = pd.read_csv(os.path.join(args.data, "participant_metadata.csv"), low_memory=False)
    C = C.merge(p[["participant_identifier", "recruitment_source"]],
                on="participant_identifier")
    path_of = {r.participant_identifier: os.path.join(args.audio_root, r.cough_file_name)
               for r in C.itertuples()}
    dur_of = dict(zip(F.participant_identifier, F.duration_s))

    ck = hashlib.sha256()
    for f in sorted(os.listdir(args.model)):
        if f.endswith((".bin", ".safetensors")):
            ck.update(open(os.path.join(args.model, f), "rb").read(1 << 20))
    ckpt_hash = ck.hexdigest()[:16]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:LAYERS])
    m = m.eval().to(dev)
    print(f"spec {SPEC_VERSION}  AST first {LAYERS} layers  ckpt {ckpt_hash}")

    if args.check:
        S = build_check_set(C, F)
        print(f"\nstratified check set: {len(S)} recordings")
        print(S.cell.value_counts().to_string())
        rep, fails = [], []
        for r in S.itertuples():
            pid = r.participant_identifier
            dur = dur_of[pid]
            e1, meta, nw = embed(m, fe, torch, path_of[pid], dur, dev)
            e2, _, _ = embed(m, fe, torch, path_of[pid], dur, dev)
            starts = [x["start_s"] for x in meta]
            covers_head = starts[0] == 0.0
            covers_tail = abs((starts[-1] + WIN_S) - max(dur, WIN_S)) < 1e-6
            exp_win = max(1, int(np.ceil(dur / WIN_S)))
            row = {"pid": pid, "cell": r.cell, "duration_s": float(dur), "n_win": nw,
                   "n_win_expected": exp_win,
                   "valid_patches": [x["n_valid_patches"] for x in meta],
                   "covers_head": bool(covers_head), "covers_tail": bool(covers_tail),
                   "bitwise_identical": bool(np.array_equal(e1, e2)),
                   "emb_finite": bool(np.isfinite(e1).all()),
                   "emb_nonzero": bool(np.abs(e1).sum() > 0)}
            for k, why in [("n_win_expected", nw != exp_win), ("covers_head", not covers_head),
                           ("covers_tail", not covers_tail),
                           ("bitwise_identical", not row["bitwise_identical"]),
                           ("emb_finite", not row["emb_finite"]),
                           ("emb_nonzero", not row["emb_nonzero"])]:
                if why:
                    fails.append((pid, k))
            # the mask rule, recomputed independently of embed()
            for x in meta:
                if x["n_valid_patches"] != n_valid_patches(x["n_valid_frames"]):
                    fails.append((pid, "mask_rule"))
            rep.append(row)
        R = pd.DataFrame(rep)
        print(f"\nwindow counts match ceil(duration/10.24): "
              f"{int((R.n_win == R.n_win_expected).sum())}/{len(R)}")
        print(f"head covered {int(R.covers_head.sum())}/{len(R)}   "
              f"tail covered {int(R.covers_tail.sum())}/{len(R)}")
        print(f"bit-identical on repeat {int(R.bitwise_identical.sum())}/{len(R)}   "
              f"finite {int(R.emb_finite.sum())}/{len(R)}   "
              f"non-zero {int(R.emb_nonzero.sum())}/{len(R)}")
        pv = [v for vs in R.valid_patches for v in vs]
        print(f"valid patches per window: min {min(pv)} median {int(np.median(pv))} "
              f"max {max(pv)} of {N_TIME}")
        out = {"spec": SPEC_VERSION, "ckpt": ckpt_hash, "n": len(R),
               "failures": [{"pid": a, "check": b} for a, b in fails],
               "passed": not fails}
        json.dump(out, open(args.check_out, "w"), indent=1)
        R.to_csv(args.check_out.replace(".json", ".csv"), index=False)
        print(f"\nPREFLIGHT {'PASSED' if not fails else 'FAILED: ' + str(fails[:5])}")
        print(f"wrote {args.check_out}")
        return

    if args.full:
        pf = json.load(open(args.check_out))
        assert pf.get("passed") and pf["spec"] == SPEC_VERSION and pf["ckpt"] == ckpt_hash, \
            "preflight did not pass under this spec and checkpoint; refusing to extract"
        pids, E, W, V = [], [], [], []
        for i, pid in enumerate(C.participant_identifier):
            e, meta, nw = embed(m, fe, torch, path_of[pid], dur_of[pid], dev)
            pids.append(pid); E.append(e); W.append(nw)
            V.append(sum(x["n_valid_patches"] for x in meta))
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(C)}", flush=True)
        np.savez_compressed(args.out, participants=np.array(pids),
                            embeddings=np.stack(E).astype(np.float32),
                            n_windows=np.array(W), n_valid_patches=np.array(V),
                            spec=SPEC_VERSION, ckpt=ckpt_hash)
        print(f"\nwrote {args.out}  {len(pids)} participants")


if __name__ == "__main__":
    main()
