"""Anti-shortcut controls for the synthetic grounding result.

The Step-2 "pass" is under suspicion of being coordinate arithmetic rather than
grounding. The query was built as

    q = [onset + dur/2,  dur,  f0/1000]

while the target patch is computed as `t_idx = f(onset + dur/2)` and
`f_idx = g(f0)` — both **deterministic functions of the query itself**. If that is
what the scorer learned, the audio was never needed, and every number in the pilot
follows: shuffling the query breaks the arithmetic, and swapping the query to the
distractor's coordinates routes to the distractor's patch, all without listening.

The `energy_tonality` baseline that should have caught this was itself broken — it
scored `np.linalg.norm(P, axis=-1)`, the magnitude of AST's embeddings, which never
touches the spectrogram. Its 0.099 said nothing about task difficulty.

Three controls, in order of how decisive they are:

  coordinate_only   no audio whatsoever — score = −|t − q_t| − |f − q_f|.
                    If this scores near the learned model, the answer is in the query.
  audio_shuffled    learned scorer, but each sample sees ANOTHER sample's patch grid.
                    Position structure intact, acoustic content destroyed. High
                    performance here means the audio is unused.
  dsp_band_energy   a real spectral heuristic: log-mel energy inside the queried
                    band, maximised over time. The baseline the pilot claimed to have.

    python src/grounding_controls.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 900
"""
import os, json, argparse
import numpy as np

SR, PATCH, STRIDE = 16000, 16, 10
N_FREQ, N_TIME, N_MELS, MAX_FRAMES = 12, 101, 128, 1024


def main():
    import torch, torch.nn as nn
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from grounding_pilot import inject_wheeze, target_patch

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model", default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=900); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/grounding_controls.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    norm = [e for e in seg if e["label"] == "normal"
            and os.path.basename(e["path"]) in smap and e.get("duration", 0) >= 1.2]
    rng.shuffle(norm); norm = norm[:args.n]

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)

    F0_TR, F0_TE = [(350, 700), (700, 1100)], [(1100, 1500)]
    rows = []
    for e in norm:
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        dur, sid = len(y) / SR, os.path.basename(p)
        is_test = smap[sid] == "test"
        bands = F0_TE if is_test else F0_TR
        d = rng.uniform(0.25, min(0.6, max(0.3, dur * 0.35)))
        if dur < 2 * d + 0.3: continue
        on_a = rng.uniform(0, dur / 2 - d) if dur / 2 - d > 0 else 0.0
        on_b = rng.uniform(dur / 2 + 0.05, max(dur / 2 + 0.06, dur - d))
        lo, hi = bands[rng.randint(len(bands))]
        f_a = rng.uniform(lo, hi); f_b = f_a * rng.uniform(1.9, 2.4)
        yi = inject_wheeze(y, on_a, d, f_a)
        if yi is None: continue
        yi = inject_wheeze(yi, on_b, d, f_b)
        if yi is None: continue
        nvf = min(int(len(yi) / SR * 100), MAX_FRAMES)
        pick_b = bool(rng.rand() < 0.5)
        on_q, f_q = (on_b, f_b) if pick_b else (on_a, f_a)
        on_d, f_d = (on_a, f_a) if pick_b else (on_b, f_b)
        t_i, f_i, occ = target_patch(on_q, d, f_q, nvf)
        t_d, f_di, _ = target_patch(on_d, d, f_d, nvf)
        if t_i == t_d and f_i == f_di: continue
        rows.append({"y": yi, "test": is_test, "t": t_i, "f": f_i, "occ": occ,
                     "f0": f_q, "on": on_q, "dur": d, "t_d": t_d, "f_d": f_di})
    print(f"{len(rows)} clips | train {sum(1 for r in rows if not r['test'])} "
          f"test {sum(1 for r in rows if r['test'])}")

    P, FB = [], []
    for i in range(0, len(rows), 8):
        inp = fe([r["y"] for r in rows[i:i+8]], sampling_rate=SR, return_tensors="pt")
        FB.append(inp["input_values"].numpy())
        with torch.no_grad():
            h = m(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state[:, 2:, :]
        P.append(h.reshape(len(h), N_FREQ, N_TIME, -1).cpu().numpy().astype(np.float32))
    P = np.concatenate(P); FB = np.concatenate(FB)          # FB: (N, frames, mels)

    occ = np.stack([r["occ"] for r in rows])
    tt = np.array([r["t"] for r in rows]); ff = np.array([r["f"] for r in rows])
    tdist = np.array([r["t_d"] for r in rows]); fdist = np.array([r["f_d"] for r in rows])
    te = np.array([r["test"] for r in rows])
    q = np.stack([[r["on"] + r["dur"] / 2, r["dur"], r["f0"] / 1000.0]
                  for r in rows]).astype(np.float32)

    def ev(score):
        s = np.where(occ[te][:, None, :] > 0, score, -1e9)
        pf, pt = np.unravel_index(s.reshape(len(s), -1).argmax(1), (N_FREQ, N_TIME))
        return {"hit1pm_t": float(np.mean(np.abs(pt - tt[te]) <= 1)),
                "freq_acc": float(np.mean(pf == ff[te])),
                "point2d": float(np.mean((pt == tt[te]) & (pf == ff[te]))),
                "distractor": float(np.mean((np.abs(pt - tdist[te]) <= 1) & (pf == fdist[te])))}

    res = {}

    # ---- 1. coordinate-only: no audio at all
    ti = np.arange(N_TIME)[None, None, :]; fi = np.arange(N_FREQ)[None, :, None]
    qt = np.array([target_patch(r["on"], r["dur"], r["f0"], 1000)[0] for r in rows])
    qf = np.array([target_patch(r["on"], r["dur"], r["f0"], 1000)[1] for r in rows])
    res["coordinate_only"] = ev(-(np.abs(ti - qt[te][:, None, None]) +
                                  3.0 * np.abs(fi - qf[te][:, None, None])).astype(np.float32))

    # ---- 2. proper DSP: log-mel energy in the queried band, over time
    mel_max = 2595 * np.log10(1 + (SR / 2) / 700.0)
    dsp = np.zeros((int(te.sum()), N_FREQ, N_TIME), np.float32)
    for n, idx in enumerate(np.where(te)[0]):
        fb = FB[idx]                                          # (frames, mels)
        for t in range(N_TIME):
            w = fb[t * STRIDE:t * STRIDE + PATCH]
            if not len(w): continue
            for fr in range(N_FREQ):
                b = w[:, fr * STRIDE:fr * STRIDE + PATCH]
                dsp[n, fr, t] = b.mean() if b.size else -1e9
        lo = 2595 * np.log10(1 + rows[idx]["f0"] / 700.0) / mel_max * N_MELS
        fr_q = int(np.clip(round((lo - PATCH / 2) / STRIDE), 0, N_FREQ - 1))
        # the query names a band, so the heuristic is allowed to use it
        boost = np.zeros((N_FREQ, 1), np.float32); boost[fr_q] = 3.0
        dsp[n] += boost
    res["dsp_band_energy"] = ev(dsp)

    # ---- 3. learned scorers, with and without real audio content
    for name, scramble in [("typed_intact", False), ("audio_shuffled", True)]:
        runs = []
        for sd in args.seeds:
            torch.manual_seed(sd); rs = np.random.RandomState(200 + sd)
            Pu = P.copy()
            if scramble:
                # each sample sees another sample's patch grid: positional structure
                # preserved, acoustic content destroyed
                Pu = Pu[rs.permutation(len(Pu))]
            enc = nn.Sequential(nn.Linear(3, 128), nn.ReLU(),
                                nn.Linear(128, P.shape[-1])).to(dev)
            opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
            Pt = torch.tensor(Pu, device=dev); Qt = torch.tensor(q, device=dev)
            Ot = torch.tensor(occ, device=dev)
            tgt = torch.tensor(ff * N_TIME + tt, device=dev)
            tr_idx = np.where(~te)[0]
            for _ in range(args.epochs):
                for s0 in range(0, len(tr_idx), 32):
                    b = torch.tensor(tr_idx[s0:s0+32], device=dev)
                    sc = (Pt[b] * enc(Qt[b])[:, None, None, :]).sum(-1)
                    sc = sc.masked_fill(Ot[b][:, None, :] <= 0, -1e9)
                    loss = nn.functional.cross_entropy(sc.flatten(1), tgt[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            enc.eval()
            with torch.no_grad():
                idx = torch.tensor(np.where(te)[0], device=dev)
                sc = (Pt[idx] * enc(Qt[idx])[:, None, None, :]).sum(-1).cpu().numpy()
            runs.append(ev(sc))
        res[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}

    print(f"\ntest n={int(te.sum())}\n")
    print("%-18s %9s %9s %9s %11s" % ("scorer", "Hit@±1", "freq_acc", "2D", "distractor"))
    for k in ["typed_intact", "audio_shuffled", "coordinate_only", "dsp_band_energy"]:
        r = res[k]
        print("%-18s %9.3f %9.3f %9.3f %11.3f" % (k, r["hit1pm_t"], r["freq_acc"],
                                                  r["point2d"], r["distractor"]))
    co, ai, sh = (res["coordinate_only"]["hit1pm_t"], res["typed_intact"]["hit1pm_t"],
                  res["audio_shuffled"]["hit1pm_t"])
    print(f"\ncoordinate_only reaches {co:.3f} vs the learned model's {ai:.3f}")
    print(f"audio_shuffled reaches {sh:.3f} — content destroyed, positions kept")
    if co > 0.8 * ai or sh > 0.8 * ai:
        print("\nSHORTCUT CONFIRMED: the answer is recoverable without the audio, so the "
              "Step-2 pass measured coordinate routing, not grounding.")
    else:
        print("\nNo shortcut: the learned model needs the acoustic content.")
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
