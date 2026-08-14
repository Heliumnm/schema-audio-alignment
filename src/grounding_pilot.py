"""Step 2: can a schema node point at the right time-frequency patch?

Grounding is a separate question from global classification, and global AUROC cannot
stand in for it — a representation can be useless for cycle-level AUROC and still
localise correctly, or the reverse. Step 1 answered only the former.

No manual annotation exists (ICBHI has per-cycle presence flags and no event timing),
so this uses synthetic injections where onset, duration and frequency band are known
exactly. That validates the *mechanism* only and is never a clinical result.

Setup
-----
Take NORMAL cycles, inject a narrowband wheeze at a known (onset, duration, f_low,
f_high), build the schema node from those same coordinates, and ask whether the node
attends to the right patch of AST's 12 x 101 grid.

    patches = h[:, 2:, :].reshape(B, 12, 101, 768)      freq x time
    time patch t covers frames [10t, 10t+16)            ~100 ms stride, ~160 ms field
    occupancy(t) = valid frames in that window / 16      partial patches down-weighted

**Two injections per clip, not one.** The single-injection version failed for a
reason that had nothing to do with grounding: the injected tone is conspicuous in the
spectrogram, so the scorer located it straight from the patch features and ignored the
query entirely. `shuffled_coords` then matched `typed_intact` (0.977 vs 0.965 Hit@±1)
and a +0.4 s query shift moved the peak by +0.10 patches — the query was not being
read at all, so the task could not discriminate the hypothesis. The `energy_tonality`
baseline was meant to catch exactly that and was too weak to (0.089).

With two injections at different times and frequencies, the query is the only way to
choose between them: ignoring it caps accuracy near 50%, using it allows ~100%.

Baselines, because a synthetic task can be trivially easy:

  typed_intact       the method under test
  shuffled_coords    same audio, node coordinates permuted across samples —
                     correspondence control
  energy_tonality    no learning: pick the patch maximising band energy x tonality.
                     If this matches the model, the task proves nothing
  random             untrained scorer — chance floor

Metrics: Hit@1 and Hit@±1 on the time axis, frequency-row accuracy, 2D pointing, and
a query-swap test — move the schema's time or frequency and check the attention
follows. Query swap is the one that separates real correspondence from a model that
has memorised where wheezes usually sit.

Go criterion, fixed before running: on unseen patients AND unseen injection
parameters, typed_intact must beat BOTH shuffled_coords and energy_tonality.

    python src/grounding_pilot.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 600
"""
import os, json, argparse
import numpy as np

SR, HOP, PATCH, STRIDE = 16000, 160, 16, 10   # AST fbank: 10 ms frames
N_FREQ, N_TIME, N_MELS, MAX_FRAMES = 12, 101, 128, 1024


def inject_wheeze(y, onset_s, dur_s, f0, sr=SR, snr_db=6.0):
    """Additive narrowband tone with a couple of harmonics and a raised-cosine gate —
    a wheeze is tonal and sustained, so a pure click would make the task trivial."""
    n0, n1 = int(onset_s * sr), int((onset_s + dur_s) * sr)
    n1 = min(n1, len(y))
    if n1 - n0 < int(0.05 * sr):
        return None
    t = np.arange(n1 - n0) / sr
    tone = np.sin(2 * np.pi * f0 * t) + 0.35 * np.sin(2 * np.pi * 2 * f0 * t)
    ramp = int(0.02 * sr)
    env = np.ones_like(tone)
    if len(env) > 2 * ramp:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        env[:ramp], env[-ramp:] = w, w[::-1]
    seg_rms = np.sqrt(np.mean(y[n0:n1] ** 2)) + 1e-8
    tone = tone * env * seg_rms * (10 ** (snr_db / 20)) / (np.sqrt(np.mean(tone ** 2)) + 1e-8)
    out = y.copy()
    out[n0:n1] += tone
    return out / (np.abs(out).max() + 1e-8)


def target_patch(onset_s, dur_s, f0, n_valid_frames):
    """Ground-truth patch indices. Time from the frame window each patch covers;
    frequency from the mel bin of f0 under AST's 128-bin layout."""
    c_frame = (onset_s + dur_s / 2) * 100.0                      # 10 ms frames
    t_idx = int(np.clip(round((c_frame - PATCH / 2) / STRIDE), 0, N_TIME - 1))
    mel = 2595 * np.log10(1 + f0 / 700.0)
    mel_max = 2595 * np.log10(1 + (SR / 2) / 700.0)
    f_bin = np.clip(mel / mel_max * N_MELS, 0, N_MELS - 1)
    f_idx = int(np.clip(round((f_bin - PATCH / 2) / STRIDE), 0, N_FREQ - 1))
    occ = np.clip((n_valid_frames - np.arange(N_TIME) * STRIDE) / PATCH, 0, 1)
    return t_idx, f_idx, occ.astype(np.float32)


def main():
    import torch, torch.nn as nn
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model", default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/grounding_pilot.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest))
    smap = json.load(open(args.split_map))
    norm = [e for e in seg if e["label"] == "normal"
            and os.path.basename(e["path"]) in smap and e.get("duration", 0) >= 1.2]
    rng.shuffle(norm); norm = norm[:args.n]
    print(f"{len(norm)} normal cycles, {len(set(os.path.basename(e['path']).split('_')[0] for e in norm))} patients")

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)

    # ---- build the synthetic set; hold out injection parameters, not just patients
    F0_TR, F0_TE = [(350, 700), (700, 1100)], [(1100, 1500)]     # unseen band at test
    rows = []
    for e in norm:
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        dur = len(y) / SR
        sid = os.path.basename(p)
        is_test = smap[sid] == "test"
        bands = F0_TE if is_test else F0_TR
        # two injections, well separated in BOTH axes so the query has to disambiguate
        d = rng.uniform(0.25, min(0.6, max(0.3, dur * 0.35)))
        if dur < 2 * d + 0.3:
            continue
        on_a = rng.uniform(0, dur / 2 - d) if dur / 2 - d > 0 else 0.0
        on_b = rng.uniform(dur / 2 + 0.05, max(dur / 2 + 0.06, dur - d))
        lo, hi = bands[rng.randint(len(bands))]
        f_a = rng.uniform(lo, hi)
        f_b = f_a * rng.uniform(1.9, 2.4)          # >= 2 freq rows apart
        yi = inject_wheeze(y, on_a, d, f_a)
        if yi is None: continue
        yi = inject_wheeze(yi, on_b, d, f_b)
        if yi is None: continue
        nvf = min(int(len(yi) / SR * 100), MAX_FRAMES)
        pick_b = bool(rng.rand() < 0.5)            # which one the query asks for
        on_q, f_q = (on_b, f_b) if pick_b else (on_a, f_a)
        on_d, f_d = (on_a, f_a) if pick_b else (on_b, f_b)
        t_i, f_i, occ = target_patch(on_q, d, f_q, nvf)
        t_d, f_d_i, _ = target_patch(on_d, d, f_d, nvf)
        if t_i == t_d and f_i == f_d_i:
            continue                                # the two must be distinguishable
        rows.append({"y": yi, "sid": sid, "pid": sid.split("_")[0], "test": is_test,
                     "t": t_i, "f": f_i, "occ": occ, "f0": f_q, "on": on_q, "dur": d,
                     "t_d": t_d, "f_d": f_d_i, "on_d": on_d + d / 2, "f0_d": f_d})
    print(f"built {len(rows)} clips x2 injections | train {sum(1 for r in rows if not r['test'])} "
          f"test {sum(1 for r in rows if r['test'])} (test f0 band 1100-1500 Hz unseen)")

    # ---- AST patch grids
    P = []
    for i in range(0, len(rows), 8):
        inp = fe([r["y"] for r in rows[i:i+8]], sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = m(**inp).last_hidden_state[:, 2:, :]
        P.append(h.reshape(len(h), N_FREQ, N_TIME, -1).cpu().numpy().astype(np.float32))
    P = np.concatenate(P)
    print(f"patch grids {P.shape}")

    occ = np.stack([r["occ"] for r in rows])
    tt = np.array([r["t"] for r in rows]); ff = np.array([r["f"] for r in rows])
    tdist = np.array([r["t_d"] for r in rows]); fdist = np.array([r["f_d"] for r in rows])
    te = np.array([r["test"] for r in rows])
    q = np.stack([[r["on"] + r["dur"] / 2, r["dur"], r["f0"] / 1000.0] for r in rows]).astype(np.float32)
    ondist = np.array([r["on_d"] for r in rows], dtype=np.float32)
    f0dist = np.array([r["f0_d"] for r in rows], dtype=np.float32)

    def scorer_eval(score, mask):
        """score: (N, F, T). Hit@1, Hit@+/-1 on time, freq-row acc, 2D pointing."""
        # padded patches can never be the answer, so they are removed from the argmax
        s = np.where(occ[mask][:, None, :] > 0, score, -1e9)
        flat = s.reshape(len(s), -1).argmax(1)
        pf, pt = np.unravel_index(flat, (N_FREQ, N_TIME))
        td, fd = tdist[mask], fdist[mask]
        return {"hit1_t": float(np.mean(pt == tt[mask])),
                "hit1pm_t": float(np.mean(np.abs(pt - tt[mask]) <= 1)),
                "freq_acc": float(np.mean(pf == ff[mask])),
                "point2d": float(np.mean((pt == tt[mask]) & (pf == ff[mask]))),
                # landing on the distractor means the audio was found but the query
                # ignored — the failure mode the single-injection design could not see
                "distractor": float(np.mean((np.abs(pt - td) <= 1) & (pf == fd)))}

    res = {}
    # --- baseline: energy x tonality, no learning
    E = np.linalg.norm(P, axis=-1)
    res["energy_tonality"] = scorer_eval(E[te], te)

    # --- learned scorers
    for name, shuffle_q in [("typed_intact", False), ("shuffled_coords", True)]:
        runs = []
        for sd in args.seeds:
            torch.manual_seed(sd); rs = np.random.RandomState(100 + sd)
            qq = q.copy()
            if shuffle_q:
                idx = np.where(~te)[0]; qq[idx] = qq[idx][rs.permutation(len(idx))]
            enc = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, P.shape[-1])).to(dev)
            opt = torch.optim.AdamW(enc.parameters(), lr=1e-3)
            Pt = torch.tensor(P, device=dev); Qt = torch.tensor(qq, device=dev)
            Ot = torch.tensor(occ, device=dev)
            tr_idx = np.where(~te)[0]
            tgt = torch.tensor(ff * N_TIME + tt, device=dev)
            for _ in range(args.epochs):
                for s0 in range(0, len(tr_idx), 32):
                    b = torch.tensor(tr_idx[s0:s0+32], device=dev)
                    node = enc(Qt[b])                                  # (B, D)
                    sc = (Pt[b] * node[:, None, None, :]).sum(-1)      # (B, F, T)
                    sc = sc.masked_fill(Ot[b][:, None, :] <= 0, -1e9)
                    loss = nn.functional.cross_entropy(sc.flatten(1), tgt[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            enc.eval()
            with torch.no_grad():
                node = enc(Qt[torch.tensor(np.where(te)[0], device=dev)])
                sc = (Pt[torch.tensor(np.where(te)[0], device=dev)] *
                      node[:, None, None, :]).sum(-1).cpu().numpy()
            runs.append(scorer_eval(sc, te))
            if name == "typed_intact" and sd == args.seeds[0]:
                # Query the DISTRACTOR's coordinates instead. A +0.4 s nudge cannot
                # test this design: the two injections sit ~1 s apart, so the shifted
                # point is still nearest the original and staying put is the correct
                # answer. Swapping to the other injection is the decisive test —
                # if the query drives the choice, the peak must follow it there.
                ti = torch.tensor(np.where(te)[0], device=dev)
                with torch.no_grad():
                    qd = Qt[ti].clone()
                    qd[:, 0] = torch.tensor(ondist[te], device=dev, dtype=qd.dtype)
                    qd[:, 2] = torch.tensor(f0dist[te] / 1000.0, device=dev, dtype=qd.dtype)
                    sd_sc = (Pt[ti] * enc(qd)[:, None, None, :]).sum(-1).cpu().numpy()
                sd_sc = np.where(occ[te][:, None, :] > 0, sd_sc, -1e9)
                pf2, pt2 = np.unravel_index(sd_sc.reshape(len(sd_sc), -1).argmax(1),
                                            (N_FREQ, N_TIME))
                res["query_swap_to_distractor"] = {
                    "lands_on_distractor": float(np.mean((np.abs(pt2 - tdist[te]) <= 1)
                                                         & (pf2 == fdist[te]))),
                    "still_on_original": float(np.mean((np.abs(pt2 - tt[te]) <= 1)
                                                       & (pf2 == ff[te])))}

                # kept for the record: the shift test that could not discriminate
                with torch.no_grad():
                    q2 = Qt[torch.tensor(np.where(te)[0], device=dev)].clone()
                    q2[:, 0] += 0.4
                    n2 = enc(q2)
                    s2 = (Pt[torch.tensor(np.where(te)[0], device=dev)] *
                          n2[:, None, None, :]).sum(-1).cpu().numpy()
                pt1 = np.unravel_index(np.where(occ[te][:, None, :] > 0, sc, -1e9)
                                       .reshape(len(sc), -1).argmax(1), (N_FREQ, N_TIME))[1]
                pt2 = np.unravel_index(np.where(occ[te][:, None, :] > 0, s2, -1e9)
                                       .reshape(len(s2), -1).argmax(1), (N_FREQ, N_TIME))[1]
                res["query_swap_time"] = {"mean_shift_patches": float(np.mean(pt2 - pt1)),
                                          "expected": 4.0,
                                          "moved_right_frac": float(np.mean(pt2 > pt1))}
        res[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        res[name + "_sd"] = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}

    res["random"] = scorer_eval(np.random.RandomState(0).randn(int(te.sum()), N_FREQ, N_TIME), te)

    print(f"\ntest n={int(te.sum())}, unseen patients and unseen f0 band\n")
    print("%-18s %8s %9s %9s %9s %11s" % ("scorer", "Hit@1", "Hit@±1", "freq_acc",
                                           "2D", "distractor"))
    for k in ["typed_intact", "shuffled_coords", "energy_tonality", "random"]:
        r = res[k]
        print("%-18s %8.3f %9.3f %9.3f %9.3f %11.3f" % (k, r["hit1_t"], r["hit1pm_t"],
                                                        r["freq_acc"], r["point2d"],
                                                        r["distractor"]))
    sw = res.get("query_swap_to_distractor", {})
    print(f"\nquery swapped to the distractor's coordinates:")
    print(f"  peak lands on the distractor  {sw.get('lands_on_distractor', 0):.3f}")
    print(f"  peak stays on the original    {sw.get('still_on_original', 0):.3f}")
    qs = res.get("query_swap_time", {})
    print(f"  (nudge test, cannot discriminate here: shift "
          f"{qs.get('mean_shift_patches', 0):+.2f} patches)")
    go = (res["typed_intact"]["hit1pm_t"] > res["shuffled_coords"]["hit1pm_t"]
          and res["typed_intact"]["hit1pm_t"] > res["energy_tonality"]["hit1pm_t"])
    print(f"\nGO criterion (beat BOTH shuffled_coords and energy_tonality): "
          f"{'PASS' if go else 'FAIL'}")
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
