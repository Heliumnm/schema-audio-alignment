"""Grounding v2 — synthesis, matching checks, and the AST feasibility gate.

Two earlier designs failed because the answer was reachable without the audio.

  v0  one injection per clip; the tone was conspicuous enough to find from the patch
      features alone, so shuffling the query changed nothing (0.977 vs 0.965).
  v1  two injections, query = [onset + dur/2, dur, f0/1000]. The target is a
      deterministic function of exactly those numbers — a coordinate-only baseline
      with no audio scored **1.000**, above the learned model's 0.935.

v2 makes the query a single bit: which *character* to locate, mono or poly. Stated
precisely, because the earlier phrasing was too strong:

  **exact coordinate leakage is impossible by construction** — one bit cannot encode
  a position — while **statistical positional leakage is excluded by matched
  distributions, counterbalanced assignment, and the position-only control**, which
  are empirical claims this file checks rather than assumes.

Design
------
* one audio, two events, two queries: the paired comparison happens *within* a clip,
  so per-clip confounds cancel;
* mono = a harmonic pair (fc, 2fc); poly = three simultaneous **inharmonic** tones.
  No frequency modulation — bundling two acoustic properties would make a failure
  impossible to attribute. Harmonicity is the only manipulated variable;
* **poly's ratios are normalised to unit geometric mean** (0.7396, 1.0133, 1.3387).
  The raw ratios (1, 1.37, 1.81) have geometric mean 1.352, which would put poly's
  spectral centre 35% above mono's — a leak that needs no listening;
* slot properties (onset, fc, SNR, duration) are drawn **per slot, before any
  character is assigned**, and assignment is **counterbalanced** rather than left to
  the RNG, so mono-first is exactly 50% within each split;
* the target is the **union of the occupied frequency bands**, not a rectangle
  spanning the empty gaps between poly's components.

Feasibility gate — run before any grounding claim
-------------------------------------------------
If AST patches cannot separate mono from poly at all, a localisation failure says
nothing about grounding. Frozen AST, fixed-size crops at known event locations,
linear probe, train/dev only, patient-level bootstrap:

    95% CI covers 0.5, or AUROC < 0.60   -> infeasible; change the events, draw no
                                            grounding conclusion
    95% CI lower bound > 0.60            -> enough separable information, proceed
    otherwise                            -> insufficient; revise before freezing

An `event vs background` probe runs alongside, since separating injections from
breath sound is a weaker prerequisite than telling the two characters apart.
"""
import os, json, argparse
import numpy as np

SR, PATCH, STRIDE = 16000, 16, 10
N_FREQ, N_TIME, N_MELS, MAX_FRAMES = 12, 101, 128, 1024
MEL_MAX = 2595 * np.log10(1 + (SR / 2) / 700.0)

# inharmonic ratios normalised to unit geometric mean, so poly and mono share a
# log-frequency centre; without this poly sits 35% higher by construction
_RAW = np.array([1.0, 1.37, 1.81])
POLY_RATIOS = _RAW / np.exp(np.mean(np.log(_RAW)))
MONO_RATIOS = np.array([1.0, 2.0])


def mel_row(f_hz):
    mel = 2595 * np.log10(1 + f_hz / 700.0)
    b = np.clip(mel / MEL_MAX * N_MELS, 0, N_MELS - 1)
    return int(np.clip(round((b - PATCH / 2) / STRIDE), 0, N_FREQ - 1))


def event_region(onset_s, dur_s, freqs, n_valid):
    """Union of the rows the components occupy — not a box spanning the gaps."""
    f0_frame, f1_frame = onset_s * 100.0, (onset_s + dur_s) * 100.0
    t_lo = int(np.clip(np.floor((f0_frame - PATCH) / STRIDE) + 1, 0, N_TIME - 1))
    t_hi = int(np.clip(np.floor(f1_frame / STRIDE), 0, N_TIME - 1))
    mask = np.zeros((N_FREQ, N_TIME), bool)
    for f in freqs:
        mask[mel_row(f), t_lo:t_hi + 1] = True
    occ = np.clip((n_valid - np.arange(N_TIME) * STRIDE) / PATCH, 0, 1).astype(np.float32)
    return mask, occ


def synth_event(y, onset_s, dur_s, fc, character, snr_db, rng, sr=SR):
    """Matched on total RMS, SNR, duration, envelope and centre frequency; component
    phases randomised for both. Harmonicity is the only difference."""
    n0, n1 = int(onset_s * sr), min(int((onset_s + dur_s) * sr), len(y))
    if n1 - n0 < int(0.08 * sr):
        return None, None
    t = np.arange(n1 - n0) / sr
    ratios = MONO_RATIOS if character == "mono" else POLY_RATIOS
    freqs = fc * ratios
    sig = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs)
    ramp = int(0.02 * sr)
    env = np.ones_like(sig)
    if len(env) > 2 * ramp:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        env[:ramp], env[-ramp:] = w, w[::-1]
    sig = sig * env
    local = np.sqrt(np.mean(y[n0:n1] ** 2)) + 1e-8
    sig = sig * local * (10 ** (snr_db / 20)) / (np.sqrt(np.mean(sig ** 2)) + 1e-8)
    out = y.copy()
    out[n0:n1] += sig
    return out, freqs


def build_clip(y, dur, rng, f_band, mono_first):
    """Slot properties first, character assigned afterwards and counterbalanced."""
    d = rng.uniform(0.25, min(0.55, max(0.3, dur * 0.32)))
    if dur < 2 * d + 0.25:
        return None
    slots = [dict(on=rng.uniform(0, max(0.01, dur / 2 - d)), dur=d,
                  fc=rng.uniform(*f_band), snr=rng.uniform(4, 9)),
             dict(on=rng.uniform(dur / 2 + 0.05, max(dur / 2 + 0.06, dur - d)), dur=d,
                  fc=rng.uniform(*f_band), snr=rng.uniform(4, 9))]
    order = ["mono", "poly"] if mono_first else ["poly", "mono"]
    ev = {ch: dict(slots[k], slot=k) for k, ch in enumerate(order)}
    yi = y
    for ch in ("mono", "poly"):
        e = ev[ch]
        yi, freqs = synth_event(yi, e["on"], e["dur"], e["fc"], ch, e["snr"], rng)
        if yi is None:
            return None
        e["freqs"] = freqs
    yi = yi / (np.abs(yi).max() + 1e-8)
    nvf = min(int(len(yi) / SR * 100), MAX_FRAMES)
    for ch in ("mono", "poly"):
        e = ev[ch]
        e["mask"], occ = event_region(e["on"], e["dur"], e["freqs"], nvf)
    return yi, ev, occ


def matching_report(rows):
    """Every claim the design rests on, measured rather than asserted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    print("\n--- matching checks ---")
    for k in ["on", "fc", "dur", "snr"]:
        m = np.array([r["ev"]["mono"][k] for r in rows])
        p = np.array([r["ev"]["poly"][k] for r in rows])
        print(f"  {k:4s}  mono {m.mean():8.3f}±{m.std():6.3f}   "
              f"poly {p.mean():8.3f}±{p.std():6.3f}   Δ {abs(m.mean()-p.mean()):.4f}")
    mf = np.mean([r["ev"]["mono"]["slot"] == 0 for r in rows])
    print(f"  mono-first fraction {mf:.4f}  (counterbalanced, must be ~0.5)")
    sm = np.array([r["ev"]["mono"]["mask"].sum() for r in rows])
    sp = np.array([r["ev"]["poly"]["mask"].sum() for r in rows])
    print(f"  target patches  mono {sm.mean():.1f}±{sm.std():.1f}   "
          f"poly {sp.mean():.1f}±{sp.std():.1f}   Δ {abs(sm.mean()-sp.mean()):.2f}")
    ov = np.mean([(r["ev"]["mono"]["mask"] & r["ev"]["poly"]["mask"]).sum() > 0 for r in rows])
    print(f"  clips whose two regions overlap: {ov:.3f}  (unscorable, excluded)")
    # can character be read off the nuisance variables alone?
    X = np.array([[r["ev"][c][k] for k in ("on", "fc", "dur", "snr")]
                  for r in rows for c in ("mono", "poly")])
    yy = np.array([0 if c == "mono" else 1 for r in rows for c in ("mono", "poly")])
    auc = roc_auc_score(yy, LogisticRegression(max_iter=5000).fit(X, yy)
                        .predict_proba(X)[:, 1])
    print(f"  character from (onset, fc, dur, snr): AUROC {auc:.3f}  "
          f"(in-sample upper bound; must be ~0.5)")
    return auc


def crops(P, ev, occ, size=3):
    """Fixed-size crop centred on an event, so mask extent cannot reveal character."""
    out = []
    for ch in ("mono", "poly"):
        m = ev[ch]["mask"]
        fr, ti = np.where(m)
        cf, ct = int(np.median(fr)), int(np.median(ti))
        f0 = np.clip(cf - size // 2, 0, N_FREQ - size); t0 = np.clip(ct - size // 2, 0, N_TIME - size)
        out.append((ch, P[f0:f0 + size, t0:t0 + size].reshape(-1)))
    return out


def background_crop(P, ev, occ, rng, size=3):
    valid = np.where(occ > 0)[0]
    both = ev["mono"]["mask"] | ev["poly"]["mask"]
    for _ in range(40):
        t0 = int(rng.choice(valid[:max(1, len(valid) - size)]))
        f0 = int(rng.randint(0, N_FREQ - size))
        if not both[f0:f0 + size, t0:t0 + size].any():
            return P[f0:f0 + size, t0:t0 + size].reshape(-1)
    return None


def probe(X, y, groups, name, boot=2000):
    """Linear probe with a patient-level bootstrap CI, on train/dev only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    g = np.array(groups)
    pats = np.array(sorted(set(g)))
    rs = np.random.RandomState(0); rs.shuffle(pats)
    dev_p = set(pats[:max(2, len(pats) // 3)])
    dv = np.isin(g, list(dev_p)); tr = ~dv
    clf = LogisticRegression(max_iter=20000, class_weight="balanced").fit(X[tr], y[tr])
    s = clf.predict_proba(X[dv])[:, 1]
    auc = roc_auc_score(y[dv], s)
    gd = g[dv]; up = np.array(sorted(set(gd))); by = {p: np.where(gd == p)[0] for p in up}
    rb = np.random.RandomState(1); bs = []
    for _ in range(boot):
        take = np.concatenate([by[p] for p in rb.choice(up, len(up))])
        if len(set(y[dv][take])) < 2: continue
        bs.append(roc_auc_score(y[dv][take], s[take]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    if hi < 0.5 or lo <= 0.5 or auc < 0.60:
        v = "INFEASIBLE"
    elif lo > 0.60:
        v = "FEASIBLE"
    else:
        v = "insufficient"
    print(f"  {name:<22} AUROC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  "
          f"n_dev={int(dv.sum())}  -> {v}")
    return {"auroc": float(auc), "ci": [float(lo), float(hi)], "verdict": v}


def main():
    import torch
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model", default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", default="results/grounding_v2_feasibility.json")
    ap.add_argument("--manifest_out", default="results/grounding_v2_manifest.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    # TRAIN SPLIT ONLY — the official test set is not touched until the design freezes
    pool = [e for e in seg if e["label"] == "normal" and e.get("duration", 0) >= 1.2
            and smap.get(os.path.basename(e["path"])) == "train"]
    rng.shuffle(pool); pool = pool[:args.n]
    print(f"pool: {len(pool)} normal TRAIN cycles, "
          f"{len(set(os.path.basename(e['path']).split('_')[0] for e in pool))} patients")

    rows = []
    for k, e in enumerate(pool):
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        r = build_clip(y, len(y) / SR, rng, (350, 1100), mono_first=(k % 2 == 0))
        if r is None: continue
        yi, ev, occ = r
        if (ev["mono"]["mask"] & ev["poly"]["mask"]).any(): continue
        rows.append({"y": yi, "ev": ev, "occ": occ,
                     "pid": os.path.basename(p).split("_")[0], "sid": os.path.basename(p)})
    print(f"built {len(rows)} clips")
    assoc = matching_report(rows)

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)
    grids = []
    for i in range(0, len(rows), 8):
        inp = fe([r["y"] for r in rows[i:i+8]], sampling_rate=SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            h = m(**inp).last_hidden_state[:, 2:, :]
        grids.append(h.reshape(len(h), N_FREQ, N_TIME, -1).cpu().numpy().astype(np.float32))
    G = np.concatenate(grids)
    print(f"patch grids {G.shape}")

    Xc, yc, gc, Xb, yb, gb = [], [], [], [], [], []
    rb = np.random.RandomState(2)
    for n, r in enumerate(rows):
        for ch, v in crops(G[n], r["ev"], r["occ"]):
            Xc.append(v); yc.append(0 if ch == "mono" else 1); gc.append(r["pid"])
            Xb.append(v); yb.append(1); gb.append(r["pid"])
        bg = background_crop(G[n], r["ev"], r["occ"], rb)
        if bg is not None:
            Xb.append(bg); yb.append(0); gb.append(r["pid"])

    print("\n--- AST feasibility gate (train/dev only) ---")
    res = {"association_auroc": float(assoc)}
    res["mono_vs_poly"] = probe(np.array(Xc), np.array(yc), gc, "mono vs poly")
    res["event_vs_background"] = probe(np.array(Xb), np.array(yb), gb, "event vs background")

    json.dump(res, open(args.out, "w"), indent=1)
    # Freeze the synthesis as bounds rather than dumping the boolean masks: the
    # manifest has to be re-readable and diffable, and a 12x101 array per event is
    # neither.
    def freeze(r):
        out = {"sid": r["sid"], "pid": r["pid"], "events": {}}
        for ch in ("mono", "poly"):
            e = r["ev"][ch]
            fr, ti = np.where(e["mask"])
            out["events"][ch] = {
                "onset_s": float(e["on"]), "dur_s": float(e["dur"]),
                "fc_hz": float(e["fc"]), "snr_db": float(e["snr"]),
                "slot": int(e["slot"]), "freqs_hz": [float(x) for x in e["freqs"]],
                "rows": sorted(set(int(x) for x in fr)),
                "t_lo": int(ti.min()), "t_hi": int(ti.max())}
        out["n_valid_time_patches"] = int((r["occ"] > 0).sum())
        return out
    json.dump([freeze(r) for r in rows], open(args.manifest_out, "w"), indent=1)
    print(f"\nwrote {args.out} and the frozen synthesis manifest {args.manifest_out}")
    if res["mono_vs_poly"]["verdict"] != "FEASIBLE":
        print("\nGATE NOT PASSED — revise the synthetic events before freezing. A "
              "localisation failure at this point would say nothing about grounding.")


if __name__ == "__main__":
    main()
