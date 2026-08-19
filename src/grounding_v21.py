"""Grounding v2.1 — leak-free synthesis, matching checks, and the AST feasibility gate.

Three designs have now failed because the answer was reachable without the audio.

  v0    one injection per clip; the tone was conspicuous enough to find from the patch
        features alone, so shuffling the query changed nothing (0.977 vs 0.965).
  v1    two injections, query = [onset + dur/2, dur, f0/1000]. The target is a
        deterministic function of exactly those numbers — a coordinate-only baseline
        with no audio scored **1.000**, above the learned model's 0.935.
  v2    query reduced to one bit (mono vs poly), which does make exact coordinate
        leakage impossible. It leaked anyway, geometrically. Only poly's ratios were
        normalised to unit geometric mean; mono's `{1, 2}` has geometric mean
        sqrt(2) = 1.414, so mono's spectral centre sat 41% above poly's, and mono
        spanned an octave against poly's 1.81. Crop centre, mask extent and patch count
        therefore all tracked the character: **a probe on crop geometry alone, with no
        audio whatsoever, reached AUROC 0.982**, so the gate's apparent AUROC 1.000
        measured where the crop was, not what was inside it.

The lesson is the same each time and is now enforced structurally rather than checked
afterwards: **anything the target mask reveals is a leak**, so the mask must be
identical for both characters by construction.

v2.1 design
-----------
Both characters are three simultaneous tones with **endpoints fixed at {1, 4} x fc**:

    harmonic     fc * {1, 2.000, 4}          middle partial = the 2nd harmonic
    inharmonic   fc * {1, 1.550, 4}   or     middle partial displaced by -/+ 0.368
                 fc * {1, 2.581, 4}          octave, counterbalanced

Consequences, all by construction rather than by measurement:

* lowest and highest partials are identical, so `row(fc)` and `row(4fc)` match;
* the target mask is the **whole band** `row(fc) .. row(4fc)`, not the union of the
  occupied rows — v2's union was itself a readout of the spacing it had to hide. Mask
  extent, patch count and crop centre are therefore identical across characters;
* three components in both arms, so component count cannot be counted off;
* the displacement is symmetric in log-frequency and counterbalanced, so the mean log
  spectral centroid of the inharmonic arm equals the harmonic arm's exactly;
* fc is drawn from the **admissible set** on which `row(fc) < row(1.550 fc) <
  row(2 fc) < row(2.581 fc) < row(4 fc)` strictly — 73.7% of 300–1200 Hz. Admissibility
  does not depend on character, so the restriction cannot leak. Outside that set the
  manipulation is not representable on a 12-row grid at all, which is a fact about
  AST's frequency resolution rather than about what the model can hear.

The only thing that differs is **which row inside an identical band carries energy**.
No coordinate reveals that; it has to come from the patch features.

What this costs
---------------
This is no longer a monophonic-vs-polyphonic wheeze proxy. Clinical monophonic and
polyphonic wheezes differ in how many tones sound at once, and component count is
exactly what had to be equalised to stop the leak. What remains is harmonic vs
inharmonic partial structure — a mechanism probe on whether AST patches carry
resolvable spectral fine structure. Any result here is about that and nothing else.

Feasibility gate — run before any grounding claim
-------------------------------------------------
If AST patches cannot separate the two characters at all, a localisation failure says
nothing about grounding. Frozen AST, fixed crops at known event locations, linear
probe, train/dev only, patient-level bootstrap:

    95% CI covers 0.5, or AUROC < 0.60   -> infeasible; the events are not resolved,
                                            draw no grounding conclusion
    95% CI lower bound > 0.60            -> enough separable information, proceed
    otherwise                            -> insufficient; revise before freezing

A **geometry-only** arm runs alongside and is not optional: the same probe on features
made only of crop and mask coordinates, no audio at all. It is the control that caught
v2, and the gate is void unless it sits at chance. An `event vs background` arm runs
too, since separating injections from breath sound is a weaker prerequisite than
telling the two characters apart.

    python src/grounding_v21.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np

SR, PATCH, STRIDE = 16000, 16, 10
N_FREQ, N_TIME, N_MELS, MAX_FRAMES = 12, 101, 128, 1024
MEL_MAX = 2595 * np.log10(1 + (SR / 2) / 700.0)
CROP_F, CROP_T = 5, 3          # 5 rows spans the whole band; 3 frames spans the event

# Endpoints shared, middle partial displaced symmetrically in log-frequency.
HARM_RATIOS = np.array([1.0, 2.0, 4.0])
INHARM_RATIOS = [np.array([1.0, 2.0 * 2 ** -0.368, 4.0]),
                 np.array([1.0, 2.0 * 2 ** +0.368, 4.0])]
CHARS = ("harm", "inharm")


def mel_row(f_hz):
    mel = 2595 * np.log10(1 + f_hz / 700.0)
    b = np.clip(mel / MEL_MAX * N_MELS, 0, N_MELS - 1)
    return int(np.clip(round((b - PATCH / 2) / STRIDE), 0, N_FREQ - 1))


def admissible_fc(lo=300.0, hi=1200.0, step=1.0):
    """fc values on which the manipulation is representable at all: the harmonic middle
    row must sit strictly between the two inharmonic ones and strictly inside the band.
    Independent of character, so restricting to this set cannot leak."""
    ml, mh = INHARM_RATIOS[0][1], INHARM_RATIOS[1][1]
    return np.array([f for f in np.arange(lo, hi, step)
                     if mel_row(f) < mel_row(f * ml) < mel_row(f * 2.0)
                     < mel_row(f * mh) < mel_row(f * 4.0)])


def event_region(onset_s, dur_s, freqs, n_valid):
    """The whole occupied band, lowest partial row to highest — identical for both
    characters because the endpoints are."""
    f0_frame, f1_frame = onset_s * 100.0, (onset_s + dur_s) * 100.0
    t_lo = int(np.clip(np.floor((f0_frame - PATCH) / STRIDE) + 1, 0, N_TIME - 1))
    t_hi = int(np.clip(np.floor(f1_frame / STRIDE), 0, N_TIME - 1))
    r_lo, r_hi = mel_row(min(freqs)), mel_row(max(freqs))
    mask = np.zeros((N_FREQ, N_TIME), bool)
    mask[r_lo:r_hi + 1, t_lo:t_hi + 1] = True
    occ = np.clip((n_valid - np.arange(N_TIME) * STRIDE) / PATCH, 0, 1).astype(np.float32)
    return mask, occ


def synth_event(y, onset_s, dur_s, fc, character, variant, snr_db, rng, sr=SR):
    """Matched on total RMS, SNR, duration, envelope, endpoints and component count;
    component phases randomised. The middle partial's position is the only difference."""
    n0, n1 = int(onset_s * sr), min(int((onset_s + dur_s) * sr), len(y))
    if n1 - n0 < int(0.08 * sr):
        return None, None
    t = np.arange(n1 - n0) / sr
    ratios = HARM_RATIOS if character == "harm" else INHARM_RATIOS[variant]
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


def crop_origin(mask):
    fr, ti = np.where(mask)
    cf, ct = int(np.median(fr)), int(np.median(ti))
    return (int(np.clip(cf - CROP_F // 2, 0, N_FREQ - CROP_F)),
            int(np.clip(ct - CROP_T // 2, 0, N_TIME - CROP_T)))


def build_clip(y, dur, rng, fc_pool, harm_first, variant):
    """Slot properties first, character assigned afterwards and counterbalanced."""
    d = rng.uniform(0.25, min(0.55, max(0.3, dur * 0.32)))
    if dur < 2 * d + 0.25:
        return None
    slots = [dict(on=rng.uniform(0, max(0.01, dur / 2 - d)), dur=d,
                  fc=float(rng.choice(fc_pool)), snr=rng.uniform(4, 9)),
             dict(on=rng.uniform(dur / 2 + 0.05, max(dur / 2 + 0.06, dur - d)), dur=d,
                  fc=float(rng.choice(fc_pool)), snr=rng.uniform(4, 9))]
    order = ["harm", "inharm"] if harm_first else ["inharm", "harm"]
    ev = {ch: dict(slots[k], slot=k) for k, ch in enumerate(order)}
    yi = y
    for ch in CHARS:
        e = ev[ch]
        e["variant"] = variant if ch == "inharm" else -1
        yi, freqs = synth_event(yi, e["on"], e["dur"], e["fc"], ch, variant, e["snr"], rng)
        if yi is None:
            return None
        e["freqs"] = freqs
    yi = yi / (np.abs(yi).max() + 1e-8)
    nvf = min(int(len(yi) / SR * 100), MAX_FRAMES)
    for ch in CHARS:
        e = ev[ch]
        e["mask"], occ = event_region(e["on"], e["dur"], e["freqs"], nvf)
    return yi, ev, occ


def geometry_features(ev, ch):
    """Everything about an event knowable without listening. If a probe on these
    separates the characters, the design leaks and nothing else in the run matters."""
    e = ev[ch]
    fr, ti = np.where(e["mask"])
    r_lo, r_hi, t_lo, t_hi = int(fr.min()), int(fr.max()), int(ti.min()), int(ti.max())
    cf, ct = crop_origin(e["mask"])
    return [cf, ct, r_lo, r_hi, r_hi - r_lo, t_lo, t_hi, t_hi - t_lo,
            int(e["mask"].sum()), e["on"], e["dur"], e["fc"], e["snr"], e["slot"]]


def matching_report(rows):
    """Every claim the design rests on, measured rather than asserted."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    print("\n--- matching checks ---")
    for k in ["on", "fc", "dur", "snr"]:
        a = np.array([r["ev"]["harm"][k] for r in rows])
        b = np.array([r["ev"]["inharm"][k] for r in rows])
        print(f"  {k:4s} harm {a.mean():8.3f} +/-{a.std():6.3f} | "
              f"inharm {b.mean():8.3f} +/-{b.std():6.3f} | delta {abs(a.mean()-b.mean()):.4f}")
    hf = np.mean([r["ev"]["harm"]["slot"] == 0 for r in rows])
    vh = np.mean([r["ev"]["inharm"]["variant"] == 1 for r in rows])
    print(f"  harm-first {hf:.4f}   upward-variant {vh:.4f}   (both must be ~0.5)")
    sa = np.array([r["ev"]["harm"]["mask"].sum() for r in rows])
    sb = np.array([r["ev"]["inharm"]["mask"].sum() for r in rows])
    print(f"  mask patches  harm {sa.mean():.2f} +/-{sa.std():.2f} | "
          f"inharm {sb.mean():.2f} +/-{sb.std():.2f} | delta {abs(sa.mean()-sb.mean()):.4f}")
    ca = np.array([np.mean(np.log2(r["ev"]["harm"]["freqs"] / r["ev"]["harm"]["fc"]))
                   for r in rows])
    cb = np.array([np.mean(np.log2(r["ev"]["inharm"]["freqs"] / r["ev"]["inharm"]["fc"]))
                   for r in rows])
    print(f"  log2 centroid/fc  harm {ca.mean():+.4f} | inharm {cb.mean():+.4f} | "
          f"delta {abs(ca.mean()-cb.mean()):.4f} octave   (v2: 0.500)")
    ov = np.mean([(r["ev"]["harm"]["mask"] & r["ev"]["inharm"]["mask"]).sum() > 0
                  for r in rows])
    print(f"  clips whose two regions overlap: {ov:.3f}  (unscorable, excluded)")
    X = np.array([geometry_features(r["ev"], c) for r in rows for c in CHARS], float)
    yy = np.array([0 if c == "harm" else 1 for r in rows for c in CHARS])
    auc = roc_auc_score(yy, LogisticRegression(max_iter=5000).fit(X, yy).predict_proba(X)[:, 1])
    print(f"  character from geometry alone, no audio: AUROC {auc:.3f}  "
          f"(in-sample upper bound; v2 scored 0.982 here)")
    return auc


def crops(P, ev):
    """Fixed-size crop centred on the band, identical placement rule for both."""
    out = []
    for ch in CHARS:
        f0, t0 = crop_origin(ev[ch]["mask"])
        out.append((ch, P[f0:f0 + CROP_F, t0:t0 + CROP_T].reshape(-1)))
    return out


def background_crop(P, ev, occ, rng):
    valid = np.where(occ > 0)[0]
    both = ev["harm"]["mask"] | ev["inharm"]["mask"]
    for _ in range(40):
        t0 = int(rng.choice(valid[:max(1, len(valid) - CROP_T)]))
        f0 = int(rng.randint(0, N_FREQ - CROP_F))
        if not both[f0:f0 + CROP_F, t0:t0 + CROP_T].any():
            return P[f0:f0 + CROP_F, t0:t0 + CROP_T].reshape(-1)
    return None


def probe(X, y, groups, name, boot=2000):
    """Linear probe with a patient-level bootstrap CI, on train/dev only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    g = np.array(groups)
    pats = np.array(sorted(set(g)))
    rs = np.random.RandomState(0); rs.shuffle(pats)
    dev_p = set(pats[:max(2, len(pats) // 3)])
    dv = np.isin(g, list(dev_p)); tr = ~dv
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=20000, class_weight="balanced"))
    clf.fit(X[tr], y[tr])
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
    print(f"  {name:<26s} AUROC {auc:.3f}  95pct CI [{lo:.3f}, {hi:.3f}]  "
          f"n_dev={int(dv.sum())}  -> {v}")
    return {"auroc": float(auc), "ci": [float(lo), float(hi)], "verdict": v}


def main():
    import torch
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", default="results/grounding_v21_feasibility.json")
    ap.add_argument("--manifest_out", default="results/grounding_v21_manifest.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    # TRAIN SPLIT ONLY — the official test set is untouched until the design freezes
    pool = [e for e in seg if e["label"] == "normal" and e.get("duration", 0) >= 1.2
            and smap.get(os.path.basename(e["path"])) == "train"]
    rng.shuffle(pool); pool = pool[:args.n]
    fc_pool = admissible_fc()
    print(f"pool: {len(pool)} normal TRAIN cycles, "
          f"{len(set(os.path.basename(e['path']).split('_')[0] for e in pool))} patients")
    print(f"admissible fc: {len(fc_pool)} values in {fc_pool.min():.0f}-{fc_pool.max():.0f} Hz")

    rows = []
    for k, e in enumerate(pool):
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        # both binary nuisances counterbalanced on the index, not left to the RNG
        r = build_clip(y, len(y) / SR, rng, fc_pool,
                       harm_first=(k % 2 == 0), variant=(k // 2) % 2)
        if r is None: continue
        yi, ev, occ = r
        if (ev["harm"]["mask"] & ev["inharm"]["mask"]).any(): continue
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

    Xc, yc, gc, Xb, yb, gb, Xg = [], [], [], [], [], [], []
    rb = np.random.RandomState(2)
    for n, r in enumerate(rows):
        for ch, v in crops(G[n], r["ev"]):
            Xc.append(v); yc.append(0 if ch == "harm" else 1); gc.append(r["pid"])
            Xg.append(geometry_features(r["ev"], ch))
            Xb.append(v); yb.append(1); gb.append(r["pid"])
        bg = background_crop(G[n], r["ev"], r["occ"], rb)
        if bg is not None:
            Xb.append(bg); yb.append(0); gb.append(r["pid"])

    print("\n--- AST feasibility gate (train/dev only) ---")
    res = {"geometry_auroc_insample": float(assoc)}
    res["geometry_only"] = probe(np.array(Xg, float), np.array(yc), gc, "geometry only (control)")
    res["harm_vs_inharm"] = probe(np.array(Xc), np.array(yc), gc, "harmonic vs inharmonic")
    res["event_vs_background"] = probe(np.array(Xb), np.array(yb), gb, "event vs background")
    json.dump(res, open(args.out, "w"), indent=1)

    def freeze(r):
        """Bounds, not boolean masks: the manifest must stay readable and diffable."""
        out = {"sid": r["sid"], "pid": r["pid"], "events": {}}
        for ch in CHARS:
            e = r["ev"][ch]
            fr, ti = np.where(e["mask"])
            out["events"][ch] = {
                "onset_s": float(e["on"]), "dur_s": float(e["dur"]),
                "fc_hz": float(e["fc"]), "snr_db": float(e["snr"]),
                "slot": int(e["slot"]), "variant": int(e["variant"]),
                "freqs_hz": [float(x) for x in e["freqs"]],
                "r_lo": int(fr.min()), "r_hi": int(fr.max()),
                "t_lo": int(ti.min()), "t_hi": int(ti.max())}
        out["n_valid_time_patches"] = int((r["occ"] > 0).sum())
        return out
    json.dump([freeze(r) for r in rows], open(args.manifest_out, "w"), indent=1)
    print(f"\nwrote {args.out} and the frozen synthesis manifest {args.manifest_out}")

    geo, aud = res["geometry_only"], res["harm_vs_inharm"]
    if geo["ci"][1] > 0.60:
        print(f"\nGATE VOID — geometry alone reaches {geo['auroc']:.3f} (CI upper "
              f"{geo['ci'][1]:.3f}). The design still leaks, so no number below it means "
              f"anything. Fix the synthesis; do not interpret the audio probe.")
    elif aud["verdict"] != "FEASIBLE":
        print(f"\nGATE NOT PASSED — geometry is at chance, so this is a real negative: "
              f"AST patches do not resolve the manipulation ({aud['auroc']:.3f}). Per the "
              f"pre-registered plan, draw no grounding conclusion on this axis.")
    else:
        print(f"\nGATE PASSED — geometry at chance ({geo['auroc']:.3f}), audio at "
              f"{aud['auroc']:.3f}. The separability is acoustic. Proceed to the "
              f"shortcut-control suite, still on train/dev only.")


if __name__ == "__main__":
    main()
