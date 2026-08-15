"""H12 Step 0 — is the H11 gap neighbourhood reading, or is it search?

H11 left two numbers on the same dev examples: a per-patch bilinear head at 0.569 and
an *oracle-location* crop classifier at 1.000. The oracle differs from the head in two
ways at once — it reads a 5x3 crop **jointly**, and it is **handed both ground-truth
locations**. A region adapter addresses only the first. This separates them before any
H12 code is written.

The readout is the one dense CLIP will use, not a probe on pooled features: score every
valid window with the frozen 5x3 crop representation, conditioned on the query, and take
the max over windows.

    s(q, f, t) = <query embedding, crop(P, f, t)>
    region wins if  max over its windows  >  max over the distractor's windows

The query embedding is the oracle's own fitted direction, +/-w for inharmonic/harmonic,
with the pipeline's StandardScaler folded into the weights so an arbitrary window can be
scored by one dot product. Nothing is trained here that H11 did not already train: the
same classifier, the same crop representation, the same dev patients. The only thing
removed is free localisation.

  max-over-window ~ 1.000  -> joint neighbourhood reading is sufficient; the region
                             adapter is motivated by evidence and H12 proceeds
  collapses toward chance  -> the bottleneck is SEARCH, not neighbourhood reading, and
                             adding conv + attention would be an uncontrolled change

Not H7. H7 pooled features and probed for classification with no query. Here the query
enters the score and the metric is within-clip 2AFC localisation; a variant that
reduced to pooling-then-probing would be H7 and is not run.

    python src/h12_step0_window.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_v21 import (SR, N_FREQ, N_TIME, CROP_F, CROP_T, CHARS,
                           admissible_fc, build_clip, crop_origin)
from grounding_train import dev_patients, boot_ci

NF0, NT0 = N_FREQ - CROP_F + 1, N_TIME - CROP_T + 1        # 8 x 99 window origins


def verify_manifest(rows, path):
    """The frozen v2.1 synthesis is the input to this diagnostic. Re-running the seeded
    pipeline must reproduce it exactly, or the dev examples are not H11's and no paired
    comparison is legitimate."""
    man = json.load(open(path))
    assert len(man) == len(rows), f"clip count {len(rows)} != frozen {len(man)}"
    for r, f in zip(rows, man):
        assert r["sid"] == f["sid"] and r["pid"] == f["pid"], f"{r['sid']} != {f['sid']}"
        for ch in CHARS:
            e, g = r["ev"][ch], f["events"][ch]
            for k, kk in [("on", "onset_s"), ("dur", "dur_s"), ("fc", "fc_hz"),
                          ("snr", "snr_db")]:
                assert abs(float(e[k]) - g[kk]) < 1e-9, f"{r['sid']} {ch} {k}"
            assert int(e["slot"]) == g["slot"] and int(e["variant"]) == g["variant"]
            fr, ti = np.where(e["mask"])
            assert (int(fr.min()), int(fr.max()), int(ti.min()), int(ti.max())) == \
                   (g["r_lo"], g["r_hi"], g["t_lo"], g["t_hi"]), f"{r['sid']} {ch} mask"
    print(f"frozen-synthesis check: {len(rows)} clips reproduce "
          f"{os.path.basename(path)} exactly")


def window_masks(mask, occ):
    """Windows assigned to a region: centre patch inside the mask, all three time
    columns valid. Both characters get the identical rule, and their counts are
    reported so a max over a larger candidate set cannot be mistaken for signal."""
    ok_t = np.array([occ[t0 + CROP_T - 1] > 0 for t0 in range(NT0)])
    w = np.zeros((NF0, NT0), bool)
    ctr = mask[CROP_F // 2:CROP_F // 2 + NF0, CROP_T // 2:CROP_T // 2 + NT0]
    w[ctr] = True
    return w & ok_t[None, :]


def valid_windows(occ):
    ok_t = np.array([occ[t0 + CROP_T - 1] > 0 for t0 in range(NT0)])
    return np.ones((NF0, NT0), bool) & ok_t[None, :]


def afc(wins, pid, name, extra=""):
    lo, hi = boot_ci(wins, pid)
    print(f"  {name:<34s} 2AFC {wins.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]{extra}")
    return {"afc": float(wins.mean()), "ci": [lo, hi]}


def main():
    import torch
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--frozen", default="results/grounding_v21_manifest.json")
    ap.add_argument("--h11_preds", default="results/grounding_v21_preds.npz")
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--out", default="results/h12_step0_window.json")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    # TRAIN SPLIT ONLY, identical pool and RNG consumption to grounding_train.py
    pool = [e for e in seg if e["label"] == "normal" and e.get("duration", 0) >= 1.2
            and smap.get(os.path.basename(e["path"])) == "train"]
    rng.shuffle(pool); pool = pool[:args.n]
    fc_pool = admissible_fc()

    rows = []
    for k, e in enumerate(pool):
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        r = build_clip(y, len(y) / SR, rng, fc_pool,
                       harm_first=(k % 2 == 0), variant=(k // 2) % 2)
        if r is None: continue
        yi, ev, occ = r
        if (ev["harm"]["mask"] & ev["inharm"]["mask"]).any(): continue
        rows.append({"y": yi, "ev": ev, "occ": occ,
                     "pid": os.path.basename(p).split("_")[0], "sid": os.path.basename(p)})
    print(f"{len(rows)} clips, {len(set(r['pid'] for r in rows))} patients (TRAIN split only)")
    verify_manifest(rows, args.frozen)

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
    G = np.concatenate(grids); del grids
    print(f"patch grids {G.shape}")

    # same example construction and same dev patients as H11
    ci = np.repeat(np.arange(len(rows)), 2)
    qb = np.tile([0, 1], len(rows))                        # 0 = harm, 1 = inharm
    pid = np.array([rows[c]["pid"] for c in ci])
    dvp = dev_patients([r["pid"] for r in rows])
    dv = np.isin(pid, list(dvp)); tr = ~dv
    dvi = np.where(dv)[0]
    print(f"train {tr.sum()} / dev {dv.sum()} examples, patient-disjoint")

    # ---- the frozen crop representation: H11's oracle classifier, refit identically
    def crop_at(c, ch):
        f0, t0 = crop_origin(rows[c]["ev"][ch]["mask"])
        return G[c, f0:f0 + CROP_F, t0:t0 + CROP_T].reshape(-1)
    tri = sorted(set(ci[np.where(tr)[0]]))
    Xo = np.array([crop_at(c, ch) for c in tri for ch in CHARS])
    yo = np.array([0 if ch == "harm" else 1 for c in tri for ch in CHARS])
    oc = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=20000, class_weight="balanced")).fit(Xo, yo)
    sc_, lr_ = oc.named_steps["standardscaler"], oc.named_steps["logisticregression"]
    # fold the scaler into the weights so any window is one dot product
    w_eff = (lr_.coef_[0] / sc_.scale_).astype(np.float32)
    b_eff = float(lr_.intercept_[0] - (lr_.coef_[0] * sc_.mean_ / sc_.scale_).sum())

    # ---- score every valid window: 15 per-patch projections, then shift-and-add
    D = G.shape[-1]
    Wt = torch.tensor(w_eff.reshape(CROP_F * CROP_T, D).T, device=dev)     # (D, 15)
    Pw = np.empty((len(G), N_FREQ, N_TIME, CROP_F * CROP_T), np.float32)
    for i in range(0, len(G), 32):
        g = torch.tensor(G[i:i+32], device=dev)
        Pw[i:i+32] = (g.reshape(-1, D) @ Wt).reshape(len(g), N_FREQ, N_TIME, -1).cpu().numpy()
    S = np.full((len(G), NF0, NT0), b_eff, np.float32)
    for k in range(CROP_F * CROP_T):
        df, dt = divmod(k, CROP_T)
        S += Pw[:, df:df + NF0, dt:dt + NT0, k]
    del Pw
    print(f"window score maps {S.shape}  (every valid 5x3 window, not just event crops)")

    WM = {c: {ch: window_masks(rows[c]["ev"][ch]["mask"], rows[c]["occ"]) for ch in CHARS}
          for c in set(ci[dvi])}
    VW = {c: valid_windows(rows[c]["occ"]) for c in set(ci[dvi])}
    nh = np.array([WM[c][ "harm" ].sum() for c in sorted(WM)])
    nih = np.array([WM[c]["inharm"].sum() for c in sorted(WM)])
    print(f"candidate windows per region: harm {nh.mean():.2f} +/-{nh.std():.2f} | "
          f"inharm {nih.mean():.2f} +/-{nih.std():.2f} | delta {abs(nh.mean()-nih.mean()):.3f}"
          f"   valid windows/clip {np.mean([VW[c].sum() for c in VW]):.1f}")

    # ---- readouts. sign of the query embedding: +w for inharmonic, -w for harmonic
    empty = sum(1 for c in WM for ch in CHARS if WM[c][ch].sum() == 0)
    assert empty == 0, f"{empty} regions have no candidate window; the rule is degenerate"

    er = np.random.RandomState(7)
    def wins_of(reduce, q_of=None, src=None, tie_rng=er):
        out = []
        for j in dvi:
            c = ci[j]; q = qb[j] if q_of is None else q_of[j]
            s = S[c if src is None else src[j]] * (1.0 if q == 1 else -1.0)
            a = reduce(s[WM[c][CHARS[qb[j]]]]); b = reduce(s[WM[c][CHARS[1 - qb[j]]]])
            out.append(tie_rng.rand() < 0.5 if abs(a - b) < 1e-9 else a > b)
        return np.array(out, float)

    res, rep = {}, {}
    print("\n--- Step 0: query-conditioned window search (dev, patient-cluster bootstrap) ---")

    # sanity: evaluating S at the oracle crop origin must reproduce H11's 1.000
    orc = []
    for j in dvi:
        c, q = ci[j], qb[j]
        sgn = 1.0 if q == 1 else -1.0
        f0, t0 = crop_origin(rows[c]["ev"][CHARS[q]]["mask"])
        f1, t1 = crop_origin(rows[c]["ev"][CHARS[1 - q]]["mask"])
        orc.append(sgn * S[c, f0, t0] > sgn * S[c, f1, t1])
    res["oracle_single_window"] = afc(np.array(orc, float), pid[dvi],
                                      "oracle single window (H11 = 1.000)")

    res["max_over_windows"] = afc(wins_of(np.max), pid[dvi],
                                  "max over windows in region  [PRIMARY]")
    res["mean_over_windows"] = afc(wins_of(np.mean), pid[dvi],
                                   "mean over windows in region")

    # pure search: does the global argmax over ALL valid windows land in the queried region
    hit, chance = [], []
    for j in dvi:
        c, q = ci[j], qb[j]
        s = np.where(VW[c], S[c] * (1.0 if q == 1 else -1.0), -1e9)
        f0, t0 = np.unravel_index(int(s.argmax()), s.shape)
        hit.append(bool(WM[c][CHARS[q]][f0, t0]))
        chance.append(WM[c][CHARS[q]].sum() / max(VW[c].sum(), 1))
    rep["hit_at_argmax_window"] = float(np.mean(hit))
    rep["hit_at_argmax_chance"] = float(np.mean(chance))
    print(f"  {'hit@argmax over all valid windows':<34s} {np.mean(hit):.3f}   "
          f"(chance {np.mean(chance):.3f})")

    # ---- controls
    print("\n--- controls (all must sit at chance) ---")
    rs = np.random.RandomState(13)
    res["query_shuffled"] = afc(wins_of(np.max, q_of=dict(zip(dvi, rs.permutation(qb[dvi])))),
                                pid[dvi], "query_shuffled")
    res["query_constant"] = afc(wins_of(np.max, q_of={j: 1 for j in dvi}), pid[dvi],
                                "query_constant")
    src = dict(zip(dvi, ci[dvi][rs.permutation(len(dvi))]))
    res["audio_shuffled"] = afc(wins_of(np.max, src=src), pid[dvi], "audio_shuffled")

    # ---- paired delta against H11's per-patch head on the same dev examples
    z = np.load(args.h11_preds, allow_pickle=True)
    assert list(z["pid"]) == list(pid[dvi]) and list(z["query_bit"]) == list(qb[dvi]) \
        and list(z["clip_idx"]) == list(ci[dvi]), \
        "dev examples do not match H11's; the paired delta would be meaningless"
    print(f"\npairing check: {len(z['pid'])} dev examples match "
          f"{os.path.basename(args.h11_preds)} in order")
    h11 = z["wins"][list(z["arms"]).index("intact")].mean(0)
    rep["h11_intact_afc"] = float(h11.mean())
    print("\npaired delta vs H11 per-patch head (same dev examples, patient bootstrap)")
    res["paired_delta"] = {}
    for k, v in [("max_over_windows", wins_of(np.max)),
                 ("oracle_single_window", np.array(orc, float))]:
        d = v - h11
        lo, hi = boot_ci(d, pid[dvi])
        res["paired_delta"][k] = {"delta": float(d.mean()), "ci": [lo, hi]}
        print(f"  {k:<24s} - h11 {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"excludes 0: {'yes' if (lo > 0 or hi < 0) else 'no'}")

    mx = res["max_over_windows"]["afc"]; orl = res["oracle_single_window"]["afc"]
    print()
    if mx >= 0.95:
        print(f"JOINT READING IS SUFFICIENT: max over windows reaches {mx:.3f} with "
              f"localisation no longer given away. The gap between H11's 0.569 and the "
              f"oracle's {orl:.3f} is neighbourhood reading, the region adapter is "
              f"motivated by evidence, and H12 proceeds as designed.")
    elif mx <= 0.60:
        print(f"THE BOTTLENECK IS SEARCH, NOT NEIGHBOURHOOD READING: the same crop "
              f"representation that scores {orl:.3f} at known locations falls to "
              f"{mx:.3f} once it has to find them. A region adapter addresses the wrong "
              f"half of the oracle gap. Per the pre-registration, do not add conv + "
              f"attention; redesign the loss or the candidate mechanism.")
    else:
        print(f"PARTIAL: max over windows {mx:.3f} against the oracle's {orl:.3f}. "
              f"Neither branch of the pre-registered dichotomy fires cleanly; both "
              f"joint reading and search carry part of the gap, and H12's design "
              f"question is not settled by this run.")

    res["report"] = rep
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
