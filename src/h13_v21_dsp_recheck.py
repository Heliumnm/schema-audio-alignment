"""Did the grounding line ever beat a hand-written detector?

On the H13 pre-check, a crude character detector scored 0.948 on the same harmonic /
inharmonic distinction H11 and H12 were built on. v2.1's own DSP arm scored 0.549 there,
which is why the H11 write-up recorded that "the DSP arm's CI covers 0.5, so it never
functioned as a ceiling". The two detectors differ in one thing: this one subtracts a
per-bin background median before profiling, which removes the breath sound's spectral
tilt.

If that single change carries the difference on the **frozen v2.1 data**, then H11's 0.569
and H12 Stage 1A's 0.580 were measured against a baseline nobody had measured properly,
and the grounding line never competed with a hand-written detector at all.

Three variants, decomposing the change one step at a time, all on the same 452 dev
examples so every comparison is paired:

  A  v2.1 as published    tight event window, oracle endpoint frequencies, raw filterbank
  B  A + background       the only change is subtracting the per-bin background median
  C  fully audio-driven   B, but fc estimated from the audio instead of handed over.
                          This calls h13_dsp_ceiling.score_character verbatim, so it is
                          the exact detector that scored 0.948, not a re-tuned one.

A must reproduce `dsp_query_cond` from grounding_v21_preds.npz, or the harness differs and
nothing below is comparable.

No training, no AST model, no GPU.

    python src/h13_v21_dsp_recheck.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_v21 import SR, N_MELS, MEL_MAX, CHARS, admissible_fc, build_clip
from grounding_train import dsp_scores, boot_ci
from h13_dsp_ceiling import score_character, est_fc, hz2bin, bin2hz

CHAR_NAME = {"harm": "harmonic", "inharm": "inharmonic"}


def excess_window(FB, e, bg):
    t0, t1 = int(e["on"] * 100), int((e["on"] + e["dur"]) * 100)
    seg = FB[max(t0, 0):max(t1, t0 + 1), :]
    return np.maximum(seg - bg[None, :], 0.0)


def char_p_oracle(E, freqs):
    """Variant B: identical to the published detector except that it profiles the
    background-subtracted excess. Endpoints are still handed over."""
    f_lo, f_hi = float(min(freqs)), float(max(freqs))
    b_lo, b_hi = int(np.floor(hz2bin(f_lo))), int(np.ceil(hz2bin(f_hi)))
    if not len(E) or b_hi - b_lo < 4:
        return 0.5
    prof = E.mean(0)
    inner = np.arange(b_lo + 2, min(b_hi - 1, N_MELS))
    if len(inner) < 2:
        return 0.5
    f_pk = float(bin2hz(inner[int(np.argmax(prof[inner]))]))
    return float(np.clip((np.log(max(f_pk, 1.0)) - np.log(f_lo)) /
                         (np.log(f_hi) - np.log(f_lo)), 0.0, 1.0))


def p_to_win(pq, pd, q):
    """q = 0 means the query names the harmonic event, which sits at p = 0.5."""
    sq = -abs(pq - 0.5) if q == 0 else abs(pq - 0.5)
    sd = -abs(pd - 0.5) if q == 0 else abs(pd - 0.5)
    return float(sq > sd)


def main():
    import soundfile as sf, librosa
    from transformers import AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--frozen", default="results/grounding_v21_manifest.json")
    ap.add_argument("--h11_preds", default="results/grounding_v21_preds.npz")
    ap.add_argument("--h12_preds", default="results/h12_stage1a_preds.npz")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--out", default="results/h13_v21_dsp_recheck.json")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
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
        rows.append({"y": yi, "ev": ev, "occ": occ, "sid": os.path.basename(p),
                     "pid": os.path.basename(p).split("_")[0]})
    man = json.load(open(args.frozen))
    assert len(man) == len(rows) and all(
        r["sid"] == f["sid"] and abs(r["ev"][c]["on"] - f["events"][c]["onset_s"]) < 1e-9
        for r, f in zip(rows, man) for c in CHARS), "synthesis differs from the frozen v2.1"
    print(f"{len(rows)} clips reproduce {os.path.basename(args.frozen)}")

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    FBs = []
    for i in range(0, len(rows), 16):
        inp = fe([r["y"] for r in rows[i:i+16]], sampling_rate=SR, return_tensors="np")
        FBs.append(inp["input_values"].astype(np.float32))
    FB = np.concatenate(FBs); del FBs
    print(f"filterbanks {FB.shape}")

    ci = np.repeat(np.arange(len(rows)), 2)
    qb = np.tile([0, 1], len(rows))
    pid = np.array([rows[c]["pid"] for c in ci])
    z = np.load(args.h11_preds, allow_pickle=True)
    # the dev examples are H11's, identified by matching its stored index arrays
    key = {(int(c), int(q)): i for i, (c, q) in enumerate(zip(ci, qb))}
    dvi = np.array([key[(int(c), int(q))] for c, q in zip(z["clip_idx"], z["query_bit"])])
    assert list(pid[dvi]) == list(z["pid"]), "dev examples do not match H11's"
    print(f"pairing check: {len(dvi)} dev examples match "
          f"{os.path.basename(args.h11_preds)} in order")

    A, B, C = [], [], []
    for j in dvi:
        c, q = int(ci[j]), int(qb[j])
        r = rows[c]; fb = FB[c]
        nvalid = min(int(len(r["y"]) / SR * 100), len(fb))
        bg = np.median(fb[:nvalid], axis=0)
        eq, ed = r["ev"][CHARS[q]], r["ev"][CHARS[1 - q]]
        # A — published detector, raw filterbank, oracle endpoints
        A.append(p_to_win(dsp_scores(fb, r["ev"], CHARS[q]),
                          dsp_scores(fb, r["ev"], CHARS[1 - q]), q))
        # B — same, on the background-subtracted excess
        Eq, Ed = excess_window(fb, eq, bg), excess_window(fb, ed, bg)
        B.append(p_to_win(char_p_oracle(Eq, eq["freqs"]),
                          char_p_oracle(Ed, ed["freqs"]), q))
        # C — the H13 detector verbatim: fc estimated from the audio
        val = CHAR_NAME[CHARS[q]]
        C.append(float(score_character(Eq, val) > score_character(Ed, val)))
    A, B, C = np.array(A), np.array(B), np.array(C)

    dsp0 = z["wins"][list(z["arms"]).index("dsp_query_cond")][0]
    mism = int((A != dsp0).sum())
    print(f"\nvariant A vs published dsp_query_cond: {mism} / {len(A)} differ "
          f"({A.mean():.4f} vs {dsp0.mean():.4f})")
    assert mism == 0, "variant A does not reproduce the published DSP arm"
    print("PASS — the harness is identical, so B and C differ only where stated.")

    h11 = z["wins"][list(z["arms"]).index("intact")].mean(0)
    zz = np.load(args.h12_preds, allow_pickle=True)
    assert list(zz["pid"]) == list(z["pid"]), "H12 dev examples do not match"
    h12 = zz["wins"][list(zz["arms"]).index("region_mass:intact")].mean(0)

    res, rows_out = {}, [
        ("A  published DSP (raw filterbank)", A),
        ("B  + background subtraction", B),
        ("C  + fc estimated from audio", C),
        ("H11 per-patch head", h11),
        ("H12 region-mass head", h12)]
    print(f"\n{'arm':<36s}{'2AFC':>8s}   95% CI")
    for name, w in rows_out:
        lo, hi = boot_ci(w, pid[dvi])
        res[name.strip()] = {"afc": float(w.mean()), "ci": [lo, hi]}
        print(f"{name:<36s}{w.mean():8.3f}   [{lo:.3f}, {hi:.3f}]")

    print("\npaired deltas (same dev examples, patient-cluster bootstrap)")
    res["paired"] = {}
    for n1, w1 in [("B", B), ("C", C)]:
        for n2, w2 in [("A", A), ("H11", h11), ("H12", h12)]:
            d = w1 - w2; lo, hi = boot_ci(d, pid[dvi])
            res["paired"][f"{n1} - {n2}"] = {"delta": float(d.mean()), "ci": [lo, hi]}
            print(f"  {n1} - {n2:<6s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"excludes 0: {'yes' if (lo > 0 or hi < 0) else 'no'}")

    print()
    best = max(B.mean(), C.mean())
    if best > max(h11.mean(), h12.mean()) and res["paired"]["B - H11"]["ci"][0] > 0:
        print(f"THE GROUNDING LINE NEVER BEAT A HAND-WRITTEN DETECTOR. On the frozen v2.1 "
              f"data, subtracting a per-bin background median takes the same published "
              f"detector from {A.mean():.3f} to {B.mean():.3f}, above H11's "
              f"{h11.mean():.3f} and H12's {h12.mean():.3f}. The published DSP arm was "
              f"weak, not the task, and 'the DSP arm never functioned as a ceiling' has "
              f"to be retracted.")
    elif best <= h11.mean():
        print(f"THE PUBLISHED BASELINE HOLDS. Background subtraction moves DSP from "
              f"{A.mean():.3f} to {B.mean():.3f} / {C.mean():.3f}, still not above H11's "
              f"{h11.mean():.3f}. The H13 pre-check's 0.948 came from the new synthesis "
              f"(longer events, fixed windows), not from the detector, and H11's ceiling "
              f"claim stands.")
    else:
        print(f"PARTIAL: background subtraction moves DSP to {B.mean():.3f} / "
              f"{C.mean():.3f} against H11's {h11.mean():.3f}. Read the paired CIs before "
              f"concluding either way.")
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
