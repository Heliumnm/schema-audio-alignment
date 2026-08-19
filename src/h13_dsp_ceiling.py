"""H13 pre-check — can a hand-written detector already solve the proposed task?

Before H13 is worth four to eight weeks, one number has to exist: what does a
query-conditioned DSP detector score on the multi-field pointing task? Three of the four
proposed fields — pitch, duration, intensity — are quantities a spectrogram exposes
directly. If DSP solves them, the aggregate metric is dominated by the easy fields, a
model clearing the gate would be matching a hand-written baseline, and the gate would not
be measuring what it claims to measure.

No training, no AST model, no GPU. Synthesis + filterbank + arithmetic.

The task, exactly as H13 proposes it
------------------------------------
Two events per clip. They share three field values and differ on exactly one; the
differing field rotates across trials; time positions and event order are randomised.
The query names the full combination:

    [character=harmonic] [pitch=high] [duration=short] [intensity=faint]

Both candidate windows are **identical in size**, so duration cannot be read off window
extent — it has to be measured from the audio inside the window, which is exactly what a
DSP detector is allowed to do.

What is reported
----------------
  per-field ceiling   the field's own detector, on trials where that field differs.
                      An oracle that is told which field carries the answer. This is the
                      upper bound for that field.
  combined            majority vote over all four detectors, told the query but NOT which
                      field differs. This is what the model actually faces, and it is the
                      number the H13 gate would be read against.
  temporal Hit@argmax the combined detector slid over every valid window position; does
                      the argmax land in the queried event's window? H13's other primary.
  query_shuffled      the same detectors against a permuted query. Must sit at chance,
                      otherwise the 2AFC is not measuring query conditioning.

Reading it
----------
  per-field ceiling near 1.000 on pitch / duration / intensity
        -> the aggregate is carried by fields a filter already solves, and H13 as
           currently drafted cannot support a claim about text pointing at spectral
           evidence. Redesign the field set before spending the weeks.
  combined near chance, character near v2.1's 0.549
        -> the task is not DSP-solvable and H13's ceiling is worth competing for.

    python src/h13_dsp_ceiling.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_v21 import SR, N_MELS, MEL_MAX, mel_row, admissible_fc
from grounding_train import boot_ci

# ---- the proposed field set, frozen here so the check and the design cannot drift
PITCH_BANDS = {"low": (300.0, 500.0), "mid": (500.0, 800.0), "high": (800.0, 1200.0)}
DURATIONS = {"short": (0.20, 0.28), "long": (0.45, 0.60)}
INTENSITIES = {"faint": (2.0, 4.0), "moderate": (8.0, 11.0)}
CHARACTERS = ("harmonic", "inharmonic")
FIELDS = ("character", "pitch", "duration", "intensity")
VALUES = {"character": list(CHARACTERS), "pitch": list(PITCH_BANDS),
          "duration": list(DURATIONS), "intensity": list(INTENSITIES)}

HARM_RATIOS = np.array([1.0, 2.0, 4.0])
INHARM_RATIOS = [np.array([1.0, 2.0 * 2 ** -0.368, 4.0]),
                 np.array([1.0, 2.0 * 2 ** +0.368, 4.0])]
FPS = 100.0                      # AST: 10 ms hop
WIN = 80                         # fixed candidate window, frames. Covers the longest event.


def hz2bin(f):
    return 2595 * np.log10(1 + np.asarray(f, float) / 700.0) / MEL_MAX * N_MELS


def bin2hz(b):
    return 700.0 * (10 ** (np.asarray(b, float) / N_MELS * MEL_MAX / 2595) - 1)


def synth(y, onset_s, dur_s, fc, character, variant, snr_db, rng):
    n0, n1 = int(onset_s * SR), min(int((onset_s + dur_s) * SR), len(y))
    if n1 - n0 < int(0.08 * SR):
        return None
    t = np.arange(n1 - n0) / SR
    ratios = HARM_RATIOS if character == "harmonic" else INHARM_RATIOS[variant]
    sig = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in fc * ratios)
    ramp = int(0.02 * SR)
    env = np.ones_like(sig)
    if len(env) > 2 * ramp:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        env[:ramp], env[-ramp:] = w, w[::-1]
    sig = sig * env
    local = np.sqrt(np.mean(y[n0:n1] ** 2)) + 1e-8
    sig = sig * local * (10 ** (snr_db / 20)) / (np.sqrt(np.mean(sig ** 2)) + 1e-8)
    y[n0:n1] += sig
    return fc * ratios


def draw_combo(rng):
    return {f: VALUES[f][rng.randint(len(VALUES[f]))] for f in FIELDS}


def realise(combo, rng, fc_pool):
    """Field values -> concrete acoustic parameters, sampled inside each value's range."""
    lo, hi = PITCH_BANDS[combo["pitch"]]
    band = fc_pool[(fc_pool >= lo) & (fc_pool < hi)]
    d0, d1 = DURATIONS[combo["duration"]]
    s0, s1 = INTENSITIES[combo["intensity"]]
    return {"fc": float(rng.choice(band)), "dur": float(rng.uniform(d0, d1)),
            "snr": float(rng.uniform(s0, s1)),
            "variant": int(rng.randint(2)) if combo["character"] == "inharmonic" else -1}


def build(y, dur_s, rng, fc_pool):
    """Two events sharing three fields and differing on exactly one. Order randomised."""
    diff = FIELDS[rng.randint(len(FIELDS))]
    a = draw_combo(rng)
    b = dict(a)
    alt = [v for v in VALUES[diff] if v != a[diff]]
    b[diff] = alt[rng.randint(len(alt))]
    combos = [a, b]
    pars = [realise(c, rng, fc_pool) for c in combos]

    half = WIN / FPS / 2
    c0 = rng.uniform(half, dur_s / 2 - half * 0.2)
    c1 = rng.uniform(dur_s / 2 + half * 0.2, dur_s - half)
    if c1 - c0 < WIN / FPS:
        return None
    centres = [c0, c1]
    if rng.rand() < 0.5:                        # which combination sits first
        combos, pars = combos[::-1], pars[::-1]
    ev = []
    for combo, p, c in zip(combos, pars, centres):
        on = c - p["dur"] / 2
        if on < 0 or on + p["dur"] > dur_s:
            return None
        ev.append(dict(combo=combo, centre=c, on=on, **p))
    return diff, ev


# ---------- detectors. Each is told the queried value and scores one window. ----------

def excess(FB, t0, bg):
    w = FB[t0:t0 + WIN]
    return np.maximum(w - bg[None, :], 0.0)


def est_fc(E):
    """Lowest strong partial in the pitch range -> fc."""
    b0, b1 = int(np.floor(hz2bin(280.0))), int(np.ceil(hz2bin(1300.0)))
    prof = E[:, b0:b1].mean(0)
    if prof.max() <= 0:
        return None
    return float(bin2hz(b0 + int(np.argmax(prof))))


def est_active(E):
    """Frames carrying the event, from broadband excess -> duration in seconds."""
    b0, b1 = int(np.floor(hz2bin(250.0))), int(np.ceil(hz2bin(5000.0)))
    e = E[:, b0:b1].mean(1)
    if e.max() <= 0:
        return 0.0, e
    return float((e > 0.5 * e.max()).sum()) / FPS, e


def score_pitch(E, val):
    f = est_fc(E)
    if f is None:
        return 0.0
    lo, hi = PITCH_BANDS[val]
    return -abs(np.log2(max(f, 1.0)) - np.log2(np.sqrt(lo * hi)))


def score_duration(E, val):
    d, _ = est_active(E)
    lo, hi = DURATIONS[val]
    return -abs(d - (lo + hi) / 2)


def score_intensity(E, val):
    _, e = est_active(E)
    lo, hi = INTENSITIES[val]
    # peak excess in dB-like units; the filterbank is already logarithmic
    return -abs(float(e.max()) - {"faint": 1.0, "moderate": 2.2}[val])


def score_character(E, val):
    """v2.1's detector: the middle partial's normalised log-frequency position between the
    endpoints. Harmonic sits at 0.5, inharmonic at 0.316 or 0.684."""
    f = est_fc(E)
    if f is None:
        return 0.0
    b_lo, b_hi = int(np.floor(hz2bin(f))), int(np.ceil(hz2bin(4 * f)))
    if b_hi - b_lo < 4 or b_hi >= N_MELS:
        return 0.0
    prof = E.mean(0)
    inner = np.arange(b_lo + 2, min(b_hi - 1, N_MELS))
    if len(inner) < 2:
        return 0.0
    pk = float(bin2hz(inner[int(np.argmax(prof[inner]))]))
    p = np.clip((np.log(max(pk, 1.0)) - np.log(f)) / (np.log(4 * f) - np.log(f)), 0, 1)
    return -abs(p - 0.5) if val == "harmonic" else abs(p - 0.5)


SCORERS = {"character": score_character, "pitch": score_pitch,
           "duration": score_duration, "intensity": score_intensity}


def main():
    import soundfile as sf, librosa
    from transformers import AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--out", default="results/h13_dsp_ceiling.json")
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    # TRAIN SPLIT ONLY. Two events plus margins need a longer cycle than v2.1 required.
    pool = [e for e in seg if e["label"] == "normal" and e.get("duration", 0) >= 2.0
            and smap.get(os.path.basename(e["path"])) == "train"]
    rng.shuffle(pool); pool = pool[:args.n]
    fc_pool = admissible_fc()
    for v, (lo, hi) in PITCH_BANDS.items():
        n = int(((fc_pool >= lo) & (fc_pool < hi)).sum())
        rows = sorted({mel_row(f) for f in fc_pool[(fc_pool >= lo) & (fc_pool < hi)]})
        print(f"pitch={v:<9s} {n:4d} admissible fc, mel rows {rows}")
        assert n > 0, f"pitch band {v} is empty on the admissible set"

    rows_, skipped = [], 0
    for e in pool:
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        r = build(y, len(y) / SR, rng, fc_pool)
        if r is None:
            skipped += 1; continue
        diff, ev = r
        yi = y.copy()
        ok = True
        for v in ev:
            f = synth(yi, v["on"], v["dur"], v["fc"], v["combo"]["character"],
                      v["variant"], v["snr"], rng)
            if f is None: ok = False; break
            v["freqs"] = f
        if not ok:
            skipped += 1; continue
        yi = yi / (np.abs(yi).max() + 1e-8)
        rows_.append({"y": yi, "diff": diff, "ev": ev,
                      "pid": os.path.basename(p).split("_")[0]})
    print(f"\nbuilt {len(rows_)} clips, {len(set(r['pid'] for r in rows_))} patients "
          f"({skipped} skipped for length)")

    # ---- matching checks: the design's own claims, measured
    print("\n--- matching checks ---")
    dc = {f: sum(r["diff"] == f for r in rows_) for f in FIELDS}
    print("  differing field: " + "  ".join(f"{k} {v}" for k, v in dc.items()))
    for f in FIELDS:
        first = np.mean([r["ev"][0]["combo"][f] == VALUES[f][0] for r in rows_])
        print(f"  {f:<10s} first-slot value[0] rate {first:.3f}  (must be ~"
              f"{1/len(VALUES[f]):.3f}–0.5, order is randomised)")

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    FB = []
    for i in range(0, len(rows_), 16):
        inp = fe([r["y"] for r in rows_[i:i+16]], sampling_rate=SR, return_tensors="np")
        FB.append(inp["input_values"].astype(np.float32))
    FB = np.concatenate(FB)
    print(f"filterbanks {FB.shape}")

    # ---- score both candidate windows, per field
    per_field, combined, qs_combined, hits, chances, pids = {f: [] for f in FIELDS}, [], [], [], [], []
    rs = np.random.RandomState(3)
    shuffled = [rows_[i]["ev"] for i in rs.permutation(len(rows_))]
    for k, r in enumerate(rows_):
        fb = FB[k]
        nvalid = min(int(len(r["y"]) / SR * FPS), len(fb))
        bg = np.median(fb[:nvalid], axis=0)
        q = rs.randint(2)                              # which event the query names
        qcombo = r["ev"][q]["combo"]
        E = [excess(fb, int(v["centre"] * FPS) - WIN // 2, bg) for v in r["ev"]]

        votes = 0
        for f in FIELDS:
            a, b = SCORERS[f](E[q], qcombo[f]), SCORERS[f](E[1 - q], qcombo[f])
            win = rs.rand() < 0.5 if abs(a - b) < 1e-12 else a > b
            votes += int(win)
            if r["diff"] == f:
                per_field[f].append(float(win))
        combined.append(float(votes > 2 if votes != 2 else rs.rand() < 0.5))

        # query_shuffled: another clip's query against this clip's audio
        sq = shuffled[k][rs.randint(2)]["combo"]
        v2 = sum(int(SCORERS[f](E[q], sq[f]) > SCORERS[f](E[1 - q], sq[f])) for f in FIELDS)
        qs_combined.append(float(v2 > 2 if v2 != 2 else rs.rand() < 0.5))

        # temporal Hit@argmax: slide the combined detector over every valid window
        best, arg = -1e18, 0
        for t0 in range(0, max(1, nvalid - WIN), 5):
            Ew = excess(fb, t0, bg)
            s = sum(SCORERS[f](Ew, qcombo[f]) for f in FIELDS)
            if s > best: best, arg = s, t0
        tgt = int(r["ev"][q]["centre"] * FPS) - WIN // 2
        hits.append(float(abs(arg - tgt) <= WIN // 2))
        chances.append(WIN / max(nvalid - WIN, 1))
        pids.append(r["pid"])

    pids = np.array(pids)
    res = {"n_clips": len(rows_), "window_frames": WIN,
           "differing_field_counts": dc}
    print("\n--- per-field ceiling (that field's own detector, on trials where it differs) ---")
    print(f"{'field':<12s}{'n':>6s}{'2AFC':>9s}   95% CI")
    for f in FIELDS:
        w = np.array(per_field[f])
        idx = [i for i, r in enumerate(rows_) if r["diff"] == f]
        lo, hi = boot_ci(w, pids[idx])
        res[f"ceiling_{f}"] = {"afc": float(w.mean()), "n": len(w), "ci": [lo, hi]}
        print(f"{f:<12s}{len(w):6d}{w.mean():9.3f}   [{lo:.3f}, {hi:.3f}]")

    cw = np.array(combined); qw = np.array(qs_combined); hw = np.array(hits)
    lo, hi = boot_ci(cw, pids); res["combined"] = {"afc": float(cw.mean()), "ci": [lo, hi]}
    print(f"\n{'combined (query, not told which field differs)':<48s}"
          f"{cw.mean():7.3f}   [{lo:.3f}, {hi:.3f}]")
    lo2, hi2 = boot_ci(qw, pids); res["query_shuffled"] = {"afc": float(qw.mean()), "ci": [lo2, hi2]}
    print(f"{'query_shuffled (must be at chance)':<48s}{qw.mean():7.3f}   [{lo2:.3f}, {hi2:.3f}]")
    lo3, hi3 = boot_ci(hw, pids)
    res["temporal_hit_argmax"] = {"hit": float(hw.mean()), "ci": [lo3, hi3],
                                  "chance": float(np.mean(chances))}
    print(f"{'temporal Hit@argmax (combined detector)':<48s}{hw.mean():7.3f}   "
          f"[{lo3:.3f}, {hi3:.3f}]   chance {np.mean(chances):.3f}")

    easy = [f for f in ("pitch", "duration", "intensity") if res[f"ceiling_{f}"]["afc"] >= 0.90]
    print()
    if easy:
        print(f"DSP ALREADY SOLVES {', '.join(easy)} (>= 0.900). Those fields are directly "
              f"readable from the spectrogram, so the aggregate metric is carried by them "
              f"and a model clearing the H13 gate would be matching a hand-written "
              f"baseline. Redesign the field set before spending the weeks.")
    elif res["combined"]["ci"][0] <= 0.5:
        print(f"DSP is at chance on the combined task ({cw.mean():.3f}). The ceiling is "
              f"worth competing for and H13's field set survives this check.")
    else:
        print(f"PARTIAL: combined DSP reaches {cw.mean():.3f} with no single field above "
              f"0.900. Report the per-field breakdown and decide field by field.")
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
