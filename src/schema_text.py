"""
Provenance-controlled schema -> text generation for audio-text contrastive alignment.

Builds the structured clinical schema for each ICBHI respiratory cycle, tags every
field with its PROVENANCE, then emits one natural-language text per provenance
condition. The provenance conditions are the paper's core independent variable:
they separate alignment signal that is genuinely independent of the audio from
signal that is a deterministic (or learned) function of it.

    dataset : recording metadata only (chest location, device, duration, condition)
              -> recorded by humans at collection time, INDEPENDENT of the signal
    signal  : dataset + deterministic signal measurements (band energy, transient
              sharpness, peak count, pitch, intensity)  -> derived from audio
    model   : dataset + a learned model's estimates (OPERA-CT crackle/wheeze
              likelihood)                                -> most circular
    all     : everything

CRITICAL: `expert_annotation` (the crackle/wheeze ground truth) is stored in the
schema for bookkeeping but is NEVER rendered into any text. Putting the target
label into the text would turn contrastive alignment into disguised supervision
and invalidate every downstream number.

Usage:
    python src/schema_text.py \
        --manifest data/segments/manifest.json \
        --feats results/opera_feats_clean.npy --index results/opera_index_clean.json \
        --crackle_probe results/judges/judge_A_crackle.pt \
        --wheeze_probe  results/judges/judge_A_wheeze.pt \
        --out results/schema_text.json --explain --show 2
"""
import os, json, argparse
import numpy as np
import torch, torch.nn as nn
import soundfile as sf
import librosa
from scipy.signal import find_peaks
from scipy.stats import kurtosis

SR = 16000; N_FFT = 1024; HOP = 256
WHEEZE_BAND = (550, 2400)
LOC = {"A": "anterior", "P": "posterior", "L": "lateral", "T": "tracheal"}
SIDE = {"l": "left", "r": "right", "c": "central"}

# provenance tag -> which conditions include it
PROVENANCE = {
    "dataset":     {"dataset", "signal", "model", "all"},
    "judge_d":     {"signal", "all"},
    "signal-proxy": {"signal", "all"},
    "opera-ct":    {"model", "all"},
}
CONDITIONS = ["dataset", "signal", "model", "all"]


class LinearProbe(nn.Module):
    def __init__(self, d): super().__init__(); self.fc = nn.Linear(d, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)


# ---------- signal-level cue extraction ----------
def extract_cues(y, sr):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    total = P.sum() + 1e-10

    hf = S[freqs >= 2000, :].sum(axis=0)
    hf_kurt = float(kurtosis(hf / (np.max(hf) + 1e-10))) if np.max(hf) > 1e-10 else 0.0
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    top = np.argsort(flux)[-max(1, len(flux) // 5):]
    burst = S[:, top].mean(axis=1)
    transient_pitch = float((freqs * burst).sum() / (burst.sum() + 1e-10))

    wb = (freqs >= WHEEZE_BAND[0]) & (freqs <= WHEEZE_BAND[1])
    wheeze_band = float(P[wb, :].sum() / total)
    mean_spec = S.mean(axis=1)
    band = (freqs >= 200) & (freqs <= 2000)
    bspec = mean_spec[band]
    if bspec.max() > 1e-9:
        pk, _ = find_peaks(bspec / bspec.max(), height=0.35, distance=5)
        n_peaks, dom_pitch = int(len(pk)), float(freqs[band][np.argmax(bspec)])
    else:
        n_peaks, dom_pitch = 0, 0.0

    return {
        "hf_kurtosis": hf_kurt, "transient_pitch": transient_pitch,
        "wheeze_band": wheeze_band, "wheeze_n_peaks": n_peaks, "wheeze_pitch": dom_pitch,
        "breath_intensity": float(np.sqrt(np.mean(y ** 2))),
    }


# ---------- descriptors ----------
def tertiles(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)])
    return float(np.percentile(a, 33)), float(np.percentile(a, 66))


def lvl(v, t):
    return "low" if v < t[0] else ("high" if v > t[1] else "medium")


LIKELIHOOD = {"low": "a low likelihood of", "medium": "a moderate likelihood of",
              "high": "a high likelihood of"}
PITCH = {"low": "low-pitched", "medium": "medium-pitched", "high": "high-pitched"}
SHARP = {"low": "low", "medium": "moderate", "high": "high"}
INTENSITY = {"low": "diminished", "medium": "normal", "high": "increased"}

CUE_EXPLAIN = {
    "crackle_likelihood": {
        "what": "Crackle likelihood — a frozen respiratory foundation model's estimate "
                "that this cycle contains crackles (discontinuous, explosive sounds).",
        "low": "low: the detector does not favour crackles.",
        "medium": "moderate: the detector is uncertain.",
        "high": "high: the detector strongly favours crackles, which point to "
                "alveolar/airway disease (e.g. pneumonia, fibrosis, bronchiectasis).",
        "caveat": "This is a model estimate, not a label — weigh it against the audio.",
    },
    "transient_sharpness": {
        "what": "Transient sharpness — how peaky and time-localised the high-frequency "
                "(>2 kHz) energy bursts are.",
        "low": "low: energy is smooth over time, unlike sharp crackle bursts.",
        "medium": "moderate.",
        "high": "high: sharp, well-localised bursts, the signature of crackles; "
                "higher-pitched bursts suggest fine crackles, lower coarse.",
        "caveat": "Sharp bursts can also come from friction/artefact, not only crackles.",
    },
    "wheeze_likelihood": {
        "what": "Wheeze likelihood — the foundation model's estimate that this cycle "
                "contains wheezes (continuous, musical sounds).",
        "low": "low: the detector does not favour wheeze.",
        "medium": "moderate: the detector is uncertain.",
        "high": "high: the detector favours wheeze, which points to airway narrowing "
                "(e.g. asthma, COPD).",
        "caveat": "A model estimate; low-pitched wheeze can be missed.",
    },
    "musical_band_energy": {
        "what": "Musical-band energy — the fraction of energy in the 550–2400 Hz band "
                "where wheeze's tonal component lives.",
        "low": "low: little tonal energy in the wheeze band.",
        "medium": "moderate tonal energy.",
        "high": "high: strong tonal energy consistent with a musical wheeze.",
        "caveat": "Normal breath and background tones also add energy here.",
    },
    "wheeze_type": {
        "monophonic": "monophonic (a single tonal peak → one narrowed airway).",
        "polyphonic": "polyphonic (several simultaneous tones → diffuse airway "
                      "narrowing, more typical of COPD/asthma).",
    },
    "wheeze_pitch": {
        "low": "low-pitched.", "medium": "medium-pitched.",
        "high": "high-pitched (higher pitch suggests tighter airway narrowing).",
    },
    "breath_intensity": {
        "what": "Breath-sound intensity — overall loudness of the cycle.",
        "diminished": "diminished: quiet breath sounds, which can indicate reduced "
                      "air entry (effusion, pneumothorax, severe obstruction).",
        "normal": "normal loudness.",
        "increased": "increased loudness.",
        "caveat": "Intensity also depends on recording gain and body habitus.",
    },
}


# ---------- schema ----------
def parse_loc(code):
    return {"region": LOC.get(code[:1], "unknown"),
            "side": SIDE.get(code[1:2].lower(), "unspecified") if len(code) > 1 else "unspecified",
            "code": code}


def build_schema(e, cues, thr):
    parts = e["filename"].split("_")
    # ICBHI names are 101_1b1_Al_sc_Meditron.wav -> field 4 carries the extension
    dev = os.path.splitext(parts[4])[0] if len(parts) > 4 else "unknown"
    crackle_char = "fine" if lvl(cues["transient_pitch"], thr["transient_pitch"]) == "high" else "coarse"
    wheeze_type = ("polyphonic" if cues["wheeze_n_peaks"] >= 2
                   else ("monophonic" if cues["wheeze_n_peaks"] == 1 else "none"))
    return {
        "segment_id": e["filename"],
        "recording": {
            "chest_location": {"value": parse_loc(parts[2]), "source": "dataset"},
            "device": {"value": dev, "source": "dataset"},
            "duration_s": {"value": round(e.get("duration", 0), 2), "source": "dataset"},
            "signal_quality": {"value": e.get("condition", "clean"), "source": "dataset"},
        },
        "adventitious_sounds": {
            "crackle": {
                "likelihood": {"level": lvl(cues["crackle_conf"], thr["crackle_conf"]), "source": "opera-ct"},
                "transient_sharpness": {"level": lvl(cues["hf_kurtosis"], thr["hf_kurtosis"]), "source": "judge_d"},
                "character": {"value": crackle_char, "proxy": "transient_pitch", "source": "signal-proxy"},
            },
            "wheeze": {
                "likelihood": {"level": lvl(cues["wheeze_conf"], thr["wheeze_conf"]), "source": "opera-ct"},
                "musical_band_energy": {"level": lvl(cues["wheeze_band"], thr["wheeze_band"]),
                                        "band_hz": list(WHEEZE_BAND), "source": "judge_d"},
                "type": {"value": wheeze_type, "proxy": "spectral_peak_count", "source": "signal-proxy"},
                "pitch": {"level": lvl(cues["wheeze_pitch"], thr["wheeze_pitch"]), "source": "signal-proxy"},
            },
        },
        "breath_sound": {
            "intensity": {"level": lvl(cues["breath_intensity"], thr["breath_intensity"]), "source": "signal-proxy"},
        },
        # bookkeeping ONLY — never rendered into text (see module docstring)
        "expert_annotation": {"crackle": bool(e["crackle"]), "wheeze": bool(e["wheeze"]), "label": e["label"]},
    }


def keep(field, condition):
    """Is this schema field visible under the given provenance condition?"""
    return field is not None and condition in PROVENANCE.get(field.get("source", ""), set())


# ---------- verbalizers (provenance-aware; degrade gracefully) ----------
def verbalize(s, condition="all"):
    rec, cr, wh = s["recording"], s["adventitious_sounds"]["crackle"], s["adventitious_sounds"]["wheeze"]
    parts = []

    if keep(rec["chest_location"], condition):
        loc = rec["chest_location"]["value"]
        site = f"{loc['region']} {loc['side']}".strip()
        dev = rec["device"]["value"] if keep(rec["device"], condition) else "unspecified"
        qual = rec["signal_quality"]["value"] if keep(rec["signal_quality"], condition) else "unspecified"
        parts.append(f"Auscultation at the {site} chest ({dev} stethoscope, {qual} recording).")

    # crackle
    has_cl, has_ts = keep(cr["likelihood"], condition), keep(cr["transient_sharpness"], condition)
    if has_cl and has_ts:
        cl = cr["likelihood"]["level"]
        if cl == "low":
            parts.append(f"There is {LIKELIHOOD[cl]} crackles; transient sharpness is "
                         f"{SHARP[cr['transient_sharpness']['level']]}.")
        else:
            char = (f" suggesting {cr['character']['value']} crackles"
                    if keep(cr["character"], condition) else "")
            parts.append(f"There is {LIKELIHOOD[cl]} crackles, with "
                         f"{SHARP[cr['transient_sharpness']['level']]} transient sharpness{char}.")
    elif has_cl:
        parts.append(f"There is {LIKELIHOOD[cr['likelihood']['level']]} crackles.")
    elif has_ts:
        char = (f", suggesting {cr['character']['value']} crackles"
                if keep(cr["character"], condition) else "")
        parts.append(f"Transient sharpness is {SHARP[cr['transient_sharpness']['level']]}{char}.")

    # wheeze
    has_wl, has_mb = keep(wh["likelihood"], condition), keep(wh["musical_band_energy"], condition)
    wtype = wh["type"]["value"] if keep(wh["type"], condition) else "none"
    char = (f"{wtype}, {PITCH[wh['pitch']['level']]}, "
            if wtype in ("monophonic", "polyphonic") and keep(wh["pitch"], condition) else "")
    if has_wl and has_mb:
        parts.append(f"There is {LIKELIHOOD[wh['likelihood']['level']]} wheeze, {char}with "
                     f"{wh['musical_band_energy']['level']} musical-band energy.")
    elif has_wl:
        parts.append(f"There is {LIKELIHOOD[wh['likelihood']['level']]} wheeze{', ' + char.rstrip(', ') if char else ''}.")
    elif has_mb:
        parts.append(f"Musical-band energy is {wh['musical_band_energy']['level']}"
                     f"{', with ' + char.rstrip(', ') + ' character' if char else ''}.")

    if keep(s["breath_sound"]["intensity"], condition):
        parts.append(f"Breath-sound intensity is {INTENSITY[s['breath_sound']['intensity']['level']]}.")

    return " ".join(parts) if parts else "No auscultation cues available for this recording."


def verbalize_explained(s, condition="all"):
    """Long-form: every visible cue carries meaning + clinical direction + caveat."""
    rec, cr, wh = s["recording"], s["adventitious_sounds"]["crackle"], s["adventitious_sounds"]["wheeze"]
    E, lines = CUE_EXPLAIN, []

    if keep(rec["chest_location"], condition):
        loc = rec["chest_location"]["value"]
        site = f"{loc['region']} {loc['side']}".strip()
        dev = rec["device"]["value"] if keep(rec["device"], condition) else "an unspecified"
        qual = rec["signal_quality"]["value"] if keep(rec["signal_quality"], condition) else "unspecified"
        lines.append(f"Auscultation at the {site} chest, recorded with a {dev} stethoscope "
                     f"({qual} signal).")

    def cue(name, level):
        e = E[name]
        return f"- {e['what']} This recording is {e[level]} ({e['caveat']})"

    body = []
    cr_lines = []
    if keep(cr["likelihood"], condition):
        cr_lines.append(cue("crackle_likelihood", cr["likelihood"]["level"]))
    if keep(cr["transient_sharpness"], condition):
        cr_lines.append(cue("transient_sharpness", cr["transient_sharpness"]["level"]))
    if cr_lines:
        body += ["Crackle assessment:"] + cr_lines

    wh_lines = []
    if keep(wh["likelihood"], condition):
        wh_lines.append(cue("wheeze_likelihood", wh["likelihood"]["level"]))
    if keep(wh["musical_band_energy"], condition):
        wh_lines.append(cue("musical_band_energy", wh["musical_band_energy"]["level"]))
    wtype = wh["type"]["value"] if keep(wh["type"], condition) else "none"
    if wtype in E["wheeze_type"] and keep(wh["pitch"], condition):
        wh_lines.append(f"- Wheeze character: {E['wheeze_type'][wtype]} "
                        f"Pitch is {E['wheeze_pitch'][wh['pitch']['level']]}")
    if wh_lines:
        body += ["Wheeze assessment:"] + wh_lines

    if keep(s["breath_sound"]["intensity"], condition):
        body += ["Breath sounds:",
                 cue("breath_intensity", INTENSITY[s["breath_sound"]["intensity"]["level"]])]

    if body:
        lines.append("The cues below are automated decision-support signals graded relative "
                     "to the training distribution (low/medium/high), not absolute clinical "
                     "thresholds.")
        lines += body
    return "\n".join(lines) if lines else "No auscultation cues available for this recording."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feats", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--crackle_probe", required=True); ap.add_argument("--wheeze_probe", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--key", default="path")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--show", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = json.load(open(args.manifest))
    X = np.load(args.feats); idx = json.load(open(args.index))
    pc = LinearProbe(768).to(device)
    pc.load_state_dict(torch.load(args.crackle_probe, map_location=device, weights_only=True)); pc.eval()
    pw = LinearProbe(768).to(device)
    pw.load_state_dict(torch.load(args.wheeze_probe, map_location=device, weights_only=True)); pw.eval()

    recs = {}
    for i, e in enumerate(seg):
        path = e.get(args.key, e["path"])
        k = os.path.basename(path)
        try:
            y, sr = sf.read(path, dtype="float32")
            if y.ndim > 1: y = y.mean(axis=1)
        except Exception:
            continue
        cues = extract_cues(y, sr)
        row = idx.get(k)
        if row is not None:
            f = torch.tensor(X[row]).float().to(device)
            with torch.no_grad():
                cues["crackle_conf"] = float(torch.sigmoid(pc(f)))
                cues["wheeze_conf"] = float(torch.sigmoid(pw(f)))
        else:
            cues["crackle_conf"] = cues["wheeze_conf"] = 0.0
        recs[k] = {"entry": e, "cues": cues}
        if (i + 1) % 500 == 0: print(f"  {i+1}/{len(seg)}", flush=True)

    # tertiles from TRAIN only — no test information enters the descriptors
    thr = {c: tertiles([r["cues"][c] for r in recs.values() if r["entry"]["split"] == "train"])
           for c in ["crackle_conf", "wheeze_conf", "wheeze_pitch",
                     "hf_kurtosis", "transient_pitch", "wheeze_band", "breath_intensity"]}

    out = {}
    for k, r in recs.items():
        sch = build_schema(r["entry"], r["cues"], thr)
        rec = {"schema": sch, "split": r["entry"]["split"], "label": r["entry"]["label"],
               "text": {c: verbalize(sch, c) for c in args.conditions}}
        if args.explain:
            rec["text_explained"] = {c: verbalize_explained(sch, c) for c in args.conditions}
        out[k] = rec

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nBuilt {len(out)} records x {len(args.conditions)} provenance conditions -> {args.out}")

    # sanity: the ground-truth label must never appear in any emitted text
    leaked = [k for k, v in out.items()
              for t in list(v["text"].values()) + list(v.get("text_explained", {}).values())
              if "expert" in t.lower() or "ground truth" in t.lower()]
    assert not leaked, f"LABEL LEAK: expert annotation surfaced in text for {leaked[:3]}"
    print("Leak check passed: expert_annotation never rendered into text.\n")

    shown = {}
    for k, v in out.items():
        shown.setdefault(v["label"], k)
    for lab in ["crackle", "wheeze", "both", "normal"]:
        if lab in shown and args.show > 0:
            args.show -= 1
            print(f"===== {lab} =====")
            for c in args.conditions:
                print(f"[{c}] {out[shown[lab]]['text'][c]}")
            print()


if __name__ == "__main__":
    main()
