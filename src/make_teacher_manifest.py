"""Freeze the teacher check's 70/30 split, before any GPU call.

Grouped by **recording**, so a recording's several annotator rows never straddle the
split, and stratified by **annotator × dry/wet** so each slice keeps each annotator's own
wet rate. Duration terciles are computed from the audio and stored here because the
offline permutation control stratifies on annotator × quality × duration tercile, and that
stratification has to be fixed before any score exists.

    python src/make_teacher_manifest.py --long .../annotator_long.csv \\
        --audio_root .../COUGHVID/extracted/public_dataset
"""
import os, json, argparse, hashlib
import numpy as np
import pandas as pd

SPLIT_SEED = 20260816       # committed; never re-drawn
DEV_FRAC = 0.30


def decode_and_measure(uuids, audio_root, wav_dir):
    """COUGHVID ships .webm (19,213) and .ogg (859). The webm files are MediaRecorder
    output with **no duration field in the container**: ffprobe reports N/A and
    librosa's audioread fallback returns 0.0 without raising. Reading either would have
    frozen a manifest whose duration terciles were two-thirds zeros.

    So decode once to 16 kHz mono wav -- which the model needs anyway -- and measure the
    decoded stream. Any zero or missing duration is a hard failure, not a fill value."""
    import subprocess
    import soundfile as sf
    os.makedirs(wav_dir, exist_ok=True)
    out, failed = {}, []
    for i, u in enumerate(uuids):
        w = os.path.join(wav_dir, u + ".wav")
        if not os.path.exists(w):
            src = next((os.path.join(audio_root, u + e) for e in (".webm", ".ogg", ".wav")
                        if os.path.exists(os.path.join(audio_root, u + e))), None)
            if src is None:
                failed.append((u, "no source file")); continue
            r = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "1",
                                "-ar", "16000", "-y", w], capture_output=True)
            if r.returncode != 0:
                failed.append((u, r.stderr.decode()[:80])); continue
        try:
            info = sf.info(w)
            d = info.frames / info.samplerate
        except Exception as e:
            failed.append((u, str(e)[:80])); continue
        if not np.isfinite(d) or d <= 0:
            failed.append((u, f"duration {d}")); continue
        out[u] = d
        if (i + 1) % 400 == 0:
            print(f"  decoded {i+1}/{len(uuids)}", flush=True)
    assert not failed, (f"{len(failed)} recordings have no usable duration, e.g. "
                        f"{failed[:3]}; resolve before freezing")
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--wav_dir", default="/mnt/hd/data_heliu/resp_datasets/COUGHVID/wav16k")
    ap.add_argument("--out", default="results/teacher_split_manifest.csv")
    ap.add_argument("--config_out", default="results/teacher_run_config.json")
    args = ap.parse_args()

    L = pd.read_csv(args.long)
    L = L[L.label.isin(["dry", "wet"])].copy()
    print(f"{len(L)} annotator rows over {L.uuid.nunique()} recordings")

    dur = decode_and_measure(sorted(L.uuid.unique()), args.audio_root, args.wav_dir)
    print(f"durations measured on decoded audio for {len(dur)} recordings; "
          f"min {dur.min():.2f}s median {dur.median():.2f}s max {dur.max():.2f}s")
    assert (dur > 0).all(), "a zero duration survived; do not freeze"
    L["duration_s"] = L.uuid.map(dur)
    L["duration_tercile"] = pd.qcut(L.uuid.map(dur), 3, labels=["t1", "t2", "t3"])
    L["quality"] = L["quality"].fillna("unknown")

    # one stratum label per RECORDING: its annotator/label signature, so the split is
    # grouped by recording and stratified by annotator x dry/wet at the same time
    sig = (L.sort_values(["annotator"])
             .groupby("uuid")
             .apply(lambda g: "|".join(f"{a}:{l}" for a, l in zip(g.annotator, g.label)),
                    include_groups=False))
    counts = sig.value_counts()
    # strata with a single recording cannot be split proportionally; pool them so the
    # assignment stays reproducible instead of silently falling back
    rare = set(counts[counts < 2].index)
    strat = sig.map(lambda s: "RARE" if s in rare else s)
    print(f"{sig.nunique()} recording signatures, {len(rare)} pooled into RARE")

    rs = np.random.RandomState(SPLIT_SEED)
    slice_of = {}
    for key, grp in strat.groupby(strat):
        u = np.array(sorted(grp.index)); rs.shuffle(u)
        k = int(round(DEV_FRAC * len(u)))
        for i, uu in enumerate(u):
            slice_of[uu] = "dev" if i < k else "report"
    L["slice"] = L.uuid.map(slice_of)

    print("\nslice sizes (annotator rows / recordings) and wet rate per annotator")
    for s in ("dev", "report"):
        m = L.slice == s
        print(f"  {s:<7s} rows={int(m.sum()):5d}  recordings={L.loc[m,'uuid'].nunique():5d}")
        for a, g in L[m].groupby("annotator"):
            print(f"      annotator {a}: n={len(g):4d}  wet={(g.label=='wet').mean():.3f}")
    straddle = L.groupby("uuid")["slice"].nunique().max()
    assert straddle == 1, "a recording straddles the split"
    print("\ngroup check: no recording appears in both slices")

    cols = ["uuid", "annotator", "label", "quality", "duration_s", "duration_tercile",
            "cough_detected", "slice"]
    L[cols].sort_values(["uuid", "annotator"]).to_csv(args.out, index=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    cfg = {"wav_dir": args.wav_dir, "split_seed": SPLIT_SEED, "dev_frac": DEV_FRAC, "permutation_seed": 20260817,
           "n_permutations": 1000, "manifest_sha256_16": h,
           "model": "/mnt/hd/data_ycyang/models/Qwen2-Audio-7B-Instruct",
           "n_rows": int(len(L)), "n_recordings": int(L.uuid.nunique())}
    json.dump(cfg, open(args.config_out, "w"), indent=1)
    print(f"\nwrote {args.out}  sha256[:16]={h}")
    print(f"wrote {args.config_out}")


if __name__ == "__main__":
    main()
