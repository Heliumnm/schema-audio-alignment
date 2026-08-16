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


def durations(uuids, audio_root):
    """COUGHVID ships .webm and .ogg. soundfile reads ogg; webm needs librosa/audioread."""
    import soundfile as sf
    out = {}
    for u in uuids:
        d = np.nan
        for ext in (".ogg", ".webm", ".wav"):
            p = os.path.join(audio_root, u + ext)
            if not os.path.exists(p):
                continue
            try:
                info = sf.info(p)
                d = info.frames / info.samplerate
            except Exception:
                try:
                    import librosa
                    d = librosa.get_duration(path=p)
                except Exception:
                    d = np.nan
            break
        out[u] = d
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out", default="results/teacher_split_manifest.csv")
    ap.add_argument("--config_out", default="results/teacher_run_config.json")
    args = ap.parse_args()

    L = pd.read_csv(args.long)
    L = L[L.label.isin(["dry", "wet"])].copy()
    print(f"{len(L)} annotator rows over {L.uuid.nunique()} recordings")

    dur = durations(sorted(L.uuid.unique()), args.audio_root)
    miss = int(dur.isna().sum())
    print(f"durations read for {len(dur)-miss}/{len(dur)} recordings ({miss} missing)")
    assert miss == 0, f"{miss} recordings have no readable audio; resolve before freezing"
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
    cfg = {"split_seed": SPLIT_SEED, "dev_frac": DEV_FRAC, "permutation_seed": 20260817,
           "n_permutations": 1000, "manifest_sha256_16": h,
           "model": "/mnt/hd/data_ycyang/models/Qwen2-Audio-7B-Instruct",
           "n_rows": int(len(L)), "n_recordings": int(L.uuid.nunique())}
    json.dump(cfg, open(args.config_out, "w"), indent=1)
    print(f"\nwrote {args.out}  sha256[:16]={h}")
    print(f"wrote {args.config_out}")


if __name__ == "__main__":
    main()
