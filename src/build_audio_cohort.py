"""Freeze the UKCOVID cohort that every audio arm will share.

Coverage, file integrity, duplicates and split composition **only**. No disease-label
performance is computed here and no model is fitted: the cohort has to be fixed before
anyone sees how well anything predicts on it, or the cohort itself becomes a tuning knob.

`audio_metadata.csv` carries one row per participant with three recording types —
`exhalation`, `cough` (single) and `three_cough` — each with a filename, size, sample rate,
frame count, channel count, length, amplitude and SNR, plus a `missing_audio` flag. There
are no repeated sessions, so no per-person aggregation rule is needed; what must be fixed
in advance is **which recording type** the experiments use.

Decision, fixed here and not revisited: the primary type is `cough`, with `three_cough`
as the pre-declared secondary. `exhalation` is not used in the first version. Sentence
audio does not exist in the open-access release at all.

Checks run before the cohort is frozen:
  coverage      how many participants have a filename for each type
  existence     the file is actually on disk
  decodability  soundfile can open it and report frames
  duration      the decoded length agrees with the manifest's `*_length`
  duplicates    the same filename, or the same (frames, size, amplitude) signature,
                appearing under more than one participant
  splits        n and positive rate per Standard train/val/test, matched, matched_long

    python src/build_audio_cohort.py --data <metadata dir> --audio_root <extracted audio>
"""
import os, json, argparse, hashlib
import numpy as np
import pandas as pd

UNIT, LABEL, POSITIVE = "participant_identifier", "covid_test_result", "Positive"
PRIMARY_TYPE = "cough"                 # frozen; three_cough is the declared secondary
MIN_DURATION_S = 0.5                   # frozen: a cough plus context cannot fit in less
TYPES = ["cough", "three_cough", "exhalation"]
SPLIT_SETS = {"train": ("splits", "train"), "val": ("splits", "val"),
              "test": ("splits", "test"),
              "matched": ("in_matched_rebalanced_test", True),
              "matched_long": ("in_matched_rebalanced_long_test", True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/Seagate/resp_datasets/ukcovid")
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--report", default="results/ukcovid_audio_cohort_audit.json")
    ap.add_argument("--check_n", type=int, default=0,
                    help="decode-verify a random subset; 0 = all files of the primary type")
    args = ap.parse_args()

    p = pd.read_csv(os.path.join(args.data, "participant_metadata.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(args.data, "train_test_splits.csv"), low_memory=False)
    a = pd.read_csv(os.path.join(args.data, "audio_metadata.csv"), low_memory=False)
    d = p.merge(s, on=UNIT, validate="one_to_one").merge(a, on=UNIT, validate="one_to_one")
    print(f"joined: {len(d)} participants")
    audit = {"n_joined": int(len(d)), "primary_type": PRIMARY_TYPE}

    lab = d[LABEL].notna()
    print(f"with a COVID label: {int(lab.sum())}")
    audit["n_labelled"] = int(lab.sum())

    print("\n--- coverage by recording type ---")
    audit["coverage"] = {}
    for t in TYPES:
        has = d[f"{t}_file_name"].notna()
        audit["coverage"][t] = int(has.sum())
        print(f"  {t:<13s} filename present {int(has.sum()):6d}  "
              f"({has.mean()*100:.2f}%)   also labelled {int((has & lab).sum()):6d}")
    print(f"  missing_audio flag True: {int((d['missing_audio'] == True).sum())}")  # noqa: E712
    audit["missing_audio_flag"] = int((d["missing_audio"] == True).sum())           # noqa: E712

    print("\n--- duplicates ---")
    audit["duplicates"] = {}
    for t in TYPES:
        f = d[f"{t}_file_name"].dropna()
        dup_name = int(len(f) - f.nunique())
        sig = d.loc[f.index, [f"{t}_frames", f"{t}_size", f"{t}_amplitude"]]
        dup_sig = int(sig.duplicated().sum())
        audit["duplicates"][t] = {"same_filename": dup_name, "same_signature": dup_sig}
        print(f"  {t:<13s} identical filename {dup_name:5d}   "
              f"identical (frames,size,amplitude) {dup_sig:6d}")
    print("  note: an identical signature is not proof of a duplicate recording; it is "
          "reported so the count is visible rather than assumed to be zero")

    # ---- file existence and decodability, primary type, labelled participants only
    import soundfile as sf
    cand = d[lab & d[f"{PRIMARY_TYPE}_file_name"].notna()].copy()
    names = cand[f"{PRIMARY_TYPE}_file_name"].to_numpy()
    idx = np.arange(len(cand))
    if args.check_n and args.check_n < len(idx):
        idx = np.random.RandomState(0).choice(idx, args.check_n, replace=False)
    print(f"\n--- integrity check on {len(idx)} {PRIMARY_TYPE} files ---")
    missing, undecodable, dur = [], [], {}
    for j, i in enumerate(idx):
        fn = names[i]
        path = None
        for sub in ("", PRIMARY_TYPE, os.path.join("audio", PRIMARY_TYPE)):
            q = os.path.join(args.audio_root, sub, fn)
            if os.path.exists(q):
                path = q; break
        if path is None:
            missing.append(fn); continue
        try:
            info = sf.info(path)
            dur[fn] = info.frames / info.samplerate
        except Exception as e:
            undecodable.append((fn, str(e)[:60]))
        if (j + 1) % 5000 == 0:
            print(f"  {j+1}/{len(idx)}", flush=True)
    print(f"  missing on disk: {len(missing)}   undecodable: {len(undecodable)}")
    audit["integrity"] = {"checked": int(len(idx)), "missing": len(missing),
                          "undecodable": len(undecodable),
                          "missing_examples": missing[:5],
                          "undecodable_examples": [u[0] for u in undecodable[:5]]}

    if dur:
        D = pd.Series(dur)
        claimed = cand.set_index(f"{PRIMARY_TYPE}_file_name")[f"{PRIMARY_TYPE}_length"]
        both = pd.concat([D.rename("decoded"), claimed.rename("claimed")], axis=1).dropna()
        diff = (both.decoded - both.claimed).abs()
        print(f"  decoded duration: min {D.min():.2f}s median {D.median():.2f}s "
              f"max {D.max():.2f}s")
        print(f"  |decoded - manifest length|: max {diff.max():.4f}s, "
              f"n>0.01s: {int((diff > 0.01).sum())}")
        audit["duration"] = {"min": float(D.min()), "median": float(D.median()),
                             "max": float(D.max()), "max_abs_mismatch": float(diff.max()),
                             "n_mismatch_gt_10ms": int((diff > 0.01).sum())}

    # a file that opens is not the same as a file that is usable: the decoded lengths run
    # down to 0.085 s, which cannot contain a cough. The floor is declared, not tuned.
    ok = {f for f, v in dur.items() if v >= MIN_DURATION_S}
    dropped_short = len(dur) - len(ok)
    print(f"  shorter than {MIN_DURATION_S}s, excluded: {dropped_short}")
    audit["integrity"]["too_short"] = int(dropped_short)
    audit["integrity"]["min_duration_s"] = MIN_DURATION_S
    clipped = int((cand[f"{PRIMARY_TYPE}_amplitude"] == 32768.0).sum())
    audit["clipped_at_int16_max"] = clipped
    print(f"  peak amplitude at the int16 ceiling (clipped): {clipped} "
          f"({clipped/max(len(cand),1)*100:.1f}%) -- recorded because clipping may track "
          f"device or recruitment source")
    cohort = cand[cand[f"{PRIMARY_TYPE}_file_name"].isin(ok)].copy()
    cohort["y"] = (cohort[LABEL] == POSITIVE).astype(int)
    print(f"\n--- frozen cohort: {len(cohort)} participants with a usable "
          f"{PRIMARY_TYPE} recording and a label ---")

    print(f"\n{'split':<14s}{'n_all':>8s}{'pos_all':>9s}{'n_audio':>9s}"
          f"{'pos_audio':>11s}{'retained':>10s}")
    audit["splits"] = {}
    for name, (col, val) in SPLIT_SETS.items():
        m_all = (d[col] == val) & lab
        m_aud = (cohort[col] == val)
        rec = {"n_all": int(m_all.sum()), "pos_all": float(d.loc[m_all, LABEL].eq(POSITIVE).mean()),
               "n_audio": int(m_aud.sum()), "pos_audio": float(cohort.loc[m_aud, "y"].mean()),
               "retained": float(m_aud.sum() / max(m_all.sum(), 1))}
        audit["splits"][name] = rec
        print(f"{name:<14s}{rec['n_all']:8d}{rec['pos_all']:9.4f}{rec['n_audio']:9d}"
              f"{rec['pos_audio']:11.4f}{rec['retained']:10.4f}")

    cols = [UNIT, f"{PRIMARY_TYPE}_file_name", f"{PRIMARY_TYPE}_length",
            f"{PRIMARY_TYPE}_sample_rate", "splits", "naive_splits",
            "in_matched_rebalanced_test", "in_matched_rebalanced_long_test", "y"]
    cohort[cols].sort_values(UNIT).to_csv(args.out, index=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()[:16]
    audit["cohort_sha256_16"] = h
    audit["n_cohort"] = int(len(cohort))
    json.dump(audit, open(args.report, "w"), indent=1)
    print(f"\nwrote {args.out}  sha256[:16]={h}")
    print(f"wrote {args.report}")
    print("\nNo model was fitted and no disease-label performance was computed here. "
          "The cohort is frozen before any of that.")


if __name__ == "__main__":
    main()
