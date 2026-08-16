"""Artefact baseline — what a recording's shape alone predicts, before any model hears it.

32.0% of the cohort's cough recordings are clipped at the int16 ceiling. Clipping,
loudness and length are properties of the device and the recording situation, not of the
respiratory tract, so they are exactly the kind of thing that can carry recruitment source
and, through it, disease prevalence. This baseline has to exist **before** an audio model
is fitted, or every later number is uninterpretable: an AST arm that beats metadata could
be hearing a cough or could be reading a microphone.

Two targets, same features, same cohort, same splits:

    covid                the disease label -- how much apparent skill is available with
                         no acoustic content at all
    recruitment_source   REACT vs Test and Trace -- how legible the study design is from
                         the waveform's shape

Features are measured from the audio, not copied from the manifest: `cough_amplitude` is a
peak, and peak alone cannot separate a loud recording from a clipped one.

    python src/artefact_baseline.py --cohort results/ukcovid_audio_cohort.csv \\
        --audio_root .../ukcovid/audio/audio --data .../ukcovid
"""
import os, json, argparse
import numpy as np
import pandas as pd

INT16_MAX = 32767
FEATS = ["duration_s", "rms", "peak", "clip_frac", "near_silence_frac", "zero_frac",
         "size_bytes", "sample_rate", "log_rms", "crest_factor"]


def measure(path):
    import soundfile as sf
    try:
        y, sr = sf.read(path, dtype="int16", always_2d=True)
    except Exception:
        return None
    y = y[:, 0].astype(np.float64)
    if len(y) == 0:
        return None
    a = np.abs(y)
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(a.max())
    return {"duration_s": len(y) / sr, "rms": rms, "peak": peak,
            "clip_frac": float((a >= INT16_MAX).mean()),
            "near_silence_frac": float((a < 32).mean()),
            "zero_frac": float((y == 0).mean()),
            "size_bytes": float(os.path.getsize(path)), "sample_rate": float(sr),
            "log_rms": float(np.log1p(rms)),
            "crest_factor": float(peak / (rms + 1e-9))}


def _one(args):
    pid, path = args
    m = measure(path)
    return (pid, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--data", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--feats_out", default="results/artefact_features.csv")
    ap.add_argument("--out", default="results/artefact_baseline.json")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args()

    C = pd.read_csv(args.cohort)
    print(f"cohort {len(C)} participants")

    if os.path.exists(args.feats_out):
        F = pd.read_csv(args.feats_out)
        print(f"reusing {args.feats_out} ({len(F)} rows)")
    else:
        from multiprocessing import Pool
        jobs = [(r.participant_identifier,
                 os.path.join(args.audio_root, r.cough_file_name))
                for r in C.itertuples()]
        rows = []
        with Pool(args.jobs) as pool:
            for i, (pid, m) in enumerate(pool.imap_unordered(_one, jobs, chunksize=200)):
                if m is not None:
                    rows.append({"participant_identifier": pid, **m})
                if (i + 1) % 10000 == 0:
                    print(f"  measured {i+1}/{len(jobs)}", flush=True)
        F = pd.DataFrame(rows)
        F.to_csv(args.feats_out, index=False)
        print(f"wrote {args.feats_out} ({len(F)} rows)")

    p = pd.read_csv(os.path.join(args.data, "participant_metadata.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(args.data, "train_test_splits.csv"), low_memory=False)
    d = (C.merge(F, on="participant_identifier")
          .merge(p[["participant_identifier", "recruitment_source", "covid_test_result"]],
                 on="participant_identifier")
          .merge(s[["participant_identifier", "splits", "in_matched_rebalanced_test",
                    "in_matched_rebalanced_long_test"]], on="participant_identifier",
                 suffixes=("", "_s")))
    print(f"joined {len(d)} participants with measured audio")

    print("\n--- artefact features, by recruitment source ---")
    for f in ["duration_s", "rms", "clip_frac", "size_bytes", "crest_factor"]:
        g = d.groupby("recruitment_source")[f].median()
        print(f"  {f:<20s} " + "   ".join(f"{k}: {v:,.4g}" for k, v in g.items()))
    print("\n--- clipping rate by recruitment source ---")
    print(d.groupby("recruitment_source").apply(
        lambda g: pd.Series({"n": len(g), "any_clipping": (g.clip_frac > 0).mean(),
                             "clip_frac_median": g.clip_frac.median()}),
        include_groups=False).round(4).to_string())

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = d[FEATS].to_numpy()
    tr = (d["splits"] == "train").to_numpy()
    tests = {"standard": (d["splits"] == "test").to_numpy(),
             "matched": (d["in_matched_rebalanced_test"] == True).to_numpy(),      # noqa: E712
             "matched_long": (d["in_matched_rebalanced_long_test"] == True).to_numpy()}  # noqa: E712

    res = {}
    for target, y in [("covid", (d["covid_test_result"] == "Positive").astype(int).to_numpy()),
                      ("recruitment_source",
                       (d["recruitment_source"] == "Test and Trace").astype(int).to_numpy())]:
        print(f"\n--- artefact-only -> {target} ---")
        P = np.stack([HistGradientBoostingClassifier(random_state=s_)
                      .fit(X[tr], y[tr]).predict_proba(X)[:, 1] for s_ in range(5)])
        res[target] = {}
        for name, m in tests.items():
            a = float(np.mean([roc_auc_score(y[m], P[k][m]) for k in range(5)]))
            rs = np.random.RandomState(0)
            bs = []
            for _ in range(2000):
                idx = rs.choice(int(m.sum()), int(m.sum()))
                yy = y[m][idx]
                if len(set(yy)) < 2:
                    continue
                bs.append(np.mean([roc_auc_score(yy, P[k][m][idx]) for k in
                                   rs.choice(5, 5)]))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[target][name] = {"auroc": a, "ci": [float(lo), float(hi)],
                                 "n": int(m.sum())}
            print(f"  {name:<14s} n={int(m.sum()):6d}  AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]")
        np.save(f"results/artefact_preds_{target}.npy", P)

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    print("\nThese are the floors. An audio model that does not clear them is reading the "
          "microphone, not the cough.")


if __name__ == "__main__":
    main()
