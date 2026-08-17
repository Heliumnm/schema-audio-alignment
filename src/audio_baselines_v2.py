"""Three audio baselines, corrected. Supersedes preliminary_superseded/audio_baselines.py.

What changed and why
--------------------
**The one-standard-error rule now has a standard error.** The first version estimated it
across five `LogisticRegression` seeds, which is deterministic under lbfgs — the measured
seed-to-seed difference was exactly 0.000e+00 — so the SE was zero and the rule degenerated
into argmax. Here the SE comes from **five fixed folds of the full Standard validation,
stratified by label and recruitment source**, frozen once and reused by every arm and every
C. The selection metric stays AUROC, as originally frozen; only the SE estimator is
repaired.

**One calibrator, because there is only one validation domain.** An earlier version fitted
a second calibrator on 1,036 participants it called "matched-like". They are not: within
them recruitment source predicts the label at AUROC 0.9990, against 0.5000 on the matched
test sets. No target-like validation exists in this release and none is claimed. Everything
is selected and calibrated on the full Standard val, and **matched / matched_long are
out-of-distribution stress tests that tune and calibrate nothing** — their log-losses
measure calibration transfer out of the source domain, not probability quality after
recalibrating inside the target population. See docs/PROTOCOL_HISTORY.md.

**No fake seed axis.** A deterministic head has one solution, so predictions are stored
with a seed axis of length one and the intervals are honestly called participant
bootstraps. The axis is kept so that later arms with genuine optimiser noise slot in
without changing the file format.

**Missing probe labels are dropped, never imputed as negative.**

Added measurements, all of which the earlier claims needed and did not have:
  artifacts+AST − AST      paired, to test whether artefacts add anything on top of AST
  AST → artefacts          can the representation decode duration, RMS, size, clipping
  calibration diagnostics  slope, intercept, Brier, on each test set
  probe intervals          and per-participant predictions, so an aligned representation
                           can later be compared against these pairwise

    python src/audio_baselines_v2.py --probes
"""
import os, json, argparse
import numpy as np
import pandas as pd

UNIT = "participant_identifier"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]      # ascending = weaker
N_FOLDS = 5
FOLD_SEED = 20260817
ARTEFACT_FEATS = ["duration_s", "rms", "peak", "clip_frac", "near_silence_frac",
                  "zero_frac", "size_bytes", "sample_rate", "log_rms", "crest_factor"]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def logloss(y, s, eps=1e-6):
    s = np.clip(s, eps, 1 - eps)
    return -(y * np.log(s) + (1 - y) * np.log(1 - s))


def fit_head(X, y, tr, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X)[:, 1]


def make_folds(d, va):
    """Five folds over the FULL Standard validation, stratified by label and recruitment
    source. Frozen once; every arm and every C sees the identical partition.

    There is no target-like validation in this dataset and none is pretended: recruitment
    source predicts the label at AUROC 0.9966 in Standard train, 0.9984 in Standard val and
    0.999 in the 1,036-participant subset that was previously called matched-like, while
    the matched test sets sit at exactly 0.5000. Selection therefore happens entirely in
    the source domain, and matched / matched_long are out-of-distribution stress tests that
    tune and calibrate nothing."""
    from sklearn.model_selection import StratifiedKFold
    idx = np.where(va)[0]
    strat = (d.loc[idx, "y"].astype(str) + "|" +
             d.loc[idx, "recruitment_source"].fillna("NA").astype(str))
    vc = strat.value_counts()
    strat = strat.map(lambda s: s if vc[s] >= N_FOLDS else s.split("|")[0] + "|RARE")
    folds = np.full(len(d), -1)
    for k, (_, te) in enumerate(StratifiedKFold(
            N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(idx, strat)):
        folds[idx[te]] = k
    return folds


def select_C(X, y, tr, folds, name):
    """Mean fold AUROC picks the best C; the SE across folds sets the one-SE band; the
    strongest regularisation inside the band wins. Simpler takes ties."""
    means, ses, per = {}, {}, {}
    for C in C_GRID:
        p = fit_head(X, y, tr, C)
        vals = [auroc(y[folds == k], p[folds == k]) for k in range(N_FOLDS)]
        means[C], ses[C] = float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(N_FOLDS))
        per[C] = [float(v) for v in vals]
    best = max(means, key=means.get)
    band = means[best] - ses[best]
    chosen = min([C for C in C_GRID if means[C] >= band])
    print(f"  [{name}] " + "  ".join(f"{C}:{means[C]:.4f}" for C in C_GRID))
    print(f"  [{name}] best C={best} ({means[best]:.4f} +/- {ses[best]:.4f} SE); "
          f"1-SE band >= {band:.4f} -> C={chosen}")
    return chosen, {"mean": means, "se": ses, "folds": per, "best": best, "chosen": chosen}


def platt(p_fit, y_fit):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p_fit, 1e-6, 1 - 1e-6) / (1 - np.clip(p_fit, 1e-6, 1 - 1e-6)))
    return LogisticRegression(max_iter=1000).fit(z[:, None], y_fit)


def apply_platt(cal, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return cal.predict_proba(z[:, None])[:, 1]


def calib_diag(y, p):
    """Slope and intercept of the logistic recalibration, plus Brier. Slope 1 and
    intercept 0 is perfect; slope < 1 means over-confident."""
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(max_iter=1000).fit(z[:, None], y)
    return {"slope": float(m.coef_[0][0]), "intercept": float(m.intercept_[0]),
            "brier": float(np.mean((p - y) ** 2))}


def boot_paired(a, b, y, fn, boot=2000, seed=0):
    obs = float(fn(y, a) - fn(y, b))
    rs = np.random.RandomState(seed); out = []
    n = len(y)
    for _ in range(boot):
        i = rs.choice(n, n)
        if len(set(y[i])) < 2:
            continue
        out.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return obs, [float(lo), float(hi)]


def boot_ci(y, p, fn=auroc, boot=2000, seed=0):
    rs = np.random.RandomState(seed); out = []
    n = len(y)
    for _ in range(boot):
        i = rs.choice(n, n)
        if len(set(y[i])) < 2:
            continue
        out.append(fn(y[i], p[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--feats", default="results/artefact_features.csv")
    ap.add_argument("--emb", default="results/ast_embeddings.npz")
    ap.add_argument("--out", default="results/audio_baselines_v3.json")
    ap.add_argument("--probes", action="store_true")
    args = ap.parse_args()

    C = pd.read_csv(args.cohort)
    F = pd.read_csv(args.feats)
    p = pd.read_csv(os.path.join(args.data, "participant_metadata.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(args.data, "train_test_splits.csv"), low_memory=False)
    z = np.load(args.emb, allow_pickle=True)
    assert np.array_equal(z["participants"], C[UNIT].to_numpy())
    E = z["embeddings"]

    keep_p = [UNIT, "recruitment_source", "age", "gender", "symptom_cough_any",
              "symptom_none", "covid_test_result"]
    keep_s = [UNIT, "splits", "in_matched_rebalanced_train", "in_matched_rebalanced_test",
              "in_matched_rebalanced_long_test", "stratum_matched_rebalanced_train"]
    d = C.merge(F, on=UNIT).merge(p[keep_p], on=UNIT).merge(s[keep_s], on=UNIT,
                                                            suffixes=("", "_s"))
    order = {q: i for i, q in enumerate(C[UNIT])}
    d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
    d["y"] = (d["covid_test_result"] == "Positive").astype(int)
    y = d["y"].to_numpy()

    tr = (d["splits"] == "train").to_numpy()
    va_s = (d["splits"] == "val").to_numpy()          # the only validation set
    va_m = va_s                                        # kept for interface compatibility
    tests = {"standard": (d["splits"] == "test").to_numpy(),
             "matched": (d["in_matched_rebalanced_test"] == True).to_numpy(),          # noqa: E712
             "matched_long": (d["in_matched_rebalanced_long_test"] == True).to_numpy()}  # noqa: E712
    # One calibrator, fitted on the source-domain validation. The matched log-losses are
    # therefore a measure of CALIBRATION TRANSFER out of the source domain, not of
    # probability quality after recalibrating inside the target population.
    CAL_OF = {"standard": "standard", "matched": "standard", "matched_long": "standard"}

    folds = make_folds(d, va_s)
    fc = np.bincount(folds[folds >= 0])
    print(f"train {tr.sum()}  Standard val {va_s.sum()} folds {list(fc)}  "
          f"(matched/matched_long are OOD stress tests only)")

    A = d[ARTEFACT_FEATS].to_numpy(float)
    X = {"artifacts_only": A, "ast_only": E,
         "artifacts_plus_ast": np.concatenate([A, E], 1)}

    res, P = {"selection": {}, "arms": {}}, {}
    for name, Xi in X.items():
        print(f"\n--- {name} ({Xi.shape[1]} features) ---")
        C_sel, info = select_C(Xi, y, tr, folds, "primary")
        raw = fit_head(Xi, y, tr, C_sel)
        cal = {"standard": platt(raw[va_s], y[va_s])}
        Pc = {k: apply_platt(cal[CAL_OF[k]], raw) for k in tests}
        P[name] = {"raw": raw, **Pc}
        res["selection"][name] = info
        res["arms"][name] = {}
        for tn, m in tests.items():
            pc = Pc[tn][m]
            a = auroc(y[m], pc)
            res["arms"][name][tn] = {"auroc": a, "ci": boot_ci(y[m], pc),
                                     "logloss": float(logloss(y[m], pc).mean()),
                                     "calibration": calib_diag(y[m], pc), "n": int(m.sum())}
            cd = res["arms"][name][tn]["calibration"]
            print(f"  {tn:<14s} AUROC {a:.3f}  logloss {res['arms'][name][tn]['logloss']:.4f}"
                  f"  slope {cd['slope']:.3f}  intercept {cd['intercept']:+.3f}"
                  f"  Brier {cd['brier']:.4f}")

    print("\n=== paired increments, same participants, same head, participant bootstrap ===")
    res["paired"] = {}
    pairs = [("ast_only", "artifacts_only"), ("artifacts_plus_ast", "artifacts_only"),
             ("artifacts_plus_ast", "ast_only")]
    for tn, m in tests.items():
        print(f"\n[{tn}] n={int(m.sum())}")
        for a1, a0 in pairs:
            da, ca = boot_paired(P[a1][tn][m], P[a0][tn][m], y[m], auroc)
            dl, cl = boot_paired(P[a1][tn][m], P[a0][tn][m], y[m],
                                 lambda yy, ss: -logloss(yy, ss).mean())
            res["paired"][f"{tn}:{a1}-{a0}"] = {"d_auroc": da, "ci": ca,
                                                "d_neg_logloss": dl, "ll_ci": cl}
            star = "excludes 0" if (ca[0] > 0 or ca[1] < 0) else "includes 0"
            print(f"  {a1:<20s} - {a0:<18s} ΔAUROC {da:+.4f} [{ca[0]:+.4f},{ca[1]:+.4f}]"
                  f" {star}   Δ(-ll) {dl:+.4f} [{cl[0]:+.4f},{cl[1]:+.4f}]")

    np.savez_compressed("results/audio_baseline_preds_v3.npz",
                        participants=d[UNIT].to_numpy(), seeds=np.array([0]),
                        note="one deterministic solution per arm; seed axis length 1",
                        **{f"{k}__{t}": P[k][t][None, :] for k in P for t in list(tests) + ["raw"]})

    if args.probes:
        print("\n=== probes on the frozen raw AST representation ===")
        res["probes"] = {}
        targets = {
            "recruitment_source": (d.recruitment_source == "Test and Trace").where(
                d.recruitment_source.notna()),
            "gender": (d.gender == "Female").where(d.gender.notna()),
            "age_65plus": (d.age == "65+").where(d.age.notna()),
            "symptom_cough_any": d.symptom_cough_any.where(d.symptom_cough_any.notna()),
            "symptom_none": d.symptom_none.where(d.symptom_none.notna()),
            # artefact decodability: does the representation carry recording shape?
            "clipped": (d.clip_frac > 0).where(d.clip_frac.notna()),
            "long_recording": (d.duration_s > d.duration_s.median()).where(d.duration_s.notna()),
            "loud_recording": (d.rms > d.rms.median()).where(d.rms.notna()),
            "large_file": (d.size_bytes > d.size_bytes.median()).where(d.size_bytes.notna()),
        }
        probe_preds = {}
        for t, yt_raw in targets.items():
            obs = yt_raw.notna().to_numpy()              # missing labels dropped, not zeroed
            yt = yt_raw.fillna(0).astype(int).to_numpy()
            trt, fol = tr & obs, np.where(obs, folds, -1)
            C_sel, _ = select_C(E[obs], yt[obs], trt[obs], fol[obs], f"probe:{t}")
            pt = fit_head(E, yt, trt, C_sel)
            probe_preds[t] = pt
            row = {"C": C_sel, "n_labelled": int(obs.sum())}
            for tn, m in tests.items():
                mm = m & obs
                row[tn] = {"auroc": auroc(yt[mm], pt[mm]), "ci": boot_ci(yt[mm], pt[mm]),
                           "n": int(mm.sum())}
            res["probes"][t] = row
            print(f"  {t:<20s} " + "  ".join(
                f"{k} {row[k]['auroc']:.3f}[{row[k]['ci'][0]:.3f},{row[k]['ci'][1]:.3f}]"
                for k in tests))
        np.savez_compressed("results/ast_probe_preds_v3.npz",
                            participants=d[UNIT].to_numpy(), **probe_preds)

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
