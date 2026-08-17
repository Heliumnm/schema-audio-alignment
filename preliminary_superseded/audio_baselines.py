"""Three audio baselines under one classification protocol, frozen.

The load-bearing quantity is **`artifacts+AST − artifacts_only`, paired, same patients,
same classifier**. If the arms used different downstream models, an increment could come
from the classifier rather than from AST, so all three share one regularised logistic head,
one standardisation, one selection procedure and one calibration.

Validation, decided from the shipped columns rather than invented:

    primary       splits == 'val'  AND  in_matched_rebalanced_train == True
                  1,036 participants, positive rate 0.5077, and zero overlap with Standard
                  train, Standard test, the matched test and matched_long
    robustness    the full Standard val, 5,179 in the audio cohort

The dataset's own `matched_train_splits == 'matched_validation'` is **not** used: 895 of
its 1,124 participants sit in Standard train, so selecting on it would be selecting on
training data.

Rules, fixed before the first fit:

* every arm and every regularisation strength trains only on Standard train;
* the regularisation is chosen on the 1,036-participant primary validation;
* calibration is stratified 5-fold cross-fitted Platt on those same 1,036, and the final
  calibrator is refitted on all 1,036 once the choice is made;
* if the full Standard val prefers a different strength, that difference is **reported and
  not acted on** -- the primary selection stands;
* **tie-break: within one standard error of the best, take the stronger regularisation.**
  Simpler wins ties, so noise cannot buy complexity;
* the test sets are run once, with the primary-selected model.

    python src/audio_baselines.py --probes
"""
import os, json, argparse
import numpy as np
import pandas as pd

UNIT = "participant_identifier"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]      # stronger regularisation first
SEEDS = [0, 1, 2, 3, 4]
ARTEFACT_FEATS = ["duration_s", "rms", "peak", "clip_frac", "near_silence_frac",
                  "zero_frac", "size_bytes", "sample_rate", "log_rms", "crest_factor"]


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def logloss(y, s, eps=1e-6):
    s = np.clip(s, eps, 1 - eps)
    return -(y * np.log(s) + (1 - y) * np.log(1 - s))


def fit_head(X, y, tr, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=5000, random_state=seed))
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X)[:, 1]


def select_C(X, y, tr, va, name):
    """Best mean validation AUROC across seeds, then the one-standard-error tie-break in
    favour of stronger regularisation."""
    scores, sds = {}, {}
    for C in C_GRID:
        v = [auroc(y[va], fit_head(X, y, tr, C, s)[va]) for s in SEEDS]
        scores[C], sds[C] = float(np.mean(v)), float(np.std(v))
    best = max(scores, key=scores.get)
    thresh = scores[best] - sds[best] / np.sqrt(len(SEEDS))
    chosen = min([C for C in C_GRID if scores[C] >= thresh])   # C_GRID ascending = weaker
    print(f"  [{name}] " + "  ".join(f"C={C}:{scores[C]:.4f}" for C in C_GRID))
    print(f"  [{name}] best C={best} ({scores[best]:.4f}); within 1 SE -> C={chosen}")
    return chosen, scores


def cross_fitted_platt(p_val, y_val, seed=0):
    """Stratified 5-fold cross-fitted Platt on the validation set, then a final calibrator
    refitted on all of it. Cross-fitting is what keeps the calibrated validation
    log-loss honest."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros_like(p_val)
    z = np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6)))
    for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=seed).split(z, y_val):
        c = LogisticRegression(max_iter=1000).fit(z[tr_i, None], y_val[tr_i])
        oof[te_i] = c.predict_proba(z[te_i, None])[:, 1]
    final = LogisticRegression(max_iter=1000).fit(z[:, None], y_val)
    return oof, final


def apply_platt(cal, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return cal.predict_proba(z[:, None])[:, 1]


def paired_ci(A, B, y, fn, boot=2000, seed=0):
    """Observed difference plus a participant x seed hierarchical interval."""
    obs = float(np.mean([fn(y, A[k]) - fn(y, B[k]) for k in range(A.shape[0])]))
    rs = np.random.RandomState(seed); out = []
    n = A.shape[1]
    for _ in range(boot):
        idx = rs.choice(n, n)
        if len(set(y[idx])) < 2:
            continue
        ss = rs.choice(A.shape[0], A.shape[0])
        out.append(np.mean([fn(y[idx], A[k][idx]) - fn(y[idx], B[k][idx]) for k in ss]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return obs, [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--feats", default="results/artefact_features.csv")
    ap.add_argument("--emb", default="results/ast_embeddings.npz")
    ap.add_argument("--out", default="results/audio_baselines.json")
    ap.add_argument("--probes", action="store_true")
    args = ap.parse_args()

    C = pd.read_csv(args.cohort)
    F = pd.read_csv(args.feats)
    p = pd.read_csv(os.path.join(args.data, "participant_metadata.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(args.data, "train_test_splits.csv"), low_memory=False)
    z = np.load(args.emb, allow_pickle=True)
    assert np.array_equal(z["participants"], C[UNIT].to_numpy()), \
        "embedding order does not match the frozen cohort"
    E = z["embeddings"]

    d = C.merge(F, on=UNIT).merge(
        p[[UNIT, "recruitment_source", "age", "gender", "symptom_cough_any",
           "symptom_none", "covid_test_result"]], on=UNIT).merge(
        s[[UNIT, "splits", "in_matched_rebalanced_train", "in_matched_rebalanced_test",
           "in_matched_rebalanced_long_test"]], on=UNIT, suffixes=("", "_s"))
    order = {q: i for i, q in enumerate(C[UNIT])}
    d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
    assert np.array_equal(d[UNIT].to_numpy(), C[UNIT].to_numpy())

    y = (d["covid_test_result"] == "Positive").astype(int).to_numpy()
    tr = (d["splits"] == "train").to_numpy()
    va = ((d["splits"] == "val") & (d["in_matched_rebalanced_train"] == True)).to_numpy()  # noqa: E712
    va_rob = (d["splits"] == "val").to_numpy()
    tests = {"standard": (d["splits"] == "test").to_numpy(),
             "matched": (d["in_matched_rebalanced_test"] == True).to_numpy(),          # noqa: E712
             "matched_long": (d["in_matched_rebalanced_long_test"] == True).to_numpy()}  # noqa: E712
    print(f"train {tr.sum()}  primary val {va.sum()} (pos {y[va].mean():.4f})  "
          f"robustness val {va_rob.sum()}")
    for k, m in tests.items():
        print(f"  test:{k:<13s} n={int(m.sum()):6d}  pos {y[m].mean():.4f}")

    A = d[ARTEFACT_FEATS].to_numpy(float)
    X = {"artifacts_only": A, "ast_only": E,
         "artifacts_plus_ast": np.concatenate([A, E], 1)}

    res, P = {"validation": {}}, {}
    for name, Xi in X.items():
        print(f"\n--- {name} ({Xi.shape[1]} features) ---")
        C_sel, sc = select_C(Xi, y, tr, va, "primary")
        C_rob, sc_r = select_C(Xi, y, tr, va_rob, "robustness")
        if C_rob != C_sel:
            print(f"  NOTE robustness val prefers C={C_rob}; reported, not acted on")
        Pk = np.stack([fit_head(Xi, y, tr, C_sel, s_) for s_ in SEEDS])
        cals = []
        for k in range(len(SEEDS)):
            _, cal = cross_fitted_platt(Pk[k][va], y[va])
            cals.append(cal)
        Pc = np.stack([apply_platt(cals[k], Pk[k]) for k in range(len(SEEDS))])
        P[name] = Pc
        res["validation"][name] = {"C_primary": C_sel, "C_robustness": C_rob,
                                   "val_scores": sc, "val_scores_robustness": sc_r}
        res[name] = {}
        for tn, m in tests.items():
            a = float(np.mean([auroc(y[m], Pc[k][m]) for k in range(len(SEEDS))]))
            ll = float(np.mean([logloss(y[m], Pc[k][m]).mean() for k in range(len(SEEDS))]))
            res[name][tn] = {"auroc": a, "logloss": ll, "n": int(m.sum())}
            print(f"  {tn:<14s} AUROC {a:.3f}   calibrated logloss {ll:.4f}")

    print("\n=== paired increments, same patients, same classifier ===")
    res["paired"] = {}
    for tn, m in tests.items():
        print(f"\n[{tn}]  n={int(m.sum())}")
        for arm in ("ast_only", "artifacts_plus_ast"):
            da, ca = paired_ci(P[arm][:, m], P["artifacts_only"][:, m], y[m], auroc)
            dl, cl = paired_ci(P[arm][:, m], P["artifacts_only"][:, m], y[m],
                               lambda yy, ss: -logloss(yy, ss).mean())
            res["paired"][f"{tn}:{arm}-artifacts_only"] = {
                "d_auroc": da, "ci": ca, "d_neg_logloss": dl, "ll_ci": cl}
            star = "excludes 0" if (ca[0] > 0 or ca[1] < 0) else "includes 0"
            print(f"  {arm:<20s} - artifacts_only  ΔAUROC {da:+.4f} "
                  f"[{ca[0]:+.4f},{ca[1]:+.4f}] {star}   Δ(-logloss) {dl:+.4f} "
                  f"[{cl[0]:+.4f},{cl[1]:+.4f}]")

    np.savez_compressed("results/audio_baseline_preds.npz",
                        participants=d[UNIT].to_numpy(), seeds=np.array(SEEDS),
                        **{k: v for k, v in P.items()})

    if args.probes:
        print("\n=== raw AST probes (frozen reference for any aligned representation) ===")
        res["probes"] = {}
        targets = {"recruitment_source": (d.recruitment_source == "Test and Trace").astype(int),
                   "gender": (d.gender == "Female").astype(int),
                   "age_65plus": (d.age == "65+").astype(int),
                   "symptom_cough_any": d.symptom_cough_any.fillna(0).astype(int),
                   "symptom_none": d.symptom_none.fillna(0).astype(int)}
        for t, yt in targets.items():
            yt = yt.to_numpy()
            C_sel, _ = select_C(E, yt, tr, va, f"probe:{t}")
            Pt = np.stack([fit_head(E, yt, tr, C_sel, s_) for s_ in SEEDS])
            row = {}
            for tn, m in tests.items():
                a = float(np.mean([auroc(yt[m], Pt[k][m]) for k in range(len(SEEDS))]))
                row[tn] = a
            res["probes"][t] = {"C": C_sel, **row}
            print(f"  {t:<20s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()))
        np.savez_compressed("results/ast_probe_preds.npz",
                            participants=d[UNIT].to_numpy())

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
