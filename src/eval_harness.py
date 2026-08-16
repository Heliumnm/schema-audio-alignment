"""UKCOVID evaluation harness — two questions, one fixed patient set, no split leakage.

  Q1  total predictive ability   AUROC on the Standard test set
  Q2  audio increment over a
      symptom questionnaire      AUROC on the covariate-matched test set, plus a
                                 per-patient paired logloss delta against the reference

Split discipline, verified rather than assumed
----------------------------------------------
The shipped splits are not interchangeable, and picking the wrong pair leaks:

    in_matched_original_test    overlaps Standard train by 292 and val by 78  -> UNUSABLE
                                with Standard training (370 / 1968 = 18.8% seen)
    in_matched_rebalanced_test  overlaps Standard train/val/long by 0         -> USE THIS
    naive_test                  overlaps Standard train by 3003, val by 758   -> only
                                valid against a model trained on naive_splits == 'train'

So there are two **training regimes**, and their numbers are never placed in the same
comparison:

    standard   train splits=='train' (20,717), select on splits=='val' (5,180),
               report on splits=='test' (11,121) and in_matched_rebalanced_test (1,814),
               with in_matched_rebalanced_long_test (4,196) for power
    naive      train naive_splits=='train' (47,489), select on naive_splits=='val',
               report on naive_splits=='test' (10,177). Its only job is to show how much
               the naive split inflates; it is not a result about any method.

Every overlap above is re-asserted in code at load time, so a future edit that swaps a
split cannot silently reintroduce the leak.

Predictions
-----------
Arms hand back a **(n_seeds, n_participants) matrix**, not a seed-averaged vector, over one
fixed participant index shared by every arm. Seeds stay a dimension all the way to the
confidence interval: CIs are a participant x seed hierarchical bootstrap, so optimiser
noise is inside the interval instead of being averaged away before anyone can see it.

Model selection
---------------
The reference arm is chosen on **validation only** and then frozen. Which family won is
recorded in the output; nothing downstream may re-open that choice, because a reference
picked on the test set would turn every audio increment into a comparison against a
handicapped baseline.

On the sensitivity statistic
----------------------------
`G = AUC_standard - AUC_matched` measures sensitivity to population composition, **not
confounding**. A near-chance model also scores a small G. In the debug run a zero-audio
symptom model scored the largest G in the experiment, which is the whole argument for
keeping G secondary and answering the mechanism question with the decodability probe
(`PROBE_TARGETS`) instead.

    python src/eval_harness.py --data /Volumes/Seagate/resp_datasets/ukcovid
"""
import os, json, argparse
import numpy as np
import pandas as pd

LABEL, POSITIVE, UNIT = "covid_test_result", "Positive", "participant_identifier"

FEATURES = ["age", "gender", "smoker_status",
            "respiratory_condition_asthma", "respiratory_condition_other"]
SYMPTOM_PREFIX = "symptom_"
# study logistics, outcome-derived quantities, and timing. recruitment_source is also a
# probe target: putting it in the features would make the reference a recruitment classifier
EXCLUDED = ["recruitment_source", "survey_phase", "submission_date", "submission_delay",
            "covid_ct_value", "covid_ct_gene", "covid_ct_mean", "covid_viral_load",
            "covid_viral_load_category", "covid_test_method", "covid_test_date",
            "covid_test_processed_date", "influenza_a_test_result", "influenza_a_ct_value",
            "influenza_b_test_result", "influenza_b_ct_value", "region_name",
            "wearing_mask", "symptom_onset"]

PROBE_TARGETS = ["age", "gender", "recruitment_source", "symptom_cough_any", "symptom_none"]

REGIMES = {
    "standard": {"train": ("splits", "train"), "val": ("splits", "val"),
                 "tests": {"standard": ("splits", "test"),
                           "matched": ("in_matched_rebalanced_test", True),
                           "matched_long": ("in_matched_rebalanced_long_test", True)}},
    "naive": {"train": ("naive_splits", "train"), "val": ("naive_splits", "val"),
              "tests": {"naive": ("naive_splits", "test")}},
}


def _mask(d, spec):
    col, val = spec
    return (d[col] == val).to_numpy()


def load(data_dir, verbose=True):
    p = pd.read_csv(os.path.join(data_dir, "participant_metadata.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(data_dir, "train_test_splits.csv"), low_memory=False)
    d = p.merge(s, on=UNIT, validate="one_to_one")
    d = d[d[LABEL].notna()].reset_index(drop=True)
    d["y"] = (d[LABEL] == POSITIVE).astype(int)
    assert d[UNIT].is_unique, "participant_identifier is not unique; the unit is not a patient"

    # the leak guard. Any test set must be disjoint from its own regime's train and val.
    for rname, r in REGIMES.items():
        tr, va = _mask(d, r["train"]), _mask(d, r["val"])
        for tname, spec in r["tests"].items():
            te = _mask(d, spec)
            ntr, nva = int((te & tr).sum()), int((te & va).sum())
            assert ntr == 0 and nva == 0, (
                f"LEAK: {rname} test '{tname}' overlaps train by {ntr} and val by {nva}")
        assert int((tr & va).sum()) == 0, f"{rname}: train and val overlap"
    if verbose:
        print("split leak guard: every test set is disjoint from its regime's train and val")
    return d


def report_splits(d):
    print(f"\n{len(d)} participants with a label "
          f"(positive rate {d['y'].mean():.4f})")
    for rname, r in REGIMES.items():
        print(f"\n[{rname} regime]")
        for role in ("train", "val"):
            m = _mask(d, r[role])
            print(f"  {role:<14s} n={int(m.sum()):6d}  positive {d.loc[m,'y'].mean():.4f}")
        for tname, spec in r["tests"].items():
            m = _mask(d, spec)
            print(f"  test:{tname:<9s} n={int(m.sum()):6d}  positive {d.loc[m,'y'].mean():.4f}")
    print("\ncovariate balance on the matched test (the reason it is the Q2 set)")
    m = _mask(d, REGIMES["standard"]["tests"]["matched"])
    for c in ["age", "gender", "symptom_cough_any", "symptom_none", "recruitment_source"]:
        t = pd.crosstab(d.loc[m, c], d.loc[m, "y"], normalize="columns")
        cells = "  ".join(f"{i}:{t.loc[i,0]:.3f}/{t.loc[i,1]:.3f}" for i in t.index)
        print(f"  {c:<22s} neg/pos  {cells}")


def feature_frame(d):
    sym = [c for c in d.columns if c.startswith(SYMPTOM_PREFIX) and c not in EXCLUDED]
    cols = FEATURES + sym
    assert not [c for c in cols if c in EXCLUDED], "excluded column leaked into features"
    return pd.get_dummies(d[cols].astype(object), dummy_na=True).astype(float), cols


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def logloss(y, s, eps=1e-6):
    s = np.clip(s, eps, 1 - eps)
    return -(y * np.log(s) + (1 - y) * np.log(1 - s))


def hier_ci(P, y, fn=auroc, boot=2000, seed=0):
    """Participant x seed hierarchical bootstrap. P is (n_seeds, n_patients)."""
    rs = np.random.RandomState(seed); out = []
    n = P.shape[1]
    for _ in range(boot):
        idx = rs.choice(n, n)
        if len(set(y[idx])) < 2:
            continue
        ss = rs.choice(P.shape[0], P.shape[0])
        out.append(np.mean([fn(y[idx], P[k][idx]) for k in ss]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def hier_ci_paired(A, B, y, fn, boot=2000, seed=0):
    """Same participants and same seed draw for both arms.

    Returns the **observed** difference on the real sample, not the mean of the bootstrap
    replicates. The bootstrap mean is a different quantity — it carries the resampling
    bias — and reporting it as the point estimate would put a number in the table that
    nothing in the data ever produced."""
    rs = np.random.RandomState(seed); out = []
    n = A.shape[1]
    observed = float(np.mean([fn(y, A[k]) - fn(y, B[k]) for k in range(A.shape[0])]))
    for _ in range(boot):
        idx = rs.choice(n, n)
        if len(set(y[idx])) < 2:
            continue
        ss = rs.choice(A.shape[0], A.shape[0])
        out.append(np.mean([fn(y[idx], A[k][idx]) - fn(y[idx], B[k][idx]) for k in ss]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return observed, [float(lo), float(hi)]


def fit_metadata(d, regime, family, seeds):
    """Returns (n_seeds, n_participants) predictions over the full fixed index."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X, _ = feature_frame(d)
    tr = _mask(d, REGIMES[regime]["train"])
    P = []
    for s in seeds:
        clf = (make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, random_state=s))
               if family == "lr" else HistGradientBoostingClassifier(random_state=s))
        clf.fit(X[tr], d.loc[tr, "y"])
        P.append(clf.predict_proba(X)[:, 1])
    return np.stack(P)


def select_reference(d, regime, seeds, families=("lr", "gbm")):
    """Chosen on validation only, then frozen."""
    va = _mask(d, REGIMES[regime]["val"]); y = d.loc[va, "y"].to_numpy()
    scores, preds = {}, {}
    for f in families:
        P = fit_metadata(d, regime, f, seeds)
        preds[f] = P
        scores[f] = float(np.mean([auroc(y, P[k][va]) for k in range(len(seeds))]))
        print(f"  validation AUROC  metadata_{f:<4s} {scores[f]:.4f}")
    best = max(scores, key=scores.get)
    print(f"  -> reference frozen as metadata_{best} on validation; not revisited")
    return best, preds[best], scores


def evaluate(d, regime, preds, ref_name, out_path=None, tag="", extra=None):
    """preds: {arm: (n_seeds, n_participants) array on the full fixed index}."""
    res = {"regime": regime, "reference": ref_name, "arms": {}}
    if extra:
        res.update(extra)          # merged BEFORE the file is written, not after
    tests = REGIMES[regime]["tests"]
    print(f"\n[{regime}] {'arm':<26s}" + "".join(f"{k:>24s}" for k in tests))
    for arm, P in preds.items():
        res["arms"][arm] = {}
        line = f"[{regime}] {arm:<26s}"
        for tname, spec in tests.items():
            m = _mask(d, spec); y = d.loc[m, "y"].to_numpy()
            Pm = P[:, m]
            a = float(np.mean([auroc(y, Pm[k]) for k in range(Pm.shape[0])]))
            sd = float(np.std([auroc(y, Pm[k]) for k in range(Pm.shape[0])]))
            lo, hi = hier_ci(Pm, y)
            res["arms"][arm][tname] = {"auroc": a, "seed_sd": sd, "ci": [lo, hi],
                                       "n": int(m.sum())}
            line += f"  {a:.3f}±{sd:.3f}[{lo:.3f},{hi:.3f}]"
        print(line)

    if "standard" in tests and "matched" in tests:
        print(f"\n[{regime}] G = AUC_standard - AUC_matched  (sensitivity to population "
              f"composition, NOT confounding)")
        for arm in preds:
            g = res["arms"][arm]["standard"]["auroc"] - res["arms"][arm]["matched"]["auroc"]
            res["arms"][arm]["G"] = float(g)
            print(f"[{regime}]   {arm:<26s} {g:+.3f}")

    if ref_name in preds and "matched" in tests:
        m = _mask(d, tests["matched"]); y = d.loc[m, "y"].to_numpy()
        B = preds[ref_name][:, m]
        print(f"\n[{regime}] paired vs {ref_name} on the matched test "
              f"(participant x seed bootstrap)")
        for arm in preds:
            if arm == ref_name:
                continue
            A = preds[arm][:, m]
            dll, dci = hier_ci_paired(A, B, y, lambda yy, s: -logloss(yy, s).mean())
            dau, aci = hier_ci_paired(A, B, y, auroc)
            res["arms"][arm]["paired_vs_ref"] = {
                "delta_neg_logloss": dll, "ci": dci, "delta_auroc": dau, "auroc_ci": aci}
            print(f"[{regime}]   {arm:<26s} Δ(-logloss) {dll:+.4f} [{dci[0]:+.4f},{dci[1]:+.4f}]"
                  f"   ΔAUROC {dau:+.4f} [{aci[0]:+.4f},{aci[1]:+.4f}]")

    if out_path:
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"\nwrote {out_path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/Seagate/resp_datasets/ukcovid")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    d = load(args.data)
    report_splits(d)
    os.makedirs(args.outdir, exist_ok=True)
    np.save(os.path.join(args.outdir, "participant_index.npy"), d[UNIT].to_numpy())
    print(f"\nfixed participant index saved: {len(d)} rows, shared by every arm")

    for regime in ("standard", "naive"):
        print(f"\n=== {regime} regime: reference selection (validation only) ===")
        best, P, scores = select_reference(d, regime, args.seeds)
        ref = f"metadata_{best}"
        preds = {ref: P}
        evaluate(d, regime, preds, ref,
                 os.path.join(args.outdir, f"ukcovid_metadata_{regime}.json"), regime,
                 extra={"validation_selection": scores, "seeds": list(args.seeds)})
        np.savez_compressed(os.path.join(args.outdir, f"preds_{regime}.npz"),
                            **{ref: P}, participants=d[UNIT].to_numpy(),
                            seeds=np.array(args.seeds))
    print("\nAudio arms add entries to `preds` as (n_seeds, n_participants) arrays "
          "on the same index. Nothing else changes.")


if __name__ == "__main__":
    main()
