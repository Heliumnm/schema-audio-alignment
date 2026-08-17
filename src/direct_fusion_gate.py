"""Direct-fusion gate — read on the matched validation only. The test sets are not touched.

If AST cannot add information to metadata even when concatenated directly, then whatever
alignment does later cannot be sold as improved diagnosis; the story would have to be
about representation and shortcut mechanism instead. That is worth knowing before a
projector is trained, and it costs nothing.

    metadata_only          the frozen reference
    metadata_artifacts     does recording shape add over symptoms?
    metadata_ast           does the representation add over symptoms?
    metadata_artifacts_ast both

Primary: cross-fitted calibrated Δ(-logloss) against metadata_only. Cross-fitting matters
because the calibrator is fitted on the same 1,036 participants the gate is read on; the
out-of-fold predictions are what keep that honest.
Secondary: paired ΔAUROC.
"""
import os, json
import numpy as np, pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import (UNIT, ARTEFACT_FEATS, auroc, logloss, fit_head, select_C,
                                make_folds, boot_paired, platt, apply_platt)

D = "/mnt/hd/data_heliu/resp_datasets/ukcovid"
META = ["age", "gender", "smoker_status", "respiratory_condition_asthma",
        "respiratory_condition_other"]

C = pd.read_csv("results/ukcovid_audio_cohort.csv")
F = pd.read_csv("results/artefact_features.csv")
p = pd.read_csv(os.path.join(D, "participant_metadata.csv"), low_memory=False)
s = pd.read_csv(os.path.join(D, "train_test_splits.csv"), low_memory=False)
z = np.load("results/ast_embeddings.npz", allow_pickle=True)
assert np.array_equal(z["participants"], C[UNIT].to_numpy())
E = z["embeddings"]

sym = [c for c in p.columns if c.startswith("symptom_") and c != "symptom_onset"]
d = C.merge(F, on=UNIT).merge(p[[UNIT, "covid_test_result", "recruitment_source"] + META + sym], on=UNIT).merge(
    s[[UNIT, "splits", "in_matched_rebalanced_train"]],
    on=UNIT, suffixes=("", "_s"))
order = {q: i for i, q in enumerate(C[UNIT])}
d = d.sort_values(UNIT, key=lambda c: c.map(order)).reset_index(drop=True)
d["y"] = (d.covid_test_result == "Positive").astype(int)
y = d.y.to_numpy()
tr = (d.splits == "train").to_numpy()
# There is no leak-free target-like validation in this dataset: recruitment source
# predicts the label at AUROC 0.9966 in Standard train, 0.9984 in Standard val and 0.999
# in the 1,036-participant subset, while the matched test sets sit at exactly 0.5000. The
# 1,036 are balanced on label prevalence only; they are the most confounded slice, not a
# target-like one. So selection uses the FULL Standard val, and matched/matched_long are
# out-of-distribution stress tests that never tune or calibrate anything.
va = (d.splits == "val").to_numpy()
from sklearn.model_selection import StratifiedKFold as _SKF
folds = np.full(len(d), -1)
_vi = np.where(va)[0]
_st = d.loc[_vi, "y"].astype(str) + "|" + d.loc[_vi, "recruitment_source"].astype(str)
for _k, (_, _te) in enumerate(_SKF(5, shuffle=True, random_state=20260817).split(_vi, _st)):
    folds[_vi[_te]] = _k

M = pd.get_dummies(d[META + sym].astype(object), dummy_na=True).astype(float).to_numpy()
A = d[ARTEFACT_FEATS].to_numpy(float)
X = {"metadata_only": M, "metadata_artifacts": np.concatenate([M, A], 1),
     "metadata_ast": np.concatenate([M, E], 1),
     "metadata_artifacts_ast": np.concatenate([M, A, E], 1)}
print(f"train {tr.sum()}  FULL Standard val {va.sum()}  (test untouched)")
print("source-domain gate: this validation is NOT target-like; recruitment predicts the\n"
      "label here at AUROC ~0.998, and the matched test sets sit at 0.5000")

from sklearn.model_selection import StratifiedKFold
res, P = {}, {}
for name, Xi in X.items():
    print(f"\n--- {name} ({Xi.shape[1]} features) ---")
    C_sel, info = select_C(Xi, y, tr, folds, name)
    raw = fit_head(Xi, y, tr, C_sel)
    # cross-fitted calibration inside the validation set itself
    oof = np.zeros(va.sum())
    vi = np.where(va)[0]
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(vi, y[vi]):
        cal = platt(raw[vi[a]], y[vi[a]])
        oof[b] = apply_platt(cal, raw[vi[b]])
    P[name] = oof
    res[name] = {"C": C_sel, "auroc": auroc(y[vi], oof),
                 "logloss": float(logloss(y[vi], oof).mean())}
    print(f"  source-val AUROC {res[name]['auroc']:.4f}   "
          f"cross-fitted calibrated logloss {res[name]['logloss']:.4f}")

vi = np.where(va)[0]
print("\n=== paired vs metadata_only on the FULL Standard val (source domain) ===")
res["paired"] = {}
for name in X:
    if name == "metadata_only":
        continue
    dl, cl = boot_paired(P[name], P["metadata_only"], y[vi],
                         lambda yy, ss: -logloss(yy, ss).mean())
    da, ca = boot_paired(P[name], P["metadata_only"], y[vi], auroc)
    res["paired"][name] = {"d_neg_logloss": dl, "ll_ci": cl, "d_auroc": da, "auroc_ci": ca}
    st = "excludes 0" if (cl[0] > 0 or cl[1] < 0) else "includes 0"
    print(f"  {name:<24s} Δ(-logloss) {dl:+.4f} [{cl[0]:+.4f},{cl[1]:+.4f}] {st}   "
          f"ΔAUROC {da:+.4f} [{ca[0]:+.4f},{ca[1]:+.4f}]")

json.dump(res, open("results/direct_fusion_gate_stdval.json", "w"), indent=1)
print("\nwrote results/direct_fusion_gate_stdval.json   (no test set was read)")
