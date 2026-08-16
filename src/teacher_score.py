"""Teacher check — scoring and the gate. No GPU, no model, no forward passes.

Kept apart from `teacher_run.py` on purpose: the gate can be recomputed from stored scores
without touching the model, and the script that produces scores cannot be edited into one
that also decides the outcome.

The score
---------
Averaging the two option orders **in probability space is wrong**, and the dev diagnostic
showed why: under one order the model puts 0.93 on the second option with a standard
deviation of 0.03, under the other it sits near 0.5 with a standard deviation of 0.11. The
two orders are on different scales and one of them is saturated, so their arithmetic mean
is a point between two failure modes rather than an estimate of anything.

The position bias is additive in **log-odds**, so it cancels there:

    z = ½ [ log(P_wet/P_dry) under the normal order  +  log(P_wet/P_dry) under the swap ]

With options `A) dry  B) wet` normally and `A) wet  B) dry` swapped, that is

    z = ½ [ log(P(B)/P(A))_normal + log(P(A)/P(B))_swapped ]

exactly. A constant preference for whichever letter comes second enters the two terms with
opposite sign and drops out; what survives is the part that tracks the audio.

The decision statistic
----------------------
Per-annotator AUROC, then the **equal-weight macro-average**. Pooled AUROC is computed and
printed, and it decides nothing: within an annotator the wet rate is fixed, so the
macro-average closes the route where a model scores well by recognising which batch of
recordings an annotator saw.

    python src/teacher_score.py --scores results/dev_diag_scores.csv \\
        --manifest results/dev_diag_manifest.csv
"""
import argparse
import numpy as np
import pandas as pd

EPS = 1e-9


def symmetrised_logodds(S):
    """S has one row per (uuid, arm, prompt, swap). Returns z per (uuid, arm, prompt)."""
    S = S.copy()
    # p_A / p_B are raw masses on the two option letters; wet is B normally, A when swapped
    wet = np.where(S["swap"], S["p_A"], S["p_B"])
    dry = np.where(S["swap"], S["p_B"], S["p_A"])
    S["lo"] = np.log(np.clip(wet, EPS, None)) - np.log(np.clip(dry, EPS, None))
    z = S.groupby(["uuid", "arm", "prompt"], as_index=False)["lo"].mean()
    n = S.groupby(["uuid", "arm", "prompt"])["swap"].nunique()
    assert (n == 2).all(), "a recording is missing one of the two option orders"
    return z.rename(columns={"lo": "z"})


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def macro_auroc(df, boot=0, seed=0):
    """Per-annotator AUROC, then equal-weight macro-average. Optional recording-level
    bootstrap resampling within each annotator."""
    per = {}
    for a, g in df.groupby("annotator"):
        y = (g.label == "wet").astype(int).to_numpy()
        if len(set(y)) < 2:
            continue
        per[int(a)] = auroc(y, g.z.to_numpy())
    macro = float(np.mean(list(per.values()))) if per else np.nan
    if not boot:
        return macro, per, None
    rs = np.random.RandomState(seed); out = []
    groups = {a: g.reset_index(drop=True) for a, g in df.groupby("annotator")}
    for _ in range(boot):
        vals = []
        for a, g in groups.items():
            idx = rs.choice(len(g), len(g))
            y = (g.label.iloc[idx] == "wet").astype(int).to_numpy()
            if len(set(y)) < 2:
                continue
            vals.append(auroc(y, g.z.iloc[idx].to_numpy()))
        if vals:
            out.append(np.mean(vals))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return macro, per, [float(lo), float(hi)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    S = pd.read_csv(args.scores)
    M = pd.read_csv(args.manifest)[["uuid", "annotator", "label", "quality",
                                    "duration_tercile"]]
    Z = symmetrised_logodds(S)
    D = M.merge(Z[Z.arm == "intact"], on="uuid")
    print(f"{len(D)} annotator rows over {D.uuid.nunique()} recordings, "
          f"{D.prompt.nunique()} prompts")
    print("\nper-annotator label counts")
    print(D[D.prompt == D.prompt.iloc[0]].groupby(["annotator", "label"]).size()
          .unstack(fill_value=0).to_string())

    print(f"\n{'prompt':<15s}{'macro-AUROC':>13s}{'95% CI':>20s}   per annotator")
    best, rows = None, {}
    for p, g in D.groupby("prompt"):
        macro, per, ci = macro_auroc(g, boot=args.boot)
        rows[p] = (macro, per, ci)
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        ps = "  ".join(f"a{a}:{v:.3f}" for a, v in sorted(per.items()))
        print(f"{p:<15s}{macro:13.3f}{cis:>20s}   {ps}")
        if best is None or macro > rows[best][0]:
            best = p

    from sklearn.metrics import roc_auc_score
    print("\npooled AUROC (printed for completeness; decides nothing)")
    for p, g in D.groupby("prompt"):
        y = (g.label == "wet").astype(int)
        print(f"  {p:<15s} {roc_auc_score(y, g.z):.3f}")

    sil = Z[Z.arm == "silence"].merge(M, on="uuid")
    print("\nsilence arm mean z (model prior, auxiliary)")
    for p, g in sil.groupby("prompt"):
        print(f"  {p:<15s} {g.z.mean():+.3f}")

    macro, per, ci = rows[best]
    n_above = sum(1 for v in per.values() if v > 0.5)
    print(f"\ndev feasibility gate — best prompt '{best}'")
    print(f"  macro-AUROC {macro:.3f} (need >= 0.60)         "
          f"{'PASS' if macro >= 0.60 else 'FAIL'}")
    print(f"  annotators with AUROC > 0.5: {n_above}/3 (need >= 2)   "
          f"{'PASS' if n_above >= 2 else 'FAIL'}")
    if macro >= 0.60 and n_above >= 2:
        print("\n  -> worth completing all 427 dev recordings.")
    else:
        print("\n  -> the development-set feasibility screen did not establish reliable "
              "audible evidence\n     under the current Qwen2-Audio, the COUGHVID "
              "annotations and this forced two-way\n     measurement protocol. The held-out "
              "report slice is not touched and no LLM\n     reasoning captions are "
              "generated. This is NOT a finding that the model cannot\n     hear wet cough, "
              "and NOT a finding that the signal does not exist: label noise,\n     model "
              "capability and protocol failure are not separable here.")


if __name__ == "__main__":
    main()
