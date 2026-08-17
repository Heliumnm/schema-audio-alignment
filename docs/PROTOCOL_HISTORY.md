# Protocol history — what each version got wrong, and what replaced it

Three versions of the evaluation protocol exist. Only **v3** is used in the main text.

## v1 — split leakage

`in_matched_original_test` was used as the matched evaluation set for models trained on
Standard train. It overlaps Standard train by 292 and Standard val by 78: **370 of its
1,968 participants, 18.8%, had been seen in training.** `naive_test` was also reported
against a Standard-trained model, overlapping Standard train by 3,003.

Replaced by `in_matched_rebalanced_test`, which overlaps Standard train, val and long by
zero, and by a separate naive regime that trains on `naive_splits == 'train'`. A
disjointness assertion now runs at load time, so a later edit that swaps a split fails
loudly.

## v2 — a highly confounded subset was misidentified as "matched-like"

Regularisation was selected, and the matched calibrator fitted, on the 1,036 participants
with `splits == 'val' AND in_matched_rebalanced_train == True`. Their positive rate is
0.5077, which is why they were taken for a target-like validation set. They are not.

| slice | AUROC(recruitment source → label) |
|---|---:|
| Standard train | 0.9966 |
| Standard val | 0.9984 |
| **the 1,036 subset** | **0.9990** |
| matched test | **0.5000** |
| matched_long test | **0.5000** |

Within those 1,036: REACT 510 negative / 1 positive; Test and Trace 0 negative / 525
positive. Balanced on prevalence, and simultaneously the most confounded slice in the
release.

The one-standard-error rule was also inoperative in v1 and v2 until v2's fold-based fix:
the SE had been estimated across five `LogisticRegression` seeds, and lbfgs is
deterministic, so the measured seed-to-seed difference was exactly `0.000e+00`.

Archived at `results/superseded/v2_misidentified_validation/` with its own README. Its one
surviving contribution is a calibration-source sensitivity: identical models and identical
AUROC give matched Δ(-logloss) of −0.0265 under the 1,036 calibrator and −0.1155 under the
Standard-val calibrator. Appendix material only.

## v3 — one source-domain validation, honestly labelled

**No target-like validation exists in this release and none is claimed.** Everything is
selected in the source domain:

* train on Standard train (20,714);
* select regularisation, early stopping and the single Platt calibrator on the **full
  Standard val** (5,179), five folds stratified by label × recruitment source;
* **matched (1,814) and matched_long (4,196) are out-of-distribution stress tests.** They
  tune nothing and calibrate nothing. Their log-losses measure **calibration transfer out
  of the source domain**, not probability quality after recalibrating inside the target
  population.

No validation set is called "matched-like" anywhere. The rule is not revised again.
