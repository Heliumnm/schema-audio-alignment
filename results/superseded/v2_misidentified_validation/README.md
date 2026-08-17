# Superseded — v2 selected and calibrated on a misidentified validation set

**Not a matched-like result. Do not place in the main table.**

v2 selected regularisation and fitted the matched calibrator on the 1,036 participants
satisfying `splits == 'val' AND in_matched_rebalanced_train == True`, which was described
as "matched-like" because its positive rate is 0.5077. That description was wrong.

Recruitment source predicts the COVID label at:

| slice | AUROC(recruitment → label) |
|---|---:|
| Standard train | 0.9966 |
| Standard val | 0.9984 |
| **the 1,036 subset** | **0.9990** |
| matched test | **0.5000** |
| matched_long test | **0.5000** |

Inside those 1,036: REACT 510 negative / 1 positive, Test and Trace 0 negative / 525
positive. It is balanced on label prevalence and is simultaneously **the most confounded
slice in the dataset** — the opposite of target-like. No leak-free target-like validation
exists in this release, and v3 stops pretending one does.

## What survives

One thing, and it is worth keeping: **the population a calibrator is fitted on changes the
matched log-loss enormously.** Same models, same AUROC, different calibrator:

| | matched Δ(-logloss), ast_only − artifacts_only |
|---|---:|
| v2, calibrator from the 1,036 | −0.0265 |
| v3, calibrator from Standard val | **−0.1155** |

AUROC is identical in both, being rank-based. This belongs in an appendix as a
calibration-source sensitivity analysis, nowhere else.

Also retained here: `direct_fusion_gate_1036subset.json`, whose +0.0144 [+0.0066, +0.0220]
increment for `metadata_ast` does not survive on the full Standard val (+0.0032
[-0.0003, +0.0069], includes 0).

Code, configuration, predictions and hashes are kept intact. Nothing here is deleted.
