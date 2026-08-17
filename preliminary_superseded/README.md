# Superseded — do not cite these numbers

The three audio baselines and the raw-AST probes in this directory were produced by a
protocol with three implementation faults. They are kept as the record of what was run,
not as results.

**1. The one-standard-error rule never executed.** The standard error was estimated across
five `LogisticRegression` seeds, but that estimator is deterministic under lbfgs — measured
seed-to-seed difference is exactly `0.000e+00` on all three arms. The SE was therefore
zero and the rule collapsed to plain argmax of validation AUROC. That is why `ast_only`
and `artifacts_plus_ast` selected the weakest regularisation available, `C = 3.0`.

**2. The same fault made the seed axis vacuous.** The "participant × seed hierarchical
bootstrap" resampled five identical copies, so every interval reported was in fact a
participant-only bootstrap. The intervals are not wrong, but their description was.

**3. One calibrator was used for two prevalences.** A Platt calibrator fitted on the
50/50 matched-like validation set was applied to the Standard test set, whose prevalence
is 0.3435. The Standard calibrated log-losses here are therefore uninterpretable. AUROC is
unaffected, being rank-based.

**4. Probe predictions were not saved.** `ast_probe_preds.npz` holds participant IDs and
nothing else, so no probe here can be paired against a later aligned representation.

**5. Missing probe labels were imputed as negative** via `fillna(0)` instead of being
excluded.

Also corrected in the summary that accompanied these numbers: `artifacts_only` selected
`C = 0.001` on the primary validation, not `C = 3.0`. Only the two AST arms selected 3.0.

Two claims made from these numbers are downgraded and must not be repeated as they stood:

* "AST has already encoded the artefacts" — the evidence was only that adding artefact
  features barely moved the point estimate. The supportable claim is exactly that. Testing
  the stronger one needs `artifacts+AST − AST` with a paired interval, and a direct decode
  of duration, RMS, file size and clipping from the AST representation.
* "calibration collapsed" — the evidence was one calibrated log-loss, produced by the
  mismatched calibrator described above. Establishing it needs calibration slope and
  intercept, Brier score and a reliability curve.
