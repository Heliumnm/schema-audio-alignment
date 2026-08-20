# UKCOVID matched direct-fusion control: frozen protocol

Date frozen: 2026-08-20  
Standing: exploratory robustness analysis; official UKCOVID tests informed earlier work.

## Question

The alignment audit shows weak source gains but worse calibrated transfer for correct
pairing relative to within-label shuffling. That contrast alone cannot tell whether the
failure is alignment-specific or whether frozen respiratory audio carries no incremental
disease information once clinical metadata is available. This control asks:

> Does direct concatenation preserve matched-population disease information that the
> metadata-aligned projector fails to preserve?

No result from this analysis can become confirmatory. It is a missing attribution control
inside the already exploratory UKCOVID case study.

## Frozen arms

All arms use the same participant order, Standard train, complete Standard validation,
one-standard-error regularisation rule, final Standard-validation Platt calibrator, five
formal alignment seeds and downstream implementation.

| arm | input to downstream head |
|---|---|
| `metadata_only` | fixed metadata design matrix |
| `metadata_raw_ast` | metadata + frozen raw AST-6L |
| `metadata_correct` | metadata + correct-pairing projector output |
| `metadata_within_label` | metadata + within-label-shuffled projector output |
| `metadata_global` | metadata + globally shuffled projector output |

The three projector arms use the frozen raw post-ReLU representation. `metadata_only` and
`metadata_raw_ast` are deterministic and are copied across the five seed indices only for
paired uncertainty calculations; they are not five independent fits.

## Metadata identity

The metadata matrix contains exactly the semantic fields used by the frozen alignment
text: age, sex, smoking, asthma, other respiratory condition, and every symptom column
except `symptom_onset` and `symptom_prefer_not_to_say`. Every enumerated field level and
`[MISSING]` receives a fixed one-hot column. Categories are defined by `metadata_text.py`,
not inferred from test data. COVID label/test details, recruitment source, timestamps and
recording artefacts are excluded.

## Selection, calibration and locked evaluation

- standardisation is fitted on Standard train only;
- each arm/seed selects logistic-regression C only on complete Standard validation using
  the frozen fold-based one-SE rule;
- the final Platt calibrator is fitted on complete Standard validation;
- Standard test, matched and matched-long choose no parameter;
- per-participant, per-seed probabilities are written before metrics;
- uncertainty uses the existing paired participant x seed hierarchical bootstrap.

The explicit `--evaluate-tests` switch is required. No result-driven arm, field, C grid,
calibrator or representation change is allowed.

## Comparisons and interpretation

Primary attribution control:

1. `metadata_correct - metadata_within_label`, paired Delta(-NLL), matched.

Key secondary controls:

2. `metadata_raw_ast - metadata_only`: direct audio increment;
3. `metadata_correct - metadata_raw_ast`: alignment versus unaligned direct fusion;
4. `metadata_within_label - metadata_global`: label-level co-occurrence;
5. all paired DeltaAUROC and matched-long effects.

Interpretation is frozen:

- raw fusion improves but correct alignment does not: evidence that the tested alignment
  discards or miscalibrates transferable information available to direct fusion;
- neither raw nor aligned fusion improves: the case study cannot attribute the null to
  alignment because the frozen audio representation has no detectable metadata-conditional
  disease increment;
- correct exceeds within-label after fusion: individual pairing contributes at the final
  predictor and the current alignment conclusion must be narrowed;
- intervals spanning zero are reported as imprecise, not equivalence.

