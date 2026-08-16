# Teacher check — dev feasibility NO-GO

**This is a development-set feasibility screen that did not pass. It is not a test-set
result, and it is not a finding about what Qwen2-Audio can hear.**

## What was run

Sixty recordings from the **dev slice only**, 720 forward passes, format compliance
1.0000. Three prompts × two option orders × two arms (intact, silence).

**The held-out report slice — 990 recordings, 1,098 annotator rows — was never read.**
No LLM reasoning captions were generated. The remaining 427 dev recordings were not run.

## The measurement had to be corrected first

The pre-registration averaged the two option orders in probability space. The raw
diagnostic showed that is the wrong ruler rather than a noisy one:

| | `plain`, intact |
|---|---|
| normal order | mean 0.925, **sd 0.030** |
| swapped order | mean 0.471, **sd 0.108** |
| correlation between orders | r = +0.39 |
| position effect | **0.454** |
| dry/wet effect within one order | **0.003** |

The two orders sit on different scales and one is saturated, so their arithmetic mean is a
point between two failure modes. Position bias is additive in log-odds, so the score is
symmetrised there:

$$z = \tfrac{1}{2}\left[\log\tfrac{P_{\text{wet}}}{P_{\text{dry}}}\Big|_{\text{normal}} + \log\tfrac{P_{\text{wet}}}{P_{\text{dry}}}\Big|_{\text{swapped}}\right]$$

A constant preference for whichever letter comes second enters the two terms with opposite
sign and cancels.

## Result

Per-annotator AUROC, then equal-weight macro-average. Pooled AUROC decides nothing.

| prompt | macro-AUROC | 95% CI | a1 | a2 | a3 |
|---|---:|---|---:|---:|---:|
| `cue` | **0.451** | [0.272, 0.653] | 0.375 | 0.494 | 0.484 |
| `cue_evidence` | 0.436 | [0.256, 0.634] | 0.375 | 0.494 | 0.440 |
| `plain` | 0.446 | [0.262, 0.652] | 0.375 | 0.494 | 0.467 |

The `silence` arm leans wet with no audio at all, z = +0.34 to +0.62.

**Dev screen: best-prompt macro-AUROC ≥ 0.60 → 0.451, FAIL. At least two annotators above
0.5 → 0/3, FAIL.**

## Why the wide interval did not change the decision

The macro CI upper bound (0.653) crosses 0.60, because annotator 1 contributes three wet
cases and annotator 2 contributes five. That width means the estimate is imprecise; it is
not evidence of a positive signal.

This is a **pass-to-continue** gate, not an attempt to prove the model ineffective.
Insufficient evidence stops the route. Widening the rule because the interval is wide
would be moving the threshold after seeing the result, which is the failure mode this
project has retracted results to before.

Nine numbers — three prompts × three annotators — and not one exceeds 0.5.

## The recorded conclusion

> Under the current Qwen2-Audio, the COUGHVID annotations and this forced two-way
> measurement protocol, the development-set feasibility screen did not establish reliable
> audible evidence. The held-out report slice is therefore not entered and no LLM
> reasoning captions are generated.

**This may not be written as "Qwen2-Audio cannot hear wet cough" or "the signal does not
exist."** Label noise, model capability and protocol failure are not separable here:
annotator wet rates run 14.2% / 18.9% / 46.5%, 1,324 of 1,417 recordings carry a single
annotator, and only 70 recordings have two or more annotators who agree.

## Consequence for the paper

The audible-evidence alignment arm is closed. The remaining plan does not depend on it:
the question becomes whether aligning audio to clinical metadata adds disease information
or mainly writes population and symptom shortcuts into the audio representation. That is
answered with UKCOVID's own matched evaluation, a mismatched-metadata control, and probes
that decode age, sex, symptoms and recruitment source out of the audio representation —
none of which needs an LLM caption.

## Artefacts

`results/dev_diag_scores.csv` (per-recording `p_A`/`p_B`, prompt, order, arm),
`results/dev_diag_manifest.csv`, `results/dev_diag.log`, `src/teacher_run.py`,
`src/teacher_score.py`, `src/teacher_prompts.py`, and the frozen
`results/teacher_split_manifest.csv` whose report slice remains unread.
