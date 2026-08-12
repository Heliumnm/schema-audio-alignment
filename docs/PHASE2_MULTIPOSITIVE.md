# Phase 2 — reopening the negative result under a better framing

Phase 1 closed seven directions and concluded that contrastive audio–text alignment
never beat probing the frozen features ([`../README.md`](../README.md)). Phase 2
reopens it under a reframing that survives that result:

> Not *"is schema text better than natural text"* but *"can an explicit structured
> acoustic representation learn a field → spectrogram-patch correspondence"* — where
> the payoff may be **grounding and queryability**, not classification AUROC.

That escape is legitimate, because Phase 1 only ever measured cycle-level
classification. Two Phase-1 conclusions had to be narrowed before it could proceed.

## Two Phase-1 overstatements, corrected

**"Local alignment is closed."** The pooling diagnostic (§2.99) showed the global
mean is the best *readout for cycle-level classification*, with zero headroom from
time-resolved readouts. That closes *"local alignment will raise AUROC"*. It says
nothing about whether patch-level structure can support **localisation**, which is a
different question and the one Phase 2 asks.

**"The negative queue result is clean."** Exact-duplicate texts were masked at
cosine > 0.999, but near-neighbours — schema strings differing in a single tier —
were not. With 3,552 distinct strings over 4,142 training cycles those are common,
so "more negatives made it worse" (§2.92) had a live alternative explanation:
more negatives meant more *false* negatives.

## AST patch geometry — measured, and the stride/receptive-field distinction

Verified on the checkpoint rather than derived from config alone:

```
n_mels 128, max_length 1024, patch 16, stride_f 10, stride_t 10
freq patches (128−16)/10+1 = 12
time patches (1024−16)/10+1 = 101
tokens 12 × 101 + 2 special = 1214   ✓ matches forward pass
patches = h[:, 2:, :].reshape(B, 12, 101, 768)
```

The temporal figure is easy to get wrong, and an earlier draft did:

| quantity | value |
|---|---|
| patch **stride** | ~100 ms |
| patch **receptive field** (16 frames) | ~160 ms |
| overlap between adjacent patches | ~60 ms |

So the correct statement is **not** "crackles cannot be localised". A 5–20 ms crackle
falls inside one or two overlapping patches, which supports **Hit@±1 coarse
pointing** but not credible millisecond IoU. Wheeze (250 ms+) spans 2–3 patches and
is the target for the main grounding experiments.

**Padding dominates and needs occupancy weighting.** AST pads to a fixed 1024 frames
while ICBHI cycles average 2.7 s, so only ~26 of 101 time positions carry signal —
73–74% is padding. A binary mask is not enough: patches straddling the boundary are
partially valid.

```
occupancy(t) = valid fbank frames covered by patch t / 16
  occupancy == 0        hard-mask in cross-attention
  0 < occupancy < 1     keep, down-weight pooling and local loss by occupancy
  occupancy == 1        normal
```

Derive it from the pre-padding fbank frame count, never from `duration / 0.1` rounded.

## Day 1 — three loss conditions

Single → soft-multi changes two things at once (are multiple positives allowed; do
near-neighbours become soft positives). A third condition separates them.

```
S(i,j) = mean over jointly-present fields of (field matches)
fields: crackle_tier, wheeze_tier, wheeze_type, intensity_tier
missing values are excluded from the denominator, not counted as matches
```

| condition | positives | answers |
|---|---|---|
| **single** | diagonal only | the original result |
| **exact** | S = 1.0 | exact-duplicate false negatives |
| **soft** | target ∝ S | near-neighbour false negatives |

Identical optimiser, batch, seeds and epochs; only the target distribution changes.
Condition is never chosen on the test set.

### Diagnostics printed before results — and they mattered

| condition | positives / anchor | target entropy (max 5.545) |
|---|---:|---:|
| single | 1.0 | 0.000 |
| exact | 8.6 (max 15) | 1.933 |
| **soft** | **218.6 (max 247)** | **5.276** |

**The plain soft target is degenerate.** It marks 218 of 256 batch entries as
positive and its entropy sits at 95% of uniform — that is not a contrastive
objective. Its number cannot be read as evidence about near-neighbours.

The original guard only checked whether >25% of *pairs* were exact matches (3.4%, so
it never fired) and missed the flat-target case entirely. Replaced with an explicit
entropy check against the uniform ceiling for every non-trivial condition.

### Results

| condition | AUROC | MCC |
|---|---:|---:|
| single | 0.636 ± 0.021 | 0.200 |
| **exact** | **0.654 ± 0.011** | **0.232** |
| soft *(degenerate)* | 0.651 ± 0.004 | 0.167 ↓ |

Patient-cluster bootstrap, 2,000 resamples over the 47 test patients:

```
exact  ΔAUROC +0.0224  95% CI [−0.0088, +0.0581]  3/3 seeds same sign
soft   ΔAUROC +0.0238  95% CI [−0.0460, +0.0964]  3/3 seeds same sign
```

**Verdict under the pre-registered rule: no effect.** Both point estimates are
positive and consistently signed across seeds, but the patient-clustered CI crosses
zero — underpowered at 47 test patients. The honest reading is *suggestive, not
established*.

**More importantly, the magnitude settles the question anyway:**

```
best multi-positive variant   0.654
no alignment at all (RAW AST) 0.796
                              ───────
                              0.142 deficit
```

A +0.022 correction against a 0.142 deficit. **False negatives are not the
explanation for Phase 1's negative result.**

MCC splits the two conditions in a way consistent with the diagnostics: `exact`
improves it (0.200 → 0.232) while degenerate `soft` *hurts* it (→ 0.167) even as
AUROC holds — AUROC survives a flattened objective, the decision threshold does not.

**Reproducibility check:** `single` scored 0.636, matching the independent Phase-1
run `off_frozen_wheeze_all` (0.636) exactly.

### Follow-up: soft-sharp

Because the soft arm was degenerate it was not a fair test of near-neighbour
correction. Added `soft-sharp`, thresholding at S ≥ 0.75 (≥3 of 4 fields agreeing)
and raising the remainder to the fourth power, keeping near-neighbours as positives
without flattening the target.

## Go / no-go

Pre-registered before running, and unchanged by the results:

1. If multi-positive still fails to improve the templated schema → **pause local
   alignment**, move to the content-matched typed-schema comparison, since the
   bottleneck is then representation rather than loss.
2. Typed schema must beat matched template **at identical information content** for
   "structure" to mean anything.
3. Local alignment must first localise correctly on a synthetic grounding set.
4. Only then ask whether it beats the raw-AST classification baseline.

## Claim boundary for week 1

Week 1 can answer **whether false negatives explain the Phase-1 global-alignment
result**, and **whether typed schema improves the global representation at matched
information**.

It cannot answer whether structure has **grounding** value. That needs the 2D local
alignment and real annotations. Phase 1 produced two overstatements by crossing
exactly this kind of line — prototype scores reported as zero-shot, and a pooling
diagnostic read as closing local alignment — so the boundary is written down rather
than remembered.

## Naming

`T1` carried baggage from Phase 1 and conflated an external baseline with the causal
comparison. Conditions are renamed:

| name | what it is |
|---|---|
| `audio_llm_freeform` | Qwen2-Audio narration — **external baseline only** |
| `matched_template` | deterministic English over the same schema fields |
| `matched_serialized` | `field=value; provenance=…` over the same fields |
| `typed_schema` | structured node encoder over the same fields |

The causal comparison is **matched_template vs matched_serialized vs typed_schema**.
`audio_llm_freeform` cannot carry it: its information content and hallucination rate
differ (its clinical assertions score AUROC 0.509 against ground truth), so it tests
a different variable.

Matching must go down to every field — value bins, confidence and provenance must
appear in the text arms too, or typed schema simply carries more information and the
comparison is not content-matched.

## Annotation plan

The grounding claim depends on annotation, and ICBHI has no event-level timing —
only per-cycle presence flags. So:

- **synthetic sanity set** — inject narrowband wheeze and short crackles at known
  times and frequencies into normal cycles; reuses the fidelity-oracle perturbation
  machinery. Validates the mapping mechanism only, **never a clinical result**.
- **20-cycle wheeze pilot first**, not 200. The pilot fixes: whether one wheeze may
  carry multiple time–frequency boxes; how mono/poly is marked; how uncertain
  boundaries are expressed; how background tones are distinguished from true wheeze;
  and the real per-cycle cost.
- Freeze the guideline before the main pass. Annotators must not see model attention.
- For a "clinical grounding" claim: 100–200 annotated, **30–50 double-annotated**,
  reporting onset tolerance / frequency overlap / box IoU agreement. With a single
  annotator this is **"human-annotated exploratory grounding"**, not clinical
  validation.


---

# Day 2–3 — the loss is not the bottleneck; the representation may be

## Multi-positive: no effect on either target

`soft-sharp` (S ≥ 0.75, then S⁴) was added because the plain soft target was
degenerate. All three variants, both targets, same criterion:

| target | single | exact | soft | soft-sharp |
|---|---:|---:|---:|---:|
| wheeze | 0.636 | 0.654 | 0.651 *(degenerate)* | 0.626 |
| crackle | 0.689 | 0.693 | 0.676 | 0.681 |

Every Δ fails: CI crosses zero, or seeds disagree, or both. **soft-sharp is *worse*
than soft** (0.626 vs 0.651 on wheeze) — sharpening removes the near-neighbour
positives and lands back near `single`, which means soft's apparent +0.024 came from
target flattening, not from near-neighbour correction. Running only `single` vs
`soft` would have supported the opposite reading; the third condition is what
separates them.

**Go/no-go criterion 1 fires: pause local alignment.** False negatives do not explain
the Phase-1 negative result.

## Content-matched string arms: format does nothing, and the apparent gain was a confound

Schema v2 emits one field dict three ways, asserted content-matched down to the value
strings. An early version silently failed this: `str.capitalize()` lowercased every
value in the template arm ("Meditron" → "meditron") while the other arms kept case,
so the arms differed at exactly the level the tokeniser sees. The assertion checked
serialized-vs-typed only and passed anyway; it now covers the template arm too.

| arm | wheeze AUROC |
|---|---:|
| RAW AST (no alignment) | **0.796** |
| matched_serialized | 0.696 |
| matched_template | 0.675 |

+0.021 looked like the first evidence for structure. It is not — **the sign flips
when the recording block is removed**:

| condition | template | serialized | Δ | verdict |
|---|---:|---:|---:|---|
| with recording (has `duration_s`) | 0.675 | 0.696 | +0.026, CI [−0.017, +0.073] | no effect |
| acoustic only | 0.659 | 0.651 | **−0.021**, seeds disagree | no effect |

v2's text carries continuous `duration_s`, and duration alone reaches AUROC 0.644 on
crackle. It also drives the diversity: 6,812 distinct strings with the recording
block, 1,430 without. **Serialisation format does nothing at matched content; the
apparent effect was the duration confound arriving through the text.**

## Typed set encoder: the first result to pass the criterion

| arm | wheeze AUROC | MCC |
|---|---:|---:|
| string_frozen | 0.664 ± 0.016 | 0.206 |
| string_trainable *(capacity-matched)* | 0.660 ± 0.007 | 0.202 |
| **typed** | **0.706 ± 0.015** | **0.249** |

```
string_trainable vs string_frozen   Δ +0.0013  CI [−0.083, +0.075]  seeds disagree  -> no effect
typed vs string_trainable           Δ +0.0495  CI [+0.0047, +0.0954]  3/3 same sign -> IMPROVES
typed vs string_frozen              Δ +0.0511  CI [−0.024, +0.129]    3/3 same sign -> no effect
```

Two things this establishes:

**Capacity is not the explanation.** Giving the string arm a trainable text MLP of
the same shape moves nothing (+0.0013, seeds disagree). The confound that would have
made this uninterpretable is ruled out by the arm built to rule it out.

**typed beats the capacity-matched string arm, and it is the first comparison in this
project to clear the pre-registered bar.** Go/no-go criterion 2 — *typed must beat
matched template at identical content for "structure" to mean anything* — is met.

### Stated precisely, because the margin is thin

The lower CI bound is +0.0047. The comparison against `string_frozen` has a slightly
*larger* point estimate (+0.0511) but a wider CI that crosses zero, because
`string_frozen` has higher seed variance (±0.016 vs ±0.007) and its scores correlate
less with typed's. So: **the comparison designed to isolate structure passes; the
comparison against the original baseline does not.** Real, but marginal, and on one
target.

**And it still loses to doing nothing.** typed 0.706 vs RAW AST 0.796. Structure
helps *within* alignment; alignment still costs 0.09 against not aligning.

## Tension between the two go/no-go criteria

Criterion 1 (multi-positive fails → pause local alignment) fired. Criterion 2 (typed
must beat matched template) passed, and it establishes the premise local alignment
needs — that the schema carries usable structure. They point opposite ways, so
resuming local alignment is a judgement call rather than something the rules settle.

Two cheap things run first, because a marginal single-target result should not be
built on:

- **crackle replication** — does it hold on the other target;
- **shuffle controls** — permute whole schema records, permute each field's values
  independently, and strip field identity. A gain that survives shuffling is not
  coming from schema content.
