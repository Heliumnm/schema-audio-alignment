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


---

# Day 3 controls — the typed result does not survive them

Both pre-registered controls came back against it.

## Shuffling the schema barely dents the gain

| schema | typed AUROC | Δ vs string_trainable | verdict |
|---|---:|---:|---|
| **intact** | **0.706 ± 0.015** | **+0.0495** CI [+0.005, +0.095] | IMPROVES |
| records permuted across segments | 0.651 ± 0.039 | +0.0323 | no effect |
| each field's values permuted | 0.650 ± 0.043 | +0.0376 | no effect |
| field identity stripped | 0.689 ± 0.011 | +0.0347 | no effect |

The shuffled arms fail the criterion because seed variance rises, **not because the
effect disappears** — the point estimates fall only from +0.0495 to +0.032…+0.038.
Destroying the audio↔schema correspondence entirely leaves roughly three quarters of
the advantage in place.

So at most ~+0.012–0.018 of the +0.0495 is attributable to schema *content*; the rest
comes from the set encoder's architecture or optimisation behaviour. That residual is
inside the noise band of effects already rejected.

## Crackle does not replicate — it reverses

| arm | crackle AUROC | MCC |
|---|---:|---:|
| **string_frozen** | **0.703 ± 0.009** | 0.293 |
| string_trainable | 0.694 ± 0.009 | 0.265 |
| typed | 0.666 ± 0.008 | 0.224 |

`typed vs string_trainable` is **−0.0292**, 3/3 seeds agreeing on the negative sign.
On the second target the typed encoder is *worse* than the strings it beat on the
first.

## Retraction

The previous entry recorded criterion 2 (*typed must beat matched template at
identical content*) as met. **It is not.** That reading rested on a single target
with the controls still running. With them in:

- not replicable — the sign reverses on crackle;
- not schema content — most of the gain survives shuffling the schema.

Both go/no-go criteria now point the same way, and the tension between them
disappears.

**This is what the pre-registered controls were for.** Acting on the Day-2 reading
would have meant 3–4 weeks of local alignment built on an effect that reverses on the
other target and persists when its supposed cause is destroyed. The two controls cost
under an hour.

## Phase 2 outcome

| # | hypothesis | verdict |
|---|---|---|
| 8 | False negatives explain the Phase-1 negative result | ❌ no effect, both targets, 3 loss variants |
| 9 | Serialisation format matters at matched content | ❌ sign flips with the duration confound |
| 10 | A typed set encoder beats strings at matched content | ❌ reverses on crackle; survives schema shuffling |

Three more hypotheses, controlled, on top of Phase 1's seven. The grounding reframing
remains untested — but it now rests on no evidence that the schema representation
carries usable structure at all, and the annotation needed to evaluate it has not been
collected.


---

# Day 4 — the Day-3 rejection was itself wrong

The Day-3 entry rejected hypothesis 10 on the strength of "shuffling keeps ~3/4 of
the gain". **That reading is withdrawn.** It compared each shuffled arm against
`string_trainable`; the question is `typed_intact` vs `typed_shuffled`, and on that
axis:

```
intact 0.706  ->  record-shuffled 0.651     a drop of 0.055
```

which points the *opposite* way — toward correspondence mattering. Four defects made
the number unusable regardless:

| defect | why it invalidates |
|---|---|
| permutation applied to train **and** test | the test schema was scrambled too, so the arm was never "trained on noise, evaluated honestly" |
| one fixed permutation | the shuffled score carries no sampling variance |
| `fields` control zeroed the id channel only | value keys are `field#value`, so field identity leaked straight back |
| bootstrap on seed-averaged predictions | the table reported seed-averaged AUROC — different estimands |

## Two loss bugs, fixed

- `S[i,i]` is already 1.0 (a record matches itself on every field), and the code then
  added an identity — **the anchor's own positive was double-weighted**. Now clamped
  with `maximum` instead.
- The bidirectional loss reused the row-normalised `Q` against `lg.T`. `P/rowsum` is
  not symmetric even when `S` is, so the reverse direction was targeting the wrong
  distribution. It now normalises `S.T` separately.

Neither is likely to close a 0.14 gap, but the multi-positive conclusion needs one
clean re-run before it is stated as final.

## Two conclusions narrowed

**"The +0.021 serialisation gap was the duration confound."** Overstated — both arms
contain duration in the with-recording condition, and *both* CIs cross zero. The
defensible statement is only: **no reproducible serialisation-format effect was
found.** The sign flip is as consistent with small-sample noise as with a mechanism.

**"Schema carries no usable structure."** Out of scope. What was tested is a *coarse
categorical* schema — low/medium/high tiers, wheeze type, crackle character,
cycle-level likelihood. The fine-grained schema actually proposed — event start/end,
frequency interval, raw pitch in Hz, duration, harmonicity, periodicity, confidence,
phase, evidence mask — was never built. The result covers coarse typed encoding for
global classification, nothing more.

**Crackle is not a strict replication of wheeze.** The typed encoder is trained by a
target-agnostic contrastive loss; the target enters only at the linear probe
afterwards. Wheeze up and crackle down means one representation serves two downstream
labels differently — not that the effect failed to replicate. Real replication needs
another patient split, group cross-validation, or another wheeze-labelled corpus.
Crackle also carries the device, location, duration and time-resolution problems
already documented.

## The audit now running

Pre-registered before launch, unchanged by any result:

> `typed_intact` beats record-shuffled, **and** all 5 seeds agree in sign, **and** the
> paired patient-cluster bootstrap CI on (intact − shuffled) excludes zero.

- shuffle applied to **train only**, test schema untouched;
- **10 permutations**, so the shuffled arm has sampling variance;
- **constant-schema** arm — every segment gets identical schema, removing all
  per-sample signal;
- **nofield** arm strips field identity from the value keys as well as the id channel;
- paired bootstrap computed **within each seed**, then aggregated;
- per-test-id predictions saved, not just AUROC.

This answers whether the coarse schema carries global-representation signal. It does
not touch grounding, which stays a separate question that global AUROC cannot stand
in for.


---

# Day 4 audit — the constant-schema control settles it

Rebuilt with train-only shuffling, 10 permutations, per-seed paired bootstrap and
proper field-identity removal. wheeze, acoustic block, 5 seeds.

| arm | AUROC |
|---|---:|
| *RAW AST, no alignment (reference)* | *0.796* |
| **constant — every segment gets the same schema** | **0.750 ± 0.018** |
| intact | 0.705 ± 0.011 |
| nofield | 0.699 ± 0.030 |
| record-shuffled (10 permutations) | 0.650 ± 0.025 |

```
intact − record-shuffled   Δ +0.0510  CI [−0.0127, +0.1248]  5/5 same sign  -> inconclusive
intact − constant          Δ −0.0458  CI [−0.1181, +0.0295]  5/5 same sign  -> inconclusive
intact − nofield           Δ +0.0068  CI [−0.0936, +0.1289]  signs disagree -> inconclusive
```

**The pre-registered criterion is not met.** `intact > shuffled` is +0.051 with all
five seeds agreeing, but the paired CI crosses zero. No correspondence signal is
established.

## What the constant arm shows

A constant schema removes the audio↔text correspondence entirely — every text
embedding is identical, so InfoNCE has nothing to discriminate. It scores **higher
than the real schema**, with all five seeds agreeing on the sign.

The full ordering is monotone and mechanistically coherent:

```
no alignment    0.796
constant        0.750     no correspondence      (neutral)
intact          0.705     true correspondence
shuffled        0.650     false correspondence   (actively wrong)
```

So the variable is not *whether the schema carries structure* but **how much the
alignment objective damages the features**: none < uninformative < correct <
incorrect. Any per-sample text target costs something; wrong targets cost more than
right ones. Phase 1's conclusion arrives again by a wholly independent route.

`nofield ≈ intact` also says field identity contributes nothing once values are
present.

## Verdict on hypothesis 10

**Rejected** — but on the constant control, not the shuffle comparison the Day-3
entry used. The Day-3 rejection reached the right answer through an analysis that
did not support it; this one is properly powered and points the same way, plus it
supplies a mechanism the earlier reading lacked.

Worth noting which control did the work: the shuffle comparison, which both the
Day-3 analysis and the audit's own primary criterion rest on, came back
**inconclusive**. The decisive evidence came from the constant-schema arm added
during review.

## Where step 1 leaves things

Answered: **the coarse categorical schema carries no global-representation signal**,
and per-sample text alignment degrades the features monotonically with how wrong the
correspondence is.

Not answered: whether a **fine-grained** schema — event onsets, frequency intervals,
raw Hz, harmonicity, confidence, phase — supports field→patch grounding. That schema
was never built, and global AUROC cannot stand in for grounding. Step 2 is the small
synthetic wheeze pilot, which needs no manual annotation to test whether the mapping
mechanism works at all.


---

# Step 2 — synthetic grounding pilot: the first positive result

Grounding is a different question from global classification, and Step 1 answered
only the latter. No manual annotation exists, so this uses synthetic injections with
known onset, duration and frequency. **It validates the mechanism only and is never a
clinical result.**

## The single-injection version could not test the hypothesis

| scorer | Hit@1 | Hit@±1 | freq_acc | 2D |
|---|---:|---:|---:|---:|
| typed_intact | 0.584 | 0.965 | 0.962 | 0.567 |
| **shuffled_coords** | **0.613** | **0.977** | 0.955 | **0.586** |
| energy_tonality | 0.012 | 0.089 | 0.158 | 0.000 |

Shuffling the query coordinates changed nothing, and a +0.4 s query shift moved the
peak by +0.10 patches. The injected tone is conspicuous enough that the scorer found
it straight from the patch features and **never read the query** — so the task could
not discriminate the hypothesis either way. The `energy_tonality` baseline existed to
catch exactly this and was too weak to (0.089).

This is a design failure, not a negative result: the experiment did not test the
thing. Fixed by making the query load-bearing rather than by relaxing anything.

## Two injections make the query the only way to choose

Each clip gets two injections, separated in both time and frequency; the query names
one. Ignoring it caps accuracy near 50%.

| scorer | Hit@1 | Hit@±1 | freq_acc | 2D | lands on distractor |
|---|---:|---:|---:|---:|---:|
| **typed_intact** | **0.617** | **0.935** | **0.894** | **0.582** | **0.050** |
| shuffled_coords | 0.305 | 0.494 | 0.488 | 0.290 | 0.456 |
| energy_tonality | 0.026 | 0.099 | 0.081 | 0.000 | 0.000 |
| random | 0.029 | 0.102 | 0.094 | 0.003 | 0.010 |

`shuffled_coords` now behaves as a correspondence control should — 0.494 Hit@±1 and
0.456 on the distractor is a coin flip between the two injections. `typed_intact`
lands on the distractor 5% of the time.

## Query swap: the decisive test

The +0.4 s nudge cannot discriminate in this design — the injections sit ~1 s apart,
so the shifted point is still nearest the original and staying put is the *correct*
answer. Swapping the query to the **distractor's** coordinates is the real test:

```
peak lands on the distractor   0.901
peak stays on the original     0.031
```

Asked for the other injection, it points at the other injection. The query drives the
choice.

## Verdict

**GO criterion passes**, on unseen patients *and* an unseen frequency band
(1100–1500 Hz never appears in training):

- 0.935 Hit@±1 with the correct query;
- 0.901 follow-through when the query moves to the distractor;
- both controls at chance or below.

This is the **first positive, controlled result in the project**. AST's patch
features do carry time–frequency localised information, and a schema node can address
it.

**What it does not establish.** Injections are additive tones; real wheezes arise from
airway dynamics and sit inside the breath sound rather than on top of it. The
pre-registered limit stands: synthetic evidence validates the mapping mechanism, never
clinical grounding.

Under the plan's ordering, passing here is what justifies the **20-cycle real wheeze
annotation pilot** — which is now the next step, and the one thing in this project
that cannot be done without listening.
