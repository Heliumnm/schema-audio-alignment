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


---

# Step 2 RETRACTED — the query contained the answer

The pass is withdrawn. The query was

    q = [onset + dur/2,  dur,  f0/1000]

while the target is `t_idx = f(onset + dur/2)` and `f_idx = g(f0)` — both
**deterministic functions of the query**. The task was coordinate arithmetic; the
audio was never required.

| scorer | Hit@±1 | freq_acc | 2D |
|---|---:|---:|---:|
| typed_intact | 0.935 | 0.894 | 0.582 |
| **coordinate_only — no audio at all** | **1.000** | **1.000** | **1.000** |
| audio_shuffled | 0.326 | 0.391 | 0.047 |
| dsp_band_energy | 0.306 | 1.000 | 0.118 |

A no-audio baseline scores **perfectly**, above the learned model. Every Step-2
number is explained without listening: shuffled queries break the arithmetic, and
swapping the query to the distractor's coordinates routes to the distractor's patch.

`audio_shuffled` at 0.326 does show the learned scorer uses acoustic content — but
that rescues nothing. The task does not *require* content, so there is no way to tell
grounding from a noisier route to the same coordinate answer. **When a no-audio
baseline is perfect, no positive number on that task is interpretable.**

The `energy_tonality` baseline that existed to catch this was itself broken: it scored
`np.linalg.norm(P, axis=-1)`, the magnitude of AST's embeddings, never touching the
spectrogram. Its 0.099 said nothing about difficulty. The replacement DSP baseline
leaks too — its `boost` term injects the queried frequency row directly, which is why
its `freq_acc` is 1.000.

## What this cost, and what it says about the process

Step 2 was reported as "the first positive, controlled result in the project". It was
neither positive nor controlled. The controls that exposed it — a coordinate-only
baseline and an audio-content ablation — are the obvious ones for any grounding claim
and should have been in the first design, not added after a pass was announced.

The pattern to note: the previous single-injection failure was correctly diagnosed as
a task-design flaw, the fix made the query load-bearing for *choosing between two
events*, and the check stopped there. It never asked whether the query determined the
answer outright.

## Redesign — one attempt, criteria fixed in advance

The query must not contain what is being localised.

| evaluating | query may contain | query may not contain |
|---|---|---|
| time | type, pitch category, mono/poly | onset, offset |
| frequency | type, duration, phase | exact frequency |
| 2D | type, morphology, phase | onset and exact Hz |

Design: each clip carries two acoustically distinct events — **A** monophonic with a
stable fundamental, **B** polyphonic with frequency modulation — whose positions are
randomised independently. The query names only `character: polyphonic`, so the model
must follow the acoustic content to find which one it is.

Go criteria, all required:

1. `coordinate_only` and `attribute_only` fall to chance — the query cannot locate;
2. audio zeroed or shuffled drops to chance;
3. swapping the queried *attribute* moves the peak to the other event;
4. intact beats shuffled-query, random and a correctly implemented DSP baseline;
5. holds on unseen patients and unseen synthesis parameters.

**One redesign, one run.** If it fails, grounding is sealed on this data rather than
re-specified again.

## Redesign attempt 1 (v2): retracted before it could be used

The v2 design replaced the coordinate query with a single bit — *which character to
locate*, monophonic or polyphonic. That much worked: one bit cannot encode a position,
so exact coordinate leakage is impossible by construction.

It leaked anyway, through the target mask.

Poly's ratios `(1, 1.37, 1.81)` were normalised to unit geometric mean, and mono's were
not. `{1, 2}` has geometric mean √2 = 1.414, so **mono's spectral centre sat 41% above
poly's** — the same class of confound the normalisation was introduced to remove, moved
to the other arm and made larger. Mono also spanned a full octave against poly's 1.81,
and the mask was the union of the occupied rows, so mask extent and patch count tracked
the character too (mono 11.0 ± 1.9 patches, poly 13.6 ± 3.6).

A probe on **crop and mask geometry alone, with no audio of any kind**, separated the
two characters at **AUROC 0.982**:

| feature, no audio | AUROC |
|---|---:|
| crop origin row | 0.782 |
| highest occupied row | 0.790 |
| row span | 0.744 |
| number of occupied rows | 0.740 |
| **all geometry, logistic regression** | **0.982** |

The v2 gate reported AUROC 1.000 with a zero-width CI. That number measured where the
crop was, not what was inside it. **The v2 feasibility pass is withdrawn.**

The perfect score is what exposed it. A gate that clears its threshold by the largest
possible margin is evidence about the design, not about the model.

## Redesign attempt 2 (v2.1): geometry equalised by construction

The rule the first three attempts each violated in a different way: **anything the
target mask reveals is a leak.** v2.1 enforces it structurally instead of checking for
it afterwards. Both characters are three simultaneous tones with endpoints fixed at
`{1, 4} × fc`:

    harmonic     fc × {1, 2.000, 4}        middle partial = 2nd harmonic
    inharmonic   fc × {1, 1.550, 4}  or    middle partial displaced ∓0.368 octave,
                 fc × {1, 2.581, 4}        counterbalanced

so that identical endpoints give identical `row(fc)` and `row(4fc)`; the mask is the
**whole band** rather than the union of occupied rows; component count is equal; and the
symmetric, counterbalanced displacement matches the mean log spectral centroid exactly.
`fc` is drawn from the admissible set on which the three middle rows are strictly
ordered — 621 values in 362–1199 Hz, 73.7% of the range. Admissibility does not depend
on character, so the restriction cannot leak either.

The only remaining difference is **which row inside an identical band carries energy**.

### What this costs

This is no longer a monophonic-vs-polyphonic wheeze proxy. Clinical mono/poly wheezes
differ in how many tones sound at once, and component count is precisely what had to be
equalised. What remains is harmonic vs inharmonic partial structure — a mechanism probe
on whether AST patches carry resolvable spectral fine structure, and nothing more.

### Feasibility gate, 695 clips, official TRAIN split only

| arm | AUROC | 95% CI |
|---|---:|---|
| **geometry only (no audio)** | **0.477** | [0.412, 0.538] |
| harmonic vs inharmonic | 0.999 | [0.997, 1.000] |
| event vs background | 1.000 | [1.000, 1.000] |

Matching checks: log₂ centroid difference **0.0002 octave** (v2: 0.500), mask patch
difference 0.28 (v2: 2.65), harm-first 0.5022, upward-variant 0.5007, region overlap
0.000, geometry in-sample upper bound 0.544 (v2: 0.982).

**Gate passed.** The geometry arm is at chance, so the 0.999 is acoustic. Note what that
implies for the grounding task that follows: discrimination at a *known* location is
essentially free, so all of the difficulty now sits in localisation. A grounding failure
from here cannot be blamed on the encoder being unable to hear the manipulation.

The AST frequency grid is the binding constraint throughout. Twelve rows cover
0–8000 Hz, with boundaries at 205 / 405 / 645 / 935 / 1290 / 1720 / 2250 Hz — roughly
half an octave to an octave per row in the wheeze band. That is why 26.3% of `fc` values
are inadmissible: outside the admissible set the manipulation is not representable at
all. This is the same resolution wall that limits crackle localisation to coarse
Hit@±1.

## Grounding, hypothesis 11: weak query-conditioned localisation, below the gate

One attempt, criteria fixed before the run, official TRAIN split only (695 clips, 67
patients, patient-disjoint train/dev, seeds 0/1/2). Query = one bit. Primary metric =
within-clip 2AFC between the two candidate regions, chance exactly 0.5. Per-example,
per-seed predictions are saved under the frozen configuration in
`results/grounding_v21_preds.npz`, and every number below is computed from them.

| arm | 2AFC | 95% CI | paired Δ vs intact | Δ 95% CI |
|---|---:|---|---:|---|
| **oracle-location crop classifier** | **1.000** | [1.000, 1.000] | −0.431 | [−0.462, −0.396] |
| **learned per-patch bilinear head** | **0.569** | [0.538, 0.605] | — | — |
| query-conditioned DSP | 0.549 | [0.481, 0.607] | +0.021 | [−0.043, +0.088] |
| query_shuffled | 0.493 | — | +0.077 | [+0.035, +0.122] |
| query_only | 0.500 | — | +0.069 | [+0.010, +0.138] |
| query_constant | 0.500 | — | +0.069 | [+0.038, +0.105] |
| target_permuted | 0.503 | — | +0.066 | [+0.036, +0.100] |
| audio_zeroed | 0.507 | — | +0.062 | [−0.005, +0.136] |
| position_only | 0.510 | — | +0.060 | [+0.031, +0.096] |
| audio_shuffled | 0.521 | — | +0.049 | [+0.013, +0.084] |
| *chance* | 0.500 | — | +0.069 | [+0.038, +0.105] |

Deltas are paired on the same dev examples and bootstrapped by patient cluster.
`audio_zeroed` produces a coin flip per example, so its delta CI is inflated by that
arm's own variance rather than by any ambiguity about the model.

### What the numbers support

**The effect is real but small.** The model beats chance by +0.069 [+0.038, +0.105], and
the paired deltas separate it from `query_shuffled`, `query_constant`, `target_permuted`,
`position_only` and `audio_shuffled` — the arms that break the query correspondence, the
target, or the acoustic content. So the head does condition on the query and does use the
audio.

**It did not reach the development gate.** 0.569 against the 0.60 criterion chosen before
the run. Consequently the official test set was not touched and no real annotation was
commissioned. Both were gated on this number and both remain unspent.

**The distinction is fully present in the patches.** The *oracle-location crop
classifier* — the feasibility gate's own linear classifier, handed both ground-truth
event locations and asked only which crop matches the query — scores 1.000. It is not a
competing method and not a tuned model; it isolates how much of the correspondence
survives in AST's patches once localisation is free. The gap of −0.431 is the part the
simple head does not extract.

**Detection and selection come apart.** `hit@argmax` chance is about 0.033 (roughly 32
mask patches among ~960 valid ones). The intact model reaches 0.546 and even
`query_constant` reaches 0.442. Finding *an* event is nearly free; identifying *which*
event the query names is where almost all of the difficulty sits.

### On `query_flip`

Reported earlier as corroboration that the head reads the query. It is not, and it has
been demoted to an algebraic sanity check. A clip contributes two examples with the same
two regions and swapped labels, so evaluating one with the flipped query reproduces the
other's scores; the win vector is the complement of intact's **by construction**, and
0.431 = 1 − 0.569 is an identity rather than a measurement. It confirms the evaluation
plumbing is consistent and nothing more. `query_shuffled`, `query_constant` and their
paired deltas carry that evidence instead, and `query_flip` is excluded from the delta
table.

### What is and is not claimed

Claimed: a simple query-to-patch bilinear head learns weak, genuinely query-conditioned
localisation of a synthetic acoustic distinction that AST's patches encode completely.

Not claimed: that the correspondence is unlearnable. One rank-1 scoring architecture was
tested — a linear patch projection dotted with a query embedding. **Region-aware or
cross-attention heads are untested and are not ruled out.** The pre-registration allowed
one attempt, so trying a second here would have converted a clean result into a search;
that remains available as new, separately pre-registered work.

Also not claimed: anything clinical. The events are synthetic, and the manipulation is
harmonic vs inharmonic partial structure rather than monophonic vs polyphonic wheeze —
component count was the variable that had to be equalised to close the geometry leak.
The DSP arm's CI covers 0.5, so it never functioned as a ceiling and the "loses to DSP"
branch of the plan did not apply.

### One line

AST patches encode the synthetic acoustic distinction, but a simple query-to-patch
bilinear head extracts only weak query-conditioned localisation — enough to exceed
chance, not enough to justify held-out testing or real annotation.
