# H12 — Dense schema→spectrogram alignment

**Read [`RUNBOOK.md`](RUNBOOK.md) first** — how to reach the server, where the data and
weights are, and the traps that have each cost a re-run.

Step 0 has run and is recorded below. Everything after it is frozen before any run;
Stage 1A and Stage 1B carry separate gates so that a result cannot be produced by
changing the loss, the text, the data and the head at the same time.

## The hypothesis

Keep the mel spectrogram's 2-D patch grid instead of mean-pooling it, split the schema
into **field-level text queries**, and learn a field→region correspondence with a
**local** contrastive loss.

    mel spectrogram → AST first 6 layers → 12 × 101 patches   (no mean pooling)
                                         → [2-D region adapter]  ← increment, not premise
                                         → P[f, t]
    schema fields   → BERT / ClinicalBERT                     → Q[k]
                    → Q[k] · P[f, t] → field-to-patch similarity map → heat map

This is not a departure from the "text ↔ mel" route; it is that route done properly.
CLIP never compares text to RGB pixels either — both sides pass through encoders. What
changed is that Phase 1 averaged all patches into one clip vector, which can only test
global alignment and cannot test the original idea at all.

**What has actually been tested so far:** global mean-pooled alignment (21 runs, rejected
under every text source and training setup) and **one** minimal local readout — a 1-bit
query dotted with individually projected patches, uniform-positive loss, no text encoder,
no neighbourhood modelling (H11, 0.569 2AFC, below its gate). The field-level dense route
is untested. H12 is not a repeat.

## Step 0 — RUN, and it moved the premise

*Pre-registered question:* H11 left a per-patch bilinear head at 0.569 and an
*oracle-location* crop classifier at 1.000. It was tempting to read the gap as "the model
cannot read neighbouring patches jointly", and the region adapter followed from that
reading. But the oracle differs in **two** ways at once — it reads a 5×3 crop jointly
*and* it is handed both ground-truth locations. A region adapter addresses only the first.

*Pre-registered readout:* score every valid window with the frozen 5×3 crop
representation, **conditioned on the query**, and take the max over windows — the same
readout dense CLIP will use, not a probe on pooled features.

    s(q, f, t) = <query embedding, crop(P, f, t)>
    region wins if  max over its windows  >  max over the distractor's windows

*Pre-registered branches:* max-over-window ≈ 1.000 → joint neighbourhood reading is
sufficient and the region adapter is motivated by evidence; collapses toward chance → the
bottleneck is **search**, and adding conv + attention would be an uncontrolled change.

### What it returned

`src/h12_step0_window.py`, `results/h12_step0_window.json`. Same 452 dev examples as H11,
patient-cluster bootstrap.

| readout | 2AFC | 95% CI |
|---|---:|---|
| oracle single window (H11's 1.000) | **1.000** | [1.000, 1.000] |
| **max over windows in region** | **0.619** | [0.575, 0.658] |
| mean over windows in region | 0.655 | [0.579, 0.725] |
| H11 per-patch head (reference) | 0.569 | — |
| query_shuffled | 0.496 | [0.461, 0.529] |
| query_constant | 0.500 | — |
| audio_shuffled | 0.478 | [0.435, 0.515] |

Paired against H11 on the same examples: **+0.050 [+0.013, +0.084]**, excluding 0. The
oracle's paired delta is +0.431 [+0.395, +0.462]. `hit@argmax` over all valid windows is
0.268 against a chance of 0.164, at ~223 valid windows per clip.

**Neither branch fired cleanly, and the reading is still clear.** Joint neighbourhood
reading is worth about +0.050 — roughly **12% of the 0.431 oracle gap**. The remaining
~88% is knowing where to look. Two qualifications keep this from being read as more than
it is: the 0.60 numeric cut that made the script print "PARTIAL" was the script's, not
this document's (Step 0's branches were stated qualitatively), and 0.619 is still a
comparison **between two real event regions** — it is a relaxation of the oracle, not a
full global search. The global-search number is `hit@argmax` = 0.268, and it is much
worse.

*Verified rather than assumed:* the seeded pipeline reproduces
`grounding_v21_manifest.json` field by field over all 695 clips; the 452 dev examples
match `grounding_v21_preds.npz` in order; and folding the scaler into the weights
reproduces the oracle's 1.000 exactly, which is the check that this is H11's own
classifier rather than a second model. Candidate windows per region match across
characters (30.51 vs 30.70), so the max is not exploiting a larger candidate set.

### What it changes

**The region adapter is demoted from premise to increment.** It was going to be justified
by the 0.431 gap; it can be justified by about 0.050 of it. It stays in the design as an
arm — an expected gain of that size is worth one arm — but no H12 conclusion may rest on
it.

**Search becomes the main variable.** What must change first is the training objective's
relationship to localisation, not the head's receptive field.

## Where the new variable actually is

An earlier draft of this document claimed the novelty was putting background patches in
the denominator. **That was wrong and is retracted here.** H11 already did that —
[`grounding_train.py:259`](../src/grounding_train.py#L259) masks invalid patches to −1e9,
takes `log_softmax` over the whole grid, and weights the mask uniformly:

$$L_{\text{H11}} = \log\!\!\sum_{\text{valid}}\!\exp s \;-\; \operatorname{mean}_{p\in M} s_p$$

The Stage-1 loss written below is:

$$L_{\text{MIL}} = \log\!\!\sum_{\text{valid}}\!\exp s \;-\; \log\!\!\sum_{p\in M}\!\exp s_p$$

The denominators are identical. The difference is the **positive** term:

* H11 forces **every** patch inside the region to score high;
* `L_MIL` asks only that the region **as a whole** collect enough probability mass;
* `L_MIL` is the objective that matches a max-over-window localisation metric, which is
  what Step 0 measured and what dense CLIP reads out.

The formula in the original draft was already the region-mass form, so what is corrected
is the narrative around it, not the mathematics. The correction matters because it names
the actual variable under test, and a variable that is misnamed cannot be controlled.

## Stage 1A — loss only, on the frozen v2.1 data

**Nothing about the data, the query, or the architecture changes.** The point is to
isolate the objective. Because the synthesis is untouched, the 452 dev examples are still
H11's and `results/grounding_v21_preds.npz` supports a **per-example paired** comparison —
an advantage Stage 1B will not have.

| arm | change | standing |
|---|---|---|
| **H11** | original query bit, original uniform-positive loss | fixed baseline |
| **MIL-search** | same model, region-mass / logsumexp positive term only | **primary mechanism arm** |
| **MIL + region** | previous arm plus the small 2-D adapter | secondary increment |
| two-sentence BERT query | the bit replaced by two matched English sentences | **smoke test only** |

The fourth arm is a **code smoke test, not a scientific arm**. Two strings produce two
fixed BERT vectors, which is a 1-bit query with an encoder bolted on; it can show that the
text path is wired correctly and nothing else. It must reproduce the bit version within
seed noise. Any difference is a finding about the pipeline, and it may not be reported as
evidence about text.

### Gate for Stage 1A — fixed now

Stage 1A is a bridge, not the H12 result, so it carries **no absolute 2AFC threshold**.
Its job is to answer one question: does the region-mass objective find events better than
the uniform-positive one?

1. **primary** — paired delta MIL-search minus H11, same 452 dev examples,
   patient-cluster bootstrap, 95% CI excluding 0;
2. `hit@argmax` moves in the same direction. The claim is about search, so the search
   metric has to move too; a 2AFC gain with a flat `hit@argmax` is not a search gain and
   must be reported as such;
3. `query_shuffled`, `query_constant`, `position_only`, `audio_shuffled`, `audio_zeroed`
   and `target_permuted` all back at chance;
4. three seeds agreeing in sign;
5. padding masked throughout, asserted in code.

These criteria are an operationalisation of "if MIL-search does not beat H11, the hope for
Dense CLIP cannot be attributed to the search objective". They are frozen before the run.

**If Stage 1A fails**, ~~the search objective is not the mechanism,~~ Stage 1B is not run,
and the multi-field synthesis is not built. Failing here does **not** license swapping in a
third loss inside H12.

> **Narrowed after the run, and marked rather than silently edited.** The struck clause was
> an interpretation written into the pre-registration, not one of its criteria, and it was
> too strong: this gate can only speak for the metric it named. The consequence it gates —
> Stage 1B not run, synthesis not built — is unchanged and was applied. The accurate
> statement is the one recorded below: **region-mass did not pass H12's pre-specified
> mean-region 2AFC primary criterion, so H12 does not support proceeding to Stage 1B; the
> pre-registered secondary Hit@argmax showed a strong and stable query-conditioned pointing
> signal.**

### What Stage 1A returned — the primary criterion fails

`src/h12_mil_search.py`, `results/h12_stage1a_mil.json`. The `uniform` arm reproduces
`grounding_v21_preds.npz` **bit for bit** — 0 of 1356 per-example, per-seed wins differ,
and the per-seed 2AFC matches to four decimals (0.6150 / 0.5819 / 0.5111). The harness is
therefore provably identical and the positive aggregation is the only difference.

| arm | 2AFC | seed sd | hier 95% CI | patient-only CI | hit@argmax |
|---|---:|---:|---|---|---:|
| uniform (H11) | 0.569 | 0.043 | [0.511, 0.631] | [0.538, 0.605] | 0.546 |
| **region-mass** | **0.580** | 0.032 | [0.528, 0.627] | [0.547, 0.611] | **0.869** |
| region-mass query_only | 0.502 | — | — | — | 0.018 |
| region-mass query_constant | 0.500 | — | — | — | 0.487 |
| region-mass position_only | 0.474 | — | — | — | 0.273 |
| region-mass query_shuffled | 0.494 | — | — | — | 0.486 |
| region-mass audio_shuffled | 0.498 | — | — | — | 0.191 |
| region-mass audio_zeroed | 0.510 | — | — | — | 0.010 |
| region-mass target_permuted | 0.507 | — | — | — | 0.136 |

Paired, patient × seed hierarchical bootstrap, seeds matched by initialisation:

* 2AFC: **+0.0103, 95% CI [−0.0656, +0.0990]** — does not exclude 0;
* hit@argmax: **+0.3237, 95% CI [+0.2485, +0.3893]** — excludes 0.

Per-seed 2AFC deltas are −0.022, −0.047, **+0.100**: the sign disagrees across seeds and
the mean is carried by one of the three.

**Criterion 1 fails and criterion 4 fails. By the gate frozen above, Stage 1A does not
pass, Stage 1B is not run, and the multi-field synthesis is not built.**

Stated precisely, and this is the sentence the rest of the document should be read
against: **region-mass did not pass H12's pre-specified mean-region 2AFC primary
criterion, so H12 does not support proceeding to Stage 1B; the pre-registered secondary
Hit@argmax showed a strong and stable query-conditioned pointing signal.** The gate speaks
for the metric it named and for nothing wider — it does not establish that the objective
is inert, and no claim of that form is made here.

### The pattern is not noise, and it is not a rescue

Criterion 2 — a pre-registered *secondary* requirement, not a primary — moved decisively
and in the same direction on all three seeds (0.580 / 0.527 / 0.531 → 0.863 / 0.894 /
0.852). It is genuinely query-conditioned: breaking the query collapses it to chance
between the two regions (`query_shuffled` 0.486, `query_constant` 0.487), and breaking the
target or the audio collapses it further (`target_permuted` 0.136, `audio_shuffled` 0.191).

So the two metrics disagree, and the disagreement has a mechanical explanation. **2AFC
scores the *mean* score inside each region; the region-mass objective concentrates mass on
a few patches rather than raising all of them.** An objective that optimises a max-like
quantity shows its gain in a max-like metric and not in a mean-like one. On this reading
the primary metric was mis-specified *for this objective* — it was chosen for H11, whose
uniform positive term it matches exactly.

**This does not change the verdict.** Promoting a secondary metric to primary after seeing
the result is precisely what this project has retracted results for, and the gate was
frozen before the run for that reason. Two things follow instead:

1. H12 is sealed here. Stage 1B stays unrun and `dense_schema_clip.py` stays unwritten;
2. the pointing signal becomes the premise of a **new pre-registered hypothesis**,
   [`H13_COMPOSITIONAL_POINTING.md`](H13_COMPOSITIONAL_POINTING.md), with max-based
   metrics named primary *before* anything is run.

**The old data is not re-scored.** Re-running Stage 1A to compute a more favourable
max-based 2AFC on the same models would be picking the metric after seeing the result, and
it is explicitly not done — the score maps were not saved and will not be regenerated for
that purpose. H13 earns its metrics on new synthesis instead.

One measurement worth carrying forward regardless: under a patient × seed hierarchical
bootstrap, **H11's own intact CI is [0.511, 0.631]**, against the published patient-only
[0.538, 0.605]. The above-chance claim survives, but with much less margin than reported,
and with a seed sd of 0.043 across three seeds a criterion asking for a +0.01 2AFC effect
had very little power to begin with. That is a fact about the gate's design, recorded here
for whoever pre-registers the next one — not a reason to reopen this one.

## Stage 1B — multi-field compositional text, on new synthesis

**HELD.** Stage 1A did not pass, so nothing below has been run or built.

Run only if Stage 1A passes. Official TRAIN split only; the test set is untouched at every
stage.

### The query is a full combination, not one field

    [character=harmonic] [pitch=high] [duration=short] [intensity=faint]

Each clip carries two events that **share three field values and differ on exactly one**,
with the differing field rotated across trials and the time positions fully randomised.
Dev holds out field-value combinations never seen in training.

Querying the whole combination rather than a single field is what makes this a test of
**compositional generalisation** — whether the text encoder can address a combination the
model never saw — instead of another test of a 1-bit distinction with more vocabulary
around it.

Templates, length and vocabulary complexity stay matched across values so the text encoder
cannot separate them on surface form.

### The shortcut problem this design has to survive

Adding fields adds geometry. `character` was safe in v2.1 because the mask is identical
for both values by construction. `pitch` and `duration` are not: pitch moves the answer in
frequency and duration moves it in time, both legitimately, and the target mask therefore
reveals the answer whenever one of them is the differing field. Three designs have already
been retracted for exactly this class of leak.

So the evaluation is constrained rather than the synthesis:

* the primary metric is a **time 2AFC over fixed-size candidate windows** — the two
  candidate windows are **identical in size** in every trial, so mask extent cannot be
  read off;
* the two events' order in time is randomised;
* a `pitch` query is **not** scored for frequency localisation;
* a `duration` query may **not** be answerable from target-mask size — the fixed-size
  window rule is what enforces this;
* **every field gets its own `position_only` / geometry-only control, and each must be
  back at chance**, reported in the same table as the arm it guards.

### Controls, every arm

`query_only`, `query_constant`, `position_only`, `query_shuffled`, `audio_shuffled`,
`audio_zeroed`, `target_permuted`, geometry-only **per field**, and a query-conditioned DSP
baseline. `query_flip` is **not** evidence — with a clip's two examples carrying swapped
labels its win vector is the complement of the intact one by construction, so it is a
plumbing check only.

### The old predictions stop being comparable

New synthesis means a new manifest and new dev examples. **`grounding_v21_preds.npz` can
no longer be used for per-example pairing at Stage 1B.** H11, Dense and Region-aware must
all be re-run on the new manifest so that every arm in the table is measured on the same
samples. Stage 1B's "beats H11" criterion refers to that re-run, not to 0.569.

### GO criteria for Stage 1B — fixed now

1. mean 2AFC ≥ **0.65**;
2. 95% CI lower bound > **0.60** (patient-cluster bootstrap);
3. beats the **re-run** H11 per-patch head on a paired delta, same dev examples,
   patient-cluster bootstrap, CI excluding 0;
4. `query_shuffled`, `audio_shuffled` and `position_only` all back at chance, and every
   per-field geometry-only control at chance;
5. three seeds agreeing in sign;
6. padding masked throughout, asserted in code;
7. held-out field combinations evaluated separately and reported, whatever they show.

Note on power: H11's dev set was 452 examples over 22 patients and the observed CI
half-width was ≈ ±0.033. Clearing (1) and (2) together is feasible but not comfortable at
that size — **raise `--n` from 700 toward the full TRAIN pool** before running, not after
seeing the result.

Failing the gate means the synthetic task does not support further investment. It does
**not** license a third architecture attempt inside H12.

## The five arms and what each is for

| arm | standing |
|---|---|
| Global CLIP | historical/global reference. Carries no H12 conclusion. |
| H11 per-patch | the old baseline, re-run on whatever data the stage uses |
| **Loss-only bridge** | same bit query, loss swapped. **Attributes the gain.** |
| **Dense Schema CLIP** | the main arm: multi-field text + region-mass loss |
| Region-aware Dense Schema CLIP | tests the ≈ +0.05 neighbourhood increment on top |

The loss-only bridge is what keeps the final number interpretable. Without it, a Stage-1B
gain is compatible with the loss, the text, the data and the head all having done the
work, and this project has twice had to retract a result for less.

### Region adapter

Deliberately small — after Step 0 the claim it supports is an increment, not a mechanism:

    AST patches → 3×3 2-D convolution → 2 layers of local transformer / cross-attention
                → query-conditioned region score

It must be able to read adjacent frequency rows and time steps together. That is the whole
point of the arm, and Step 0 has already sized the expected gain at about +0.05.

### On multi-positive

**Multi-positive across samples is a variant to test, not a default.** Hypothesis 8 already
tested exact / soft / soft-sharp multi-positive targets in the global setting and found no
effect on either target across three loss variants. That does not transfer automatically to
the local setting, which is exactly why it has to be an arm rather than an assumption baked
into the loss.

## Stage 2 — real wheeze, only if Stage 1B passes

Annotate 100–200 real wheeze cycles. First pass records only:

* onset time, offset time;
* optional coarse frequency band;
* annotator confidence.

**Wheeze only, not crackle.** AST's ~100 ms hop and ~160 ms receptive field cannot support
a claim about localising 5–20 ms crackles; that constraint is recorded in the README and
has not changed.

Three text conditions at **identical information content**, varying only expression:

| condition | example |
|---|---|
| natural | A continuous, high-pitched musical wheeze. |
| schema | `[event=wheeze] [pitch=high] [continuity=continuous]` |
| typed schema | field, value, confidence and provenance encoded separately |

Evaluation keeps the query honest:

* scoring **time** localisation — the query may carry pitch, never onset/offset;
* scoring **frequency** localisation — the query may not carry exact frequency;
* primary metrics: 2AFC, pointing accuracy, region mass, temporal IoU;
* classification AUROC is secondary and is not what this is for.

## Code

Done:

* `src/h12_step0_window.py` — Step 0, the query-conditioned window search.

Reuse:

* `grounding_v21.py` — the frozen v2.1 synthesis, region masks, the geometry-only control;
* `grounding_train.py` — the control harness, 2AFC, patient-cluster bootstrap;
* `clip_finetune.py` — mel input + AST, for the Global CLIP arm.

**Do not reuse `ast_features.py`** — its vectors are already mean-pooled and the local
information is gone.

New, in this order and not before:

* Stage 1A → `src/h12_mil_search.py`: the region-mass loss inside `grounding_train.py`'s
  existing harness, so that the loss is the only thing that differs from H11;
* Stage 1B → `src/h12_synth_fields.py` (multi-field synthesis, per-field geometry controls)
  and `src/dense_schema_clip.py` (field encoder, 2-D region adapter, local contrastive
  loss).

`dense_schema_clip.py` is **not** written until Stage 1A reports.

## Order of work

1. ~~commit Step 0~~ — done, `ad9b0b6`;
2. ~~update this document's bottleneck explanation~~ — done, `edba6ca`;
3. ~~Stage 1A: loss-only validation on the frozen v2.1 data~~ — **done, did not pass**;
4. ~~only if it passes: rebuild the multi-field compositional synthesis~~ — **held**;
5. ~~Dense Schema CLIP as the main Stage-1B arm~~ — **held**;
6. ~~region adapter as the increment on top~~ — **held**;
7. ~~real wheeze annotation~~ — **held**.

The open question is not "which loss next" — the pre-registration forbids a third attempt
inside H12. It is whether the metric mismatch Stage 1A exposed is worth a new
pre-registered hypothesis with a max-based localisation metric named as primary in
advance.

## The one-shot rule

The official test set is spent **once**, and only after Stage 1B passes and the manifest,
seeds, model, loss, metrics and GO criteria are committed. Every number before that is
train/dev. This document is the pre-registration; changing a threshold after seeing a
result is a new hypothesis with a new document, recorded as such.
