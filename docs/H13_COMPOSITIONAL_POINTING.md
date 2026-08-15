# H13 — Compositional schema→spectrogram pointing

**Pre-registration. Frozen before any run.** Read [`RUNBOOK.md`](RUNBOOK.md) first — how to
reach the server, where the data and weights are, and the traps that have each cost a
re-run.

## Standing — what H13 inherits, and what it does not

**H13 is motivated by H12's pre-registered *secondary* result. It is not a confirmatory
conclusion of H12.**

H12 failed its primary criterion and is sealed
([`H12_DENSE_SCHEMA_CLIP.md`](H12_DENSE_SCHEMA_CLIP.md), commits `3602ad5` / `cbe67c1`).
Region-mass did not pass the pre-specified mean-region 2AFC criterion; the pre-registered
secondary Hit@argmax showed a strong and stable query-conditioned pointing signal
(0.546 → 0.869, all three seeds, `query_shuffled` 0.486 and `query_constant` 0.487 at
chance). That signal is a *reason to ask a new question*, not evidence for an answer.

Three consequences, and they bind:

* no H12 number is evidence for H13. H13's claims rest on H13's own runs;
* **H12's data is not re-scored.** Regenerating score maps to compute a more favourable
  max-based 2AFC on H12's models would be choosing the metric after seeing the result. It
  is not done;
* H13 runs on **new synthesis**, so `grounding_v21_preds.npz` and `h12_stage1a_preds.npz`
  are not pairable here. Every arm is re-run on the new manifest.

## The hypothesis

A multi-field schema text, encoded by a text encoder, points at the right place on the mel
map — and does so for field combinations it was never trained on.

    mel spectrogram   → AST first 6 layers → 12 × 101 patches   (no mean pooling)
                      → [region adapter]   ← optional arm
                      → P[f, t]
    multi-field text  → BERT / ClinicalBERT → Q
                      → Q · P[f, t] → response map → argmax over the whole map

Training is the region-mass local contrastive objective, unchanged from H12 Stage 1A:

$$L = \log\!\!\sum_{\text{valid}}\!\exp s \;-\; \log\!\!\sum_{p\in M}\!\exp s_p$$

Padding is excluded from the denominator entirely, never merely down-weighted, and this is
asserted in code. No temperature is introduced; H11 and H12 Stage 1A both ran at τ = 1 and
a temperature would be a second variable.

This is the first stage of the project where the text side is real. It is worth being
precise about what it is and is not: both towers encode, as in CLIP, but the contrast runs
over **spatial positions inside one clip**, not over other samples in the batch. This is a
local/region-level contrastive *pointing* objective in the GLoRIA / BioViL line, not
CLIP-style instance discrimination, and it must be described that way.

## Why the metrics change

H12's primary metric scored the **mean** response inside a region. That matched H11's
uniform positive term exactly, and it is mismatched to a region-mass objective, which
concentrates mass on a few patches instead of raising all of them. H12's two metrics
disagreed for that mechanical reason.

H13 therefore names **max-based metrics as primary, in advance**, and demotes the mean.

### Primary metrics — fixed now

1. **Temporal Hit@argmax** — the time patch of the global maximum falls inside the queried
   event's fixed-size time window. Chance is a property of the synthesis, not of the model:
   it is `window_width / n_valid_time_patches`, computed from the frozen manifest and
   **reported before any model is trained**.
2. **Max-based 2AFC** — of two equal-size candidate windows, which contains the higher
   maximum response. Chance is exactly 0.5 by construction.

### Diagnostic only

3. **mean-region 2AFC** — reported in every table for continuity with H11 and H12, and
   carrying **no conclusion**. If it disagrees with the primary metrics, that disagreement
   is a finding to report, not a verdict to apply.

## The query is a full combination

    [character=harmonic] [pitch=high] [duration=short] [intensity=faint]

Each clip carries two events that **share three field values and differ on exactly one**,
with the differing field rotated across trials and the two events' time positions fully
randomised. Dev holds out field-value combinations never seen in training.

Querying the whole combination rather than a single field is what makes this a test of
**compositional generalisation** — whether the encoder can address a combination the model
never saw — rather than another test of a 1-bit distinction with more vocabulary around it.
Templates, length and vocabulary complexity stay matched across values so the text encoder
cannot separate them on surface form.

**Seen and unseen combinations are evaluated separately and both reported, whatever they
show.**

## The shortcut problem this design has to survive

Adding fields adds geometry. `character` was safe in v2.1 because the mask is identical for
both values by construction. `pitch` and `duration` are not: pitch moves the answer in
frequency and duration moves it in time, both legitimately, and the target mask therefore
reveals the answer whenever one of them is the differing field. Four designs have already
been retracted over this class of leak.

The evaluation is constrained rather than the synthesis:

* candidate windows are **identical in size** in every trial, so mask extent cannot be read
  off — this is what stops a `duration` query being answerable from target-mask size;
* the two events' order in time is randomised;
* a `pitch` query is **not** scored for frequency localisation;
* **every field gets its own `position_only` / geometry-only control, and each must be back
  at chance**, reported in the same table as the arm it guards.

### Representability constraints, to fix before the synthesis is written

AST's grid is the binding constraint, exactly as in v2.1:

* **pitch** — 12 mel rows over 0–8000 Hz. Field values must be separable on that grid and
  each must fall in the admissible set (`admissible_fc`, 73.7% of 300–1200 Hz in v2.1). The
  number of distinct pitch values is bounded by the grid, not by taste;
* **duration** — ~100 ms hop and ~160 ms receptive field. `short` and `long` must differ by
  enough time patches to be representable at all, and the fixed-size candidate window must
  still contain either;
* **intensity** — SNR, matched in every other respect. v2.1 drew 4–9 dB; discrete values
  must stay inside a range where the event is neither inaudible nor trivially findable.

A field value that is not representable is not a hard task, it is an unmeasurable one.

## Arms

| arm | question it answers |
|---|---|
| discrete field/value embedding | does a structured query work at all without real language? |
| **multi-field schema text + BERT** | can the text encoder read field combinations? |
| `query_shuffled` | does it depend on the text correspondence at all? |
| `position_only` / `audio_shuffled` | does it depend on acoustic content rather than coordinates? |
| region adapter (optional) | how much does neighbourhood reading add on top? |

The discrete-embedding arm is what makes the BERT arm interpretable: without it, a text
result cannot be separated from "any structured query would have done this". The region
adapter stays optional and last — Step 0 sized its expected contribution at about +0.05,
and no H13 conclusion may rest on it.

### Full control suite, every arm

`query_only`, `query_constant`, `position_only`, `query_shuffled`, `audio_shuffled`,
`audio_zeroed`, `target_permuted`, geometry-only **per field**, and a query-conditioned DSP
baseline. `query_flip` is **not** evidence — with a clip's two examples carrying swapped
labels its win vector is the complement of the intact one by construction, so it is a
plumbing check only.

## GO criteria — fixed now

On the **primary** metrics, official TRAIN split only, patient-disjoint train/dev:

1. **max-based 2AFC** mean ≥ **0.65**, and the 95% CI lower bound > **0.60**;
2. **Temporal Hit@argmax** ≥ **0.50** absolute and at least **5×** the chance computed from
   the frozen manifest, with the CI on (hit − chance) excluding 0;
3. the **multi-field BERT arm** beats the **discrete field/value arm** on a paired delta,
   CI excluding 0 — otherwise the result is about structured queries, not about text;
4. `query_shuffled`, `audio_shuffled`, `audio_zeroed`, `position_only`, `target_permuted`
   and every per-field geometry-only control back at chance;
5. **five seeds**, all agreeing in sign on both primary metrics;
6. held-out combinations reported separately; the gate is read on the **unseen** set;
7. padding masked throughout, asserted in code.

All CIs are a **patient × seed hierarchical bootstrap** — patients resampled for sampling
error, seeds for optimiser noise, seeds matched across arms by initialisation for paired
deltas.

> These thresholds are proposed rather than inherited: (1) and the control criteria carry
> over from H12, (2), (3) and (5) are new. Nothing has been run, so they can still be
> overwritten — but only **now**, in this document, before the first run.

### Two lessons from H12 that are binding here

**Five seeds, not three.** Stage 1A measured a seed sd of 0.043 on 2AFC across three seeds.
Under a patient × seed hierarchical bootstrap that left almost no power against a +0.01
effect, and it widened H11's own intact CI from the published [0.538, 0.605] to
[0.511, 0.631]. Three seeds is not enough to gate on.

**Raise `--n` toward the full TRAIN pool** before running, not after seeing the result. H11
and H12 both ran at `--n 700`, giving 452 dev examples over 22 patients.

## What failing means

Failing the gate means the compositional pointing claim is not supported on synthetic data.
It does **not** license a second architecture, a second loss, or a re-scored metric inside
H13. As with H11 and H12, a threshold changed after seeing a result is a new hypothesis
with a new document.

## What stays unspent

**The official test set and real wheeze annotation are not touched until H13's development
gate is cleared.** Both were gated on H11, then on H12, and both remain unspent. When they
are spent, it is once, and only after the manifest, seeds, model, loss, metrics and GO
criteria are committed.

Real annotation, when it happens, is **wheeze only, not crackle** — AST's ~100 ms hop and
~160 ms receptive field cannot support a claim about localising 5–20 ms crackles, and that
constraint has not changed.

## Code

Reuse:

* `grounding_v21.py` — mel-row machinery, `admissible_fc`, region masks, the geometry-only
  control;
* `grounding_train.py` — the control harness and the 2AFC plumbing;
* `h12_mil_search.py` — the region-mass loss and the patient × seed hierarchical bootstrap;
* `clip_finetune.py` — mel input + AST.

**Do not reuse `ast_features.py`** — its vectors are already mean-pooled and the local
information is gone.

New:

* `src/h13_synth_fields.py` — multi-field synthesis, per-field geometry-only controls, a
  frozen manifest that records the chance level for Temporal Hit@argmax;
* `src/h13_pointing.py` — discrete and BERT field encoders, region-mass loss, optional
  region adapter, both primary metrics and the mean-region diagnostic.

The synthesis is written and its matching report inspected **before** the model, in that
order, so that a design leak is caught while it is still cheap.
