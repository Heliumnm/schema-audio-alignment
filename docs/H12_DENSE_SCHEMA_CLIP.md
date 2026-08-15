# H12 — Dense schema→spectrogram alignment

**Frozen before any run. Read [`RUNBOOK.md`](RUNBOOK.md) first** — how to reach the
server, where the data and weights are, and the traps that have each cost a re-run.

## The hypothesis

Keep the mel spectrogram's 2-D patch grid instead of mean-pooling it, split the schema
into **field-level text queries**, and learn a field→region correspondence with a
**local** contrastive loss.

    mel spectrogram → AST first 6 layers → 12 × 101 patches   (no mean pooling)
                                         → 2-D region adapter → P[f, t]
    schema field    → BERT / ClinicalBERT                     → Q[k]
                    → Q[k] · P[f, t] → field-to-patch similarity map → heat map

This is not a departure from the "text ↔ mel" route; it is that route done properly.
CLIP never compares text to RGB pixels either — both sides pass through encoders. What
changed is that Phase 1 averaged all patches into one clip vector, which can only test
global alignment and cannot test the original idea at all.

**What has actually been tested so far:** global mean-pooled alignment (21 runs, rejected
under every text source and training setup) and **one** minimal local readout — a 1-bit
query dotted with individually projected patches, no text encoder, no neighbourhood
modelling (H11, 0.569 2AFC, below its gate). The field-level dense route is untested.
H12 is not a repeat.

## Step 0 — the diagnostic that must run first

H11 left two numbers: a per-patch bilinear head at 0.569, and an *oracle-location* crop
classifier at 1.000. It is tempting to read the gap as "the model cannot read neighbouring
patches jointly", and the region adapter follows from that reading. **But the oracle
differs in two ways at once**: it reads a 5×3 crop jointly *and* it is handed both
ground-truth locations. A region adapter addresses only the first.

Separate them before building anything. Score every valid window with the frozen 5×3 crop
representation, **conditioned on the query**, and take the max over windows — the same
readout dense CLIP will use, not a probe on pooled features:

    s(q, f, t) = <query embedding, crop(P, f, t)>
    region wins if  max over its windows  >  max over the distractor's windows

* max-over-window ≈ 1.000 → joint neighbourhood reading is sufficient; the region adapter
  is motivated by evidence and H12 proceeds as designed;
* it collapses toward chance → the bottleneck is **search**, not neighbourhood reading,
  and adding conv + attention is an uncontrolled change of exactly the kind this project
  has twice had to retract. Redesign the loss or the candidate mechanism instead.

### Why this is not H7 again

H7 (`pooling_diagnostic.py`) already compared `mean` / `max` / `mean+max` / `segments`
readouts on these same frozen AST features and found headroom 0.000, so a sliding window
is, as a *feature* operation, just `segments` with overlap. The distinction is the
estimand, and it has to stay explicit:

| | H7 | Step 0 |
|---|---|---|
| query | none | one bit, conditions the score |
| readout | pool features, then probe | query-conditioned similarity per window |
| metric | classification AUROC (wheeze/crackle) | 2AFC localisation between two regions |
| question | does time resolution raise accuracy? | can joint local reading *find* the region? |

**Degeneracy guard:** if any Step-0 variant reduces to pooling features and probing for
classification, it is H7, its answer is already known, and it must not be run or reported
as new evidence. The query must enter the score, and the metric must be localisation.

Uses only existing frozen data (`results/grounding_v21_manifest.json`, the v2.1 synthesis)
and costs well under an hour. **No other H12 code is written until this reports.**

## Stage 1 — mechanism, on synthetic data only

Official TRAIN split only, patient-disjoint train/dev inside it. The test set is not
touched at Stage 1 under any outcome.

| arm | what it isolates |
|---|---|
| Global CLIP | the old route: mean-pooled clip vector ↔ sentence vector |
| per-patch head (H11) | the 0.569 baseline, unchanged |
| Dense CLIP | field token ↔ independently projected patches |
| Region-aware Dense CLIP | neighbourhood modelling, then field↔region alignment |

### Region adapter

Deliberately small — the claim is about whether joint local reading helps, not about
capacity:

    AST patches → 3×3 2-D convolution → 2 layers of local transformer / cross-attention
                → query-conditioned region score

It must be able to read adjacent frequency rows and time steps together. That is the
whole point of the arm.

### Loss

For field \(q_k\) with region \(M_k\), computed inside the true occupancy mask:

$$
L_{\text{local}} = -\log \frac{\sum_{p \in M_k} \exp(s(q_k, p)/\tau)}
                              {\sum_{p \in \text{valid}} \exp(s(q_k, p)/\tau)}
$$

* positives: the patches of the region the field describes;
* hard negatives: **the other event in the same clip** — this is what makes the query
  necessary rather than decorative;
* padding is excluded from the denominator entirely, never merely down-weighted;
* **multi-positive across samples is a variant to test, not a default.** Hypothesis 8
  already tested exact / soft / soft-sharp multi-positive targets in the global setting
  and found no effect on either target across three loss variants. That does not transfer
  automatically to the local setting, which is exactly why it has to be an arm rather than
  an assumption baked into the loss.

### The text axis has to be real

Two strings — "an event with harmonic spectral structure" versus "...inharmonic..." —
produce two fixed BERT vectors. That is a 1-bit query with an encoder bolted on, and it
cannot show what text is supposed to buy: **querying field values the model never saw in
that combination.**

So Stage 1 needs synthesis that varies along **several** schema fields, not one:

    [character = harmonic | inharmonic]
    [pitch     = low | mid | high]
    [duration  = short | long]
    [intensity = faint | moderate]

Each trial queries one field; the two events are matched on every other field and differ
on the queried one, counterbalanced across fields. Hold out some field-value combinations
from training to test composition. Templates, length and vocabulary complexity must be
matched across values so the text encoder cannot separate them on surface form.

**Every added field needs its own geometry-only control.** Pitch legitimately determines
where in frequency the answer lies; that is signal, not leakage. But the v2.1 experience
is unambiguous — three designs leaked through mask geometry, and one reported a perfect
gate while measuring crop position. Run the geometry-only probe **per queried field** and
report it in the same table.

### Controls, every arm

`query_only`, `query_constant`, `position_only`, `query_shuffled`, `audio_shuffled`,
`audio_zeroed`, `target_permuted`, geometry-only (per field), and a query-conditioned DSP
baseline. `query_flip` is **not** evidence — with a clip's two examples carrying swapped
labels its win vector is the complement of the intact one by construction, so it is a
plumbing check only.

### GO criteria — fixed now

1. mean 2AFC ≥ **0.65**;
2. 95% CI lower bound > **0.60** (patient-cluster bootstrap);
3. beats the H11 per-patch head on a **paired** delta — same dev examples, patient-cluster
   bootstrap, CI excluding 0. `results/grounding_v21_preds.npz` holds H11's per-example,
   per-seed win vectors, so this is directly computable;
4. `query_shuffled`, `audio_shuffled` and `position_only` all back at chance;
5. three seeds agreeing in sign;
6. padding masked throughout, asserted in code.

Note on power: H11's dev set was 452 examples over 22 patients and the observed CI
half-width was ≈ ±0.033. Clearing (1) and (2) together is feasible but not comfortable at
that size — **raise `--n` from 700 toward the full TRAIN pool** before running, not after
seeing the result.

Failing the gate means the synthetic task does not support further investment even with
neighbourhood reading. It does **not** license a third architecture attempt inside H12.

## Stage 2 — real wheeze, only if Stage 1 passes

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

Reuse `clip_finetune.py` (mel input + AST), `grounding_v21.py` (clean synthesis, region
masks, the geometry-only control), `grounding_train.py` (control harness and 2AFC).
**Do not reuse `ast_features.py`** — its vectors are already mean-pooled and the local
information is gone.

New: `src/dense_schema_clip.py` — field encoder, 2-D region adapter, local contrastive
loss.

## The one-shot rule

The official test set is spent **once**, and only after Stage 1 passes and the manifest,
seeds, model, loss, metrics and GO criteria are committed. Every number before that is
train/dev. This document is the pre-registration; changing a threshold after seeing a
result is a new hypothesis with a new document, recorded as such.
