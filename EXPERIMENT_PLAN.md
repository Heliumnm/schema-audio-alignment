# Schema-grounded contrastive alignment for respiratory audio

**Target:** ICASSP (deadline ~early September 2026) · **First author:** Liu He · **Advisor:** Yuanchao Li

> **What text should respiratory audio be aligned to?**
> We show that structured, provenance-tagged schema text outperforms both patient
> metadata and free-form LLM narration for audio–text contrastive alignment — and we
> quantify how much of the apparent gain from audio-derived text is circular rather
> than transferable.

---

## 1. Motivation

RespiraMFM (arXiv 2606.09966) established that contrastive audio–text alignment
before instruction tuning improves respiratory disease identification, especially
zero-shot. Its text is **patient metadata** (age, sex, fever, night sweats, HIV
status). That leaves two gaps:

| Gap | Why it matters |
|---|---|
| **Metadata is often unavailable at inference** | Real deployment has the sound, not the questionnaire. |
| **Metadata describes the *patient*, not the *sound*** | Alignment anchors audio to epidemiological priors, not to acoustic evidence. |

So the question becomes: **can we align to descriptions of the acoustic evidence
itself — generated automatically — and does the *structure* of those descriptions
matter?**

## 2. Contributions (revised after the W1 measurements in §2.5)

1. **A text-source taxonomy** for respiratory audio–text alignment: metadata /
   free-form LLM narration / structured schema, held under one fixed architecture
   and one fixed hyper-parameter set.
2. **Resolution, not fluency, is what contrastive alignment needs.** Metadata text
   collapses 6,898 recordings onto **27 distinct strings**; schema text yields
   **3,552**. Below some resolution the InfoNCE objective has no usable negatives —
   most in-batch "negatives" carry identical text. This turns 师兄's hypothesis
   ("schema gives a finer information mapping than natural description") from an
   intuition into a **measurable quantity: distinguishable text states per corpus**,
   and connects it directly to the false-negative literature the project already
   builds on (Chuang et al. 2020; Huynh et al. 2020). *(Headline claim.)*
3. **Two confound controls that reviewers would otherwise demand.**
   - *Circularity*: how much of the gain is self-recovery when the text is a
     function of the audio (provenance ablation E2, audio-tower ablation E3).
   - *Recording confound*: on ICBHI, chest location + device alone predict crackle
     as well as every acoustic cue combined (§2.5). Any crackle result that does not
     report this baseline is uninterpretable.

## 2.5 W1 measurements (real, on all 6,898 ICBHI cycles)

`schema_text.py` ran clean: 6,898/6,898 read, 0 failures, 3,450 train, leak check
passed. Then a text-only probe (TF-IDF + logistic regression, patient-independent
official split) measured what each condition actually contains.

**Text resolution — the finding that reframes the paper:**

| condition | distinct texts / 6,898 | mean words |
|---|---:|---:|
| dataset | **27** | 10.0 |
| model | 193 | 24.0 |
| signal | 1,786 | 27.7 |
| all | **3,552** | 39.3 |

**Label information carried by the text alone (AUROC):**

| condition | crackle | wheeze |
|---|---:|---:|
| dataset | 0.637 | 0.390 |
| signal | 0.637 | 0.514 |
| model | 0.626 | 0.505 |
| all | 0.633 | **0.579** |

Three consequences:

1. **Circularity is mild, not fatal.** The most an audio-derived condition leaks is
   AUROC 0.637. Alignment gains cannot be explained away as label leakage through
   text — which is what made this design risky in the first place.
2. **A recording confound dominates crackle.** `dataset` (chest location + device
   only, no acoustics at all) reaches 0.637 — *identical* to `signal` and `all`. On
   ICBHI, the acoustic cues add **nothing** over knowing where and with what the
   recording was made. This is the known ICBHI device/site confound, quantified.
   **Report it as a baseline on every crackle result.**
3. **Wheeze behaves differently and is the cleaner target.** `dataset` alone is
   *below* chance (0.390) and the cues genuinely add signal (→ 0.579). Where crackle
   is confounded, wheeze is not — consider leading with wheeze.

---

## 2.6 First alignment sweep — a negative result (wheeze, 2026-08-07)

Ran E1/E2 on wheeze: 4 provenance conditions × 3 seeds, patient-independent ICBHI
split (3,450 train / 3,448 test, zero patient overlap asserted). The frozen-feature
baseline is a linear probe on the raw OPERA-CT embeddings, **same split, same seeds**
— not the 600-sample balanced subset used in the fidelity-oracle work, which would
not have been comparable.

| condition | distinct texts | MCC | AUROC | ΔAUROC |
|---|---:|---|---|---:|
| **RAW features (no alignment)** | — | **0.183** | **0.640** | — |
| align / dataset | 27 | 0.005 ± 0.025 | 0.515 ± 0.012 | −0.125 |
| align / signal | 1,786 | 0.041 ± 0.012 | 0.524 ± 0.006 | −0.116 |
| align / model | 193 | 0.137 ± 0.015 | 0.586 ± 0.002 | −0.054 |
| align / all | 3,552 | 0.078 ± 0.013 | 0.542 ± 0.006 | −0.098 |

**Every condition is below the baseline.** Contrastive alignment on frozen features
is discarding task information, not adding it. Training loss barely moved across 500
epochs (59.49 → 59.34), consistent with the objective learning very little.

**The resolution hypothesis of §2.2 is falsified.** Ranking does not track distinct
text count (27 → 0.515; 193 → **0.586**; 1,786 → 0.524; 3,552 → 0.542). It tracks
**circularity**: `model` wins because its text *is* OPERA-CT's own wheeze estimate
while the audio tower *is* OPERA-CT, so the projection head only has to recover that
probe. The failure mode flagged in §2.5 as a risk is the effect actually driving the
results.

### T1 (free-form Qwen2-Audio narration), measured the same way

All 6,898 descriptions generated (~3.4 h, batch 8, NF4 on a shared GPU).

| | distinct texts | mean words | text-only AUROC (wheeze) |
|---|---:|---:|---:|
| dataset | 27 | 10.0 | 0.390 |
| model | 193 | 24.0 | 0.505 |
| signal | 1,786 | 27.7 | 0.514 |
| all | 3,552 | 39.3 | 0.579 |
| **T1 free-form** | **2,448** | 54.1 | **0.590** |

Two things worth keeping:

- **T1 carries the most label information of any condition** — despite hallucinating
  badly. A crude negation check found adventitious sounds asserted in 68.9% of
  descriptions of `normal` cycles (upper bound; the rule is rough). An unreliable
  detector still carries signal, which is why T1 must be in E1 rather than dismissed.
- **T1 is not the maximally diverse condition.** Qwen2-Audio repeats itself: 2,448
  distinct texts, between `signal` and `all`. The predicted "6,898 unique" did not
  happen, which further undercuts resolution as the operative variable.

### Boundary of this claim — three untested substitutions

1. **Audio tower is OPERA-CT, not AST.** §3 specifies AST for main runs precisely to
   avoid the COLA-family circularity; OPERA-CT was used because it is on the server
   and AST is not. Strictly, this run *is* the E3 circular arm, not the main arm.
2. **Text tower is `bert-base-uncased`**, the only English encoder cached on the
   offline server. RespiraMFM used Phi-2. A weak text tower is a live alternative
   explanation.
3. **This probes the projection directly.** RespiraMFM uses the aligned projector as
   *initialisation for instruction tuning* and measures end-to-end. Probing tests
   their Fig. 5 clustering claim, which is fair, but it is not their pipeline.

So the defensible statement is narrow: *with an OPERA-CT audio tower and a
general-purpose text tower, none of the four text conditions yields a representation
better than the raw features, and the apparent ranking is explained by circular
dependence between text and audio towers.* Whether an AST tower plus a medical text
encoder changes this is **untested**, and both weights need downloading onto a server
with no outbound network.

## 2.7 Full matrix — 15/15 alignment runs below baseline (2026-08-07)

Re-ran everything in the configuration §3 actually specifies: **AST audio tower**
(ImageNet/AudioSet, no lineage with any text-generating model) and
**Bio_ClinicalBERT text tower**, five text conditions, both targets, three seeds,
official patient-independent ICBHI split.

### Frozen-feature baselines (linear probe, identical split and seeds)

| audio tower | wheeze MCC | wheeze AUROC | crackle MCC | crackle AUROC |
|---|---:|---:|---:|---:|
| OPERA-CT (respiratory FM) | 0.184 | 0.640 | 0.159 | 0.610 |
| **AST (general AudioSet)** | **0.516** | **0.857** | **0.254** | **0.689** |

### Alignment results, AST + Bio_ClinicalBERT

| condition | wheeze AUROC | Δ | crackle AUROC | Δ |
|---|---:|---:|---:|---:|
| **RAW (no alignment)** | **0.857** | — | **0.689** | — |
| t1_qwen2 | 0.629 | −0.228 | 0.606 | −0.083 |
| all | 0.613 | −0.244 | 0.615 | −0.074 |
| model | 0.579 | −0.278 | 0.599 | −0.090 |
| signal | 0.521 | −0.336 | 0.608 | −0.081 |
| dataset | 0.455 | −0.402 | 0.617 | −0.072 |

**Across both audio towers, both targets and five text conditions — 15 alignment
runs — not one beats simply probing the frozen features.** The better the audio
tower, the more alignment destroys (−0.054 worst case on OPERA-CT/wheeze, −0.402 on
AST/wheeze).

### The circularity mechanism, confirmed by a stated-in-advance prediction

Before running the AST arm we predicted: *if circularity is the mechanism, `model`
should lose its advantage once the audio tower is no longer OPERA-CT, because
`model` text is OPERA-CT's own likelihood output.*

| condition | OPERA-CT tower | AST tower |
|---|---:|---:|
| model | **0.586 (best)** | 0.579 (3rd) |
| all | 0.542 | 0.613 |
| t1_qwen2 | 0.533 | **0.629 (best)** |

`model` lost the top position exactly as predicted. Its apparent advantage was
dependence between the text and that specific audio representation, not clinical
grounding. A pre-stated prediction that held is much stronger evidence than a
post-hoc reading of the first sweep.

### A separate and arguably bigger finding: AST ≫ OPERA-CT

**AST beats the OPERA-CT respiratory foundation model by +0.217 AUROC on ICBHI
wheeze (0.857 vs 0.640) and +0.079 on crackle**, on OPERA's own benchmark corpus,
while OPERA reports beating general-audio models on 16 of 19 tasks.

Verified before believing it — this number is surprising enough to deserve controls:

| control | wheeze AUROC | expected |
|---|---:|---|
| real | **0.857** | — |
| shuffled train labels | 0.451 | ≈0.5 ✅ |
| shuffled feature↔id pairing | 0.488 | ≈0.5 ✅ |
| train/test patient overlap | 0 patients | 0 ✅ |

Also ruled out a duration artefact: AST pads to a fixed 1024 frames, and its
features do encode duration (max |r| = 0.72 over sampled dims), but **duration alone
gives wheeze AUROC 0.534** — nowhere near 0.857. The wheeze result is not a padding
artefact.

**But duration alone gives crackle AUROC 0.644 — above the OPERA-CT probe (0.610).**
That is a *second* crackle confound alongside the device/site one in §2.5. Crackle on
ICBHI is confounded from at least two directions; treat every crackle number here as
provisional.

## 2.75 Zero-shot from text closes the last frozen-tower escape route

The sweeps above evaluated the alignment with a linear probe on the projected audio.
That is not what contrastive alignment is *for* — CLIP's selling point is
classification from text prompts with no labelled test data. Two variants were added
(AST + Bio_ClinicalBERT, 3 seeds):

- **prompt** — true zero-shot, hand-written positive/negative sentences written in
  the training text's own style (an out-of-style prompt would test the text tower
  rather than the alignment);
- **prototype** — mean *train* text embedding per class. It uses train labels, so it
  is an upper bound on what this text space can express, not zero-shot.

| | probe | zs_prompt | zs_prototype | RAW probe |
|---|---:|---:|---:|---:|
| **wheeze** best | 0.629 (t1) | 0.613 (model) | 0.568 (t1) | **0.857** |
| **crackle** best | 0.617 (dataset) | 0.567 (all) | 0.631 (all) | **0.689** |

Zero-shot does not rescue the method. The best figure anywhere — crackle prototype
at 0.631 — is still below simply probing the raw features (0.689), and on wheeze the
gap is far larger (0.613 vs 0.857).

Prompt-based zero-shot beat the probe in only 2 of 10 condition×target cells
(wheeze/dataset, wheeze/model), so there is no general "zero-shot works where
probing fails" effect. An earlier read of three partial seed-0 rows suggested one;
the full sweep does not support it.

**With probing and both zero-shot variants exhausted, the frozen-tower design has no
remaining escape route.** That is precisely why the trainable-encoder variant (§2.9)
is the experiment that matters: a frozen encoder's features are fixed, so the
projector can only remap them, never reorganise them the way CLIP's image tower does.

## 2.9 Trainable encoder (CLIP-style) — the variant under test

`src/clip_finetune.py` fine-tunes AST itself from AudioSet weights against frozen
text embeddings. The text tower stays frozen: at 3,061 training cycles, training both
would add overfitting without adding capacity where it matters.

Guards, because overfitting is the obvious failure mode at this scale:
patient-level train/val carve-out (asserted, never segment-level), early stopping on
validation InfoNCE with best-weight restore, three seeds, and the frozen-encoder
number printed alongside as the bar to clear.

## 2.95 The split was not the official one (found 2026-08-07)

Everything above ran on a split the fidelity-oracle pipeline created in
`step1_parse_and_segment.py:77`:

```python
def get_split(patient_id: int) -> str:
    """ICBHI official 60/40 patient-independent split."""
    train_patients = set(range(101, 161))
```

**The docstring is wrong.** ICBHI ships an explicit per-recording partition
(`ICBHI_challenge_train_test.txt`) and the id threshold is not it. Checked directly:

| | official file | id-threshold split |
|---|---|---|
| ratio | 540 / 381 recordings (**58.6 / 41.4 ≈ 60/40** ✓) | 3,450 / 3,448 cycles (**50/50**) |
| train patients | 79 | 60 |
| test patients | 49 | 66 |
| **agreement** | — | **53.5%** of recordings (492/920) |

428 recordings sit on the opposite side from the official assignment.

**The official partition is also not strictly patient-independent**: patients **156**
and **218** appear on both sides. `src/official_split.py --strict` drops their test
segments, giving 4,142 train / 2,636 test (61/39) over 79 + 47 patients with zero
overlap.

The two splits trade off: ours was non-standard but strictly clean, the official one
is standard but slightly leaky. Report both.

### What this invalidates, and what survives

Internal comparisons are unaffected — every condition ran on the same split, so the
relative findings stand. **Absolute numbers are not comparable to any published
ICBHI result** and must never be quoted against RespiraMFM's or OPERA's tables.

The headline finding was re-checked under the official split and holds:

| audio tower | wheeze (old → official) | crackle (old → official) |
|---|---|---|
| OPERA-CT | 0.640 → 0.606 | 0.610 → 0.621 |
| **AST** | 0.857 → **0.796** | 0.689 → **0.729** |
| **AST − OPERA-CT** | +0.217 → **+0.190** | +0.079 → **+0.108** |

AST still beats the respiratory foundation model by a wide margin on both targets;
on crackle the gap widens. Absolute values move (wheeze down, crackle up), which is
exactly why the split had to be pinned down before any number went into a paper.

## 2.96 Both splits, side by side — every conclusion is split-invariant

wheeze, AST audio tower, Bio_ClinicalBERT text tower, 3 seeds:

| setting | id-threshold AUROC | official AUROC | official zs_proto |
|---|---:|---:|---:|
| **RAW (no alignment)** | **0.857** | **0.796** | — |
| trainable enc / t1_qwen2 | 0.828 | **0.771** | 0.650 |
| trainable enc / all | 0.771 | 0.739 | 0.586 |
| frozen proj / t1_qwen2 | 0.629 | 0.657 | 0.582 |
| frozen proj / all | 0.613 | 0.636 | 0.602 |

All four conclusions reproduce under both splits:

1. **Trainable encoder ≫ frozen projector** — +0.199 (id-threshold), +0.114 (official).
2. **Free-form LLM narration beats the structured schema** — t1 > all in **4 of 4**
   split × training-mode cells. 师兄's hypothesis that a designed schema gives a
   finer information mapping than natural description is contradicted every time.
3. **Alignment never beats not aligning** — best 0.771 vs RAW 0.796.
4. **AST ≫ OPERA-CT** — +0.190 (official), +0.217 (id-threshold).

That the conclusions survive a split disagreeing on 46.5% of recordings matters more
than any individual number.

### The zero-shot jump was sample size, not an easier split

Official-split zero-shot (0.58–0.65) sits far above the id-threshold trainable runs
(0.48–0.59), which raised the question of whether the official partition simply makes
the task easier. It does not: the *frozen* official runs reach the same 0.58–0.60 band
as the *trainable* id-threshold runs, so the driver is the larger training set
(4,142 vs 3,450) giving a steadier prototype estimate. No extra control run needed.

### The one thing alignment actually buys

`trainable / t1_qwen2` on the official split: probe 0.771, **zero-shot prototype
0.650**. Zero-shot gives up 0.121 while requiring **no labelled test data at all**,
and the raw features cannot do it — they have no text space to query.

So the only defensible positive claim from this line is not accuracy but capability:
*contrastive alignment trades ~0.12 AUROC for a label-free inference path.* Whether
that trade is worth making is a deployment argument, not a benchmark one.

## 2.8 Where this leaves the paper

The originally planned contribution ("structured schema is the best text to align
to") is dead: no text condition helps, so their ranking is not a result worth
reporting on its own. What survives is stronger and honestly obtained:

1. **Contrastive audio–text alignment on frozen respiratory encoders does not
   produce a usable representation.** 15/15 runs below baseline, two audio towers,
   two text towers, five text sources. The failure is systematic, not a bad-condition
   accident.
2. **Apparent gains come from circular dependence between the text source and the
   audio tower** — established by a prediction made before the confirming run.
3. **A general AudioSet model substantially outperforms the domain-specific
   respiratory foundation model on ICBHI adventitious-sound detection**, with shuffle
   and duration controls.

Finding 3 does not depend on contrastive learning working at all, which makes it the
most robust asset here.

### Still-open caveat

This probes the projection directly. RespiraMFM uses the aligned projector to
*initialise instruction tuning* and measures end-to-end. Our result refutes the
claim that alignment yields a better representation (their Fig. 5), but does not
test their full pipeline. Say so explicitly in any writeup.

## 3. Architecture (fixed — do not tune)

```
text  →  frozen text encoder (MedGemma-4B-IT / Phi-2)  →  e_t
audio →  log-mel spectrogram → frozen AST/ViT          →  e_a
                     ↓
        MLP projection head   ← the only trained module
        768 → 1024 → D_LLM,  LayerNorm + ReLU + dropout 0.1
                     ↓
        InfoNCE,  τ = 0.07,  lr 1e-3,  500 epochs,  AdamW
```

Configuration is copied from RespiraMFM Appendix D verbatim. **The independent
variable is the text, not the optimiser.** Any hyper-parameter change contaminates
the ablation.

### Audio tower: use AST, not OPERA-CT

StethoLM's audio encoder is a **domain-adapted COLA** encoder. OPERA-CT is
**HTS-AT + COLA-style contrastive pretraining** — the same family. Aligning
OPERA-CT features to text produced by a COLA-based model maximises circularity: the
projection head only has to re-discover a near-identical encoder.

- **Main experiments: AST / spectrogram ViT** (cross-architecture, clean).
- **OPERA-CT appears only in E3**, as a *demonstration* of the circularity effect.

---

## 4. Text conditions (the core variable)

| ID | Text source | Relation to audio | Role |
|---|---|---|---|
| **T0** | Patient metadata (age, sex, chest location, device, diagnosis) | **independent** | baseline to beat (RespiraMFM-style) |
| **T1** | StethoLM free-form clinical narrative | derived | 师兄's original proposal |
| **T2** | Structured schema, **full** | mixed | the hypothesis |
| **T2a** | Schema, **expert-annotated fields only** (crackle/wheeze label, chest location, device) | **independent** | circularity control |
| **T2b** | Schema, **model/signal-derived fields only** (OPERA-CT likelihood, band energy, transient sharpness) | **fully derived** | circularity control |

T2/T2a/T2b come from `clinical_report.py --provenance`, reusing the provenance tags
already in the schema (`opera-ct` / `judge_d` / `signal-proxy` / `ICBHI_expert`).

---

## 5. Experiments

| | Experiment | Question |
|---|---|---|
| **E0** | **Recording-confound baseline**: `dataset` text (location + device only) | **Mandatory on every crackle number.** It already reaches text-only AUROC 0.637 — any gain not measured against it is uninterpretable. |
| **E1** | T0 vs T1 vs T2 | **Main result.** Is structured schema best, and does the ranking track distinguishable-state count? |
| **E2** | provenance sweep: dataset / signal / model / all | **How much is circular?** High retrieval + flat downstream = caught. |
| **E3** | Audio tower: AST vs OPERA-CT | Architecture-level confirmation of E2. |
| **E4** | Zero-shot transfer to KAUH | Deployment argument: what to do when metadata is missing. |
| **E5** *(if time)* | Training-data fraction curve | Mirrors RespiraMFM Fig. 3. |

**Run wheeze first.** §2.5 shows crackle is confounded by device/site on ICBHI while
wheeze is not, so wheeze gives the cleaner read on whether the text condition matters
at all. Crackle still runs, always beside E0.

### Evaluation — the one rule

**Never report alignment/retrieval score as the headline.** Under a circular text
source it is guaranteed to look good and means nothing.

- **Primary metric: downstream AUROC**, following RespiraMFM's protocol so numbers
  are directly comparable (supervised in-domain + zero-shot held-out).
- **Report alignment score in a separate column.** The *gap* between alignment
  quality and downstream transfer **is** the circularity evidence — and it is the
  most interesting figure in the paper.

---

## 6. Datasets — verified availability (2026-08-06)

### Freely downloadable, no application

| Dataset | Link | Has expert acoustic labels? | Has symptom metadata? |
|---|---|---|---|
| **ICBHI 2017** | [bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr) | ✅ **event-level crackle/wheeze** | ✗ (age, sex, BMI, device placement only) |
| **Coswara** | [iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) | ✗ | ✅ **rich** (fever, cough, fatigue, smell loss, …) |
| **COUGHVID** | [Zenodo 4048312](https://zenodo.org/records/4048312) | ✗ | ⚠️ limited (fever, muscle pain) |
| **KAUH** | [Mendeley jwyy9np4gv/3](https://data.mendeley.com/datasets/jwyy9np4gv/3) | ⚠️ disease labels, device placement | ✗ |

The **OPERA benchmark repo** ([evelyn0414/OPERA](https://github.com/evelyn0414/OPERA))
already scripts the download and preprocessing for ICBHI, Coswara, COUGHVID and
KAUH — reuse it rather than rebuilding loaders.

### Requires application (do not block on these)

| Dataset | Access |
|---|---|
| **UK COVID-19 Vocal Audio** | Full edition: DUA via `DataAccess@ukhsa.gov.uk`. **An open-access edition exists on [Zenodo 10043978](https://zenodo.org/records/10043978)** (cough/exhalation, no speech) — check whether it suffices. |
| **TBscreen** | Request from authors (Sharma et al., *Sci Adv* 2024). |
| **CodaTB** | CODA TB DREAM Challenge — registration/DUA. |

**Decision: build the paper on ICBHI + Coswara + COUGHVID + KAUH.** Four free
corpora are enough for E1–E4. Start the UK COVID-19 open-access download in W1 as a
bonus; never let it gate the timeline.

### ⚠️ Design tension to resolve in W1

**Only ICBHI has event-level crackle/wheeze annotations** — and the schema (T2/T2a)
is built on exactly those. Meanwhile **T0 needs symptom-rich metadata**, which ICBHI
does not have and Coswara does.

No single corpus supports all five text conditions. Two options:

- **Option A (recommended).** Run the **main comparison E1–E3 on ICBHI**, where all
  of T0/T1/T2/T2a/T2b can be constructed (T0 from ICBHI's own demographics + device
  + location). Use **Coswara / COUGHVID / KAUH for E4 zero-shot** only.
  → Clean, self-consistent, one corpus per claim.
- **Option B.** Run E1 across corpora with a per-corpus "best available metadata"
  T0. → Broader, but T0 is no longer a fixed condition, weakening the comparison.

**Go with A unless the meeting says otherwise.** Note in the paper that ICBHI's
metadata is thin, which makes T0 a *conservative* baseline — that is an honest
limitation, not a flaw.

---

## 7. Six-week schedule

```
W1  Infrastructure
    ├ StethoLM inference running; 20 ICBHI descriptions generated and eyeballed
    ├ clinical_report.py --provenance  →  T2 / T2a / T2b
    ├ Download ICBHI (have) + Coswara + COUGHVID + KAUH via OPERA scripts
    └ RESOLVE the Option A/B design tension
W2  Pipeline
    ├ Two-tower + InfoNCE training script (RespiraMFM config, verbatim)
    └ T0 baseline runs and produces a sane number
W3  Main experiments
    └ T1 / T2 / T2a / T2b, supervised AUROC
W4  Ablations + zero-shot
    ├ E3 audio tower (AST vs OPERA-CT)
    └ E4 zero-shot transfer
W5  Analysis and figures
    └ Fig 1 main result · Fig 2 provenance ablation
      Fig 3 alignment-vs-transfer gap · Fig 4 zero-shot
W6  Writing
```

**Highest-risk item: W1 dataset acquisition.** If Coswara/COUGHVID/KAUH are not on
the server by end of W1, E4 is at risk. Start these downloads on day 1.

---

## 8. Protocol discipline (carried over from the fidelity-oracle work)

Non-negotiable, because these are the exact traps that produced a false conclusion
last time:

1. **Patient-level splits only.** Never split at segment level — assert zero patient
   overlap between train/val/test and keep the assertion in code.
2. **Report MCC and AUROC, never accuracy alone.** Also report the
   predicted-positive rate; a degenerate all-negative classifier still scores ~0.49
   on a balanced set.
3. **Multi-seed.** Three seeds minimum, report mean ± sd. A single-seed delta
   between text conditions is not a result.
4. **Verify before summarising.** Before writing any number into the paper, re-check
   the split, the feature source, and which code actually ran.

---

## 9. What to bring to meeting #1

1. The experiment matrix (§4–§5).
2. **Why the main runs use AST, not OPERA-CT** (COLA-family circularity).
3. **The circularity risk, converted into the provenance ablation** (§4, E2).
4. StethoLM output samples + the dataset availability table (§6).
5. The Option A/B design tension (§6) — a decision to make together.

Items 2 and 3 matter most: both were absent from the original proposal, both
*support* the hypothesis rather than undercutting it, and both show design rather
than execution.
