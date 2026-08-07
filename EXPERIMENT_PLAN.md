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
