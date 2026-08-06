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

## 2. Contributions (honest ranking)

1. **A text-source taxonomy** for respiratory audio–text alignment: metadata /
   free-form LLM narration / structured schema, held under one fixed architecture
   and one fixed hyper-parameter set.
2. **Structure beats fluency.** Schema text with explicit clinical dimensions
   provides a finer information mapping to the spectrogram than free-form
   description of the same audio. *(This is the headline claim.)*
3. **The circularity control.** When text is generated *from* the audio, part of
   the alignment signal is self-recovery, not clinical grounding. We separate the
   two with a provenance ablation and an audio-tower ablation, and show that the
   **expert-grounded** portion of the schema is what carries transferable signal.

Contribution 3 is what makes this more than an ablation row. It also pre-empts the
reviewer question that would otherwise sink the paper.

---

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
| **E1** | T0 vs T1 vs T2 | **Main result.** Is structured schema best? |
| **E2** | T2a vs T2b vs T2 | **How much is circular?** T2b high alignment + low transfer = caught. |
| **E3** | Audio tower: AST vs OPERA-CT | Architecture-level confirmation of E2. |
| **E4** | Zero-shot transfer to held-out corpora | Deployment argument: what to do when metadata is missing. |
| **E5** *(if time)* | Training-data fraction curve | Mirrors RespiraMFM Fig. 3. |

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
