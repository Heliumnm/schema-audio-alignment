# Does Clinical Metadata Alignment Transfer Disease Evidence?
# A Pairing-Controlled Audit of Respiratory Audio under Cohort Shift

> **Draft status (2026-08-19).** UKCOVID is a discovery audit because its official test
> sets informed earlier protocol development. Coswara is a preregistered, pending external
> patient-level stress test; no Coswara model score is included. Citation keys and figure
> panels remain production placeholders.

## Abstract

Respiratory audio models increasingly align recordings with clinical context, although
age, symptoms, and medical history are not necessarily audible and may encode cohort
construction. We test whether such alignment learns participant correspondence or disease
evidence that transfers across populations. Frozen AST-6L and Phi-2 encoders are connected
by a RespiraMFM-style audio projector trained with correct metadata, metadata shuffled
within COVID label, or globally shuffled metadata. Within-label shuffling preserves
label-level co-occurrence while removing individual correspondence. In UKCOVID,
recruitment source predicts COVID with AUROC 0.9966 in Standard train but 0.5000 in a
covariate-matched test set. Correct pairing yields a small Standard AUROC gain over
within-label shuffling (+0.0140, 95% CI [+0.0040, +0.0235]). In the matched population,
however, the preregistered paired change in negative log-likelihood is worse
(Delta(-NLL) -0.0249, 95% CI [-0.0443, -0.0069]), while the AUROC difference is uncertain
(+0.0062, 95% CI [-0.0144, +0.0279]). Probes show that correct pairing retains real
participant correspondence, especially sex, but this does not become transferable disease
benefit. These discovery results motivate within-label pairing controls and
covariate-balanced evaluation as minimum audits for clinical audio--metadata alignment.

## 1. Introduction

Clinical audio models often use symptoms, demographics, medical history, or text assembled
from those fields. Contrastive alignment can make this context accessible to an audio
representation [RespiraMFM]. Yet, unlike an audible description of pitch or timing,
clinical metadata may describe the participant or recruitment process rather than the
waveform. A model can therefore learn the intended mapping while exploiting cohort
structure that does not transfer.

UKCOVID makes this ambiguity measurable. Its Standard training distribution combines a
largely positive Test-and-Trace cohort with a largely negative population-surveillance
cohort. Recruitment source alone predicts COVID with AUROC 0.9966 in Standard train but
0.5000 in the covariate-matched test population. Symptoms are also a strong recruitment
proxy: “no symptoms” occurs in 0.786 of training negatives and 0.019 of positives. Aligning
audio to symptom text may thus reinforce a source-specific association [UKCOVID-Audit].

A global shuffle cannot isolate this mechanism because it removes both participant
correspondence and disease-level co-occurrence. We introduce a **within-label shuffle**:
each recording is paired with another same-label participant's metadata. Correct and
within-label arms share label-level statistics and differ only in individual pairing.

Our contributions are:

1. a pairing-controlled protocol separating participant correspondence from label-level
   audio--metadata co-occurrence;
2. a transfer audit using source and covariate-matched populations, with calibrated NLL as
   the primary outcome rather than source AUROC alone; and
3. a probe-based decomposition showing that alignment learns genuine participant mapping
   without producing measurable matched-population disease benefit.

![Pairing-controlled audit](../figures/icassp_pairing_design.svg)

**Fig. 1.** Frozen audio/text branches, the three pairing controls, and evaluation under
Standard versus covariate-matched populations. Correct and within-label arms preserve the
same COVID-label co-occurrence.

## 2. Method

### 2.1 Data and representations

We learn representations on the patient-level UKCOVID Standard train split. After the
frozen audio-availability audit, it contains 20,714 participants. We evaluate Standard, the
primary covariate-matched test set, and the participant-disjoint matched-long sensitivity
set. Because official tests informed earlier protocol development, these results are a
**discovery audit**, not independent confirmation.

Audio is encoded by frozen six-layer Audio Spectrogram Transformer features [AST]; text is
encoded by frozen Phi-2 [Phi-2]. Metadata is deterministically rendered from age, sex,
smoking, asthma, other respiratory conditions, and symptoms, with missingness explicit.
COVID results, test details, recruitment source, timestamps, and recording artefacts are
excluded.

Only an audio-side projector is trained, following RespiraMFM Stage 1 [RespiraMFM]. It maps
768 to 1,024 to 2,560 dimensions using LayerNorm and ReLU after both linear layers and
dropout 0.1 after the first. For frozen audio embedding \(a_i\), frozen text embedding
\(t_i\), and projector \(f_\theta\), the one-directional objective is

\[
\mathcal{L}=-\frac{1}{N}\sum_i\log
\frac{\exp(\bar f_\theta(a_i)^\top\bar t_i/\tau)}
{\sum_j\exp(\bar f_\theta(a_i)^\top\bar t_j/\tau)},\quad\tau=0.07,
\]

where bars denote L2 normalization inside the loss. The primary downstream representation
is the raw post-ReLU output; L2-normalized output is a prespecified sensitivity.

### 2.2 Pairing arms and evaluation

The trained arms are **correct**, **within-label shuffled**, and **globally shuffled**.
Within each of five seeds, they share initialization, audio batch order, and dropout stream;
only pairing differs. A shuffled mapping is sampled once per seed, forbids self-pairs, and
remains fixed for 500 epochs. We use batch size 64 and Adam at \(10^{-3}\), with no scheduler
or weight decay. All 15 projector runs completed, and only their final checkpoints are
evaluated; no epoch is selected against disease outcomes. Raw frozen AST is the unaligned
reference. This is a Stage-1-style
reimplementation, not a reproduction of RespiraMFM's full Phi-2/LoRA system.

All representations use the same regularized logistic readout. Regularization is selected
per seed on complete Standard validation, where a source-domain Platt calibrator is also
fit. Matched results select no model or calibration parameter. The preregistered primary
comparison is correct minus within-label on matched, measured as paired
\(\Delta(-\mathrm{NLL})\); positive values favour correct pairing. AUROC is secondary.
Intervals hierarchically resample participants and seeds while retaining paired arm
predictions. Linear probes for recruitment source, sex, age at least 65, cough, and no
symptoms measure decodability, not causal classifier use.

### 2.3 External stress test

Before any external model score, we froze a Coswara v1.0 protocol using one
`cough-heavy.wav` per participant, objective waveform QC, decoded-PCM duplicate removal,
patient-disjoint development splits, and 1:1 covariate matching. Formal execution requires
at least 100 balanced pairs and adequate development class counts. A metadata-only capacity
audit provides a ceiling of 114 pairs and first passes deterministic balance trimming at
101; waveform QC can therefore close the gate. If it passes, the same pairing arms and raw,
artefact, metadata, and direct-fusion baselines are read once. UKCOVID and Coswara are not
pooled. This small analysis is an **external patient-level stress test**, never
confirmatory proof.

## 3. Results

### 3.1 Disease transfer

Table 1 reports absolute performance. Correct exceeds within-label AUROC in Standard
(0.649 versus 0.635). Under matching, aligned representations approach chance ranking and
the globally shuffled projector has lower NLL than the correct projector. Consequently,
correct-versus-raw improvement cannot alone establish semantic alignment: projection or
regularization can change calibration without using correct pairing.

**Table 1. COVID performance as AUROC / NLL (lower NLL is better).**

| Evaluation | Raw AST | Correct | Within-label | Global |
|---|---:|---:|---:|---:|
| Standard | 0.708 / 0.604 | 0.649 / 0.628 | 0.635 / 0.628 | 0.571 / 0.640 |
| Matched | 0.538 / 0.857 | 0.529 / 0.795 | 0.523 / 0.770 | 0.510 / 0.739 |
| Matched-long | 0.548 / 0.850 | 0.532 / 0.794 | 0.530 / 0.773 | 0.524 / 0.737 |

The paired contrast separates source correspondence from matched transfer (Table 2).
Correct pairing gives a detectable Standard AUROC increment of +0.0140, while its Standard
NLL difference is uncertain. On matched, the primary \(\Delta(-\mathrm{NLL})\) is -0.0249
(95% CI [-0.0443, -0.0069]); all five seed effects are negative. Matched AUROC changes by
only +0.0062 with an interval crossing zero. Matched-long shows the same NLL direction and
near-zero AUROC difference. L2 normalization preserves the direction: normalized matched
\(\Delta(-\mathrm{NLL})\) is -0.0179 [-0.0371, +0.0011], and matched-long is -0.0204
[-0.0328, -0.0082].

**Table 2. Correct minus within-label paired effects.**

| Evaluation | \(\Delta(-\mathrm{NLL})\) [95% CI] | \(\Delta\)AUROC [95% CI] |
|---|---:|---:|
| Standard | +0.0008 [-0.0047, +0.0062] | +0.0140 [+0.0040, +0.0235] |
| **Matched** | **-0.0249 [-0.0443, -0.0069]** | +0.0062 [-0.0144, +0.0279] |
| Matched-long | -0.0214 [-0.0330, -0.0101] | +0.0013 [-0.0121, +0.0155] |

### 3.2 What correspondence was learned?

The correct arm did not simply fail to learn its mapping. On matched, correct minus
within-label probe AUROC is +0.1912 [+0.1677, +0.2156] for sex and +0.0465
[+0.0069, +0.0854] for age at least 65. Effects are uncertain for recruitment source
(+0.0138 [-0.0205, +0.0473]), cough (+0.0098 [-0.0114, +0.0309]), and no symptoms
(+0.0151 [-0.0096, +0.0407]). Symptoms instead improve mainly between global and
within-label arms, consistent with label-level co-occurrence. Raw AST has higher absolute
AUROC than correct alignment on every probe (e.g., sex 0.871 versus 0.812 and age 0.742
versus 0.657). We therefore infer selective retention relative to the within-label
projector, not that alignment adds demographic information beyond raw AST.

![Paired effects](../figures/icassp_pairing_forest.svg)

**Fig. 2.** Correct-minus-within-label \(\Delta\)AUROC (A) and
\(\Delta(-\mathrm{NLL})\) (B) across Standard, matched, and matched-long. Any later
Coswara estimate is displayed separately and never pooled.

### 3.3 Coswara status

The Coswara source, modality, QC, matching, split, model family, outcomes, and interpretation
are frozen. Extraction and post-QC power gating remain pending, so no external result or
cross-dataset confirmation is claimed. The final paper will report either a separate
patient-level stress-test estimate if the 100-pair gate passes, or a transparent NO-GO if
the cohort is underpowered or imbalanced.

## 4. Discussion

Correct pairing produced both a source-domain ranking gain and strong participant-level sex
correspondence; calling alignment wholly ineffective would therefore be inaccurate.
Nevertheless, the preregistered matched NLL comparison was negative, matched AUROC was
uncertain, and matched-long agreed in NLL direction. In this setting, learning genuine
clinical metadata correspondence did not guarantee portable disease evidence.

Within-label shuffling makes this interpretation possible. Unlike global shuffling, it
retains the disease--metadata association and removes only participant identity. It should
therefore accompany global controls whenever cohort metadata is label-associated.

The study does not establish that metadata alignment is universally harmful or that AST
contains no COVID information. It tests one Stage-1 projector, one frozen backbone, and a
linear readout in an unusually confounded dataset. UKCOVID is exploratory; matched-long is
not independent replication; probe decodability does not establish causal classifier use;
and the pending Coswara cohort is near its minimum gate. A clear opposite-direction
Coswara result would define a boundary condition, not be hidden or used to move thresholds.
Future work should use untouched cohorts, additional backbones and readouts, and explicitly
separate audible evidence from independent clinical context.

## 5. Conclusion

In UKCOVID, correct audio--metadata pairing learns real participant correspondence and a
small source-domain ranking advantage, but not improved calibrated disease prediction
after covariate balancing. Clinical audio--metadata studies should minimally include a
within-label shuffle and a cohort-balanced transfer evaluation; otherwise, source gains
may be mistaken for portable disease evidence.

## References to resolve before IEEE formatting

- **[AST]** Audio Spectrogram Transformer original paper and implementation.
- **[Phi-2]** Phi-2 technical report/model card.
- **[RespiraMFM]** *RespiraMFM: A Multimodal Foundation Model with Contrastive
  Audio-Language Alignment for Respiratory Disease Identification*.
- **[UKCOVID-Audit]** UKCOVID cohort paper and matched confounding analysis.
