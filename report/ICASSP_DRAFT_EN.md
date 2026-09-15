# Successful Clinical Audio–Metadata Alignment Does Not Imply Disease Transfer: A Pairing-Controlled Audit

> This is the editable English narrative source. The submission-formatted version is
> [`../paper/icassp2027/main.tex`](../paper/icassp2027/main.tex). The Chinese evidence
> ledger remains [`ICASSP_DRAFT_ZH.md`](ICASSP_DRAFT_ZH.md).

## Abstract

Clinical audio models increasingly align recordings with participant metadata, but
successful correspondence does not establish a transferable disease representation.
Metadata mixes disease-related information with demographics, symptoms, history, and
cohort structure. We introduce a Pairing-Controlled Transfer Audit comparing correct pairs,
within-disease-label shuffles, label-and-recorded-sex-preserving shuffles, and global
shuffles. On UKCOVID, Correct improved profile
retrieval and sex decodability over Within-label for AST-6L and OPERA-CT, while matched
COVID effects were +0.006 and -0.002. A post-hoc control shuffling within both label and
recorded sex substantially attenuated the sex-probe contrasts across AST, OPERA, and HeAR,
whereas small residual retrieval differences remained. Source-only and target-assisted
readouts did not establish consistent disease gains across backbones; intervals still
allowed beneficial effects in individual settings. Secondary analyses on CODA TB,
Cambridge COVID-19 Sounds, and Coswara extended the descriptive pattern without
establishing a universal null effect.
Correspondence learning, retained participant information, and disease transfer are
distinct empirical claims requiring distinct controls.

## 1. Introduction

Clinical audio models increasingly use participant metadata not only as prediction-time
input, but also as supervision for representation learning. Pretrained audio encoders range
from general-audio models such as the Audio Spectrogram Transformer (AST) to respiratory-
and health-specific representations such as OPERA and HeAR. Recent multimodal systems
further align each recording with clinical text derived from the same participant.
RespiraMFM, for example, freezes its audio and text encoders and trains an audio-side
projector for audio-clinical-text alignment before downstream multimodal prediction.

This setup creates an evaluation problem: a model can become better at matching an audio
recording to its associated participant profile without becoming better at predicting
disease in a different patient population. A contrastive objective rewards any audio
feature that helps identify the assigned metadata profile, while clinical metadata contains
demographics, symptoms, medical history, missingness patterns, and cohort structure in
addition to disease-related information. Successful alignment can therefore reflect at
least three phenomena: **profile correspondence**, **label-conditioned population
association**, or **disease information that transfers across populations**. Alignment
loss, profile retrieval, or source-cohort AUROC alone cannot distinguish among them.

The missing control is the **pairing relation itself**. Consider a COVID-positive
participant. In our Within-label control, that participant's audio is paired with metadata
from a different COVID-positive participant: the disease class is preserved, but the
original participant profile is not. Comparing Correct with Within-label therefore asks
what the observed audio-profile pairing adds beyond metadata structure already associated
with the disease label. Comparing Within-label with Global measures the contribution of
label-conditioned population association. Disease labels construct these control pairings;
they are not included in the metadata text presented to either encoder.

We use these interventions in a **Pairing-Controlled Transfer Audit** that asks three
sequential questions: **Was pairing learned? What became more decodable? Does disease
prediction improve after population balancing?** Profile retrieval is a manipulation check,
information probes characterize the resulting representation, and participant-disjoint
matched evaluation tests disease transfer against both Within-label and frozen raw-audio
references.

Across UKCOVID and secondary analyses on CODA TB, Cambridge COVID-19 Sounds, and Coswara,
we observe a consistent separation between these outcomes. Correct pairing repeatedly
improves profile correspondence; its largest probe contrast involved sex but was
substantially attenuated by a sex-preserving control. In contrast, matched disease effects
are small and directionally inconsistent across backbones, and Correct does not consistently
outperform the corresponding frozen raw representation. Our
claim is not that metadata alignment is generally ineffective. Rather, successful profile
alignment and improved disease transfer are separate empirical claims requiring separate
controls.

## 2. Related Work

AST, OPERA, and HeAR learn general-purpose representations from spectrograms or
health-related audio. These methods ask whether respiratory recordings contain useful
health information, but not what additional information is retained when audio is aligned
with participant metadata. Metadata has also been used directly as supervision for
respiratory contrastive pretraining.

Clinical multimodal systems combine respiratory audio with demographics, symptoms, and
medical history. Explicit alignment methods go further by using clinical text as
representation-level supervision. RespiraMFM aligns frozen audio and text encoders through
an audio-side projector before a downstream Phi-2/LoRA stage. We audit a paper-faithful
implementation of that Stage-1 alignment form, not its complete downstream system.

Respiratory-audio performance can also be strongly affected by symptom-conditioned
recruitment, recording protocol, and cohort composition. Prior work has largely asked
whether fusion or alignment improves downstream accuracy. We ask what that gain represents
and whether it survives population balancing.

## 3. Data and Methods

### 3.1 Datasets and evidence hierarchy

UKCOVID is the discovery dataset. After objective audio filtering, its Standard training
split contains 20,714 participants. We evaluate Standard test (11,119 participants), a
covariate-matched test (1,814), and a participant-disjoint matched-long sensitivity set
(4,196). Recruitment source alone predicts COVID with AUROC 0.9966 in Standard training but
0.5000 after matching. Because UKCOVID test data informed earlier protocol development, the
analysis is explicitly exploratory rather than untouched confirmation.

Three external analyses test whether the result shape recurs:

- **CODA TB:** 1,081 eligible participants and a frozen 200-participant matched target from
  the released training data;
- **Cambridge COVID-19 Sounds:** 989 participants and a 200-participant target under a
  predefined strict-COVID reconstruction of its Task-2 cough subset; and
- **Coswara:** 1,701 participants and a 200-participant target obtained through a
  model-blind secondary matching procedure.

CODA is a match-first secondary analysis, Cambridge is not an official benchmark
replication, and Coswara is a post-hoc stress test. Matching balances prespecified measured
covariates; it does not remove unmeasured confounding.

### 3.2 Audio representations and pairing interventions

We use two frozen audio backbones for the primary UKCOVID audit, chosen to represent
complementary pretraining regimes. **AST-6L** retains the first six Transformer blocks of
the 12-layer AudioSet-finetuned AST encoder. This depth was fixed before the present audit
from an independent respiratory-audio comparison: six blocks outperformed the full
12-layer model on both ICBHI wheeze and crackle detection while using about half the
parameters, and the same comparison favored AST-6L on three of four patient-level KAUH
tasks. AST-6L was not selected using UKCOVID transfer performance.

**OPERA-CT** is the released frozen 768-dimensional contrastive Transformer from the OPERA
respiratory acoustic foundation-model family. Because its pretraining corpus includes
UKCOVID training audio, we treat it as a domain-pretrained comparator rather than an
independent pretraining condition; the present UKCOVID targets remain participant-disjoint
from our training split. **HeAR** is added in the external analyses as a third frozen
health-acoustic representation.

Recordings use each backbone's frozen preprocessing pipeline and are aggregated to one
participant-level audio representation before alignment. A frozen Phi-2 encoder represents
a deterministic schema of available demographics, smoking, respiratory history, symptoms,
and explicit missing values.

The text excludes disease labels, participant identifiers, recruitment source,
country/site, platform, device, timestamps, and recording artefacts. The audited risk is
therefore statistical association through patient context, not direct target or domain
leakage.

Only the audio-side projector is trained. Each arm uses five seeds and 500 epochs, and only
the final checkpoint is evaluated. Within each seed, the arms share initialization, audio
batches, and dropout streams; only the pairing changes. Pair assignments remain fixed
during training. For the primary source-only endpoint, no matched split selects the
projector, readout, or hyperparameters. The experiment audits a Stage-1-style alignment
mechanism rather than reproducing the complete downstream RespiraMFM system.

### 3.3 Pairing-controlled intervention

For audio `a_i`, metadata `m_i`, and disease label `y_i`:

```text
Correct:       a_i <-> m_i
Within-label:  a_i <-> m_j, where y_i = y_j and i != j
Global:        a_i <-> m_k, where i != k
```

Within-label breaks the observed participant-profile relationship while preserving
label-conditioned metadata structure in expectation. Global additionally removes systematic
label-conditioned pairing, although accidental same-label or duplicate-profile matches can
remain. We denote Correct-minus-Within as `Delta_pair = C - W`, the increment associated
with observed participant-profile pairing beyond label-conditioned association. We denote
Within-minus-Global as `Delta_label = W - G`, which measures label-conditioned population
association. Neither is a causal estimand, and `Delta_pair` is not a participant-identity
effect. `Correct - Raw` tests whether alignment improves on the frozen audio representation.

A post-hoc sex-preserving control, `W(y,s)`, pairs `a_i` with `m_j` subject to
`y_i = y_j`, `s_i = s_j`, and `i != j`, where `s` is recorded sex. It uses the same
20,714 training participants and permits neither self-pairing, singleton-stratum fallback,
nor cross-stratum fallback; the five-person positive/missing-sex stratum remains valid. A
different participant can still share the same schema text. Exact-schema collisions were
6.35-6.61% in `W(y,s)` versus 3.03-3.33% in `W`, so this control is described as
"different participant," not "different profile."

### 3.4 Audit endpoints

- **Correspondence:** audio queries retrieve metadata profiles on source validation. Because
  multiple participants can share a schema, identical profiles count as the same semantic
  target. We use macro-profile MRR so frequent schemas do not dominate.
- **Information:** identical regularized probes decode sex, age, symptoms, acquisition and
  cohort variables, and disease. Decodability establishes availability, not causal use.
- **Transfer:** for the primary endpoint, source data alone select and calibrate downstream
  readouts; paired AUROC, NLL, and Brier score are read once on matched targets. A separately
  reported target-assisted diagnostic trains, selects, and calibrates its readout only on
  matched-long before transport to participant-disjoint matched.
- **Alternative explanations:** frozen raw-audio and metadata-only references, a fixed MLP,
  and target calibration test simpler explanations for the result.

Retrieval intervals jointly resample unique profiles and seeds after averaging repeated
participant queries within profile. Probe and disease intervals jointly resample
participants and seeds. Arm and seed pairing is preserved throughout; UKCOVID participants,
not presumed matched-pair blocks, are the resampling units.

## 4. Results

### 4.1 Pairing controls separate sex decodability from residual retrieval

On UKCOVID validation, Correct improved macro-profile MRR over Within-label for both
backbones. `Delta_pair` was +0.00318 [0.00161, 0.00492] for AST and +0.00323
[0.00155, 0.00500] for OPERA. All five seed-level effects were positive. Retrieval is a
manipulation check: later disease results cannot be explained simply by failure to learn the
assigned pairing.

Across the 12 dataset-by-backbone settings, all retrieval point estimates were
positive and 11/12 confidence intervals excluded zero. We do not pool these effects because
profile libraries and evidence status differ across datasets.

The strongest probe effect was sex. On UKCOVID matched participants, `Delta_pair` sex-probe
AUROC was +0.1912 [0.1669, 0.2151] for AST and +0.1125 [0.0943, 0.1311] for OPERA. Age
increased for AST but not OPERA; acquisition and cohort effects were less consistent.

Sex `Delta_pair` intervals excluded zero in all 12 settings. Raw audio usually
had higher absolute demographic decodability than aligned representations, so alignment did
not create demographic information absent from audio. It selectively retained that channel
relative to Within-label.

A post-hoc label-and-sex-preserving shuffle, `W(y,s)`, substantially attenuated the
UKCOVID sex-probe contrast. Correct-minus-`W(y,s)` was -0.0004 [-0.0153, 0.0147] for AST,
+0.0067 [-0.0054, 0.0192] for OPERA, and +0.0064 [-0.0011, 0.0136] for HeAR;
`W(y,s)`-minus-Within was +0.1916 [0.1683, 0.2142], +0.1058 [0.0896, 0.1222], and
+0.0797 [0.0667, 0.0928]. This shows sensitivity to recorded-sex consistency in the pairing
rule, not causal use of sex or equivalence between Correct and `W(y,s)`.

Residual retrieval must be interpreted separately. Correct-minus-`W(y,s)` macro-profile
MRR was +0.00164 [0.00001, 0.00335], +0.00099 [-0.00096, 0.00300], and +0.00394
[0.00175, 0.00637] for AST, OPERA, and HeAR. Restricting the original Correct-versus-Within
candidate library to the same recorded sex gave +0.00140 [-0.00065, 0.00350], +0.00060
[-0.00143, 0.00256], and +0.00356 [0.00096, 0.00638]. Restricting it to the same label and
sex gave +0.00245 [-0.00001, 0.00499], +0.00094 [-0.00149, 0.00329], and +0.00451
[0.00115, 0.00813]. Candidate control therefore attenuated, but did not universally remove,
profile correspondence.

### 4.2 Correspondence gains lack consistent disease-transfer gains

On UKCOVID matched, disease `Delta_pair` AUROC was +0.0062
[-0.0144, 0.0279] for AST and -0.0020 [-0.0191, 0.0168] for OPERA. For AST, raw, Correct,
and Within AUROC were 0.538, 0.529, and 0.523; for OPERA they were 0.577, 0.555, and 0.557.
A fixed MLP did not reveal a hidden effect.

For the primary matched disease contrast, Correct-minus-`W(y,s)`, point estimates were
positive in five of the 12 dataset-by-backbone settings and negative in seven. No confidence
interval established a positive gain. In the nine locked secondary settings, estimates
ranged from -0.0634 to +0.0476: CODA was +0.0476, -0.0077, and -0.0102; Cambridge was
-0.0037, +0.0054, and -0.0634; and Coswara was +0.0132, -0.0152, and -0.0076 for AST,
OPERA-CT, and HeAR, respectively. All nine intervals included zero. Correct also showed no
consistent advantage over frozen raw audio. This is a descriptive recurrence, not a
meta-analysis, an equivalence result, or proof of a universal null effect.

A target-assisted diagnostic trained and calibrated the fixed linear readout on matched-long
before applying it to participant-disjoint matched. Correct-minus-Within AUROC was -0.0082
[-0.0337, 0.0177], +0.0287 [-0.0017, 0.0626], and +0.0047 [-0.0278, 0.0408] for AST,
OPERA, and HeAR. The positive OPERA estimate remains uncertain but allows a potentially
meaningful benefit. These results did not establish a cross-backbone advantage, and no
Correct-minus-Raw advantage was established. Because source-only and target-assisted
estimates were not directly contrasted, this diagnostic neither proves nor excludes an
effect of readout training.

The source-only `W(y,s)` disease results were likewise inconclusive. Absolute `W(y,s)`
AUROC was 0.529 [0.504, 0.554], 0.554 [0.529, 0.577], and 0.532 [0.507, 0.557] for AST,
OPERA, and HeAR. Correct-minus-`W(y,s)` was -0.0003 [-0.0228, 0.0241], +0.0009
[-0.0169, 0.0196], and +0.0130 [-0.0023, 0.0287]; `W(y,s)`-minus-Within was +0.0065
[-0.0197, 0.0313], -0.0029 [-0.0219, 0.0164], and -0.0102 [-0.0256, 0.0053]. These
intervals do not establish equivalence or rule out individual positive effects.

### 4.3 Label association, fusion, and calibration

Within-label and Global separated in the source cohort, showing that label-conditioned
metadata association was learnable and would be mixed with exact pairing by a
Correct-versus-Global comparison alone.

Direct fusion did not provide a reproducible positive result. Adding raw audio to metadata
changed matched AUROC by only +0.0028 [-0.0016, 0.0073] for AST and +0.0041
[-0.0006, 0.0087] for OPERA. Metadata plus Correct exceeded metadata plus Within for AST
(+0.0156 [0.0034, 0.0276]) but not OPERA (+0.0021 [-0.0068, 0.0107]); neither Correct
fusion beat metadata plus raw audio.

With source calibration, Correct had worse matched NLL than Within even though AUROC
differences were negligible. Target calibration reduced both NLL gaps to approximately zero
without changing AUROC. This pattern is consistent with probability-scale mismatch after
cohort shift, but it does not identify the mechanism; recalibration created no discrimination
gain.

## 5. Discussion

The audit separates two outcomes that conventional evaluation can conflate. Correct
audio-metadata alignment repeatedly improved retrieval of participant profiles, showing
that the pairing intervention produced a measurable correspondence signal. Yet the most
reproducible additional channel was participant sex, while disease improvements were small,
directionally inconsistent, and
not robust to raw-audio controls or readout changes.

In plain terms, the model became better at matching a recording to the kind of participant
described by a metadata card, but not reliably better at diagnosing disease after measured
patient composition was balanced.

Within-label shuffling is essential. Correct versus Global alone mixes exact correspondence
with label-conditioned population association. Retrieval verifies the manipulation, probes
describe retained channels, and matched prediction tests clinical transfer. Future studies
should report all three together with a frozen raw reference and calibration under
population shift.

The same correspondence-transfer separation appeared with both a general-audio AST
representation and a respiratory-domain OPERA representation, reducing the likelihood that
the finding is specific to one encoder family. This does not establish backbone independence;
HeAR was added to UKCOVID only as a separately frozen post-hoc robustness analysis.

The sex result also requires careful wording. A probe shows decodability, not causal disease-
classifier use. The `W(y,s)` control substantially attenuated the original sex-probe contrast,
showing that it is sensitive to recorded-sex consistency retained by the pairing rule. It is
not a causal decomposition or equivalence test. Raw audio often contains still more
demographic information, so alignment did not create a new attribute. Residual retrieval
after sex control is a separate result and indicates that sex is not the complete pairing
signal.

The evidence remains bounded. UKCOVID is a discovery audit; CODA, Cambridge, and Coswara
are secondary or post-hoc sensitivities with 100-pair targets and wide intervals. Matching
balances measured covariates only; UKCOVID intervals resample participants rather than
matched-pair blocks; probes do not establish causal use; and the experiment
audits one Stage-1-style projector rather than a complete multimodal reasoning system. Our
synthetic study failed its preregistered general-mechanism gates, so we do not claim a
universal causal law. A proposed disease-invariant repair was not trained because the
single-dataset cohorts lacked prespecified support for the required counterfactual pairs;
this is a feasibility limitation, not a negative result for the repair.

## 6. Conclusion

Correct pairing reproducibly improved participant-profile correspondence, while the largest
sex-probe contrast was substantially attenuated by the sex-preserving control. Neither
result yielded a robust cross-backbone improvement in covariate-balanced disease prediction
or consistently outperformed raw audio. Successful audio-metadata correspondence is
therefore not sufficient evidence of a transferable disease representation. Correspondence,
retained information, and disease transfer must be evaluated separately.
