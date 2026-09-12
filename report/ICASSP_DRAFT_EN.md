# Successful Clinical Audio–Metadata Alignment Does Not Imply Disease Transfer: A Pairing-Controlled Audit

> This is the editable English narrative source. The submission-formatted version is
> [`../paper/icassp2027/main.tex`](../paper/icassp2027/main.tex). The Chinese evidence
> ledger remains [`ICASSP_DRAFT_ZH.md`](ICASSP_DRAFT_ZH.md).

## Abstract

Clinical audio models increasingly align recordings with participant metadata, but
successful correspondence does not establish a transferable disease representation.
Metadata mixes disease-related information with demographics, symptoms, history, and
cohort structure. We introduce a Pairing-Controlled Transfer Audit comparing correct pairs,
metadata shuffled within disease labels, and globally shuffled metadata. On UKCOVID,
Correct pairing improved profile retrieval for AST-6L and OPERA-CT and increased sex
decodability relative to Within-label (delta AUROC +0.191 and +0.113), showing that the
pairing intervention produced a measurable representational difference. Matched COVID
prediction changed by only +0.006 and -0.002.
Across a post-hoc UKCOVID HeAR extension and secondary analyses on CODA TB, Cambridge
COVID-19 Sounds, and Coswara, retrieval and sex decodability increased consistently,
whereas 0/12 dataset-by-backbone settings
established a positive matched-disease gain, and Correct did not consistently outperform
frozen raw audio. Correspondence learning, retained participant information, and disease
transfer are therefore distinct empirical claims requiring distinct controls.

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
improves profile correspondence and, most consistently, sex decodability. In contrast,
matched disease effects are small and directionally inconsistent across backbones, and
Correct does not consistently outperform the corresponding frozen raw representation. Our
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
during training. Matched data select no epoch, projector, readout, or hyperparameter. The
experiment audits a Stage-1-style alignment mechanism rather than reproducing the complete
downstream RespiraMFM system.

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

### 3.4 Audit endpoints

- **Correspondence:** audio queries retrieve metadata profiles on source validation. Because
  multiple participants can share a schema, identical profiles count as the same semantic
  target. We use macro-profile MRR so frequent schemas do not dominate.
- **Information:** identical regularized probes decode sex, age, symptoms, acquisition and
  cohort variables, and disease. Decodability establishes availability, not causal use.
- **Transfer:** source data alone select and calibrate downstream readouts; paired AUROC,
  NLL, and Brier score are read once on matched targets.
- **Alternative explanations:** frozen raw-audio and metadata-only references, a fixed MLP,
  and target calibration test simpler explanations for the result.

Intervals use participant-by-seed hierarchical bootstrap and preserve paired predictions.

## 4. Results

### 4.1 Correct pairing improves profile correspondence and sex decodability

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

A post-hoc label-and-sex-preserving shuffle, `W(y,s)`, localized the UKCOVID sex result.
Correct-minus-`W(y,s)` sex effects were -0.0004 [-0.0153, 0.0147] for AST, +0.0067
[-0.0054, 0.0192] for OPERA, and +0.0064 [-0.0011, 0.0136] for HeAR, while
`W(y,s)`-minus-Within was clearly positive for all three. Residual profile MRR after this
control was small and backbone dependent (+0.00164, +0.00099, and +0.00394).

### 4.2 Correspondence gains lack consistent disease-transfer gains

On UKCOVID matched, disease `Delta_pair` AUROC was +0.0062
[-0.0144, 0.0279] for AST and -0.0020 [-0.0191, 0.0168] for OPERA. For AST, raw, Correct,
and Within AUROC were 0.538, 0.529, and 0.523; for OPERA they were 0.577, 0.555, and 0.557.
A fixed MLP did not reveal a hidden effect.

Across all 12 settings, matched disease point estimates were positive in six and negative
in six. No confidence interval established a positive gain. Correct also showed no
consistent advantage over frozen raw audio. This is a descriptive recurrence, not a
meta-analysis and not proof of exact equivalence.

A target-assisted diagnostic trained and calibrated the fixed linear readout on matched-long
before applying it to participant-disjoint matched. Correct-minus-Within AUROC was -0.0082
[-0.0337, 0.0177], +0.0287 [-0.0017, 0.0626], and +0.0047 [-0.0278, 0.0408] for AST,
OPERA, and HeAR. It therefore did not reveal a stable disease benefit hidden by the source
readout and did not establish a Correct-minus-Raw advantage.

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
without changing AUROC. The source NLL penalty therefore reflected unsupported confidence
after cohort shift; recalibration repaired probability scale but created no transferable
discrimination.

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
classifier use. The `W(y,s)` control shows more specifically that the original effect largely
reflects retention of recorded-sex consistency. Raw audio often contains still more
demographic information, so alignment did not create a new attribute. Small residual
retrieval after sex control indicates that sex is not the complete pairing signal.

The evidence remains bounded. UKCOVID is a discovery audit; CODA, Cambridge, and Coswara
are secondary or post-hoc sensitivities with 100-pair targets and wide intervals. Matching
balances measured covariates only, probes do not establish causal use, and the experiment
audits one Stage-1-style projector rather than a complete multimodal reasoning system. Our
synthetic study failed its preregistered general-mechanism gates, so we do not claim a
universal causal law. A proposed disease-invariant repair was not trained because the
single-dataset cohorts lacked prespecified support for the required counterfactual pairs;
this is a feasibility limitation, not a negative result for the repair.

## 6. Conclusion

Correct pairing reproducibly improved participant-profile correspondence, especially sex,
but did not yield a robust cross-backbone improvement in covariate-balanced disease
prediction or consistently outperform raw audio. Successful audio-metadata correspondence
is therefore not sufficient evidence of a transferable disease representation.
Correspondence, retained information, and disease transfer must be evaluated separately.
