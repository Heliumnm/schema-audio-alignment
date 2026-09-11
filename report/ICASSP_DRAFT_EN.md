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
Correct pairing improved profile retrieval for AST-6L and OPERA-CT and retained more sex
information than Within-label (delta AUROC +0.191 and +0.112), confirming that exact pairing
changed the representation. Matched COVID prediction changed by only +0.006 and -0.002.
Across secondary analyses on CODA TB, Cambridge COVID-19 Sounds, and Coswara, retrieval and
sex decodability increased consistently, whereas 0/11 dataset-by-backbone settings
established a positive matched-disease gain, and Correct did not consistently outperform
frozen raw audio. Correspondence learning, retained participant information, and disease
transfer are therefore distinct empirical claims requiring distinct controls.

## 1. Introduction

Respiratory audio models increasingly combine cough, breathing, or lung sounds with patient
information such as age, sex, symptoms, smoking status, and medical history. General-purpose
representations including AST, OPERA, and HeAR provide strong audio encoders, while metadata
has also been used as supervision for respiratory contrastive learning. More recent
multimodal systems explicitly align each recording with clinical text from the same
participant. RespiraMFM, for example, freezes audio and text encoders and trains an
audio-side projector before downstream multimodal prediction.

This alignment creates an evaluation ambiguity. A contrastive objective rewards any audio
feature that helps recover the assigned metadata profile. Clinical metadata contains not
only disease-related information but also demographics, symptoms, medical history,
missingness patterns, and variables associated with cohort composition. Successful
alignment can therefore reflect participant-profile correspondence, label-conditioned
population association, or disease information that remains useful after population shift.
Alignment loss, profile retrieval, or source-cohort AUROC alone cannot distinguish them.

The missing control is the **pairing relation itself**. Comparing correctly paired metadata
only with globally shuffled metadata changes both exact correspondence and label-conditioned
statistics. We introduce a **Pairing-Controlled Transfer Audit** with three otherwise
identical arms: Correct preserves the observed pair, Within-label assigns metadata from
another participant with the same disease label, and Global removes systematic participant-
and label-level pairing. Correct-minus-Within asks what exact profile pairing adds beyond
label-conditioned association, whereas Within-minus-Global measures the latter. These
contrasts are descriptive, not causal estimands.

The audit asks three sequential questions: **Was pairing learned? What information was
retained? Does that information transfer?** Duplicate-aware profile retrieval is a
manipulation check; probes identify decodable channels; and participant-disjoint,
covariate-balanced disease evaluation tests transfer against Within-label and frozen raw
audio. Across four datasets, Correct repeatedly improves profile correspondence and, most
consistently, sex decodability. Its disease effect is small, directionally inconsistent, and
does not consistently exceed raw audio. Successful correspondence and transferable disease
representation are therefore separate empirical claims.

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

### 3.2 Representations and metadata

Frozen AST-6L and OPERA-CT are the primary UKCOVID audio backbones; HeAR is included in the
external backbone analyses. A frozen Phi-2 encoder represents a deterministic schema of
available demographics, smoking, respiratory history, and symptoms, with missingness
explicit.

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

`Correct - Within-label` measures the increment associated with exact profile pairing after
retaining label-conditioned metadata statistics in expectation. It is not a pure identity
effect or a causal estimand. `Within-label - Global` describes label-conditioned population
association, and `Correct - Raw` tests whether alignment improves on the frozen audio
representation. Global removes systematic participant- and label-conditioned pairing,
although accidental same-label or duplicate-profile matches can remain.

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

### 4.1 Alignment genuinely occurred

On UKCOVID validation, Correct improved macro-profile MRR over Within-label for both
backbones: +0.00318 [0.00161, 0.00492] for AST and +0.00323 [0.00155, 0.00500] for OPERA.
All five seed-level effects were positive. A null transfer result therefore cannot be
explained by a projector that learned nothing.

Across the 11 formal dataset-by-backbone settings, all retrieval point estimates were
positive and 10/11 confidence intervals excluded zero. We do not pool these effects because
profile libraries and evidence status differ across datasets.

### 4.2 Correct pairing preferentially retained participant information

On UKCOVID matched participants, Correct retained substantially more sex information than
Within-label: delta AUROC was +0.1912 [0.1669, 0.2151] for AST and +0.1125 [0.0943, 0.1311]
for OPERA. Age increased for AST but not OPERA; acquisition and cohort effects were less
consistent.

Sex `Correct - Within-label` intervals excluded zero in all 11 settings. Raw audio usually
had higher absolute demographic decodability than aligned representations, so alignment did
not create demographic information absent from audio. It selectively retained that channel
relative to Within-label.

### 4.3 Correspondence did not robustly improve disease transfer

On UKCOVID matched, disease `Correct - Within-label` delta AUROC was +0.0062
[-0.0144, 0.0279] for AST and -0.0020 [-0.0191, 0.0168] for OPERA. For AST, raw, Correct,
and Within AUROC were 0.538, 0.529, and 0.523; for OPERA they were 0.577, 0.555, and 0.557.
A fixed MLP did not reveal a hidden effect.

Across all 11 settings, matched disease point estimates were positive in five and negative
in six. No confidence interval established a positive gain. Correct also showed no
consistent advantage over frozen raw audio. This is a descriptive recurrence, not a
meta-analysis and not proof of exact equivalence.

### 4.4 Label association, fusion, and calibration

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
audio-metadata alignment repeatedly improved retrieval of participant profiles, proving
that genuine correspondence was learned. Yet the most reproducible additional channel was
participant sex, while disease improvements were small, directionally inconsistent, and
not robust to raw-audio controls or readout changes.

In plain terms, the model became better at matching a recording to the kind of participant
described by a metadata card, but not reliably better at diagnosing disease after measured
patient composition was balanced.

Within-label shuffling is essential. Correct versus Global alone mixes exact correspondence
with label-conditioned population association. Retrieval verifies the manipulation, probes
describe retained channels, and matched prediction tests clinical transfer. Future studies
should report all three together with a frozen raw reference and calibration under
population shift.

The sex result also requires careful wording. A probe shows that sex is decodable, not that
the disease classifier causally relies on it. Moreover, raw audio often contains more
demographic information than aligned representations. Correct therefore appears to preserve
this channel relative to a Within-label projector rather than create a new attribute.

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
