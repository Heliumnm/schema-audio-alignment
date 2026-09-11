# Pairing-Controlled Auditing of Clinical Audio-Metadata Alignment: Separating Participant Correspondence from Disease Transfer

> This is the editable English narrative source. The submission-formatted version is
> [`../paper/icassp2027/main.tex`](../paper/icassp2027/main.tex). The Chinese evidence
> ledger remains [`ICASSP_DRAFT_ZH.md`](ICASSP_DRAFT_ZH.md).

## Abstract

Clinical audio models increasingly align recordings with patient metadata, implicitly
treating successful correspondence as improved disease representation. Yet metadata mixes
symptoms with demographic and cohort information. We introduce a Pairing-Controlled
Transfer Audit comparing correct pairs, metadata shuffled within disease labels, and
globally shuffled metadata. This separates exact participant-profile correspondence from
label-conditioned association. In UKCOVID, correct pairing improved profile retrieval for
AST-6L and OPERA-CT and strongly retained sex information (matched probe delta AUROC +0.191
and +0.112), while COVID gains were small and uncertain (+0.006 and -0.002). Across
UKCOVID and secondary analyses on CODA TB, Cambridge COVID-19 Sounds, and Coswara, 10/11
retrieval and 11/11 sex-probe intervals excluded zero, but 0/11 matched disease intervals
did. Correct alignment also failed to consistently outperform raw audio. Clinical
audio-metadata studies should therefore evaluate correspondence, retained information, and
disease transfer separately.

## 1. Introduction

Respiratory audio models increasingly combine cough, breathing, or lung sounds with age,
sex, symptoms, smoking status, and medical history. Recent systems go beyond prediction-time
fusion and explicitly align each recording with clinical text derived from the same
participant. RespiraMFM, for example, freezes audio and text encoders and learns an
audio-side projector before downstream multimodal prediction.

The premise is appealing: better audio-metadata correspondence should produce a more
clinically useful representation. It is not guaranteed. Clinical metadata mixes
disease-associated symptoms with demographic background, medical history, missingness, and
variables correlated with recruitment. A contrastive loss cannot distinguish portable
disease evidence from these alternatives; it rewards any feature that identifies the
assigned positive pair. Improved alignment loss, retrieval, or source-cohort AUROC may
therefore reflect three different abilities:

1. exact participant-profile correspondence;
2. label-conditioned population association; or
3. transferable disease evidence.

We introduce a **Pairing-Controlled Transfer Audit** to separate these claims. Three
otherwise identical projectors receive correct metadata, metadata from another participant
with the same disease label, or globally shuffled metadata. Duplicate-aware retrieval first
checks whether exact pairing was learned; probes then identify retained information; and
covariate-balanced targets test whether that information improves disease prediction. Raw
audio, nonlinear readouts, and calibration transport address alternative explanations.

Our contributions are:

- a pairing intervention that separates exact profile correspondence from
  label-conditioned association;
- an audit sequence linking retrieval, information probes, raw references, and matched
  transfer; and
- a multi-dataset, multi-backbone study showing reproducible profile correspondence--most
  consistently for sex--without a robust gain in covariate-balanced disease prediction.

Our claim is not that metadata alignment is universally ineffective. Correspondence
learning and disease transfer are distinct empirical claims.

## 2. Related Work

AST, OPERA, and HeAR learn general-purpose representations from spectrograms or
health-related audio. These methods ask whether respiratory recordings contain useful
health information, but not what additional information is retained when audio is aligned
with participant metadata.

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
  custom strict-COVID reconstruction of its Task-2 cough subset; and
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
batches, and dropout streams; only the pairing changes. Matched data select no epoch,
projector, readout, or hyperparameter.

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
representation.

### 3.4 Audit endpoints

- **Correspondence:** audio queries retrieve metadata profiles on source validation. We use
  macro-profile MRR so repeated schemas do not dominate.
- **Information:** identical regularized probes decode sex, age, symptoms, acquisition and
  cohort variables, and disease. Decodability establishes availability, not causal use.
- **Transfer:** source validation selects and calibrates downstream readouts; paired AUROC,
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
