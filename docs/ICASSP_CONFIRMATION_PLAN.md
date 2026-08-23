# ICASSP confirmation plan

Date frozen: 2026-08-19

> **Outcome update (2026-08-23).** The Coswara branch failed its frozen covariate-balance
> data gate before any representation, classifier, or model score was produced. The current
> paper therefore has no external confirmation and remains a UKCOVID discovery audit. Final
> status: `docs/ICASSP_EXECUTION_STATUS_ZH.md`.

## Purpose

The UKCOVID metadata-alignment results are a discovery audit, not a confirmatory result.
The official Standard, matched and matched-long test sets informed earlier protocol
versions.  The ICASSP paper therefore requires a participant-level confirmation that is
specified before model scores are read.

The claim to be tested is deliberately narrow:

> In a strongly confounded respiratory-audio setting, correct audio--metadata pairing can
> preserve patient metadata correspondence without producing disease evidence that
> transfers to a covariate-balanced population.

This does not claim that metadata alignment is universally ineffective, that the frozen
AST contains no disease information, or that the full RespiraMFM system has been
reproduced.

## Retracted option: residual UKCOVID Standard-long cohort

The first candidate was the residual Standard-long cohort after excluding the official
matched-long test:

- audio-QC Standard-long: 30,815 participants;
- already evaluated official matched-long: 4,196 participants;
- residual pool: 26,619 participants (18,404 negative, 8,215 positive);
- overlap between residual and official matched-long: zero.

This option was rejected **before reading any residual-cohort model score**.

The official matched-long construction stratifies by recruitment source, ten-year age
band, sex, cough, sore throat, asthma, shortness of breath, runny/blocked nose and the
presence of at least one symptom, then samples equal positive and negative counts within
stratum.  The open metadata exposes only three broad age groups for residual participants;
the ten-year age band appears only in the published stratum string for the 4,196 selected
participants.  The official strata therefore cannot be reconstructed for the residual
pool.

Using the available three-level age variable while keeping the remaining fields fixed
would yield at most 42 positive--negative pairs (84 participants).  That is not an
adequately powered confirmation set.  No residual manifest was created and no residual
metric was computed.  Existing pipelines did generate full-cohort probabilities, so the
pool must also not be described as never predicted; the accurate statement is that no
recorded label--prediction joint evaluation was performed.

## Required external confirmation

The next dataset must satisfy all of the following before training begins:

1. a stable participant identifier and participant-level split;
2. a disease label with documented provenance;
3. one prespecified respiratory recording type per participant;
4. metadata sufficient to construct the same semantic field families and a
   covariate-balanced evaluation;
5. enough positive and negative participants after matching for paired uncertainty
   estimates;
6. no sample-level leakage through repeated recordings, augmented files or filtered
   copies;
7. all matching, split, pairing, metrics and decision rules frozen before model scores are
   read.

Candidate datasets are audited using metadata and filesystem state only.  No external
model training or outcome evaluation may start until one candidate passes these gates and
its own preregistration document and participant manifest are committed.

## Candidate audit, 2026-08-19

### Coswara: conditional GO for download and cohort audit only

Coswara is the only current candidate that passes enough gates to justify further work:

- it assigns a stable participant ID and contains multiple recordings under that ID;
- it provides cough recordings, participant-reported COVID status, test-status fields,
  age, sex, smoking, symptoms and respiratory comorbidities;
- one recording type can be fixed in advance (`cough-heavy`) and all splits can be made at
  participant level;
- its metadata substantially overlaps the UKCOVID schema.

The dataset is not yet approved for training.  The changing GitHub `master` contains
2,746 IDs and is not the frozen paper release.  The selected source is Zenodo v1.0:

- record: `7188627`;
- file: `iiscleap/Coswara-Data-dataset-paper-publication.zip`;
- size: 12,984,309,908 bytes;
- MD5: `53721d9c106f99872bf7f878c8196d31`.

The archive root is pinned to upstream commit
`bf300ae9dc47918be4a30de90436fe7563fafb45`.  Its metadata CSV SHA-256 is
`e462c503bee3408214195855975b0eda08dd1188c0d521b494d93d388d60a72d`.
The Zenodo record labels the release CC BY 4.0, while the README inside the pinned archive
states CC BY-NC-ND 4.0.  This licence conflict is recorded rather than silently resolved;
the paper may cite and analyse the dataset, but no audio or derived data are redistributed
until the data owner clarifies the applicable terms.

The fixed release must first be downloaded, checksum-verified and audited for repeated
participants/sessions, exact and near-duplicate audio, decodability, duration, signal
quality and the availability of `cough-heavy`.  Recovered and under-validation statuses
are excluded.  Returning users and the meaning of missing symptom values must be resolved
from the frozen archive rather than inferred from the live summary CSV.

The outer ZIP contains 43 date directories and 153 split gzip-tar parts.  The nine audio
types are interleaved inside those streams, so `cough-heavy` cannot be downloaded alone.
After the full ZIP passes its byte-count, MD5 and ZIP integrity checks, each date stream is
reassembled in order and only `cough-heavy.wav` is written.  Other audio types and
intermediate tar parts are not materialised.

The pinned metadata CSV contains 2,746 unique IDs.  The paper's reported 2,635-person
cohort is reproduced exactly by removing 81 `under_validation` rows and 30 participants
outside its stated age range of 15--90 years.  The remaining counts are 1,819 non-COVID,
674 positive and 142 recovered.  This resolves the apparent version discrepancy: it is an
eligibility rule, not a different metadata release.

Manual `cough-heavy` quality annotation is incomplete and calendar-structured.  The
pinned file covers 2,233 IDs (1,927 excellent, 167 moderate, 139 poor) and stops in
September 2021; 427 members of the paper cohort have no annotation.  Manual quality is
therefore not an entry criterion for the primary cohort: missing, 0, 1 and 2 remain
visible while every file receives the same objective decode/duration/signal QC.
Restricting to manual quality 1--2 is a predeclared, separately matched sensitivity
analysis, not a way to select the primary cohort after outcomes are seen.

Label provenance is a limitation: the paper reports that most positives had clinical
testing, while many non-COVID participants lack an explicit negative test.  The external
document must therefore distinguish the broad paper-defined train/validation label from a
test-status-confirmed confirmation subset.  Location and calendar wave are acquisition
confounders and are used for balancing/audit, not placed in the alignment text.

The pinned metadata and annotations were used only for capacity planning.  Before waveform
QC, the paper-age-range, non-returning, test-status-concordant subset has 788 participants
(570 positive, 218 negative).  Exact matching on age decade, sex, cough, fever, fatigue,
sore throat, breathing difficulty, asthma and other respiratory disease leaves 114 pairs
before audio QC.  Exact matching of location and calendar quarter reduces the count below
100, so their balancing rule and
acceptable post-match imbalance were frozen before objective waveform QC.  With official
checkbox semantics and the two respiratory-comorbidity fields combined, the metadata-only
ceiling is 114 exact pairs and the deterministic balance path first passes at 101 pairs.
These are capacity estimates, not a selected test manifest.  If the final audio-QC cohort
cannot supply at least 100 prespecified matched pairs, Coswara is demoted to a descriptive
external sensitivity analysis.  Even at 100--130 pairs, its power is sufficient only for
large effects, so the frozen terminology is *external patient-level stress test*, not
confirmatory proof.

No Coswara model score has been computed and no projector training has started.

### COUGHVID: rejected as the primary confirmation

COUGHVID has adequate recording counts and some overlapping metadata, but the released
UUID identifies a recording rather than a participant.  The data therefore cannot exclude
repeat submissions by the same person across splits, cannot support participant-level
bootstrap uncertainty and cannot make `correct` versus `within-label` a clean
person-to-person comparison.  Its COVID status is also participant-reported rather than a
verified test result.  It may be used only as an explicitly recording-level exploratory
stress test; no COUGHVID disease model is started for the ICASSP confirmation.

### Other local candidates: rejected for the primary confirmation

- ICBHI is small, uses a different disease endpoint and has already informed this project;
  its released recording split also places patients 156 and 218 on both sides, so it is not
  used as a patient-independent confirmation.
- KAUH contains only 112 participants, with three filtered versions of the same recording
  per participant and too few cases per disease after matching.
- TBscreen and CodaTB are not locally available and require access steps whose timing is
  incompatible with the current submission window.

## Frozen comparison family for the eventual external dataset

The minimum comparison is:

- recording artefacts only;
- frozen raw audio representation;
- metadata only;
- metadata plus raw audio by direct concatenation;
- correct audio--metadata alignment;
- within-label metadata shuffle;
- global metadata shuffle;
- metadata plus each aligned representation by the same direct-concatenation head.

The architecture, optimiser and downstream selection protocol must be identical between
the three alignment arms.  Direct concatenation is used instead of introducing a new
late-fusion weighting scheme.

The primary alignment comparison remains `correct - within_label` on calibrated negative
log-loss.  Paired AUROC, Brier score and calibration slope are secondary.  A result whose
interval includes zero is described as uncertain or as no detectable gain; it is not
evidence that the representation contains no disease information.  Discovery and
confirmation estimates are reported separately and are never pooled.

## One-shot rule

External training heads may use only the frozen external train and validation partitions.
The confirmation manifest is not accepted as an input to model-selection code.  The final
evaluator must verify the committed participant and file hashes and require an explicit
unlock flag.  After the confirmation metrics are read, matching fields, exclusions,
primary outcomes and arm definitions cannot change.
