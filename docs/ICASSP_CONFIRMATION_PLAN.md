# ICASSP confirmation plan

Date frozen: 2026-08-19

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
