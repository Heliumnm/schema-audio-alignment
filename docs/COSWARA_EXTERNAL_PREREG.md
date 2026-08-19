# Coswara external patient-level stress test: frozen preregistration

Date frozen: 2026-08-19
Dataset: Coswara Zenodo v1.0, record `7188627`
Archive root: upstream commit `bf300ae9dc47918be4a30de90436fe7563fafb45`

## Status and purpose

Coswara is not allowed to rescue or overwrite the UKCOVID discovery result.  Its purpose
is a separately reported, participant-level external stress test of one frozen question:

> Does correct audio--metadata pairing improve calibrated disease transfer relative to
> the same alignment architecture trained with metadata shuffled within disease label?

The expected matched sample is small.  Even if the data gate passes, this analysis is
called an **external patient-level stress test**, not proof of external confirmation or
equivalence.  UKCOVID and Coswara estimates are never pooled.

No model prediction may be read until the archive, audio-QC manifest, participant split,
matched pairs, metadata template, pairings, model hashes and evaluation code are frozen.

## Frozen source and modality

- archive bytes: `12,984,309,908`;
- archive MD5: `53721d9c106f99872bf7f878c8196d31`;
- metadata SHA-256: `e462c503bee3408214195855975b0eda08dd1188c0d521b494d93d388d60a72d`;
- manual cough-quality SHA-256:
  `ab41f10875796818f44022c3f067bc4b3724bc48100bff447edc95354d98a9bb`;
- one recording per participant: `cough-heavy.wav`;
- `cough-shallow` is not substituted when the primary recording is absent.  It may be
  added only as a clearly labelled, separately frozen sensitivity analysis.

The Zenodo page states CC BY 4.0 while the pinned README states CC BY-NC-ND 4.0.  The
analysis may cite the source, but the project does not redistribute audio or derived data
until the owner clarifies the applicable licence.

## Eligibility and labels

All cohorts first require:

- age from 15 through 90 inclusive;
- `rU == n` (known non-returning participant);
- status not `under_validation` or `recovered_full`;
- exactly one participant ID and one usable `cough-heavy.wav` after objective QC.

Broad development labels are:

- positive: `positive_mild`, `positive_moderate`, `positive_asymp`;
- negative: `healthy`, `no_resp_illness_exposed`,
  `resp_illness_not_identified`.

The confirmation candidate is stricter:

- a positive broad label and `test_status == p`; or
- a negative broad label and `test_status == n`.

Known `testType` in `{rtpcr, rat}` is a predeclared sensitivity cohort.  It is not used to
replace the primary cohort if the primary result is inconvenient.

## Objective audio QC and duplicates

Every extracted recording is audited.  Primary inclusion requires:

- regular file, non-empty and larger than a WAV header;
- complete PCM decode with finite samples and matching header/decoded frame counts;
- duration at least 0.5 s;
- waveform not identically zero;
- stable raw-file and canonical decoded-PCM SHA-256.

Manual `QUALITY` does **not** determine primary inclusion.  Its annotation stops in
September 2021 and its missingness is calendar- and label-structured.  Missing remains a
separate category; it is not mapped to 0, 1, 2 or `NO`.

- `QUALITY in {1,2}` is a separately matched sensitivity analysis;
- `QUALITY == 2` is descriptive only;
- known `QUALITY == 0` remains visible in the primary cohort and in balance diagnostics.

Cross-ID identical decoded PCM is a leakage group.  Within each such group, only the ID
with the smallest SHA-256 of the literal string `coswara-dedup-v1|PARTICIPANT_ID` is kept;
all other IDs are excluded before any split.  The rule uses neither label nor model score.
Raw duplicates and repeated participant IDs are reported separately.

## Confirmation set is selected first

The matched confirmation set is constructed from the strict test-status candidate after
audio QC and duplicate resolution.  Matching is 1:1 without replacement.

The following variables are exact within every pair:

- ten-year age band;
- sex;
- cough;
- fever;
- fatigue;
- sore throat;
- breathing difficulty;
- asthma;
- other respiratory disease.

Unchecked official multi-select checkboxes are `NO`.  This includes symptoms, asthma, and
the three boxes combined into `other respiratory disease` (`others_resp`, chronic lung
disease, or pneumonia).  Blank values in smoking, vaccination, mask use, location and
quality are explicit missing categories, not `NO`.

Within each exact stratum, a deterministic minimum-cost bipartite assignment pairs every
possible minority-class ID to a deterministic subset of the majority class.  Pairing cost
was frozen before any model output or prediction was read:

- absolute continuous-age difference divided by 10: weight 1;
- mismatch in smoker, diarrhoea and loss of smell: weight 1 each;
- mismatch in country, collapsed province and recording half-year: weight 2 each;
- mismatch in vaccination, mask use and manual-quality category: weight 1 each;
- final ties: SHA-256 of `coswara-match-v1|NEGATIVE_ID|POSITIVE_ID`.

First, maximum-cardinality matching is solved under those exact fields.  If its balance
fails and more than the applicable minimum number of pairs remains, one pair at a time is
removed.  The removed pair is the deterministic choice that minimises

`max(|age SMD|/0.10, max categorical SMD/0.10, max level difference/0.05)`;

ties are resolved by pair ID.  Trimming stops at the **first** passing set encountered on
this deterministic greedy path, or at the power floor.  It does not search alternative
paths for a more favourable cohort and cannot continue below the floor to manufacture a
cosmetically balanced result.

Province is collapsed to `Tamil Nadu`, `Karnataka`, `other India`, or `non-India`.
Recording wave is the calendar half-year derived from `record_date`.

After matching, all exact fields must be perfectly balanced; every numeric/binary audit
field must have absolute standardised mean difference at most 0.12; each level of country,
collapsed province, recording half-year, vaccination, mask use and quality availability
must differ in proportion by at most 0.08.  Country levels below ten eligible candidates
are grouped as `OTHER` before matching and auditing.  These limits were frozen during a
metadata-only capacity test, before waveform QC or any model prediction was available.
With all metadata rows provisionally treated as audio-valid, maximum exact matching gives
114 pairs and deterministic trimming first passes at 101 pairs; this is only a feasibility
ceiling, not the final manifest or a result.

The archive date directory is an upload-batch identifier and need not equal the
participant-reported `record_date`; disagreement is recorded as a diagnostic, not used as
an exclusion.  Unambiguous linkage instead requires one stable participant ID, one pinned
archive path and matching raw/decoded-PCM hashes.

Participant, file, PCM, label, stratum, pair ID, metadata and manifest hashes are frozen.
Confirmation IDs cannot enter any training, validation or source-test partition.
The independently matched manual-quality and known-test-type sensitivity manifests are
also frozen at this stage; the union of every primary or sensitivity evaluation ID is
removed from development, even when an ID is absent from the primary matched set.

## Development split

After confirmation IDs are removed, all remaining broad-label eligible participants are
split deterministically 70%/15%/15% into train, validation and source-distribution test.
Splitting is participant-level and stratified by `label × sex × recording half-year`;
within stratum, IDs are ordered by SHA-256 of `coswara-split-v1|PARTICIPANT_ID`.

- train fits the three projectors and downstream heads;
- validation selects downstream regularisation and fits calibration;
- source test is read only after all choices are frozen;
- matched confirmation requires an explicit one-shot unlock.

An explicitly tested participant not selected into the matched confirmation may enter the
development pool: selection used only frozen metadata/audio QC and no model prediction.

## Data and power gates

The primary stress test proceeds only if all conditions hold after audio QC:

1. at least 100 matched pairs (200 participants);
2. all exact and fine-balance limits above pass;
3. train at least 900 participants and at least 250 per class;
4. validation and source test each at least 200 participants and at least 50 per class;
5. no participant or PCM duplicate crosses any split;
6. fixed IDs map unambiguously between metadata and the Zenodo audio.

There is no modality fallback: missing or failed `cough-heavy` files remain excluded.  Their
impact is decided only by the prespecified pair-count, balance and development-size gates;
the recording type cannot be changed to recover sample size.

The manual-quality sensitivity requires at least 60 independently matched pairs **and**
the same balance gate; the known-test-type sensitivity has the same requirements.  Below
those limits, or when balance fails, it is descriptive only.  The metadata-only ceiling
predicts that neither sensitivity analysis will be powered (57 quality pairs; 62 known-
test-type pairs whose balance still fails after trimming to 60).

If the primary set has fewer than 100 pairs or fails balance, Coswara is a **NO-GO for
formal external confirmation**.  It may still be reported as a small external stress test,
but the manuscript must not claim cross-dataset confirmation.

## Frozen experimental family

- recording artefacts only;
- frozen raw AST-6L representation;
- metadata only;
- metadata + raw AST by direct concatenation;
- correct metadata alignment;
- within-label metadata shuffle;
- global metadata shuffle;
- metadata + each aligned representation by the same direct-concatenation head.

The alignment architecture and optimiser stay identical to the frozen UKCOVID audit.  The
Coswara text template uses only the same semantic families available in Coswara; country,
province, calendar, test details and disease status are never alignment-text fields.

## Outcomes and one-shot interpretation

Primary comparison: `correct - within_label` paired difference in calibrated negative
log-loss.  Secondary outcomes are paired AUROC, Brier score and calibration slope.  The
fusion counterpart (`metadata + correct - metadata + within_label`) and direct raw-audio
increment (`metadata + raw - metadata only`) are prespecified diagnostics.

The primary 95% interval uses a hierarchical bootstrap over exact matching strata and
model seed.  A sampled stratum contributes all of its pairs, so members of a pair are never
split.  Pair-level bootstrap over matched pairs and seed is a prespecified sensitivity
analysis.  Concretely, each of 10,000 replicates draws the observed number `S` of exact
strata with replacement and all pairs belonging to each selected stratum, then draws five
seed IDs with replacement from the five fitted seeds.  A seed ID remains paired across
`correct` and `within_label` (and across their fusion counterparts).  The replicate is the
mean paired loss difference across all selected participants and seeds, including the
multiplicity induced by resampling.  The point estimate uses every observed stratum and all
five seeds once; the interval is the 2.5th and 97.5th percentiles with bootstrap RNG seed
`20260819`.  Pair-level sensitivity replaces only the stratum draw with `P` matched-pair
draws from the observed `P` pairs.  Five seeds do not increase the participant sample size.

Interpretation is frozen:

1. negative difference with CI excluding zero: correct pairing is worse in this Coswara
   cohort, not universally harmful;
2. UKCOVID-consistent direction with CI including zero: directionally consistent but
   externally imprecise;
3. opposite direction with CI including zero: externally inconsistent and imprecise;
4. positive difference with CI excluding zero: the UKCOVID negative pattern did not
   reproduce, and the paper must say so.

No field, matching rule, exclusion, modality, primary outcome or arm may change after the
confirmation evaluator is unlocked.
