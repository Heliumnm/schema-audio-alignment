# Teacher check — pre-registration

**Frozen before any GPU run.** This document, the split manifest, the prompt texts, the
model revision and the seeds are committed together; the commit hash is the audit record
that the freeze preceded the first forward pass. Nothing below may change afterwards.

The whole audible-evidence route rests on one assumption: an audio LLM asked about a
recording can report a cue that a clinician also hears. If it cannot, the descriptions are
not evidence and no amount of contrastive training on them will make them so. This is the
gate.

## Task

**Primary: dry vs wet cough**, on COUGHVID's expert-annotated subset.

**Exploratory only: wheezing.** No gate, no conclusion. Positive counts are 55 / 83 / 2
across the three annotators — the third marks wheeze essentially never while the second
marks it 40× more often. Reporting that as a result would be indefensible.

## The label is not clean, and the design is built around that

| annotator | dry/wet rows | wet rate |
|---|---:|---:|
| expert 1 | 415 | 14.2% |
| expert 2 | 619 | 18.9% |
| expert 3 | 542 | **46.5%** |

**Annotator 3 calls a cough wet three times as often as annotator 1**, and the annotations
barely overlap: of the 1,417 recordings in the analysis set, **1,324 carry one annotator**,
27 carry two, 66 carry three.

That combination is dangerous in a specific way: a pooled score can be driven by the model
recognising *which batch of recordings a given annotator saw* rather than by hearing
anything. So:

> **The primary metric is the equal-weight macro-average of the three within-annotator
> AUROCs.** Pooled AUROC is supplementary and carries no gate.

Within an annotator the wet rate is fixed, so the annotator-identity route is closed by
construction rather than argued away.

Inter-annotator agreement cannot be estimated at 93 overlapping recordings. We therefore
claim **no label ceiling**, and a low AUROC is not evidence that the model cannot hear — it
may be the label. The gate is one-directional: passing licenses the next stage, failing
stops it, and neither outcome is a statement about ceilings.

### Analysis set

* annotator-level dry/wet rows with `cough_detected ≥ 0.8`: **1,576**
* unique recordings: **1,417**
* quality: good 935, ok 584, poor 56, no_cough 1
* COUGHVID is anonymous per submission with no participant identifier, so the unit is the
  **recording**. Stated as a limitation, not worked around.

## Scoring — one procedure, one generation path

Every prompt is scored the same way: a forced two-way choice read from the model's
probability over the two option tokens.

    A) dry      B) wet        ->    p = P(B) / (P(A) + P(B))

**The A/B answer is emitted first in every prompt.** Any description or justification comes
*after* the answer token. A prompt that describes first and answers second would condition
the answer on its own generated text, which is a different inference path and cannot be
compared with a first-token probability. This round asks one question only — can the model
hear the attribute — and **whether its generated descriptions are trustworthy is a
separate, later gate**, not this one.

No free-text parsing anywhere: parsing rewards prompts that happen to produce parseable
output, which is a prompt-style artefact this project has already retracted a result to.
Option order is counterbalanced across recordings and the two orders averaged, so a
positional preference cannot masquerade as hearing.

## Prompts

Three, differing only in the framing that precedes the question — all three emit the
choice first:

| id | framing |
|---|---|
| `plain` | the forced choice alone |
| `cue` | an itemised audible-cue checklist is requested, answer first, checklist after |
| `cue_evidence` | as `cue`, with a one-line justification per item, again after the answer |

No prompt may mention COVID, a diagnosis, or the words dry/wet outside the option list.
The verbatim texts are committed alongside this document.

**Multiplicity is handled by splitting, not by picking.** The analysis set is split 70/30
**grouped by recording** (so a recording's multiple annotator rows never straddle the
split) and **stratified by annotator and dry/wet**. The prompt is chosen on the 30%
development slice; **the gate is read once on the 70% reporting slice**. All three
development numbers are reported so the choice is auditable.

## Controls

**`audio_shuffled` is computed offline from the real scores — it costs no GPU time.**
Within strata of **annotator × quality × duration tercile**, the model's scores are
permuted across recordings while the labels stay put. The stratification deliberately does
*not* include dry/wet: permuting within the label class would destroy the very signal the
test is meant to detect and would make the null trivially true.

**1,000 fixed permutations** (seed committed) give the null distribution of the macro-AUROC,
and the test is `p < 0.01`. A single shuffle's confidence interval is not used — one draw
cannot characterise a null, and asking whether its CI covers 0.5 is a much weaker question
than asking where the observed statistic falls in a thousand draws.

`silence` (zero-signal input) is a **secondary** arm and does need GPU. It reveals the
model's prior. It supports no chance claim, because it changes the input distribution as
well as the correspondence.

## GO criteria — fixed now

Read once on the 70% reporting slice:

1. **macro-average of the three within-annotator AUROCs ≥ 0.65**, with 95% CI lower bound
   > **0.60** (recording-level bootstrap, 2,000 resamples);
2. **all three within-annotator point estimates > 0.5**;
3. **stratified permutation test `p < 0.01`** (1,000 permutations, as above);
4. **A/B format compliance ≥ 95%**, i.e. fewer than 5% of calls fail to place
   interpretable mass on the two option tokens.

Pooled AUROC, AUPRC and the `silence` arm are **auxiliary** and gate nothing.

Failing any of 1–4 stops the audible-evidence route. It does **not** license a fourth
prompt, a different cue, or a change of corpus to find a passing number.

## What this number may not be compared to

**Not to the 0.509 Qwen2-Audio scored on ICBHI wheeze.** Different corpus, different
acoustic event, different label definition, different annotation protocol, and decisively
a different annotator structure. That number is why this gate exists; it is not a baseline
this gate is measured against, and no sentence in the write-up may place the two side by
side as if commensurable.

## Model, cost, and what is committed before the run

Qwen2-Audio-7B-Instruct at the revision recorded in `results/teacher_run_config.json`,
already on the server. 1,417 recordings × 3 prompts × 2 option orders, plus the `silence`
arm — no training, one GPU, well under a day. `audio_shuffled` adds nothing to that cost.

Committed before the first forward pass, and referenced by hash in the results:

* this document;
* `results/teacher_split_manifest.csv` — the frozen 70/30 assignment, with uuid,
  annotator, label, quality, duration tercile and slice;
* `src/teacher_prompts.py` — the verbatim prompt texts;
* the model revision, the split seed, and the permutation seed.

## Order

pre-registration → commit → manifest frozen and committed → GPU. UKCOVID extraction runs
in parallel and gates nothing here.
