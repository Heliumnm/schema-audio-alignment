# Contrastive audio–text alignment for respiratory sound

> 中文完整故事版：[docs/项目完整故事_中文.md](docs/项目完整故事_中文.md)

## Current ICASSP audit extension (final Route-A run, August 2026)

The Route-A audit is complete. It asks whether correct audio--metadata pairing learns
portable disease evidence or participant/cohort correspondence. The controlled arms are
correct pairing, within-label shuffling (label-level association preserved, individual
pairing broken), and global shuffling.

The manipulation worked: correct pairing improves unique-profile retrieval over
within-label shuffling for both AST-6L and OPERA-CT, with all five seed effects positive.
Information-channel probes show strong retained participant correspondence, especially
sex, but no stable correct-minus-within COVID increment. Under covariate-matched evaluation,
disease ranking gains are not robust across backbones or linear/nonlinear readouts, and
correct alignment does not beat the frozen raw-audio representation. Probability-transport
analysis further shows that the adverse source-calibrated NLL is mainly unsupported
confidence: target-domain recalibration removes the NLL gap without creating a disease
ranking gain.

The controlled synthetic sweep learned correspondence but failed its preregistered
mechanism gates, so it is not used as causal evidence. The Coswara external branch stopped
at its frozen data-balance gate before any model score. UKCOVID therefore remains a
single-dataset discovery audit; the results do not establish that metadata alignment is
universally harmful or ineffective.

Start with:

- [Chinese final execution status](docs/ICASSP_EXECUTION_STATUS_ZH.md)
- [Chinese ICASSP paper blueprint](docs/ICASSP_PAPER_BLUEPRINT_ZH.md)
- [English ICASSP draft](docs/ICASSP_DRAFT_EN.md)
- [Frozen strict follow-up design](docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md)
- [Auditable final Route-A JSON results and hashes](results/route_a_final/README.md)
- [Future disease-invariant positive-pair design (not run)](docs/DISEASE_INVARIANT_ALIGNMENT_FUTURE_PLAN_ZH.md)

## Historical ICBHI/schema track

**A negative result, thoroughly controlled — plus one finding that outlived it.**

The project set out to test whether structured clinical schema text is a better
alignment target than free-form LLM narration for respiratory audio, CLIP-style. It
is not, and neither is anything else tried: **no form of contrastive audio–text
alignment beat simply probing the frozen audio features.** Ten hypotheses were falsified with controls across two phases, and an eleventh —
patch-level grounding — produced a weak effect that fell short of its pre-registered
development gate. Along the way a separate result held up: **a general-purpose
AudioSet encoder substantially outperforms every respiratory-specific foundation
model tested.**

Full detail in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); this is the summary.

**Phase 2 is complete.** Its reframing — *can an explicit structured acoustic
representation learn a field → spectrogram-patch correspondence*, with the payoff in
grounding rather than classification — reached a pre-registered verdict on synthetic
events: a weak, well-controlled effect that did not clear its development gate. Phase 2
also corrects two Phase-1 overstatements. See
[`docs/PHASE2_MULTIPOSITIVE.md`](docs/PHASE2_MULTIPOSITIVE.md).

## What was tested and what happened

| # | Hypothesis | Verdict |
|---|---|---|
| 1 | Structured schema text > free-form narration | ❌ narration wins the probe, both splits, both targets |
| 2 | Text *resolution* (distinct strings) drives alignment quality | ❌ falsified twice — 27 strings ≈ 3,552 strings |
| 3 | Frozen towers were the bottleneck | ⚠️ **partly true** — training the encoder gains +0.10–0.20, still loses to no alignment |
| 4 | Alignment buys a label-free (zero-shot) inference path | ❌ true zero-shot peaks at 0.614, and 0.425 on wheeze |
| 5 | Schema helps *promptability* | ❌ prompt-style artefact; rewording flips the ranking |
| 6 | Too few in-batch negatives (23 vs CLIP's 32,767) | ❌ more negatives made it **worse**, saturating by 1,024 |
| 7 | Global mean-pooling destroys crackle transients; local alignment would fix it | ❌ premise false — global mean is the *best* readout, headroom 0.000 |
| 8 | False negatives explain it (multi-positive loss) | ❌ no effect on either target, across 3 loss variants |
| 9 | Serialisation format matters at matched content | ❌ sign flips once the duration confound is removed |
| 10 | A typed set encoder beats strings at matched content | ❌ a **constant** schema scores higher (0.750 vs 0.705, 5/5 seeds); alignment damage is monotone in how wrong the correspondence is |
| 11 | A query→patch objective learns a field↔region correspondence | ⚠️ **weak but real** — 0.569 2AFC [0.538, 0.605], significantly above chance, below the 0.60 development gate; an oracle-location crop classifier reads the same correspondence at 1.000 |

**21 alignment runs across 2 splits, 2 audio towers, 2 text towers, 5 text sources,
2 targets. Not one beat the frozen-feature baseline.** The last direction with a
mechanism behind it — patch-level alignment — was closed by measuring its premise
rather than building it: time-resolved readouts are no better than the global mean,
so there was no headroom to compete for.

## Why it fails

Two measurements explain the whole pattern.

**The text carries no label information.** Qwen2-Audio asserts a wheeze at 0.344 when
one is present and 0.327 when absent — AUROC 0.509. Yet its narration still wins the
probe. A target with no clinical content cannot be producing a clinical gain: what
the encoder gets from alignment is a *diverse per-sample target*, i.e. functionally
self-supervised training that happens to be text-shaped.

**The encoders cannot hear the pathology.** StethoLM's audio encoder scores 0.512 on
ICBHI wheeze — chance. It inherits that from OPERA-CE (0.531), so the weakness
predates any domain adaptation. Descriptions cannot be grounded in something the
encoder never resolved.

## The finding that survived

On identical data, splits and probes, a general AudioSet model beats every
respiratory-specific encoder:

| encoder | params | ICBHI wheeze | ICBHI crackle |
|---|---:|---:|---:|
| **AST (first 6 layers)** | **43.7 M** | **0.807** | **0.750** |
| AST (full) | 86.2 M | 0.796 | 0.729 |
| OPERA-CT | 32.4 M | 0.606 | 0.621 |
| OPERA-CE | 5.0 M | 0.531 | 0.594 |
| StethoLM | 5.0 M | 0.512 | 0.562 |

Three controls, all passed: bandwidth explains +0.037 of a 0.265 gap; halving AST
*improves* it, so scale is not the mechanism; and it replicates on KAUH's
patient-level disease tasks (3 of 4, with COPD the one counterexample).

**Use AST's first 6 layers** — better than the full model on most tasks at half the
compute, and far above the domain-specific alternatives.

*Not claimed:* that OPERA is wrong in general. Four encoders, two corpora; HeAR, CLAP
and AudioMAE were never tested, and OPERA reports 19 tasks.

## Grounding: a claimed positive, retracted

A synthetic grounding pilot was reported as passing — 0.935 Hit@±1 with correspondence
controls at chance. **It was wrong.** The query encoded `[onset + dur/2, dur, f0]`
while the target patch is a deterministic function of exactly those quantities, so the
task was coordinate arithmetic. A **no-audio coordinate-only baseline scores 1.000**,
above the learned model's 0.935.

An audio-content ablation drops the learned scorer to 0.326, so it does use the audio
— but when a no-audio baseline is perfect, no positive number on that task can be
interpreted. The pass is withdrawn.

The first redesign (v2) reduced the query to a single bit — which character to locate —
which does make coordinate leakage impossible. It leaked through the target mask
instead: only one arm's frequency ratios were normalised, so the two characters differed
in spectral centre, span and patch count, and **a probe on crop geometry alone with no
audio reached 0.982**. That gate reported 1.000; it was measuring where the crop was.
**Withdrawn.**

v2.1 equalises geometry by construction rather than checking for it afterwards — three
partials in both arms, endpoints pinned at `{1, 4}×fc`, mask spanning the whole band,
and a counterbalanced ±0.368-octave displacement of the middle partial that matches the
log spectral centroid to 0.0002 octave. Its gate passes properly: **geometry-only 0.477
[0.412, 0.538], audio 0.999**. The cost is that it is no longer a mono/poly wheeze
proxy — component count was the thing that had to be equalised — so it tests harmonic
vs inharmonic partial structure and nothing more.

## Grounding: weak, and stopped at the development gate

v2.1's gate passed properly, so the grounding experiment ran once, criteria fixed in
advance, official TRAIN split only, with every shortcut control the design admits.

| arm | 2AFC | 95% CI | paired Δ vs model |
|---|---:|---|---:|
| **oracle-location crop classifier** | **1.000** | [1.000, 1.000] | −0.431 [−0.462, −0.396] |
| **query→patch bilinear head** | **0.569** | [0.538, 0.605] | — |
| query-conditioned DSP | 0.549 | [0.481, 0.607] | +0.021 [−0.043, +0.088] |
| 8 shortcut controls | 0.493–0.521 | at chance | +0.049 to +0.077 |

**The effect is weak but real.** +0.069 above chance [+0.038, +0.105], with paired
deltas separating it from `query_shuffled`, `query_constant`, `target_permuted`,
`position_only` and `audio_shuffled`. The head does condition on the query and does use
the audio.

**It did not clear the 0.60 development gate**, so the official test set was not touched
and no real annotation was commissioned. Both were gated on this number; both remain
unspent.

**The distinction is entirely present in the patches.** The *oracle-location crop
classifier* — the gate's own linear classifier, handed both ground-truth event locations
and asked only which crop matches the query — scores 1.000. It is not a competing
method; it isolates how much correspondence survives once localisation is free. Detection
and selection come apart the same way: `hit@argmax` reaches 0.546 against ~0.033 chance,
while choosing *which* event the query names stays near chance.

**One rank-1 scoring architecture was tested.** Region-aware and cross-attention heads
are untested and **not ruled out**; the pre-registration allowed a single attempt, so a
second belongs to separately pre-registered work.

> AST patches encode the synthetic acoustic distinction, but a simple query-to-patch
> bilinear head extracts only weak query-conditioned localisation — enough to exceed
> chance, not enough to justify held-out testing or real annotation.

## Where this stands

The next hypothesis is pre-registered in
[`docs/H12_DENSE_SCHEMA_CLIP.md`](docs/H12_DENSE_SCHEMA_CLIP.md): keep the 2-D patch grid,
split the schema into field-level text queries, and learn field→region correspondence
with a local contrastive loss. Nothing there has been run.

Global alignment is finished: every hypothesis about it has been run with controls and
rejected. Grounding stopped at its development gate — a weak, controlled positive that
does not justify spending the held-out test set or annotator time.

**Eleven hypotheses across two phases: ten rejected, one stopped at its gate.** Phase 1
asked whether any text source or training setup makes contrastive alignment work; Phase 2
asked whether the loss or the representation was at fault. The answer in both cases is
neither — nothing beats probing the frozen features, and the schema representation shows
no evidence of carrying usable structure. The eleventh, patch-level grounding, produced a
weak but genuine effect that fell short of the bar set for it in advance.

**Grounding has now been tested and stopped short of its gate.** The reframing that
opened Phase 2 (structure may pay off in localisation rather than AUROC) reached a
pre-registered verdict on synthetic data, and it also lacked its premise and its data for
anything beyond that:

- its premise is gone: ten falsified hypotheses leave no evidence the schema
  representation carries structure the model can use, and the eleventh reached only a
  weak effect on synthetic events;
- the annotation it needs does not exist. ICBHI has per-cycle presence flags and no
  event timing, so grounding can only be measured against a synthetic set (mechanism
  only, never a clinical result) or against manual annotation that has not been
  collected;
- AST's patches have a ~160 ms receptive field, so crackles (5–20 ms) support only
  coarse Hit@±1 pointing.

**What the repository is worth as it stands:** a complete, controlled negative result
with a diagnosis for each failure, plus one positive finding that does not depend on
contrastive learning working at all — AST's first six layers outperform every
respiratory-specific encoder tested, with bandwidth, scale and cross-corpus controls.

It will not improve with more runs. It would get worse by adding an uncontrolled
"improvement" — Phase 2 Day 2 produced exactly one such result, and two controls
costing under an hour reversed it.

### The methodological record

Several conclusions here were stated, then corrected by a later measurement. That
trail is kept deliberately, in commit order:

| stated | corrected by |
|---|---|
| structured schema outperforms narration | narration wins the probe; its text carries no label information (AUROC 0.509) |
| alignment buys a label-free inference path | prototype scores use train labels; true zero-shot reaches 0.425 on wheeze |
| schema helps promptability | rewording the prompts in each arm's own idiom flips the ranking |
| local alignment is closed by the pooling diagnostic | that closed "raises AUROC", not "supports localisation" |
| the split is ICBHI's official 60/40 | the upstream docstring was wrong; the two agree on 53.5% of recordings |
| typed schema clears the pre-registered bar | reverses on crackle; ~¾ of the gain survives shuffling the schema |
| the grounding pilot passes at 0.935 | the query contained the answer; a no-audio coordinate baseline scores 1.000 |
| the v2 feasibility gate passes at 1.000 | only one arm's ratios were normalised; crop geometry alone scores 0.982 |

Eight retractions across eleven hypotheses. Each came from a control that was cheaper than
the work it prevented.

## Protocol

Every trap that produced a wrong answer here, encoded as a rule:

- **Patient-level splits, asserted in code.** Never segment-level.
- **The official ICBHI partition is not `patient_id ≤ 160`.** The upstream pipeline
  claimed it was; the two agree on 53.5% of recordings. `src/official_split.py`
  builds the real one, and results are reported under both.
- **MCC and AUROC, never accuracy alone**, plus the predicted-positive rate — a
  degenerate all-negative classifier scores ~0.49 on a balanced set.
- **Three seeds, mean ± sd.** A single-seed delta between conditions is not a result.
- **Always print the do-nothing baseline in the same table.** The question is never
  which condition wins but whether any beats not doing it.
- **Verify the plumbing before believing a number.** A surprising result gets shuffle
  controls; a suspicious one gets its preprocessing read from source, not assumed.

## Layout

```
src/schema_text.py         provenance-controlled schema → text (dataset/signal/model/all)
src/stetholm_describe.py   free-form LLM narration (StethoLM backend + Qwen2-Audio fallback)
src/ast_features.py        AST embeddings
src/stetholm_encoder.py    StethoLM / OPERA-CE encoders, standalone
src/contrastive_align.py   frozen two-tower + trained projector, probe & zero-shot
src/clip_finetune.py       trainable AST encoder, patient-level val, early stopping
src/kauh_replication.py    cross-corpus, cross-task control
src/official_split.py      the real ICBHI partition
src/summarize.py           results table, baseline always included
src/ctrl.py                bandwidth and depth controls
scripts/                   server drivers; download_datasets.sh fetches the corpora
docs/H12_DENSE_SCHEMA_CLIP.md  the next hypothesis, pre-registered and frozen
docs/RUNBOOK.md           machines, the edit/run loop, fixed paths, operational traps
docs/DATASET_NOTES.md      verified dataset structure and its traps
docs/ENVIRONMENT.md        how to get weights onto the offline server
```

Audio, features and model weights are not tracked — several GB, all regenerable.
