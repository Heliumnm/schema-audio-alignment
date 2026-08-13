# Contrastive audio–text alignment for respiratory sound

**A negative result, thoroughly controlled — plus one finding that outlived it.**

The project set out to test whether structured clinical schema text is a better
alignment target than free-form LLM narration for respiratory audio, CLIP-style. It
is not, and neither is anything else tried: **no form of contrastive audio–text
alignment beat simply probing the frozen audio features.** Ten hypotheses were
falsified with controls, across two phases. Along the way a separate result held up: **a general-purpose
AudioSet encoder substantially outperforms every respiratory-specific foundation
model tested.**

Full detail in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md); this is the summary.

**Phase 2 is under way** under a reframing that survives the above — *can an explicit
structured acoustic representation learn a field → spectrogram-patch correspondence*,
where the payoff is grounding rather than classification. It also corrects two
Phase-1 overstatements. See [`docs/PHASE2_MULTIPOSITIVE.md`](docs/PHASE2_MULTIPOSITIVE.md).

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
| 10 | A typed set encoder beats strings at matched content | ❌ reverses on crackle; ~¾ of the gain survives shuffling the schema |

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

## Where this stands

The line is finished. Every testable hypothesis has been run with controls, and the
one direction left untested is not viable on the current data.

**Ten hypotheses, two phases, all rejected.** Phase 1 asked whether any text source or
training setup makes contrastive alignment work; Phase 2 asked whether the loss or the
representation was at fault. The answer in both cases is neither — nothing beats
probing the frozen features, and the schema representation shows no evidence of
carrying usable structure.

**The remaining direction — grounding — lacks both its premise and its data.** The
reframing that opened Phase 2 (structure may pay off in localisation rather than
AUROC) is still formally untested, but:

- its premise is gone: ten falsified hypotheses leave no evidence the schema
  representation carries structure the model can use;
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

Six retractions across ten hypotheses. Each came from a control that was cheaper than
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
docs/DATASET_NOTES.md      verified dataset structure and its traps
docs/ENVIRONMENT.md        how to get weights onto the offline server
```

Audio, features and model weights are not tracked — several GB, all regenerable.
