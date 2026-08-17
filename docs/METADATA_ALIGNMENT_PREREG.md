# Metadata alignment — pre-registration

**Frozen before any projector is trained.** Read
[`PROTOCOL_HISTORY.md`](PROTOCOL_HISTORY.md) first: it records two protocol faults that
were found and withdrawn, and the validation rule that is now fixed and not revised again.

## The question, as it now stands

The original framing — "does aligning audio to audible evidence beat aligning it to
clinical metadata on diagnosis" — no longer survives its own gates. The teacher check
returned a development NO-GO, and the source-domain fusion gate found no detectable
increment from AST over a symptom questionnaire on the full Standard val
(+0.0032 [−0.0003, +0.0069], includes 0), because metadata alone already reaches 0.9764
there. So the question is not whether alignment improves diagnosis. It is:

> **Does metadata alignment amplify source-specific information, or does it produce disease
> information that transfers to the covariate-matched OOD population?**

This is answerable, it is the mechanism the dataset is unusually well suited to expose, and
it does not depend on any of the arms that failed.

### What makes the question sharp here

Recruitment source predicts the COVID label at **AUROC 0.9966 in Standard train** and
**0.5000 in the matched test sets**. Symptoms are a near-perfect proxy for recruitment
(`symptom_none` is 0.786 among train negatives and 0.019 among positives). So a model that
aligns audio to symptom text in this training distribution is, mechanically, being pushed
to encode a variable that carries the label in the source and carries nothing in the
target. Whether it actually does so is the measurement.

## Four arms, one projector architecture

| arm | pairing |
|---|---|
| `raw_ast` | no alignment — the frozen v3 reference |
| `correct_metadata` | audio paired with that participant's own metadata text |
| `within_label_shuffled` | metadata permuted **within COVID label**, preserving group-level co-occurrence and destroying the individual correspondence |
| `global_shuffled` | metadata permuted globally — the pure negative control |

The three trained arms share the projector, training steps, batch size, temperature and
seeds exactly. Only the pairing differs.

## The loss, taken from the paper

RespiraMFM's Stage 1, verbatim from the paper:

$$z^a_i = \frac{f_\theta(e^a_i)}{\|f_\theta(e^a_i)\|},\qquad z^t_i = \frac{e^t_i}{\|e^t_i\|}$$

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(z^a_i\cdot z^t_i/\tau)}{\sum_{j=1}^{N}\exp(z^a_i\cdot z^t_j/\tau)}$$

Verified against the paper rather than assumed: **both encoders frozen**; only the
projector `f_θ` trains, and **only on the audio side** — the text vector is normalised and
passed through unchanged; the loss is **one-directional audio→text**, not CLIP's symmetric
form (they cite Chen et al. 2020, not Radford et al.); negatives are **in-batch**; and
**identical metadata texts are not given any special treatment**.

## Everything frozen from the official source

Pinned to RespiraMFM commit
**`b4224f231f947f3ac3ba8dcafe1f11d8a9f3e525`**. Read from that source rather than inferred
from the paper:

| | value | where |
|---|---|---|
| projector | **768 → 1024 → 2560** | `model_projector.py`; `out_dim = d_llm`, and Phi-2's `d_llm` is 2560 |
| block 1 | Linear, LayerNorm, ReLU, **Dropout 0.1** | `ContrastiveProjectionHead` |
| block 2 | Linear, **LayerNorm, ReLU** | the trailing norm+activation the paper's appendix does not mention |
| loss | one-directional audio→text | `symmetric_loss = False` is hard-coded; the symmetric branch exists but is unreachable |
| temperature | 0.07 | `contrastive_loss` default |
| batch | **64**, `shuffle=True`, `drop_last=False` | `DataLoader(..., batch_size=64, shuffle=True)` |
| optimiser | **Adam, lr 1e-3**, PyTorch defaults, **no scheduler, no weight decay, no clipping** | `torch.optim.Adam(model.parameters(), lr=learning_rate)` |
| epochs | **500**, and the **final** state is saved | `train.py` passes `num_epochs=500`, overriding the function's default of 100; `..._final.pth` is written after the loop |
| text pooling | mean over the sequence | `last_hidden_state.mean(dim=1)` |
| text max length | 125 | `truncation=True, max_length=125` |

**Two deliberate departures, both forced by our texts and recorded as ours:**

* **`max_length = 200`.** Our schema texts are 188–200 tokens, so 125 fits none of the
  12,992. This says our texts are long; it is not a claim about RespiraMFM's own contexts,
  which we have not measured.
* **mask-aware mean rather than plain mean.** The official code encodes one context at a
  time, so its `padding=True` adds no padding and its plain mean already *is* a valid-token
  mean. Batching the cache preserves that semantics; it does not change it.

**An implementation property worth stating plainly:** the projector ends in ReLU, so its
output is non-negative and every projected audio vector lies in the positive orthant. After
`F.normalize` all pairwise cosines among audio projections are therefore ≥ 0. This is
copied faithfully and is recorded as a property of the architecture, not as a defect or a
predicted cause of failure.

### Seeds and shuffled pairings

Seeds `0, 1, 2, 3, 4`. Within a seed the three trained arms share initialisation, audio
batch order and the dropout random stream, so only the pairing differs.

Each shuffled pairing is **drawn once per seed and held fixed for all 500 epochs**, never
resampled per epoch. `within_label_shuffled` permutes participants inside each COVID label;
`global_shuffled` permutes across the whole training set. Both forbid self-pairing. The
mapping and its hash are saved. Landing on an identical schema by chance is **not**
prevented — the collision rate is measured and reported.

### The duplicate-induced loss floor, from the real manifest

20,714 training participants at batch 64 with `drop_last=False` gives 323 full batches plus
one of 42. Mean `log K` over that manifest is **0.5553 nats** (sd 0.0020 over 20 shuffles).
Training reports the manifest value; the iid batch-64 approximation of 0.5658 is not used.
Standard train carries **5,437** distinct texts over those 20,714 participants.

### Single-positive stays; multi-positive is a sanity check, not an arm

20,714 training participants produce only **5,437** distinct schema texts, and the single
most common profile covers 7.9% of them, so a sizeable share of each anchor's in-batch
negatives carries a text identical to its own.

That is **a loss floor, not a learning pathology**. With K identical texts in a batch,

$$L_{\text{multi}} = L_{\text{single}} - \log K$$

a constant in the parameters. Measured on real tensors: loss difference 0.335955 against a
predicted 0.335955, max gradient difference 6.3e-08, gradient cosine 1.0000000
(`src/test_multipositive_equivalence.py`). Identical texts also cannot be pushed apart from
one another, the text encoder being frozen. So exact-text multi-positive gets no arm and no
seeds; it stays as the sanity check it is. Treating **near-neighbour** texts (one field
apart) as positives would be a genuinely different hypothesis and needs its own
pre-registration, with the field-similarity metric and threshold fixed in advance.

## Metadata content, frozen

Only what the metadata-only baseline already uses: **age, sex, smoking status, asthma,
other respiratory condition, and the symptom fields.**

Excluded, and the exclusions are the point: COVID result and test information, viral load,
CT values, **recruitment source**, timestamps, file size, duration, loudness, clipping, and
anything generated by an LLM.

Deterministic structured text, missing values written explicitly:

    [AGE=40-49] [SEX=FEMALE] [SMOKER=NO] [ASTHMA=NO] [COUGH_ANY=YES] [NONE=NO] ...

`[MISSING]` is a value. A missing field is never silently rendered as "no".

*Divergence from the paper, recorded:* its UKCOVID field table lists asthma but not smoking
status or other respiratory condition. Ours includes both, so that the alignment text and
the metadata-only baseline describe the same information.

## Text encoder

Phi-2, revision pinned to `810d367871c1d460086d9f82db8696f2e0a0fcd0`, `eval()`, dropout
off, fully frozen, embeddings computed once and cached. Tokenizer, Transformers version and
precision recorded with the run.

**This is a paper-faithful reimplementation of RespiraMFM Stage 1, not an exact
reproduction.** The paper does not state which Phi-2 layer supplies `e^t`, whether pooling
is mean or last-valid-token, the aligner's maximum text length, or whether special tokens
enter the pooling. Those are our choices, fixed in the text-axis check below and recorded
as ours. The downstream head is also ours: RespiraMFM's Stage 2 is Phi-2 with LoRA, which
we do not run, so nothing here speaks to the accuracy of their full system.

## Text-axis feasibility check — before any training

1. identical schema text yields **bit-identical** embeddings;
2. **every metadata field is linearly decodable** from the text embedding;
3. texts differing in **one field** are closer than random text pairs;
4. retrieval computed **duplicate-aware** — every participant carrying an identical profile
   counts as a correct answer, or the 7.9% modal profile makes retrieval look broken when
   it is only ambiguous;
5. the modal profile's effect on retrieval and on the loss floor is quantified, not assumed.

## Training

Standard train only. Both encoders frozen; only the projector trains. **The paper fixes 500
epochs, so the epoch count is fixed at 500 and is not selected against any COVID
validation.** Five seeds. Downstream: the same regularised logistic head, the same one-SE
fold selection on the full Standard val, and the same single source-domain calibrator as
the v3 reference. The test sets are read once.

### Smoke test first, one seed per arm, few steps

Loss decreases; embeddings do not collapse; correct pairing beats global shuffle on
*training-set* retrieval; the five-seed plumbing and per-participant outputs are correct.
**No test-set disease metric is read.** Only after the smoke test and the text-axis check
both pass does the five-seed run start.

## Frozen interpretation

Disease, on the matched OOD test: paired `Δ(-logloss)` primary, paired `ΔAUROC` secondary,
`correct_metadata − raw_ast` and `correct_metadata − within_label_shuffled`.
Representation: `probe(aligned) − probe(raw AST)` with paired CIs, reported separately for
sex, age, symptoms and recruitment source.

| pattern | reading |
|---|---|
| Standard up, matched flat, metadata probes stronger | **alignment amplifies the training-distribution shortcut** |
| `correct` beats both shuffles **and** matched improves | individual-level information that transfers |
| `correct` ≈ `within_label_shuffled` | the effect is label-level co-occurrence, not individual pairing |
| `correct` ≈ `global_shuffled` | alignment changed the representation in no useful way |

## Standing of any result here

**Exploratory.** The official test sets have already informed design decisions across three
protocol versions. A positive alignment result therefore requires confirmation on a fresh
untouched holdout or an external dataset before it is claimed. Only if correct metadata
alignment produces a measurable change at all does the next stage — schema text versus
natural description at identical information content — become worth running.
