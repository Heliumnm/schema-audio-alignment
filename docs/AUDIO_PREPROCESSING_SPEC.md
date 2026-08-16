# Audio preprocessing and the interpretation thresholds

**Frozen before any embedding is extracted.** Every arm shares this specification exactly;
a change here invalidates every arm, not one.

## Two interpretation corrections, recorded because both were stated wrongly first

**AUROC does not decompose.** It was written earlier that "of the 0.619 an audio model
reaches on the matched set, a substantial part is the 0.554 that artefacts reach". That is
not a valid operation on AUROC and the claim is withdrawn. The correct statement:

> Simple recording artefacts — duration, RMS, clipping fraction, file size — already reach
> **0.554 [0.528, 0.580]** on the matched test set. Any audio model must therefore be
> compared **directly and pairwise against the artefact model on the same patients**, not
> against 0.5.

**The recruitment probe is not a threshold at 0.578.** Artefacts decode recruitment source
at 0.578 [0.559, 0.596] on matched_long, and that is a *reference line for what file
properties alone carry* — not the bar an aligned representation must clear. The question
"does alignment write more source information into the representation" is answered only by

    probe(aligned representation) − probe(raw AST representation)

with the same probe, the same patients and a paired CI. An absolute number against 0.578
answers a different question.

**One superseded guess, recorded rather than deleted.** Clipping was expected to track
recruitment source. It does not: REACT 39.3% of recordings show clipping against Test and
Trace's 37.0%, and the median clipping fraction is 0 in both. What separates the two arms
is file size and duration (REACT 3.44e5 bytes / 3.67 s, T+T 3.69e5 / 3.84 s).

## The three audio baselines, frozen

| arm | what it establishes |
|---|---|
| `artifacts_only` | the floor from recording shape alone |
| `ast_only` | what the frozen representation reaches |
| `artifacts_plus_ast` | whether AST adds anything the shape did not already carry |

**The quantity that matters is the paired increment `artifacts_plus_ast − artifacts_only`**,
not whether `ast_only` happens to exceed 0.554.

### Result wording, fixed in advance

* `ast_only > artifacts_only` (paired, CI excluding 0) → *AST is stronger than a simple
  artefact model.*
* `artifacts_plus_ast > artifacts_only` (paired, CI excluding 0) → *AST carries information
  beyond simple artefacts.*
* Neither may be written as *"the model hears pathological cough."* The most that is
  supported is **"the representation contains additional waveform information."**

## Preprocessing, frozen

1. **Resample to 16 kHz, mono.** No per-file peak normalisation — normalising would erase
   the loudness and clipping differences the artefact baseline just showed are real, and
   would silently change what the artefact control is controlling for.
2. **≤ 10.24 s:** right-pad with zeros to 10.24 s and keep an explicit valid-frame count.
   Never centre-pad; the pad must be on one known side.
3. **> 10.24 s: no truncation and no centre crop.** A long recording may hold the cough
   anywhere, so the whole recording is covered by
   `n_win = ceil(duration / 10.24)` windows whose starts are spread uniformly:

       start_i = i · (duration − 10.24) / (n_win − 1),   i = 0 … n_win−1

   so window 0 begins at 0 and the last ends exactly at the end of the recording. The
   longest cohort recording is 48.90 s → 5 windows.
4. **One participant contributes one loss.** Windows are averaged into a single embedding;
   they are never treated as independent training examples.

## The valid-patch mask, computed from frames and stride — never from `duration / 101`

AST's front end is 128 mel bins × 1024 frames at 100 fps, cut into 16×16 patches with
stride 10, giving a 12 × 101 grid. Time patch `t` covers mel frames `[10t, 10t + 16)`.

With `n_valid = min(round(seconds × 100), 1024)` real mel frames in a window:

> **A time patch is kept only if it is fully covered: `10t + 16 ≤ n_valid`.**
> Boundary patches that straddle the pad are **discarded**, not down-weighted.

so `n_patch_valid = max(0, floor((n_valid − 16) / 10) + 1)`.

Discarding rather than fractionally weighting costs at most one patch (160 ms) at the tail
and guarantees that no averaged patch contains any padding at all. At the cohort median of
3.75 s that leaves 36 of 101 time patches; at the 0.5 s floor it leaves 4. Any window
yielding 0 valid patches is a hard error, not a silent zero vector.

## The representation, frozen

* AST, **first six layers** — the depth this project already established, not re-chosen;
* **frozen** in round one; only a shared classification head trains;
* the two leading special tokens are dropped before reshaping to 12 × 101;
* patches are averaged over **valid time patches only**, across all 12 frequency rows;
* window embeddings are then averaged with equal weight.

## What is saved with every extraction

participant id, each window's start time, the valid-patch count per window, the window
count, the final embedding, this specification's version, and the checkpoint hash — so a
later run can be shown to be the same run.

## Pre-extraction check

100 recordings, **stratified rather than random**, covering: the shortest recordings; the
region either side of 10.24 s; the longest; single-window and multi-window; clipped and
unclipped; REACT and Test and Trace; Standard, matched and matched_long. Verified: head and
tail are both covered by some window; short-recording masks are right; window counts match
`ceil(duration / 10.24)`; no all-zero mel or degenerate embedding; and a repeated run is
bit-identical.

Full extraction does not start until every one of those passes.
