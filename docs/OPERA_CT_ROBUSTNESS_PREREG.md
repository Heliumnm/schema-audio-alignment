# OPERA-CT second-backbone robustness: frozen protocol

Date frozen: 2026-08-20  
Standing: exploratory robustness analysis on the same UKCOVID discovery cohort.

## Purpose

The AST-6L result may be specific to one general-audio backbone. This analysis repeats the
same raw/correct/within-label/global comparison with frozen OPERA-CT, a respiratory audio
foundation model. It tests backbone sensitivity; it is not an independent dataset and
cannot repair the exploratory standing of UKCOVID.

## Frozen encoder and preprocessing

- OPERA repository already installed on the server;
- frozen UKCOVID waveform root: `ukcovid/audio/audio`;
- checkpoint: `encoder-operaCT.ckpt`;
- checkpoint SHA-256:
  `83c35b435518ad5f395bf4d34e552caa088faf9e63f6b8058d5288e9abb350ae`;
- `Cola(encoder="htsat")`, strict state-dict loading with zero missing/unexpected keys;
- frozen `extract_feature(x, 768)` output before the contrastive projection head;
- 16 kHz mono, native OPERA silence trim, 64 mel bins, 50--8,000 Hz, FFT 1,024,
  hop 512 and native per-window dB/min-max processing;
- native 8-second input; recordings shorter than 8 seconds use OPERA's repeat padding;
- after native silence trimming, recordings longer than 8 seconds use
  `ceil(trimmed_duration/8)` uniformly spaced windows covering the full signal; window
  embeddings are averaged equally to one participant embedding;
- no disease label, split, metadata or recording artefact controls preprocessing.

The full-coverage rule is our explicit adaptation because OPERA's released
`get_entire_signal_librosa` does not define a fixed-length policy for recordings longer
than its expected 8-second inputs. It is frozen before any OPERA UKCOVID disease score.

The first preflight launch stopped before model loading because both the frozen cohort
and a redundantly merged split table supplied a `splits` column. No embedding or outcome
was produced. The redundant merge was removed; the frozen cohort's existing split flags
are now the sole source used for stratified technical sampling.

## Technical gates

A stratified 100-recording preflight must cover short/long recordings, the 8-second
boundary, clipped/unclipped audio, both recruitment sources and all evaluation splits. It
must verify strict checkpoint loading, exact 251x64 mel shape, full-window coverage,
finite/nonzero 768-dimensional outputs and bit-identical repeated inference.

Full extraction is sharded across three GPUs. Every participant must appear exactly once;
all shard hashes, checkpoint/script/cohort hashes, window counts and failures are frozen.
The merged participant order must exactly match `results/ukcovid_audio_cohort.csv`.

## Alignment and evaluation

The frozen metadata texts, Phi-2 cache, pairings, projector architecture, batch size,
optimizer, 500 epochs, five seeds and final-checkpoint rule are unchanged. Only the audio
embedding cache changes from AST-6L to OPERA-CT.

Arms:

1. raw frozen OPERA-CT;
2. correct metadata pairing;
3. metadata shuffled within COVID label;
4. metadata shuffled globally.

The existing evaluator is reused without changing selection or calibration: Standard
train scaling, complete Standard-validation one-SE C selection, one final source-domain
Platt calibrator, then locked Standard/matched/matched-long scoring. Raw post-ReLU
projector output is primary; L2-normalized output is the prespecified sensitivity. The
evaluator's legacy `raw_ast` result key means the raw embedding supplied with this OPERA
run and must be relabelled `raw_opera_ct` in summaries.

Primary robustness contrast: correct minus within-label paired Delta(-NLL) on matched.
Secondary: paired DeltaAUROC, matched-long, correct minus raw OPERA-CT and within-label
minus global. Direction agreement with AST supports backbone robustness; disagreement is
a boundary condition and must be reported rather than averaged across backbones.
