# Coswara data-gate runbook

Frozen implementation commit: `6d783a4`

Execution outcome (2026-08-20): **NO-GO at the preregistered balance gate; no model was
fitted and no Coswara prediction was read.** See `docs/COSWARA_DATA_GATE_OUTCOME_ZH.md`
and `results/coswara_external_data_gate.json`.

This runbook stops before representation extraction, projector training or disease-model
evaluation.  Its only outcome is a frozen participant/audio manifest and a GO/NO-GO data
gate.

## 1. Finish and verify the pinned download

Do not restart a healthy resumable download.  Completion is recognised only when the
13-GB archive has the pinned byte count and MD5, the aria2 sidecar is gone, ZIP CRC passes,
and the fixed 418-entry/43-date/153-part structure passes.  File size or a downloader log
alone is not completion.

The archive must be:

- `Coswara-Data-dataset-paper-publication.zip`;
- 12,984,309,908 bytes;
- MD5 `53721d9c106f99872bf7f878c8196d31`.

Retain the download log and tool versions.  Do not redistribute the archive because the
Zenodo page and pinned README disagree on the licence.

## 2. Pin the two small metadata inputs

Fetch both files from upstream commit
`bf300ae9dc47918be4a30de90436fe7563fafb45`, then require:

- `combined_data.csv` SHA-256
  `e462c503bee3408214195855975b0eda08dd1188c0d521b494d93d388d60a72d`;
- `annotations/cough-heavy_labels_debottam.csv` SHA-256
  `ab41f10875796818f44022c3f067bc4b3724bc48100bff447edc95354d98a9bb`.

The quality file is diagnostic and defines sensitivity cohorts; it never controls primary
audio inclusion.

## 3. One-date extraction smoke

Use the real archive and date `20200424`.  Run the frozen streaming extractor with an
empty output root.  It must materialise only paths of the form
`20200424/PARTICIPANT/cough-heavy.wav`, publish one v2 completion marker, and pass the
partial QC audit.  Smoke outputs use separate filenames and never overwrite the future
formal QC outputs.

The smoke is accepted only if:

- the archive re-verifies before extraction;
- every split part is contiguous and the gzip/tar stream reaches EOF;
- no link, unsafe path, duplicate target or unexpected file is accepted;
- the partial QC manifest and audit report publish once without overwrite.

## 4. Full extraction and objective QC

Run the same extractor without `--date`.  It resumes the verified smoke date and processes
all 43 dates.  A full summary must verify 43 completion markers and all 153 split members.

Then run `src/audit_coswara_audio.py` once without `--allow-partial`.  Save:

- `results/coswara_audio_qc.csv`;
- `results/coswara_audio_qc_audit.json`;
- the extractor, metadata, quality and manifest hashes;
- the complete decode/failure and duplicate lists.

`record_date` versus archive batch-date disagreement is reported only as a diagnostic.
Primary objective QC requires a regular decodable PCM WAV, at least 0.5 seconds, matching
header/decoded frame counts, finite nonzero samples, and stable raw/PCM hashes.

## 5. Freeze pairs and obey the gate

Run `src/freeze_coswara_external_split.py` exactly once on the formal QC manifest.  It
writes the participant manifest, pair manifest and split audit before returning its gate
status.

GO requires all of the following:

- at least 100 primary pairs after the deterministic balance path;
- the frozen exact/fine-balance limits pass;
- the frozen train/validation/source-test size and class-count limits pass;
- no participant or decoded-PCM duplicate crosses a split.

The metadata-only ceiling is only 101 passing pairs.  If real audio QC leaves 99 or fewer,
or balance fails, exit status 3 is the planned **NO-GO**.  Do not change modality,
thresholds, labels, matching fields or balance limits.  Record the external dataset as a
descriptive sensitivity only and continue the ICASSP paper as a UKCOVID audit.

If the gate passes, commit the three frozen outputs and their hashes before writing or
running any AST/Phi-2 representation or disease evaluator.
