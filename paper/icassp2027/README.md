# ICASSP 2027 paper package

This directory is the submission-oriented English manuscript built with the
official ICASSP 2027 LaTeX kit. The paper is a scientific rewrite of
`../../report/ICASSP_DRAFT_ZH.md`, not a sentence-by-sentence translation.

## Current status

- The PDF builds successfully as four pages total and has been visually checked.
- The abstract contains 138 words and the keyword list contains five items.
- There are no unresolved citations or horizontal overflow warnings.
- The figures are generated from `figure_results.csv` and are already included
  as PDF assets.
- Author names and affiliations are still placeholders in `main.tex`. ICASSP is
  not double blind, so they must be replaced before submission.

## Build locally

```bash
bash build.sh
```

The compiled manuscript is written to
`../../output/pdf/icassp2027_pairing_audit.pdf`.

## Upload to Overleaf

Upload the prepared ZIP from `../../output/overleaf/`, choose `main.tex` as the
main document if Overleaf does not detect it automatically, and compile with
pdfLaTeX. The ZIP contains only the TeX source, bibliography, official style
files, figures, and submission checklist; it contains no private dataset or
participant-level result.

The complete experiment ledger remains in the repository. This folder contains
only claims and results that fit the four-page paper and whose evidence status is
stated in the manuscript.
