# ICASSP 2027 submission checklist

## Required manual edits

- [ ] Replace the author-name placeholder in `main.tex`.
- [ ] Replace the affiliation placeholder in `main.tex`.
- [ ] Confirm the title, author order, and affiliations with every co-author.
- [ ] Confirm whether acknowledgements or funding disclosures are required.
- [ ] Confirm the final wording of the ethical-compliance statement.

## Scientific checks

- [ ] Keep UKCOVID labelled as an exploratory discovery audit.
- [ ] Keep CODA TB, Cambridge, and Coswara labelled as secondary or post-hoc
      sensitivity analyses, not untouched external confirmation.
- [ ] Do not describe `Correct - Within-label` as a causal identity effect.
- [ ] Do not claim equivalence or zero effect from confidence intervals that
      include zero.
- [ ] Keep UKCOVID-HeAR labelled as a post-hoc third-backbone robustness analysis,
      not a primary UKCOVID backbone.
- [ ] Keep the central count fixed unless the evidence ledger changes:
      retrieval 11/12, sex probe 12/12, matched disease 0/12.

## Format checks

- [ ] Compile with the official `spconf.sty` and `IEEEbib.bst` in this folder.
- [ ] Confirm that the abstract remains between 100 and 150 words.
- [ ] Confirm that there are no more than five index terms.
- [ ] Confirm the final PDF page count and inspect every page after author edits.
- [ ] Run the official submission format checker when it becomes available.
- [ ] Verify that all fonts are embedded in the uploaded PDF.

## Reproducibility and policy

- [ ] Preserve the frozen pairings, five seeds, and final-checkpoint-only rule.
- [ ] Keep matched targets out of model, epoch, readout, and hyperparameter
      selection.
- [ ] Archive the final Git commit hash and the PDF hash with the submission.
- [ ] Review the current ICASSP policy on generative-AI assistance and add any
      required disclosure.
- [ ] Confirm that the upload contains no controlled data or participant-level
      predictions.
