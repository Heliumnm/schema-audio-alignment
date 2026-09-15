# ICASSP 2027 submission checklist

## Required manual edits

- [x] Replace the author-name placeholder in `main.tex`.
- [x] Replace the affiliation placeholder in `main.tex`.
- [ ] Confirm the title, author order, and affiliations with every co-author.
- [ ] Confirm whether acknowledgements or funding disclosures are required.
- [ ] Confirm the final wording of the ethical-compliance statement.

## Scientific checks

- [x] Keep UKCOVID labelled as an exploratory discovery audit.
- [x] Keep CODA TB, Cambridge, and Coswara labelled as secondary or post-hoc
      sensitivity analyses, not untouched external confirmation.
- [x] Do not describe `Correct - Within-label` as a causal identity effect.
- [x] Do not claim equivalence or zero effect from confidence intervals that
      include zero.
- [x] Keep UKCOVID-HeAR labelled as a post-hoc third-backbone robustness analysis,
      not a primary UKCOVID backbone.
- [x] Keep E1--E3 labelled as targeted post-hoc diagnostics; do not mix the
      target-assisted readout with source-only transfer results.
- [x] Describe `W(y,s)` as a sex-preserving pairing control, not a causal
      decomposition of participant identity.
- [x] Say that `W(y,s)` substantially attenuates the sex-probe contrast; do not
      claim that it explains all residual profile-retrieval gain.
- [x] Keep source-only matched evaluation separate from the target-assisted
      matched-long readout diagnostic; the latter does not rule out a readout effect.
- [x] Report the actual resampling units: profiles by seed for macro retrieval,
      participants by seed for UKCOVID probe and disease endpoints.
- [x] State that `W(y,s)` changes participant identity, not necessarily the exact
      schema text; retain the audited W and W(y,s) collision ranges.
- [x] Keep the 11/12 `C-W` retrieval, 12/12 `C-W` sex-probe, and 0/12 positive
      primary matched-disease `C-W(y,s)` intervals descriptive and secondary to
      the UKCOVID `W(y,s)` result; note that the disease point estimates have
      five positive and seven negative signs.

## Format checks

- [x] Compile with the official `spconf.sty` and `IEEEbib.bst` in this folder.
- [x] Confirm that the abstract remains between 100 and 150 words (133 words).
- [x] Confirm that there are no more than five index terms.
- [x] Confirm the final PDF page count and inspect every page after author edits
      (five pages total; page 5 contains references only).
- [ ] Run the official submission format checker when it becomes available.
- [x] Verify that all fonts are embedded in the uploaded PDF.

## Reproducibility and policy

- [x] Preserve the frozen pairings, five seeds, and final-checkpoint-only rule.
- [x] Keep matched targets out of model, epoch, readout, and hyperparameter
      selection.
- [ ] Archive the final Git commit hash and the PDF hash with the submission.
- [ ] Review the current ICASSP policy on generative-AI assistance and add any
      required disclosure.
- [x] Confirm that the upload contains no controlled data or participant-level
      predictions.
