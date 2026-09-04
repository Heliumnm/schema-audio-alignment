# ICASSP paper workspace

`icassp2027_draft.tex` is a layout-tested draft of the ICASSP paper. It currently uses
the official ICASSP 2026 `spconf.sty` and `IEEEbib.bst` as a provisional typesetting kit.
The ICASSP 2027 kit must replace those files when it is released or becomes downloadable.

The full Chinese paper-structured source draft is
[`../report/ICASSP_DRAFT_ZH.md`](../report/ICASSP_DRAFT_ZH.md). It integrates the current
UKCOVID discovery audit and the CODA TB, Cambridge Task-2, and Coswara sensitivity results
under Abstract / Introduction / Related Work / Data / Method / Results / Discussion /
Limitations / Conclusion headings. The shorter English and LaTeX drafts should be updated
from that source only after the claim and number audit is complete.

The current PDF is intentionally **not submission-ready**:

- author names and affiliations are explicit placeholders;
- UKCOVID is labelled a discovery audit because its official test sets informed earlier
  protocol development;
- the LaTeX text predates the completed CODA TB, Cambridge Task-2, and Coswara sensitivity
  results; their current evidence levels and numbers are consolidated in the Chinese source
  draft;
- CODA TB is a training-release match-first secondary analysis, Cambridge uses a custom
  reconstructed strict-COVID endpoint, and Coswara is a post-hoc stress test; none is labelled
  as untouched external confirmation;
- the matched direct-fusion attribution control and OPERA-CT second-backbone robustness
  analysis are included, but both remain part of the UKCOVID discovery audit.

Build the layout test with:

```bash
bash scripts/build_icassp_layout.sh /absolute/path/to/ICASSP2026_Paper_Templates
```

The command writes the PDF to `output/pdf/icassp2027_draft.pdf` and compilation files to
the ignored `tmp/pdfs/` directory. The verified references are in `references.bib`.
