# ICASSP paper workspace

`icassp2027_draft.tex` is a layout-tested draft of the ICASSP paper. It currently uses
the official ICASSP 2026 `spconf.sty` and `IEEEbib.bst` as a provisional typesetting kit.
The ICASSP 2027 kit must replace those files when it is released or becomes downloadable.

The current PDF is intentionally **not submission-ready**:

- author names and affiliations are explicit placeholders;
- UKCOVID is labelled a discovery audit because its official test sets informed earlier
  protocol development;
- Coswara contains no model score until the frozen post-QC 100-pair gate is executed;
- any Coswara result will be reported as an external patient-level stress test, not as
  confirmatory proof.

Build the layout test with:

```bash
bash scripts/build_icassp_layout.sh /absolute/path/to/ICASSP2026_Paper_Templates
```

The command writes the PDF to `output/pdf/icassp2027_draft.pdf` and compilation files to
the ignored `tmp/pdfs/` directory. The verified references are in `references.bib`.
