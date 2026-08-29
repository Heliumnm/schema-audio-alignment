# CODA TB external transfer audit

> **Outcome (2026-08-29): NO-GO at the frozen model-blind balance gate.** All 9,772
> solicited-cough WAV files passed objective QC, but the final 100 matched pairs had maximum
> absolute SMD 0.276 (limit 0.12) and maximum categorical level difference 0.13 (limit
> 0.08). No representation, projector, classifier, or model score was produced. See
> [`../../docs/CODA_TB_DATA_GATE_OUTCOME_ZH.md`](../../docs/CODA_TB_DATA_GATE_OUTCOME_ZH.md)
> and [`../../results/coda_tb_data_gate_summary.json`](../../results/coda_tb_data_gate_summary.json).

This directory contains the model-blind data gate for applying the frozen
Pairing-Controlled Transfer Audit to the controlled-access CODA TB training data.

The raw data are governed by the Synapse access terms.  Never commit, redistribute, or
package participant-level metadata, audio, participant manifests, matched-pair manifests,
or derived embeddings.  Only code and aggregate, disclosure-checked summaries belong in
the public repository.

## First stage: model-blind feasibility only

The first run is deliberately limited to participant linkage, objective waveform QC,
duplicate detection, missingness, deterministic participant splitting, and covariate
matching.  It must not import an audio encoder or read any model prediction.

On the server:

```bash
bash external_validation/coda_tb/run_feasibility.sh \
  /mnt/hd/data_heliu/resp_datasets/CODA_TB/raw \
  /mnt/hd/data_heliu/resp_datasets/CODA_TB/audit
```

The command writes controlled participant and pair manifests to the supplied audit
directory plus an aggregate `coda_tb_data_gate.json`.  A zero exit code means every frozen
GO gate passed.  Exit code 3 is a planned NO-GO, not a software failure.

Formal AST/OPERA extraction and Correct/Within-label/Global alignment are forbidden until
the aggregate gate has been reviewed and the controlled manifests have been hashed and
frozen. The formal gate has now been reviewed and returned NO-GO, so those model stages
remain closed under this preregistration.

## Frozen data assumptions

- Synapse Train folder: `syn39711065`;
- solicited-cough folder: `syn40358494`;
- clinical table: `meta_data/Clinical/CODA_TB_Clinical_Meta_Info.csv`;
- country/domain table: `meta_data/Clinical/CODA_TB_additional_variables_train.csv`;
- audio map: `meta_data/Cough Metadata/CODA_TB_Solicited_Meta_Info.csv`;
- expected metadata rows: 1,105 clinical participants and 9,772 solicited WAV mappings;
- primary reference label: `tb_status` (microbiologic reference standard);
- primary audio: solicited cough only; longitudinal cough is not downloaded or substituted.

See `PREREGISTRATION_ZH.md` for the scientific contract.
