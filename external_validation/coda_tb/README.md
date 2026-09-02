# CODA TB external transfer audit

> **Secondary v2 update (2026-09-03): model-blind GO.** Match-first v2 selected the
> frozen 100-pair target from all 1,081 eligible participants before splitting the
> remainder. Maximum absolute SMD was 0.0819 (limit 0.12) and maximum categorical level
> difference was 0.040 (limit 0.08); every frozen data gate passed in two bit-identical
> runs. This permits a separately preregistered model experiment but is not itself a model
> result. See [`PREREGISTRATION_MATCH_FIRST_V2_ZH.md`](PREREGISTRATION_MATCH_FIRST_V2_ZH.md)
> and [`../../docs/CODA_TB_MATCH_FIRST_V2_OUTCOME_ZH.md`](../../docs/CODA_TB_MATCH_FIRST_V2_OUTCOME_ZH.md).

> **Formal model protocol frozen (2026-09-03).** The permitted v2 model experiment is now
> specified before any CODA representation or model score is read: frozen AST-6L and
> OPERA-CT, Correct/Within-label/Global pairings, five seeds, source-only model selection,
> and a single read of the 100-pair matched target. See
> [`PREREGISTRATION_FORMAL_MODELS_V1_ZH.md`](PREREGISTRATION_FORMAL_MODELS_V1_ZH.md).

> **Original v1 outcome (2026-08-29): NO-GO at the frozen model-blind balance gate.** All 9,772
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

Formal AST/OPERA extraction and Correct/Within-label/Global alignment were forbidden under
the original split-first v1 after its NO-GO. The separately frozen match-first v2 returned
GO, and its model protocol is now frozen in
`PREREGISTRATION_FORMAL_MODELS_V1_ZH.md`. Only that secondary protocol may unlock model
execution; v1 remains closed and is not overwritten.

## Secondary match-first v2

After the aggregate v1 gate was inspected—but before any CODA representation or model score
was produced—a single model-blind sensitivity analysis was frozen. It changes only the order:
the 100-pair matched target is selected from all eligible participants before the remaining
participants are divided into train/validation/source-test. All matching fields, costs,
thresholds and power requirements remain unchanged.

See [`PREREGISTRATION_MATCH_FIRST_V2_ZH.md`](PREREGISTRATION_MATCH_FIRST_V2_ZH.md). Run it in
a new controlled output directory so the v1 record is never overwritten:

```bash
bash external_validation/coda_tb/run_match_first_v2.sh \
  /mnt/hd/data_heliu/resp_datasets/CODA_TB/raw \
  /mnt/hd/data_heliu/resp_datasets/CODA_TB/audit_match_first_v2
```

This is a secondary external sensitivity analysis, not a replacement for the v1
preregistration and not the official CODA challenge validation.

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
