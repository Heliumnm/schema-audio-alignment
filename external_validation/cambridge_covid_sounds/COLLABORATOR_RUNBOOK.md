# Cambridge COVID-19 Sounds controlled-access audit — collaborator runbook

This package is designed to run inside the authorised institution. It must not copy raw
audio, row-level metadata, participant identifiers, matched-pair manifests, embeddings, or
predictions outside the Data Transfer Agreement environment.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The formal model stage additionally requires local, already-authorised copies of the
frozen AST checkpoint, Phi-2, the OPERA repository and the frozen OPERA-CT checkpoint.

## 2. Map the DTA release without changing the protocol

```bash
cp config.example.json config.local.json
```

Edit only:

- local input/output/model paths;
- raw CSV column names under `columns`;
- raw value maps needed to map the Cambridge release into the already named canonical
  fields.

Do not edit the disease definition, official fold, matching fields, thresholds, model
arms, seeds, epochs or equivalence margins. If a required field is genuinely absent, stop
and report that fact rather than substituting another variable.

For the public benchmark directory, leaving `audio_manifest_csv=null` enables the official
`0426_EN_used_task2` scanner. For a different DTA layout, supply a CSV with participant ID,
relative audio path and modality; paths remain private.

Participant and metadata tables may be CSV, XLSX or XLSM. They may be supplied separately;
the gate performs a validated participant-ID join before checking the canonical columns.

## 3. Run the model-blind data gate

```bash
bash run_once.sh config.local.json gate
```

Review `output/.../public/data_gate.json`. A `NO_GO` is a valid final outcome. Return only
`output/.../public/PUBLIC_RESULTS.zip`; do not run a model.

## 4. Formal unlock after independent review

If and only if the gate is `GO`, send the aggregate gate JSON to the project lead. After
approval, set:

```json
"formal_unlock": "FROZEN_PROTOCOL_AND_DATA_GATE_GO"
```

Then run:

```bash
bash run_once.sh config.local.json formal
```

The command is resumable at completed artefacts. It refuses to overwrite an incompatible
cache, representation or training manifest.

## 5. Return contract

Return only:

```text
output/.../public/PUBLIC_RESULTS.zip
git commit SHA
Python/CUDA/GPU environment summary
whether the DTA permits publication of the aggregate tables
```

Do not return `output/private`. The public bundle deliberately contains no participant ID,
audio path or row-level prediction. If an authorised data steward imposes stricter rules,
those rules override this package.

## 6. Troubleshooting rule

Technical failures may be fixed only when they do not change the frozen scientific
definition. Any proposed change to labels, modality, cohort, matching, thresholds, arms,
seeds, epochs or metrics must stop execution and be discussed before any model score is
read. Keep the failing log and current commit unchanged.
