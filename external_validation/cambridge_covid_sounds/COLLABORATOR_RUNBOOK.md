# Cambridge COVID-19 Sounds controlled-access audit — collaborator runbook

This package is designed to run inside the authorised institution. It must not copy raw
audio, row-level metadata, participant identifiers, matched-pair manifests, embeddings, or
predictions outside the Data Transfer Agreement environment.

## 1. Environment

For the model-blind gate only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gate.txt
```

The formal model stage requires `pip install -r requirements.txt` and local, already-authorised copies of the
frozen AST checkpoint, Phi-2, the OPERA repository and the frozen OPERA-CT checkpoint.

## 2. Current release: match-first v2 model-blind gate (recommended)

The delivered `task1/` and `task2/` directories contain audio but not the
`data_0426_en_task2.csv` file referenced by the public repository. Do not invent that file or
describe a new random split as the official benchmark. Run:

```bash
bash run_reconstructed_match_first_v2.sh \
  /DTA/covid19/metadata \
  /DTA/test2 \
  /DTA/cambridge_match_first_v2
```

The second argument must be the Task-2-only audio tree and must retain
`participant-ID/collection-time/cough-file` linkage. The reconstruction keeps English
`positiveLast14/last14` versus `negativeNever` sessions, excludes participants observed under
both labels, and selects one audio-linked session per participant by a fixed label-blind hash.
It first freezes exactly 100 matched positive/negative pairs, then splits only the remaining
participants 70/15/15 within label x platform into train, validation and source-test. It is a
versioned secondary sensitivity endpoint, not an official split reproduction or an untouched
confirmation.

If `public/data_gate.json` says `NO_GO`, return the public bundle and stop. Do not change the
match-first order, split ratio, label definition, 100-pair, 0.12-SMD or 0.08 fine-balance
criteria. The earlier `run_reconstructed_gate.sh` command is the preserved split-first v1 and
already returned NO-GO; do not substitute it for v2.

## 3. Optional route if the official Task-2 CSV is later recovered

Use Task 2 (COVID-positive versus COVID-negative), not Task 1 (respiratory-symptom
prediction). No manual metadata merge is needed. Provide exactly these four paths:

```bash
bash run_task2_gate.sh \
  /DTA/task2/data_0426_en_task2.csv \
  /DTA/covid19/metadata \
  /DTA/covid19 \
  /DTA/cambridge_audit_output
```

The audio root may be the full `covid19` directory with
`participant-ID/collection-time/audio_file_cough.wav`; the scanner enters only participant
IDs listed in the Task-2 CSV. The raw adapter joins `uid/label/fold` to Android/iOS/Web metadata, normalises the released
age bands, expands multi-select `Symptoms` and `Medhistory`, derives platform, audits cough
audio and runs the frozen matching gate. It reads no embedding, prediction or model score.

Review:

```text
/DTA/cambridge_audit_output/public/data_gate.json
/DTA/cambridge_audit_output/public/DATA_GATE_REPORT.md
```

If the verdict is `NO_GO`, return `PUBLIC_RESULTS.zip` and stop. Do not relax the 100-pair,
0.12-SMD or 0.08 fine-balance criteria.

## 4. Canonical-table route (advanced)

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

## 5. Run the model-blind data gate for the advanced route

```bash
bash run_once.sh config.local.json gate
```

Review `output/.../public/data_gate.json`. A `NO_GO` is a valid final outcome. Return only
`output/.../public/PUBLIC_RESULTS.zip`; do not run a model.

## 6. Formal unlock after independent review

If and only if the gate is `GO`, send the aggregate gate JSON to the project lead. After
approval, set:

```json
"formal_unlock": "FROZEN_PROTOCOL_AND_DATA_GATE_GO"
```

Then run:

```bash
bash run_once.sh config.local.json formal
```

On a Slurm cluster, prefer the checked GPU wrapper after filling the same private config:

```bash
sbatch submit_formal.slurm /absolute/path/to/config.local.json
```

The Chinese handoff is in `CAMBRIDGE_FORMAL_NEXT_STEPS_ZH.txt`.

The command is resumable at completed artefacts. It refuses to overwrite an incompatible
cache, representation or training manifest.

## 7. Return contract

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

## 8. Troubleshooting rule

Technical failures may be fixed only when they do not change the frozen scientific
definition. Any proposed change to labels, modality, cohort, matching, thresholds, arms,
seeds, epochs or metrics must stop execution and be discussed before any model score is
read. Keep the failing log and current commit unchanged.
