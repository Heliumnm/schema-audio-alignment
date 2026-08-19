# Dataset notes (verified, not assumed)

Everything here was checked against the downloaded files, not read off a paper.

## KAUH — downloaded 2026-08-06, 336 wav, 45 MB

`data.mendeley.com/datasets/jwyy9np4gv/3` → `AudioFiles/` (flat, no subdirs).

Filename format, after collapsing whitespace:

```
{filter}P{id}_{diagnosis},{sound_type},{location},{age},{sex}.wav
BP33_Asthma,E W,P R M,43,F.wav
```

### Three traps

**1. 336 files = 112 patients × 3 filter settings — not 336 recordings.**

Verified exactly: `{'B': 112, 'D': 112, 'E': 112}`, and every patient has exactly 3
files. B/D/E are the Bell / Diaphragm / Extended export filters of the recording
software — **three filtered views of the same physical auscultation**.

> Treating them as independent samples inflates n by 3× *and* puts the same
> recording on both sides of any random split. **Split by patient id, always.**

**2. Filenames contain spaces.** `find | xargs basename` word-splits them and makes
one name look like several. Use `find -print0 | xargs -0`, or quote every path.
(They do *not* contain newlines — the multi-line output from a naive `xargs` is an
artefact of the splitting, not of the names.)

**3. Diagnosis strings are inconsistently cased and must be normalised**, or the
same class gets counted twice:

| normalised | raw variants | patients |
|---|---|---:|
| normal | `N` | 35 |
| asthma | `Asthma` (17) + `asthma` (15) | 32 |
| heart failure | `heart failure` (15) + `Heart Failure` (3) | 18 |
| COPD | `COPD` (8) + `copd` (1) | 9 |
| pneumonia | `pneumonia` | 5 |
| lung fibrosis | `Lung Fibrosis` | 4 |
| bronchiectasis | `BRON` | 3 |
| pleural effusion | `Plueral Effusion` *(sic — dataset typo)* | 2 |

Plus three comorbid entries that need an explicit policy: `Heart Failure + COPD`
(2), `Asthma and lung fibrosis` (1), `Heart Failure + Lung Fibrosis` (1).

### RespiraMFM's KAUH tasks reproduce exactly

Their zero-shot test sizes are `{disease patients} + {35 normals}` × 3 filters:

| task | disease | patients | (d + 35) × 3 | RespiraMFM reports |
|---|---|---:|---:|---:|
| T7 | COPD | 9 | **132** | 132 ✅ |
| T8 | asthma | 32 | **201** | 201 ✅ |
| T9 | pneumonia | 5 | **120** | 120 ✅ |

All three match to the file. So RespiraMFM **does** use the three filter versions as
separate samples, disease-vs-normal, all patients. We can rebuild their exact
evaluation sets — our E4 zero-shot numbers will be directly comparable to their
Table 3, with no guesswork.

*(Note this also means their reported n's carry the 3× filter duplication. Fine for
comparability; worth a sentence in our limitations.)*

## COUGHVID — not yet downloaded

Zenodo record `4048312`, API verified: **one file**, `public_dataset.zip`, 951.4 MB,
`links.self` present. The download script resolves it through the API rather than
hard-coding a filename.

## Coswara — not yet downloaded

`github.com/iiscleap/Coswara-Data`, split tar archives per recording date plus an
`extract_data.py` that joins them. ~8 GB.

**Question its inclusion.** Under EXPERIMENT_PLAN Option A the main experiments run
on ICBHI alone, so Coswara's original role (a metadata-rich corpus for T0) is gone.
What remains is zero-shot transfer — but Coswara is crowd-sourced cough and speech,
not stethoscope auscultation, so it is a *domain* shift as much as a corpus shift.
**KAUH is the natural zero-shot target for a lung-auscultation schema.** Decide
before spending 8 GB and a day of transfer.

## ICBHI

Already on the server. 6,898 cycles, 126 patients, official 60/40 patient-independent
split (train pid ≤ 160). The only corpus here with **event-level crackle/wheeze
annotation**, which is why the schema conditions are built on it.

## Local disk constraint

The Mac had **14 GB free** at download time. The server has no outbound network, so
everything is fetched locally and rsynced. Use the per-dataset push:

```bash
bash scripts/download_datasets.sh --only coughvid \
  --push heliu@202.38.72.99:/mnt/hd/data_heliu/resp_datasets/ --clean
```

`--clean` deletes each local copy after a successful rsync, so peak local usage is
bounded by the largest single corpus instead of their sum.
