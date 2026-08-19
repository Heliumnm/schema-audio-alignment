# Runbook — machines, the edit/run loop, and the traps

`ENVIRONMENT.md` covers how to get model weights onto the offline box. This covers
everything else needed to run an experiment here from a cold start.

**No credentials appear in this repository.** GitHub and Hugging Face tokens were typed
directly into `gh auth login` / `huggingface-cli login` on the machine that needed them
and were never written to a file, a script, or a commit. Both hosts are already
authenticated; if a token ever expires, re-run those commands interactively rather than
storing anything.

## Machines

| | Mac (authoritative) | GPU server |
|---|---|---|
| address | local | `ssh heliu@202.38.72.99` (host `ahu`, key auth, no password) |
| repo | `/Users/heliumnm/Downloads/euro-grad-apply-main/schema_audio_alignment` | `/mnt/hd/data_heliu/schema_align` |
| role | git, editing, write-up | execution only |
| GitHub | `Heliumnm/schema-audio-alignment` (private), branch `main` | — |
| disk | system ~12 GB free; `/Volumes/Seagate` 909 GB | `/mnt/hd` ~3 TB |
| GPUs | — | 6 × RTX 4090, 24 GB each, **shared with other users** |

### The server is a run target, not a repository

`/mnt/hd/data_heliu/schema_align/.git` exists and **is a trap**:

* it has **no remote**, so `git pull` there cannot work;
* it sits at an old commit (`16bd636`) with ~39 modified/untracked paths;
* `results/` — every output this project has produced — is untracked there.

**Never run `git checkout`, `git reset`, `git stash` or `git clean` on the server.** They
would revert or delete files that only exist because they were rsync'd there. Treat that
`.git` as dead. The Mac copy is authoritative and is the only one that pushes.

## The loop

Edit on the Mac, sync the one file you changed, run over SSH, pull results back, commit
on the Mac.

```bash
cd /Users/heliumnm/Downloads/euro-grad-apply-main/schema_audio_alignment
rsync -q src/<script>.py heliu@202.38.72.99:/mnt/hd/data_heliu/schema_align/src/
```

```bash
ssh heliu@202.38.72.99 'cd /mnt/hd/data_heliu/schema_align && source ~/anaconda3/etc/profile.d/conda.sh && conda activate qwen-audio && R=/mnt/hd/data_heliu/icbhi_pathology_fidelity && CUDA_VISIBLE_DEVICES=0 python src/<script>.py --manifest $R/data/segments/manifest.json --audio_root $R'
```

```bash
rsync -q heliu@202.38.72.99:/mnt/hd/data_heliu/schema_align/results/<file> results/
```

Pick a free GPU first — three of the six are usually busy with someone else's job:

```bash
ssh heliu@202.38.72.99 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```

## Fixed paths on the server

| what | path |
|---|---|
| ICBHI root | `/mnt/hd/data_heliu/icbhi_pathology_fidelity` |
| segment manifest | `$R/data/segments/manifest.json` — 6,898 per-cycle segments |
| official split map | `results/split_official.json` (repo-relative), built by `src/official_split.py --strict` |
| AST weights | `/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593` |
| Qwen2-Audio | `/mnt/hd/data_ycyang/models/Qwen2-Audio-7B-Instruct` |
| conda env | `qwen-audio` (transformers 4.56, torch 2.8); `base` for sklearn/numpy on 112 cores |

## Reproducing the current results

All three run on the **official TRAIN split only** and touch no test data.

```bash
python src/grounding_v21.py --manifest $R/data/segments/manifest.json --audio_root $R --n 700
```

```bash
python src/grounding_train.py --manifest $R/data/segments/manifest.json --audio_root $R --n 700
```

Both are fully seeded and deterministic — `grounding_train.py` reproduces
0.569 ± 0.043 exactly on re-run, which is the check that the configuration is frozen.
Outputs: `results/grounding_v21_feasibility.json`, `grounding_v21_manifest.json`,
`grounding_v21_train.json`, `grounding_v21_preds.npz` (per-example, per-seed win vectors,
shape `(11 arms, 3 seeds, 452 dev examples)`).

Rough cost: AST feature extraction for 695 clips ≈ 2.6 GB of patch grids and a couple of
minutes on one 4090; the control suite is 8 arms × 3 seeds and dominates wall time.

## Traps that have cost a re-run

**`grep -vE "    "` eats your results.** That pattern matches any line *containing* four
consecutive spaces, which is every column-aligned table this project prints. It silently
swallowed a whole feasibility-gate readout and cost a full re-run. Filter with `^\s{4}`
or on keywords, and prefer reading the JSON output over parsing stdout.

**`pkill -f <script>` kills your own shell.** The pattern matches the SSH command line
that issued it. Two long restarts died this way. Write a pidfile and kill by PID.

**System `python3` on the server has no numpy.** Always
`source ~/anaconda3/etc/profile.d/conda.sh && conda activate qwen-audio` first, including
for one-line analysis snippets.

**`HF_HOME` moves token lookup, not just the cache.** Pointing it at the external drive
produced a `GatedRepoError 401` immediately after a successful login. Use `HF_HUB_CACHE`
when you only mean to relocate the cache. (`ENVIRONMENT.md` recommends `HF_HOME` for the
Mac-side gated download — that works only if you log in *after* setting it.)

**Nested ndarrays break `json.dump`.** A one-level comprehension is not enough; write an
explicit freezer that converts bounds to ints, as `grounding_v21.py:freeze` does.

**Blind textual patching across files.** A `--split_map` flag was once patched into two
structurally different scripts by string replacement; in one it referenced variables that
did not exist there, and in both the argparse line was never registered. Verify with
`--help` after any cross-file edit.

## Protocol reminders that outrank convenience

* patient-level splits, asserted in code — never segment-level;
* the official ICBHI partition is a per-recording file, **not** `patient_id ≤ 160`;
* three seeds, mean ± sd, and the do-nothing baseline in the same table;
* a control that costs an hour is cheaper than the work a wrong result causes — six of
  this project's eight retractions were caught that way, and two were not.
