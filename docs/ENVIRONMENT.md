# Getting weights onto the offline server

The GPU box has no route to `huggingface.co`. Most of this project's setup cost went
into working around that, usually the slow way. The routes below are ordered by how
much time they save; check them in order before starting any download.

## Verified network paths (2026-08-07)

| from → to | result |
|---|---|
| server → `hf-mirror.com` | ✅ **200, 0.44 s** |
| server → `hf-mirror.com`, **gated** repo | ❌ the mirror does not proxy gated repos |
| server → `huggingface.co` | ❌ unreachable |
| Mac → `huggingface.co` | ✅ but **needs a token**, see below |
| Mac → `zenodo.org` | ⚠️ intermittent — timed out entirely for one stretch |

## Route 1 — ungated weights: download **on the server** via the mirror

This is the fast path and it was missed for most of the project. AST,
Bio_ClinicalBERT, CLAP, AudioMAE and similar need no local hop at all:

```bash
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
python -c "from huggingface_hub import snapshot_download; snapshot_download('MIT/ast-finetuned-audioset-10-10-0.4593')"
```

Verified working against `bert-base-uncased`. Note the pipeline scripts export
`HF_HUB_OFFLINE=1`; unset it for the download, set it again for runs.

## Route 2 — gated weights (MedGemma, Llama, …): Mac → external drive → rsync

The mirror will not serve these, so they go through the Mac. Two prerequisites that
each cost a round trip when missed:

1. **Accept the licence** on the model page (`google/medgemma-4b-it` is
   `gated=auto`, so it takes effect immediately).
2. **Log in on the Mac.** Accepting the licence is not enough — the download
   authenticates with a token, and without one it fails `GatedRepoError 401` in a
   way that looks identical to not having accepted:

   ```bash
   /Users/heliumnm/anaconda3/bin/huggingface-cli login
   ```

**Send both the cache and the destination to the external drive.** The system disk
sits at ~12 GB free, and `snapshot_download` fills its cache before materialising
`local_dir` — targeting only `local_dir` still overflows the boot volume:

```python
os.environ["HF_HOME"] = "/Volumes/Seagate/model/hf_cache"
snapshot_download(repo_id, local_dir="/Volumes/Seagate/model/<name>")
```

Storage on hand: `/Volumes/Seagate` 909 GB free; Mac system disk ~12 GB;
server `/mnt/hd` ~3 TB.

## Model-specific notes

**StethoLM (`askyishan/StethoLM`)** ships **only `stetholm_adapter.pt` (696 MB)** —
no config, no base weights. Loading it with `AutoModelForCausalLM.from_pretrained`
cannot work. Running it needs, per the authors' repo:

- `google/medgemma-4b-it` (8.6 GB, gated) as the LM backbone,
- a **COLA audio encoder whose source the repo does not state**, and
- their `predict.py` to assemble the three.

**Audio Flamingo 3 (`nvidia/audio-flamingo-3-hf`)** is 34.5 GB and ungated — too
large for the Mac system disk, fine on the external drive or straight to the server
via the mirror.

**Qwen2-Audio-7B-Instruct** is already on the server at
`/mnt/hd/data_ycyang/models/Qwen2-Audio-7B-Instruct` (complete, `model_type:
qwen2_audio`). The copy under `data_heliu/from_home/SpeechLLM/.../
Qwen2-Audio-7B-Instruct-local` has **no `config.json`** and does not load.

## Environments on the server

`qwen-audio` (transformers 4.56, torch 2.8) runs everything here. `base` is fine for
the sklearn/numpy analysis steps and has 112 cores — CPU is the uncontended
resource on this box, so text encoding and the frozen-projector sweeps run there
rather than queueing for a GPU.
