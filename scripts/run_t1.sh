#!/usr/bin/env bash
# T1 free-form description generation. Writes a pidfile so it can be stopped
# without pkill -f, which self-matches any shell whose command line mentions the
# script name (this cost two failed restarts).
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
echo $$ > logs/t1.pid
export CUDA_VISIBLE_DEVICES=1
exec python src/stetholm_describe.py \
  --manifest /mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json \
  --audio_root /mnt/hd/data_heliu/icbhi_pathology_fidelity \
  --out results/t1_qwen2.jsonl --backend qwen2 --load_4bit --batch 8 \
  --model_id /mnt/hd/data_ycyang/models/Qwen2-Audio-7B-Instruct
