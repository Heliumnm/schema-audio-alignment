#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAN=/mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json
SP=results/split_official.json
M=/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593

# frozen projector, official split
for c in all t1_qwen2; do
  CUDA_VISIBLE_DEVICES=0 python src/contrastive_align.py run \
    --audio_emb results/ast_feats_clean.npy --audio_index results/ast_feats_clean_index.json \
    --text_emb results/text_emb_bio/$c.npz --manifest $MAN --split_map $SP \
    --target wheeze --seeds 0 1 2 --device cuda \
    --prompt_emb results/class_prompts_bio.npz \
    --out results/off_frozen_wheeze_$c.json > logs/off_frozen_wheeze_$c.log 2>&1
  echo "frozen $c done"
done

# trainable encoder, official split
CUDA_VISIBLE_DEVICES=0 python src/clip_finetune.py train --inputs results/ast_inputs \
  --text_emb results/text_emb_bio/all.npz --manifest $MAN --split_map $SP --model $M \
  --target wheeze --seeds 0 1 2 --epochs 15 --patience 3 --batch 24 --lr 1e-5 \
  --out results/off_clip_wheeze_all.json > logs/off_clip_wheeze_all.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python src/clip_finetune.py train --inputs results/ast_inputs \
  --text_emb results/text_emb_bio/t1_qwen2.npz --manifest $MAN --split_map $SP --model $M \
  --target wheeze --seeds 0 1 2 --epochs 15 --patience 3 --batch 24 --lr 1e-5 \
  --out results/off_clip_wheeze_t1_qwen2.json > logs/off_clip_wheeze_t1.log 2>&1 &
wait
echo OFFICIAL_DONE
