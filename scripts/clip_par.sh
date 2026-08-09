#!/usr/bin/env bash
# Remaining trainable-encoder conditions, one per GPU in parallel.
# The decisive comparison is dataset (27 distinct texts) vs all (3,552): if text
# resolution matters at all, it should show up now that the encoder can reorganise.
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593
MAN=/mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json

run() {  # run <gpu> <target> <cond>
  CUDA_VISIBLE_DEVICES=$1 python src/clip_finetune.py train \
    --inputs results/ast_inputs --text_emb results/text_emb_bio/$3.npz \
    --manifest $MAN --model $M --target $2 --seeds 0 1 2 \
    --epochs 15 --patience 3 --batch 24 --lr 1e-5 \
    --out results/clip_$2_$3.json > logs/clip_$2_$3.log 2>&1
  echo "done $2/$3"
}

run 0 wheeze dataset &
run 1 wheeze t1_qwen2 &
run 2 crackle all &
wait
run 0 wheeze model &
run 1 wheeze signal &
wait
echo "CLIP PARALLEL DONE"
