#!/usr/bin/env bash
# Does the negative count explain why trainable AST still loses to raw features?
# Everything except --queue_size is held fixed against the runs already in the repo,
# so the delta is attributable.
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
MAN=/mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json
SP=results/split_official.json
M=/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593
PR=results/class_prompts_bio.npz

run() {  # run <gpu> <cond> <queue>
  CUDA_VISIBLE_DEVICES=$1 python src/clip_finetune.py train \
    --inputs results/ast_inputs --text_emb results/text_emb_bio/$2.npz \
    --manifest $MAN --split_map $SP --model $M --prompt_emb $PR \
    --target wheeze --seeds 0 1 2 --epochs 15 --patience 3 --batch 24 --lr 1e-5 \
    --queue_size $3 --out results/q${3}_wheeze_$2.json > logs/q${3}_wheeze_$2.log 2>&1
  echo "done queue=$3 cond=$2"
}

run 0 all      4096 &
run 1 t1_qwen2 4096 &
run 2 all      1024 &
wait
echo QUEUE_SWEEP_DONE
