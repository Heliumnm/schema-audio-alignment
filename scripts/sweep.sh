#!/usr/bin/env bash
# Full matrix: 2 audio towers x 2 text towers x 5 text conditions x 2 targets.
# AST + Bio_ClinicalBERT is the configuration EXPERIMENT_PLAN section 3 specifies;
# OPERA-CT + bert-base was the first (circular) sweep and is kept for comparison.
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=/mnt/hd/data_heliu/icbhi_pathology_fidelity
M=$R/data/segments/manifest.json

for tgt in wheeze crackle; do
  for c in dataset signal model all t1_qwen2; do
    o=results/ast_bio_${tgt}_${c}.json
    [ -f "$o" ] && { echo "skip $o"; continue; }
    echo "########## AST+Bio | $tgt | $c ##########"
    python src/contrastive_align.py run \
      --audio_emb results/ast_feats_clean.npy \
      --audio_index results/ast_feats_clean_index.json \
      --text_emb results/text_emb_bio/$c.npz \
      --manifest $M --target $tgt --seeds 0 1 2 --device cuda \
      --out $o 2>&1 | grep -E "^  seed .*: MCC|paired|train .*test|^===|^  (mcc|auroc|retrieval)"
  done
done
echo "SWEEP DONE"
