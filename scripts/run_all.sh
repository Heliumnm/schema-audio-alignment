#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate base
cd /mnt/hd/data_heliu/schema_align
export OMP_NUM_THREADS=16
R=/mnt/hd/data_heliu/icbhi_pathology_fidelity
python src/contrastive_align.py run \
  --audio_emb $R/results/opera_feats_clean.npy \
  --audio_index $R/results/opera_index_clean.json \
  --text_emb results/text_emb/all.npz \
  --manifest $R/data/segments/manifest.json \
  --target wheeze --seeds 0 1 2 --device cpu \
  --out results/align_wheeze_all.json
