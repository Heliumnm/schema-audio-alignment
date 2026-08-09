source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python src/clip_finetune.py train \
  --inputs results/ast_inputs \
  --text_emb results/text_emb_bio/all.npz \
  --manifest /mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json \
  --model /mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593 \
  --target wheeze --seeds 0 1 2 --epochs 15 --patience 3 --batch 24 --lr 1e-5 \
  --out results/clip_wheeze_all.json
