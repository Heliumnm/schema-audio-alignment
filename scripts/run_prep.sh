source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=24
python src/clip_finetune.py prep \
  --manifest /mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json \
  --audio_root /mnt/hd/data_heliu/icbhi_pathology_fidelity \
  --model /mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593 \
  --out results/ast_inputs --batch 32
