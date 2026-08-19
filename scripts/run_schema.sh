source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate liuhe 2>/dev/null || conda activate base
cd /mnt/hd/data_heliu/schema_align
python src/schema_text.py   --manifest /mnt/hd/data_heliu/icbhi_pathology_fidelity/data/segments/manifest.json   --feats /mnt/hd/data_heliu/icbhi_pathology_fidelity/results/opera_feats_clean.npy   --index /mnt/hd/data_heliu/icbhi_pathology_fidelity/results/opera_index_clean.json   --crackle_probe /mnt/hd/data_heliu/icbhi_pathology_fidelity/results/judges_opera/judge_A_crackle.pt   --wheeze_probe  /mnt/hd/data_heliu/icbhi_pathology_fidelity/results/judges_opera/judge_A_wheeze.pt   --audio_root /mnt/hd/data_heliu/icbhi_pathology_fidelity --out results/schema_text.json --explain --show 3
