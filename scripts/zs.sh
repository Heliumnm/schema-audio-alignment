#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# encode the class prompts with the SAME frozen text tower used for training
python - <<PY
import numpy as np, torch, sys
sys.path.insert(0,"src")
from contrastive_align import CLASS_PROMPTS
from transformers import AutoTokenizer, AutoModel
M="/mnt/hd/data_heliu/hf_models/Bio_ClinicalBERT"
tok=AutoTokenizer.from_pretrained(M); mdl=AutoModel.from_pretrained(M).eval().cuda()
tg,E=[],[]
for t,(pos,neg) in CLASS_PROMPTS.items():
    enc=tok([pos,neg],padding=True,truncation=True,max_length=256,return_tensors="pt").to("cuda")
    with torch.no_grad(): h=mdl(**enc).last_hidden_state
    m=enc["attention_mask"].unsqueeze(-1).float()
    E.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy()); tg.append(t)
np.savez("results/class_prompts_bio.npz", targets=np.array(tg), emb=np.stack(E))
print("prompts:", np.stack(E).shape, tg)
PY
R=/mnt/hd/data_heliu/icbhi_pathology_fidelity
for tgt in wheeze crackle; do
  for c in dataset signal model all t1_qwen2; do
    python src/contrastive_align.py run \
      --audio_emb results/ast_feats_clean.npy \
      --audio_index results/ast_feats_clean_index.json \
      --text_emb results/text_emb_bio/$c.npz \
      --manifest $R/data/segments/manifest.json \
      --target $tgt --seeds 0 1 2 --device cuda \
      --prompt_emb results/class_prompts_bio.npz \
      --out results/zs_${tgt}_${c}.json 2>&1 | grep -E "^  seed .*: MCC"
    echo "--- $tgt / $c done"
  done
done
echo "ZS SWEEP DONE"
