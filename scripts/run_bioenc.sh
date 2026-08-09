source ~/anaconda3/etc/profile.d/conda.sh; conda activate qwen-audio
cd /mnt/hd/data_heliu/schema_align
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
M=/mnt/hd/data_heliu/hf_models/Bio_ClinicalBERT
python src/contrastive_align.py encode --schema_text results/schema_text.json \
  --out results/text_emb_bio --text_model $M --device cuda --batch 128
python - <<PY
import json, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
M="/mnt/hd/data_heliu/hf_models/Bio_ClinicalBERT"
rows=[json.loads(l) for l in open("results/t1_qwen2.jsonl")]
tok=AutoTokenizer.from_pretrained(M); mdl=AutoModel.from_pretrained(M).eval().cuda()
ids=[r["id"] for r in rows]; texts=[r["text"] for r in rows]; E=[]
for s in range(0,len(texts),128):
    enc=tok(texts[s:s+128],padding=True,truncation=True,max_length=256,return_tensors="pt").to("cuda")
    with torch.no_grad(): h=mdl(**enc).last_hidden_state
    m=enc["attention_mask"].unsqueeze(-1).float()
    E.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
E=np.concatenate(E).astype(np.float32)
np.savez("results/text_emb_bio/t1_qwen2.npz", ids=np.array(ids), emb=E)
print("t1_qwen2.npz", E.shape)
PY
