"""Acoustic-only arms: drop the recording block entirely.

v2 puts continuous duration_s into the text, and duration alone reaches AUROC 0.644
on crackle. Any v2 gain could be that confound arriving through the text rather than
anything structural, so the same two arms are re-rendered over the acoustic block
only and encoded identically.
"""
import os, sys, json, numpy as np, torch
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "src")
from schema_v2 import fields, render
from transformers import AutoTokenizer, AutoModel

M = "/mnt/hd/data_heliu/hf_models/Bio_ClinicalBERT"
D = json.load(open("results/schema_text.json"))
ids = list(D)
tok = AutoTokenizer.from_pretrained(M); mdl = AutoModel.from_pretrained(M).eval().cuda()
os.makedirs("results/text_emb_v2", exist_ok=True)
for arm in ["matched_template", "matched_serialized"]:
    texts = [render(fields(D[i]), arm, blocks=("acoustic",)) for i in ids]
    E = []
    for s in range(0, len(texts), 64):
        e = tok(texts[s:s+64], padding=True, truncation=True, max_length=384,
                return_tensors="pt").to("cuda")
        with torch.no_grad(): h = mdl(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        E.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
    E = np.concatenate(E).astype(np.float32)
    np.savez(f"results/text_emb_v2/{arm}_acoustic.npz", ids=np.array(ids), emb=E)
    print(f"{arm}_acoustic: {E.shape}  unique {len(set(texts))}", flush=True)
    print("  e.g.", texts[0][:170])
