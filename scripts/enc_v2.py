import os, json, numpy as np, torch
os.environ["HF_HUB_OFFLINE"]="1"
from transformers import AutoTokenizer, AutoModel
M="/mnt/hd/data_heliu/hf_models/Bio_ClinicalBERT"
D=json.load(open("results/schema_v2.json"))
ids=list(D)
tok=AutoTokenizer.from_pretrained(M); mdl=AutoModel.from_pretrained(M).eval().cuda()
os.makedirs("results/text_emb_v2", exist_ok=True)
for arm in ["matched_template","matched_serialized"]:
    texts=[D[i][arm] for i in ids]; E=[]
    for s in range(0,len(texts),64):
        e=tok(texts[s:s+64],padding=True,truncation=True,max_length=384,return_tensors="pt").to("cuda")
        with torch.no_grad(): h=mdl(**e).last_hidden_state
        m=e["attention_mask"].unsqueeze(-1).float()
        E.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).cpu().numpy())
        if (s//64)%25==0: print(f"  {arm}: {s}/{len(texts)}", flush=True)
    E=np.concatenate(E).astype(np.float32)
    np.savez(f"results/text_emb_v2/{arm}.npz", ids=np.array(ids), emb=E)
    L=[len(tok(t)["input_ids"]) for t in texts[:500]]
    print(f"{arm}: {E.shape}  tokens mean {np.mean(L):.0f} max {max(L)}  "
          f"unique texts {len(set(texts))}", flush=True)
