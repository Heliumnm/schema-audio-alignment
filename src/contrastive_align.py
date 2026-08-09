"""
Two-tower contrastive alignment of respiratory audio to text  (experiments E1-E3).

Both towers are FROZEN; only a small MLP projection head is trained, exactly as in
RespiraMFM (App. D: 768 -> 1024 -> D, LayerNorm + ReLU + dropout 0.1, InfoNCE,
tau = 0.07, lr 1e-3, 500 epochs). Because nothing upstream is trained, both towers'
embeddings are computed ONCE and cached — so a full provenance sweep
(4 conditions x 3 seeds) is minutes, not hours.

The independent variable is the TEXT (see schema_text.py). Do not tune the
optimiser between conditions or the ablation stops being an ablation.

Reported metrics
----------------
  downstream  (PRIMARY)  linear probe on projected audio -> MCC / AUROC,
                         and zero-shot AUROC via class-prompt cosine.
  retrieval   (context)  R@1 audio->text. NEVER the headline: when the text is a
                         function of the audio this is high by construction. The
                         GAP between retrieval and downstream is the circularity
                         evidence this paper is built on.

    # 1. cache text embeddings for every provenance condition
    python src/contrastive_align.py encode \
        --schema_text results/schema_text.json --out results/text_emb

    # 2. train + evaluate one condition
    python src/contrastive_align.py run \
        --audio_emb results/opera_feats_clean.npy \
        --audio_index results/opera_index_clean.json \
        --text_emb results/text_emb/signal.npz \
        --manifest data/segments/manifest.json \
        --target crackle --seeds 0 1 2 --out results/align_signal.json
"""
import os, json, argparse, warnings
import numpy as np

TAU = 0.07
EPOCHS = 500
LR = 1e-3
HIDDEN = 1024
DROPOUT = 0.1


def patient_of(seg_id):
    """ICBHI segment ids look like 101_1b1_Al_sc_Meditron.wav -> patient 101."""
    return os.path.basename(seg_id).split("_")[0]


# ------------------------------------------------------------------ encode
def cmd_encode(args):
    import torch
    from transformers import AutoTokenizer, AutoModel

    data = json.load(open(args.schema_text))
    conditions = sorted({c for v in data.values() for c in v["text"]})
    print(f"{len(data)} records, conditions: {conditions}")

    tok = AutoTokenizer.from_pretrained(args.text_model)
    mdl = AutoModel.from_pretrained(args.text_model, torch_dtype=torch.float32).eval()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    mdl = mdl.to(dev)
    print(f"text tower: {args.text_model} on {dev}")

    os.makedirs(args.out, exist_ok=True)
    ids = list(data.keys())
    for cond in conditions:
        texts = [data[i]["text"][cond] for i in ids]
        embs = []
        for s in range(0, len(texts), args.batch):
            chunk = texts[s:s + args.batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt").to(dev)
            with torch.no_grad():
                h = mdl(**enc).last_hidden_state
            # mean-pool over non-padding tokens
            m = enc["attention_mask"].unsqueeze(-1).float()
            embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
            if (s // args.batch) % 20 == 0:
                print(f"  {cond}: {s}/{len(texts)}", flush=True)
        E = np.concatenate(embs).astype(np.float32)
        np.savez(os.path.join(args.out, f"{cond}.npz"), ids=np.array(ids), emb=E)
        print(f"  wrote {cond}.npz  {E.shape}")


# ------------------------------------------------------------------- align
def build_head(d_in, d_out, torch, nn):
    return nn.Sequential(
        nn.Linear(d_in, HIDDEN), nn.LayerNorm(HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, d_out),
    )


def info_nce(za, zt, torch, tau=TAU):
    za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    zt = zt / zt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    logits = za @ zt.T / tau
    tgt = torch.arange(len(za), device=za.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, tgt) +
                  torch.nn.functional.cross_entropy(logits.T, tgt))


def metrics(y, score, pred=None):
    from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
    if pred is None:
        pred = (score > np.median(score)).astype(int)
    out = {"mcc": float(matthews_corrcoef(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "pred_pos_rate": float(np.mean(pred))}
    try:
        out["auroc"] = float(roc_auc_score(y, score))
    except ValueError:
        out["auroc"] = float("nan")
    out["degenerate"] = bool(out["pred_pos_rate"] < 0.05 or out["pred_pos_rate"] > 0.95)
    return out



# ------------------------------------------------------------- zero-shot
# CLIP's actual selling point is not "better features" but classification from text
# prompts with no labelled test data. The linear probe above does not test that, so
# both variants are reported:
#   prompt     true zero-shot — hand-written pos/neg sentences in the training text's
#              own style (out-of-style prompts would test the text tower, not the
#              alignment)
#   prototype  mean TRAIN text embedding per class. Uses train labels, so it is an
#              upper bound on what this text space can express, not zero-shot.
CLASS_PROMPTS = {
    "wheeze": (
        "There is a high likelihood of wheeze, polyphonic, high-pitched, with high "
        "musical-band energy.",
        "There is a low likelihood of wheeze, with low musical-band energy.",
    ),
    "crackle": (
        "There is a high likelihood of crackles, with high transient sharpness "
        "suggesting coarse crackles.",
        "There is a low likelihood of crackles; transient sharpness is low.",
    ),
}


def zero_shot(Pa_te, Xt, y_tr, tr_mask, prompt_emb, np_):
    """Return (prompt_auroc, prototype_auroc); nan when a variant is unavailable."""
    from sklearn.metrics import roc_auc_score
    def unit(x):
        return x / np_.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)
    out = {}
    U = unit(Pa_te)
    if prompt_emb is not None:
        P = unit(prompt_emb)                       # (2, D): [positive, negative]
        out["prompt"] = U @ P[0] - U @ P[1]
    proto_pos = Xt[tr_mask][y_tr == 1].mean(0)
    proto_neg = Xt[tr_mask][y_tr == 0].mean(0)
    P2 = unit(np_.stack([proto_pos, proto_neg]))
    out["prototype"] = U @ P2[0] - U @ P2[1]
    return out


def cmd_run(args):
    import torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    man = json.load(open(args.manifest))
    meta = {os.path.basename(e.get("path", e["filename"])): e for e in man}

    A = np.load(args.audio_emb)
    aidx = json.load(open(args.audio_index))
    tz = np.load(args.text_emb, allow_pickle=True)
    tids, T = list(tz["ids"]), tz["emb"]
    tpos = {i: k for k, i in enumerate(tids)}

    # keep only ids present in audio, text and manifest
    ids = [i for i in tids if i in aidx and i in meta]
    print(f"paired: {len(ids)} / audio {len(aidx)} / text {len(tids)}")
    Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
    Xt = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)

    prompt_emb = None
    if args.prompt_emb and os.path.exists(args.prompt_emb):
        pz = np.load(args.prompt_emb, allow_pickle=True)
        pt = {str(k): v for k, v in zip(pz["targets"], pz["emb"])}
        if args.target in pt:
            prompt_emb = pt[args.target].astype(np.float32)
            print(f"class prompts loaded for {args.target}")

    lab = {"crackle": lambda e: int(e["label"] in ("crackle", "both")),
           "wheeze":  lambda e: int(e["label"] in ("wheeze", "both"))}[args.target]
    y = np.array([lab(meta[i]) for i in ids])
    smap = json.load(open(args.split_map)) if args.split_map else None
    if smap is not None:
        ids = [i for i in ids if i in smap]
        Xa = np.stack([A[aidx[i]] for i in ids]).astype(np.float32)
        Xt = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)
        y = np.array([lab(meta[i]) for i in ids])
        print(f"split_map: {os.path.basename(args.split_map)} -> {len(ids)} segments")
    split = np.array([(smap[i] if smap else meta[i]["split"]) for i in ids])
    pid = np.array([patient_of(i) for i in ids])

    tr, te = split == "train", split == "test"
    overlap = set(pid[tr]) & set(pid[te])
    assert not overlap, f"PATIENT LEAK between train and test: {sorted(overlap)[:5]}"
    print(f"train {tr.sum()} ({len(set(pid[tr]))} patients) | "
          f"test {te.sum()} ({len(set(pid[te]))} patients) | no overlap OK")
    print(f"target={args.target}  positives: train {y[tr].mean():.3f} test {y[te].mean():.3f}")

    runs = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        head = build_head(Xa.shape[1], Xt.shape[1], torch, nn).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.1)
        at = torch.tensor(Xa[tr], device=dev); tt = torch.tensor(Xt[tr], device=dev)

        head.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(at), device=dev)
            tot = 0.0
            for s in range(0, len(perm), args.batch):
                b = perm[s:s + args.batch]
                if len(b) < 2:           # InfoNCE needs negatives in-batch
                    continue
                loss = info_nce(head(at[b]), tt[b], torch)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            if (ep + 1) % 100 == 0:
                print(f"  seed {seed} epoch {ep+1}/{EPOCHS} loss {tot:.3f}", flush=True)

        head.eval()
        with torch.no_grad():
            Pa = head(torch.tensor(Xa, device=dev)).cpu().numpy()

        # --- context metric: retrieval (inflated under circular text by design)
        def unit(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        S = unit(Pa[te]) @ unit(Xt[te]).T
        r1 = float(np.mean(np.argmax(S, axis=1) == np.arange(S.shape[0])))

        # --- PRIMARY: linear probe on the projected audio representation
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Pa[tr], y[tr])
        sc = clf.predict_proba(Pa[te])[:, 1]
        m = metrics(y[te], sc, clf.predict(Pa[te]))

        # --- zero-shot from text (what contrastive alignment is actually for)
        from sklearn.metrics import roc_auc_score
        zs = zero_shot(Pa[te], Xt, y[tr], tr, prompt_emb, np)
        zsr = {}
        for k, score in zs.items():
            try:
                zsr[f"zs_{k}_auroc"] = float(roc_auc_score(y[te], score))
            except ValueError:
                zsr[f"zs_{k}_auroc"] = float("nan")

        if args.save_head:
            torch.save(head.state_dict(), f"{args.save_head}_seed{seed}.pt")

        runs.append({"seed": seed, "retrieval_r1": r1, **m, **zsr})
        print(f"  seed {seed}: MCC {m['mcc']:.3f}  AUROC {m['auroc']:.3f}  "
              f"zs_prompt {zsr.get('zs_prompt_auroc', float('nan')):.3f}  "
              f"zs_proto {zsr.get('zs_prototype_auroc', float('nan')):.3f}  "
              f"R@1 {r1:.3f}{'  [DEGENERATE]' if m['degenerate'] else ''}")

    agg = {k: {"mean": float(np.mean([r[k] for r in runs])),
               "sd": float(np.std([r[k] for r in runs]))}
           for k in ["mcc", "auroc", "macro_f1", "pred_pos_rate", "retrieval_r1",
                     "zs_prompt_auroc", "zs_prototype_auroc"] if k in runs[0]}
    out = {"text_emb": os.path.basename(args.text_emb), "target": args.target,
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "seeds": args.seeds, "runs": runs, "aggregate": agg}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)

    print(f"\n=== {os.path.basename(args.text_emb)} / {args.target} "
          f"({len(args.seeds)} seeds) ===")
    for k in ["mcc", "auroc", "zs_prompt_auroc", "zs_prototype_auroc", "retrieval_r1"]:
        if k not in agg: continue
        print(f"  {k:<14} {agg[k]['mean']:.3f} ± {agg[k]['sd']:.3f}")
    print("\nReport downstream MCC/AUROC as the result; retrieval only as context.")
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="cache frozen text embeddings per condition")
    e.add_argument("--schema_text", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--text_model", default="bert-base-uncased",
                   help="frozen text tower; bert-base is cached on the server "
                        "and runs fine on CPU, avoiding the busy GPUs")
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--max_len", type=int, default=256)

    r = sub.add_parser("run", help="train projection head and evaluate")
    r.add_argument("--audio_emb", required=True); r.add_argument("--audio_index", required=True)
    r.add_argument("--text_emb", required=True); r.add_argument("--manifest", required=True)
    r.add_argument("--target", default="crackle", choices=["crackle", "wheeze"])
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--out", default="")
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--prompt_emb", default="", help="npz of encoded class prompts")
    r.add_argument("--save_head", default="", help="prefix to save trained heads")

    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    (cmd_encode if args.cmd == "encode" else cmd_run)(args)


if __name__ == "__main__":
    main()
