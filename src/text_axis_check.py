"""Text-axis feasibility check, corrected. Freezes the text cache protocol.

Fixed decisions, ours and recorded as ours because the paper states none of them:

    layer        last hidden layer
    pooling      MASK-AWARE MEAN over valid tokens. The official source encodes one
                 context at a time, so its `padding=True` adds no padding and its plain
                 last_hidden_state.mean(dim=1) already equals a valid-token mean. Batching
                 the cache preserves that semantics rather than changing it.
    padding      RIGHT. Not left: the last-token index would be wrong under left padding.
    max length   the true maximum over ALL unique texts, checked, not sampled
    batch        fixed size, the final batch topped up with placeholder texts so every
                 batch has the same shape
    precision    bfloat16, recorded

Cache protocol, and the reason it is enough: **each unique text is encoded once** and every
participant carrying that profile reuses the same vector. So identical profiles are
identical by construction downstream, and the exact-text multi-positive equivalence
(L_multi = L_single − log K) holds regardless of what batching does. Bit-stability is then
verified directly: encode the unique texts, shuffle their order, encode again, sort back by
text id, and require the two caches to be bit-identical with matching hashes.

Three statistics from the first version are corrected or withdrawn:

* **single-field similarity** — the first version sampled random pairs and hit exactly one
  one-field-apart pair in 3,000 draws. Pairs are now constructed explicitly and reported
  per field. Report it as **1 − cosine** with several digits, never as a rounded 1.0000:
  no pair here is mathematically identical (0 pairs at 1 − cos < 1e-12). This is a
  **descriptive diagnostic, not a gate**, and the right name for it is high anisotropy —
  geometric compression — not a defect. If `correct` and `within_label_shuffled` later come
  out close, this may be listed among the possible reasons; it may not be presented as the
  explanation.
* **"duplicate-aware top-1"** — 0.6465 was not retrieval performance. It says roughly 65%
  of the sampled participants have someone else in the sample with an identical profile,
  and the naive 0.0000 was forced by excluding self. Both are reported as *ambiguity*
  statistics. Real audio→text retrieval belongs to the projector smoke test.
* **loss floor** — 0.51 / 0.85 / 1.31 nats are withdrawn. They were log(E[K]); the floor
  the sampler actually imposes is E[log K], computed here by simulating the sampler.

Field decodability is now split **by unique text**, so identical profiles never straddle
train and val, and it is evaluated on full field values rather than one binary question
per field.

    python src/text_axis_check.py --model /mnt/hd/data_heliu/hf_models/phi-2
"""
import os, json, argparse, hashlib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metadata_text import UNIT, FIELDS, BINARY_MAP, build, expected_log_K

BATCH = 16
REVISION = "810d367871c1d460086d9f82db8696f2e0a0fcd0"
PLACEHOLDER = "[PAD_TEXT]"


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="results/text_axis_check.json")
    ap.add_argument("--texts_out", default="results/metadata_texts.csv")
    ap.add_argument("--emb_out", default="results/metadata_text_embeddings.npz")
    args = ap.parse_args()

    d, sym, audit = build(os.path.join(args.data, "participant_metadata.csv"), args.cohort,
                          os.path.join(args.data, "train_test_splits.csv"))
    print("--- value audit (unknown values would have raised) ---")
    for k, v in audit.items():
        flag = "  UNKNOWN=" + str(v["unknown"]) if v["unknown"] else ""
        print(f"  {k:<34s} {len(v['values'])} values, {v['n_missing']:5d} missing{flag}")
    d[[UNIT, "splits", "text"]].to_csv(args.texts_out, index=False)
    uniq = sorted(d.text.unique())
    tid = {t: i for i, t in enumerate(uniq)}
    print(f"\n{len(d)} participants, {len(uniq)} distinct schema texts")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    DTYPE = torch.bfloat16
    model = AutoModel.from_pretrained(args.model, dtype=DTYPE).eval().cuda()
    for q in model.parameters():
        q.requires_grad_(False)

    lens = [len(tok(t)["input_ids"]) for t in uniq]        # ALL of them, not a sample
    MAX_LEN = 200          # frozen. The official source sets 125, which is incompatible
                           # with OUR long schema texts. That is a statement about our
                           # texts, not a defect in RespiraMFM: their own contexts may
                           # well fit inside 125 and we have not measured them.
    print(f"token length over all {len(uniq)} texts: min {min(lens)} max {max(lens)}; "
          f"MAX_LEN {MAX_LEN} (official 125 would truncate {sum(l > 125 for l in lens)} "
          f"of {len(uniq)})")
    assert max(lens) <= MAX_LEN, "raise MAX_LEN before freezing"

    @torch.no_grad()
    def encode(texts):
        """Fixed batch shape: fixed size, fixed length, final batch topped up."""
        n = len(texts)
        pad_n = (-n) % BATCH
        padded_list = list(texts) + [PLACEHOLDER] * pad_n
        out = []
        for i in range(0, len(padded_list), BATCH):
            b = tok(padded_list[i:i+BATCH], return_tensors="pt", padding="max_length",
                    truncation=True, max_length=MAX_LEN).to("cuda")
            h = model(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())   # mask-aware mean
        return np.concatenate(out)[:n]

    U = encode(uniq)
    print(f"unique-text embeddings {U.shape}")

    # ---- 1. cache stability under a shuffled encoding order
    perm = np.random.RandomState(0).permutation(len(uniq))
    U2 = encode([uniq[i] for i in perm])
    back = np.empty_like(U2); back[perm] = U2
    same = bool(np.array_equal(U, back))
    h1 = hashlib.sha256(U.tobytes()).hexdigest()[:16]
    h2 = hashlib.sha256(back.tobytes()).hexdigest()[:16]
    print(f"\n1. cache stability under shuffled order: bit-identical {same}   "
          f"hash {h1} vs {h2}")
    if not same:
        print(f"   max abs difference {np.abs(U - back).max():.3e}  "
              f"(fall back to BATCH=1 if this is not zero)")
    res = {"n_participants": int(len(d)), "n_unique_texts": len(uniq),
           "max_token_len": MAX_LEN, "batch": BATCH, "padding_side": "right",
           "layer": "last_hidden_state", "pooling": "mask_aware_mean",
           "dtype": str(DTYPE), "revision": REVISION,
           "cache_bit_identical_shuffled": same, "cache_sha256_16": h1,
           "value_audit": audit}

    E = U[[tid[t] for t in d.text]]

    # ---- 2. field decodability, split BY UNIQUE TEXT so profiles never straddle
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score
    ud = d.drop_duplicates("text").set_index("text").loc[uniq].reset_index()
    rs = np.random.RandomState(1)
    m = rs.rand(len(uniq)) < 0.7
    print(f"\n2. field decodability, unique texts split {int(m.sum())}/{int((~m).sum())} "
          f"(identical profiles cannot straddle)")
    res["field_decodability"] = {}
    all_fields = FIELDS + [(c.replace("symptom_", "").upper(), c, BINARY_MAP) for c in sym]
    for tag, col, _ in all_fields:
        yy = ud[col].astype(str).fillna("[MISSING]")
        if yy.nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(U[m], yy[m])
        pred = clf.predict(U[~m])
        acc = float(accuracy_score(yy[~m], pred))
        entry = {"n_classes": int(yy.nunique()), "accuracy": acc}
        if yy.nunique() == 2:
            pos = sorted(yy.unique())[-1]
            entry["auroc"] = float(roc_auc_score((yy[~m] == pos).astype(int),
                                                 clf.predict_proba(U[~m])[:, list(clf.classes_).index(pos)]))
        res["field_decodability"][tag] = entry
        extra = f"  AUROC {entry['auroc']:.4f}" if "auroc" in entry else ""
        print(f"   {tag:<34s} {entry['n_classes']} classes  acc {acc:.4f}{extra}")

    # ---- 3. one-field-apart similarity, constructed explicitly, per field. DIAGNOSTIC.
    Un = U / np.linalg.norm(U, axis=1, keepdims=True)
    toks = np.array([t.split(" ") for t in uniq])
    print("\n3. one-field-apart cosine similarity, constructed per field "
          "(descriptive diagnostic, not a gate)")
    res["one_field_similarity"] = {}
    rs = np.random.RandomState(2)
    ri, rj = rs.randint(0, len(uniq), 5000), rs.randint(0, len(uniq), 5000)
    rand_sim = float(np.mean(np.sum(Un[ri] * Un[rj], 1)))
    for pos, (tag, _, _) in enumerate(all_fields):
        key = np.array([" ".join(np.delete(row, pos)) for row in toks])
        srt = np.argsort(key, kind="stable")
        pairs = []
        for a, b in zip(srt[:-1], srt[1:]):
            if key[a] == key[b] and toks[a][pos] != toks[b][pos]:
                pairs.append((a, b))
        if len(pairs) < 20:
            continue
        pa = np.array(pairs)
        sim = float(np.mean(np.sum(Un[pa[:, 0]] * Un[pa[:, 1]], 1)))
        res["one_field_similarity"][tag] = {"n_pairs": len(pairs), "sim": sim}
        print(f"   {tag:<34s} n={len(pairs):6d}  sim {sim:.4f}")
    res["random_pair_similarity"] = rand_sim
    print(f"   {'random pairs':<34s} n=  5000  sim {rand_sim:.4f}")

    # ---- 4. ambiguity, not retrieval
    trd = d[d.splits == "train"]
    vc = trd.text.value_counts()
    res["ambiguity"] = {
        "train_n": int(len(trd)), "train_unique_texts": int(trd.text.nunique()),
        "share_with_a_duplicate": float((vc[trd.text].to_numpy() > 1).mean()),
        "modal_share": float(vc.iloc[0] / len(trd))}
    print(f"\n4. ambiguity (NOT retrieval): {res['ambiguity']['train_unique_texts']} texts "
          f"for {res['ambiguity']['train_n']} train participants; "
          f"{res['ambiguity']['share_with_a_duplicate']*100:.1f}% share a profile with "
          f"someone else; modal profile {res['ambiguity']['modal_share']*100:.1f}%")
    print("   audio→text retrieval is measured in the projector smoke test, not here")

    # ---- 5. the InfoNCE floor the sampler actually imposes: E[log K], not log(E[K])
    # 5 — the floor computed over the ACTUAL batch manifest the official settings produce:
    # 20,714 participants, batch 64, drop_last=False, reshuffled every epoch. That is 323
    # full batches plus a final batch of 42, so the last batch has a different floor and
    # the epoch average is not the batch-64 theoretical value.
    print("\n5. duplicate-induced loss floor over the REAL batch manifest "
          "(batch 64, drop_last=False; log(E[K]) withdrawn)")
    texts = trd.text.to_numpy()
    res["loss_floor"] = {"n_train": int(len(texts)), "batch": 64, "drop_last": False}
    per_epoch = []
    for ep in range(20):
        order = np.random.RandomState(1000 + ep).permutation(len(texts))
        logK = []
        for i in range(0, len(order), 64):
            b = texts[order[i:i+64]]
            _, cnt = np.unique(b, return_counts=True)
            inv = {t: c for t, c in zip(*np.unique(b, return_counts=True))}
            logK.extend(np.log([inv[t] for t in b]))
        per_epoch.append(float(np.mean(logK)))
    nb_full, last = divmod(len(texts), 64)
    res["loss_floor"].update({
        "n_full_batches": int(nb_full), "last_batch_size": int(last),
        "mean_log_K_per_epoch": float(np.mean(per_epoch)),
        "sd_across_epochs": float(np.std(per_epoch)),
        "theoretical_batch64": float(expected_log_K(trd.text, 64)[0])})
    r5 = res["loss_floor"]
    print(f"   {r5['n_train']} train participants -> {nb_full} full batches of 64 "
          f"plus one of {last}")
    print(f"   mean log K over the real manifest: {r5['mean_log_K_per_epoch']:.4f} nats "
          f"(sd {r5['sd_across_epochs']:.4f} across 20 shuffles)")
    print(f"   the iid batch-64 approximation would have said "
          f"{r5['theoretical_batch64']:.4f}; the manifest value is what training reports")

    np.savez_compressed(args.emb_out, participants=d[UNIT].to_numpy(),
                        text_id=np.array([tid[t] for t in d.text]),
                        unique_texts=np.array(uniq), unique_embeddings=U.astype(np.float32),
                        layer="last_hidden_state", pooling="mask_aware_mean",
                        revision=REVISION, sha256_16=h1)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}, {args.texts_out}, {args.emb_out}")


if __name__ == "__main__":
    main()
