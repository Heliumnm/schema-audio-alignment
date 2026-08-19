"""
CLIP-style contrastive learning with a TRAINABLE spectrogram encoder.

This is the variant the frozen-tower sweep never tested. There, both towers were
frozen and only an MLP projector was trained (RespiraMFM's design) — every one of
15 runs landed below a plain linear probe on the raw features. A frozen encoder's
features are fixed, so the projector can only remap them; it cannot reorganise the
representation the way CLIP's image tower does. Here the audio encoder itself is
fine-tuned from AudioSet weights against the text embeddings.

The text tower stays frozen. With 3,450 training cycles, training both towers would
overfit harder without adding capacity where it matters.

Overfitting is the main risk at this scale, so:
  - train/val is split at the PATIENT level (never segment level) and asserted;
  - early stopping on validation InfoNCE, patience-based, best weights restored;
  - three seeds, mean ± sd reported;
  - the frozen-encoder result is printed alongside as the number to beat.

Spectrograms are precomputed once and memmapped — the fbank extraction, not the
transformer, dominates wall clock if done per epoch.

    # 1. cache AST input features (once, ~3.6 GB)
    python src/clip_finetune.py prep --manifest .../manifest.json \
        --audio_root .../icbhi_pathology_fidelity \
        --model .../ast-finetuned-audioset-10-10-0.4593 --out results/ast_inputs

    # 2. fine-tune against one text condition
    python src/clip_finetune.py train --inputs results/ast_inputs \
        --text_emb results/text_emb_bio/all.npz --manifest .../manifest.json \
        --model .../ast-finetuned-audioset-10-10-0.4593 \
        --target wheeze --seeds 0 1 2 --out results/clip_wheeze_all.json
"""
import os, json, argparse
import numpy as np

TAU = 0.07
PROJ_DIM = 768


def patient_of(seg_id):
    return os.path.basename(seg_id).split("_")[0]


# ------------------------------------------------------------------- prep
def cmd_prep(args):
    import torch
    import soundfile as sf
    import librosa
    from transformers import AutoFeatureExtractor

    SR = 16000
    fe = AutoFeatureExtractor.from_pretrained(args.model)
    seg = json.load(open(args.manifest))

    def resolve(e):
        p = e.get("path", e["filename"])
        return p if (os.path.isabs(p) or not args.audio_root) else os.path.join(args.audio_root, p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ids, arr, n_err = [], None, 0
    for s0 in range(0, len(seg), args.batch):
        ys, keys = [], []
        for e in seg[s0:s0 + args.batch]:
            p = resolve(e)
            try:
                y, sr = sf.read(p, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if sr != SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                ys.append(y); keys.append(os.path.basename(p))
            except Exception:
                n_err += 1
        if not ys:
            if s0 == 0:
                raise SystemExit("First batch read 0 files — pass --audio_root.")
            continue
        v = fe(ys, sampling_rate=SR, return_tensors="np")["input_values"]
        if arr is None:
            arr = np.lib.format.open_memmap(
                args.out + ".npy", mode="w+", dtype=np.float32,
                shape=(len(seg), v.shape[1], v.shape[2]))
        arr[len(ids):len(ids) + len(v)] = v
        ids.extend(keys)
        if (s0 // args.batch) % 25 == 0:
            print(f"  {s0}/{len(seg)}", flush=True)
    arr.flush()
    json.dump({k: i for i, k in enumerate(ids)}, open(args.out + "_index.json", "w"))
    print(f"\ncached {len(ids)} x {arr.shape[1:]} -> {args.out}.npy ({n_err} failures)")


# ------------------------------------------------------------------ train
def cmd_train(args):
    import torch, torch.nn as nn
    from transformers import ASTModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, matthews_corrcoef

    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    X = np.load(args.inputs + ".npy", mmap_mode="r")
    xidx = json.load(open(args.inputs + "_index.json"))
    tz = np.load(args.text_emb, allow_pickle=True)
    tpos = {str(i): k for k, i in enumerate(tz["ids"])}
    T = tz["emb"]
    meta = {os.path.basename(e.get("path", e["filename"])): e
            for e in json.load(open(args.manifest))}

    prompt_emb = None
    if args.prompt_emb and os.path.exists(args.prompt_emb):
        pz = np.load(args.prompt_emb, allow_pickle=True)
        pt = {str(k): v for k, v in zip(pz["targets"], pz["emb"])}
        if args.target in pt:
            prompt_emb = pt[args.target].astype(np.float32)
            print(f"class prompts loaded for {args.target}")

    smap = json.load(open(args.split_map)) if args.split_map else None
    ids = [i for i in xidx if i in tpos and i in meta and (smap is None or i in smap)]
    if smap is not None:
        print(f"split_map: {os.path.basename(args.split_map)} -> {len(ids)} segments")
    rows = np.array([xidx[i] for i in ids])
    Xt = np.stack([T[tpos[i]] for i in ids]).astype(np.float32)
    y = np.array([int(meta[i]["label"] in (args.target, "both")) for i in ids])
    split = np.array([(smap[i] if smap else meta[i]["split"]) for i in ids])
    pid = np.array([patient_of(i) for i in ids])

    trall, te = split == "train", split == "test"
    assert not (set(pid[trall]) & set(pid[te])), "PATIENT LEAK train/test"

    results = []
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        # patient-level val carve-out — a segment-level split would put the same
        # patient on both sides and make early stopping meaningless
        pats = sorted(set(pid[trall]))
        rng.shuffle(pats)
        val_p = set(pats[:max(1, int(len(pats) * args.val_frac))])
        tr = trall & ~np.isin(pid, list(val_p))
        va = trall & np.isin(pid, list(val_p))
        assert not (set(pid[tr]) & set(pid[va])), "PATIENT LEAK train/val"
        print(f"\nseed {seed}: train {tr.sum()} ({len(set(pid[tr]))}p) | "
              f"val {va.sum()} ({len(set(pid[va]))}p) | test {te.sum()} ({len(set(pid[te]))}p)")

        enc = ASTModel.from_pretrained(args.model).to(dev)
        proj = nn.Sequential(nn.Linear(enc.config.hidden_size, PROJ_DIM)).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()),
                                lr=args.lr, weight_decay=0.05)
        Tt = torch.tensor(Xt, device=dev)

        def embed(mask, train=False, bs=None):
            bs = bs or args.batch
            enc.train(train); proj.train(train)
            idx = np.where(mask)[0]
            out = []
            for s in range(0, len(idx), bs):
                b = idx[s:s + bs]
                xb = torch.tensor(np.array(X[rows[b]]), device=dev)
                h = enc(input_values=xb).last_hidden_state.mean(1)
                out.append(proj(h))
            return torch.cat(out), idx

        def nce(za, zt):
            za = za / za.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            zt = zt / zt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            lg = za @ zt.T / TAU
            t = torch.arange(len(za), device=za.device)
            return 0.5 * (nn.functional.cross_entropy(lg, t) +
                          nn.functional.cross_entropy(lg.T, t))

        # ---- extra negatives ------------------------------------------------
        # InfoNCE sees batch_size - 1 negatives. CLIP gets 32,767; fine-tuning AST
        # caps the batch at ~24, leaving 23. Since the TEXT tower is frozen, extra
        # text negatives are free — they are already computed. Audio negatives come
        # from a FIFO queue of recent detached embeddings (MoCo-style, without a
        # momentum encoder: the queue is short relative to how fast the encoder
        # moves at lr 1e-5).
        #
        # False negatives matter here: `all` has 3,552 distinct texts over 4,142
        # training cycles, so a sampled "negative" is often the positive's exact
        # text. Those logits are masked out — otherwise the objective is punished
        # for matching text it should match, which is precisely the failure mode
        # the debiased-contrastive literature describes.
        def nce_big(za, zt, neg_t, neg_a):
            u = lambda x: x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            za, zt, neg_t, neg_a = u(za), u(zt), u(neg_t), u(neg_a)
            t = torch.arange(len(za), device=za.device)

            a2t_extra = za @ neg_t.T
            dup = (zt @ neg_t.T) > 0.999          # negative carries the positive's text
            a2t_extra = a2t_extra.masked_fill(dup, float("-inf"))
            lg_a = torch.cat([za @ zt.T, a2t_extra], 1) / TAU

            if len(neg_a):
                lg_t = torch.cat([zt @ za.T, zt @ neg_a.T], 1) / TAU
            else:
                lg_t = (zt @ za.T) / TAU
            return 0.5 * (nn.functional.cross_entropy(lg_a, t) +
                          nn.functional.cross_entropy(lg_t, t))

        queue = torch.zeros(0, PROJ_DIM, device=dev)

        tr_idx = np.where(tr)[0]
        best, bad, best_state = float("inf"), 0, None
        for ep in range(args.epochs):
            enc.train(); proj.train()
            perm = rng.permutation(tr_idx)
            tot = 0.0
            for s in range(0, len(perm), args.batch):
                b = perm[s:s + args.batch]
                if len(b) < 2:
                    continue
                xb = torch.tensor(np.array(X[rows[b]]), device=dev)
                h = enc(input_values=xb).last_hidden_state.mean(1)
                za = proj(h)
                if args.queue_size > 0:
                    nt = rng.choice(tr_idx, min(args.queue_size, len(tr_idx)), replace=False)
                    loss = nce_big(za, Tt[b], Tt[nt], queue)
                    queue = torch.cat([queue, za.detach()])[-args.queue_size:]
                else:
                    loss = nce(za, Tt[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(b)
            enc.eval(); proj.eval()
            with torch.no_grad():
                vz, vidx = embed(va)
                vloss = float(nce(vz, Tt[vidx]))
            print(f"  ep {ep+1}/{args.epochs}  train {tot/len(perm):.4f}  val {vloss:.4f}",
                  flush=True)
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                best_state = ({k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop at epoch {ep+1} (best val {best:.4f})")
                    break
        if best_state is not None:
            enc.load_state_dict(best_state[0]); proj.load_state_dict(best_state[1])
            enc.to(dev); proj.to(dev)

        enc.eval(); proj.eval()
        with torch.no_grad():
            Ztr, itr = embed(tr); Zte, ite = embed(te)
        Ztr, Zte = Ztr.cpu().numpy(), Zte.cpu().numpy()
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ztr, y[itr])
        probe = roc_auc_score(y[ite], clf.predict_proba(Zte)[:, 1])
        mcc = matthews_corrcoef(y[ite], clf.predict(Zte))

        def unit(a): return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(1e-8)
        # prototype: mean TRAIN text embedding per class. Uses train labels, so it
        # bounds what this text space can express — not zero-shot.
        pp = unit(Xt[itr][y[itr] == 1].mean(0)); pn = unit(Xt[itr][y[itr] == 0].mean(0))
        zs = roc_auc_score(y[ite], unit(Zte) @ pp - unit(Zte) @ pn)

        # prompt: TRUE zero-shot — no labels touched at any point. This is the only
        # honest measure of the one capability alignment buys over a plain probe.
        zsp = float("nan")
        if prompt_emb is not None:
            P = unit(prompt_emb)
            zsp = roc_auc_score(y[ite], unit(Zte) @ P[0] - unit(Zte) @ P[1])

        print(f"  seed {seed}: probe AUROC {probe:.3f}  MCC {mcc:.3f}  "
              f"zs_proto {zs:.3f}  zs_prompt {zsp:.3f}")
        results.append({"seed": seed, "auroc": float(probe), "mcc": float(mcc),
                        "zs_prototype_auroc": float(zs),
                        "zs_prompt_auroc": float(zsp), "best_val_nce": best})
        del enc, proj, opt
        if dev == "cuda":
            torch.cuda.empty_cache()

    agg = {k: {"mean": float(np.mean([r[k] for r in results])),
               "sd": float(np.std([r[k] for r in results]))}
           for k in ["auroc", "mcc", "zs_prototype_auroc", "zs_prompt_auroc"]}
    out = {"text_emb": os.path.basename(args.text_emb), "target": args.target,
           "trainable_encoder": True, "runs": results, "aggregate": agg}
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n=== trainable AST / {os.path.basename(args.text_emb)} / {args.target} ===")
    for k in ["auroc", "mcc", "zs_prototype_auroc", "zs_prompt_auroc"]:
        print(f"  {k:<20} {agg[k]['mean']:.3f} ± {agg[k]['sd']:.3f}")
    print(f"  frozen-encoder reference (raw AST probe): "
          f"{'0.857' if args.target == 'wheeze' else '0.689'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--manifest", required=True); p.add_argument("--audio_root", default="")
    p.add_argument("--model", required=True); p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=32)
    t = sub.add_parser("train")
    t.add_argument("--inputs", required=True); t.add_argument("--text_emb", required=True)
    t.add_argument("--manifest", required=True); t.add_argument("--model", required=True)
    t.add_argument("--target", default="wheeze", choices=["wheeze", "crackle"])
    t.add_argument("--split_map", default="", help="json {segment_id: train|test} overriding the manifest split; use for the OFFICIAL ICBHI partition")
    t.add_argument("--prompt_emb", default="", help="npz of encoded class prompts -> true zero-shot")
    t.add_argument("--queue_size", type=int, default=0,
                   help="extra negatives beyond the batch (0 = in-batch only, "
                        "reproducing the earlier runs exactly)")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--batch", type=int, default=24)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--val_frac", type=float, default=0.15)
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--out", default="")
    args = ap.parse_args()
    (cmd_prep if args.cmd == "prep" else cmd_train)(args)


if __name__ == "__main__":
    main()
