"""H12 Stage 1A — the positive aggregation, and nothing else.

Step 0 demoted the region adapter from premise to increment: joint 5x3 reading is worth
about +0.050 of the 0.431 oracle gap, and the rest is knowing where to look. So the first
thing to change is the objective's relationship to localisation, not the head.

H11 and this share a denominator. `grounding_train.py:259` already masks invalid patches
and takes log_softmax over the whole valid grid; the mask is weighted uniformly. Written
out, the two objectives are

    L_uniform      =  logsumexp_valid(s)  -  mean_{p in M} s_p        (H11)
    L_region_mass  =  logsumexp_valid(s)  -  logsumexp_{p in M} s_p   (this)

H11 forces **every** patch inside the region to score high. The region-mass form asks only
that the region **as a whole** collect enough probability mass, which is the objective that
matches a max-over-window readout. No temperature is introduced: H11 runs at tau = 1 and so
does this, because a temperature would be a second variable.

Everything else is held fixed by construction — same synthesis, same seeds, same patient
split, same model, optimiser, lr, epochs, batch and hidden width, same control suite, same
2AFC. The guarantee is not asserted, it is **proved**: the `uniform` arm must reproduce
`results/grounding_v21_preds.npz` bit for bit, per example and per seed, before the
region-mass arm is allowed to mean anything.

Saved per example AND per seed, for both arms: the 2AFC win vector and the patch-level
global `hit@argmax` (argmax over the whole valid patch grid lands inside the queried mask —
the H11 definition, not Step 0's window-level one).

CIs are a **patient x seed hierarchical bootstrap**: patients resampled with replacement to
capture sampling error, seeds resampled with replacement to capture optimiser noise. The
patient-only CI is reported alongside because that is what H11 published.

    python src/h12_mil_search.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_v21 import SR, N_FREQ, N_TIME, CHARS, admissible_fc, build_clip
from grounding_train import CONTROLS, dev_patients, boot_ci, region_scores

AGGS = ("uniform", "region_mass")


def hier_ci(W, pids, boot=2000, seed=11):
    """Patient x seed hierarchical bootstrap. W is (n_seeds, n_examples): patients are
    resampled with replacement for sampling error, seeds for optimiser noise."""
    g = np.array(pids); up = np.array(sorted(set(g)))
    by = {p: np.where(g == p)[0] for p in up}
    rb = np.random.RandomState(seed); bs = []
    for _ in range(boot):
        idx = np.concatenate([by[p] for p in rb.choice(up, len(up))])
        ss = rb.choice(W.shape[0], W.shape[0])
        bs.append(W[np.ix_(ss, idx)].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


def hier_ci_paired(A, B, pids, boot=2000, seed=11):
    """Same resample applied to both arms, so the delta is paired at the example level and
    at the seed level. Seeds are matched across arms by initialisation."""
    g = np.array(pids); up = np.array(sorted(set(g)))
    by = {p: np.where(g == p)[0] for p in up}
    rb = np.random.RandomState(seed); bs = []
    for _ in range(boot):
        idx = np.concatenate([by[p] for p in rb.choice(up, len(up))])
        ss = rb.choice(A.shape[0], A.shape[0])
        bs.append(A[np.ix_(ss, idx)].mean() - B[np.ix_(ss, idx)].mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    import torch, torch.nn as nn
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--frozen", default="results/grounding_v21_manifest.json")
    ap.add_argument("--h11_preds", default="results/grounding_v21_preds.npz")
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/h12_stage1a_mil.json")
    ap.add_argument("--preds_out", default="results/h12_stage1a_preds.npz")
    args = ap.parse_args()
    # anything below would silently break the "loss is the only difference" guarantee
    assert (args.n, args.layers, args.epochs, args.batch, args.hidden, args.lr,
            list(args.seeds)) == (700, 6, 25, 32, 256, 1e-3, [0, 1, 2]), \
        "hyperparameters differ from grounding_train.py; the arms would not be comparable"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(0)
    seg = json.load(open(args.manifest)); smap = json.load(open(args.split_map))
    pool = [e for e in seg if e["label"] == "normal" and e.get("duration", 0) >= 1.2
            and smap.get(os.path.basename(e["path"])) == "train"]
    rng.shuffle(pool); pool = pool[:args.n]
    fc_pool = admissible_fc()

    rows = []
    for k, e in enumerate(pool):
        p = e["path"] if os.path.isabs(e["path"]) else os.path.join(args.audio_root, e["path"])
        try:
            y, sr = sf.read(p, dtype="float32")
        except Exception:
            continue
        if y.ndim > 1: y = y.mean(1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        r = build_clip(y, len(y) / SR, rng, fc_pool,
                       harm_first=(k % 2 == 0), variant=(k // 2) % 2)
        if r is None: continue
        yi, ev, occ = r
        if (ev["harm"]["mask"] & ev["inharm"]["mask"]).any(): continue
        rows.append({"y": yi, "ev": ev, "occ": occ, "sid": os.path.basename(p),
                     "pid": os.path.basename(p).split("_")[0]})
    print(f"{len(rows)} clips, {len(set(r['pid'] for r in rows))} patients (TRAIN split only)")

    man = json.load(open(args.frozen))
    assert len(man) == len(rows) and all(
        r["sid"] == f["sid"] and abs(r["ev"][c]["on"] - f["events"][c]["onset_s"]) < 1e-9
        for r, f in zip(rows, man) for c in CHARS), "synthesis differs from the frozen v2.1"
    print(f"frozen-synthesis check: {len(rows)} clips reproduce "
          f"{os.path.basename(args.frozen)}")

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)
    grids = []
    for i in range(0, len(rows), 8):
        inp = fe([r["y"] for r in rows[i:i+8]], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            h = m(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state[:, 2:, :]
        grids.append(h.reshape(len(h), N_FREQ, N_TIME, -1).cpu().numpy().astype(np.float32))
    G = np.concatenate(grids); del grids
    print(f"patch grids {G.shape}")

    ci = np.repeat(np.arange(len(rows)), 2)
    qb = np.tile([0, 1], len(rows))
    M = np.stack([rows[c]["ev"][CHARS[q]]["mask"] for c, q in zip(ci, qb)]).astype(np.float32)
    MD = np.stack([rows[c]["ev"][CHARS[1 - q]]["mask"] for c, q in zip(ci, qb)]).astype(np.float32)
    OC = np.stack([rows[c]["occ"] for c in ci])
    pid = np.array([rows[c]["pid"] for c in ci])
    dvp = dev_patients([r["pid"] for r in rows])
    dv = np.isin(pid, list(dvp)); tr = ~dv
    dvi = np.where(dv)[0]
    print(f"train {tr.sum()} / dev {dv.sum()} examples, patient-disjoint")

    POSF = np.zeros((N_FREQ, N_TIME, N_FREQ + N_TIME), np.float32)
    for f in range(N_FREQ):
        for t in range(N_TIME):
            POSF[f, t, f] = 1.0; POSF[f, t, N_FREQ + t] = 1.0

    Gt = torch.tensor(G); Mt = torch.tensor(M); Ot = torch.tensor(OC)

    def run(control, seed, agg):
        """Verbatim H11 `run`, with one branch on the positive term. Everything that
        touches an RNG happens in the same order, which is what makes the `uniform` arm
        reproduce grounding_v21_preds.npz exactly."""
        torch.manual_seed(seed); rs = np.random.RandomState(500 + seed)
        q_in = qb.copy()
        idx_src = np.arange(len(ci))
        d_in = {"query_only": 1, "audio_zeroed": 1,
                "position_only": N_FREQ + N_TIME}.get(control, G.shape[-1])
        use_grid = control not in ("query_only", "audio_zeroed", "position_only")
        if control == "query_shuffled":
            q_in = q_in[rs.permutation(len(q_in))]
        elif control == "audio_shuffled":
            idx_src = rs.permutation(len(ci))
        tgt = Mt
        if control == "target_permuted":
            tgt = Mt[torch.tensor(rs.permutation(len(ci)))]

        proj = nn.Linear(d_in, args.hidden).to(dev)
        qemb = nn.Embedding(2, args.hidden).to(dev)
        opt = torch.optim.AdamW(list(proj.parameters()) + list(qemb.parameters()), lr=args.lr)
        qt = torch.tensor(q_in, device=dev)
        tr_idx = np.where(tr)[0]

        def score(b):
            if use_grid:
                f = Gt[torch.tensor(ci[idx_src[b]])].to(dev)
            elif control == "query_only":
                f = torch.ones(len(b), N_FREQ, N_TIME, 1, device=dev)
            elif control == "audio_zeroed":
                f = torch.zeros(len(b), N_FREQ, N_TIME, 1, device=dev)
            else:
                f = torch.tensor(POSF, device=dev)[None].expand(len(b), -1, -1, -1)
            e = qemb(torch.zeros(len(b), dtype=torch.long, device=dev)
                     if control == "query_constant" else qt[torch.tensor(b, device=dev)])
            return (proj(f) * e[:, None, None, :]).sum(-1)

        for _ in range(args.epochs):
            perm = rs.permutation(tr_idx)
            for s0 in range(0, len(perm), args.batch):
                b = perm[s0:s0 + args.batch]
                sc = score(b)
                mk = tgt[torch.tensor(b)].to(dev)
                ok = Ot[torch.tensor(b)].to(dev)[:, None, :] > 0
                sc = sc.masked_fill(~ok, -1e9)
                # ---- THE ONLY DIFFERENCE BETWEEN THE TWO ARMS ----
                if agg == "uniform":
                    p = (mk / mk.sum((1, 2), keepdim=True).clamp(min=1)).flatten(1)
                    loss = -(p * sc.flatten(1).log_softmax(1)).sum(1).mean()
                else:
                    flat = sc.flatten(1)
                    pos = flat.masked_fill(mk.flatten(1) <= 0, float("-inf"))
                    loss = (flat.logsumexp(1) - pos.logsumexp(1)).mean()
                # --------------------------------------------------
                opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            S = np.concatenate([score(dvi[i:i+32]).cpu().numpy()
                                for i in range(0, len(dvi), 32)])
        return S

    def per_example(S):
        """2AFC win and patch-level global hit@argmax, per example, same tie-breaking and
        the same RandomState(7) H11 used."""
        q, d = region_scores(S, M[dvi], MD[dvi])
        r1 = np.random.RandomState(7)
        tie = np.abs(q - d) < 1e-9
        win = np.where(tie, r1.rand(len(q)) < 0.5, q > d).astype(float)
        r2 = np.random.RandomState(7)
        s = np.where(OC[dvi][:, None, :] > 0, S, -1e9)
        s = s + r2.rand(*s.shape).astype(np.float32) * 1e-6
        fi, ti = np.unravel_index(s.reshape(len(s), -1).argmax(1), (N_FREQ, N_TIME))
        hit = M[dvi][np.arange(len(s)), fi, ti].astype(float)
        return win, hit

    # ---- uniform: reproduction check only, then region-mass with the full control suite
    plan = [("uniform", ["intact"]), ("region_mass", CONTROLS)]
    WIN, HIT = {}, {}
    for agg, controls in plan:
        for c in controls:
            w, h = [], []
            for s in args.seeds:
                S = run(c, s, agg)
                a, b = per_example(S)
                w.append(a); h.append(b)
            WIN[f"{agg}:{c}"] = np.stack(w); HIT[f"{agg}:{c}"] = np.stack(h)
            print(f"  {agg:<12s} {c:<16s} 2AFC {np.mean(w):.3f} +/-{np.std(np.mean(w, 1)):.3f}"
                  f"   hit@argmax {np.mean(h):.3f}")

    # ---- the guarantee: uniform must be H11, bit for bit, per example and per seed
    z = np.load(args.h11_preds, allow_pickle=True)
    assert list(z["pid"]) == list(pid[dvi]) and list(z["query_bit"]) == list(qb[dvi]) \
        and list(z["clip_idx"]) == list(ci[dvi]), "dev examples do not match H11's"
    h11 = z["wins"][list(z["arms"]).index("intact")]
    mism = int((h11 != WIN["uniform:intact"]).sum())
    print(f"\nreproduction check vs {os.path.basename(args.h11_preds)}: "
          f"{mism} / {h11.size} per-example per-seed wins differ")
    for k, s in enumerate(args.seeds):
        d = int((h11[k] != WIN["uniform:intact"][k]).sum())
        print(f"  seed {s}: {d} differ, 2AFC {h11[k].mean():.4f} (H11) vs "
              f"{WIN['uniform:intact'][k].mean():.4f} (here)")
    assert mism == 0, ("the uniform arm does not reproduce H11 exactly, so the harness is "
                       "not identical and the region-mass delta cannot be attributed to "
                       "the loss. Fix this before interpreting anything below.")
    print("PASS — the harness is identical, so the loss is the only difference below.")

    # ---- results
    res = {}
    print(f"\n{'arm':<28s}{'2AFC':>8s}{'sd':>7s}{'hier 95% CI':>20s}"
          f"{'patient-only CI':>20s}{'hit@argmax':>12s}")
    for k in WIN:
        w, h = WIN[k], HIT[k]
        hc = hier_ci(w, pid[dvi]); pc = boot_ci(w.mean(0), pid[dvi])
        res[k] = {"afc": float(w.mean()), "afc_seed_sd": float(np.std(w.mean(1))),
                  "ci_hier": hc, "ci_patient": pc,
                  "hit_argmax": float(h.mean()), "hit_ci_hier": hier_ci(h, pid[dvi])}
        print(f"{k:<28s}{w.mean():8.3f}{np.std(w.mean(1)):7.3f}"
              f"   [{hc[0]:.3f}, {hc[1]:.3f}]   [{pc[0]:.3f}, {pc[1]:.3f}]{h.mean():12.3f}")

    print("\npaired delta, patient x seed hierarchical bootstrap (seeds matched by init)")
    res["paired"] = {}
    for name, A, B in [("2afc: region_mass - uniform",
                        WIN["region_mass:intact"], WIN["uniform:intact"]),
                       ("hit:  region_mass - uniform",
                        HIT["region_mass:intact"], HIT["uniform:intact"])]:
        d = float(A.mean() - B.mean()); lo, hi = hier_ci_paired(A, B, pid[dvi])
        res["paired"][name] = {"delta": d, "ci": [lo, hi]}
        print(f"  {name:<32s} {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"excludes 0: {'yes' if (lo > 0 or hi < 0) else 'no'}")

    np.savez_compressed(args.preds_out,
                        arms=np.array(list(WIN.keys())),
                        wins=np.array([WIN[k] for k in WIN], dtype=np.float32),
                        hits=np.array([HIT[k] for k in HIT], dtype=np.float32),
                        seeds=np.array(args.seeds), pid=pid[dvi], query_bit=qb[dvi],
                        clip_idx=ci[dvi])
    json.dump(res, open(args.out, "w"), indent=1)

    # ---- the pre-registered gate for Stage 1A
    d2 = res["paired"]["2afc: region_mass - uniform"]
    dh = res["paired"]["hit:  region_mass - uniform"]
    ctl = {k: res[k]["afc"] for k in WIN if k.startswith("region_mass:") and "intact" not in k}
    worst = max(ctl, key=ctl.get)
    sgn = [float((WIN["region_mass:intact"][k] - WIN["uniform:intact"][k]).mean())
           for k in range(len(args.seeds))]
    print()
    if ctl[worst] > 0.60:
        print(f"SHORTCUT: {worst} reaches {ctl[worst]:.3f} without doing the task; the "
              f"region-mass number cannot be interpreted.")
    elif d2["ci"][0] > 0:
        c2 = ("met" if dh["delta"] > 0 else
              "NOT met — a 2AFC gain without a search gain is not a search gain")
        c4 = "met" if all(x > 0 for x in sgn) else "NOT met"
        print(f"STAGE 1A PASSES criterion 1: region-mass beats uniform by "
              f"{d2['delta']:+.4f}, CI [{d2['ci'][0]:+.4f}, {d2['ci'][1]:+.4f}].")
        print(f"  hit@argmax {dh['delta']:+.4f} [{dh['ci'][0]:+.4f}, {dh['ci'][1]:+.4f}]"
              f"  -> criterion 2 {c2}")
        print(f"  seed signs {['%+.3f' % x for x in sgn]}  -> criterion 4 {c4}")
    else:
        print(f"STAGE 1A FAILS: region-mass minus uniform is {d2['delta']:+.4f}, CI "
              f"[{d2['ci'][0]:+.4f}, {d2['ci'][1]:+.4f}], which does not exclude 0. The "
              f"search objective is not the mechanism. Per the pre-registration, Stage 1B "
              f"is not run and the multi-field synthesis is not built.")
    print(f"\nwrote {args.out} and {args.preds_out}")


if __name__ == "__main__":
    main()
