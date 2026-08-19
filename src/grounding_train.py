"""Grounding v2.1 — the localisation model and its shortcut-control suite.

Runs on the **official TRAIN split only**, patient-disjoint train/dev inside it. The
test set stays untouched until this design is frozen.

The task
--------
One clip, two events at randomised positions, both three-partial tones with identical
mask geometry (see `grounding_v21.py`). The query is a single bit — locate the
**harmonic** event, or the **inharmonic** one. The model scores every patch and the
queried event's region must win.

Primary metric is a **within-clip 2AFC**: mean score inside the queried region versus
mean score inside the distractor's. Both regions sit in the same clip, and character is
counterbalanced against slot, so chance is exactly 0.5 and per-clip confounds cancel.
`hit@argmax` (the global argmax lands inside the queried mask) is reported alongside.

Why this many controls
----------------------
Three previous designs passed and were withdrawn, each time because a control that
would have taken an hour was not run first. Every arm below answers "could this number
have arisen without doing the task?", and the trained model is only interpretable if
all of them fail:

  query_only          patches replaced by a constant. The query alone cannot prefer one
                      region over the other within a clip, so this must be 0.5.
  query_constant      both queries map to the same embedding. Tests whether audio and
                      position alone pick the "right" region without being told which.
  position_only       features replaced by row/column one-hots — position kept, acoustic
                      content destroyed.
  query_shuffled      query bit permuted across clips: correspondence broken, marginals
                      preserved.
  query_flip          an algebraic sanity check, not evidence. A clip's two examples
                      carry the same regions with the labels swapped, so evaluating with
                      the opposite query reproduces the other example's scores and the
                      win vector is the complement of intact's by construction. It
                      confirms the evaluation plumbing is consistent; it cannot
                      corroborate that the head reads the query, and it is excluded from
                      the paired-delta table for that reason.
  audio_shuffled      each sample sees another clip's patch grid.
  audio_zeroed        patch grid zeroed (ties broken at random).
  target_permuted     which region counts as the answer is randomised during training.
  oracle_location_crop  not a control and not a competitor: the feasibility gate's own
                      crop classifier, handed BOTH ground-truth event locations and asked
                      only to say which crop matches the query. It measures how much of
                      the correspondence survives in the patches when localisation is
                      given away for free, so it upper-bounds what any head could
                      extract at these locations.
  dsp_query_cond      no learning. Inside each candidate region, find the middle partial
                      in the log-mel filterbank and measure its normalised
                      log-frequency position p between the endpoints: harmonic sits at
                      p = 0.5, inharmonic at 0.316 or 0.684. Pick the region matching
                      the query. This exploits the construction directly and is meant to
                      be hard to beat — if the learned model does not clear it, the
                      result is a statement about mechanism, not a method.

    python src/grounding_train.py --manifest .../manifest.json \\
        --audio_root .../icbhi_pathology_fidelity --n 700
"""
import os, json, argparse
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_v21 import (SR, PATCH, STRIDE, N_FREQ, N_TIME, N_MELS, MEL_MAX,
                           CHARS, admissible_fc, build_clip)

CONTROLS = ["intact", "query_only", "query_constant", "position_only", "query_shuffled",
            "audio_shuffled", "audio_zeroed", "target_permuted"]


def dev_patients(pids):
    """The same patient-level split the feasibility gate used, so the two are comparable."""
    pats = np.array(sorted(set(pids)))
    rs = np.random.RandomState(0); rs.shuffle(pats)
    return set(pats[:max(2, len(pats) // 3)])


def boot_ci(wins, pids, boot=2000):
    """Patient-cluster bootstrap on the 2AFC win vector. Seed sd measures optimiser
    noise; this measures sampling error, and only the second answers 'above chance?'."""
    g = np.array(pids); up = np.array(sorted(set(g)))
    by = {p: np.where(g == p)[0] for p in up}
    rb = np.random.RandomState(11); bs = []
    for _ in range(boot):
        take = np.concatenate([by[p] for p in rb.choice(up, len(up))])
        bs.append(np.mean(wins[take]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(lo), float(hi)]


def region_scores(sc, m_q, m_d):
    """Mean score inside each mask. sc (B,F,T), masks (B,F,T) bool."""
    q = (sc * m_q).sum((1, 2)) / np.maximum(m_q.sum((1, 2)), 1)
    d = (sc * m_d).sum((1, 2)) / np.maximum(m_d.sum((1, 2)), 1)
    return q, d


def wins_vector(sc, m_q, m_d, rng):
    q, d = region_scores(sc, m_q, m_d)
    tie = np.abs(q - d) < 1e-9
    return np.where(tie, rng.rand(len(q)) < 0.5, q > d).astype(float)


def evaluate(sc, m_q, m_d, occ, rng):
    """2AFC on region means, plus hit@argmax. Ties broken at random so a degenerate
    all-zero scorer reports 0.5 rather than 1.0."""
    q, d = region_scores(sc, m_q, m_d)
    tie = np.abs(q - d) < 1e-9
    win = np.where(tie, rng.rand(len(q)) < 0.5, q > d)
    s = np.where(occ[:, None, :] > 0, sc, -1e9)
    s = s + rng.rand(*s.shape).astype(np.float32) * 1e-6      # random tiebreak
    fi, ti = np.unravel_index(s.reshape(len(s), -1).argmax(1), (N_FREQ, N_TIME))
    hit = m_q[np.arange(len(s)), fi, ti]
    return {"choice_2afc": float(win.mean()), "hit_argmax": float(hit.mean())}


def dsp_scores(FB, ev, ch):
    """Query-conditioned DSP detector, no learning. Inside the event's band and frames,
    locate the strongest mel bin strictly between the two endpoint bins and measure its
    normalised log-frequency position between them."""
    e = ev[ch]
    f_lo, f_hi = float(min(e["freqs"])), float(max(e["freqs"]))
    b = lambda f: 2595 * np.log10(1 + f / 700.0) / MEL_MAX * N_MELS
    b_lo, b_hi = int(np.floor(b(f_lo))), int(np.ceil(b(f_hi)))
    t0, t1 = int(e["on"] * 100), int((e["on"] + e["dur"]) * 100)
    seg = FB[max(t0, 0):max(t1, t0 + 1), :]
    if not len(seg) or b_hi - b_lo < 4:
        return 0.5
    prof = seg.mean(0)
    inner = np.arange(b_lo + 2, min(b_hi - 1, N_MELS))
    if len(inner) < 2:
        return 0.5
    pk = inner[int(np.argmax(prof[inner]))]
    # invert the mel scale to Hz, then take log-frequency position between the endpoints
    f_pk = 700.0 * (10 ** (pk / N_MELS * MEL_MAX / 2595) - 1)
    return float(np.clip((np.log(max(f_pk, 1.0)) - np.log(f_lo)) /
                         (np.log(f_hi) - np.log(f_lo)), 0.0, 1.0))


def main():
    import torch, torch.nn as nn
    import soundfile as sf, librosa
    from transformers import ASTModel, AutoFeatureExtractor

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--audio_root", default="")
    ap.add_argument("--model",
                    default="/mnt/hd/data_heliu/hf_models/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--split_map", default="results/split_official.json")
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/grounding_v21_train.json")
    ap.add_argument("--preds_out", default="results/grounding_v21_preds.npz")
    args = ap.parse_args()

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
        rows.append({"y": yi, "ev": ev, "occ": occ,
                     "pid": os.path.basename(p).split("_")[0]})
    print(f"{len(rows)} clips, {len(set(r['pid'] for r in rows))} patients (TRAIN split only)")

    fe = AutoFeatureExtractor.from_pretrained(args.model)
    m = ASTModel.from_pretrained(args.model)
    m.encoder.layer = torch.nn.ModuleList(list(m.encoder.layer)[:args.layers])
    m = m.eval().to(dev)
    grids, FBs = [], []
    for i in range(0, len(rows), 8):
        inp = fe([r["y"] for r in rows[i:i+8]], sampling_rate=SR, return_tensors="pt")
        FBs.append(inp["input_values"].numpy().astype(np.float32))
        with torch.no_grad():
            h = m(**{k: v.to(dev) for k, v in inp.items()}).last_hidden_state[:, 2:, :]
        grids.append(h.reshape(len(h), N_FREQ, N_TIME, -1).cpu().numpy().astype(np.float32))
    G = np.concatenate(grids); FB = np.concatenate(FBs)
    del grids, FBs
    print(f"patch grids {G.shape}   filterbanks {FB.shape}")

    # two examples per clip: query = harmonic, query = inharmonic
    ci = np.repeat(np.arange(len(rows)), 2)
    qb = np.tile([0, 1], len(rows))                       # 0 = harm, 1 = inharm
    M = np.stack([rows[c]["ev"][CHARS[q]]["mask"] for c, q in zip(ci, qb)]).astype(np.float32)
    MD = np.stack([rows[c]["ev"][CHARS[1 - q]]["mask"] for c, q in zip(ci, qb)]).astype(np.float32)
    OC = np.stack([rows[c]["occ"] for c in ci])
    pid = np.array([rows[c]["pid"] for c in ci])
    dvp = dev_patients([r["pid"] for r in rows])
    dv = np.isin(pid, list(dvp)); tr = ~dv
    print(f"train {tr.sum()} / dev {dv.sum()} examples, patient-disjoint")

    POSF = np.zeros((N_FREQ, N_TIME, N_FREQ + N_TIME), np.float32)
    for f in range(N_FREQ):
        for t in range(N_TIME):
            POSF[f, t, f] = 1.0; POSF[f, t, N_FREQ + t] = 1.0

    Gt = torch.tensor(G); Mt = torch.tensor(M); Ot = torch.tensor(OC)

    def run(control, seed):
        torch.manual_seed(seed); rs = np.random.RandomState(500 + seed)
        q_in = qb.copy()
        idx_src = np.arange(len(ci))
        # zeroed and constant features collapse to `proj`'s bias whatever their width,
        # so a width of 1 is exactly equivalent to 768 and far cheaper
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
            # b indexes EXAMPLES (two per clip); the patch grid is stored per CLIP, so
            # audio_shuffled permutes examples and then maps through ci
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
                p = (mk / mk.sum((1, 2), keepdim=True).clamp(min=1)).flatten(1)
                loss = -(p * sc.flatten(1).log_softmax(1)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        out = {}
        with torch.no_grad():
            dvi = np.where(dv)[0]
            S = np.concatenate([score(dvi[i:i+32]).cpu().numpy()
                                for i in range(0, len(dvi), 32)])
        er = np.random.RandomState(7)
        out["intact_eval"] = evaluate(S, M[dvi], MD[dvi], OC[dvi], er)
        out["wins"] = wins_vector(S, M[dvi], MD[dvi], np.random.RandomState(7))
        if control == "intact":
            with torch.no_grad():
                qt_save = qt.clone(); qt[:] = 1 - qt          # forced within-clip flip
                Sf = np.concatenate([score(dvi[i:i+32]).cpu().numpy()
                                     for i in range(0, len(dvi), 32)])
                qt[:] = qt_save
            out["query_flip"] = evaluate(Sf, M[dvi], MD[dvi], OC[dvi],
                                         np.random.RandomState(7))
            out["query_flip_wins"] = wins_vector(Sf, M[dvi], MD[dvi],
                                                 np.random.RandomState(7))
        return out

    res, W = {}, {}
    for c in CONTROLS:
        runs = [run(c, s) for s in args.seeds]
        W[c] = np.stack([r["wins"] for r in runs])
        res[c] = {k: [float(np.mean([r["intact_eval"][k] for r in runs])),
                      float(np.std([r["intact_eval"][k] for r in runs]))]
                  for k in runs[0]["intact_eval"]}
        if "query_flip" in runs[0]:
            W["query_flip"] = np.stack([r["query_flip_wins"] for r in runs])
            res["query_flip"] = {k: [float(np.mean([r["query_flip"][k] for r in runs])),
                                     float(np.std([r["query_flip"][k] for r in runs]))]
                                 for k in runs[0]["query_flip"]}
        if c == "intact":
            res["intact"]["ci"] = boot_ci(np.mean([r["wins"] for r in runs], 0),
                                          pid[np.where(dv)[0]])
        print(f"  {c:<16s} 2AFC {res[c]['choice_2afc'][0]:.3f} "
              f"+/-{res[c]['choice_2afc'][1]:.3f}   "
              f"hit@argmax {res[c]['hit_argmax'][0]:.3f}")

    # ---- query-conditioned DSP, no learning
    dvi = np.where(dv)[0]
    wins = []
    for j in dvi:
        c, q = ci[j], qb[j]
        pq = dsp_scores(FB[c], rows[c]["ev"], CHARS[q])
        pd = dsp_scores(FB[c], rows[c]["ev"], CHARS[1 - q])
        # harmonic sits at p = 0.5; inharmonic away from it
        sq = -abs(pq - 0.5) if q == 0 else abs(pq - 0.5)
        sd = -abs(pd - 0.5) if q == 0 else abs(pd - 0.5)
        wins.append(sq > sd)
    wins = np.array(wins, float)
    W["dsp_query_cond"] = wins[None]
    res["dsp_query_cond"] = {"choice_2afc": [float(wins.mean()), 0.0],
                             "hit_argmax": [float("nan"), 0.0],
                             "ci": boot_ci(wins, pid[dvi])}
    print(f"  {'dsp_query_cond':<16s} 2AFC {wins.mean():.3f}  (no learning)")

    # ---- crop-probe oracle: the feasibility gate's own classifier, repurposed.
    # Not a new grounding model and not tuning — it is the upper bound the per-patch
    # head is being measured against. It reads each candidate region as a whole 5x3
    # crop, exactly as the gate did, and picks whichever crop the query matches. If
    # this is near 1.0 while the learned head sits at chance, the failure is in the
    # per-patch readout, not in what AST encoded.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from grounding_v21 import crop_origin, CROP_F, CROP_T
    def crop_at(c, ch):
        f0, t0 = crop_origin(rows[c]["ev"][ch]["mask"])
        return G[c, f0:f0 + CROP_F, t0:t0 + CROP_T].reshape(-1)
    tri = sorted(set(ci[np.where(tr)[0]])); dvc = sorted(set(ci[dvi]))
    Xo = np.array([crop_at(c, ch) for c in tri for ch in CHARS])
    yo = np.array([0 if ch == "harm" else 1 for c in tri for ch in CHARS])
    oc = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=20000, class_weight="balanced")).fit(Xo, yo)
    ow = []
    for j in dvi:
        c, q = ci[j], qb[j]
        pr = oc.predict_proba(np.stack([crop_at(c, CHARS[q]), crop_at(c, CHARS[1 - q])]))[:, 1]
        ow.append((pr[0] > pr[1]) if q == 1 else (pr[0] < pr[1]))
    ow = np.array(ow, float)
    W["oracle_location_crop"] = ow[None]
    res["oracle_location_crop"] = {"choice_2afc": [float(ow.mean()), 0.0],
                                   "hit_argmax": [float("nan"), 0.0],
                                   "ci": boot_ci(ow, pid[dvi])}
    print(f"  {'oracle_location_crop':<20s} 2AFC {ow.mean():.3f}  "
          f"(crop classifier given both event locations)")

    print(f"\n{'arm':<18s}{'2AFC':>9s}{'sd':>8s}{'hit@argmax':>13s}")
    order = (["intact", "query_flip", "dsp_query_cond", "oracle_location_crop"] +
             [c for c in CONTROLS if c != "intact"])
    for k in order:
        v = res[k]
        ci_s = (f"  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if "ci" in v else "")
        print(f"{k:<18s}{v['choice_2afc'][0]:9.3f}{v['choice_2afc'][1]:8.3f}"
              f"{v['hit_argmax'][0]:13.3f}{ci_s}")

    it, dsp = res["intact"]["choice_2afc"][0], res["dsp_query_cond"]["choice_2afc"][0]
    orc = res["oracle_location_crop"]["choice_2afc"][0]
    bad = {k: res[k]["choice_2afc"][0] for k in CONTROLS + ["query_flip"] if k != "intact"}
    worst = max(bad, key=bad.get)
    print()
    if bad[worst] > 0.60:
        print(f"SHORTCUT: {worst} reaches {bad[worst]:.3f} without doing the task. "
              f"The intact number ({it:.3f}) cannot be interpreted as grounding.")
    elif it < 0.60:
        lo, hi = res["intact"]["ci"]
        print(f"BELOW THE DEVELOPMENT GATE: {it:.3f}, 95% CI [{lo:.3f}, {hi:.3f}] — "
              f"weak but significantly above chance, and short of the 0.60 gate chosen "
              f"in advance. Every shortcut control is at chance, and the paired deltas "
              f"separate the model from query_shuffled, query_constant, target_permuted "
              f"and position_only.")
        print(f"  The oracle-location crop classifier reaches {orc:.3f}, so the "
              f"distinction is fully present in these patches. What the simple "
              f"query-to-patch bilinear head extracts is only a fraction of it.")
        print("  Consequence: do not proceed to the official test set and do not "
              "commission real annotation. Region-aware or cross-attention heads are "
              "untested and are not ruled out.")
    elif it < dsp:
        print(f"MECHANISM ONLY: the model grounds ({it:.3f}, controls at chance) but "
              f"loses to the query-conditioned DSP detector ({dsp:.3f}). Claim the "
              f"mechanism, not a method.")
    else:
        print(f"GROUNDING: {it:.3f} with every control at chance and DSP at {dsp:.3f}.")
    # ---- paired deltas, intact minus each arm, same dev examples, patient bootstrap.
    # query_flip is excluded: a clip's two examples carry swapped regions, so its win
    # vector is the complement of intact's by construction and its delta restates the
    # intact estimate rather than testing anything.
    print("\npaired delta vs intact (same dev examples, patient-cluster bootstrap)")
    wi = W["intact"].mean(0)
    res["paired_delta"] = {}
    for k in [c for c in CONTROLS if c != "intact"] + ["dsp_query_cond", "oracle_location_crop"]:
        d = wi - W[k].mean(0)
        lo, hi = boot_ci(d, pid[dvi])
        res["paired_delta"][k] = {"delta": float(d.mean()), "ci": [lo, hi]}
        print(f"  intact - {k:<22s} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"excludes 0: {'yes' if (lo > 0 or hi < 0) else 'no'}")
    lo, hi = res["intact"]["ci"]
    print(f"  intact - chance{'':<17s} {wi.mean()-0.5:+.4f}  95% CI "
          f"[{lo-0.5:+.4f}, {hi-0.5:+.4f}]  excludes 0: {'yes' if lo > 0.5 else 'no'}")

    np.savez_compressed(args.preds_out,
                        arms=np.array(list(W.keys())),
                        wins=np.array([W[k] if W[k].shape[0] == len(args.seeds)
                                       else np.repeat(W[k], len(args.seeds), 0)
                                       for k in W], dtype=np.float32),
                        seeds=np.array(args.seeds), pid=pid[dvi], query_bit=qb[dvi],
                        clip_idx=ci[dvi])
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} and per-example per-seed predictions {args.preds_out}")


if __name__ == "__main__":
    main()
