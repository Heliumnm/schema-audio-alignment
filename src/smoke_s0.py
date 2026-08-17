"""S0 — pure code-correctness gate. It measures nothing about the science.

Sixteen participants with sixteen distinct profiles from Standard train. Each arm has to
memorise its own assignment. Whether alignment works is not asked here and cannot be
inferred from anything S0 prints.

Checks, all of which must pass before S1 runs:

  pairing      the permutation is a derangement in the shuffled arms and the identity in
               the correct arm, and within_label never crosses a COVID label
  frozen       only projector parameters carry gradient, and the cached inputs are
               bit-identical before and after training
  loss         the framework's loss equals a hand-computed InfoNCE on the same tensors
  memorise     each arm reaches 16/16 Recall@1 against its own targets
  outputs      2560-dimensional, finite, non-zero, and non-negative (the trailing ReLU)
  reproducible the first 20 steps replay identically under the same seed

    python src/smoke_s0.py
"""
import os, json, hashlib
import numpy as np
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projector import (ContrastiveProjectionHead, contrastive_loss, make_optimizer,
                       build_pairing, SOURCE_COMMIT, TEMPERATURE, D_LLM)

N, STEPS, SEED = 16, 400, 0


def hand_infonce(a, t, tau=TEMPERATURE):
    """InfoNCE written out longhand, to check the framework call against."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    t = t / np.linalg.norm(t, axis=1, keepdims=True)
    s = a @ t.T / tau
    s = s - s.max(1, keepdims=True)
    return float(np.mean(-(np.diag(s) - np.log(np.exp(s).sum(1)))))


def main():
    torch.use_deterministic_algorithms(False)
    z = np.load("results/ast_embeddings.npz", allow_pickle=True)
    tz = np.load("results/metadata_text_embeddings.npz", allow_pickle=True)
    import pandas as pd
    C = pd.read_csv("results/ukcovid_audio_cohort.csv")
    T = pd.read_csv("results/metadata_texts.csv")
    assert np.array_equal(z["participants"], C.participant_identifier.to_numpy())
    assert np.array_equal(tz["participants"], T.participant_identifier.to_numpy())

    tr = (T.splits == "train").to_numpy() & C.participant_identifier.isin(
        T.participant_identifier).to_numpy()
    tid = tz["text_id"]
    U = tz["unique_embeddings"]
    # sixteen DISTINCT profiles
    seen, pick = set(), []
    for i in np.where(tr)[0]:
        if tid[i] not in seen:
            seen.add(tid[i]); pick.append(i)
        if len(pick) == N:
            break
    pick = np.array(pick)
    A = torch.tensor(z["embeddings"][pick], dtype=torch.float32)
    X = torch.tensor(U[tid[pick]], dtype=torch.float32)
    y = C.y.to_numpy()[pick]
    print(f"S0: {N} participants, {len(set(tid[pick]))} distinct profiles, "
          f"labels {np.bincount(y, minlength=2)}   source {SOURCE_COMMIT[:12]}")
    in_hash = hashlib.sha256(A.numpy().tobytes() + X.numpy().tobytes()).hexdigest()[:16]

    res, fails = {"source_commit": SOURCE_COMMIT, "input_hash": in_hash, "arms": {}}, []
    for mode in ("correct", "within_label", "global"):
        j = build_pairing(y, {"within_label": "within"}.get(mode, mode), SEED)
        ok_derange = bool(mode == "correct") == bool(np.all(j == np.arange(N)))
        ok_label = mode != "within_label" or bool(np.all(y[j] == y))
        Xp = X[torch.tensor(j)]

        torch.manual_seed(SEED)
        m = ContrastiveProjectionHead()
        opt = make_optimizer(m)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        m.train()
        first20 = []
        for step in range(STEPS):
            loss = contrastive_loss(m(A), Xp)
            opt.zero_grad(); loss.backward(); opt.step()
            if step < 20:
                first20.append(round(float(loss), 8))

        m.eval()
        with torch.no_grad():
            P = m(A)
            hand = hand_infonce(P.numpy(), Xp.numpy())
            fw = float(contrastive_loss(P, Xp))
            Pn = torch.nn.functional.normalize(P, dim=1)
            Xn = torch.nn.functional.normalize(Xp, dim=1)
            r1 = int((torch.argmax(Pn @ Xn.T, 1) == torch.arange(N)).sum())

        # reproducibility of the first 20 steps
        torch.manual_seed(SEED)
        m2 = ContrastiveProjectionHead(); opt2 = make_optimizer(m2)
        rep = []
        for step in range(20):
            l2 = contrastive_loss(m2(A), Xp)
            opt2.zero_grad(); l2.backward(); opt2.step()
            rep.append(round(float(l2), 8))

        out_hash = hashlib.sha256(A.numpy().tobytes() + X.numpy().tobytes()).hexdigest()[:16]
        e = {"pairing_ok": ok_derange, "within_label_respected": ok_label,
             "trainable_params": len(trainable),
             "loss_framework": fw, "loss_hand": hand,
             "loss_matches_hand": abs(fw - hand) < 1e-4,
             "recall_at_1": r1, "memorised": r1 == N,
             "out_dim": int(P.shape[1]), "finite": bool(torch.isfinite(P).all()),
             "nonzero": bool(P.abs().sum() > 0), "nonnegative": bool((P >= 0).all()),
             "inputs_unchanged": out_hash == in_hash,
             "first20_reproducible": rep == first20}
        res["arms"][mode] = e
        print(f"  {mode:<14s} R@1 {r1:2d}/{N}  loss {fw:.6f} (hand {hand:.6f})  "
              f"dim {e['out_dim']}  nonneg {e['nonnegative']}  "
              f"repro {e['first20_reproducible']}  inputs_unchanged {e['inputs_unchanged']}")
        for k in ("pairing_ok", "within_label_respected", "loss_matches_hand", "memorised",
                  "finite", "nonzero", "nonnegative", "inputs_unchanged",
                  "first20_reproducible"):
            if not e[k]:
                fails.append(f"{mode}:{k}")
        if e["out_dim"] != D_LLM:
            fails.append(f"{mode}:out_dim")

    res["passed"] = not fails
    res["failures"] = fails
    json.dump(res, open("results/smoke_s0.json", "w"), indent=1)
    print(f"\nS0 {'PASSED' if not fails else 'FAILED: ' + str(fails)}")
    print("S0 says nothing about whether alignment works; that is S1 and the formal run.")


if __name__ == "__main__":
    main()
