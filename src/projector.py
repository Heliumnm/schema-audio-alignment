"""The projector and the loss, copied from RespiraMFM b4224f231f947f3ac3ba8dcafe1f11d8a9f3e525.

Kept in one small module so the training script, S0 and S1 cannot drift apart from each
other or from the source they claim to reproduce.

Two things here are faithful copies that look like mistakes and are not ours to fix:

* the trailing `LayerNorm + ReLU` after the **second** linear, which the paper's appendix
  does not mention. The output is therefore non-negative, so after normalisation every
  pairwise cosine among projected audio vectors is >= 0;
* the loss is one-directional audio→text. The official `contrastive_loss` contains a
  symmetric branch behind a hard-coded `symmetric_loss = False`, so that branch is
  unreachable and is reproduced as unreachable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_COMMIT = "b4224f231f947f3ac3ba8dcafe1f11d8a9f3e525"
TEMPERATURE = 0.07
D_AUDIO, D_HIDDEN, D_LLM = 768, 1024, 2560       # Phi-2's d_llm is 2560
DROPOUT = 0.1


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, in_dim=D_AUDIO, out_dim=D_LLM, hidden_dim=D_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.projector(x)


def contrastive_loss(audio_proj, text_embed, temperature=TEMPERATURE):
    """One-directional audio→text InfoNCE over in-batch negatives, identical texts given
    no special treatment — exactly as the source runs it."""
    audio_proj = F.normalize(audio_proj, dim=1)
    text_embed = F.normalize(text_embed, dim=1)
    logits = torch.matmul(audio_proj, text_embed.T) / temperature
    labels = torch.arange(audio_proj.size(0), device=audio_proj.device)
    return F.cross_entropy(logits, labels)


def make_optimizer(model, lr=1e-3):
    """Adam on PyTorch defaults. No scheduler, no weight decay, no gradient clipping."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def build_pairing(labels, mode, seed):
    """Drawn once per seed and held fixed for all 500 epochs; never resampled per epoch.

    Returns an index array `j` such that audio i is paired with text j[i]. Self-pairing is
    forbidden. Landing on an identical schema by chance is not prevented — the caller
    measures the collision rate instead.
    """
    import numpy as np
    n = len(labels)
    if mode == "correct":
        return np.arange(n)
    rs = np.random.RandomState(10_000 + seed)
    j = np.arange(n)
    groups = [np.arange(n)] if mode == "global" else \
             [np.where(labels == v)[0] for v in np.unique(labels)]
    for g in groups:
        if len(g) < 2:
            continue
        for _ in range(1000):                      # derangement by retry, then repair
            p = rs.permutation(g)
            if not np.any(p == g):
                break
        fixed = np.where(p == g)[0]
        for k in fixed:                            # swap any remaining fixed point away
            m = (k + 1) % len(g)
            p[k], p[m] = p[m], p[k]
        j[g] = p
    assert not np.any(j == np.arange(n)), "self-pairing survived"
    return j


def build_stratified_pairing(*strata, seed, seed_base=70_000):
    """Build a deterministic no-self bijection within joint strata.

    This is the generic implementation used by the locked ``W_{y,s}`` arm.  Every
    participant is retained, and every donor has the same value on every supplied
    stratum (disease label and recorded sex in the present audit).  A singleton
    stratum is a hard protocol failure: the caller must not silently self-pair,
    cross a stratum, or delete rows only for this arm.
    """
    import numpy as np

    if not strata:
        raise ValueError("at least one stratification array is required")
    values = [np.asarray(value) for value in strata]
    n = len(values[0])
    if any(len(value) != n for value in values):
        raise ValueError("stratification arrays have different lengths")

    keys = np.asarray(list(zip(*(value.astype(str) for value in values))), dtype=object)
    grouped = {}
    for index, key in enumerate(map(tuple, keys.tolist())):
        grouped.setdefault(key, []).append(index)
    counts = {key: len(indices) for key, indices in grouped.items()}
    singletons = {key: size for key, size in counts.items() if size < 2}
    if singletons:
        raise ValueError(f"singleton joint stratum; frozen cohort is not eligible: {singletons}")

    rng = np.random.RandomState(int(seed_base) + int(seed))
    pairing = np.full(n, -1, dtype=np.int64)
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        group = np.asarray(grouped[key], dtype=np.int64)
        cycle = rng.permutation(group)
        pairing[cycle] = np.roll(cycle, -1)

    assert np.array_equal(np.sort(pairing), np.arange(n))
    assert not np.any(pairing == np.arange(n))
    for value in values:
        assert np.all(value[pairing] == value)
    return pairing
