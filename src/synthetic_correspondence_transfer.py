"""Frozen synthetic correspondence--transfer stress test (E4).

This is an executable implementation of
``docs/SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md``.  The formal grid is
deliberately encoded as constants and cannot be changed from the command line.  A tiny
``smoke`` mode exists only to exercise the same code path before the formal run.

The experiment asks whether correct audio--metadata pairing can improve instance/profile
correspondence while failing to improve disease transfer when a shared source-domain
shortcut S stops tracking disease in the target.  It is a sufficient-mechanism stress
test, not a model of UKCOVID and not an external clinical confirmation.

Examples
--------
Pure implementation checks (no result files)::

    python src/synthetic_correspondence_transfer.py --self-test

Small end-to-end technical smoke::

    python src/synthetic_correspondence_transfer.py \
        --mode smoke --out-dir results/synthetic_correspondence_transfer_smoke

Formal run (the only scientific configuration)::

    python src/synthetic_correspondence_transfer.py \
        --mode formal --out-dir results/synthetic_correspondence_transfer

The run is safely shardable.  Shards write disjoint per-simulation files; summarize only
after every shard completes::

    python src/synthetic_correspondence_transfer.py --mode formal --out-dir OUT \
        --num-shards 4 --shard-index 0 --skip-summary
    python src/synthetic_correspondence_transfer.py --mode formal --out-dir OUT \
        --summarize-only

The preregistered language-faithful sensitivity is a separate one-cell run and requires a
precomputed, frozen cache with all 81 tier codes and Phi-2 embeddings::

    python src/synthetic_correspondence_transfer.py --mode formal \
        --language-profile-cache results/synthetic_phi2_profiles.npz \
        --out-dir results/synthetic_correspondence_transfer_phi2

Outputs are no-clobber by default.  ``--resume`` skips a simulation only when both its
prediction and metric files exist and their frozen configuration hash matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Frozen scientific configuration.  Formal values are not CLI options.

FORMAL_RHOS = (0.0, 0.50, 0.95)
FORMAL_KAPPAS = (0.5, 1.0, 2.0)
FORMAL_SEEDS = tuple(range(10))
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
ARMS = ("correct", "within_label", "global")
ALIGNED_REPS = ARMS
REPRESENTATIONS = (
    "raw", "correct", "within_label", "global",
    "raw_plus_correct", "raw_plus_within_label", "raw_plus_global",
)
PROBE_REPS = ("raw", "correct", "within_label", "global")
ENDPOINTS = ("source_test", "target_source_relation", "target_half", "target_zero")

MU_D = 0.37
TEMPERATURE = 0.07
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-3
DROPOUT = 0.1
N_FOLDS = 5
BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_820
PROFILE_THRESHOLDS = (-0.43, 0.43)
MASTER_SEED = 20_260_820


@dataclass(frozen=True)
class RunConfig:
    mode: str
    rhos: tuple[float, ...]
    kappas: tuple[float, ...]
    seeds: tuple[int, ...]
    n_train: int
    n_val: int
    n_source_test: int
    n_target: int
    epochs: int
    batch_size: int
    bootstrap: int
    mu_d: float = MU_D
    temperature: float = TEMPERATURE
    learning_rate: float = LEARNING_RATE
    dropout: float = DROPOUT
    thresholds: tuple[float, float] = PROFILE_THRESHOLDS
    c_grid: tuple[float, ...] = C_GRID
    source_prevalence: float = 0.36
    target_prevalence: float = 0.50
    n_context: int = 4
    n_audio_noise: int = 11
    projector_hidden: int = 64
    projector_out: int = 12
    metadata_variant: str = "one_hot"
    metadata_cache_sha256: str = ""


FORMAL_CONFIG = RunConfig(
    mode="formal", rhos=FORMAL_RHOS, kappas=FORMAL_KAPPAS, seeds=FORMAL_SEEDS,
    n_train=4096, n_val=2048, n_source_test=4096, n_target=4096,
    epochs=EPOCHS, batch_size=BATCH_SIZE, bootstrap=BOOTSTRAP,
)

SMOKE_CONFIG = RunConfig(
    mode="smoke", rhos=(0.50,), kappas=(1.0,), seeds=(0,),
    n_train=256, n_val=128, n_source_test=256, n_target=256,
    epochs=3, batch_size=BATCH_SIZE, bootstrap=100,
)


# ---------------------------------------------------------------------------
# Small utilities and provenance.

def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"cannot JSON encode {type(x)}")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=_json_default).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(x: np.ndarray) -> str:
    a = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(canonical_json(a.shape))
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def hash_arrays(items: Iterable[tuple[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode())
        h.update(sha256_array(value).encode())
    return h.hexdigest()


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def float_tag(x: float) -> str:
    return f"{x:.2f}".replace("-", "m").replace(".", "p")


def simulation_tag(rho: float, kappa: float, seed: int) -> str:
    return f"rho{float_tag(rho)}_kappa{float_tag(kappa)}_seed{seed:02d}"


def config_hash(config: RunConfig) -> str:
    return sha256_bytes(canonical_json(asdict(config)))


def stable_rng(seed: int, stream: int) -> np.random.RandomState:
    # SeedSequence avoids accidental collisions and is invariant to grid execution order.
    ss = np.random.SeedSequence([MASTER_SEED, int(seed), int(stream)])
    return np.random.RandomState(ss.generate_state(1, dtype=np.uint32)[0])


def state_hash(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def effective_rank(x: np.ndarray) -> float:
    xc = np.asarray(x, dtype=np.float64) - np.mean(x, axis=0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    p = (s * s) / max(float(np.sum(s * s)), 1e-30)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


# ---------------------------------------------------------------------------
# Data-generating mechanism.

@dataclass
class SyntheticSet:
    y: np.ndarray
    y_sign: np.ndarray
    s: np.ndarray
    tiers: np.ndarray
    metadata: np.ndarray
    x_raw: np.ndarray
    eps_d: np.ndarray
    eps_s: np.ndarray
    eps_noise: np.ndarray
    eta: np.ndarray
    eta_cf: np.ndarray
    rho: float
    kappa: float


@dataclass(frozen=True)
class ProfileEmbeddingCache:
    codes: np.ndarray
    embeddings: np.ndarray
    sha256: str


def exact_binary_labels(n: int, prevalence: float, rng: np.random.RandomState) -> np.ndarray:
    n_pos = int(round(n * prevalence))
    y = np.r_[np.ones(n_pos, dtype=np.int8), np.zeros(n - n_pos, dtype=np.int8)]
    return y[rng.permutation(n)]


def tiers_to_metadata(tiers: np.ndarray) -> np.ndarray:
    tiers = np.asarray(tiers, dtype=np.int64)
    assert tiers.ndim == 2 and tiers.shape[1] == 4
    eye = np.eye(3, dtype=np.float32)
    out = eye[tiers].reshape(len(tiers), 12)
    out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-12)
    return out.astype(np.float32, copy=False)


def generate_set(n: int, prevalence: float, rho: float, kappa: float,
                 seed: int, stream: int) -> SyntheticSet:
    rng = stable_rng(seed, stream)
    y = exact_binary_labels(n, prevalence, rng)
    ys = (2 * y - 1).astype(np.float32)
    eta = rng.standard_normal((n, 4)).astype(np.float32)
    eta_cf = rng.standard_normal((n, 4)).astype(np.float32)
    s = rho * ys[:, None] + math.sqrt(max(0.0, 1.0 - rho * rho)) * eta
    tiers = np.digitize(s, PROFILE_THRESHOLDS).astype(np.int8)
    metadata = tiers_to_metadata(tiers)
    eps_d = rng.standard_normal(n).astype(np.float32)
    eps_s = rng.standard_normal((n, 4)).astype(np.float32)
    eps_noise = rng.standard_normal((n, 11)).astype(np.float32)
    x = np.concatenate([
        (MU_D * ys + eps_d)[:, None],
        kappa * s + eps_s,
        eps_noise,
    ], axis=1).astype(np.float32)
    assert x.shape == (n, 16) and metadata.shape == (n, 12)
    return SyntheticSet(y, ys, s.astype(np.float32), tiers, metadata, x,
                        eps_d, eps_s, eps_noise, eta, eta_cf, rho, kappa)


def audio_counterfactual(ds: SyntheticSet, kind: str) -> np.ndarray:
    """Change S only; stable D coordinate and every additive noise draw stay fixed."""
    if kind == "flip":
        s_cf = -ds.s
    elif kind == "resample":
        s_cf = (ds.rho * ds.y_sign[:, None] +
                math.sqrt(max(0.0, 1.0 - ds.rho * ds.rho)) * ds.eta_cf)
    else:
        raise ValueError(kind)
    x = np.concatenate([
        (MU_D * ds.y_sign + ds.eps_d)[:, None],
        ds.kappa * s_cf + ds.eps_s,
        ds.eps_noise,
    ], axis=1).astype(np.float32)
    assert np.array_equal(x[:, 0], ds.x_raw[:, 0])
    assert np.array_equal(x[:, 5:], ds.x_raw[:, 5:])
    return x


def build_datasets(config: RunConfig, rho: float, kappa: float,
                   seed: int) -> dict[str, SyntheticSet]:
    # Streams do not depend on rho/kappa: within a simulation seed the same y, eta and
    # additive noise underlie the complete dose grid.  Only the frozen interventions vary.
    return {
        "train": generate_set(config.n_train, config.source_prevalence, rho, kappa,
                              seed, 10),
        "val": generate_set(config.n_val, config.source_prevalence, rho, kappa,
                            seed, 20),
        "source_test": generate_set(config.n_source_test, config.source_prevalence,
                                    rho, kappa, seed, 30),
        "target_source_relation": generate_set(
            config.n_target, config.target_prevalence, rho, kappa, seed, 40),
        "target_half": generate_set(
            config.n_target, config.target_prevalence, rho / 2.0, kappa, seed, 50),
        "target_zero": generate_set(
            config.n_target, config.target_prevalence, 0.0, kappa, seed, 60),
    }


def load_profile_embedding_cache(path: Path) -> ProfileEmbeddingCache:
    """Load the optional frozen Phi-2 sensitivity cache.

    The cache is intentionally prepared outside this experiment and must contain every
    one of the 3^4 tier profiles.  Accepted keys are ``profile_codes``/``codes`` and
    ``embeddings``/``emb``.  No text model is downloaded or changed here.
    """
    with np.load(path, allow_pickle=False) as z:
        code_key = "profile_codes" if "profile_codes" in z else "codes"
        emb_key = "embeddings" if "embeddings" in z else "emb"
        codes = np.asarray(z[code_key], dtype=np.int8)
        embeddings = np.asarray(z[emb_key], dtype=np.float32)
    assert codes.shape == (81, 4), f"expected all 81 profile codes, got {codes.shape}"
    assert embeddings.ndim == 2 and embeddings.shape[0] == 81
    assert np.isfinite(embeddings).all() and np.all(np.linalg.norm(embeddings, axis=1) > 0)
    expected = np.asarray(np.meshgrid(*([np.arange(3)] * 4))).reshape(4, -1).T
    assert {tuple(v) for v in codes.tolist()} == {tuple(v) for v in expected.tolist()}
    assert len({tuple(v) for v in codes.tolist()}) == 81
    return ProfileEmbeddingCache(codes, embeddings, sha256_file(path))


def apply_profile_embedding_cache(datasets: dict[str, SyntheticSet],
                                  cache: ProfileEmbeddingCache) -> None:
    lookup = {tuple(code): cache.embeddings[i] for i, code in enumerate(cache.codes.tolist())}
    for ds in datasets.values():
        ds.metadata = np.stack([lookup[tuple(code)] for code in ds.tiers.tolist()]).astype(np.float32)


def standardize_audio(datasets: dict[str, SyntheticSet]) -> tuple[dict[str, np.ndarray],
                                                                   np.ndarray, np.ndarray]:
    mean = datasets["train"].x_raw.mean(axis=0, keepdims=True)
    std = datasets["train"].x_raw.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x = {name: ((ds.x_raw - mean) / std).astype(np.float32)
         for name, ds in datasets.items()}
    return x, mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Frozen pairing, projector and InfoNCE.

def cycle_derangement(indices: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("a no-self derangement is impossible for a singleton group")
    p = rng.permutation(indices)
    out = np.empty(len(indices), dtype=np.int64)
    # A single random cycle is always a derangement.
    position = {int(v): i for i, v in enumerate(indices)}
    for a, b in zip(p, np.roll(p, -1)):
        out[position[int(a)]] = b
    assert not np.any(out == indices)
    return out


def build_pairing(y: np.ndarray, arm: str, seed: int) -> np.ndarray:
    n = len(y)
    if arm == "correct":
        return np.arange(n, dtype=np.int64)
    rng = stable_rng(seed, 100 if arm == "within_label" else 110)
    j = np.empty(n, dtype=np.int64)
    groups = [np.where(y == v)[0] for v in np.unique(y)] if arm == "within_label" \
        else [np.arange(n)]
    for g in groups:
        j[g] = cycle_derangement(g, rng)
    assert sorted(j.tolist()) == list(range(n))
    assert not np.any(j == np.arange(n))
    if arm == "within_label":
        assert np.array_equal(y[j], y)
    return j


def build_projector(in_dim: int = 16, out_dim: int = 12):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT),
        nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.ReLU(inplace=True),
    )


def info_nce(audio_projection, metadata, temperature: float = TEMPERATURE):
    import torch
    import torch.nn.functional as F
    a = F.normalize(audio_projection, dim=1)
    m = F.normalize(metadata, dim=1)
    logits = a @ m.T / temperature
    target = torch.arange(len(a), device=a.device)
    return F.cross_entropy(logits, target)


def epoch_order(n: int, seed: int, epoch: int) -> np.ndarray:
    return stable_rng(seed, 1000 + epoch).permutation(n).astype(np.int64)


def torch_seed(seed: int) -> int:
    return 700_000 + int(seed)


def project_numpy(model: Any, x: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            t = torch.as_tensor(x[start:start + batch], dtype=torch.float32, device=device)
            out.append(model(t).cpu().numpy())
    z = np.concatenate(out).astype(np.float32, copy=False)
    assert np.isfinite(z).all() and np.any(z != 0)
    return z


def train_projector(x_train: np.ndarray, metadata: np.ndarray, y: np.ndarray,
                    pairing: np.ndarray, seed: int, epochs: int, batch_size: int,
                    device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    # Reset before every arm.  Initialization, epoch order and dropout stream are thus
    # identical across paired arms; only metadata[pairing] differs.
    torch.manual_seed(torch_seed(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed(seed))
    model = build_projector(x_train.shape[1], metadata.shape[1]).to(device)
    initial_hash = state_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    xt = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    mt = torch.as_tensor(metadata[pairing], dtype=torch.float32, device=device)
    losses, grad_norms = [], []
    order_hash = hashlib.sha256()
    model.train()
    for epoch in range(epochs):
        order = epoch_order(len(x_train), seed, epoch)
        order_hash.update(order.tobytes())
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                raise RuntimeError("InfoNCE final batch has fewer than two samples")
            b = torch.as_tensor(idx, dtype=torch.long, device=device)
            loss = info_nce(model(xt[b]), mt[b])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite InfoNCE loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_sq += float(torch.sum(p.grad.detach() ** 2).cpu())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(math.sqrt(grad_sq))
    z_train = project_numpy(model, x_train, device)
    audit = {
        "initial_state_sha256": initial_hash,
        "final_state_sha256": state_hash(model),
        "epoch_order_sha256": order_hash.hexdigest(),
        "pairing_sha256": sha256_array(pairing),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_last50_mean": float(np.mean(losses[-min(50, len(losses)):])),
        "gradient_norm_first": grad_norms[0],
        "gradient_norm_last": grad_norms[-1],
        "effective_rank": effective_rank(z_train),
        "identical_profile_collision": None if np.array_equal(pairing, np.arange(len(y)))
        else float(np.mean(np.all(metadata[pairing] == metadata, axis=1))),
    }
    assert audit["effective_rank"] > 1.01, "projector collapsed"
    return model, audit


# ---------------------------------------------------------------------------
# Retrieval and representation geometry.

def profile_bank(tiers: np.ndarray, metadata: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes, inv = np.unique(np.asarray(tiers, dtype=np.int8), axis=0, return_inverse=True)
    if metadata is None:
        candidates = tiers_to_metadata(codes)
    else:
        metadata = np.asarray(metadata, dtype=np.float32)
        candidates = []
        for p in range(len(codes)):
            rows = metadata[inv == p]
            assert len(rows) and np.all(rows == rows[0]), \
                "one profile maps to multiple frozen text embeddings"
            candidates.append(rows[0])
        candidates = np.stack(candidates).astype(np.float32)
    return codes, candidates, inv.astype(np.int16)


def retrieval_ranks(z: np.ndarray, tiers: np.ndarray, metadata: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    codes, candidates, true_id = profile_bank(tiers, metadata)
    zu = z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    cu = candidates / np.linalg.norm(candidates, axis=1, keepdims=True).clip(1e-12)
    scores = np.asarray(zu @ cu.T, dtype=np.float64)
    true_score = scores[np.arange(len(scores)), true_id]
    greater = np.sum(scores > true_score[:, None], axis=1)
    # Exact score ties receive their average rank; candidate ordering cannot help a model.
    equal = np.sum(scores == true_score[:, None], axis=1)
    ranks = 1.0 + greater + 0.5 * (equal - 1)
    rr = 1.0 / ranks
    macro = float(np.mean([rr[true_id == p].mean() for p in np.unique(true_id)]))
    out = {
        "n_profiles": int(len(codes)),
        "macro_profile_mrr": macro,
        "micro_mrr": float(rr.mean()),
        "r_at_1": float(np.mean(ranks <= 1.0)),
        "r_at_10": float(np.mean(ranks <= 10.0)),
        "candidate_bank_sha256": hash_arrays((("codes", codes), ("vectors", candidates))),
    }
    return ranks.astype(np.float32), true_id, out


def centered_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, "fro") ** 2
    xx = np.linalg.norm(x.T @ x, "fro")
    yy = np.linalg.norm(y.T @ y, "fro")
    return float(xy / max(xx * yy, 1e-30))


# ---------------------------------------------------------------------------
# Frozen downstream heads and probes.

def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class BinaryHead:
    scaler: Any
    classifier: Any
    calibrator: Any
    chosen_c: float
    selection: dict[str, Any]

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.classifier.decision_function(self.scaler.transform(x)),
                          dtype=np.float64)

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        return self.calibrator.predict_proba(z[:, None])[:, 1].astype(np.float64)


def validation_folds(y: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    folds = np.empty(len(y), dtype=np.int8)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=BOOTSTRAP_SEED + seed)
    for k, (_, held) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[held] = k
    return folds


def fit_binary_head(x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray, seed: int) -> BinaryHead:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    tr = scaler.transform(x_train)
    va = scaler.transform(x_val)
    folds = validation_folds(y_val, seed)
    means, ses, scores, models = {}, {}, {}, {}
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=5000, random_state=seed)
        model.fit(tr, y_train)
        z = model.decision_function(va)
        fold_scores = [auc(y_val[folds == k], z[folds == k]) for k in range(N_FOLDS)]
        means[c] = float(np.mean(fold_scores))
        ses[c] = float(np.std(fold_scores, ddof=1) / math.sqrt(N_FOLDS))
        scores[c] = fold_scores
        models[c] = model
    best = max(C_GRID, key=lambda c: means[c])
    threshold = means[best] - ses[best]
    chosen = min(c for c in C_GRID if means[c] >= threshold)
    model = models[chosen]
    val_logits = model.decision_function(va)
    calibrator = LogisticRegression(max_iter=2000).fit(val_logits[:, None], y_val)
    selection = {
        "mean_auc": {str(k): v for k, v in means.items()},
        "se": {str(k): v for k, v in ses.items()},
        "per_fold": {str(k): v for k, v in scores.items()},
        "best_c": best,
        "one_se_threshold": threshold,
        "chosen_c": chosen,
        "fold_sha256": sha256_array(folds),
    }
    return BinaryHead(scaler, model, calibrator, float(chosen), selection)


@dataclass
class TierProbe:
    scaler: Any
    classifier: Any
    chosen_c: float
    selection: dict[str, Any]

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import balanced_accuracy_score
        pred = self.classifier.predict(self.scaler.transform(x))
        return float(balanced_accuracy_score(y, pred))


def fit_tier_probe(x_train: np.ndarray, y_train: np.ndarray,
                   x_val: np.ndarray, y_val: np.ndarray, seed: int) -> TierProbe:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x_train)
    tr, va = scaler.transform(x_train), scaler.transform(x_val)
    folds = validation_folds(y_val, seed)
    means, ses, per, models = {}, {}, {}, {}
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=5000, random_state=seed)
        model.fit(tr, y_train)
        pred = model.predict(va)
        values = [balanced_accuracy_score(y_val[folds == k], pred[folds == k])
                  for k in range(N_FOLDS)]
        means[c] = float(np.mean(values))
        ses[c] = float(np.std(values, ddof=1) / math.sqrt(N_FOLDS))
        per[c] = [float(v) for v in values]
        models[c] = model
    best = max(C_GRID, key=lambda c: means[c])
    threshold = means[best] - ses[best]
    chosen = min(c for c in C_GRID if means[c] >= threshold)
    return TierProbe(scaler, models[chosen], float(chosen), {
        "mean_balanced_accuracy": {str(k): v for k, v in means.items()},
        "se": {str(k): v for k, v in ses.items()},
        "per_fold": {str(k): v for k, v in per.items()},
        "best_c": best, "one_se_threshold": threshold, "chosen_c": chosen,
        "fold_sha256": sha256_array(folds),
    })


def disease_metrics(y: np.ndarray, logits: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": auc(y, logits),
        "nll": nll(y, p),
        "neg_nll": -nll(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "sharpness_logit_sd": float(np.std(logits)),
    }


def counterfactual_metrics(y_sign: np.ndarray, original: np.ndarray,
                           counterfactual: np.ndarray) -> dict[str, float]:
    delta = original - counterfactual
    return {
        "mean_abs_logit_change": float(np.mean(np.abs(delta))),
        "rms_logit_change": float(np.sqrt(np.mean(delta ** 2))),
        "mean_label_support_removed": float(np.mean(y_sign * delta)),
        "fraction_abs_change_gt_0p1": float(np.mean(np.abs(delta) > 0.1)),
    }


# ---------------------------------------------------------------------------
# One independent simulation.

def make_representations(models: dict[str, Any], x: dict[str, np.ndarray],
                         device: str) -> dict[str, dict[str, np.ndarray]]:
    reps: dict[str, dict[str, np.ndarray]] = {}
    for split, raw in x.items():
        reps[split] = {"raw": raw}
        for arm, model in models.items():
            z = project_numpy(model, raw, device)
            reps[split][arm] = z
            reps[split][f"raw_plus_{arm}"] = np.concatenate([raw, z], axis=1)
    return reps


def counterfactual_representations(models: dict[str, Any], ds: SyntheticSet,
                                   mean: np.ndarray, std: np.ndarray,
                                   device: str, kind: str) -> dict[str, np.ndarray]:
    raw = ((audio_counterfactual(ds, kind) - mean) / std).astype(np.float32)
    out = {"raw": raw}
    for arm, model in models.items():
        z = project_numpy(model, raw, device)
        out[arm] = z
        out[f"raw_plus_{arm}"] = np.concatenate([raw, z], axis=1)
    return out


def simulation_paths(out_dir: Path, tag: str) -> tuple[Path, Path]:
    return (out_dir / "predictions" / f"{tag}.npz",
            out_dir / "metrics" / f"{tag}.json")


def save_checkpoints(models: dict[str, Any], out_dir: Path, tag: str,
                     epochs: int) -> dict[str, str]:
    import torch
    hashes = {}
    for arm, model in models.items():
        path = out_dir / "checkpoints" / f"{tag}_{arm}_epoch{epochs}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
        hashes[arm] = sha256_file(path)
    return hashes


def run_simulation(config: RunConfig, rho: float, kappa: float, seed: int,
                   out_dir: Path, device: str, script_hash: str,
                   resume: bool = False,
                   profile_cache: ProfileEmbeddingCache | None = None) -> dict[str, Any]:
    tag = simulation_tag(rho, kappa, seed)
    prediction_path, metric_path = simulation_paths(out_dir, tag)
    cfg_hash = config_hash(config)
    if prediction_path.exists() or metric_path.exists():
        if not resume or not (prediction_path.exists() and metric_path.exists()):
            raise FileExistsError(f"no-clobber: {prediction_path} or {metric_path} exists")
        with open(metric_path) as f:
            existing = json.load(f)
        if existing.get("config_sha256") != cfg_hash or existing.get("script_sha256") != script_hash:
            raise RuntimeError(f"resume hash mismatch for {tag}")
        if existing.get("prediction_sha256") != sha256_file(prediction_path):
            raise RuntimeError(f"resume prediction hash mismatch for {tag}")
        for arm in ARMS:
            checkpoint = (out_dir / "checkpoints" /
                          f"{tag}_{arm}_epoch{config.epochs}.pt")
            if not checkpoint.exists() or \
                    existing.get("checkpoint_sha256", {}).get(arm) != sha256_file(checkpoint):
                raise RuntimeError(f"resume checkpoint missing/hash mismatch: {tag}/{arm}")
        return existing

    datasets = build_datasets(config, rho, kappa, seed)
    if profile_cache is not None:
        assert config.metadata_variant == "phi2_mask_aware_mean"
        assert config.metadata_cache_sha256 == profile_cache.sha256
        apply_profile_embedding_cache(datasets, profile_cache)
    else:
        assert config.metadata_variant == "one_hot"
    x, audio_mean, audio_std = standardize_audio(datasets)
    train = datasets["train"]
    input_hash = hash_arrays(
        [(f"{name}_y", ds.y) for name, ds in datasets.items()] +
        [(f"{name}_x", ds.x_raw) for name, ds in datasets.items()] +
        [(f"{name}_metadata", ds.metadata) for name, ds in datasets.items()]
    )

    models, train_audit, pairings = {}, {}, {}
    for arm in ARMS:
        pairing = build_pairing(train.y, arm, seed)
        pairings[arm] = pairing
        model, audit = train_projector(x["train"], train.metadata, train.y, pairing,
                                       seed, config.epochs, config.batch_size, device)
        models[arm], train_audit[arm] = model, audit
    initial = {a["initial_state_sha256"] for a in train_audit.values()}
    orders = {a["epoch_order_sha256"] for a in train_audit.values()}
    assert len(initial) == 1, "paired arms did not share initialization"
    assert len(orders) == 1, "paired arms did not share batch order"

    reps = make_representations(models, x, device)
    rep_hashes = {split: {arm: sha256_array(value) for arm, value in values.items()}
                  for split, values in reps.items()}

    # Retrieval is computed before any disease head and uses the unique profile bank of
    # that evaluation set.  It never uses the disease label.
    retrieval: dict[str, dict[str, Any]] = {}
    rank_arrays: dict[str, np.ndarray] = {}
    profile_arrays: dict[str, np.ndarray] = {}
    for endpoint in ENDPOINTS:
        retrieval[endpoint] = {}
        for arm in ALIGNED_REPS:
            ranks, true_id, met = retrieval_ranks(
                reps[endpoint][arm], datasets[endpoint].tiers,
                datasets[endpoint].metadata)
            retrieval[endpoint][arm] = met
            rank_arrays[f"rank__{endpoint}__{arm}"] = ranks
            profile_arrays[f"profile_id__{endpoint}"] = true_id

    # One frozen source-trained disease head for every representation.
    heads, head_audit = {}, {}
    disease: dict[str, dict[str, Any]] = {ep: {} for ep in ENDPOINTS}
    prediction_arrays: dict[str, np.ndarray] = {}
    for rep in REPRESENTATIONS:
        head = fit_binary_head(reps["train"][rep], train.y,
                               reps["val"][rep], datasets["val"].y, seed)
        heads[rep] = head
        head_audit[rep] = {"chosen_c": head.chosen_c, "selection": head.selection}
        for endpoint in ENDPOINTS:
            logits = head.logits(reps[endpoint][rep])
            probs = head.probabilities(reps[endpoint][rep])
            disease[endpoint][rep] = disease_metrics(datasets[endpoint].y, logits, probs)
            prediction_arrays[f"logits__{endpoint}__{rep}"] = logits.astype(np.float32)
            prediction_arrays[f"prob__{endpoint}__{rep}"] = probs.astype(np.float32)

    # S-tier probes are deliberately kept separate for s_1..s_4; no composite score.
    probes: dict[str, dict[str, dict[str, float]]] = {}
    probe_audit: dict[str, Any] = {}
    for rep in PROBE_REPS:
        probes[rep] = {}
        probe_audit[rep] = {}
        for j in range(4):
            probe = fit_tier_probe(reps["train"][rep], train.tiers[:, j],
                                   reps["val"][rep], datasets["val"].tiers[:, j],
                                   seed + 100 * j)
            name = f"s{j + 1}_tier"
            probes[rep][name] = {
                endpoint: probe.score(reps[endpoint][rep], datasets[endpoint].tiers[:, j])
                for endpoint in ENDPOINTS
            }
            probe_audit[rep][name] = {
                "chosen_c": probe.chosen_c, "selection": probe.selection,
            }

    # CKA is descriptive only.  It is computed on source and the fully deconfounded end.
    cka: dict[str, dict[str, dict[str, float]]] = {}
    for endpoint in ("source_test", "target_zero"):
        cka[endpoint] = {}
        for rep in PROBE_REPS:
            cka[endpoint][rep] = {
                "disease_y": centered_linear_cka(reps[endpoint][rep],
                                                  datasets[endpoint].y_sign),
                "shortcut_s": centered_linear_cka(reps[endpoint][rep],
                                                   datasets[endpoint].s),
            }

    # Counterfactual use test: same y, D coordinate and noises; alter only S.
    counterfactual: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for endpoint in ENDPOINTS:
        counterfactual[endpoint] = {}
        cf = {kind: counterfactual_representations(
            models, datasets[endpoint], audio_mean, audio_std, device, kind)
              for kind in ("flip", "resample")}
        for rep in REPRESENTATIONS:
            original = prediction_arrays[f"logits__{endpoint}__{rep}"].astype(np.float64)
            counterfactual[endpoint][rep] = {}
            for kind in ("flip", "resample"):
                cf_logits = heads[rep].logits(cf[kind][rep])
                prediction_arrays[f"cf_{kind}_logits__{endpoint}__{rep}"] = \
                    cf_logits.astype(np.float32)
                counterfactual[endpoint][rep][kind] = counterfactual_metrics(
                    datasets[endpoint].y_sign, original, cf_logits)

    prediction_arrays.update(rank_arrays)
    prediction_arrays.update(profile_arrays)
    for endpoint in ENDPOINTS:
        prediction_arrays[f"y__{endpoint}"] = datasets[endpoint].y
        prediction_arrays[f"tiers__{endpoint}"] = datasets[endpoint].tiers
    prediction_arrays.update({
        "config_sha256": np.asarray(cfg_hash),
        "script_sha256": np.asarray(script_hash),
        "input_sha256": np.asarray(input_hash),
        "rho_s": np.asarray(rho),
        "kappa": np.asarray(kappa),
        "simulation_seed": np.asarray(seed),
    })

    # File discipline: predictions first, then final checkpoints, then metrics.
    atomic_npz(prediction_path, **prediction_arrays)
    checkpoint_hashes = save_checkpoints(models, out_dir, tag, config.epochs)
    result = {
        "tag": tag, "mode": config.mode, "rho_s": rho, "kappa": kappa, "seed": seed,
        "config": asdict(config), "config_sha256": cfg_hash,
        "script_sha256": script_hash, "input_sha256": input_hash,
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": checkpoint_hashes,
        "audio_standardization": {
            "mean_sha256": sha256_array(audio_mean), "std_sha256": sha256_array(audio_std),
        },
        "pairing_sha256": {arm: sha256_array(p) for arm, p in pairings.items()},
        "training": train_audit, "representation_sha256": rep_hashes,
        "retrieval": retrieval, "disease": disease,
        "head_audit": head_audit, "probes": probes, "probe_audit": probe_audit,
        "cka": cka, "counterfactual": counterfactual,
    }
    atomic_json(result, metric_path)
    return result


# ---------------------------------------------------------------------------
# Across-simulation summaries, paired CIs and frozen gates.

def bootstrap_ci(values: Iterable[float], boot: int = BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        raise ValueError("empty bootstrap input")
    observed = float(np.mean(values))
    rng = np.random.RandomState(seed)
    draws = np.mean(values[rng.randint(0, len(values), size=(boot, len(values)))], axis=1)
    return {
        "observed": observed,
        "ci": [float(v) for v in np.percentile(draws, [2.5, 97.5])],
        "per_seed": values.tolist(),
        "n_seeds": int(len(values)),
    }


def load_all_results(config: RunConfig, out_dir: Path, script_hash: str) -> list[dict]:
    out = []
    for rho in config.rhos:
        for kappa in config.kappas:
            for seed in config.seeds:
                tag = simulation_tag(rho, kappa, seed)
                pred, met = simulation_paths(out_dir, tag)
                if not pred.exists() or not met.exists():
                    raise FileNotFoundError(f"formal simulation incomplete: {tag}")
                with open(met) as f:
                    result = json.load(f)
                if result["config_sha256"] != config_hash(config):
                    raise RuntimeError(f"config hash mismatch: {tag}")
                if result["script_sha256"] != script_hash:
                    raise RuntimeError(f"script hash mismatch: {tag}")
                if result["prediction_sha256"] != sha256_file(pred):
                    raise RuntimeError(f"prediction hash mismatch: {tag}")
                out.append(result)
    return out


def cell_results(results: list[dict], rho: float, kappa: float) -> list[dict]:
    return sorted([r for r in results if r["rho_s"] == rho and r["kappa"] == kappa],
                  key=lambda r: r["seed"])


def delta_values(cell: list[dict], getter, arm_a: str = "correct",
                 arm_b: str = "within_label") -> list[float]:
    return [float(getter(r, arm_a) - getter(r, arm_b)) for r in cell]


def monotone_non_decreasing(values: list[float], tolerance: float = 0.0) -> bool:
    return bool(np.all(np.diff(np.asarray(values)) >= -tolerance))


def summarize(config: RunConfig, out_dir: Path, script_hash: str) -> dict[str, Any]:
    results = load_all_results(config, out_dir, script_hash)
    cells: dict[str, Any] = {}
    for rho in config.rhos:
        for kappa in config.kappas:
            cell = cell_results(results, rho, kappa)
            key = f"rho={rho:.2f}|kappa={kappa:.2f}"
            get_mrr = lambda r, a: r["retrieval"]["source_test"][a]["macro_profile_mrr"]
            get_src_auc = lambda r, a: r["disease"]["source_test"][a]["auroc"]
            get_tgt_auc = lambda r, a: r["disease"]["target_zero"][a]["auroc"]
            get_tgt_nnll = lambda r, a: r["disease"]["target_zero"][a]["neg_nll"]
            cells[key] = {
                "rho_s": rho, "kappa": kappa,
                "correspondence_gain_mrr_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_mrr), config.bootstrap),
                "source_delta_auc_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_src_auc), config.bootstrap),
                "target_zero_delta_auc_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_tgt_auc), config.bootstrap),
                "target_zero_delta_neg_nll_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_tgt_nnll), config.bootstrap),
                "raw_preserving_target_zero_delta_auc_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_tgt_auc,
                                 "raw_plus_correct", "raw_plus_within_label"),
                    config.bootstrap),
                "raw_preserving_target_zero_delta_neg_nll_C_minus_W": bootstrap_ci(
                    delta_values(cell, get_tgt_nnll,
                                 "raw_plus_correct", "raw_plus_within_label"),
                    config.bootstrap),
                "per_seed_metric_files": [r["tag"] for r in cell],
            }
            # Separate S-tier deltas, as frozen; never average into a composite score.
            cells[key]["s_tier_probe_balanced_accuracy"] = {}
            for endpoint in ("source_test", "target_zero"):
                cells[key]["s_tier_probe_balanced_accuracy"][endpoint] = {}
                for j in range(1, 5):
                    name = f"s{j}_tier"
                    c_w = [r["probes"]["correct"][name][endpoint] -
                           r["probes"]["within_label"][name][endpoint] for r in cell]
                    w_g = [r["probes"]["within_label"][name][endpoint] -
                           r["probes"]["global"][name][endpoint] for r in cell]
                    cells[key]["s_tier_probe_balanced_accuracy"][endpoint][name] = {
                        "C_minus_W": bootstrap_ci(c_w, config.bootstrap),
                        "W_minus_G": bootstrap_ci(w_g, config.bootstrap),
                    }

            # Counterfactual use is also kept factorised by endpoint, intervention and
            # statistic.  These are paired representation deltas, not a composite score.
            cells[key]["counterfactual_use"] = {}
            for endpoint in ("source_test", "target_zero"):
                cells[key]["counterfactual_use"][endpoint] = {}
                for kind in ("flip", "resample"):
                    cells[key]["counterfactual_use"][endpoint][kind] = {}
                    for statistic in ("mean_abs_logit_change", "mean_label_support_removed"):
                        c_w = [r["counterfactual"][endpoint]["correct"][kind][statistic] -
                               r["counterfactual"][endpoint]["within_label"][kind][statistic]
                               for r in cell]
                        w_g = [r["counterfactual"][endpoint]["within_label"][kind][statistic] -
                               r["counterfactual"][endpoint]["global"][kind][statistic]
                               for r in cell]
                        cells[key]["counterfactual_use"][endpoint][kind][statistic] = {
                            "C_minus_W": bootstrap_ci(c_w, config.bootstrap),
                            "W_minus_G": bootstrap_ci(w_g, config.bootstrap),
                        }

            cells[key]["cka_descriptive"] = {}
            for endpoint in ("source_test", "target_zero"):
                cells[key]["cka_descriptive"][endpoint] = {}
                for target in ("disease_y", "shortcut_s"):
                    cells[key]["cka_descriptive"][endpoint][target] = {
                        arm: bootstrap_ci(
                            [r["cka"][endpoint][arm][target] for r in cell],
                            config.bootstrap)
                        for arm in PROBE_REPS
                    }

    # Frozen main-text admission gates.  Gate 5 contains explicit diagnostics but remains
    # manual because the preregistration calls for an "interpretable boundary" rather than
    # defining a scalar equivalence test.  The script never silently promotes it.
    def c(rho: float, kappa: float) -> dict:
        return cells[f"rho={rho:.2f}|kappa={kappa:.2f}"]

    gate1_checks = []
    for rho in config.rhos:
        for kappa in config.kappas:
            if kappa >= 1.0:
                ci = c(rho, kappa)["correspondence_gain_mrr_C_minus_W"]["ci"]
                gate1_checks.append({"rho": rho, "kappa": kappa, "ci": ci,
                                     "pass": ci[0] > 0})

    high = [(rho, kappa) for rho in config.rhos for kappa in config.kappas
            if rho == 0.95 and kappa >= 1.0]
    gate2_checks = []
    for rho, kappa in high:
        src = c(rho, kappa)["source_delta_auc_C_minus_W"]["ci"]
        tgt = c(rho, kappa)["target_zero_delta_neg_nll_C_minus_W"]["ci"]
        gate2_checks.append({"rho": rho, "kappa": kappa,
                             "source_delta_auc_ci": src,
                             "target_delta_neg_nll_ci": tgt,
                             "pass": src[0] > 0 and tgt[1] < 0})

    # Dose trend is operationalised before reading formal results as monotonic endpoint
    # NLL regret (-Delta[-NLL]) across every nontrivial row and column.
    trend_checks = []
    for kappa in config.kappas:
        if kappa >= 1.0:
            vals = [-c(r, kappa)["target_zero_delta_neg_nll_C_minus_W"]["observed"]
                    for r in config.rhos]
            trend_checks.append({"axis": "rho", "fixed": kappa, "regret": vals,
                                 "pass": monotone_non_decreasing(vals)})
    for rho in config.rhos:
        if rho >= 0.50:
            vals = [-c(rho, k)["target_zero_delta_neg_nll_C_minus_W"]["observed"]
                    for k in config.kappas]
            trend_checks.append({"axis": "kappa", "fixed": rho, "regret": vals,
                                 "pass": monotone_non_decreasing(vals)})

    rope_checks = []
    if 0.0 in config.rhos:
        for kappa in config.kappas:
            aci = c(0.0, kappa)["target_zero_delta_auc_C_minus_W"]["ci"]
            nci = c(0.0, kappa)["target_zero_delta_neg_nll_C_minus_W"]["ci"]
            ok = aci[0] > -0.005 and aci[1] < 0.005 and nci[0] > -0.01 and nci[1] < 0.01
            rope_checks.append({"kappa": kappa, "delta_auc_ci": aci,
                                "delta_neg_nll_ci": nci, "pass": ok})

    raw_boundary = []
    for rho, kappa in high:
        aligned = c(rho, kappa)["target_zero_delta_neg_nll_C_minus_W"]["observed"]
        preserved = c(rho, kappa)[
            "raw_preserving_target_zero_delta_neg_nll_C_minus_W"]["observed"]
        raw_boundary.append({
            "rho": rho, "kappa": kappa,
            "aligned_delta_neg_nll": aligned,
            "raw_preserving_delta_neg_nll": preserved,
            "absolute_regret_reduced": abs(preserved) <= abs(aligned),
        })

    gates = {
        "gate1_correspondence": {"checks": gate1_checks,
                                  "pass": bool(gate1_checks) and all(x["pass"] for x in gate1_checks)},
        "gate2_high_confounding": {"checks": gate2_checks,
                                    "pass": bool(gate2_checks) and all(x["pass"] for x in gate2_checks)},
        "gate3_dose_trend": {"operational_definition":
                             "endpoint NLL regret is non-decreasing across all frozen rho/kappa rows",
                             "checks": trend_checks,
                             "pass": bool(trend_checks) and all(x["pass"] for x in trend_checks)},
        "gate4_rho_zero_rope": {"checks": rope_checks,
                                "pass": bool(rope_checks) and all(x["pass"] for x in rope_checks)},
        "gate5_raw_preserving_boundary": {
            "checks": raw_boundary,
            "automatic_pass": None,
            "manual_review_required": True,
            "reason": "the frozen preregistration requires an interpretable boundary, not a scalar gate",
        },
    }
    automatic_1_to_4 = all(gates[k]["pass"] for k in (
        "gate1_correspondence", "gate2_high_confounding", "gate3_dose_trend",
        "gate4_rho_zero_rope"))
    summary = {
        "mode": config.mode, "config": asdict(config),
        "config_sha256": config_hash(config), "script_sha256": script_hash,
        "n_simulations": len(results), "cells": cells, "main_text_gates": gates,
        "automatic_gates_1_to_4_pass": automatic_1_to_4,
        "main_text_eligible": None,
        "main_text_eligibility_note":
            "never automatic: gates 1-4 must pass and frozen gate 5 requires documented manual review",
    }
    path = out_dir / "summary.json"
    if path.exists():
        raise FileExistsError(f"no-clobber: {path}")
    atomic_json(summary, path)
    return summary


# ---------------------------------------------------------------------------
# Self-test and CLI.

def manual_info_nce(a: np.ndarray, m: np.ndarray, tau: float) -> float:
    au = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    mu = m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-12)
    logits = au @ mu.T / tau
    mx = logits.max(axis=1, keepdims=True)
    logsumexp = mx[:, 0] + np.log(np.exp(logits - mx).sum(axis=1))
    return float(np.mean(logsumexp - np.diag(logits)))


def self_test() -> None:
    import torch

    ds = generate_set(128, 0.36, 0.5, 1.0, seed=7, stream=1)
    assert ds.x_raw.shape == (128, 16) and ds.metadata.shape == (128, 12)
    assert np.allclose(np.linalg.norm(ds.metadata, axis=1), 1.0)
    for arm in ARMS:
        j = build_pairing(ds.y, arm, 7)
        if arm == "correct":
            assert np.array_equal(j, np.arange(len(j)))
        else:
            assert not np.any(j == np.arange(len(j)))
        if arm == "within_label":
            assert np.array_equal(ds.y[j], ds.y)

    torch.manual_seed(1)
    a = torch.randn(16, 12)
    m = torch.randn(16, 12)
    framework = float(info_nce(a, m).detach())
    hand = manual_info_nce(a.numpy(), m.numpy(), TEMPERATURE)
    assert abs(framework - hand) < 2e-5, (framework, hand)

    for kind in ("flip", "resample"):
        cf = audio_counterfactual(ds, kind)
        assert np.array_equal(cf[:, 0], ds.x_raw[:, 0])
        assert np.array_equal(cf[:, 5:], ds.x_raw[:, 5:])

    z = np.random.RandomState(3).normal(size=(128, 12)).astype(np.float32)
    ranks, ids, met = retrieval_ranks(z, ds.tiers)
    assert len(ranks) == len(ids) == 128 and np.all(ranks >= 1)
    assert 0 <= met["macro_profile_mrr"] <= 1

    values = [-0.1, 0.0, 0.1]
    ci = bootstrap_ci(values, boot=100, seed=1)
    assert abs(ci["observed"]) < 1e-12 and len(ci["ci"]) == 2

    # Optional language-faithful cache must cover all 81 profiles and map repeated
    # profiles to one immutable vector.
    with tempfile.TemporaryDirectory() as cache_td:
        codes = np.asarray(np.meshgrid(*([np.arange(3)] * 4))).reshape(4, -1).T
        emb = np.random.RandomState(8).normal(size=(81, 7)).astype(np.float32)
        cache_path = Path(cache_td) / "profiles.npz"
        np.savez(cache_path, profile_codes=codes, embeddings=emb)
        cache = load_profile_embedding_cache(cache_path)
        cache_sets = {"x": generate_set(96, 0.5, 0.95, 2.0, 0, 70)}
        apply_profile_embedding_cache(cache_sets, cache)
        assert cache_sets["x"].metadata.shape == (96, 7)
        _, candidate, _ = profile_bank(cache_sets["x"].tiers,
                                       cache_sets["x"].metadata)
        assert candidate.shape[1] == 7

    # A genuinely end-to-end two-epoch run in a temporary directory verifies prediction-
    # first output, checkpoint hashes, heads, probes and counterfactuals.
    tiny = RunConfig(
        mode="self_test", rhos=(0.5,), kappas=(1.0,), seeds=(0,),
        n_train=128, n_val=80, n_source_test=96, n_target=96,
        epochs=2, batch_size=64, bootstrap=50,
    )
    with tempfile.TemporaryDirectory() as td:
        script_hash = sha256_file(Path(__file__))
        result = run_simulation(tiny, 0.5, 1.0, 0, Path(td), "cpu", script_hash)
        pred, met_path = simulation_paths(Path(td), result["tag"])
        assert pred.exists() and met_path.exists()
        with np.load(pred) as p:
            assert p["logits__target_zero__correct"].shape == (96,)
            assert p["rank__source_test__correct"].shape == (96,)
        # Shared-stream assertions are also made in run_simulation; verify saved form.
        assert len({v["initial_state_sha256"] for v in result["training"].values()}) == 1
        assert all(v["effective_rank"] > 1 for v in result["training"].values())
    print("SELF-TEST PASS: generator, derangements, hand InfoNCE, retrieval, "
          "counterfactual invariants, shared streams, heads, probes and output hashes")


def resolve_device(requested: str) -> str:
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true",
                    help="skip only complete, hash-matching simulations")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--skip-summary", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument(
        "--language-profile-cache", type=Path,
        help=("optional frozen Phi-2 cache for the preregistered sensitivity; requires "
              "formal mode and runs only rho=0.95,kappa=2.0"),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.out_dir is None:
        raise SystemExit("--out-dir is required outside --self-test")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("require 0 <= shard-index < num-shards")
    profile_cache = None
    if args.language_profile_cache is not None:
        if args.mode != "formal":
            raise SystemExit("--language-profile-cache is allowed only with --mode formal")
        profile_cache = load_profile_embedding_cache(args.language_profile_cache)
        config = replace(
            FORMAL_CONFIG,
            mode="formal_language_sensitivity",
            rhos=(0.95,), kappas=(2.0,),
            projector_out=int(profile_cache.embeddings.shape[1]),
            metadata_variant="phi2_mask_aware_mean",
            metadata_cache_sha256=profile_cache.sha256,
        )
    else:
        config = FORMAL_CONFIG if args.mode == "formal" else SMOKE_CONFIG
    script_hash = sha256_file(Path(__file__))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "config.json"
    manifest = {
        "config": asdict(config), "config_sha256": config_hash(config),
        "script": str(Path(__file__).resolve()), "script_sha256": script_hash,
        "formal_configuration_is_not_cli_overridable": True,
        "source_doc": "docs/SYNTHETIC_CORRESPONDENCE_TRANSFER_PREREG_ZH.md",
        "language_profile_cache": (str(args.language_profile_cache.resolve())
                                   if args.language_profile_cache else None),
    }
    if manifest_path.exists():
        with open(manifest_path) as f:
            old = json.load(f)
        if old["config_sha256"] != manifest["config_sha256"] or \
                old["script_sha256"] != script_hash:
            raise RuntimeError("output directory belongs to a different config/script hash")
    else:
        atomic_json(manifest, manifest_path)

    if args.summarize_only:
        summary = summarize(config, args.out_dir, script_hash)
        print(json.dumps({"summary": str(args.out_dir / "summary.json"),
                          "automatic_gates_1_to_4_pass":
                              summary["automatic_gates_1_to_4_pass"]}, indent=2))
        return

    device = resolve_device(args.device)
    jobs = [(rho, kappa, seed) for rho in config.rhos for kappa in config.kappas
            for seed in config.seeds]
    selected = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_index]
    print(f"mode={config.mode} device={device} shard={args.shard_index}/{args.num_shards} "
          f"jobs={len(selected)}/{len(jobs)} config_sha256={config_hash(config)[:16]}")
    for number, (rho, kappa, seed) in enumerate(selected, 1):
        print(f"[{number}/{len(selected)}] rho={rho:.2f} kappa={kappa:.2f} seed={seed}",
              flush=True)
        result = run_simulation(config, rho, kappa, seed, args.out_dir, device,
                                script_hash, resume=args.resume,
                                profile_cache=profile_cache)
        print(f"  {result['tag']} complete: "
              f"CG={result['retrieval']['source_test']['correct']['macro_profile_mrr'] - result['retrieval']['source_test']['within_label']['macro_profile_mrr']:+.4f}",
              flush=True)
    if not args.skip_summary and args.num_shards == 1:
        summary = summarize(config, args.out_dir, script_hash)
        print(f"summary={args.out_dir / 'summary.json'} "
              f"automatic_gates_1_to_4_pass={summary['automatic_gates_1_to_4_pass']}")
    elif not args.skip_summary:
        print("sharded run complete for this shard; run --summarize-only after all shards")


if __name__ == "__main__":
    main()
