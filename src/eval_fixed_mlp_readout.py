"""Frozen nonlinear-readout sensitivity for the Route-A ICASSP audit.

This script implements E2 from ``docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md``.
It deliberately separates fitting from scoring:

* ``--fit`` opens Standard train/validation labels, trains the fixed MLP and writes
  full-cohort logits/probabilities plus immutable fit metadata.  It never computes a
  Standard/matched/matched-long metric.
* ``--score`` requires a complete fit manifest, verifies every prediction file, then
  explicitly unlocks the three exploratory test sets.
* ``--self-test`` uses synthetic data only.

The architecture and optimiser are not command-line choices.  Changing them requires a
new preregistration rather than another invocation of this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import ARTEFACT_FEATS, UNIT, auroc, calib_diag, logloss
from eval_matched_direct_fusion import build_metadata_matrix
from eval_metadata_alignment import (
    atomic_json,
    atomic_npz,
    hierarchical_ci,
    load_cohort,
    load_representation,
    paired_hierarchical_ci,
    sha16,
    verify_training_manifest,
)


SEEDS = (0, 1, 2, 3, 4)
INNER_SPLIT_SEED = 20260820
BOOTSTRAP_SEED = 20260820
HIDDEN = 128
DROPOUT = 0.20
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 100
MIN_EPOCHS = 10
PATIENCE = 15
MIN_DELTA = 1e-4
GRAD_CLIP = 5.0

COMMON_ARMS = ("metadata_only", "artifacts_only", "metadata_artifacts")
BACKBONE_ARMS = (
    "audio_raw", "audio_correct", "audio_within_label", "audio_global",
    "metadata_raw", "metadata_correct", "metadata_within_label", "metadata_global",
)
ALL_ARMS = COMMON_ARMS + BACKBONE_ARMS
ALIGNED = {"correct", "within_label", "global"}

COMPARISONS = (
    ("audio_correct", "audio_within_label", "audio_individual_correspondence_primary"),
    ("metadata_correct", "metadata_within_label", "fusion_individual_correspondence"),
    ("metadata_raw", "metadata_only", "direct_audio_increment"),
    ("audio_correct", "audio_raw", "aligned_versus_raw_audio"),
    ("metadata_correct", "metadata_raw", "aligned_versus_raw_fusion"),
    ("audio_within_label", "audio_global", "audio_label_level_cooccurrence"),
    ("metadata_within_label", "metadata_global", "fusion_label_level_cooccurrence"),
    ("metadata_artifacts", "metadata_only", "metadata_artifact_increment"),
)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk):
            h.update(block)
    return h.hexdigest()


def state_sha16(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_inner_split(d: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen 90/10 split of Standard train, stratified by label x source."""
    from sklearn.model_selection import StratifiedShuffleSplit

    idx = np.where(train)[0]
    strata = (d.loc[idx, "y"].astype(str) + "|" +
              d.loc[idx, "recruitment_source"].fillna("NA").astype(str)).to_numpy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                   random_state=INNER_SPLIT_SEED)
    fit_local, stop_local = next(split.split(idx, strata))
    fit = np.zeros(len(d), dtype=bool)
    stop = np.zeros(len(d), dtype=bool)
    fit[idx[fit_local]] = True
    stop[idx[stop_local]] = True
    assert not np.any(fit & stop)
    assert np.array_equal(fit | stop, train)
    return fit, stop


def fit_scaler(X: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(X[train], dtype=np.float64).mean(axis=0)
    std = np.asarray(X[train], dtype=np.float64).std(axis=0)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (np.asarray(X, dtype=np.float32) - mean) / std
    assert np.isfinite(out).all()
    return out.astype(np.float32, copy=False)


def build_model(d_in: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(d_in, HIDDEN),
        nn.LayerNorm(HIDDEN),
        nn.GELU(),
        nn.Dropout(DROPOUT),
        nn.Linear(HIDDEN, 1),
    )


def epoch_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """Arm-independent minibatch order for a given readout seed and epoch."""
    rng = np.random.RandomState(10_000_019 * seed + epoch + 20260820)
    return indices[rng.permutation(len(indices))]


def train_epochs(model, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                 epochs: int, seed: int, device: str,
                 stop_indices: np.ndarray | None = None,
                 early_stop: bool = False) -> tuple[list[float], list[float]]:
    import torch
    import torch.nn.functional as F

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_losses: list[float] = []
    stop_losses: list[float] = []
    best_stop = float("inf")
    stale = 0
    # Reset dropout independently of input-layer size.  Paired arms of equal shape now
    # share initialisation, minibatches and dropout stream exactly.
    torch.manual_seed(90_000 + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(90_000 + seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = epoch_order(indices, seed, epoch)
        total, count = 0.0, 0
        for start in range(0, len(order), BATCH_SIZE):
            ii = order[start:start + BATCH_SIZE]
            xb = torch.from_numpy(X[ii]).to(device)
            yb = torch.from_numpy(y[ii].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ii)
            count += len(ii)
        train_losses.append(total / count)
        if stop_indices is not None:
            stop_logits = predict_logits(model, X, stop_indices, device)
            p = 1.0 / (1.0 + np.exp(-np.clip(stop_logits, -30, 30)))
            current = float(logloss(y[stop_indices], p).mean())
            stop_losses.append(current)
            if current < best_stop - MIN_DELTA:
                best_stop, stale = current, 0
            else:
                stale += 1
            if early_stop and epoch >= MIN_EPOCHS and stale >= PATIENCE:
                break
    return train_losses, stop_losses


def predict_logits(model, X: np.ndarray, indices: np.ndarray | None, device: str) -> np.ndarray:
    import torch
    model.eval()
    if indices is None:
        indices = np.arange(len(X))
    out = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), 2048):
            ii = indices[start:start + 2048]
            xb = torch.from_numpy(X[ii]).to(device)
            out[start:start + len(ii)] = model(xb).squeeze(1).cpu().numpy()
    assert np.isfinite(out).all()
    return out


@dataclass
class FitRecord:
    arm: str
    backbone: str
    seed: int
    input_dim: int
    best_epoch: int
    best_stop_bce: float
    init_hash: str
    final_hash: str
    feature_hash: str
    mean_hash: str
    std_hash: str
    train_loss_first: float
    train_loss_last: float
    calibrator_coef: float
    calibrator_intercept: float


def fit_one(X0: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
            inner_fit: np.ndarray, inner_stop: np.ndarray, arm: str, backbone: str,
            seed: int, device: str) -> tuple[np.ndarray, np.ndarray, FitRecord]:
    from sklearn.linear_model import LogisticRegression

    mean, std = fit_scaler(X0, train)
    X = transform(X0, mean, std)
    seed_everything(seed)
    probe = build_model(X.shape[1])
    init_hash = state_sha16(probe)
    train_loss, stop_loss = train_epochs(
        probe, X, y, np.where(inner_fit)[0], MAX_EPOCHS, seed, device,
        np.where(inner_stop)[0], early_stop=True)
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    for epoch, loss in enumerate(stop_loss, start=1):
        if loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    # The exploratory trajectory may have run to 100, but the formal full-train model is
    # reset to the identical initialisation and trained exactly best_epoch steps.
    seed_everything(seed)
    final = build_model(X.shape[1])
    assert state_sha16(final) == init_hash
    final_train, _ = train_epochs(
        final, X, y, np.where(train)[0], best_epoch, seed, device, None)
    raw_logits = predict_logits(final, X, None, device).astype(np.float64)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(raw_logits[val, None], y[val])
    prob = calibrator.predict_proba(raw_logits[:, None])[:, 1].astype(np.float64)
    assert np.isfinite(prob).all() and np.ptp(prob) > 0
    record = FitRecord(
        arm=arm, backbone=backbone, seed=seed, input_dim=int(X.shape[1]),
        best_epoch=int(best_epoch), best_stop_bce=float(best_loss),
        init_hash=init_hash, final_hash=state_sha16(final), feature_hash=sha16(X0),
        mean_hash=sha16(mean), std_hash=sha16(std),
        train_loss_first=float(final_train[0]), train_loss_last=float(final_train[-1]),
        calibrator_coef=float(calibrator.coef_[0, 0]),
        calibrator_intercept=float(calibrator.intercept_[0]),
    )
    return raw_logits, prob, record


def load_artifacts(path: Path, participants: np.ndarray) -> np.ndarray:
    d = pd.read_csv(path)
    assert d[UNIT].is_unique
    order = pd.DataFrame({UNIT: participants})
    d = order.merge(d[[UNIT] + ARTEFACT_FEATS], on=UNIT, how="left", validate="one_to_one")
    assert not d[ARTEFACT_FEATS].isna().any().any()
    return d[ARTEFACT_FEATS].to_numpy(np.float32)


def feature_for_arm(arm: str, M: np.ndarray, A: np.ndarray,
                    reps: dict[str, np.ndarray]) -> np.ndarray:
    if arm == "metadata_only":
        return M
    if arm == "artifacts_only":
        return A
    if arm == "metadata_artifacts":
        return np.concatenate([M, A], axis=1)
    prefix, suffix = arm.split("_", 1)
    key = "raw_ast" if suffix == "raw" else suffix
    R = reps[key]
    if prefix == "audio":
        return R
    if prefix == "metadata":
        return np.concatenate([M, R], axis=1)
    raise KeyError(arm)


def completed_fit(path: Path, participants: np.ndarray, arm: str,
                  backbone: str, seed: int) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants)
        assert str(z["arm"].item()) == arm
        assert str(z["backbone"].item()) == backbone
        assert int(z["seed"].item()) == seed
        assert z["raw_logits"].shape == participants.shape
        assert z["calibrated_prob"].shape == participants.shape
        assert np.isfinite(z["raw_logits"]).all()
        assert np.isfinite(z["calibrated_prob"]).all()
    return True


def run_fit(args) -> None:
    seeds = tuple(args.seeds)
    assert seeds == SEEDS, "formal run requires exactly seeds 0..4"
    d, train, val, _ = load_cohort(args)
    participants = d[UNIT].to_numpy()
    y = d["y"].to_numpy(np.int64)
    inner_fit, inner_stop = make_inner_split(d, train)
    M, _, metadata_audit = build_metadata_matrix(
        Path(args.data), Path(args.cohort), participants)
    A = load_artifacts(Path(args.artifacts), participants)
    manifest = verify_training_manifest(
        Path(args.alignment_dir) / "manifest.json", list(seeds), 500)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "manifest.json").exists(), "complete manifest already exists"
    selected_arms = list(BACKBONE_ARMS) + (list(COMMON_ARMS) if args.include_common else [])
    raw_cache: dict[str, np.ndarray] = {}
    records = []
    for arm in selected_arms:
        for seed in seeds:
            target = out / f"fit_{arm}_seed{seed}.npz"
            if completed_fit(target, participants, arm, args.backbone, seed):
                print(f"resume verified {target}", flush=True)
                with np.load(target, allow_pickle=True) as z:
                    records.append(json.loads(str(z["fit_record"].item())))
                continue
            reps: dict[str, np.ndarray] = {}
            if arm not in COMMON_ARMS:
                suffix = arm.split("_", 1)[1]
                key = "raw_ast" if suffix == "raw" else suffix
                reps[key] = load_representation(
                    args, participants, key, seed, "raw", raw_cache, manifest)
            X = feature_for_arm(arm, M, A, reps)
            print(f"FIT {args.backbone} {arm} seed={seed} shape={X.shape}", flush=True)
            logits, prob, record = fit_one(
                X, y, train, val, inner_fit, inner_stop, arm, args.backbone, seed,
                args.device)
            atomic_npz(
                target, participants=participants, raw_logits=logits,
                calibrated_prob=prob, arm=np.asarray(arm),
                backbone=np.asarray(args.backbone), seed=np.asarray(seed),
                fit_record=np.asarray(json.dumps(asdict(record), sort_keys=True)),
            )
            records.append(asdict(record))
            del X, reps, logits, prob
    # Paired arms with equal input geometry must begin from the same state.
    by_seed_arm = {(r["seed"], r["arm"]): r for r in records}
    for seed in seeds:
        for group in (("audio_correct", "audio_within_label", "audio_global"),
                      ("metadata_correct", "metadata_within_label", "metadata_global")):
            hashes = {by_seed_arm[(seed, arm)]["init_hash"] for arm in group}
            assert len(hashes) == 1, f"paired init mismatch seed={seed} group={group}"
    payload = {
        "status": "complete_predictions_frozen_before_test_scoring",
        "backbone": args.backbone,
        "arms": selected_arms,
        "seeds": list(seeds),
        "protocol": {
            "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "min_epochs": MIN_EPOCHS,
            "patience": PATIENCE, "min_delta": MIN_DELTA,
            "grad_clip": GRAD_CLIP, "inner_split_seed": INNER_SPLIT_SEED,
        },
        "cohort_hash": sha16(participants), "labels_hash": sha16(y),
        "train_hash": sha16(train), "val_hash": sha16(val),
        "inner_fit_hash": sha16(inner_fit), "inner_stop_hash": sha16(inner_stop),
        "metadata_audit": metadata_audit, "artifacts_hash": sha16(A),
        "training_manifest_source_commit": manifest["source_commit"],
        "script_sha256": file_sha256(Path(__file__)), "fits": records,
    }
    atomic_json(payload, out / "manifest.json")
    print(f"FROZEN {len(records)} fit files under {out}")


def load_fit_matrix(out: Path, participants: np.ndarray, arm: str,
                    seeds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    logits, probs = [], []
    for seed in seeds:
        path = out / f"fit_{arm}_seed{seed}.npz"
        assert completed_fit(path, participants, arm, str(json.load(open(out / "manifest.json"))["backbone"]), seed)
        with np.load(path, allow_pickle=True) as z:
            logits.append(np.asarray(z["raw_logits"], dtype=np.float64))
            probs.append(np.asarray(z["calibrated_prob"], dtype=np.float64))
    return np.stack(logits), np.stack(probs)


def run_score(args) -> None:
    out = Path(args.out_dir)
    manifest = json.load(open(out / "manifest.json"))
    assert manifest["status"] == "complete_predictions_frozen_before_test_scoring"
    assert manifest["script_sha256"] == file_sha256(Path(__file__)), \
        "script changed after fitting; score with the frozen source or rerun under a new tag"
    d, _, _, tests = load_cohort(args)
    participants = d[UNIT].to_numpy()
    y_all = d["y"].to_numpy(np.int64)
    assert manifest["cohort_hash"] == sha16(participants)
    arms = tuple(manifest["arms"])
    seeds = tuple(manifest["seeds"])
    probs = {arm: load_fit_matrix(out, participants, arm, seeds)[1] for arm in arms}
    neg_nll = lambda yy, pp: -float(logloss(yy, pp).mean())
    result = {"standing": "exploratory fixed-MLP sensitivity", "arms": {},
              "comparisons": {}, "manifest_sha256": file_sha256(out / "manifest.json")}
    for arm, P in probs.items():
        result["arms"][arm] = {}
        for name, mask in tests.items():
            y, Pt = y_all[mask], P[:, mask]
            auc, auc_ci = hierarchical_ci(Pt, y, auroc, args.bootstrap, BOOTSTRAP_SEED)
            nnll, nnll_ci = hierarchical_ci(Pt, y, neg_nll, args.bootstrap,
                                             BOOTSTRAP_SEED + 1)
            result["arms"][arm][name] = {
                "n": int(mask.sum()), "auroc": auc, "auroc_ci": auc_ci,
                "neg_nll": nnll, "neg_nll_ci": nnll_ci,
                "per_seed_auroc": [auroc(y, p) for p in Pt],
                "per_seed_nll": [float(logloss(y, p).mean()) for p in Pt],
                "calibration": [calib_diag(y, p) for p in Pt],
            }
    for a, b, label in COMPARISONS:
        if a not in probs or b not in probs:
            continue
        key = f"{a}_minus_{b}"
        result["comparisons"][key] = {"interpretation": label}
        for name, mask in tests.items():
            y = y_all[mask]
            result["comparisons"][key][name] = {
                "delta_neg_nll": paired_hierarchical_ci(
                    probs[a][:, mask], probs[b][:, mask], y, neg_nll,
                    args.bootstrap, BOOTSTRAP_SEED + 2),
                "delta_auroc": paired_hierarchical_ci(
                    probs[a][:, mask], probs[b][:, mask], y, auroc,
                    args.bootstrap, BOOTSTRAP_SEED + 3),
            }
    result["primary_result_path"] = \
        "comparisons.audio_correct_minus_audio_within_label.matched.delta_neg_nll"
    target = out / "results.json"
    assert not target.exists(), f"refusing to overwrite {target}"
    atomic_json(result, target)
    print(f"SCORED frozen predictions -> {target}")


def self_test() -> None:
    import torch
    rng = np.random.RandomState(11)
    n = 500
    y = np.tile([0, 1], n // 2).astype(np.int64)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    X[:, 0] += y * 0.8
    train = np.zeros(n, bool); train[:350] = True
    val = np.zeros(n, bool); val[350:450] = True
    inner_fit = np.zeros(n, bool); inner_fit[:315] = True
    inner_stop = train & ~inner_fit
    old = (MAX_EPOCHS, MIN_EPOCHS, PATIENCE)
    # Exercise deterministic primitives without changing the formal constants.
    mean, std = fit_scaler(X, train)
    Xs = transform(X, mean, std)
    seed_everything(0)
    a = build_model(X.shape[1])
    h1 = state_sha16(a)
    loss1, _ = train_epochs(a, Xs, y, np.where(inner_fit)[0], 3, 0, "cpu",
                            np.where(inner_stop)[0])
    p1 = predict_logits(a, Xs, None, "cpu")
    seed_everything(0)
    b = build_model(X.shape[1])
    assert state_sha16(b) == h1
    loss2, _ = train_epochs(b, Xs, y, np.where(inner_fit)[0], 3, 0, "cpu",
                            np.where(inner_stop)[0])
    p2 = predict_logits(b, Xs, None, "cpu")
    assert np.array_equal(p1, p2) and loss1 == loss2
    assert loss1[-1] < loss1[0] and np.isfinite(p1).all() and np.ptp(p1) > 0
    assert old == (100, 10, 15)
    print(f"SELF-TEST PASS: deterministic fixed MLP ({torch.__version__})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--artifacts", default="results/artefact_features.csv")
    ap.add_argument("--ast-emb", dest="ast_emb", default="results/ast_embeddings.npz",
                    help="raw embedding file for the selected backbone")
    ap.add_argument("--alignment-dir", default="results/alignment")
    ap.add_argument("--out-dir", default="results/mlp_ast")
    ap.add_argument("--backbone", choices=("ast", "opera_ct"), default="ast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--include-common", action="store_true")
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=2000)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.fit:
        run_fit(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
