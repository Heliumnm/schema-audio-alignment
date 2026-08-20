"""Frozen clinical-profile retrieval for the Pairing-Controlled Transfer Audit.

This is the formal manipulation check described in
``docs/ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md``.  It deliberately evaluates only the full
Standard validation split.  It never opens Standard test, matched, or matched-long labels.

The candidate bank contains one frozen Phi-2 vector for every unique schema profile that
appears in Standard validation.  Each validation participant is a query.  The audio-side
post-ReLU projector output and candidate text vectors are L2-normalised and ranked by
cosine similarity.  Repeated profiles therefore create repeated *queries*, never repeated
candidates.

Execution is intentionally staged::

    python src/audit_profile_retrieval.py --self-test
    python src/audit_profile_retrieval.py --preflight [paths ...]
    python src/audit_profile_retrieval.py --run [paths ...]

``--run`` writes per-participant/per-seed ranks before it computes aggregate metrics.  All
formal outputs are no-clobber.  The script is generic in the audio backbone: point
``--alignment-dir`` and ``--audio-emb`` at either the frozen AST or OPERA-CT run.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_baselines_v2 import UNIT
from eval_metadata_alignment import (
    PROJECTOR_ARMS,
    atomic_json,
    atomic_npz,
    load_representation,
    sha16,
    verify_training_manifest,
)


ARMS = PROJECTOR_ARMS
BOOTSTRAP_SEED = 20260820
SCRIPT_PATH = Path(__file__).resolve()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                return h.hexdigest()
            h.update(block)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError("retrieval contains zero or non-finite embedding norms")
    out = x / norm
    assert np.isfinite(out).all()
    return out


def stable_cosine_ranks(query: np.ndarray, candidates: np.ndarray,
                        gold_position: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Return one-based ranks with a deterministic candidate-order tie break.

    Candidate order is the ascending frozen text id.  For an exact tie, the earlier
    candidate receives the earlier rank.  Counting greater scores plus earlier ties is
    equivalent to a stable descending sort without materialising a full order tensor.
    """
    q = normalize_rows(query)
    c = normalize_rows(candidates)
    gold_position = np.asarray(gold_position, dtype=np.int64)
    assert q.shape[1] == c.shape[1]
    assert gold_position.shape == (len(q),)
    assert np.all((gold_position >= 0) & (gold_position < len(c)))

    ranks = np.empty(len(q), dtype=np.int32)
    candidate_position = np.arange(len(c), dtype=np.int64)[None, :]
    for start in range(0, len(q), block_size):
        stop = min(start + block_size, len(q))
        scores = q[start:stop] @ c.T
        gp = gold_position[start:stop]
        gold_score = scores[np.arange(stop - start), gp][:, None]
        greater = np.sum(scores > gold_score, axis=1)
        earlier_tie = np.sum(
            (scores == gold_score) & (candidate_position < gp[:, None]), axis=1)
        ranks[start:stop] = 1 + greater + earlier_tie
    assert np.all((ranks >= 1) & (ranks <= len(c)))
    return ranks


def _rank_score(ranks: np.ndarray, metric: str) -> np.ndarray:
    ranks = np.asarray(ranks)
    if metric == "mrr":
        return 1.0 / ranks
    if metric == "r1":
        return (ranks <= 1).astype(float)
    if metric == "r10":
        return (ranks <= 10).astype(float)
    raise ValueError(metric)


def profile_means(ranks: np.ndarray, gold_position: np.ndarray,
                  n_profiles: int, metric: str) -> np.ndarray:
    """Return ``seed x profile`` metrics, averaging repeated queries within profile."""
    values = _rank_score(ranks, metric)
    out = np.empty((ranks.shape[0], n_profiles), dtype=np.float64)
    for profile in range(n_profiles):
        mask = gold_position == profile
        assert mask.any(), "candidate bank contains a profile with no query"
        out[:, profile] = values[:, mask].mean(axis=1)
    return out


def matrix_bootstrap_ci(values: np.ndarray, boot: int, seed: int) -> dict:
    """Bootstrap both axes of a paired seed x unit metric matrix."""
    values = np.asarray(values, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.RandomState(seed)
    draws = np.empty(boot, dtype=np.float64)
    ns, nu = values.shape
    # Multinomial counts are exactly equivalent to drawing indices with replacement.  A
    # chunked einsum avoids 10,000 Python allocations of a seed-by-participant matrix.
    chunk = 256
    seed_probability = np.full(ns, 1.0 / ns)
    unit_probability = np.full(nu, 1.0 / nu)
    for start in range(0, boot, chunk):
        stop = min(start + chunk, boot)
        count = stop - start
        seed_count = rng.multinomial(ns, seed_probability, size=count)
        unit_count = rng.multinomial(nu, unit_probability, size=count)
        draws[start:stop] = np.einsum(
            "bi,ij,bj->b", seed_count, values, unit_count,
            optimize=True) / float(ns * nu)
    return {
        "observed": observed,
        "ci": np.percentile(draws, [2.5, 97.5]).astype(float).tolist(),
        "per_seed": values.mean(axis=1).astype(float).tolist(),
        "n_seeds": int(ns),
        "n_units": int(nu),
    }


def paired_matrix_bootstrap_ci(a: np.ndarray, b: np.ndarray, boot: int,
                               seed: int) -> dict:
    assert a.shape == b.shape
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    out = matrix_bootstrap_ci(delta, boot, seed)
    out["seed_signs"] = np.sign(np.asarray(out["per_seed"])).astype(int).tolist()
    return out


def retrieval_metrics(ranks_by_arm: dict[str, np.ndarray], gold_position: np.ndarray,
                      n_profiles: int, boot: int, seed: int) -> dict:
    metrics = ("mrr", "r1", "r10")
    arm_out: dict[str, dict] = {}
    macro_cache: dict[tuple[str, str], np.ndarray] = {}
    micro_cache: dict[tuple[str, str], np.ndarray] = {}
    for arm, ranks in ranks_by_arm.items():
        arm_out[arm] = {"macro_profile": {}, "micro_participant": {}}
        for mi, metric in enumerate(metrics):
            macro = profile_means(ranks, gold_position, n_profiles, metric)
            micro = _rank_score(ranks, metric)
            macro_cache[(arm, metric)] = macro
            micro_cache[(arm, metric)] = micro
            arm_out[arm]["macro_profile"][metric] = matrix_bootstrap_ci(
                macro, boot, seed + 100 * mi)
            arm_out[arm]["micro_participant"][metric] = matrix_bootstrap_ci(
                micro, boot, seed + 100 * mi + 1)

    comparisons = {}
    for ci, (a, b, label) in enumerate((
        ("correct", "within_label", "individual_correspondence_CG_primary"),
        ("correct", "global", "total_alignment_correspondence"),
        ("within_label", "global", "label_level_cooccurrence"),
    )):
        key = f"{a}_minus_{b}"
        comparisons[key] = {"interpretation": label,
                            "macro_profile": {}, "micro_participant": {}}
        for mi, metric in enumerate(metrics):
            comparisons[key]["macro_profile"][metric] = paired_matrix_bootstrap_ci(
                macro_cache[(a, metric)], macro_cache[(b, metric)], boot,
                seed + 1000 + 100 * ci + mi)
            comparisons[key]["micro_participant"][metric] = paired_matrix_bootstrap_ci(
                micro_cache[(a, metric)], micro_cache[(b, metric)], boot,
                seed + 2000 + 100 * ci + mi)
    return {
        "arms": arm_out,
        "comparisons": comparisons,
        "primary_result_path": (
            "comparisons.correct_minus_within_label.macro_profile.mrr"),
    }


def load_text_axis(texts_path: Path, embeddings_path: Path,
                   participants: np.ndarray, val: np.ndarray) -> dict:
    texts = pd.read_csv(texts_path)
    assert np.array_equal(texts[UNIT].to_numpy(), participants), \
        "metadata-text participant order differs from frozen cohort"
    with np.load(embeddings_path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants)
        text_id = np.asarray(z["text_id"], dtype=np.int64)
        unique_embeddings = np.asarray(z["unique_embeddings"], dtype=np.float32)
    assert len(text_id) == len(participants)
    assert np.all((text_id >= 0) & (text_id < len(unique_embeddings)))
    unique_texts = sorted(texts["text"].astype(str).unique())
    assert len(unique_texts) == len(unique_embeddings)
    expected_id = {text: i for i, text in enumerate(unique_texts)}
    reconstructed = texts["text"].astype(str).map(expected_id).to_numpy(np.int64)
    assert np.array_equal(reconstructed, text_id), \
        "text ids do not match the frozen lexicographic unique-text cache"

    candidate_ids = np.unique(text_id[val])
    candidate_ids.sort()
    position = {int(text_id_): i for i, text_id_ in enumerate(candidate_ids)}
    gold_position = np.asarray([position[int(t)] for t in text_id[val]], dtype=np.int64)
    candidate_embeddings = unique_embeddings[candidate_ids]
    bank_hash = hashlib.sha256()
    bank_hash.update(np.ascontiguousarray(candidate_ids).tobytes())
    bank_hash.update(np.ascontiguousarray(candidate_embeddings).tobytes())
    return {
        "text_id": text_id,
        "candidate_ids": candidate_ids,
        "candidate_embeddings": candidate_embeddings,
        "gold_position": gold_position,
        "candidate_bank_sha256": bank_hash.hexdigest(),
        "text_id_sha16": sha16(text_id),
        "unique_embedding_sha16": sha16(unique_embeddings),
        "n_unique_global": int(len(unique_embeddings)),
    }


def load_standard_validation_only(data_dir: Path, cohort_path: Path) -> tuple[pd.DataFrame,
                                                                                 np.ndarray]:
    """Load identifiers and the Standard split flag without opening outcome metadata."""
    cohort = pd.read_csv(cohort_path, usecols=[UNIT])
    splits = pd.read_csv(data_dir / "train_test_splits.csv", usecols=[UNIT, "splits"])
    data = cohort.merge(splits, on=UNIT, how="left", validate="one_to_one")
    assert np.array_equal(data[UNIT].to_numpy(), cohort[UNIT].to_numpy())
    if data["splits"].isna().any():
        raise ValueError("frozen cohort contains participants without a Standard split")
    val = (data["splits"] == "val").to_numpy()
    if not val.any():
        raise ValueError("Standard validation is empty")
    return data, val


def prepare(args) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    seeds = list(args.seeds)
    assert len(seeds) == len(set(seeds)) == 5, "formal retrieval requires five unique seeds"
    manifest_path = Path(args.alignment_dir) / "manifest.json"
    manifest = verify_training_manifest(manifest_path, seeds, args.expected_epochs)
    # This path does not open participant_metadata.csv at all.  It sees identifiers and
    # the Standard split flag, but neither disease outcomes nor matched-set membership.
    data, val = load_standard_validation_only(Path(args.data), Path(args.cohort))
    participants = data[UNIT].to_numpy()
    text_axis = load_text_axis(
        Path(args.texts), Path(args.text_embeddings), participants, val)
    cache: dict[str, np.ndarray] = {}
    representation_audit: dict[str, dict] = {}
    for arm in ARMS:
        representation_audit[arm] = {}
        for seed in seeds:
            x = load_representation(args, participants, arm, seed, "raw", cache, manifest)
            representation_audit[arm][str(seed)] = {
                "shape": list(x.shape), "sha16": sha16(x),
                "finite": bool(np.isfinite(x).all()),
                "nonzero_rows": int(np.sum(np.linalg.norm(x, axis=1) > 0)),
            }
    audit = {
        "backbone": args.backbone,
        "standing": "Standard-validation manipulation check; no formal test labels read",
        "n_cohort": int(len(data)),
        "n_standard_validation_queries": int(val.sum()),
        "n_candidate_profiles": int(len(text_axis["candidate_ids"])),
        "candidate_bank_sha256": text_axis["candidate_bank_sha256"],
        "text_id_sha16": text_axis["text_id_sha16"],
        "unique_text_embedding_sha16": text_axis["unique_embedding_sha16"],
        "n_unique_profiles_global": text_axis["n_unique_global"],
        "alignment_manifest_sha256": sha256_file(manifest_path),
        "training_input_hashes": manifest.get("input_hashes", {}),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "metadata_texts_sha256": sha256_file(Path(args.texts)),
        "text_embedding_file_sha256": sha256_file(Path(args.text_embeddings)),
        "representations": representation_audit,
        "seeds": seeds,
        "expected_epochs": int(args.expected_epochs),
    }
    return audit, data, val, participants, {"manifest": manifest, **text_axis}


def self_test() -> None:
    candidates = np.eye(3, dtype=np.float32)
    queries = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 0.9, 0.1],
        [0.5, 0.5, 0.0],  # exact tie; candidate 0 precedes candidate 1
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    gold = np.asarray([0, 1, 1, 2])
    ranks = stable_cosine_ranks(queries, candidates, gold, block_size=2)
    assert np.array_equal(ranks, np.asarray([1, 1, 2, 1]))

    ranks_by_arm = {
        "correct": np.stack([ranks, ranks]),
        "within_label": np.stack([np.asarray([2, 2, 2, 1])] * 2),
        "global": np.stack([np.asarray([3, 3, 2, 2])] * 2),
    }
    out1 = retrieval_metrics(ranks_by_arm, gold, 3, boot=100, seed=7)
    out2 = retrieval_metrics(ranks_by_arm, gold, 3, boot=100, seed=7)
    primary = out1["comparisons"]["correct_minus_within_label"][
        "macro_profile"]["mrr"]
    assert primary["observed"] > 0
    assert out1 == out2, "bootstrap is not deterministic"
    print("SELF-TEST PASS: stable cosine ranks, macro-profile aggregation, paired bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--texts", default="results/metadata_texts.csv")
    parser.add_argument("--text-embeddings", default="results/metadata_text_embeddings.npz")
    parser.add_argument("--audio-emb", dest="ast_emb", default="results/ast_embeddings.npz")
    parser.add_argument("--alignment-dir", default="results/alignment")
    parser.add_argument("--backbone", default="AST-6L")
    parser.add_argument("--out-dir", default="results/profile_retrieval_ast")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected-epochs", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--block-size", type=int, default=512)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = out_dir / "preflight.json"
    ranks_path = out_dir / "query_ranks.npz"
    config_path = out_dir / "run_config.json"
    metrics_path = out_dir / "metrics.json"
    targets = [preflight_path] if args.preflight else [ranks_path, config_path, metrics_path]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    audit, data, val, participants, state = prepare(args)
    if args.preflight:
        atomic_json({**audit, "preflight_pass": True}, preflight_path)
        print(f"PREFLIGHT PASS: wrote {preflight_path}; no retrieval score computed")
        return

    seeds = list(args.seeds)
    val_participants = participants[val]
    gold_position = state["gold_position"]
    ranks_by_arm: dict[str, np.ndarray] = {}
    raw_cache: dict[str, np.ndarray] = {}
    for arm in ARMS:
        rows = []
        for seed in seeds:
            representation = load_representation(
                args, participants, arm, seed, "raw", raw_cache, state["manifest"])
            rank = stable_cosine_ranks(
                representation[val], state["candidate_embeddings"], gold_position,
                block_size=args.block_size)
            rows.append(rank)
            print(f"{arm} seed {seed}: ranked {len(rank)} queries", flush=True)
        ranks_by_arm[arm] = np.stack(rows)

    # Per-query results are frozen before aggregate statistics are calculated.
    atomic_npz(
        ranks_path,
        participants=val_participants,
        seeds=np.asarray(seeds, dtype=np.int64),
        arms=np.asarray(ARMS),
        true_profile_id=state["text_id"][val],
        true_candidate_position=gold_position,
        candidate_profile_ids=state["candidate_ids"],
        candidate_bank_sha256=np.asarray(state["candidate_bank_sha256"]),
        **{f"rank__{arm}": ranks_by_arm[arm] for arm in ARMS},
    )
    atomic_json({
        **audit,
        "rank_file": str(ranks_path),
        "rank_file_sha256": sha256_file(ranks_path),
        "rank_tie_break": "stable ascending candidate text_id for exact cosine ties",
        "primary_metric": "macro-profile MRR, correct minus within-label",
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "block_size": int(args.block_size),
        "formal_tests_read": False,
    }, config_path)
    if sha256_file(SCRIPT_PATH) != audit["script_sha256"]:
        raise RuntimeError("retrieval script changed between frozen ranks and aggregation")
    result = {
        "standing": "Standard-validation manipulation check; not disease transfer",
        "backbone": args.backbone,
        "n_queries": int(val.sum()),
        "n_candidate_profiles": int(len(state["candidate_ids"])),
        "candidate_bank_sha256": state["candidate_bank_sha256"],
        "metrics": retrieval_metrics(
            ranks_by_arm, gold_position, len(state["candidate_ids"]),
            args.bootstrap, args.bootstrap_seed),
    }
    atomic_json(result, metrics_path)
    print(f"wrote frozen query ranks, configuration, and Standard-val metrics to {out_dir}")


if __name__ == "__main__":
    main()
