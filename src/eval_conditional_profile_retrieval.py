"""Query-conditioned profile retrieval for the UKCOVID pairing audit.

This follow-up never trains a model and only evaluates the frozen Standard-validation
representations.  It retains one candidate per semantic metadata profile, then repeats the
retrieval manipulation check with four candidate-eligibility rules: unrestricted, same
recorded sex, same disease label, and same label plus recorded sex.

The protocol is frozen in ``docs/PAIRING_AUDIT_SUPPLEMENTARY_EXPERIMENTS_PREREG_ZH.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_profile_retrieval import (ARMS, BOOTSTRAP_SEED, normalize_rows, prepare,
                                     retrieval_metrics, sha256_file)
from audio_baselines_v2 import UNIT
from eval_metadata_alignment import atomic_json, atomic_npz, load_representation


SCRIPT_PATH = Path(__file__).resolve()
CONDITIONS = ("unrestricted", "same_sex", "same_label", "same_label_and_sex")


def recorded_sex(text: str) -> str:
    match = re.search(r"\[SEX=(.*?)\]\s+\[SMOKER=", str(text))
    if match is None:
        raise ValueError("metadata schema does not contain a parseable SEX field")
    return match.group(1)


def categorical_codes(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = sorted(map(str, np.unique(values)))
    lookup = {label: index for index, label in enumerate(labels)}
    return np.asarray([lookup[str(value)] for value in values], dtype=np.int16), labels


def group_candidate_masks(gold_position: np.ndarray, group: np.ndarray,
                          n_candidates: int) -> np.ndarray:
    n_groups = int(group.max()) + 1
    masks = np.zeros((n_groups, n_candidates), dtype=bool)
    masks[group, gold_position] = True
    if np.any(masks.sum(axis=1) == 0):
        raise ValueError("empty candidate group")
    return masks


def build_conditions(gold_position: np.ndarray, y: np.ndarray, sex: np.ndarray,
                     n_candidates: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    y_code, y_levels = categorical_codes(y)
    sex_code, sex_levels = categorical_codes(sex)
    joint_values = np.asarray([f"{label}|{value}" for label, value in zip(y, sex)])
    joint_code, joint_levels = categorical_codes(joint_values)

    conditions = {
        "unrestricted": (
            np.zeros(len(gold_position), dtype=np.int16),
            np.ones((1, n_candidates), dtype=bool),
        ),
        "same_sex": (sex_code, group_candidate_masks(
            gold_position, sex_code, n_candidates)),
        "same_label": (y_code, group_candidate_masks(
            gold_position, y_code, n_candidates)),
        "same_label_and_sex": (joint_code, group_candidate_masks(
            gold_position, joint_code, n_candidates)),
    }
    audit = {
        "disease_levels": y_levels,
        "recorded_sex_levels": sex_levels,
        "joint_levels": joint_levels,
    }
    return conditions, audit


def conditioned_cosine_ranks(query: np.ndarray, candidates: np.ndarray,
                             gold_position: np.ndarray,
                             conditions: dict[str, tuple[np.ndarray, np.ndarray]],
                             block_size: int = 256) -> dict[str, np.ndarray]:
    """One-based stable ranks for all conditions from one cosine-score pass."""
    q = normalize_rows(query)
    c = normalize_rows(candidates)
    gold_position = np.asarray(gold_position, dtype=np.int64)
    assert q.shape[1] == c.shape[1]
    assert gold_position.shape == (len(q),)
    ranks = {name: np.empty(len(q), dtype=np.int32) for name in conditions}
    sizes = {}
    for name, (query_group, group_masks) in conditions.items():
        query_group = np.asarray(query_group, dtype=np.int64)
        assert query_group.shape == (len(q),)
        assert group_masks.shape[1] == len(c)
        assert np.all((query_group >= 0) & (query_group < len(group_masks)))
        true_eligible = group_masks[query_group, gold_position]
        if not np.all(true_eligible):
            raise ValueError(
                f"a query's true semantic profile is absent from {name} candidate bank")
        sizes[name] = group_masks[query_group].sum(axis=1)

    candidate_position = np.arange(len(c), dtype=np.int64)[None, :]
    for start in range(0, len(q), block_size):
        stop = min(start + block_size, len(q))
        scores = q[start:stop] @ c.T
        gp = gold_position[start:stop]
        gold_score = scores[np.arange(stop - start), gp][:, None]
        for name, (query_group, group_masks) in conditions.items():
            eligible = group_masks[query_group[start:stop]]
            greater = np.sum((scores > gold_score) & eligible, axis=1)
            earlier_tie = np.sum(
                (scores == gold_score) & eligible &
                (candidate_position < gp[:, None]), axis=1)
            ranks[name][start:stop] = 1 + greater + earlier_tie

    for name in conditions:
        assert np.all((ranks[name] >= 1) & (ranks[name] <= sizes[name]))
    return ranks


def distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(values.min()),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def macro_average(values: np.ndarray, gold_position: np.ndarray,
                  n_profiles: int) -> float:
    per_profile = []
    for profile in range(n_profiles):
        mask = gold_position == profile
        if not mask.any():
            raise ValueError("candidate profile has no validation query")
        per_profile.append(float(np.mean(values[mask])))
    return float(np.mean(per_profile))


def random_baselines(candidate_sizes: np.ndarray, gold_position: np.ndarray,
                     n_profiles: int) -> dict:
    sizes = np.asarray(candidate_sizes, dtype=np.int64)
    maximum = int(sizes.max())
    harmonic = np.cumsum(1.0 / np.arange(1, maximum + 1, dtype=np.float64))
    values = {
        "mrr": harmonic[sizes - 1] / sizes,
        "r1": 1.0 / sizes,
        "r10": np.minimum(10, sizes) / sizes,
    }
    return {
        "analytic_random_ranking": {
            metric: {
                "macro_profile": macro_average(value, gold_position, n_profiles),
                "micro_participant": float(np.mean(value)),
            }
            for metric, value in values.items()
        }
    }


def condition_audit(query_group: np.ndarray, group_masks: np.ndarray,
                    gold_position: np.ndarray, n_profiles: int) -> dict:
    sizes = group_masks[query_group].sum(axis=1).astype(np.int64)
    true_eligible = group_masks[query_group, gold_position]
    return {
        "n_queries": int(len(query_group)),
        "n_true_profiles": int(n_profiles),
        "true_profile_coverage": float(np.mean(true_eligible)),
        "candidate_count": distribution(sizes),
        **random_baselines(sizes, gold_position, n_profiles),
    }


def self_test() -> None:
    candidates = np.eye(4, dtype=np.float32)
    query = np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.8, 0.9, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.8, 0.9],
    ], dtype=np.float32)
    gold = np.arange(4, dtype=np.int64)
    group = np.asarray([0, 0, 1, 1], dtype=np.int16)
    masks = group_candidate_masks(gold, group, 4)
    ranks = conditioned_cosine_ranks(
        query, candidates, gold, {"same_group": (group, masks)}, block_size=2)
    assert np.array_equal(ranks["same_group"], np.asarray([1, 1, 1, 1]))
    sizes = masks[group].sum(axis=1)
    baseline = random_baselines(sizes, gold, 4)
    assert baseline["analytic_random_ranking"]["r1"]["micro_participant"] == 0.5
    assert recorded_sex("[AGE=18-44] [SEX=[MISSING]] [SMOKER=NEVER]") == "[MISSING]"
    print("SELF-TEST PASS: conditional masks, stable ranks, random baselines, sex parsing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    parser.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    parser.add_argument("--texts", default="results/metadata_texts.csv")
    parser.add_argument("--text-embeddings", default="results/metadata_text_embeddings.npz")
    parser.add_argument("--audio-emb", dest="ast_emb", default="results/ast_embeddings.npz")
    parser.add_argument("--alignment-dir", default="results/alignment")
    parser.add_argument("--backbone", default="AST-6L")
    parser.add_argument("--out-dir", default="results/pairing_followup/e1_ast")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected-epochs", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--block-size", type=int, default=256)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks_path = out_dir / "conditional_query_ranks.npz"
    config_path = out_dir / "run_config.json"
    metrics_path = out_dir / "metrics.json"
    for target in (ranks_path, config_path, metrics_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")

    base_audit, data, val, participants, state = prepare(args)
    cohort = pd.read_csv(args.cohort, usecols=[UNIT, "splits", "y"])
    texts = pd.read_csv(args.texts, usecols=[UNIT, "text"])
    if not np.array_equal(cohort[UNIT].to_numpy(), participants):
        raise ValueError("cohort participant order changed")
    if not np.array_equal(texts[UNIT].to_numpy(), participants):
        raise ValueError("metadata-text participant order changed")
    if not np.array_equal((cohort["splits"] == "val").to_numpy(), val):
        raise ValueError("validation split differs between frozen inputs")

    y = cohort.loc[val, "y"].to_numpy()
    sex = texts.loc[val, "text"].map(recorded_sex).to_numpy()
    conditions, level_audit = build_conditions(
        state["gold_position"], y, sex, len(state["candidate_ids"]))
    condition_reports = {
        name: condition_audit(group, masks, state["gold_position"],
                              len(state["candidate_ids"]))
        for name, (group, masks) in conditions.items()
    }

    ranks: dict[str, dict[str, np.ndarray]] = {name: {} for name in CONDITIONS}
    representation_cache: dict[str, np.ndarray] = {}
    candidates = normalize_rows(state["candidate_embeddings"])
    for arm in ARMS:
        for condition in CONDITIONS:
            ranks[condition][arm] = []
        for seed in args.seeds:
            representation = load_representation(
                args, participants, arm, seed, "raw", representation_cache,
                state["manifest"])
            query = normalize_rows(representation[val])
            seed_ranks = conditioned_cosine_ranks(
                query, candidates, state["gold_position"], conditions,
                block_size=args.block_size)
            for condition in CONDITIONS:
                ranks[condition][arm].append(seed_ranks[condition])
            print(f"{args.backbone} {arm} seed {seed}: all four conditions ranked", flush=True)

    ranks = {
        condition: {arm: np.stack(rows) for arm, rows in arm_values.items()}
        for condition, arm_values in ranks.items()
    }
    atomic_npz(
        ranks_path,
        participants=participants[val],
        seeds=np.asarray(args.seeds, dtype=np.int64),
        arms=np.asarray(ARMS),
        conditions=np.asarray(CONDITIONS),
        true_profile_id=state["text_id"][val],
        true_candidate_position=state["gold_position"],
        candidate_profile_ids=state["candidate_ids"],
        recorded_sex=sex,
        disease_label=y,
        **{
            f"candidate_count__{condition}": masks[group].sum(axis=1).astype(np.int32)
            for condition, (group, masks) in conditions.items()
        },
        **{
            f"rank__{condition}__{arm}": ranks[condition][arm]
            for condition in CONDITIONS for arm in ARMS
        },
    )

    config = {
        **base_audit,
        "standing": "post-hoc Standard-validation conditional-retrieval diagnostic",
        "conditions": list(CONDITIONS),
        "condition_definitions": {
            "unrestricted": "all unique validation profiles",
            "same_sex": "profile occurs for at least one validation participant with query sex",
            "same_label": "profile occurs for at least one validation participant with query label",
            "same_label_and_sex": "profile occurs for at least one validation participant with both",
        },
        "level_audit": level_audit,
        "condition_audit": condition_reports,
        "cohort_sha256": sha256_file(Path(args.cohort)),
        "metadata_texts_sha256": sha256_file(Path(args.texts)),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "rank_file": str(ranks_path),
        "rank_file_sha256": sha256_file(ranks_path),
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "matched_or_test_labels_read": False,
    }
    atomic_json(config, config_path)
    if sha256_file(SCRIPT_PATH) != config["script_sha256"]:
        raise RuntimeError("script changed between rank generation and aggregation")

    metrics = {
        "standing": config["standing"],
        "backbone": args.backbone,
        "condition_audit": condition_reports,
        "metrics": {
            condition: retrieval_metrics(
                ranks[condition], state["gold_position"], len(state["candidate_ids"]),
                args.bootstrap, args.bootstrap_seed)
            for condition in CONDITIONS
        },
    }
    atomic_json(metrics, metrics_path)
    print(f"wrote conditional retrieval ranks and metrics to {out_dir}")


if __name__ == "__main__":
    main()
