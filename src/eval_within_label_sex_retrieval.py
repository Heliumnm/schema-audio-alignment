"""Standard-validation profile retrieval for the UKCOVID W_{y,s} control.

Correct and Within-label ranks are reused from the completed E1 unrestricted candidate
bank.  Only W_{y,s} ranks are newly computed.  No matched or test labels are opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from audio_baselines_v2 import UNIT
from audit_profile_retrieval import (BOOTSTRAP_SEED, atomic_json, atomic_npz,
                                     load_standard_validation_only, load_text_axis,
                                     retrieval_metrics, sha256_file,
                                     stable_cosine_ranks)
from eval_within_label_sex_control import (NEW_ARM, load_e2_representation,
                                          verify_e2_manifest)


SCRIPT_PATH = Path(__file__).resolve()


def translate_metrics(raw: dict) -> dict:
    """Rename a three-arm retrieval result used as a generic paired engine."""
    return {
        "arms": {
            "correct": raw["arms"]["correct"],
            NEW_ARM: raw["arms"]["within_label"],
            "within_label": raw["arms"]["global"],
        },
        "comparisons": {
            f"correct_minus_{NEW_ARM}": {
                **raw["comparisons"]["correct_minus_within_label"],
                "interpretation": "exact-pair increment after preserving label and recorded sex",
            },
            "correct_minus_within_label": {
                **raw["comparisons"]["correct_minus_global"],
                "interpretation": "original exact-pair increment",
            },
            f"{NEW_ARM}_minus_within_label": {
                **raw["comparisons"]["within_label_minus_global"],
                "interpretation": "increment from preserving recorded sex in shuffled pairs",
            },
        },
        "primary_result_path": (
            f"comparisons.correct_minus_{NEW_ARM}.macro_profile.mrr"),
    }


def self_test() -> None:
    base = {"arms": {name: {"x": name} for name in
                     ("correct", "within_label", "global")},
            "comparisons": {
                "correct_minus_within_label": {"x": 1},
                "correct_minus_global": {"x": 2},
                "within_label_minus_global": {"x": 3},
            }}
    out = translate_metrics(base)
    assert out["arms"][NEW_ARM]["x"] == "within_label"
    assert out["comparisons"][f"{NEW_ARM}_minus_within_label"]["x"] == 3
    print("SELF-TEST PASS: E2 retrieval arm and comparison translation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--texts", default="results/metadata_texts.csv")
    ap.add_argument("--text-embeddings", default="results/metadata_text_embeddings.npz")
    ap.add_argument("--e1-dir", default="results/pairing_followup/e1_ast")
    ap.add_argument("--e2-alignment-dir", default="results/pairing_followup/e2_alignment_ast")
    ap.add_argument("--out-dir", default="results/pairing_followup/e2_retrieval_ast")
    ap.add_argument("--backbone", default="AST-6L")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--expected-epochs", type=int, default=500)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranks_path = out_dir / "query_ranks.npz"
    config_path = out_dir / "run_config.json"
    metrics_path = out_dir / "metrics.json"
    if any(path.exists() for path in (ranks_path, config_path, metrics_path)):
        raise FileExistsError("refusing to overwrite E2 retrieval output")

    seeds = list(args.seeds)
    assert len(seeds) == len(set(seeds)) == 5
    manifest_path = Path(args.e2_alignment_dir) / "manifest.json"
    manifest = verify_e2_manifest(manifest_path, seeds, args.expected_epochs)
    data, val = load_standard_validation_only(Path(args.data), Path(args.cohort))
    participants = data[UNIT].to_numpy()
    text = load_text_axis(
        Path(args.texts), Path(args.text_embeddings), participants, val)

    e1_ranks_path = Path(args.e1_dir) / "conditional_query_ranks.npz"
    e1_config_path = Path(args.e1_dir) / "run_config.json"
    e1_config = json.load(open(e1_config_path))
    assert e1_config["matched_or_test_labels_read"] is False
    assert e1_config["rank_file_sha256"] == sha256_file(e1_ranks_path)
    with np.load(e1_ranks_path, allow_pickle=True) as z:
        assert np.array_equal(z["participants"], participants[val])
        assert np.array_equal(z["seeds"], np.asarray(seeds))
        assert np.array_equal(z["candidate_profile_ids"], text["candidate_ids"])
        assert np.array_equal(z["true_candidate_position"], text["gold_position"])
        correct = np.asarray(z["rank__unrestricted__correct"], dtype=np.int32)
        within = np.asarray(z["rank__unrestricted__within_label"], dtype=np.int32)

    wys_rows = []
    for seed in seeds:
        representation = load_e2_representation(
            Path(args.e2_alignment_dir), manifest, participants, seed)
        rank = stable_cosine_ranks(
            representation[val], text["candidate_embeddings"], text["gold_position"],
            block_size=args.block_size)
        wys_rows.append(rank)
        print(f"{args.backbone} {NEW_ARM} seed {seed}: ranked {len(rank)} queries", flush=True)
    wys = np.stack(wys_rows)

    atomic_npz(
        ranks_path,
        participants=participants[val],
        seeds=np.asarray(seeds),
        arms=np.asarray(("correct", NEW_ARM, "within_label")),
        true_profile_id=text["text_id"][val],
        true_candidate_position=text["gold_position"],
        candidate_profile_ids=text["candidate_ids"],
        rank__correct=correct,
        rank__within_label_sex=wys,
        rank__within_label=within,
    )
    config = {
        "standing": "post-hoc Standard-validation W_{y,s} retrieval diagnostic",
        "backbone": args.backbone,
        "n_queries": int(val.sum()),
        "n_candidate_profiles": int(len(text["candidate_ids"])),
        "candidate_bank_sha256": text["candidate_bank_sha256"],
        "e1_rank_file_sha256": sha256_file(e1_ranks_path),
        "e1_config_sha256": sha256_file(e1_config_path),
        "e2_manifest_sha256": sha256_file(manifest_path),
        "rank_file_sha256": sha256_file(ranks_path),
        "script_sha256": sha256_file(SCRIPT_PATH),
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "matched_or_test_labels_read": False,
    }
    atomic_json(config, config_path)
    raw = retrieval_metrics(
        {"correct": correct, "within_label": wys, "global": within},
        text["gold_position"], len(text["candidate_ids"]),
        args.bootstrap, args.bootstrap_seed)
    atomic_json({
        "standing": config["standing"],
        "backbone": args.backbone,
        "n_queries": int(val.sum()),
        "n_candidate_profiles": int(len(text["candidate_ids"])),
        "metrics": translate_metrics(raw),
    }, metrics_path)
    print(f"wrote E2 retrieval output to {out_dir}; no matched/test labels read")


if __name__ == "__main__":
    main()
