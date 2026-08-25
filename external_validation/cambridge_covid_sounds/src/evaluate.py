"""Frozen patient-level disease, retrieval and information-channel evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import (REPO_ROOT, atomic_json, load_config, output_paths, sha16_array,
                    sha256_file)

sys.path.insert(0, str(REPO_ROOT / "src"))
from audio_baselines_v2 import auroc, calib_diag  # noqa: E402
from eval_metadata_alignment import (fit_disease_head, fit_probe, hierarchical_ci,
                                     make_validation_folds, paired_hierarchical_ci)  # noqa: E402


ARMS = ("raw_audio", "correct", "within_label", "global")
CONTEXT_ARMS = ("metadata_only", "raw_audio_plus_metadata")
EVALUATION_ARMS = ARMS + CONTEXT_ARMS


def mean_nll(y: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(probability) + (1 - y) * np.log(1 - probability))))


def brier(y: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - y) ** 2))


def load_representations(paths: dict[str, Path], table: pd.DataFrame, backbone: str,
                         seeds: list[int]) -> dict[str, np.ndarray]:
    raw_file = paths["models"] / backbone / "raw_embeddings.npz"
    raw = np.load(raw_file, allow_pickle=True)
    participants = table.participant_identifier.astype(str).to_numpy()
    if not np.array_equal(raw["participants"].astype(str), participants):
        raise RuntimeError(f"{backbone} participant order mismatch")
    output = {"raw_audio": np.repeat(raw["embeddings"][None, :, :], len(seeds), axis=0)}
    directory = paths["models"] / backbone / "alignment"
    manifest = json.loads((directory / "manifest.json").read_text())
    if int(manifest["epochs"]) != 500:
        raise RuntimeError("formal evaluation only accepts epoch-500 projectors")
    for arm in ("correct", "within_label", "global"):
        values = []
        for seed in seeds:
            file = directory / f"repr_{arm}_seed{seed}.npz"
            representation = np.load(file, allow_pickle=True)
            if not np.array_equal(representation["participants"].astype(str), participants):
                raise RuntimeError(f"{file}: participant order mismatch")
            if int(representation["epochs"]) != 500:
                raise RuntimeError(f"{file}: wrong epoch")
            values.append(np.asarray(representation["raw"], dtype=np.float32))
        output[arm] = np.stack(values)
    return output


def pair_cluster_ci(A: np.ndarray, B: np.ndarray, y: np.ndarray,
                    pair_id: np.ndarray, fn: Callable, boot: int, seed: int) -> dict:
    unique = np.unique(pair_id)
    groups = {pair: np.where(pair_id == pair)[0] for pair in unique}
    if any(len(indices) != 2 for indices in groups.values()):
        raise RuntimeError("matched bootstrap requires exactly two participants per pair")
    per_seed = np.asarray([fn(y, A[k]) - fn(y, B[k]) for k in range(len(A))])
    rng, values = np.random.RandomState(seed), []
    for _ in range(boot):
        sampled_pairs = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([groups[pair] for pair in sampled_pairs])
        sampled_seeds = rng.choice(len(A), len(A), replace=True)
        values.append(float(np.mean([
            fn(y[indices], A[s, indices]) - fn(y[indices], B[s, indices])
            for s in sampled_seeds])))
    return {"observed": float(np.mean(per_seed)),
            "ci": list(map(float, np.percentile(values, [2.5, 97.5]))),
            "per_seed": per_seed.tolist(), "n_pairs": int(len(unique)),
            "n_seeds": int(len(A))}


def absolute_metrics(probability: np.ndarray, y: np.ndarray, boot: int, seed: int) -> dict:
    result = {}
    for offset, (name, fn) in enumerate((("auroc", auroc), ("nll", mean_nll),
                                         ("brier", brier))):
        observed, ci = hierarchical_ci(probability, y, fn, boot, seed + offset)
        result[name] = {"observed": observed, "ci": ci}
    result["calibration"] = calib_diag(y, probability.mean(axis=0))
    return result


def profile_mrr(representations: np.ndarray, query_indices: np.ndarray,
                text_id: np.ndarray, text_embeddings: np.ndarray) -> tuple[float, np.ndarray]:
    candidate_ids = np.unique(text_id[query_indices])
    bank = text_embeddings[candidate_ids]
    bank = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    position = {text: index for index, text in enumerate(candidate_ids)}
    target = np.asarray([position[text_id[index]] for index in query_indices])
    query = representations[query_indices]
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    reciprocal = np.empty(len(query), dtype=float)
    for start in range(0, len(query), 512):
        scores = query[start:start + 512] @ bank.T
        target_position = target[start:start + 512]
        target_score = scores[np.arange(len(scores)), target_position]
        rank = 1 + np.sum(scores > target_score[:, None] + 1e-12, axis=1)
        ties = np.abs(scores - target_score[:, None]) <= 1e-12
        rank += np.sum(ties & (np.arange(len(bank))[None, :] < target_position[:, None]), axis=1)
        reciprocal[start:start + len(scores)] = 1.0 / rank
    frame = pd.DataFrame({"profile": text_id[query_indices], "rr": reciprocal})
    return float(frame.groupby("profile").rr.mean().mean()), reciprocal


def retrieval_summary(representations: dict[str, np.ndarray], table: pd.DataFrame,
                      text_file: Path, seeds: list[int], boot: int) -> dict:
    text = np.load(text_file, allow_pickle=True)
    text_id = np.asarray(text["text_id"], dtype=int)
    embeddings = np.asarray(text["unique_embeddings"], dtype=np.float32)
    query = np.where((table.splits == "validation").to_numpy())[0]
    scores, row_rr = {}, {}
    for arm in ("correct", "within_label", "global"):
        values, rows = [], []
        for seed_index in range(len(seeds)):
            value, reciprocal = profile_mrr(
                representations[arm][seed_index], query, text_id, embeddings)
            values.append(value); rows.append(reciprocal)
        scores[arm] = {"macro_profile_mrr": float(np.mean(values)), "per_seed": values}
        row_rr[arm] = np.stack(rows)

    profiles = np.unique(text_id[query])
    by_profile = {
        arm: np.stack([
            [np.mean(row_rr[arm][seed, text_id[query] == profile]) for profile in profiles]
            for seed in range(len(seeds))])
        for arm in ("correct", "within_label", "global")
    }
    rng, values = np.random.RandomState(20260825), []
    for _ in range(boot):
        pp = rng.choice(len(profiles), len(profiles), replace=True)
        ss = rng.choice(len(seeds), len(seeds), replace=True)
        values.append(float(np.mean([
            np.mean(by_profile["correct"][s, pp] - by_profile["within_label"][s, pp])
            for s in ss])))
    scores["correct_minus_within"] = {
        "observed": float(np.mean(by_profile["correct"] - by_profile["within_label"])),
        "ci": list(map(float, np.percentile(values, [2.5, 97.5]))),
        "n_profiles": int(len(profiles)), "n_queries": int(len(query)),
    }
    return scores


def make_probe_target(table: pd.DataFrame, spec: dict) -> pd.Series:
    values = table[spec["field"]]
    observed = values.astype(str) != "[MISSING]"
    if spec["kind"] == "threshold":
        result = pd.to_numeric(values, errors="coerce") >= float(spec["threshold"])
        return result.where(observed)
    positive = {str(value).strip().casefold() for value in spec["positive_values"]}
    return values.astype(str).str.strip().str.casefold().isin(positive).where(observed)


def metadata_matrix(table: pd.DataFrame, fields: list[str], train: np.ndarray) \
        -> tuple[np.ndarray, list[str]]:
    """Training-vocabulary one-hot schema baseline; labels/domains never enter it."""

    columns, names = [], []
    for field in fields:
        values = table[field].astype(str).to_numpy()
        levels = sorted(np.unique(values[train]).tolist())
        for level in levels:
            columns.append((values == level).astype(np.float32))
            names.append(f"{field}={level}")
        columns.append((~np.isin(values, levels)).astype(np.float32))
        names.append(f"{field}=[UNSEEN_IN_TRAIN]")
    if not columns:
        raise RuntimeError("metadata-only baseline has no frozen schema fields")
    return np.stack(columns, axis=1), names


def execute(config_file: str, backbone: str) -> Path:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    gate = json.loads((paths["public"] / "data_gate.json").read_text())
    if gate["verdict"] != "GO":
        raise RuntimeError("data gate is not GO; formal evaluation is forbidden")
    table_path = paths["private"] / "participant_manifest.csv"
    table = pd.read_csv(table_path)
    table["recruitment_source"] = table[config["protocol"]["cohort_field"]].astype(str)
    train = (table.splits == "train").to_numpy()
    validation = (table.splits == "validation").to_numpy()
    standard = (table.splits == "test").to_numpy()
    matched = table.in_matched_test.astype(bool).to_numpy()
    if np.any(train & validation) or np.any((train | validation) & standard):
        raise RuntimeError("participant split overlap")
    y = table.y.astype(int).to_numpy()
    folds = make_validation_folds(table, validation)
    settings = config["models"]["alignment"]
    seeds = list(map(int, settings["seeds"]))
    boot = int(config["protocol"].get("bootstrap", 2000))
    representations = load_representations(paths, table, backbone, seeds)

    metadata, metadata_features = metadata_matrix(
        table, list(config["protocol"]["schema_fields"]), train)
    representations["metadata_only"] = np.repeat(
        metadata[None, :, :], len(seeds), axis=0)
    representations["raw_audio_plus_metadata"] = np.repeat(
        np.concatenate([representations["raw_audio"][0], metadata], axis=1)[None, :, :],
        len(seeds), axis=0)

    probabilities, logits, fits = {}, {}, {}
    for arm in EVALUATION_ARMS:
        arm_probability, arm_logits, arm_fits = [], [], []
        fit_indices = [0] if arm in ("raw_audio", *CONTEXT_ARMS) else range(len(seeds))
        for seed_index in fit_indices:
            head = fit_disease_head(
                representations[arm][seed_index], y, train, validation, folds,
                seeds[seed_index])
            arm_probability.append(head.calibrated)
            arm_logits.append(head.logits)
            arm_fits.append({"seed": seeds[seed_index], "C": head.C,
                             "selection": head.selection,
                             "calibrator_coef": head.calibrator_coef,
                             "calibrator_intercept": head.calibrator_intercept,
                             "representation_sha16": sha16_array(
                                 representations[arm][seed_index])})
        if arm in ("raw_audio", *CONTEXT_ARMS):
            arm_probability *= len(seeds); arm_logits *= len(seeds)
            arm_fits = [{**arm_fits[0], "seed": seed,
                         "replicated_deterministic_reference": True} for seed in seeds]
        probabilities[arm] = np.stack(arm_probability)
        logits[arm] = np.stack(arm_logits)
        fits[arm] = arm_fits

    populations = {"standard": standard, "matched": matched}
    metrics = {population: {} for population in populations}
    comparisons = {population: {} for population in populations}
    comparison_pairs = (("correct", "within_label", "C-W"),
                        ("within_label", "global", "W-G"),
                        ("correct", "raw_audio", "C-R"),
                        ("raw_audio_plus_metadata", "metadata_only", "RM-M"))
    for population, mask in populations.items():
        for arm in EVALUATION_ARMS:
            metrics[population][arm] = absolute_metrics(
                probabilities[arm][:, mask], y[mask], boot, 20260825)
        for left, right, name in comparison_pairs:
            comparisons[population][name] = {}
            for metric_name, fn in (("delta_auroc", auroc),
                                    ("delta_neg_nll", lambda yy, pp: -mean_nll(yy, pp)),
                                    ("delta_neg_brier", lambda yy, pp: -brier(yy, pp))):
                if population == "matched":
                    result = pair_cluster_ci(
                        probabilities[left][:, mask], probabilities[right][:, mask],
                        y[mask], table.loc[mask, "pair_id"].astype(str).to_numpy(),
                        fn, boot, 20260825)
                else:
                    result = paired_hierarchical_ci(
                        probabilities[left][:, mask], probabilities[right][:, mask],
                        y[mask], fn, boot, 20260825)
                comparisons[population][name][metric_name] = result

    retrieval = retrieval_summary(
        representations, table, paths["models"] / "text_embeddings.npz", seeds, boot)

    probes = {}
    for probe_name, spec in config["protocol"].get("probes", {}).items():
        target = make_probe_target(table, spec)
        observed = target.notna().to_numpy()
        target_y = target.fillna(False).astype(int).to_numpy()
        if np.unique(target_y[train & observed]).size < 2 or \
                np.unique(target_y[validation & observed]).size < 2:
            probes[probe_name] = {"status": "not_estimable_in_source"}
            continue
        probe_logits = {}
        for arm in ARMS:
            values = []
            fit_indices = [0] if arm == "raw_audio" else range(len(seeds))
            for seed_index in fit_indices:
                value, _, _, _ = fit_probe(
                    representations[arm][seed_index], target, train, folds,
                    seeds[seed_index])
                values.append(value)
            if arm == "raw_audio":
                values *= len(seeds)
            probe_logits[arm] = np.stack(values)
        probe_mask = matched & observed
        if np.unique(target_y[probe_mask]).size < 2:
            probes[probe_name] = {"status": "not_estimable_in_matched"}
            continue
        probes[probe_name] = {"status": "ok", "matched": {}}
        for left, right, name in comparison_pairs[:3]:
            probes[probe_name]["matched"][name] = pair_cluster_ci(
                probe_logits[left][:, probe_mask], probe_logits[right][:, probe_mask],
                target_y[probe_mask], table.loc[probe_mask, "pair_id"].astype(str).to_numpy(),
                auroc, boot, 20260825)

    correspondence_ci = retrieval["correct_minus_within"]["ci"]
    transfer_auc = comparisons["matched"]["C-W"]["delta_auroc"]
    transfer_nll = comparisons["matched"]["C-W"]["delta_neg_nll"]
    auc_margin = float(config["protocol"]["auroc_equivalence_margin"])
    nll_margin = float(config["protocol"]["nll_equivalence_margin"])
    correspondence_established = correspondence_ci[0] > 0
    transfer_positive = transfer_auc["ci"][0] > 0 and transfer_nll["ci"][0] > 0
    equivalence_compatible = (
        transfer_auc["ci"][0] >= -auc_margin and transfer_auc["ci"][1] <= auc_margin and
        transfer_nll["ci"][0] >= -nll_margin and transfer_nll["ci"][1] <= nll_margin)
    if not correspondence_established:
        branch = "alignment_correspondence_not_established"
    elif transfer_positive:
        branch = "correspondence_and_matched_transfer_gain"
    elif equivalence_compatible:
        branch = "correspondence_without_matched_transfer_equivalence_compatible"
    else:
        branch = "correspondence_gain_but_matched_transfer_inconclusive"

    public = {
        "format_version": "cambridge-external-results-v1",
        "dataset": config.get("dataset"), "backbone": backbone,
        "standing": "controlled-access external audit; interpret according to frozen branches",
        "n": {"all": len(table), "train": int(train.sum()),
              "validation": int(validation.sum()), "standard_test": int(standard.sum()),
              "matched_test": int(matched.sum()), "matched_pairs": int(matched.sum() // 2)},
        "metrics": metrics, "comparisons": comparisons,
        "profile_retrieval": retrieval, "probes": probes,
        "metadata_baseline": {
            "fields": list(config["protocol"]["schema_fields"]),
            "n_training_vocabulary_features": len(metadata_features),
            "encoding": "source-train-vocabulary one-hot with explicit unseen indicator",
            "labels_and_cohort_excluded": True,
        },
        "frozen_interpretation": {
            "branch": branch,
            "correspondence_established": correspondence_established,
            "matched_transfer_positive_on_both_metrics": transfer_positive,
            "matched_equivalence_compatible_on_both_metrics": equivalence_compatible,
            "margins": {"delta_auroc": auc_margin, "delta_neg_nll": nll_margin},
            "note": ("A confidence interval containing zero is inconclusive unless the "
                     "entire interval is also inside the frozen equivalence margin."),
        },
        "fits": fits, "cohort_sha256": sha256_file(table_path),
        "privacy": "aggregate output; no participant identifiers or row-level predictions",
    }
    output = paths["public"] / f"formal_results_{backbone}.json"
    atomic_json(output, public)
    private = paths["private"] / f"predictions_{backbone}.npz"
    arrays = {"participants": table.participant_identifier.astype(str).to_numpy(),
              "y": y, "seeds": np.asarray(seeds)}
    for arm in EVALUATION_ARMS:
        arrays[f"{arm}__calibrated"] = probabilities[arm]
        arrays[f"{arm}__logits"] = logits[arm]
    np.savez_compressed(private, **arrays)
    print(f"wrote aggregate external results {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ast", "opera_ct"), required=True)
    args = parser.parse_args()
    execute(args.config, args.backbone)


if __name__ == "__main__":
    main()
