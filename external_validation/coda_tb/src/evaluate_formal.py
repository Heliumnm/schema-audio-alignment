"""Frozen CODA disease, retrieval, fusion, and information-channel evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
ENGINE = REPO / "external_validation" / "cambridge_covid_sounds" / "src"
sys.path.insert(0, str(ENGINE)); sys.path.insert(0, str(REPO / "src"))

from common import atomic_json, load_config, output_paths, sha16_array, sha256_file
from evaluate import (absolute_metrics, brier, load_representations, make_probe_target,
                      mean_nll, metadata_matrix, pair_cluster_ci, profile_mrr)
from eval_metadata_alignment import fit_disease_head, fit_probe, paired_hierarchical_ci
from audio_baselines_v2 import auroc


SEEDS = [0, 1, 2, 3, 4]
PAIRED_ARMS = ("correct", "within_label", "within_label_sex", "global")
AUDIO_ARMS = ("raw_audio", *PAIRED_ARMS)
FUSION_ARMS = ("metadata_only", "metadata_plus_raw", "metadata_plus_correct",
               "metadata_plus_within", "metadata_plus_within_label_sex",
               "metadata_plus_global")
ALL_ARMS = AUDIO_ARMS + FUSION_ARMS


def validation_folds(table: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    indices = np.where(mask)[0]
    folds = np.full(len(table), -1, dtype=np.int16)
    splitter = StratifiedKFold(5, shuffle=True, random_state=20260903)
    for fold, (_, held) in enumerate(splitter.split(indices, table.loc[indices, "y"])):
        folds[indices[held]] = fold
    if any(np.unique(table.loc[folds == fold, "y"]).size != 2 for fold in range(5)):
        raise RuntimeError("a frozen validation fold lacks one TB class")
    return folds


def probe_validation_folds(target: pd.Series, mask: np.ndarray) -> np.ndarray:
    """Fixed target-stratified folds for one probe's one-SE readout selection.

    Disease-stratified folds are correct for the TB head but need not contain both classes
    of every secondary probe. Missing probe targets remain outside all folds.
    """
    observed = target.notna().to_numpy()
    indices = np.where(mask & observed)[0]
    values = target.iloc[indices].astype(int).to_numpy()
    counts = np.bincount(values, minlength=2)
    folds = np.full(len(target), -1, dtype=np.int16)
    if np.any(counts < 5):
        return folds
    splitter = StratifiedKFold(5, shuffle=True, random_state=20260903)
    for fold, (_, held) in enumerate(splitter.split(indices, values)):
        folds[indices[held]] = fold
    if any(np.unique(target.iloc[folds == fold].astype(int)).size != 2
           for fold in range(5)):
        raise RuntimeError("a frozen probe validation fold lacks one class")
    return folds


def retrieval(representations: dict[str, np.ndarray], table: pd.DataFrame,
              text_path: Path, boot: int) -> dict:
    text = np.load(text_path, allow_pickle=True)
    text_id = np.asarray(text["text_id"], dtype=int)
    bank = np.asarray(text["unique_embeddings"], dtype=np.float32)
    query = np.where((table.splits == "validation").to_numpy())[0]
    if len(np.unique(text_id[query])) != len(query):
        raise RuntimeError("CODA validation profiles are no longer unique")
    rows, summary = {}, {}
    for arm in PAIRED_ARMS:
        rr, per_seed = [], []
        for seed_index in range(5):
            value, reciprocal = profile_mrr(
                representations[arm][seed_index], query, text_id, bank)
            rr.append(reciprocal); per_seed.append(value)
        rows[arm] = np.stack(rr)
        summary[arm] = {"mrr": float(np.mean(per_seed)), "per_seed": per_seed,
                        "r_at_1": float(np.mean(rows[arm] == 1)),
                        "r_at_10": float(np.mean(rows[arm] >= 0.1))}
    rng = np.random.RandomState(20260903)
    comparisons = {}
    for left, right, name in (("correct", "within_label_sex", "C-Wys"),
                              ("within_label_sex", "within_label", "Wys-W"),
                              ("correct", "within_label", "C-W"),
                              ("within_label", "global", "W-G")):
        delta = rows[left] - rows[right]; values = []
        for _ in range(boot):
            participants = rng.choice(delta.shape[1], delta.shape[1], replace=True)
            seeds = rng.choice(delta.shape[0], delta.shape[0], replace=True)
            values.append(float(np.mean(delta[np.ix_(seeds, participants)])))
        comparisons[name] = {"observed": float(delta.mean()),
                             "ci": list(map(float, np.percentile(values, [2.5, 97.5]))),
                             "per_seed": delta.mean(axis=1).tolist()}
    return {"n_queries": int(len(query)), "n_unique_candidates": int(len(query)),
            "arms": summary, "comparisons": comparisons}


def paired(population: str, left: np.ndarray, right: np.ndarray, y: np.ndarray,
           table: pd.DataFrame, mask: np.ndarray, fn: Callable, boot: int, seed: int) -> dict:
    if population == "matched_target":
        return pair_cluster_ci(left[:, mask], right[:, mask], y[mask],
                               table.loc[mask, "pair_id"].astype(str).to_numpy(),
                               fn, boot, seed)
    return paired_hierarchical_ci(left[:, mask], right[:, mask], y[mask], fn, boot, seed)


def execute(config_file: str, backbone: str) -> Path:
    config, config_path = load_config(config_file); paths = output_paths(config, config_path)
    gate = json.loads((paths["public"] / "data_gate.json").read_text())
    if gate.get("protocol") != "coda-match-first-v2-formal-v1" or gate.get("verdict") != "GO":
        raise RuntimeError("wrong or non-GO formal input gate")
    table_path = paths["private"] / "participant_manifest.csv"
    table = pd.read_csv(table_path); y = table.y.astype(int).to_numpy()
    train = (table.splits == "train").to_numpy(); val = (table.splits == "validation").to_numpy()
    populations = {"source_test": (table.splits == "test").to_numpy(),
                   "matched_target": table.in_matched_test.astype(bool).to_numpy()}
    if any(np.any((train | val) & mask) for mask in populations.values()):
        raise RuntimeError("test overlap")
    folds = validation_folds(table, val); boot = int(config["protocol"]["bootstrap"])
    reps = load_representations(paths, table, backbone, SEEDS)
    metadata, metadata_names = metadata_matrix(
        table, list(config["protocol"]["schema_fields"]), train)
    reps["metadata_only"] = np.repeat(metadata[None], 5, axis=0)
    reps["metadata_plus_raw"] = np.repeat(
        np.concatenate([metadata, reps["raw_audio"][0]], axis=1)[None], 5, axis=0)
    for audio_arm, fusion_arm in (("correct", "metadata_plus_correct"),
                                  ("within_label", "metadata_plus_within"),
                                  ("within_label_sex", "metadata_plus_within_label_sex"),
                                  ("global", "metadata_plus_global")):
        reps[fusion_arm] = np.stack([
            np.concatenate([metadata, reps[audio_arm][seed]], axis=1) for seed in range(5)])

    probabilities, logits, fits = {}, {}, {}
    for arm in ALL_ARMS:
        deterministic = arm in ("raw_audio", "metadata_only", "metadata_plus_raw")
        seed_indices = [0] if deterministic else range(5)
        arm_prob, arm_logits, arm_fit = [], [], []
        for seed_index in seed_indices:
            head = fit_disease_head(reps[arm][seed_index], y, train, val, folds,
                                    SEEDS[seed_index])
            arm_prob.append(head.calibrated); arm_logits.append(head.logits)
            arm_fit.append({"seed": SEEDS[seed_index], "C": head.C,
                            "selection": head.selection,
                            "calibrator_coef": head.calibrator_coef,
                            "calibrator_intercept": head.calibrator_intercept,
                            "representation_sha16": sha16_array(reps[arm][seed_index])})
        if deterministic:
            arm_prob *= 5; arm_logits *= 5
            arm_fit = [{**arm_fit[0], "seed": seed,
                        "replicated_deterministic_reference": True} for seed in SEEDS]
        probabilities[arm] = np.stack(arm_prob); logits[arm] = np.stack(arm_logits)
        fits[arm] = arm_fit

    audio_comparisons = (("correct", "within_label_sex", "C-Wys"),
                   ("within_label_sex", "within_label", "Wys-W"),
                   ("correct", "within_label", "C-W"),
                   ("within_label", "global", "W-G"),
                   ("correct", "raw_audio", "C-R"))
    comparisons = (*audio_comparisons,
                   ("metadata_plus_raw", "metadata_only", "MR-M"),
                   ("metadata_plus_correct", "metadata_plus_within_label_sex", "MC-MWys"),
                   ("metadata_plus_within_label_sex", "metadata_plus_within", "MWys-MW"),
                   ("metadata_plus_correct", "metadata_plus_within", "MC-MW"),
                   ("metadata_plus_within", "metadata_plus_global", "MW-MG"),
                   ("metadata_plus_correct", "metadata_plus_raw", "MC-MR"))
    metrics, deltas = {}, {}
    for population, mask in populations.items():
        metrics[population] = {arm: absolute_metrics(probabilities[arm][:, mask], y[mask],
                                                     boot, 20260903)
                               for arm in ALL_ARMS}
        deltas[population] = {}
        for left, right, name in comparisons:
            deltas[population][name] = {
                "delta_auroc": paired(population, probabilities[left], probabilities[right],
                                       y, table, mask, auroc, boot, 20260903),
                "delta_neg_nll": paired(
                    population, probabilities[left], probabilities[right], y, table, mask,
                    lambda yy, pp: -mean_nll(yy, pp), boot, 20260904),
                "delta_neg_brier": paired(
                    population, probabilities[left], probabilities[right], y, table, mask,
                    lambda yy, pp: -brier(yy, pp), boot, 20260905),
            }

    profile = retrieval(reps, table, paths["models"] / "text_embeddings.npz", boot)
    probes = {}
    probe_logits = {}
    for probe_name, spec in config["protocol"]["probes"].items():
        target = make_probe_target(table, spec); observed = target.notna().to_numpy()
        target_y = target.fillna(False).astype(int).to_numpy()
        if np.unique(target_y[train & observed]).size < 2 or \
                np.unique(target_y[val & observed]).size < 2:
            probes[probe_name] = {"status": "not_estimable_in_source"}; continue
        probe_folds = probe_validation_folds(target, val)
        if np.any(probe_folds[val & observed] < 0):
            probes[probe_name] = {"status": "not_estimable_in_source"}; continue
        probe_logits[probe_name] = {}
        for arm in AUDIO_ARMS:
            indices = [0] if arm == "raw_audio" else range(5); values = []
            for seed_index in indices:
                value, _, _, _ = fit_probe(reps[arm][seed_index], target, train, probe_folds,
                                           SEEDS[seed_index])
                values.append(value)
            if arm == "raw_audio": values *= 5
            probe_logits[probe_name][arm] = np.stack(values)
        target_mask = populations["matched_target"] & observed
        if np.unique(target_y[target_mask]).size < 2:
            probes[probe_name] = {"status": "not_estimable_in_target"}; continue
        probes[probe_name] = {"status": "ok", "matched_target": {}}
        for left, right, name in audio_comparisons:
            probes[probe_name]["matched_target"][name] = paired(
                "matched_target", probe_logits[probe_name][left],
                probe_logits[probe_name][right], target_y, table, target_mask,
                auroc, boot, 20260906)

    # The country taxonomy endpoint is the mean of seven jointly bootstrapped one-vs-rest
    # AUROCs; fitting and resampling are shared, rather than averaging seven unrelated CIs.
    countries = [f"country_{value}" for value in ("IN", "MG", "PH", "SA", "TZ", "UG", "VN")]
    target_mask = populations["matched_target"]
    pair_ids = table.loc[target_mask, "pair_id"].astype(str).to_numpy()
    unique_pairs = np.unique(pair_ids); groups = {p: np.where(pair_ids == p)[0] for p in unique_pairs}
    country_y = {name: make_probe_target(table, config["protocol"]["probes"][name])
                 .fillna(False).astype(int).to_numpy()[target_mask] for name in countries}
    def macro(seed: int, indices: np.ndarray, arm: str) -> float:
        values = []
        for name in countries:
            yy = country_y[name][indices]
            if np.unique(yy).size == 2:
                values.append(auroc(yy, probe_logits[name][arm][seed, target_mask][indices]))
        return float(np.mean(values))
    observed = np.mean([macro(seed, np.arange(target_mask.sum()), "correct") -
                        macro(seed, np.arange(target_mask.sum()), "within_label_sex")
                        for seed in range(5)])
    rng, values = np.random.RandomState(20260907), []
    for _ in range(boot):
        sampled = rng.choice(unique_pairs, len(unique_pairs), replace=True)
        indices = np.concatenate([groups[pair] for pair in sampled])
        seeds = rng.choice(5, 5, replace=True)
        values.append(float(np.mean([macro(seed, indices, "correct") -
                                    macro(seed, indices, "within_label_sex") for seed in seeds])))
    probes["country_macro_ovr"] = {"status": "ok", "matched_target": {"C-Wys": {
        "observed": float(observed),
        "ci": list(map(float, np.percentile(values, [2.5, 97.5]))),
        "n_classes": 7, "n_pairs": 100, "n_seeds": 5}}}

    correspondence = profile["comparisons"]["C-Wys"]
    transfer_auc = deltas["matched_target"]["C-Wys"]["delta_auroc"]
    transfer_nll = deltas["matched_target"]["C-Wys"]["delta_neg_nll"]
    established = correspondence["ci"][0] > 0
    positive = transfer_auc["ci"][0] > 0 and transfer_nll["ci"][0] > 0
    equivalent = (transfer_auc["ci"][0] >= -0.02 and transfer_auc["ci"][1] <= 0.02 and
                  transfer_nll["ci"][0] >= -0.01 and transfer_nll["ci"][1] <= 0.01)
    branch = ("alignment_correspondence_not_established" if not established else
              "correspondence_and_matched_transfer_gain" if positive else
              "correspondence_without_matched_transfer_equivalence_compatible" if equivalent else
              "correspondence_gain_but_matched_transfer_inconclusive")
    result = {
        "format_version": "coda-formal-model-v2", "dataset": config["dataset"],
        "backbone": backbone,
        "rerun_protocol": config["protocol"].get("rerun_protocol"),
        "standing": config["protocol"].get(
            "standing", "locked external test; prior exploratory results archived"),
        "n": {"all": len(table), "train": int(train.sum()), "validation": int(val.sum()),
              "source_test": int(populations["source_test"].sum()),
              "matched_target": int(target_mask.sum()), "matched_pairs": 100},
        "profile_retrieval": profile, "metrics": metrics, "comparisons": deltas,
        "probes": probes, "fits": fits,
        "metadata_baseline": {"fields": config["protocol"]["schema_fields"],
                              "n_features": len(metadata_names)},
        "frozen_interpretation": {"branch": branch,
                                  "correspondence_established": established,
                                  "matched_transfer_positive_on_both_metrics": positive,
                                  "matched_equivalence_compatible_on_both_metrics": equivalent,
                                  "margins": {"delta_auroc": 0.02,
                                              "delta_neg_nll": 0.01},
                                  "primary_contrast": "C-Wys"},
        "cohort_sha256": sha256_file(table_path),
        "privacy": "aggregate output only; no participant identifiers",
    }
    output = paths["public"] / f"formal_results_{backbone}.json"; atomic_json(output, result)
    arrays = {"participants": table.participant_identifier.astype(str).to_numpy(),
              "y": y, "seeds": np.asarray(SEEDS)}
    for arm in ALL_ARMS:
        arrays[f"{arm}__calibrated"] = probabilities[arm]
        arrays[f"{arm}__logits"] = logits[arm]
    np.savez_compressed(paths["private"] / f"predictions_{backbone}.npz", **arrays)
    print(f"wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", choices=("ast", "opera_ct", "hear"), required=True)
    args = parser.parse_args(); execute(args.config, args.backbone)


if __name__ == "__main__":
    main()
