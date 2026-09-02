import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "src" / "evaluate_formal.py"
SPEC = importlib.util.spec_from_file_location("coda_evaluate_formal", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_validation_folds_cover_only_validation_and_both_classes():
    table = pd.DataFrame({"y": np.tile([0, 1], 30)})
    mask = np.zeros(len(table), dtype=bool)
    mask[10:] = True
    folds = MODULE.validation_folds(table, mask)
    assert np.all(folds[~mask] == -1)
    assert set(folds[mask]) == set(range(5))
    for fold in range(5):
        assert set(table.loc[folds == fold, "y"]) == {0, 1}


def test_probe_folds_stratify_probe_and_exclude_missing():
    target = pd.Series(([False] * 10 + [True] * 10 + [None] * 5), dtype="boolean")
    mask = np.ones(len(target), dtype=bool)
    folds = MODULE.probe_validation_folds(target, mask)
    assert np.all(folds[target.isna()] == -1)
    for fold in range(5):
        assert set(target.iloc[np.where(folds == fold)[0]].astype(bool)) == {False, True}


def test_profile_mrr_is_one_for_exact_unique_embeddings():
    bank = np.eye(4, dtype=np.float32)
    text_id = np.arange(4)
    value, reciprocal = MODULE.profile_mrr(bank, np.arange(4), text_id, bank)
    assert value == 1.0
    np.testing.assert_array_equal(reciprocal, np.ones(4))


def test_pair_cluster_ci_preserves_paired_direction():
    y = np.array([0, 1, 0, 1])
    pair_id = np.array(["a", "a", "b", "b"])
    left = np.tile(np.array([0.1, 0.9, 0.2, 0.8]), (3, 1))
    right = np.tile(np.array([0.4, 0.6, 0.45, 0.55]), (3, 1))
    result = MODULE.pair_cluster_ci(
        left, right, y, pair_id, MODULE.auroc, boot=100, seed=7)
    assert result["observed"] == 0.0  # both rank the four examples perfectly
    assert result["n_pairs"] == 2
    assert result["n_seeds"] == 3
