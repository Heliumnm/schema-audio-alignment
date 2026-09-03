"""Model-blind global feasibility sensitivity for the frozen Coswara gate.

This does not overwrite the preregistered greedy result.  It asks whether a
strictly balanced 100-pair participant set exists under constraints at least as
strict as the original gate, before any representation or prediction is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from freeze_coswara_external_split import (
    EXACT_FIELDS,
    MAX_ABS_SMD,
    MAX_LEVEL_PROPORTION_DIFFERENCE,
    PROPORTION_FIELDS,
    SMD_CATEGORICAL_FIELDS,
    atomic_csv,
    atomic_json,
    balance_report,
    country_groups,
    exact_stratum,
    match_rows,
    prepare_rows,
    sha256_file,
)


TARGET_PAIRS = 100


def allowed_binary_smd_difference(total_count: int, n_per_class: int) -> int | None:
    """Largest |positive-negative count| whose exact binary SMD is <= 0.12."""
    best = -1
    lower = max(0, total_count - n_per_class)
    upper = min(n_per_class, total_count)
    for negative in range(lower, upper + 1):
        positive = total_count - negative
        p0, p1 = negative / n_per_class, positive / n_per_class
        denominator = math.sqrt((p0 * (1 - p0) + p1 * (1 - p1)) / 2)
        if denominator == 0:
            smd = 0.0 if p0 == p1 else math.inf
        else:
            smd = abs((p1 - p0) / denominator)
        if smd <= MAX_ABS_SMD + 1e-12:
            best = max(best, abs(positive - negative))
    return None if best < 0 else best


class ConstraintBuilder:
    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        self.rows: list[int] = []
        self.cols: list[int] = []
        self.data: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []

    def add(self, coefficients: dict[int, float], lower: float, upper: float) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if value:
                self.rows.append(row)
                self.cols.append(column)
                self.data.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def build(self) -> LinearConstraint:
        matrix = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=(len(self.lower), self.n_variables),
        ).tocsr()
        return LinearConstraint(matrix, np.asarray(self.lower), np.asarray(self.upper))


def solve_selection(
    candidates: list[dict[str, object]], target_pairs: int = TARGET_PAIRS,
    time_limit_s: float = 600.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Select an exactly balanced cohort with a deterministic MILP feasibility solve."""
    ordered = sorted(candidates, key=lambda row: str(row["participant_id"]))
    x_count = len(ordered)
    level_specs: list[tuple[str, str, list[tuple[int, int, int]]]] = []
    next_index = x_count
    for field in SMD_CATEGORICAL_FIELDS:
        levels = sorted({str(row[field]) for row in ordered})
        for level in levels:
            states = []
            for total in range(2 * target_pairs + 1):
                max_difference = allowed_binary_smd_difference(total, target_pairs)
                if max_difference is not None:
                    states.append((total, next_index, max_difference))
                    next_index += 1
            level_specs.append((field, level, states))

    builder = ConstraintBuilder(next_index)
    labels = np.asarray([int(row["label"]) for row in ordered])
    for label in (0, 1):
        builder.add(
            {index: 1.0 for index in np.flatnonzero(labels == label)},
            target_pairs,
            target_pairs,
        )

    strata: dict[tuple[str, ...], dict[int, list[int]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    for index, row in enumerate(ordered):
        strata[exact_stratum(row)][int(row["label"])].append(index)
    for classes in strata.values():
        coefficients = {index: 1.0 for index in classes[0]}
        coefficients.update({index: -1.0 for index in classes[1]})
        builder.add(coefficients, 0.0, 0.0)

    # Stronger than the original age-SMD constraint and linear at fixed n.
    builder.add(
        {
            index: float(row["age"]) * (1.0 if int(row["label"]) == 1 else -1.0)
            for index, row in enumerate(ordered)
        },
        0.0,
        0.0,
    )

    for field, level, states in level_specs:
        builder.add({index: 1.0 for _, index, _ in states}, 1.0, 1.0)
        negative = [
            index for index, row in enumerate(ordered)
            if int(row["label"]) == 0 and str(row[field]) == level
        ]
        positive = [
            index for index, row in enumerate(ordered)
            if int(row["label"]) == 1 and str(row[field]) == level
        ]
        total_coefficients = {index: 1.0 for index in [*negative, *positive]}
        total_coefficients.update(
            {index: -float(total) for total, index, _ in states}
        )
        builder.add(total_coefficients, 0.0, 0.0)
        max_difference = {
            index: float(bound) for _, index, bound in states
        }
        difference = {index: -1.0 for index in negative}
        difference.update({index: 1.0 for index in positive})
        builder.add(
            {**difference, **{index: -value for index, value in max_difference.items()}},
            -math.inf,
            0.0,
        )
        builder.add(
            {**{index: -value for index, value in difference.items()},
             **{index: -value for index, value in max_difference.items()}},
            -math.inf,
            0.0,
        )

    # Use seven counts, not eight, so float reconstruction cannot turn 0.08
    # into 0.08000000000000002 and spuriously fail the original checker.
    fine_bound = math.floor(target_pairs * MAX_LEVEL_PROPORTION_DIFFERENCE - 1e-12)
    for field in PROPORTION_FIELDS:
        for level in sorted({str(row[field]) for row in ordered}):
            difference: dict[int, float] = {}
            for index, row in enumerate(ordered):
                if str(row[field]) == level:
                    difference[index] = 1.0 if int(row["label"]) == 1 else -1.0
            builder.add(difference, -fine_bound, fine_bound)

    result = milp(
        c=np.zeros(next_index, dtype=float),
        integrality=np.ones(next_index, dtype=np.uint8),
        bounds=Bounds(np.zeros(next_index), np.ones(next_index)),
        constraints=builder.build(),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0},
    )
    solver = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "n_candidate_participants": x_count,
        "n_variables": next_index,
        "n_constraints": len(builder.lower),
        "time_limit_s": float(time_limit_s),
    }
    if not result.success or result.x is None:
        return [], solver
    selected = [row for row, value in zip(ordered, result.x[:x_count]) if value > 0.5]
    if sum(int(row["label"]) == 0 for row in selected) != target_pairs:
        raise RuntimeError("solver output has wrong negative count")
    if sum(int(row["label"]) == 1 for row in selected) != target_pairs:
        raise RuntimeError("solver output has wrong positive count")
    return selected, solver


def execute(qc: Path, pairs_out: Path, report_out: Path, time_limit_s: float) -> dict:
    rows, excluded = prepare_rows(qc)
    candidates = [
        row for row in rows
        if (int(row["label"]) == 1 and row["test_status"] == "p")
        or (int(row["label"]) == 0 and row["test_status"] == "n")
    ]
    mapping = country_groups(candidates)
    for row in rows:
        row["country_group"] = mapping.get(str(row["country"]), "OTHER")
    selected, solver = solve_selection(candidates, TARGET_PAIRS, time_limit_s)
    by_id = {str(row["participant_id"]): row for row in rows}
    pairs = match_rows(selected, "global_constrained_primary") if selected else []
    balance = balance_report(pairs, by_id)
    passed = len(pairs) == TARGET_PAIRS and bool(balance.get("passes", False))
    verdict = "SECONDARY_GO" if passed else (
        "INCONCLUSIVE" if solver["status"] == 1 else "NO_FEASIBLE_STRICT_SOLUTION"
    )
    pair_fields = ["set", "pair_id", "stratum", "negative_id", "positive_id", "cost"]
    atomic_csv(pairs_out, pairs, pair_fields)
    report = {
        "format_version": "coswara-global-gate-sensitivity-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "model_scores_read": False,
        "post_hoc_data_only_sensitivity": True,
        "qc_manifest": str(qc),
        "qc_manifest_sha256": sha256_file(qc),
        "n_eligible": len(rows),
        "n_strict_candidates": len(candidates),
        "target_pairs": TARGET_PAIRS,
        "excluded": excluded,
        "solver": solver,
        "balance": balance,
        "pairs_sha256": sha256_file(pairs_out),
        "interpretation": (
            "A passing cohort permits only a post-hoc external stress test; it does not "
            "replace the preregistered Coswara NO-GO or establish external confirmation."
        ),
    }
    atomic_json(report_out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--pairs-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, default=600.0)
    args = parser.parse_args()
    report = execute(args.qc, args.pairs_out, args.report_out, args.time_limit_s)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] != "SECONDARY_GO":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
