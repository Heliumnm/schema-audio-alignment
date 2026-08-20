"""Frozen E3 probability-transport audit.

The script consumes already frozen raw disease-head logits and Standard-validation Platt
probabilities.  It never retrains an audio projector or disease head and never chooses a
transport parameter on matched outcomes.  The shrinkage grid is reported in full.

Input archives are supplied as ``LABEL=PATH`` pairs.  Every prediction prefix that has
both ``__raw_logits`` and ``__calibrated_prob`` is audited.  This supports the existing
AST/OPERA disease archives and the E1 fusion archives without result-specific code.

``--self-test`` is synthetic-only and opens no repository result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)

from audio_baselines_v2 import UNIT, auroc, calib_diag, logloss
from eval_metadata_alignment import atomic_json, atomic_npz, load_cohort, sha16


LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
EPS = 1e-8


def file_sha256(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _as_seed_matrix(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] == 1:
        x = np.repeat(x, len(SEEDS), axis=0)
    assert x.shape == (len(SEEDS), n)
    assert np.isfinite(x).all()
    return x


def parse_sources(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"source must be LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        assert label and label not in out and all(c.isalnum() or c in "-_." for c in label)
        out[label] = Path(raw_path)
    assert out, "at least one --source is required"
    return out


def discover_archive(path: Path, participants: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as z:
        assert "participants" in z.files and np.array_equal(
            np.asarray(z["participants"]), participants), f"participant mismatch: {path}"
        if "seeds" in z.files:
            seeds = np.asarray(z["seeds"], dtype=int).tolist()
            assert seeds in ([0, 1, 2, 3, 4], [0]), f"unexpected seeds in {path}: {seeds}"
        prefixes = sorted({
            key[:-len("__calibrated_prob")]
            for key in z.files
            if key.endswith("__calibrated_prob")
            and not key.endswith("__val_oof_calibrated_prob")
            and key[:-len("__calibrated_prob")] + "__raw_logits" in z.files
        })
        assert prefixes, f"no compatible prediction pair found in {path}"
        n = len(participants)
        return {
            prefix: {
                "prob": _as_seed_matrix(z[prefix + "__calibrated_prob"], n),
                "raw_logits": _as_seed_matrix(z[prefix + "__raw_logits"], n),
            }
            for prefix in prefixes
        }


def arm_name(prefix: str) -> str:
    if prefix.startswith("fusion__"):
        return prefix[len("fusion__"):]
    return prefix.split("__")[-1]


def frozen_comparisons(prefixes: list[str]) -> list[tuple[str, str, str]]:
    by_arm = {arm_name(prefix): prefix for prefix in prefixes}
    candidates = (
        ("correct", "within_label", "individual_pairing_audio_only"),
        ("correct", "raw_ast", "alignment_versus_raw_audio"),
        ("within_label", "global", "label_level_cooccurrence_audio_only"),
        ("m_c", "m_w", "individual_pairing_metadata_fusion"),
        ("m_r_c", "m_r_w", "individual_pairing_raw_preserving"),
        ("m_r", "m", "direct_audio_increment"),
        ("m_c", "m_r", "alignment_versus_raw_fusion"),
        ("m_r_c", "m_r", "alignment_increment_with_raw_preserved"),
    )
    return [(by_arm[a], by_arm[b], label) for a, b, label in candidates
            if a in by_arm and b in by_arm]


def _mean_binary_entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p), EPS, 1 - EPS)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def calibration_in_the_large(z: np.ndarray, y: np.ndarray,
                             max_iter: int = 80) -> float:
    """Fit only an intercept correction while fixing the supplied logit's slope to one."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    intercept = float(logit(np.mean(y)) - np.mean(z))
    for _ in range(max_iter):
        p = sigmoid(z + intercept)
        grad = float(np.sum(p - y))
        hess = float(np.sum(np.maximum(p * (1 - p), 1e-12)))
        step = grad / hess
        intercept -= step
        if abs(step) < 1e-11:
            break
    return float(intercept)


def probability_metrics(y: np.ndarray, source_prob: np.ndarray,
                        raw_logits: np.ndarray | None = None) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(source_prob, dtype=np.float64), EPS, 1 - EPS)
    z = logit(p)
    raw = z if raw_logits is None else np.asarray(raw_logits, dtype=np.float64)
    nll_all = float(logloss(y, p).mean())
    brier = float(np.mean((p - y) ** 2))
    cal = calib_diag(y, p)
    prevalence = float(np.mean(y))
    prior_z = float(logit(prevalence))
    return {
        "n": int(len(y)),
        "prevalence": prevalence,
        "raw_logit_auroc": float(auroc(y, raw)),
        "source_platt_auroc": float(auroc(y, p)),
        "nll": nll_all,
        "excess_nll_vs_log2": nll_all - math.log(2.0),
        "nll_positive": float(logloss(y[y == 1], p[y == 1]).mean()),
        "nll_negative": float(logloss(y[y == 0], p[y == 0]).mean()),
        "brier": brier,
        "brier_skill_vs_0.25": 1.0 - brier / 0.25,
        "calibration_in_the_large": calibration_in_the_large(z, y),
        "calibration_slope": float(cal["slope"]),
        "joint_recalibration_intercept": float(cal["intercept"]),
        "source_platt_logit_sd": float(np.std(z)),
        "predictive_entropy": _mean_binary_entropy(p),
        "prior_centered_mean_abs_logit": float(np.mean(np.abs(z - prior_z))),
    }


def prior_correct(prob: np.ndarray, pi_source: float, pi_target: float = .5) -> np.ndarray:
    shift = float(logit(pi_target) - logit(pi_source))
    return sigmoid(logit(prob) + shift)


def shrink_after_prior(prob: np.ndarray, pi_source: float, lam: float) -> np.ndarray:
    corrected_z = logit(prior_correct(prob, pi_source, .5))
    return sigmoid(float(lam) * corrected_z)


def fit_monotone_platt(z: np.ndarray, y: np.ndarray,
                       max_iter: int = 80) -> tuple[float, float]:
    """Two-parameter logistic recalibration with a numerically nonnegative slope.

    The unconstrained IRLS solution is used when its slope is positive.  If it is
    negative, the constrained optimum is on the slope=0 boundary; a 1e-12 numerical
    lower bound retains strict rank monotonicity for the frozen AUROC invariant.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert len(z) == len(y) and np.unique(y).size == 2
    beta = np.asarray([float(logit(np.mean(y))), 1.0], dtype=np.float64)
    x = np.column_stack([np.ones(len(z)), z])
    for _ in range(max_iter):
        eta = np.clip(x @ beta, -40, 40)
        p = sigmoid(eta)
        w = np.maximum(p * (1 - p), 1e-9)
        grad = x.T @ (p - y)
        hess = (x.T * w) @ x
        hess.flat[::3] += 1e-10
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-9:
            beta = beta_new
            break
        beta = np.clip(beta_new, -100, 100)
    if not np.isfinite(beta).all() or beta[1] <= 0:
        slope = 1e-12
        intercept = float(logit(np.mean(y)))
        # With slope fixed, solve the intercept score equation exactly.
        for _ in range(max_iter):
            p = sigmoid(intercept + slope * z)
            grad = float(np.sum(p - y))
            hess = float(np.sum(np.maximum(p * (1 - p), 1e-12)))
            step = grad / hess
            intercept -= step
            if abs(step) < 1e-10:
                break
        return float(intercept), float(slope)
    return float(beta[0]), float(beta[1])


def apply_monotone_platt(z: np.ndarray, fit: tuple[float, float]) -> np.ndarray:
    intercept, slope = fit
    assert slope > 0, "numerical implementation must be strictly monotone"
    return sigmoid(intercept + slope * np.asarray(z, dtype=np.float64))


def point_target_transport(prob: np.ndarray, cal_mask: np.ndarray, eval_mask: np.ndarray,
                           y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    z = logit(prob)
    out, fits = [], []
    for seed in range(prob.shape[0]):
        fit = fit_monotone_platt(z[seed, cal_mask], y[cal_mask])
        p = apply_monotone_platt(z[seed], fit)
        before = auroc(y[eval_mask], z[seed, eval_mask])
        after = auroc(y[eval_mask], p[eval_mask])
        assert before == after or abs(before - after) < 1e-15, \
            "strictly monotone target calibration changed AUROC"
        out.append(p)
        fits.append({"intercept": fit[0], "slope": fit[1]})
    return np.stack(out), fits


def _metric_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    return (float(auroc(y, a) - auroc(y, b)),
            float(-logloss(y, a).mean() + logloss(y, b).mean()),
            float(np.mean((b - y) ** 2) - np.mean((a - y) ** 2)))


def bootstrap_transport_pair(a_prob: np.ndarray, b_prob: np.ndarray,
                             y: np.ndarray, cal_mask: np.ndarray,
                             eval_mask: np.ndarray, boot: int, seed: int) -> dict:
    """Paired participant x seed bootstrap; refit both calibrators each replicate."""
    za, zb = logit(a_prob), logit(b_prob)
    cal_idx, eval_idx = np.where(cal_mask)[0], np.where(eval_mask)[0]
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(boot):
        ci = rng.choice(cal_idx, len(cal_idx), replace=True)
        ei = rng.choice(eval_idx, len(eval_idx), replace=True)
        if np.unique(y[ci]).size < 2 or np.unique(y[ei]).size < 2:
            continue
        ss = rng.choice(a_prob.shape[0], a_prob.shape[0], replace=True)
        per_seed = []
        for s in ss:
            fa = fit_monotone_platt(za[s, ci], y[ci])
            fb = fit_monotone_platt(zb[s, ci], y[ci])
            pa = apply_monotone_platt(za[s, ei], fa)
            pb = apply_monotone_platt(zb[s, ei], fb)
            per_seed.append(_metric_delta(y[ei], pa, pb))
        values.append(np.mean(per_seed, axis=0))
    assert values, "transport bootstrap produced no valid replicate"
    values = np.asarray(values)
    names = ("delta_auroc", "delta_neg_nll", "delta_neg_brier")
    return {
        name: {"ci": np.percentile(values[:, j], [2.5, 97.5]).tolist()}
        for j, name in enumerate(names)
    } | {"n_valid_bootstrap": int(len(values))}


def paired_point(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                 mask: np.ndarray) -> dict:
    per_seed = [_metric_delta(y[mask], a[s, mask], b[s, mask])
                for s in range(a.shape[0])]
    per_seed = np.asarray(per_seed)
    names = ("delta_auroc", "delta_neg_nll", "delta_neg_brier")
    return {name: {"observed": float(per_seed[:, j].mean()),
                   "per_seed": per_seed[:, j].tolist()}
            for j, name in enumerate(names)}


def self_test() -> None:
    rng = np.random.RandomState(9)
    n = 300
    y = np.tile([0, 1], n // 2)
    z = .7 * (2 * y - 1) + rng.normal(size=n)
    p = sigmoid(1.8 * z + .4)
    pc = prior_correct(p, pi_source=.3, pi_target=.5)
    assert np.isfinite(pc).all()
    assert np.allclose(shrink_after_prior(p, .3, 0), .5)
    fit = fit_monotone_platt(logit(p), y)
    pt = apply_monotone_platt(logit(p), fit)
    assert fit[1] > 0 and abs(auroc(y, p) - auroc(y, pt)) < 1e-15
    masks = np.zeros(n, bool), np.zeros(n, bool)
    masks[0][:150] = True; masks[1][150:] = True
    a = np.repeat(p[None, :], 5, axis=0)
    b = np.repeat(sigmoid(logit(p) * .8)[None, :], 5, axis=0)
    r = bootstrap_transport_pair(a, b, y, masks[0], masks[1], boot=20, seed=4)
    assert r["n_valid_bootstrap"] == 20
    m = probability_metrics(y, p, z)
    assert set(("nll", "brier", "calibration_slope")).issubset(m)
    print("SELF-TEST PASS: prior shift, fixed shrink, monotone transport and paired refit bootstrap")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", default=[], metavar="LABEL=PATH")
    ap.add_argument("--data", default="/mnt/hd/data_heliu/resp_datasets/ukcovid")
    ap.add_argument("--cohort", default="results/ukcovid_audio_cohort.csv")
    ap.add_argument("--out-dir", default="results/probability_transport")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260820)
    ap.add_argument("--evaluate-tests", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.evaluate_tests:
        ap.error("target audit is locked; pass --evaluate-tests explicitly")
    sources = parse_sources(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = {name: out_dir / name for name in ("predictions.npz", "config.json", "results.json")}
    for path in output.values():
        assert not path.exists(), f"refusing to overwrite {path}"

    cohort_args = SimpleNamespace(data=args.data, cohort=args.cohort)
    d, _, val, tests = load_cohort(cohort_args)
    participants = d[UNIT].to_numpy()
    y = d["y"].to_numpy()
    matched, matched_long = tests["matched"], tests["matched_long"]
    assert not np.any(matched & matched_long)
    pi_source = float(np.mean(y[val]))
    assert 0 < pi_source < 1 and np.mean(y[matched]) == .5 \
        and np.mean(y[matched_long]) == .5

    archives = {label: discover_archive(path, participants)
                for label, path in sources.items()}
    config = {
        "protocol": "ICASSP_STRICT_FOLLOWUP_PREREG_ZH.md/E3",
        "script_sha256": file_sha256(Path(__file__)),
        "standing": "read-only post-hoc probability-transport sensitivity",
        "source_prevalence": pi_source,
        "target_prevalence": .5,
        "lambdas_reported_without_selection": list(LAMBDAS),
        "target_transport": (
            "fit nonnegative 1D Platt on matched-long, apply unchanged to participant-disjoint matched"),
        "reverse_sensitivity": "fit matched, apply unchanged to matched-long",
        "target_tuning_for_deployment": False,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.bootstrap_seed,
        "participants_sha16": sha16(participants.astype(str)),
        "split_hashes": {name: sha16(mask) for name, mask in tests.items()} | {"val": sha16(val)},
        "input_file_sha256": {
            **{label: file_sha256(path) for label, path in sources.items()},
            "cohort": file_sha256(Path(args.cohort)),
            "participant_metadata": file_sha256(Path(args.data) / "participant_metadata.csv"),
            "train_test_splits": file_sha256(Path(args.data) / "train_test_splits.csv"),
        },
        "discovered_prefixes": {label: sorted(group) for label, group in archives.items()},
    }
    config["config_sha256"] = canonical_hash(config)

    arrays: dict[str, np.ndarray] = {
        "matched_participants": participants[matched],
        "matched_long_participants": participants[matched_long],
        "seeds": np.asarray(SEEDS, dtype=np.int64),
        "lambdas": np.asarray(LAMBDAS, dtype=np.float64),
        "config_sha256": np.asarray(config["config_sha256"]),
    }
    result: dict = {
        "protocol": "E3 probability-transport audit",
        "config_sha256": config["config_sha256"],
        "source_prevalence": pi_source,
        "target_platt_fits": None,
        "groups": {},
    }

    # Compute and freeze every transformed per-participant prediction before metrics.
    transported: dict[tuple[str, str, str], np.ndarray] = {}
    target_fits: dict = {}
    for group_label, group in archives.items():
        target_fits[group_label] = {}
        for prefix, entry in group.items():
            safe = hashlib.sha256(prefix.encode()).hexdigest()[:12]
            prob = entry["prob"]
            prior = prior_correct(prob, pi_source, .5)
            curve = np.stack([shrink_after_prior(prob, pi_source, lam) for lam in LAMBDAS])
            forward, ffit = point_target_transport(prob, matched_long, matched, y)
            reverse, rfit = point_target_transport(prob, matched, matched_long, y)
            transported[(group_label, prefix, "forward")] = forward
            transported[(group_label, prefix, "reverse")] = reverse
            target_fits[group_label][prefix] = {"forward": ffit, "reverse": rfit}
            base = f"{group_label}__{safe}"
            arrays[base + "__matched__source_prob"] = prob[:, matched]
            arrays[base + "__matched__prior_corrected"] = prior[:, matched]
            arrays[base + "__matched__lambda_curve"] = curve[:, :, matched]
            arrays[base + "__matched__target_transport"] = forward[:, matched]
            arrays[base + "__matched_long__source_prob"] = prob[:, matched_long]
            arrays[base + "__matched_long__prior_corrected"] = prior[:, matched_long]
            arrays[base + "__matched_long__lambda_curve"] = curve[:, :, matched_long]
            arrays[base + "__matched_long__reverse_target_transport"] = reverse[:, matched_long]
    arrays["prefix_map_json"] = np.asarray(json.dumps({
        label: {hashlib.sha256(prefix.encode()).hexdigest()[:12]: prefix for prefix in group}
        for label, group in archives.items()}, sort_keys=True))
    atomic_npz(output["predictions.npz"], **arrays)
    atomic_json(config, output["config.json"])
    result["target_platt_fits"] = target_fits

    # Metrics are read only after all predictions and fits are on disk.
    assert file_sha256(Path(__file__)) == config["script_sha256"], \
        "evaluator changed after its configuration was frozen"
    for group_label, group in archives.items():
        gout = {"arms": {}, "comparisons": {}}
        result["groups"][group_label] = gout
        for prefix, entry in group.items():
            prob, raw = entry["prob"], entry["raw_logits"]
            prior = prior_correct(prob, pi_source, .5)
            forward = transported[(group_label, prefix, "forward")]
            reverse = transported[(group_label, prefix, "reverse")]
            arm_out = {"splits": {}, "shrink_curve": {}}
            gout["arms"][prefix] = arm_out
            for split_name, mask in tests.items():
                arm_out["splits"][split_name] = {
                    "per_seed": [probability_metrics(y[mask], prob[s, mask], raw[s, mask])
                                 for s in range(len(SEEDS))]
                }
            for split_name, mask in (("matched", matched), ("matched_long", matched_long)):
                arm_out["splits"][split_name]["prior_corrected_per_seed"] = [
                    probability_metrics(y[mask], prior[s, mask], raw[s, mask])
                    for s in range(len(SEEDS))]
                arm_out["shrink_curve"][split_name] = {
                    str(lam): [probability_metrics(
                        y[mask], shrink_after_prior(prob[s, mask], pi_source, lam),
                        raw[s, mask]) for s in range(len(SEEDS))]
                    for lam in LAMBDAS
                }
            arm_out["target_transport"] = {
                "matched_long_to_matched_per_seed": [
                    probability_metrics(y[matched], forward[s, matched], raw[s, matched])
                    for s in range(len(SEEDS))],
                "matched_to_matched_long_per_seed": [
                    probability_metrics(y[matched_long], reverse[s, matched_long],
                                        raw[s, matched_long])
                    for s in range(len(SEEDS))],
            }

        for a, b, label in frozen_comparisons(list(group)):
            key = f"{arm_name(a)}_minus_{arm_name(b)}"
            aa, bb = group[a]["prob"], group[b]["prob"]
            fa = transported[(group_label, a, "forward")]
            fb = transported[(group_label, b, "forward")]
            ra = transported[(group_label, a, "reverse")]
            rb = transported[(group_label, b, "reverse")]
            comp = {"arms": [a, b], "interpretation": label}
            comp["source_platt_matched"] = paired_point(aa, bb, y, matched)
            comp["prior_corrected_matched"] = paired_point(
                prior_correct(aa, pi_source), prior_correct(bb, pi_source), y, matched)
            comp["shrink_curve_matched"] = {
                str(lam): paired_point(shrink_after_prior(aa, pi_source, lam),
                                       shrink_after_prior(bb, pi_source, lam), y, matched)
                for lam in LAMBDAS
            }
            comp["matched_long_to_matched"] = paired_point(fa, fb, y, matched)
            forward_boot = bootstrap_transport_pair(
                aa, bb, y, matched_long, matched, args.bootstrap,
                args.bootstrap_seed + len(gout["comparisons"]))
            for metric in ("delta_auroc", "delta_neg_nll", "delta_neg_brier"):
                comp["matched_long_to_matched"][metric]["ci"] = forward_boot[metric]["ci"]
            comp["matched_long_to_matched"]["n_valid_bootstrap"] = \
                forward_boot["n_valid_bootstrap"]
            comp["matched_to_matched_long"] = paired_point(ra, rb, y, matched_long)
            reverse_boot = bootstrap_transport_pair(
                aa, bb, y, matched, matched_long, args.bootstrap,
                args.bootstrap_seed + 100 + len(gout["comparisons"]))
            for metric in ("delta_auroc", "delta_neg_nll", "delta_neg_brier"):
                comp["matched_to_matched_long"][metric]["ci"] = reverse_boot[metric]["ci"]
            comp["matched_to_matched_long"]["n_valid_bootstrap"] = \
                reverse_boot["n_valid_bootstrap"]
            gout["comparisons"][key] = comp

    result["predictions_sha256"] = file_sha256(output["predictions.npz"])
    atomic_json(result, output["results.json"])
    print(f"wrote frozen E3 predictions/config/results to {out_dir}")


if __name__ == "__main__":
    main()
