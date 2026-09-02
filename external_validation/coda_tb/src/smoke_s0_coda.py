"""Pure projector/loss correctness gate; generates no CODA scientific result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from projector import (ContrastiveProjectionHead, D_LLM, SOURCE_COMMIT, TEMPERATURE,
                       build_pairing, contrastive_loss, make_optimizer)


def hand_infonce(audio: np.ndarray, text: np.ndarray) -> float:
    audio = audio / np.linalg.norm(audio, axis=1, keepdims=True)
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    score = audio @ text.T / TEMPERATURE
    score -= score.max(axis=1, keepdims=True)
    return float(np.mean(-(np.diag(score) - np.log(np.exp(score).sum(axis=1)))))


def run_arm(audio: torch.Tensor, text: torch.Tensor, labels: np.ndarray,
            arm: str, steps: int = 400) -> dict:
    pairing = build_pairing(labels, {"within_label": "within"}.get(arm, arm), 0)
    paired = text[torch.tensor(pairing)]
    input_hash = hashlib.sha256(audio.numpy().tobytes() + text.numpy().tobytes()).hexdigest()
    torch.manual_seed(0)
    model = ContrastiveProjectionHead(); optimiser = make_optimizer(model)
    trace = []
    for step in range(steps):
        loss = contrastive_loss(model(audio), paired)
        optimiser.zero_grad(); loss.backward(); optimiser.step()
        if step < 20:
            trace.append(round(float(loss.detach()), 8))
    model.eval()
    with torch.no_grad():
        projection = model(audio)
        framework = float(contrastive_loss(projection, paired))
        manual = hand_infonce(projection.numpy(), paired.numpy())
        score = torch.nn.functional.normalize(projection, dim=1) @ \
            torch.nn.functional.normalize(paired, dim=1).T
        recall = int((score.argmax(1) == torch.arange(len(audio))).sum())

    torch.manual_seed(0)
    replay_model = ContrastiveProjectionHead(); replay_opt = make_optimizer(replay_model)
    replay = []
    for _ in range(20):
        replay_loss = contrastive_loss(replay_model(audio), paired)
        replay_opt.zero_grad(); replay_loss.backward(); replay_opt.step()
        replay.append(round(float(replay_loss.detach()), 8))
    output_hash = hashlib.sha256(audio.numpy().tobytes() + text.numpy().tobytes()).hexdigest()
    return {
        "pairing_legal": bool(np.array_equal(np.sort(pairing), np.arange(len(labels)))),
        "no_fixed_points_if_shuffled": bool(arm == "correct" or
                                             not np.any(pairing == np.arange(len(labels)))),
        "within_label_respected": bool(arm != "within_label" or
                                        np.all(labels[pairing] == labels)),
        "loss_matches_hand": abs(framework - manual) < 1e-4,
        "recall_at_1": recall, "memorised": recall == len(labels),
        "out_dim": int(projection.shape[1]), "finite": bool(torch.isfinite(projection).all()),
        "nonzero": bool(projection.abs().sum()), "nonnegative": bool((projection >= 0).all()),
        "inputs_unchanged": input_hash == output_hash,
        "first20_reproducible": trace == replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve(); config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    if not root.is_absolute(): root = (config_path.parent / root).resolve()
    rng = np.random.RandomState(20260903)
    audio = torch.tensor(rng.normal(size=(16, 768)).astype(np.float32))
    text = torch.tensor(rng.normal(size=(16, 2560)).astype(np.float32))
    labels = np.repeat([0, 1], 8)
    results = {arm: run_arm(audio, text, labels, arm)
               for arm in ("correct", "within_label", "global")}
    failures = [f"{arm}:{key}" for arm, values in results.items()
                for key, value in values.items()
                if key != "recall_at_1" and (value is False or
                   (key == "out_dim" and value != D_LLM))]
    output = {"passed": not failures, "failures": failures,
              "source_commit": SOURCE_COMMIT, "synthetic_only": True, "arms": results}
    path = root / "public" / "smoke_s0.json"; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"S0 {'PASS' if not failures else 'FAIL'}: {path}")
    if failures: raise SystemExit(3)


if __name__ == "__main__":
    main()
