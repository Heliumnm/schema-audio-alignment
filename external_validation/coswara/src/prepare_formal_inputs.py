"""Build the private Coswara model table from the frozen secondary-GO inputs.

This program is model blind.  It refuses any changed cohort, pair or QC input and
never imports an audio encoder or reads a prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
ENGINE = REPO / "external_validation" / "cambridge_covid_sounds" / "src"
sys.path.insert(0, str(ENGINE)); sys.path.insert(0, str(REPO / "src"))

from common import atomic_json, load_config, output_paths, resolve_path, sha256_file
from freeze_coswara_external_split import prepare_rows


EXPECTED = {
    "train": {0: 669, 1: 372},
    "validation": {0: 141, 1: 76},
    "source_test": {0: 152, 1: 91},
    "external_evaluation": {0: 100, 1: 100},
}
FIELDS = (
    "age", "sex", "cough", "fever", "fatigue", "sore_throat",
    "breathing_difficulty", "asthma", "other_respiratory", "smoker",
    "diarrhoea", "loss_of_smell", "vaccination", "mask_use",
)


def safe(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "[MISSING]"
    text = str(value).strip()
    if text.casefold() == "[missing]":
        return "[MISSING]"
    if any(char in text for char in "[]\n\r\x00"):
        raise ValueError(f"unsafe schema value: {text!r}")
    return text


def schema(row: dict[str, object]) -> str:
    return " ".join(f"[{field.upper()}={safe(row[field])}]" for field in FIELDS)


def execute(config_file: str) -> Path:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    inputs = config["inputs"]
    source = {name: resolve_path(config_path, inputs[name])
              for name in ("participant_manifest", "matched_pairs", "audio_qc")}
    for name, path in source.items():
        if path is None or sha256_file(path) != inputs["expected_sha256"][name]:
            raise RuntimeError(f"frozen {name} SHA-256 mismatch")

    frozen = pd.read_csv(source["participant_manifest"])
    pairs = pd.read_csv(source["matched_pairs"])
    prepared, _ = prepare_rows(source["audio_qc"])
    by_id = {str(row["participant_id"]): row for row in prepared}
    if len(frozen) != 1701 or frozen.participant_id.nunique() != 1701:
        raise RuntimeError("secondary-GO manifest must contain 1,701 unique participants")
    observed = frozen.groupby(["split", "label"]).size().to_dict()
    expected = {(split, label): count for split, counts in EXPECTED.items()
                for label, count in counts.items()}
    if observed != expected:
        raise RuntimeError(f"frozen split counts changed: {observed}")
    if len(pairs) != 100 or pairs.pair_id.nunique() != 100:
        raise RuntimeError("matched target must contain 100 unique pairs")

    pair_id = {}
    for item in pairs.itertuples(index=False):
        for participant, label in ((str(item.negative_id), 0), (str(item.positive_id), 1)):
            if participant in pair_id:
                raise RuntimeError(f"repeated target participant: {participant}")
            row = frozen.loc[frozen.participant_id.astype(str) == participant]
            if len(row) != 1 or int(row.iloc[0].label) != label or \
                    row.iloc[0]["split"] != "external_evaluation":
                raise RuntimeError(f"pair membership mismatch: {participant}")
            pair_id[participant] = str(item.pair_id)
    target = set(frozen.loc[frozen["split"] == "external_evaluation",
                            "participant_id"].astype(str))
    if set(pair_id) != target:
        raise RuntimeError("pair file does not exactly cover external evaluation")

    audio_root = resolve_path(config_path, inputs["audio_root"])
    rows = []
    for item in frozen.itertuples(index=False):
        pid = str(item.participant_id)
        if pid not in by_id:
            raise RuntimeError(f"participant absent after frozen preparation: {pid}")
        raw = dict(by_id[pid])
        if str(raw["relative_path"]) != str(item.relative_path):
            raise RuntimeError(f"audio path changed for {pid}")
        path = (audio_root / str(item.relative_path)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        split = "matched_target" if item.split == "external_evaluation" else \
                ("test" if item.split == "source_test" else str(item.split))
        row = dict(raw)
        row.update({
            "participant_identifier": pid,
            "y": int(item.label),
            "splits": split,
            "in_matched_test": split == "matched_target",
            "pair_id": pair_id.get(pid, ""),
            "exact_stratum": str(item.exact_stratum),
            "audio_files_json": json.dumps([str(path)], separators=(",", ":")),
        })
        row["text"] = schema(row)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.text.nunique() != 1156:
        raise RuntimeError(f"profile count changed: {table.text.nunique()} != 1156")

    paths["private"].mkdir(parents=True, exist_ok=True)
    paths["public"].mkdir(parents=True, exist_ok=True)
    output = paths["private"] / "participant_manifest.csv"
    temporary = output.with_suffix(".csv.tmp")
    table.to_csv(temporary, index=False)
    if output.exists():
        if sha256_file(output) != sha256_file(temporary):
            temporary.unlink(); raise RuntimeError(f"refusing to overwrite {output}")
        temporary.unlink()
    else:
        os.replace(temporary, output)
    val = table.splits.eq("validation")
    gate = {
        "verdict": "GO",
        "protocol": "coswara-global-secondary-go-stress-v1",
        "standing": config["protocol"]["standing"],
        "original_greedy_gate": "NO-GO",
        "model_outputs_read": False,
        "n_participants": 1701,
        "n_matched_pairs": 100,
        "split_counts": EXPECTED,
        "profiles": {"all_unique": 1156, "validation_n": int(val.sum()),
                     "validation_unique": int(table.loc[val, "text"].nunique())},
        "source_sha256": {name: sha256_file(path) for name, path in source.items()},
        "formal_manifest_sha256": sha256_file(output),
    }
    atomic_json(paths["public"] / "data_gate.json", gate)
    print("COSWARA FORMAL INPUT PASS: 1,701 participants; secondary stress test only")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    execute(parser.parse_args().config)
