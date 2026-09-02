"""Verify the frozen CODA v2 manifests and build the private formal-model table.

No encoder or model is imported here.  The output adapts the controlled CODA manifest to
the already-audited external execution engine without changing participant membership,
split, label, pairing, or metadata values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pandas as pd


EXPECTED_COUNTS = {
    "train": {0: 476, 1: 127},
    "validation": {0: 98, 1: 24},
    "source_test": {0: 116, 1: 40},
    "matched_target": {0: 100, 1: 100},
}
PROFILE_FIELDS = (
    "age", "sex", "height", "weight", "reported_cough_dur", "tb_prior",
    "tb_prior_pul", "tb_prior_extrapul", "tb_prior_unknown", "hemoptysis",
    "heart_rate", "temperature", "weight_loss", "smoke_lweek", "fever",
    "night_sweats",
)
NUMERIC_FIELDS = {
    "age", "height", "weight", "reported_cough_dur", "heart_rate", "temperature"
}
TAG = {field: field.upper() for field in PROFILE_FIELDS}


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def canonical(field: str, value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "[MISSING]"
    if field in NUMERIC_FIELDS:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite {field}: {value!r}")
        return format(parsed, ".10g")
    text = str(value).strip()
    if any(character in text for character in "[]\n\r\x00"):
        raise ValueError(f"unsafe categorical value in {field}: {text!r}")
    return text


def schema_text(row: pd.Series) -> str:
    return " ".join(f"[{TAG[field]}={canonical(field, row[field])}]"
                    for field in PROFILE_FIELDS)


def verify_counts(table: pd.DataFrame) -> None:
    observed = table.groupby(["split", "label"]).size().to_dict()
    expected = {(split, label): count for split, values in EXPECTED_COUNTS.items()
                for label, count in values.items()}
    if observed != expected:
        raise RuntimeError(f"frozen split counts changed: {observed} != {expected}")


def execute(config_file: str) -> Path:
    config_path = Path(config_file).expanduser().resolve()
    config = json.loads(config_path.read_text())
    inputs = config["inputs"]
    source = {
        key: resolve(config_path, inputs[key])
        for key in ("participant_manifest", "matched_pairs", "audio_qc")
    }
    for key, path in source.items():
        actual = sha256_file(path)
        expected = inputs["expected_sha256"][key]
        if actual != expected:
            raise RuntimeError(f"{key} SHA-256 mismatch: {actual} != {expected}")

    table = pd.read_csv(source["participant_manifest"])
    pairs = pd.read_csv(source["matched_pairs"])
    qc = pd.read_csv(source["audio_qc"])
    required = {"participant", "label", "country", "sex", "split", "audio_count",
                "audio_files", *PROFILE_FIELDS}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError(f"participant manifest missing columns: {sorted(missing)}")
    if len(table) != 1081 or table.participant.nunique() != 1081:
        raise RuntimeError("frozen participant cohort must contain 1,081 unique people")
    if set(table.label) != {0, 1}:
        raise RuntimeError("TB label must be binary 0/1")
    verify_counts(table)
    if len(qc) != 9772 or not qc.qc_pass.astype(bool).all():
        raise RuntimeError("frozen audio QC must contain 9,772 passing recordings")
    if len(pairs) != 100 or pairs.pair_id.nunique() != 100:
        raise RuntimeError("frozen matched target must contain exactly 100 pairs")

    pair_id: dict[str, str] = {}
    for row in pairs.itertuples(index=False):
        for participant, label in ((str(row.negative_id), 0), (str(row.positive_id), 1)):
            if participant in pair_id:
                raise RuntimeError(f"matched participant repeated: {participant}")
            actual = table.loc[table.participant.astype(str) == participant]
            if len(actual) != 1 or int(actual.iloc[0].label) != label or \
                    str(actual.iloc[0].split) != "matched_target":
                raise RuntimeError(f"pair membership/label mismatch: {participant}")
            pair_id[participant] = str(row.pair_id)
    target = set(table.loc[table.split == "matched_target", "participant"].astype(str))
    if set(pair_id) != target or len(target) != 200:
        raise RuntimeError("pair manifest does not cover the frozen target exactly")

    qc_stats = qc.groupby("participant").duration_seconds.agg(
        audio_total_duration_s="sum", audio_min_duration_s="min",
        audio_max_duration_s="max")
    audio_root = resolve(config_path, inputs["audio_root"])
    rows, all_audio = [], []
    for item in table.itertuples(index=False):
        values = item._asdict()
        filenames = str(values["audio_files"]).split("|")
        if len(filenames) != int(values["audio_count"]):
            raise RuntimeError(f"audio count mismatch: {values['participant']}")
        paths = [str((audio_root / filename).resolve()) for filename in filenames]
        all_audio.extend(paths)
        row = dict(values)
        row.update({
            "participant_identifier": str(values["participant"]),
            "y": int(values["label"]),
            "splits": "test" if values["split"] == "source_test" else values["split"],
            "in_matched_test": values["split"] == "matched_target",
            "pair_id": pair_id.get(str(values["participant"]), ""),
            "audio_files_json": json.dumps(paths, separators=(",", ":")),
            "audio_total_duration_s": float(qc_stats.loc[values["participant"],
                                                           "audio_total_duration_s"]),
            "audio_min_duration_s": float(qc_stats.loc[values["participant"],
                                                         "audio_min_duration_s"]),
            "audio_max_duration_s": float(qc_stats.loc[values["participant"],
                                                         "audio_max_duration_s"]),
        })
        row["text"] = schema_text(pd.Series(row))
        rows.append(row)

    if len(all_audio) != 9749 or len(set(all_audio)) != 9749:
        raise RuntimeError("formal eligible-cohort audio list is not the frozen 9,749-file bijection")
    missing_audio = [path for path in all_audio if not Path(path).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"{len(missing_audio)} frozen audio files absent; first={missing_audio[0]}")
    formal = pd.DataFrame(rows)
    if formal.text.nunique() != 1081:
        raise RuntimeError("expected all 1,081 frozen CODA profiles to be unique")

    output_root = resolve(config_path, config["output_root"])
    private, public = output_root / "private", output_root / "public"
    private.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    output = private / "participant_manifest.csv"
    temporary = output.with_suffix(".csv.tmp")
    formal.to_csv(temporary, index=False)
    if output.exists():
        if sha256_file(output) != sha256_file(temporary):
            temporary.unlink()
            raise RuntimeError(f"refusing to overwrite incompatible {output}")
        temporary.unlink()
    else:
        os.replace(temporary, output)

    gate = {
        "verdict": "GO", "protocol": "coda-match-first-v2-formal-v1",
        "n_participants": 1081, "n_audio_formal_cohort": 9749,
        "n_audio_qc_total": 9772, "n_audio_excluded_with_ineligible_participants": 23,
        "n_matched_pairs": 100,
        "split_counts": EXPECTED_COUNTS,
        "source_sha256": {key: sha256_file(path) for key, path in source.items()},
        "formal_manifest_sha256": sha256_file(output),
        "model_outputs_read": False,
    }
    gate_path = public / "data_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(f"FORMAL INPUT PASS: {len(formal)} participants, {len(all_audio)} recordings")
    print(f"private manifest: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
