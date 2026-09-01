"""Shared, privacy-conscious utilities for the Cambridge external audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_bytes: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha16_array(array) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text())
    if config.get("format_version") != "cambridge-external-v1":
        raise ValueError("config format_version must be 'cambridge-external-v1'")
    return config, config_path


def resolve_path(config_path: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def output_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    root = resolve_path(config_path, config["output_root"])
    assert root is not None
    return {
        "root": root,
        "private": root / "private",
        "public": root / "public",
        "logs": root / "logs",
        "models": root / "models",
    }


def canonical_string(value: Any) -> str:
    if value is None:
        return "[MISSING]"
    try:
        import pandas as pd

        if pd.isna(value):
            return "[MISSING]"
    except (ImportError, TypeError):
        pass
    text = str(value).strip()
    return text if text else "[MISSING]"


def safe_identifier(value: Any) -> str:
    text = canonical_string(value)
    if text == "[MISSING]" or "\n" in text or "\r" in text or "\x00" in text:
        raise ValueError("blank or unsafe participant identifier")
    return text


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a path-free configuration summary safe for aggregate result bundles."""

    return {
        "format_version": config["format_version"],
        "dataset": config.get("dataset"),
        "source_adapter": {
            "name": config.get("inputs", {}).get("source_adapter", {}).get("name"),
            "metadata_glob": config.get("inputs", {}).get("source_adapter", {}).get(
                "metadata_glob"),
        } if config.get("inputs", {}).get("source_adapter") else None,
        "protocol": config.get("protocol", {}),
        "labels": config.get("labels", {}),
        "splits": config.get("splits", {}),
        "field_rules": config.get("field_rules", {}),
        "matching": config.get("matching", {}),
        "models": {
            key: {k: v for k, v in value.items() if "path" not in k and "root" not in k}
            for key, value in config.get("models", {}).items()
        },
    }
