"""Record a path-free execution environment summary for the returned aggregate bundle."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import subprocess
import sys

from common import REPO_ROOT, atomic_json, load_config, output_paths


PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "soundfile", "librosa",
            "torch", "transformers", "openpyxl")


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True,
                                timeout=15)
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def execute(config_file: str) -> None:
    config, config_path = load_config(config_file)
    paths = output_paths(config, config_path)
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git_commit = command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    gpu = command_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ])
    atomic_json(paths["public"] / "environment.json", {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": versions, "gpu": gpu, "repository_commit": git_commit,
        "privacy": "path-free aggregate execution metadata",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute(args.config)


if __name__ == "__main__":
    main()
