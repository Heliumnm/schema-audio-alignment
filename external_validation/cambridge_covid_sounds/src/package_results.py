"""Create a path-free shareable archive containing public aggregate outputs only."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from common import atomic_json, load_config, output_paths, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    paths = output_paths(config, config_path)
    public = paths["public"]
    files = sorted(path for path in public.iterdir()
                   if path.is_file() and path.name != "PUBLIC_RESULTS.zip")
    manifest = {path.name: sha256_file(path) for path in files}
    atomic_json(public / "PUBLIC_FILE_HASHES.json", manifest)
    files = sorted(path for path in public.iterdir()
                   if path.is_file() and path.name != "PUBLIC_RESULTS.zip")
    archive = public / "PUBLIC_RESULTS.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in files:
            output.write(path, arcname=path.name)
    print(f"share only {archive} (sha256 {sha256_file(archive)})")


if __name__ == "__main__":
    main()
