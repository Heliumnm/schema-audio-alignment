"""Create a local gate configuration from four Cambridge release paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2-csv", required=True)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = {
        "task2_csv": Path(args.task2_csv).expanduser().resolve(),
        "metadata_root": Path(args.metadata_root).expanduser().resolve(),
        "audio_root": Path(args.audio_root).expanduser().resolve(),
        "output_root": Path(args.output_root).expanduser().resolve(),
        "write": Path(args.write).expanduser().resolve(),
    }
    if not paths["task2_csv"].is_file():
        raise SystemExit(f"Task-2 CSV not found: {paths['task2_csv']}")
    if not paths["metadata_root"].is_dir():
        raise SystemExit(f"three-platform metadata directory not found: {paths['metadata_root']}")
    if not paths["audio_root"].is_dir():
        raise SystemExit(f"Task-2 audio directory not found: {paths['audio_root']}")
    if paths["write"].exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing config: {paths['write']}")

    template = json.loads((PACKAGE_ROOT / "config.task2_raw.example.json").read_text())
    template["inputs"]["participant_csv"] = str(paths["task2_csv"])
    template["inputs"]["source_adapter"]["metadata_root"] = str(paths["metadata_root"])
    template["inputs"]["audio_root"] = str(paths["audio_root"])
    template["output_root"] = str(paths["output_root"])
    paths["write"].parent.mkdir(parents=True, exist_ok=True)
    paths["write"].write_text(json.dumps(template, indent=2) + "\n")
    print(f"wrote local gate config: {paths['write']}")


if __name__ == "__main__":
    main()
