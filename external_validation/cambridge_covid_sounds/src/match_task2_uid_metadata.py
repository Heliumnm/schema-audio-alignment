#!/usr/bin/env python3
"""Privately join a Task-2 UID allow-list to the released Cambridge metadata.

Participant-level rows remain below OUTPUT/private.  OUTPUT/public contains aggregate counts
only and is safe to review or share.  This script does not assign a train/test split because a
UID alone cannot identify the Task-2 collection session for repeatedly surveyed participants.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


POSITIVE_VALUES = {"positiveLast14", "last14"}
NEGATIVE_VALUES = {"negativeNever"}


def read_uids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def strict_label(value: str) -> str:
    value = value.strip()
    if value in POSITIVE_VALUES:
        return "positive"
    if value in NEGATIVE_VALUES:
        return "negative"
    return ""


def is_english(value: str) -> bool:
    return value.strip().casefold() in {"en", "english"}


def read_metadata(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    for path in sorted(root.glob("**/*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if not reader.fieldnames or "Uid" not in reader.fieldnames:
                continue
            local_columns = [name for name in reader.fieldnames if name]
            for name in local_columns:
                if name not in columns:
                    columns.append(name)
            for row in reader:
                cleaned = {name: (row.get(name) or "").strip() for name in local_columns}
                cleaned["platform"] = path.stem
                cleaned["metadata_source"] = str(path.relative_to(root))
                rows.append(cleaned)
    if not rows:
        raise ValueError("no semicolon-delimited Cambridge metadata CSV with a Uid column")
    return rows, columns


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid-list", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    raw_uids = read_uids(args.uid_list)
    uid_set = set(raw_uids)
    metadata, metadata_columns = read_metadata(args.metadata_root)
    matched = [row for row in metadata if row.get("Uid", "") in uid_set]
    by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matched:
        by_uid[row["Uid"]].append(row)

    strict_rows: list[dict[str, object]] = []
    participant_rows: list[dict[str, object]] = []
    for uid in sorted(uid_set):
        rows = by_uid.get(uid, [])
        eligible = []
        for row in rows:
            label = strict_label(row.get("Covid-Tested", ""))
            if is_english(row.get("Language", "")) and label:
                item: dict[str, object] = dict(row)
                item["strict_label"] = label
                eligible.append(item)
                strict_rows.append(item)
        labels = sorted({str(row["strict_label"]) for row in eligible})
        participant_rows.append({
            "uid": uid,
            "found_in_metadata": bool(rows),
            "metadata_rows": len(rows),
            "strict_english_rows": len(eligible),
            "strict_labels": "|".join(labels),
            "label_conflict": len(labels) > 1,
            "safe_participant_label": labels[0] if len(labels) == 1 else "",
            "strict_collection_folders": len({
                str(row.get("Folder Name", "")) for row in eligible
                if str(row.get("Folder Name", ""))
            }),
        })

    private = args.output_root / "private"
    public = args.output_root / "public"
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    extra = ["platform", "metadata_source"]
    write_csv(private / "task2_uid_metadata_all_rows.csv", matched,
              metadata_columns + extra)
    write_csv(private / "task2_uid_strict_english_rows.csv", strict_rows,
              metadata_columns + extra + ["strict_label"])
    write_csv(private / "task2_uid_participant_status.csv", participant_rows,
              ["uid", "found_in_metadata", "metadata_rows", "strict_english_rows",
               "strict_labels", "label_conflict", "safe_participant_label",
               "strict_collection_folders"])

    safe = [row for row in participant_rows if row["safe_participant_label"]]
    report = {
        "model_outputs_read": False,
        "split_assigned": False,
        "uid_list": {
            "nonempty_lines": len(raw_uids),
            "unique_values": len(uid_set),
            "duplicates": len(raw_uids) - len(uid_set),
            "found_in_metadata": len(by_uid),
            "not_found_in_metadata": len(uid_set - set(by_uid)),
        },
        "metadata": {
            "total_rows": len(metadata),
            "matched_rows": len(matched),
            "uids_with_multiple_metadata_rows": sum(len(rows) > 1 for rows in by_uid.values()),
            "matched_rows_by_platform": dict(sorted(Counter(
                row["platform"] for row in matched).items())),
        },
        "strict_english": {
            "rows": len(strict_rows),
            "uids": sum(bool(row["strict_english_rows"]) for row in participant_rows),
            "participant_label_conflicts": sum(bool(row["label_conflict"])
                                               for row in participant_rows),
            "safe_participant_labels": len(safe),
            "negative": sum(row["safe_participant_label"] == "negative" for row in safe),
            "positive": sum(row["safe_participant_label"] == "positive" for row in safe),
            "uids_with_multiple_strict_collection_folders": sum(
                int(row["strict_collection_folders"]) > 1 for row in participant_rows),
        },
        "interpretation": (
            "UID membership is linked, but UID-only input does not identify the Task-2 "
            "collection session for repeated participants. UID plus Folder Name (or an "
            "audio-root session linkage) is required before freezing labels and splits."
        ),
    }
    (public / "task2_uid_metadata_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    s = report["strict_english"]
    u = report["uid_list"]
    lines = [
        "# Cambridge Task-2 UID ↔ metadata audit", "",
        "No model output was read and no split was assigned.", "",
        f"- UID-list values: {u['unique_values']}",
        f"- found in metadata: {u['found_in_metadata']}",
        f"- not found: {u['not_found_in_metadata']}",
        f"- strict-English UID: {s['uids']}",
        f"- participant label conflicts: {s['participant_label_conflicts']}",
        f"- safe UID-level labels after conflict exclusion: {s['safe_participant_labels']} "
        f"({s['negative']} negative / {s['positive']} positive)", "",
        "UID-only membership is insufficient to choose the correct collection session for "
        "repeated participants. Obtain UID + Folder Name or perform exact session linkage "
        "against the Task-2 audio tree before matching or splitting.",
    ]
    (public / "TASK2_UID_METADATA_AUDIT.md").write_text("\n".join(lines) + "\n",
                                                        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
