#!/usr/bin/env python3
"""Safely stream only ``cough-heavy.wav`` from the pinned Coswara archive.

The Zenodo ZIP contains 43 date directories.  Each date's gzip-compressed tar stream is
split across several ZIP members (``DATE.tar.gz.aa``, ``.ab``, ...), and all nine audio
modalities are interleaved inside it.  This extractor therefore reads the full verified
archive but materialises only one prespecified modality.

Safety properties:

* the outer archive must match the frozen byte count and MD5;
* the frozen root, date and split-part structure is checked before extraction;
* split parts are ordered explicitly and must have contiguous suffixes from ``aa``;
* tar paths are validated and files are copied manually, never via ``extractall``;
* every date is written to a private staging directory and atomically promoted;
* a completed date is skipped only if its provenance record matches the archive.

No metadata label is loaded and no model score is computed here.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import string
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ARCHIVE_NAME = "Coswara-Data-dataset-paper-publication.zip"
ARCHIVE_BYTES = 12_984_309_908
ARCHIVE_MD5 = "53721d9c106f99872bf7f878c8196d31"
ARCHIVE_ROOT = "iiscleap-Coswara-Data-bf300ae"
EXPECTED_ENTRIES = 418
EXPECTED_DATES = 43
EXPECTED_PARTS = 153
EXTRACTION_PROTOCOL_VERSION = 2
TARGET_NAME = "cough-heavy.wav"
PART_RE = re.compile(
    rf"^{re.escape(ARCHIVE_ROOT)}/(?P<date>[0-9]{{8}})/"
    r"(?P=date)\.tar\.gz\.(?P<suffix>[a-z]{2})$"
)
PARTICIPANT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def digest(path: Path, algorithm: str, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def suffix_sequence(n: int) -> list[str]:
    values = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
    if n > len(values):
        raise ValueError(f"too many split parts for two-letter suffixes: {n}")
    return values[:n]


def verify_archive(path: Path) -> tuple[zipfile.ZipFile, dict[str, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != ARCHIVE_NAME:
        raise ValueError(f"archive name {path.name!r}, expected {ARCHIVE_NAME!r}")
    actual_bytes = path.stat().st_size
    if actual_bytes != ARCHIVE_BYTES:
        raise ValueError(f"archive bytes {actual_bytes}, expected {ARCHIVE_BYTES}")
    actual_md5 = digest(path, "md5")
    if actual_md5 != ARCHIVE_MD5:
        raise ValueError(f"archive MD5 {actual_md5}, expected {ARCHIVE_MD5}")

    archive = zipfile.ZipFile(path)
    infos = archive.infolist()
    if len(infos) != EXPECTED_ENTRIES:
        archive.close()
        raise ValueError(f"ZIP entries {len(infos)}, expected {EXPECTED_ENTRIES}")

    parts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    outside_root: list[str] = []
    for info in infos:
        if not info.filename.startswith(f"{ARCHIVE_ROOT}/"):
            outside_root.append(info.filename)
        match = PART_RE.fullmatch(info.filename)
        if match:
            parts[match.group("date")].append((match.group("suffix"), info.filename))
    if outside_root:
        archive.close()
        raise ValueError(f"{len(outside_root)} entries lie outside the pinned root")
    if len(parts) != EXPECTED_DATES:
        archive.close()
        raise ValueError(f"date streams {len(parts)}, expected {EXPECTED_DATES}")
    if sum(len(values) for values in parts.values()) != EXPECTED_PARTS:
        archive.close()
        raise ValueError(
            f"split parts {sum(len(v) for v in parts.values())}, expected {EXPECTED_PARTS}"
        )

    ordered: dict[str, list[str]] = {}
    for date, values in sorted(parts.items()):
        values.sort()
        observed = [suffix for suffix, _ in values]
        expected = suffix_sequence(len(values))
        if observed != expected:
            archive.close()
            raise ValueError(
                f"non-contiguous split sequence for {date}: {observed}, expected {expected}"
            )
        ordered[date] = [name for _, name in values]
    return archive, ordered


class ConcatenatedZipMembers(io.RawIOBase):
    """A non-seekable reader that joins several ZIP members without buffering them."""

    def __init__(self, archive: zipfile.ZipFile, members: list[str]):
        super().__init__()
        self.archive = archive
        self.members = iter(members)
        self.current: zipfile.ZipExtFile | None = None

    def readable(self) -> bool:
        return True

    def _open_next(self) -> bool:
        if self.current is not None:
            self.current.close()
        try:
            name = next(self.members)
        except StopIteration:
            self.current = None
            return False
        self.current = self.archive.open(name, "r")
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer)
        total = 0
        while total < len(view):
            if self.current is None and not self._open_next():
                break
            assert self.current is not None
            n_read = self.current.readinto(view[total:])
            if n_read:
                total += n_read
            else:
                self.current.close()
                self.current = None
        return total

    def close(self) -> None:
        if self.current is not None:
            self.current.close()
            self.current = None
        super().close()


def validate_tar_path(name: str, date: str) -> tuple[PurePosixPath, str | None]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe tar path: {name!r}")
    participant = None
    if path.name == TARGET_NAME:
        if len(path.parts) != 3 or path.parts[0] != date:
            raise ValueError(
                f"target path must be DATE/PARTICIPANT/{TARGET_NAME}: {name!r}"
            )
        participant = path.parts[1]
        if not PARTICIPANT_RE.fullmatch(participant):
            raise ValueError(f"invalid participant ID in tar path: {name!r}")
    return path, participant


def completion_payload(
    archive_path: Path,
    stage: Path,
    date: str,
    members: list[str],
    participants: list[str],
    script_sha256: str,
) -> dict[str, object]:
    return {
        "format_version": EXTRACTION_PROTOCOL_VERSION,
        "date": date,
        "archive_name": archive_path.name,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_md5": ARCHIVE_MD5,
        "archive_root": ARCHIVE_ROOT,
        "zip_members": members,
        "n_cough_heavy": len(participants),
        "participants_sha256": hashlib.sha256(
            ("\n".join(sorted(participants)) + "\n").encode()
        ).hexdigest(),
        "files_sha256": extracted_files_digest(stage, participants),
        "script_sha256": script_sha256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def extracted_files_digest(root: Path, participants: list[str]) -> str:
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"extraction date root is not a real directory: {root}")
    aggregate = hashlib.sha256()
    for participant in sorted(participants):
        participant_dir = root / participant
        participant_stat = os.lstat(participant_dir)
        if not stat.S_ISDIR(participant_stat.st_mode) or stat.S_ISLNK(participant_stat.st_mode):
            raise ValueError(f"participant path is not a real directory: {participant_dir}")
        path = participant_dir / TARGET_NAME
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
            raise ValueError(f"target is not a regular non-link file: {path}")
        aggregate.update(participant.encode())
        aggregate.update(b"\t")
        aggregate.update(str(path_stat.st_size).encode())
        aggregate.update(b"\t")
        aggregate.update(digest(path, "sha256").encode())
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def completion_matches(
    path: Path, date: str, members: list[str], script_sha256: str
) -> bool:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        return False
    marker = path / ".complete.json"
    try:
        marker_stat = os.lstat(marker)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(marker_stat.st_mode) or stat.S_ISLNK(marker_stat.st_mode):
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    participant_dirs = []
    try:
        with os.scandir(path) as date_entries:
            for entry in date_entries:
                if entry.name == ".complete.json":
                    continue
                if (
                    not PARTICIPANT_RE.fullmatch(entry.name)
                    or entry.is_symlink()
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    return False
                participant_dir = Path(entry.path)
                with os.scandir(participant_dir) as participant_entries:
                    children = list(participant_entries)
                if len(children) != 1:
                    return False
                target = children[0]
                if (
                    target.name != TARGET_NAME
                    or target.is_symlink()
                    or not target.is_file(follow_symlinks=False)
                ):
                    return False
                participant_dirs.append(entry.name)
    except OSError:
        return False
    participant_dirs.sort()
    participant_hash = hashlib.sha256(
        ("\n".join(participant_dirs) + "\n").encode()
    ).hexdigest()
    try:
        files_hash = extracted_files_digest(path, participant_dirs)
    except (FileNotFoundError, OSError, ValueError):
        return False
    visible_entries = sorted(child.name for child in path.iterdir())
    expected_entries = sorted([".complete.json", *participant_dirs])
    return (
        payload.get("format_version") == EXTRACTION_PROTOCOL_VERSION
        and payload.get("date") == date
        and payload.get("archive_name") == ARCHIVE_NAME
        and payload.get("archive_bytes") == ARCHIVE_BYTES
        and payload.get("archive_md5") == ARCHIVE_MD5
        and payload.get("archive_root") == ARCHIVE_ROOT
        and payload.get("zip_members") == members
        and payload.get("script_sha256") == script_sha256
        and payload.get("n_cough_heavy") == len(participant_dirs)
        and payload.get("participants_sha256") == participant_hash
        and payload.get("files_sha256") == files_hash
        and visible_entries == expected_entries
    )


def extract_date(
    archive: zipfile.ZipFile,
    archive_path: Path,
    out_root: Path,
    date: str,
    members: list[str],
    script_sha256: str,
) -> int:
    final_dir = out_root / date
    if os.path.lexists(final_dir):
        if completion_matches(final_dir, date, members, script_sha256):
            payload = json.loads((final_dir / ".complete.json").read_text())
            return int(payload["n_cough_heavy"])
        raise FileExistsError(f"unverified date directory already exists: {final_dir}")

    stage_parent = out_root / ".staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage_parent_stat = os.lstat(stage_parent)
    if not stat.S_ISDIR(stage_parent_stat.st_mode) or stat.S_ISLNK(
        stage_parent_stat.st_mode
    ):
        raise ValueError(f"staging root is not a real directory: {stage_parent}")
    stage = Path(tempfile.mkdtemp(prefix=f"{date}.", dir=stage_parent))
    participants: set[str] = set()
    try:
        raw = ConcatenatedZipMembers(archive, members)
        with raw, io.BufferedReader(raw, buffer_size=4 << 20) as joined:
            with tarfile.open(fileobj=joined, mode="r|gz") as tar:
                for member in tar:
                    _, participant = validate_tar_path(member.name, date)
                    if member.issym() or member.islnk():
                        raise ValueError(f"links are not allowed in audio stream: {member.name}")
                    if participant is None:
                        continue
                    if not member.isfile():
                        raise ValueError(f"target is not a regular file: {member.name}")
                    if participant in participants:
                        raise ValueError(f"duplicate cough-heavy for {date}/{participant}")
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read tar member: {member.name}")
                    destination_dir = stage / participant
                    destination_dir.mkdir()
                    destination = destination_dir / TARGET_NAME
                    with source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1 << 20)
                    if destination.stat().st_size != member.size:
                        raise IOError(f"short extraction for {member.name}")
                    participants.add(participant)

        payload = completion_payload(
            archive_path,
            stage,
            date,
            members,
            sorted(participants),
            script_sha256,
        )
        marker_tmp = stage / ".complete.json.tmp"
        marker_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(marker_tmp, stage / ".complete.json")
        if os.path.lexists(final_dir):
            raise FileExistsError(f"date directory appeared during extraction: {final_dir}")
        os.rename(stage, final_dir)
        return len(participants)
    except Exception:
        # Keep the private staging tree as forensic evidence.  It is never merged into
        # the final output and a future retry gets a fresh unique staging directory.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--date",
        action="append",
        help="extract only this YYYYMMDD stream; may be repeated (default: all 43)",
    )
    args = parser.parse_args()

    script_sha256 = digest(Path(__file__), "sha256")
    archive, date_parts = verify_archive(args.archive)
    requested = sorted(args.date or date_parts)
    unknown = sorted(set(requested) - set(date_parts))
    if unknown:
        archive.close()
        raise ValueError(f"requested dates absent from pinned archive: {unknown}")

    args.output.mkdir(parents=True, exist_ok=True)
    output_stat = os.lstat(args.output)
    if not stat.S_ISDIR(output_stat.st_mode) or stat.S_ISLNK(output_stat.st_mode):
        archive.close()
        raise ValueError(f"output root is not a real directory: {args.output}")
    lock_path = args.output / ".extract.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            archive.close()
            raise RuntimeError(f"another extractor owns {lock_path}") from error
        try:
            for date in requested:
                count = extract_date(
                    archive,
                    args.archive,
                    args.output,
                    date,
                    date_parts[date],
                    script_sha256,
                )
                print(f"{date}: {count} cough-heavy files", flush=True)

            # Always summarise every verified completed date.  A one-date smoke run can no
            # longer overwrite a previous full summary with a partial run view.
            counts: dict[str, int] = {}
            for date, members in date_parts.items():
                final_dir = args.output / date
                if not os.path.lexists(final_dir):
                    continue
                if not completion_matches(final_dir, date, members, script_sha256):
                    raise ValueError(f"completed date failed provenance verification: {date}")
                payload = json.loads((final_dir / ".complete.json").read_text())
                counts[date] = int(payload["n_cough_heavy"])

            summary = {
                "format_version": EXTRACTION_PROTOCOL_VERSION,
                "archive": str(args.archive.resolve()),
                "archive_bytes": ARCHIVE_BYTES,
                "archive_md5": ARCHIVE_MD5,
                "archive_root": ARCHIVE_ROOT,
                "script_sha256": script_sha256,
                "dates": counts,
                "n_dates": len(counts),
                "n_cough_heavy": sum(counts.values()),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            summary_path = args.output / "extraction_summary.json"
            if os.path.lexists(summary_path):
                summary_stat = os.lstat(summary_path)
                if not stat.S_ISREG(summary_stat.st_mode) or stat.S_ISLNK(
                    summary_stat.st_mode
                ):
                    raise ValueError("existing extraction summary is not a regular file")
                existing = json.loads(summary_path.read_text())
                fixed_keys = (
                    "format_version",
                    "archive_bytes",
                    "archive_md5",
                    "archive_root",
                    "script_sha256",
                )
                if any(existing.get(key) != summary[key] for key in fixed_keys):
                    raise ValueError("existing extraction summary has conflicting provenance")
                existing_dates = existing.get("dates")
                if not isinstance(existing_dates, dict) or any(
                    counts.get(date) != count for date, count in existing_dates.items()
                ):
                    raise ValueError("existing extraction summary conflicts with date markers")
                if existing_dates == counts:
                    print(f"verified unchanged {summary_path}")
                    return
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix="extraction_summary.",
                suffix=".tmp",
                dir=args.output,
                delete=False,
            ) as handle:
                handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                temp_summary = Path(handle.name)
            os.replace(temp_summary, summary_path)
            print(f"wrote {summary_path}")
        finally:
            archive.close()


if __name__ == "__main__":
    main()
