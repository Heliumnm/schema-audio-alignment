"""Freeze a label-blind, participant-linked QC manifest for Coswara cough-heavy audio.

This stage measures file integrity and waveform properties for every extracted recording.
It does not train a model or compute disease-label performance.  Manual quality labels are
joined for later sensitivity analyses but never determine ``objective_qc_pass``.

Example
-------
python src/audit_coswara_audio.py \
  --audio-root /data/Coswara_zenodo_v1/cough_heavy \
  --metadata /data/Coswara_zenodo_v1/pinned/combined_data.csv \
  --quality /data/Coswara_zenodo_v1/pinned/cough-heavy_labels_debottam.csv
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_DIR = REPO_ROOT / "scripts"
if str(EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTOR_DIR))
import extract_coswara_cough_heavy as extractor  # noqa: E402


METADATA_SHA256 = "e462c503bee3408214195855975b0eda08dd1188c0d521b494d93d388d60a72d"
QUALITY_SHA256 = "ab41f10875796818f44022c3f067bc4b3724bc48100bff447edc95354d98a9bb"
MIN_DURATION_S = 0.5
EXPECTED_DATE_NAMES = (
    "20200413",
    "20200415",
    "20200416",
    "20200417",
    "20200418",
    "20200419",
    "20200424",
    "20200430",
    "20200502",
    "20200504",
    "20200505",
    "20200525",
    "20200604",
    "20200707",
    "20200720",
    "20200803",
    "20200814",
    "20200820",
    "20200824",
    "20200901",
    "20200911",
    "20200919",
    "20200930",
    "20201012",
    "20201031",
    "20201130",
    "20201221",
    "20210206",
    "20210406",
    "20210419",
    "20210426",
    "20210507",
    "20210523",
    "20210603",
    "20210618",
    "20210630",
    "20210714",
    "20210816",
    "20210830",
    "20210914",
    "20210930",
    "20220116",
    "20220224",
)
TARGET_NAME = extractor.TARGET_NAME
DATE_RE = re.compile(r"^[0-9]{8}$")
PARTICIPANT_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MEMBER_RE = re.compile(
    rf"^{re.escape(extractor.ARCHIVE_ROOT)}/(?P<date>[0-9]{{8}})/"
    r"(?P=date)\.tar\.gz\.(?P<suffix>[a-z]{2})$"
)


def sha256_file(path: Path, chunk_bytes: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


def read_unique_csv(path: Path, key: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise ValueError(f"{path}: missing key column {key!r}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            value = row[key]
            if not value:
                raise ValueError(f"{path}: blank {key}")
            if value in rows:
                raise ValueError(f"{path}: duplicate {key} {value}")
            rows[value] = row
    return list(reader.fieldnames), rows


def read_quality(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        expected = {"FILENAME", "QUALITY"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"unexpected quality columns: {reader.fieldnames}")
        result: dict[str, int] = {}
        for row in reader:
            filename = row["FILENAME"]
            suffix = "_cough-heavy"
            if not filename.endswith(suffix):
                raise ValueError(f"unexpected quality filename: {filename}")
            participant = filename[: -len(suffix)]
            quality = int(row["QUALITY"])
            if quality not in {0, 1, 2}:
                raise ValueError(f"unexpected quality value for {participant}: {quality}")
            if participant in result:
                raise ValueError(f"duplicate quality row for {participant}")
            result[participant] = quality
    return result


def read_regular_json(path: Path) -> dict[str, object]:
    """Read a JSON record without following a symlink or accepting a special file."""

    try:
        path_stat = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError(f"missing provenance record: {path}") from error
    if not stat.S_ISREG(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"provenance record is not a regular non-link file: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON provenance record: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"provenance record must contain a JSON object: {path}")
    return payload


def validated_marker_members(payload: dict[str, object], date: str) -> list[str]:
    """Validate the exact structural contract of one marker's split ZIP members."""

    raw_members = payload.get("zip_members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError(f"{date}: zip_members must be a non-empty list")
    if any(not isinstance(value, str) for value in raw_members):
        raise ValueError(f"{date}: zip_members contains a non-string value")
    members = list(raw_members)
    if len(set(members)) != len(members):
        raise ValueError(f"{date}: duplicate zip_members")
    suffixes: list[str] = []
    for member in members:
        match = MEMBER_RE.fullmatch(member)
        if match is None or match.group("date") != date:
            raise ValueError(f"{date}: invalid pinned ZIP member {member!r}")
        suffixes.append(match.group("suffix"))
    expected_suffixes = extractor.suffix_sequence(len(members))
    if suffixes != expected_suffixes:
        raise ValueError(
            f"{date}: non-contiguous or out-of-order ZIP members "
            f"{suffixes}, expected {expected_suffixes}"
        )
    return members


def validate_extraction_summary(
    audio_root: Path,
    marker_counts: dict[str, int],
    script_sha256: str,
    *,
    required: bool,
) -> None:
    summary_path = audio_root / "extraction_summary.json"
    if not os.path.lexists(summary_path):
        if required:
            raise ValueError(f"missing full-extraction summary: {summary_path}")
        return
    payload = read_regular_json(summary_path)
    archive_value = payload.get("archive")
    if not isinstance(archive_value, str) or Path(archive_value).name != extractor.ARCHIVE_NAME:
        raise ValueError("extraction summary does not identify the pinned archive")
    expected = {
        "format_version": extractor.EXTRACTION_PROTOCOL_VERSION,
        "archive_bytes": extractor.ARCHIVE_BYTES,
        "archive_md5": extractor.ARCHIVE_MD5,
        "archive_root": extractor.ARCHIVE_ROOT,
        "script_sha256": script_sha256,
        "n_dates": len(marker_counts),
        "n_cough_heavy": sum(marker_counts.values()),
        "dates": marker_counts,
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"extraction summary provenance mismatch: {mismatches}")


def extracted_files(audio_root: Path, allow_partial: bool) -> list[tuple[str, str, Path]]:
    """Return only recordings covered by a fully verified extractor-v2 provenance chain."""

    try:
        root_stat = os.lstat(audio_root)
    except FileNotFoundError as error:
        raise ValueError(f"audio root does not exist: {audio_root}") from error
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"audio root is not a real directory: {audio_root}")

    allowed_support = {".extract.lock", ".staging", "extraction_summary.json"}
    date_names: list[str] = []
    with os.scandir(audio_root) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"symlink is forbidden in extraction root: {entry.path}")
            if DATE_RE.fullmatch(entry.name):
                if not entry.is_dir(follow_symlinks=False):
                    raise ValueError(f"date entry is not a real directory: {entry.path}")
                date_names.append(entry.name)
                continue
            if entry.name not in allowed_support:
                raise ValueError(f"unexpected extraction-root entry: {entry.path}")
            if entry.name == ".staging" and not entry.is_dir(follow_symlinks=False):
                raise ValueError(f"staging entry is not a real directory: {entry.path}")
            if entry.name != ".staging" and not entry.is_file(follow_symlinks=False):
                raise ValueError(f"support entry is not a regular file: {entry.path}")

    observed_dates = tuple(sorted(date_names))
    unexpected_dates = sorted(set(observed_dates) - set(EXPECTED_DATE_NAMES))
    if unexpected_dates:
        raise ValueError(f"unexpected date directories: {unexpected_dates}")
    if not allow_partial and observed_dates != EXPECTED_DATE_NAMES:
        missing = sorted(set(EXPECTED_DATE_NAMES) - set(observed_dates))
        raise ValueError(
            "completed date directories do not equal the pinned 43-date set; "
            f"missing={missing}, observed={list(observed_dates)}"
        )
    if not observed_dates:
        raise ValueError("no completed date directories were found")

    script_path = Path(extractor.__file__).resolve()
    script_sha256 = extractor.digest(script_path, "sha256")
    marker_counts: dict[str, int] = {}
    total_members = 0
    result: list[tuple[str, str, Path]] = []
    for date in observed_dates:
        date_dir = audio_root / date
        marker_payload = read_regular_json(date_dir / ".complete.json")
        members = validated_marker_members(marker_payload, date)
        total_members += len(members)
        if marker_payload.get("archive_name") != extractor.ARCHIVE_NAME:
            raise ValueError(f"{date}: completion record has the wrong archive name")
        if not extractor.completion_matches(date_dir, date, members, script_sha256):
            raise ValueError(f"{date}: extractor-v2 completion verification failed")

        participants = sorted(
            child.name for child in date_dir.iterdir() if child.name != ".complete.json"
        )
        count = int(marker_payload["n_cough_heavy"])
        if count != len(participants):
            raise ValueError(f"{date}: completion count does not match participant directories")
        marker_counts[date] = count
        for participant in participants:
            result.append((date, participant, date_dir / participant / TARGET_NAME))

    if not allow_partial and total_members != extractor.EXPECTED_PARTS:
        raise ValueError(
            f"verified split members {total_members}, expected {extractor.EXPECTED_PARTS}"
        )
    validate_extraction_summary(
        audio_root,
        marker_counts,
        script_sha256,
        required=not allow_partial,
    )
    return result


def waveform_audit(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "file_bytes": path.stat().st_size,
        "raw_sha256": sha256_file(path),
        "decode_ok": False,
        "decode_error": "",
    }
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            samplerate = audio.getframerate()
            header_frames = audio.getnframes()
            compression = audio.getcomptype()
            if compression != "NONE":
                raise ValueError(f"compressed WAV is unsupported: {compression}")
            if channels <= 0 or samplerate <= 0 or sample_width not in {1, 2, 3, 4}:
                raise ValueError(
                    f"invalid PCM layout channels={channels} sr={samplerate} width={sample_width}"
                )
            pcm_hash = hashlib.sha256()
            pcm_hash.update(f"sr={samplerate};channels={channels};".encode())
            n_values = 0
            decoded_frames = 0
            finite = True
            sum_values = 0.0
            sum_squares = 0.0
            peak = 0.0
            n_zero = 0
            n_silent = 0
            n_clipped = 0
            while raw := audio.readframes(65536):
                frame_bytes = channels * sample_width
                if len(raw) % frame_bytes:
                    raise ValueError("decoded PCM block ends inside a frame")
                if sample_width == 1:
                    integer = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
                    block = integer.astype(np.float32) / 128.0
                elif sample_width == 2:
                    integer = np.frombuffer(raw, dtype="<i2")
                    block = integer.astype(np.float32) / 32768.0
                elif sample_width == 3:
                    octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                    integer = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
                    integer = (integer ^ 0x800000) - 0x800000
                    block = integer.astype(np.float32) / 8388608.0
                else:
                    integer = np.frombuffer(raw, dtype="<i4")
                    block = integer.astype(np.float32) / 2147483648.0
                decoded_frames += len(raw) // frame_bytes
                finite = finite and bool(np.isfinite(block).all())
                canonical = np.asarray(block, dtype="<f4", order="C")
                pcm_hash.update(canonical.tobytes())
                flat = canonical.astype(np.float64)
                n_values += flat.size
                sum_values += float(flat.sum())
                sum_squares += float(np.dot(flat, flat))
                absolute = np.abs(flat)
                if flat.size:
                    peak = max(peak, float(absolute.max()))
                n_zero += int(np.count_nonzero(flat == 0.0))
                n_silent += int(np.count_nonzero(absolute < 1e-4))
                n_clipped += int(np.count_nonzero(absolute >= 0.999))
        duration = decoded_frames / samplerate
        record.update(
            {
                "decode_ok": True,
                "format": "WAV",
                "subtype": f"PCM_{sample_width * 8}",
                "samplerate": int(samplerate),
                "channels": int(channels),
                "header_frames": int(header_frames),
                "decoded_frames": int(decoded_frames),
                "header_frame_match": bool(decoded_frames == header_frames),
                "duration_s": float(duration),
                "finite": finite,
                "all_zero": bool(n_values == 0 or sum_squares == 0.0),
                "rms": float(math.sqrt(sum_squares / n_values)) if n_values else math.nan,
                "peak": peak,
                "dc_mean": float(sum_values / n_values) if n_values else math.nan,
                "zero_fraction": float(n_zero / n_values) if n_values else math.nan,
                "silence_fraction": float(n_silent / n_values) if n_values else math.nan,
                "clip_fraction": float(n_clipped / n_values) if n_values else math.nan,
                "pcm_sha256": pcm_hash.hexdigest(),
            }
        )
    except Exception as error:  # record every failure; do not silently drop the row
        record["decode_error"] = f"{type(error).__name__}: {error}"[:500]
    return record


def publish_no_clobber(temporary: Path, destination: Path) -> None:
    """Atomically publish a staged file, refusing to replace any existing result."""

    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite frozen output: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        publish_no_clobber(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        publish_no_clobber(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def output_locks(paths: Iterable[Path]):
    """Hold one non-following advisory lock per final output for the whole audit."""

    destinations = sorted({path.absolute() for path in paths}, key=str)
    if len(destinations) < 2:
        raise ValueError("QC manifest and audit report must be different output paths")
    handles = []
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            lock_path = destination.parent / f".{destination.name}.lock"
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError(f"audit lock is not a regular file: {lock_path}")
            handle = os.fdopen(fd, "a+")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another audit owns {lock_path}") from error
        existing = [str(path) for path in destinations if os.path.lexists(path)]
        if existing:
            raise FileExistsError(f"refusing to overwrite frozen outputs: {existing}")
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


def duplicate_groups(
    rows: list[dict[str, object]], field: str
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = str(row.get(field, ""))
        if value:
            groups[value].append(
                {
                    "participant_id": str(row["participant_id"]),
                    "relative_path": str(row["relative_path"]),
                }
            )
    return {
        value: sorted(
            members,
            key=lambda member: (member["participant_id"], member["relative_path"]),
        )
        for value, members in groups.items()
        if len({member["participant_id"] for member in members}) > 1
    }


def annotate_duplicate_rows(
    rows: list[dict[str, object]],
) -> tuple[
    dict[str, list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
    Counter[str],
]:
    """Mark only the concrete rows whose hashes occur under another participant ID."""

    raw_duplicates = duplicate_groups(rows, "raw_sha256")
    pcm_duplicates = duplicate_groups(rows, "pcm_sha256")
    raw_duplicate_hashes = set(raw_duplicates)
    pcm_duplicate_hashes = set(pcm_duplicates)
    id_counts: Counter[str] = Counter(str(row["participant_id"]) for row in rows)
    for row in rows:
        row["participant_recording_count"] = id_counts[str(row["participant_id"])]
        row["cross_id_raw_duplicate"] = str(row.get("raw_sha256", "")) in raw_duplicate_hashes
        row["cross_id_pcm_duplicate"] = str(row.get("pcm_sha256", "")) in pcm_duplicate_hashes
    return raw_duplicates, pcm_duplicates, id_counts


def run_audit(args: argparse.Namespace) -> None:
    metadata_hash = sha256_file(args.metadata)
    quality_hash = sha256_file(args.quality)
    if metadata_hash != METADATA_SHA256:
        raise ValueError(f"metadata SHA256 {metadata_hash}, expected {METADATA_SHA256}")
    if quality_hash != QUALITY_SHA256:
        raise ValueError(f"quality SHA256 {quality_hash}, expected {QUALITY_SHA256}")

    metadata_fields, metadata = read_unique_csv(args.metadata, "id")
    quality = read_quality(args.quality)
    files = extracted_files(args.audio_root, args.allow_partial)
    rows: list[dict[str, object]] = []
    for index, (date, participant, path) in enumerate(files, start=1):
        relative_path = path.relative_to(args.audio_root).as_posix()
        record: dict[str, object] = {
            "participant_id": participant,
            "archive_date": date,
            "relative_path": relative_path,
            "metadata_present": participant in metadata,
            "manual_quality": quality.get(participant, "[MISSING]"),
        }
        record.update(waveform_audit(path))
        record["objective_qc_pass"] = bool(
            record.get("decode_ok")
            and record.get("finite")
            and not record.get("all_zero")
            and record.get("header_frame_match")
            and int(record.get("file_bytes", 0)) > 44
            and float(record.get("duration_s", 0.0)) >= MIN_DURATION_S
        )
        if participant in metadata:
            for field in metadata_fields:
                if field != "id":
                    record[f"meta_{field}"] = metadata[participant][field]
            normalized_metadata_date = metadata[participant].get("record_date", "").replace("-", "")
            record["record_date_matches_archive"] = normalized_metadata_date == date
        rows.append(record)
        if index % 250 == 0:
            print(f"audited {index}/{len(files)}", flush=True)

    if not rows:
        raise ValueError("no cough-heavy recordings were found")

    # Duplicate group IDs are deterministic hashes rather than row-order-dependent numbers.
    raw_duplicates, pcm_duplicates, id_counts = annotate_duplicate_rows(rows)

    # Union of keys is intentional: decode failures have fewer waveform fields, while
    # metadata-absent recordings have no meta_* values.  Missing output cells remain blank.
    preferred = [
        "participant_id",
        "archive_date",
        "relative_path",
        "metadata_present",
        "manual_quality",
        "objective_qc_pass",
        "participant_recording_count",
        "cross_id_raw_duplicate",
        "cross_id_pcm_duplicate",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fieldnames = preferred + sorted(all_fields - set(preferred))
    atomic_csv(args.out, rows, fieldnames)

    audio_ids = {str(row["participant_id"]) for row in rows}
    metadata_ids = set(metadata)
    durations = sorted(
        float(row["duration_s"])
        for row in rows
        if row.get("decode_ok") and math.isfinite(float(row["duration_s"]))
    )

    def quantile(values: list[float], probability: float) -> float | None:
        if not values:
            return None
        return float(np.quantile(np.asarray(values), probability))

    report: dict[str, object] = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "audio_root": str(args.audio_root.resolve()),
        "metadata_sha256": metadata_hash,
        "quality_sha256": quality_hash,
        "extractor_protocol_version": extractor.EXTRACTION_PROTOCOL_VERSION,
        "extractor_script_sha256": extractor.digest(Path(extractor.__file__).resolve(), "sha256"),
        "archive_bytes": extractor.ARCHIVE_BYTES,
        "archive_md5": extractor.ARCHIVE_MD5,
        "archive_root": extractor.ARCHIVE_ROOT,
        "verified_dates": list(EXPECTED_DATE_NAMES) if not args.allow_partial else sorted(
            {str(row["archive_date"]) for row in rows}
        ),
        "min_duration_s": MIN_DURATION_S,
        "n_recordings": len(rows),
        "n_audio_participants": len(audio_ids),
        "n_metadata_participants": len(metadata_ids),
        "n_audio_without_metadata": len(audio_ids - metadata_ids),
        "audio_without_metadata_examples": sorted(audio_ids - metadata_ids)[:20],
        "n_metadata_without_audio": len(metadata_ids - audio_ids),
        "metadata_without_audio_examples": sorted(metadata_ids - audio_ids)[:20],
        "n_participants_multiple_recordings": sum(count > 1 for count in id_counts.values()),
        "n_decode_failures": sum(not bool(row.get("decode_ok")) for row in rows),
        "n_objective_qc_pass": sum(bool(row["objective_qc_pass"]) for row in rows),
        "n_too_short": sum(
            bool(row.get("decode_ok")) and float(row.get("duration_s", 0.0)) < MIN_DURATION_S
            for row in rows
        ),
        "n_all_zero": sum(bool(row.get("all_zero")) for row in rows),
        "n_nonfinite": sum(row.get("finite") is False for row in rows),
        "n_header_frame_mismatch": sum(
            bool(row.get("decode_ok")) and not bool(row.get("header_frame_match"))
            for row in rows
        ),
        "n_record_date_mismatch": sum(
            row.get("record_date_matches_archive") is False for row in rows
        ),
        "record_date_mismatch_is_diagnostic_only": True,
        "duration_s": {
            "min": quantile(durations, 0.0),
            "p01": quantile(durations, 0.01),
            "median": quantile(durations, 0.5),
            "p99": quantile(durations, 0.99),
            "max": quantile(durations, 1.0),
        },
        "samplerate_counts": dict(
            sorted(Counter(str(row.get("samplerate", "[DECODE_FAILED]")) for row in rows).items())
        ),
        "channel_counts": dict(
            sorted(Counter(str(row.get("channels", "[DECODE_FAILED]")) for row in rows).items())
        ),
        "n_manual_quality_missing": sum(row["manual_quality"] == "[MISSING]" for row in rows),
        "manual_quality_counts": dict(
            sorted(Counter(str(row["manual_quality"]) for row in rows).items())
        ),
        "n_cross_id_raw_duplicate_groups": len(raw_duplicates),
        "n_cross_id_pcm_duplicate_groups": len(pcm_duplicates),
        "cross_id_raw_duplicate_examples": [
            {"sha256": value, "members": raw_duplicates[value]}
            for value in sorted(raw_duplicates)[:20]
        ],
        "cross_id_pcm_duplicate_examples": [
            {"sha256": value, "members": pcm_duplicates[value]}
            for value in sorted(pcm_duplicates)[:20]
        ],
    }
    report["manifest_sha256"] = sha256_file(args.out)
    atomic_json(args.report, report)
    print(f"wrote {args.out} sha256={report['manifest_sha256']}")
    print(f"wrote {args.report}")
    print("No model was fitted and no disease-label performance was computed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/coswara_audio_qc.csv"))
    parser.add_argument(
        "--report", type=Path, default=Path("results/coswara_audio_qc_audit.json")
    )
    parser.add_argument("--allow-partial", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    with output_locks((args.out, args.report)):
        run_audit(args)


if __name__ == "__main__":
    main()
