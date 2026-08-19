#!/usr/bin/env bash
# Parallel, resumable downloader for the pinned Coswara Zenodo v1.0 archive.
#
# The Zenodo endpoint supports byte ranges. aria2 records completed pieces in a sidecar
# file, so a broken connection or stopped process can resume without truncating progress.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TARGET_DIRECTORY" >&2
  exit 2
fi

target_dir=$1
work_dir="$target_dir/aria_work"
archive_name=Coswara-Data-dataset-paper-publication.zip
work_path="$work_dir/$archive_name"
final_path="$target_dir/$archive_name"
sidecar_path="$work_path.aria2"
lock_path="$target_dir/.coswara_download.lock"
url='https://zenodo.org/api/records/7188627/files/iiscleap/Coswara-Data-dataset-paper-publication.zip/content'
expected_size=12984309908
expected_md5=53721d9c106f99872bf7f878c8196d31
minimum_free_bytes=25000000000
expected_entries=418
archive_root=iiscleap-Coswara-Data-bf300ae

for command_name in aria2c awk date df du flock md5sum mv sed stat unzip wc zipinfo; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required command is missing: $command_name" >&2
    exit 10
  fi
done

mkdir -p "$target_dir" "$work_dir"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another Coswara downloader owns the lock: $lock_path" >&2
  exit 11
fi

verify_archive() {
  local path=$1
  local actual_size actual_md5 entry_count outside_root date_count csv_count part_count
  actual_size=$(stat -c '%s' "$path") || return 1
  if [[ $actual_size -ne $expected_size ]]; then
    echo "unexpected archive size: $actual_size" >&2
    return 1
  fi
  actual_md5=$(md5sum "$path" | awk '{print $1}') || return 1
  if [[ $actual_md5 != $expected_md5 ]]; then
    echo "unexpected archive MD5: $actual_md5" >&2
    return 1
  fi
  if ! unzip -tq "$path" >/dev/null; then
    echo "ZIP CRC/integrity check failed" >&2
    return 1
  fi
  entry_count=$(zipinfo -1 "$path" | wc -l | awk '{print $1}') || return 1
  if [[ $entry_count -ne $expected_entries ]]; then
    echo "unexpected ZIP entry count: $entry_count" >&2
    return 1
  fi
  outside_root=$(zipinfo -1 "$path" | awk -v root="$archive_root/" \
    'index($0, root) != 1 {n += 1} END {print n + 0}') || return 1
  date_count=$(zipinfo -1 "$path" | awk -F/ -v root="$archive_root" \
    '$1 == root && length($2) == 8 && $2 !~ /[^0-9]/ {seen[$2] = 1} \
     END {for (x in seen) n += 1; print n + 0}') || return 1
  csv_count=$(zipinfo -1 "$path" | awk -F/ -v root="$archive_root" \
    '$1 == root && length($2) == 8 && $2 !~ /[^0-9]/ && $3 == $2 ".csv" {n += 1} \
     END {print n + 0}') || return 1
  part_count=$(zipinfo -1 "$path" | awk -F/ -v root="$archive_root" \
    '$1 == root && length($2) == 8 && $2 !~ /[^0-9]/ && \
     $3 ~ /^[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]\.tar\.gz\.[a-z][a-z]$/ {n += 1} \
     END {print n + 0}') || return 1
  if [[ $outside_root -ne 0 || $date_count -ne 43 || $csv_count -ne 43 || $part_count -ne 153 ]]; then
    echo "unexpected ZIP structure: outside_root=$outside_root dates=$date_count date_csv=$csv_count tar_parts=$part_count" >&2
    return 1
  fi
}

promote_verified_work_file() {
  if [[ -e $final_path ]]; then
    verify_archive "$final_path" || return 1
    echo "a verified final archive already exists; preserving both paths for review" >&2
    return 1
  fi
  mv -n "$work_path" "$final_path"
  if [[ -e $work_path ]]; then
    echo "the no-clobber move did not promote the verified work file" >&2
    return 1
  fi
  verify_archive "$final_path"
}

if [[ -e $final_path ]]; then
  verify_archive "$final_path" || exit 3
  echo "archive already complete and verified: $final_path"
  exit 0
fi

# A data file without its aria2 control file can be a completely downloaded archive
# whose wrapper was interrupted before verification. Verify it before doing anything.
# If it is incomplete or corrupt, fail closed: never let aria2 silently overwrite it.
if [[ -e $work_path && ! -e $sidecar_path ]]; then
  if verify_archive "$work_path"; then
    promote_verified_work_file || exit 13
    echo "recovered and promoted a complete orphan work file: $final_path"
    exit 0
  fi
  echo "work data exists without an aria2 sidecar and is not complete; preserve it for review" >&2
  exit 14
fi

if [[ -e $sidecar_path && ! -e $work_path ]]; then
  echo "aria2 sidecar exists without its data file; refusing to invent download state" >&2
  exit 15
fi

# A completed archive needs no spare-space gate.  For a true resume, require only the
# remaining allocated bytes plus a 2 GB safety margin; for a fresh download retain the
# conservative 25 GB gate used by the preregistered data plan.
available_bytes=$(df -PB1 "$target_dir" | awk 'NR==2 {print $4}')
if [[ ! $available_bytes =~ ^[0-9]+$ ]]; then
  echo "could not determine free bytes on the target filesystem" >&2
  exit 12
fi
required_free=$minimum_free_bytes
if [[ -e $work_path && -e $sidecar_path ]]; then
  allocated_bytes=$(du -B1 "$work_path" | awk '{print $1}')
  if [[ ! $allocated_bytes =~ ^[0-9]+$ ]]; then
    echo "could not determine allocated bytes for the resumable work file" >&2
    exit 16
  fi
  remaining_bytes=$(( expected_size > allocated_bytes ? expected_size - allocated_bytes : 0 ))
  required_free=$(( remaining_bytes + 2000000000 ))
fi
if (( available_bytes < required_free )); then
  echo "insufficient free space: available=$available_bytes required=$required_free" >&2
  exit 12
fi

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "url=$url"
echo "expected_size=$expected_size"
echo "expected_md5=$expected_md5"
aria2c --version | sed -n '1p'

download_status=0
aria2c \
  --continue=true \
  --auto-file-renaming=false \
  --allow-overwrite=false \
  --file-allocation=none \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=1M \
  --max-tries=100 \
  --retry-wait=15 \
  --connect-timeout=30 \
  --timeout=60 \
  --lowest-speed-limit=1K \
  --dir="$work_dir" \
  --out="$archive_name" \
  "$url" || download_status=$?

if (( download_status != 0 )); then
  echo "aria2 exited with status $download_status; preserving data and sidecar for resume" >&2
  exit 4
fi

if [[ -e $sidecar_path ]]; then
  echo "aria2 returned success but its control sidecar still exists" >&2
  exit 5
fi

verify_archive "$work_path" || exit 6
promote_verified_work_file || exit 7
echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "archive complete and verified: $final_path"
