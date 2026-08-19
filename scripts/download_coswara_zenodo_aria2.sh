#!/usr/bin/env bash
# Parallel, resumable downloader for the pinned Coswara Zenodo v1.0 archive.
#
# The Zenodo endpoint supports byte ranges. aria2 records completed pieces in a sidecar
# file, so a broken connection or stopped process can resume without truncating progress.

set -u

if [[ $# -ne 1 ]]; then
  echo "usage: $0 TARGET_DIRECTORY" >&2
  exit 2
fi

target_dir=$1
work_dir="$target_dir/aria_work"
archive_name=Coswara-Data-dataset-paper-publication.zip
work_path="$work_dir/$archive_name"
final_path="$target_dir/$archive_name"
url='https://zenodo.org/api/records/7188627/files/iiscleap/Coswara-Data-dataset-paper-publication.zip/content'
expected_size=12984309908
expected_md5=53721d9c106f99872bf7f878c8196d31

mkdir -p "$work_dir"

verify_archive() {
  local path=$1
  local actual_size actual_md5
  actual_size=$(stat -c '%s' "$path")
  if [[ $actual_size -ne $expected_size ]]; then
    echo "unexpected archive size: $actual_size" >&2
    return 1
  fi
  actual_md5=$(md5sum "$path" | awk '{print $1}')
  if [[ $actual_md5 != $expected_md5 ]]; then
    echo "unexpected archive MD5: $actual_md5" >&2
    return 1
  fi
}

if [[ -f $final_path ]]; then
  verify_archive "$final_path" || exit 3
  echo "archive already complete and verified: $final_path"
  exit 0
fi

aria2c \
  --continue=true \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --file-allocation=none \
  --max-connection-per-server=8 \
  --split=8 \
  --min-split-size=1M \
  --max-tries=0 \
  --retry-wait=15 \
  --connect-timeout=30 \
  --timeout=60 \
  --lowest-speed-limit=1K \
  --dir="$work_dir" \
  --out="$archive_name" \
  "$url"
download_status=$?

if [[ $download_status -ne 0 ]]; then
  echo "aria2 exited with status $download_status; keep the file and .aria2 sidecar to resume" >&2
  exit 4
fi

if [[ -f $work_path.aria2 ]]; then
  echo "aria2 returned success but its control sidecar still exists" >&2
  exit 5
fi

verify_archive "$work_path" || exit 6
mv "$work_path" "$final_path"
echo "archive complete and verified: $final_path"
