#!/usr/bin/env bash
# Download the four free corpora for the schema-alignment paper.
#
# The GPU server has no outbound network, so everything is fetched LOCALLY and
# rsynced across afterwards (the script prints the rsync command when it finishes).
#
# Idempotent: each dataset writes a .done marker, so re-running skips completed
# work. Downloads resume on interruption (curl -C - / wget -c / git clone retry).
#
#   bash scripts/download_datasets.sh                      # all
#   bash scripts/download_datasets.sh --only kauh           # one
#   bash scripts/download_datasets.sh --dest /Volumes/ext   # elsewhere
#
# Approximate download sizes: KAUH ~0.2G · COUGHVID ~2.5G · Coswara ~8G
set -uo pipefail

DEST="${DEST:-$HOME/Downloads/resp_datasets}"
ONLY=""
PUSH=""      # e.g. heliu@202.38.72.99:/mnt/hd/data_heliu/resp_datasets/
CLEAN=0      # delete local copy after a successful push
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --push) PUSH="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$DEST"
LOG="$DEST/download.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
done_marker() { echo "$DEST/$1/.done"; }

# Timeouts matter: without them a stalled host (Zenodo rate-limits) hangs the whole
# run with no output instead of failing loudly. --speed-limit/-time aborts a
# connection that opens but then delivers nothing.
fetch() {  # fetch <url> <output-path>
  if command -v curl >/dev/null 2>&1; then
    curl -fL -C - --retry 5 --retry-delay 10 \
         --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
         -o "$2" "$1"
  else
    wget -c -t 5 --timeout=30 --read-timeout=120 -O "$2" "$1"
  fi
}

free_gb() { df -g "$DEST" 2>/dev/null | tail -1 | awk '{print $4}'; }

# Push one dataset to the offline server, then optionally reclaim local space.
# Called right after each dataset completes so peak local usage stays bounded by
# the single largest corpus rather than their sum.
push_and_clean() {  # push_and_clean <name>
  local name="$1"
  [[ -z "$PUSH" ]] && return 0
  say "$name: rsync -> $PUSH"
  if rsync -ahP --partial "$DEST/$name" "$PUSH"; then
    say "$name: pushed"
    if [[ $CLEAN -eq 1 ]]; then
      rm -rf "${DEST:?}/$name"
      say "$name: local copy removed ($(free_gb) GB free)"
    fi
  else
    say "$name: RSYNC FAILED — local copy kept"
  fi
}

# ---------------------------------------------------------------- KAUH
if want kauh && [[ ! -f "$(done_marker KAUH)" ]]; then
  say "KAUH: downloading (Mendeley jwyy9np4gv v3)"
  mkdir -p "$DEST/KAUH"
  URL="https://data.mendeley.com/public-files/datasets/jwyy9np4gv/files/99d7bd63-eb5d-4000-9c2a-a8dd31168cbc/file_downloaded"
  if fetch "$URL" "$DEST/KAUH/kauh.zip"; then
    mkdir -p "$DEST/KAUH/AudioFiles"
    if unzip -qo "$DEST/KAUH/kauh.zip" -d "$DEST/KAUH/AudioFiles"; then
      rm -f "$DEST/KAUH/kauh.zip"; touch "$(done_marker KAUH)"
      say "KAUH: done ($(find "$DEST/KAUH/AudioFiles" -name '*.wav' | wc -l | tr -d ' ') wav)"
      push_and_clean KAUH
    else say "KAUH: UNZIP FAILED — archive may be truncated, delete and rerun"; fi
  else say "KAUH: DOWNLOAD FAILED"; fi
elif want kauh; then
  say "KAUH: already downloaded"; push_and_clean KAUH
fi

# ------------------------------------------------------------ COUGHVID
# Resolve real file URLs through the Zenodo API rather than guessing filenames.
if want coughvid && [[ ! -f "$(done_marker COUGHVID)" ]]; then
  say "COUGHVID: querying Zenodo record 4048312"
  mkdir -p "$DEST/COUGHVID"
  META="$DEST/COUGHVID/_zenodo.json"
  if fetch "https://zenodo.org/api/records/4048312" "$META"; then
    python3 - "$META" "$DEST/COUGHVID/_files.txt" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as f:
    for e in rec.get("files", []):
        link = e.get("links", {}).get("self") or e.get("links", {}).get("download")
        name = e.get("key") or e.get("filename")
        if link and name:
            f.write(f"{link}\t{name}\t{e.get('size', 0)}\n")
PY
    n=0
    while IFS=$'\t' read -r link name size; do
      say "COUGHVID: fetching $name ($(( size / 1024 / 1024 )) MB)"
      fetch "$link" "$DEST/COUGHVID/$name" && n=$((n+1))
    done < "$DEST/COUGHVID/_files.txt"
    for z in "$DEST/COUGHVID"/*.zip; do
      [[ -e "$z" ]] && unzip -qo "$z" -d "$DEST/COUGHVID/" && say "COUGHVID: extracted $(basename "$z")"
    done
    [[ $n -gt 0 ]] && touch "$(done_marker COUGHVID)" && say "COUGHVID: done ($n files)" && push_and_clean COUGHVID
  else say "COUGHVID: ZENODO API FAILED"; fi
elif want coughvid; then
  say "COUGHVID: already downloaded"; push_and_clean COUGHVID
fi

# ------------------------------------------------------------- Coswara
# Split tar archives per recording date; the repo ships extract_data.py to join them.
if want coswara && [[ ! -f "$(done_marker Coswara)" ]]; then
  say "Coswara: cloning iiscleap/Coswara-Data (large, be patient)"
  if [[ -d "$DEST/Coswara/.git" ]]; then
    git -C "$DEST/Coswara" pull --ff-only
  else
    git clone --depth 1 https://github.com/iiscleap/Coswara-Data.git "$DEST/Coswara"
  fi
  if [[ -d "$DEST/Coswara" ]]; then
    if [[ -f "$DEST/Coswara/extract_data.py" ]]; then
      say "Coswara: running extract_data.py"
      ( cd "$DEST/Coswara" && python3 extract_data.py ) || say "Coswara: extract_data.py failed (run manually)"
    else
      say "Coswara: extract_data.py not found — check repo layout before extracting"
    fi
    touch "$(done_marker Coswara)"; say "Coswara: done"; push_and_clean Coswara
  else say "Coswara: CLONE FAILED"; fi
elif want coswara; then
  say "Coswara: already downloaded"; push_and_clean Coswara
fi

# ---------------------------------------------------------------- report
say "----- summary -----"
for d in KAUH COUGHVID Coswara; do
  if [[ -d "$DEST/$d" ]]; then
    printf '  %-10s %8s  %s\n' "$d" "$(du -sh "$DEST/$d" 2>/dev/null | cut -f1)" \
      "$([[ -f "$(done_marker "$d")" ]] && echo ok || echo INCOMPLETE)" | tee -a "$LOG"
  else
    printf '  %-10s %8s  %s\n' "$d" "-" "missing" | tee -a "$LOG"
  fi
done
cat <<EOF | tee -a "$LOG"

ICBHI is already on the server; nothing to fetch.

Next — push to the offline server:
  rsync -avhP --partial "$DEST/" heliu@202.38.72.99:/mnt/hd/data_heliu/resp_datasets/
EOF
