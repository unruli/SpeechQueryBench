#!/usr/bin/env bash
# Fetch AMI Mix-Headset audio for the test+dev meetings into blue (Week 2).
# Annotations (RTTM / word alignments) are already present at $AMI_ANNOT; only audio is missing.
# AMI is freely available under CC-BY-4.0. Mix-Headset (single far-mix wav per meeting) is the
# smallest useful stream for query experiments (~10-25 GB for test+dev).
#
# This DOWNLOADS data. Review the meeting list first; run with --go to actually fetch.
#   bash scripts/fetch_ami_audio.sh          # dry run: prints what it would download
#   bash scripts/fetch_ami_audio.sh --go     # actually download
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config/paths.env"

BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
LISTS=("$AMI_ANNOT/lists/test.meetings.txt" "$AMI_ANNOT/lists/dev.meetings.txt")
DEST="$AMI_AUDIO"
GO="${1:-}"

mkdir -p "$DEST"
mapfile -t MEETINGS < <(cat "${LISTS[@]}" | tr -d '\r' | awk 'NF' | sort -u)
echo "meetings: ${#MEETINGS[@]}  ->  $DEST"

for M in "${MEETINGS[@]}"; do
  URL="$BASE/$M/audio/$M.Mix-Headset.wav"
  OUT="$DEST/$M/audio/$M.Mix-Headset.wav"
  if [ "$GO" = "--go" ]; then
    mkdir -p "$(dirname "$OUT")"
    [ -f "$OUT" ] && { echo "skip $M"; continue; }
    echo "GET $URL"
    wget -q --show-progress -O "$OUT" "$URL"
  else
    echo "would GET $URL -> $OUT"
  fi
done
echo "done (${GO:-dry-run}). Next: python scripts/build_manifest_ami.py"
