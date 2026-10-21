#!/usr/bin/env bash
# Week-1 go/no-go pilot driver (run on a an HPC cluster.
#   1) build LibriSpeech recording-level manifest       (CPU, here)
#   2) generate keyword + aggregate workload            (CPU, here)
#   3) submit one ASR job per tier                       (GPU, sbatch)
#   4) submit the analysis job, dependent on all ASR     (CPU, sbatch)
# Prereq: bash scripts/setup_env.sh   (creates the venv with faster-whisper)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config/paths.env"

if [ ! -d "$ECHODB_VENV" ]; then
  echo "ERROR: venv missing. Run: bash scripts/setup_env.sh" >&2; exit 1
fi
# shellcheck disable=SC1091
source "$ECHODB_VENV/bin/activate"

MANIFEST="$ECHODB_MANIFESTS/librispeech_test_clean.csv"
WORKLOAD="$ECHODB_ROOT/workload/queries.jsonl"

echo "[1/4] manifest"
python "$HERE/scripts/build_manifest_librispeech.py" \
  --root "$LIBRISPEECH_ROOT" --split "$LIBRISPEECH_SPLIT" --out "$MANIFEST"

echo "[2/4] workload"
python "$HERE/scripts/gen_workload.py" \
  --manifest "$MANIFEST" --out "$WORKLOAD" --per-stratum 40 --seed 0

echo "[3/4] submit ASR tiers: $ASR_TIERS"
DEP=""
IFS=',' read -ra TIERS <<< "$ASR_TIERS"
for M in "${TIERS[@]}"; do
  JID=$(sbatch --parsable --export=ALL,ECHODB_HOME="$HERE",MODEL="$M",MANIFEST="$MANIFEST" "$HERE/runners/sbatch_asr.sh")
  echo "   submitted ASR $M -> job $JID"
  DEP="${DEP}:${JID}"
done

echo "[4/4] submit analysis (afterok${DEP})"
AID=$(sbatch --parsable --dependency=afterok"${DEP}" \
  --export=ALL,ECHODB_HOME="$HERE",MANIFEST="$MANIFEST",WORKLOAD="$WORKLOAD" "$HERE/runners/sbatch_analyze.sh")
echo "   submitted analysis -> job $AID"
echo
echo "watch:  squeue -u \$USER"
echo "result: $ECHODB_RESULTS/pilot_summary.csv  (+ kill-test report in echodb-analyze-$AID.out)"
