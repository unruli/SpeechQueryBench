#!/usr/bin/env bash
#SBATCH --job-name=echodb-analyze
#SBATCH --account=mygroup
#SBATCH --partition=hpg-milan
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# CPU-only: execute queries over all systems + score the kill test.
#   sbatch --export=ALL,MANIFEST=...,WORKLOAD=... runners/sbatch_analyze.sh
set -euo pipefail
# Under SLURM, $0 is the spool copy, so resolve the project root from an exported var.
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source "$ECHODB_VENV/bin/activate"

: "${MANIFEST:?set MANIFEST}"
: "${WORKLOAD:?set WORKLOAD}"

ASR_OUT_DIR="${ASR_OUT_DIR:-$ECHODB_RESULTS/asr}"
QUERIES_DIR="${QUERIES_DIR:-$ECHODB_RESULTS/queries}"
SUMMARY_CSV="${SUMMARY_CSV:-$ECHODB_RESULTS/pilot_summary.csv}"

python "$HERE/scripts/run_queries.py" \
  --manifest "$MANIFEST" --workload "$WORKLOAD" \
  --asr-dir "$ASR_OUT_DIR" --out-dir "$QUERIES_DIR"

python "$HERE/scripts/eval_pilot.py" \
  --manifest "$MANIFEST" --workload "$WORKLOAD" \
  --asr-dir "$ASR_OUT_DIR" --queries-dir "$QUERIES_DIR" \
  --out "$SUMMARY_CSV"
