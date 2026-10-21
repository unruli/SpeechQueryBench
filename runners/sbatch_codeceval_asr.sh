#!/usr/bin/env bash
#SBATCH --job-name=sqb-asrx
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Transcribe our manifests with CodecEval ASR models (Parakeet/CanaryQ/Q3ASR) -> hyps.jsonl.
#   sbatch --job-name=sqb-Parakeet --export=ALL,ECHODB_HOME=...,MANIFEST=...,MODEL=...,SYSTEM=...,OUT_DIR=... sbatch_codeceval_asr.sh
set -uo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$HERE"
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
source "${VENV_PATH:-${SQB_ROOT}/.venv/bin/activate"
export HF_HOME="${HF_HOME:-${SQB_MODELS}"
export PYTHONUNBUFFERED=1
: "${MANIFEST:?set MANIFEST}"; : "${MODEL:?set MODEL}"; : "${SYSTEM:?set SYSTEM}"; : "${OUT_DIR:?set OUT_DIR}"
python scripts/run_codeceval_asr.py --manifest "$MANIFEST" --model "$MODEL" --system "$SYSTEM" \
  --out-dir "$OUT_DIR" --chunk-sec "${CHUNK_SEC:-30}" ${LIMIT:+--limit "$LIMIT"}
