#!/usr/bin/env bash
#SBATCH --job-name=echodb-asr
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# NOTE: L4 (24GB) fits Whisper large-v3 easily and hpg-turin has idle nodes; hpg-b200 was fully
# allocated. Override for B200 with:  sbatch --partition=hpg-b200 --gpus=b200:1 ...
# Transcribe the manifest with ONE ASR tier. Pass the model via env:
#   sbatch --export=ALL,MODEL=large-v3,MANIFEST=... runners/sbatch_asr.sh
set -euo pipefail
# Under SLURM, $0 is the spool copy, so resolve the project root from an exported var.
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"

# CTranslate2/faster-whisper need CUDA 12 cuBLAS + cuDNN 9 (libcublas.so.12/libcudnn.so.9) at runtime.
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1 cudnn/9.6.0 2>/dev/null || true

source "$ECHODB_VENV/bin/activate"

: "${MODEL:?set MODEL (e.g. large-v3)}"
: "${MANIFEST:?set MANIFEST (path to manifest csv)}"

python "$HERE/scripts/run_asr.py" \
  --manifest "$MANIFEST" \
  --model "$MODEL" \
  --out-dir "${ASR_OUT_DIR:-$ECHODB_RESULTS/asr}" \
  --compute-type "${ASR_COMPUTE_TYPE:-float16}" \
  ${LIMIT:+--limit "$LIMIT"}
