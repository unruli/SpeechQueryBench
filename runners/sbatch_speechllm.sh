#!/usr/bin/env bash
#SBATCH --job-name=sqb-speechllm
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Direct speech-LLM (ASR-mode) transcription -> hyps.jsonl. Reuses the proven Qwen2-Audio env.
#   sbatch --export=ALL,ECHODB_HOME=...,MANIFEST=...,ASR_OUT_DIR=... runners/sbatch_speechllm.sh
# Qwen2-Audio-7B fp16 ~16GB; rtx_pro_6000 (idle) has ample memory. L4 (hpg-turin) is tight but may work.
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"

source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# proven Qwen2-Audio venv (.venvASR312 has transformers+qwen2_audio) + cached weights (offline)
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

: "${MANIFEST:?set MANIFEST}"
python "$HERE/scripts/run_speechllm.py" \
  --manifest "$MANIFEST" \
  --model "${SLLM_MODEL:-Qwen/Qwen2-Audio-7B-Instruct}" \
  --system "${SLLM_SYSTEM:-qwen2-audio}" \
  --out-dir "${ASR_OUT_DIR:-$ECHODB_RESULTS/asr}" \
  --chunk-sec "${CHUNK_SEC:-30}" \
  ${LIMIT:+--limit "$LIMIT"}
