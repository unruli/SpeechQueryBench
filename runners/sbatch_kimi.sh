#!/usr/bin/env bash
#SBATCH --job-name=sqb-kimi
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# Kimi-Audio ASR-mode -> hyps.jsonl. Uses Kimi-Audio/.venv (kimia_infer).
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export PYTHONPATH=${SQB_MODELS}
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=$HF_HOME/hub

: "${MANIFEST:?set MANIFEST}"
python "$HERE/scripts/run_kimi.py" \
  --manifest "$MANIFEST" \
  --model "${KIMI_MODEL:-moonshotai/Kimi-Audio-7B-Instruct}" \
  --system "${KIMI_SYSTEM:-kimi-audio}" \
  --out-dir "${ASR_OUT_DIR:-$ECHODB_RESULTS/asr}" \
  ${LIMIT:+--limit "$LIMIT"}
