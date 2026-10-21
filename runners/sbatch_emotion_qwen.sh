#!/usr/bin/env bash
#SBATCH --job-name=sqb-emoqwen
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Speech-LLM emotion-QA (paralinguistic) on CREMA-D. Reuses the proven Qwen2-Audio env.
#   sbatch --export=ALL,SCRIPT=run_emotion_qwen.py,EMO_OUT=... runners/sbatch_emotion_qwen.sh
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

python "$HERE/scripts/${SCRIPT:-run_emotion_qwen.py}" \
  --labels "$HERE/data/raw/cremad/labels.csv" \
  --out "${EMO_OUT:-$ECHODB_RESULTS/cremad/qwen.jsonl}"
