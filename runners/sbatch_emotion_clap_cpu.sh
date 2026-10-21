#!/usr/bin/env bash
#SBATCH --job-name=sqb-emoclapcpu
#SBATCH --account=mygroup
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# CLAP zero-shot emotion on CREMA-D, CPU-only (small model, 300 short clips) -- dodges the GPU GRES quota.
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export CUDA_VISIBLE_DEVICES=""

python "$HERE/scripts/run_emotion_clap.py" \
  --labels "$HERE/data/raw/cremad/labels.csv" \
  --out "${EMO_OUT:-$ECHODB_RESULTS/cremad/clap.jsonl}" --cpu
