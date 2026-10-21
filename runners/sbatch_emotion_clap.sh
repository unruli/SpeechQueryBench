#!/usr/bin/env bash
#SBATCH --job-name=sqb-emoclap
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# CLAP zero-shot emotion classification (paralinguistic family) on CREMA-D. L4 (MSCLAP venv).
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}

python "$HERE/scripts/run_emotion_clap.py" \
  --labels "$HERE/data/raw/cremad/labels.csv" \
  --out "${EMO_OUT:-$ECHODB_RESULTS/cremad/clap.jsonl}"
