#!/usr/bin/env bash
#SBATCH --job-name=sqb-ac-textllm
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
cd ${SQB_ROOT:-.}
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS} HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/run_acoustic_textllm.py --labels data/raw/acoustic/labels.csv --out results/acoustic/textllm.jsonl
