#!/usr/bin/env bash
#SBATCH --job-name=sqb-snr
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
module load ffmpeg/n6.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export TORCHAUDIO_USE_BACKEND_DISPATCHER=soundfile
python "$HERE/scripts/snr_sensitivity.py" \
  --speech-manifest "$ECHODB_MANIFESTS/librispeech_test_clean.csv" \
  --n 100 --snr-levels 0,4,8,12,16 \
  --out "$ECHODB_RESULTS/stats/snr_sensitivity.csv"
echo "=== DONE ==="
