#!/usr/bin/env bash
#SBATCH --job-name=sqb-diarize
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
# pyannote speaker diarization for the interaction/temporal family. Uses Diarizer/.venv + cached models.
# NB: run on L4 (hpg-turin) -- Diarizer/.venv torch lacks kernels for Blackwell (rtx6000/b200).
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
# pyannote models are cached here (models--pyannote--segmentation-3.0 / speaker-diarization-3.1)
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=${SQB_MODELS}
export HF_HUB_OFFLINE=1

: "${MANIFEST:?set MANIFEST}"
python "$HERE/scripts/interaction_diarize.py" \
  --manifest "$MANIFEST" \
  --out "${DIAR_OUT:-$ECHODB_RESULTS/ami/interaction/pyannote.jsonl}" \
  ${LIMIT:+--limit "$LIMIT"}
