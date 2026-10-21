#!/usr/bin/env bash
#SBATCH --job-name=sqb-clap
#SBATCH --account=mygroup
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Direct speech-embedding retrieval (MS-CLAP) for semantic queries. L4 (MSCLAP venv torch may lack
# Blackwell kernels, like the diarizer).
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
module load ffmpeg/n6.1 2>/dev/null || true   # torchcodec (CLAP audio loading) needs libav* from FFmpeg
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}

: "${MANIFEST:?set MANIFEST}"
python "$HERE/scripts/sem_clap.py" \
  --manifest "$MANIFEST" \
  --concepts "$HERE/workload/semantic_concepts.yaml" \
  --out "${CLAP_OUT:-$ECHODB_RESULTS/ami/semantic/clap.jsonl}" \
  --win "${CLAP_WIN:-10}" \
  ${LIMIT:+--limit "$LIMIT"}
