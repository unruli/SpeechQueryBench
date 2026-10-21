#!/usr/bin/env bash
#SBATCH --job-name=sqb-granite
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# IBM Granite Speech ASR -> hyps.jsonl (non-Whisper ASR). Uses .venvASR312 (transformers+peft+torchaudio).
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=$HF_HOME/hub; export HF_HUB_OFFLINE=1

: "${MANIFEST:?set MANIFEST}"
python "$HERE/scripts/run_granite.py" \
  --manifest "$MANIFEST" \
  --model "${GRANITE_MODEL:-ibm-granite/granite-speech-3.3-2b}" \
  --system "${GRANITE_SYSTEM:-granite-3.3-2b}" \
  --out-dir "${ASR_OUT_DIR:-$ECHODB_RESULTS/asr}" \
  ${LIMIT:+--limit "$LIMIT"}
