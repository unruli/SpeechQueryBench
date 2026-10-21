#!/usr/bin/env bash
#SBATCH --job-name=sqb-hybrid
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Retrieval->speech-LLM hybrid for semantic queries. Step 1: chunk-aligned ASR (faster-whisper,
# project venv). Step 2: lexical gate + Qwen2-Audio verify (.venvASR312). Then eval on the login node.
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load cuda/12.8.1 cudnn/9.6.0 2>/dev/null || true
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH; export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

SEM="$ECHODB_RESULTS/ami/semantic"; mkdir -p "$SEM/chunks"
CONC="$HERE/workload/semantic_concepts.yaml"
AMI="$ECHODB_MANIFESTS/ami.csv"
LIM="${SEM_LIMIT:-12}"

echo "===== STEP 1: chunk-aligned ASR (faster-whisper large-v3) ====="
source "$ECHODB_VENV/bin/activate"
python "$HERE/scripts/chunk_transcribe.py" --manifest "$AMI" --model large-v3 \
  --out-dir "$SEM/chunks" --chunk-sec 30 --limit "$LIM"
deactivate

echo "===== STEP 2: lexical gate + Qwen2-Audio verify ====="
source ${SQB_ROOT}/.venv/bin/activate
export HF_HOME=${SQB_MODELS}
export HF_HUB_CACHE=$HF_HOME/hub; export HF_HUB_OFFLINE=1
python "$HERE/scripts/sem_hybrid.py" --manifest "$AMI" --concepts "$CONC" \
  --chunks-dir "$SEM/chunks" --out "$SEM/hybrid.jsonl" --top-k 3 --chunk-sec 30 --limit "$LIM"

echo "===== DONE ====="
