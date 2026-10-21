#!/usr/bin/env bash
#SBATCH --job-name=sqb-semantic
#SBATCH --account=mygroup
#SBATCH --partition=hpg-rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
# Semantic-query experiment (AMI): gold (Qwen2.5 over reference) + ASR+text-LLM (Qwen2.5 over
# large-v3 transcript) + speech-LLM-QA (Qwen2-Audio over audio). Models load sequentially.
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

SEM="$ECHODB_RESULTS/ami/semantic"; mkdir -p "$SEM"
CONC="$HERE/workload/semantic_concepts.yaml"
AMI="$ECHODB_MANIFESTS/ami.csv"
LARGEV3="$ECHODB_RESULTS/ami/asr/large-v3/hyps.jsonl"
LIM="${SEM_LIMIT:-12}"

if [ "${SEM_SKIP_TEXT:-0}" != "1" ]; then
  echo "===== GOLD: Qwen2.5 over reference ====="
  python "$HERE/scripts/sem_textllm.py" --manifest "$AMI" --source reference \
    --concepts "$CONC" --out "$SEM/gold.jsonl" --limit "$LIM"

  echo "===== ASR+text-LLM: Qwen2.5 over large-v3 transcript ====="
  python "$HERE/scripts/sem_textllm.py" --manifest "$AMI" --source "$LARGEV3" \
    --concepts "$CONC" --out "$SEM/asrllm_largev3.jsonl" --limit "$LIM"
fi

echo "===== speech-LLM-QA: Qwen2-Audio over audio ====="
python "$HERE/scripts/sem_speechllm.py" --manifest "$AMI" \
  --concepts "$CONC" --out "$SEM/speechllm_qa.jsonl" --limit "$LIM"

echo "===== DONE ====="
