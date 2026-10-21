#!/usr/bin/env bash
#SBATCH --job-name=sqb-pairwise
#SBATCH --account=mygroup
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail
HERE="${ECHODB_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "$HERE/config/paths.env"
source "${ECHODB_VENV}/bin/activate"
python "$HERE/scripts/pairwise_significance.py" --results-dir "$ECHODB_RESULTS/ami" --B 5000
echo "=== DONE ==="
