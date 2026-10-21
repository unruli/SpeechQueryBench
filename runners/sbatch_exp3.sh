#!/usr/bin/env bash
#SBATCH --job-name=sqb-exp3
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
echo "=== Experiment 1: AMI-100 pairwise ==="
python "$HERE/scripts/exp_ami100_pairwise.py"
echo ""
echo "=== Experiment 2: Cross-representation effect sizes ==="
python "$HERE/scripts/exp_cross_repr_effects.py"
echo ""
echo "=== Experiment 3: Semantic threshold sweep ==="
python "$HERE/scripts/exp_semantic_threshold.py"
echo ""
echo "=== ALL DONE ==="
