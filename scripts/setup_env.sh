#!/usr/bin/env bash
# Create the pilot venv on blue. Run once, on a login node.
#   bash scripts/setup_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config/paths.env"

# an HPC cluster.
module load python/3.11 2>/dev/null || module load conda 2>/dev/null || true

if [ ! -d "$ECHODB_VENV" ]; then
  python3 -m venv "$ECHODB_VENV"
fi
# shellcheck disable=SC1091
source "$ECHODB_VENV/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -r "$HERE/requirements.txt"

echo "venv ready: $ECHODB_VENV"
python - <<'PY'
import importlib.util
for m in ["faster_whisper", "numpy", "pandas", "scipy"]:
    print(m, "OK" if importlib.util.find_spec(m) else "MISSING")
PY
