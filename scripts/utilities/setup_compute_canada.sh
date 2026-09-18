#!/bin/bash
# Set up the persistent Python environment used by Compute Canada jobs.

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_MODULE="${PYTHON_MODULE:-python/3.13}"
VENV_DIR="${VENV_DIR:-.venv}"

module purge
module load "$PYTHON_MODULE"
module load scipy-stack
module load proj/9.4.1

if [ -z "${EBROOTPROJ:-}" ]; then
  echo "The proj module did not define EBROOTPROJ." >&2
  exit 1
fi
export PROJ_DATA="$EBROOTPROJ/share/proj"
if [ ! -f "$PROJ_DATA/proj.db" ]; then
  echo "Missing PROJ database: $PROJ_DATA/proj.db" >&2
  exit 1
fi

if [ -e "$VENV_DIR" ] && [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Refusing to replace incomplete environment: $VENV_DIR" >&2
  echo "Move or remove it explicitly, then run this script again." >&2
  exit 1
fi

if [ ! -e "$VENV_DIR" ]; then
  virtualenv --no-download "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 13):
    raise SystemExit(
        "The existing virtual environment uses "
        f"Python {sys.version_info.major}.{sys.version_info.minor}; "
        "move or remove it and rerun setup with python/3.13 loaded."
    )
PY

python -m pip install --no-index --upgrade pip
# Alliance's pip configuration prefers its local wheelhouse. Some Python 3.13
# dependencies (notably pygame) are not currently mirrored there, so permit
# pip to fall back to PyPI when this setup is run on a login node.
python -m pip install -e '.[ppo]'

mkdir -p logs err out/periodic_ppo_training

python - <<'PY'
import gymnasium
import importlib.metadata
import numpy
import pandas
import torch
from pathlib import Path
from openhail.agents import PeriodicPPOAgent
from pyproj import CRS, datadir

proj_data = Path(datadir.get_data_dir())
if not (proj_data / "proj.db").is_file():
    raise SystemExit(f"Missing PROJ database: {proj_data / 'proj.db'}")
CRS.from_epsg(4326)

print("Compute Canada environment is ready.")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"pandas: {pandas.__version__}")
print(f"Gymnasium: {gymnasium.__version__}")
print(f"OpenHail: {importlib.metadata.version('openhail')}")
print(f"Periodic PPO agent: {PeriodicPPOAgent.__name__}")
print(f"PROJ data: {proj_data}")
PY

echo "Submit from the repository root with:"
echo "  sbatch scripts/arrays/run_periodic_ppo_nyc_day1_fleet20_9infra_3seeds_500ep_array.sh"
