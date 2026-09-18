#!/bin/bash
#SBATCH --job-name=openhail-analysis
#SBATCH --account=def-tms
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=1-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err

set -euo pipefail

module purge
module load python/3.13
module load scipy-stack
module load proj/9.4.1

mkdir -p logs err
if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run: bash scripts/utilities/setup_compute_canada.sh"
  exit 1
fi
source .venv/bin/activate
python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
if [ -z "${EBROOTPROJ:-}" ]; then
  echo "The proj module did not define EBROOTPROJ."
  exit 1
fi
export PROJ_DATA="$EBROOTPROJ/share/proj"
if [ ! -f "$PROJ_DATA/proj.db" ]; then
  echo "Missing PROJ database: $PROJ_DATA/proj.db"
  exit 1
fi
python -c 'from pyproj import CRS; CRS.from_epsg(4326)'
echo "PROJ data: $PROJ_DATA"

python scripts/utilities/analyze_simulator.py \
  --instances scripts/instances/simulator_analysis_instances.tsv \
  --config-index "$SLURM_ARRAY_TASK_ID" \
  --policies nearest random
