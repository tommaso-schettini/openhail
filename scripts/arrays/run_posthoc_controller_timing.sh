#!/bin/bash
# Run the entire paired timing matrix sequentially in one allocation.
# Submit from the repository root after synchronizing the benchmark source
# and all configuration-specific checkpoints from batch 1455644.
#SBATCH --job-name=posthoc-controller-timing
#SBATCH --account=def-tms
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail
module purge
module load python/3.13 scipy-stack proj/9.4.1
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PROJ_DATA="$EBROOTPROJ/share/proj"
export SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
test -f "$PROJ_DATA/proj.db"
if [ -f manifest.json ]; then
    python verify_bundle.py
fi
python -c 'import sys, torch, threadpoolctl, openhail; assert sys.version_info[:2] == (3, 13); print("OpenHail source:", openhail.__file__)'

OUTPUT_DIR="out/posthoc_controller_timing_slurm_${SLURM_JOB_ID}"
srun --cpu-bind=cores python scripts/utilities/benchmark_trained_controllers.py \
    --output-dir "$OUTPUT_DIR" --num-episodes 5
python scripts/utilities/analyze_posthoc_timings.py --input-dir "$OUTPUT_DIR"
