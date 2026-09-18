#!/bin/bash
# Replicated paper experiment: NYC day 1, fleet 20, 9 infrastructures,
# 3 independent training seeds, and 500 training episodes per run.
#SBATCH --job-name=ppo-nyc-d1-f20-9i-3s-500
#SBATCH --account=def-tms
# ---------------------------------------------------------------------
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00
# ---------------------------------------------------------------------
#SBATCH --array=1-27%9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=err/%x_%A_%a.err
# ---------------------------------------------------------------------
#SBATCH --mail-user=tommaso.schettini@concordia.ca
#SBATCH --mail-type=FAIL

set -euo pipefail
trap 'echo "[$(date)] Job ${SLURM_ARRAY_TASK_ID} failed with exit code $?"; exit 1' ERR

INSTANCE_FILE="${INSTANCE_FILE:-scripts/instances/periodic_ppo_nyc_day1_fleet20_9infra_3seeds_500ep_instances.tsv}"
LINE=$(awk -v N="$SLURM_ARRAY_TASK_ID" 'NR==N+1' "$INSTANCE_FILE")

if [ -z "$LINE" ]; then
  echo "No configuration found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $INSTANCE_FILE"
  exit 1
fi

IFS=$'\t' read -r CITY SIMULATION HORIZON FLEET INFRASTRUCTURE PYTHON_MODULE NUM_EPISODES SEED LEARNING_RATE DECISION_CLOCK UPDATE_EPOCHS MINIBATCH_DECISIONS EVAL_INTERVAL EVAL_EPISODES EVAL_FIRST_SEED EVAL_FIRST_POLICY_SEED GREEDY_EVAL SAVE_INTERVAL DEVICE OUTPUT_DIR <<< "$LINE"

module purge
module load "$PYTHON_MODULE"
module load scipy-stack
module load proj/9.4.1
if [[ "$DEVICE" == cuda* ]]; then
  module load cuda
fi

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

mkdir -p logs err

GREEDY_EVAL_ARGS=()
if [ "$GREEDY_EVAL" = "1" ]; then
  GREEDY_EVAL_ARGS+=(--greedy-eval)
fi

EXPERIMENT="nyc_day1_fleet20_9infra_3seeds_500ep"
RUN_NAME="periodic_ppo_${EXPERIMENT}_${INFRASTRUCTURE}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Running replicated periodic PPO task ${SLURM_ARRAY_TASK_ID}/27..."
echo "Experiment: $EXPERIMENT"
echo "City: $CITY | Simulation: $SIMULATION | Horizon: $HORIZON | Fleet: $FLEET"
echo "Infrastructure: $INFRASTRUCTURE | Training seed: $SEED"
echo "Python module: $PYTHON_MODULE | Python: $(python --version 2>&1)"
echo "PROJ data: $PROJ_DATA"
echo "Episodes: $NUM_EPISODES | Learning rate: $LEARNING_RATE"
echo "Decision clock: $DECISION_CLOCK | PPO epochs: $UPDATE_EPOCHS | Minibatch: $MINIBATCH_DECISIONS"
echo "Evaluation every $EVAL_INTERVAL episodes on $EVAL_EPISODES fixed seed pairs"
echo "Environment seed: $EVAL_FIRST_SEED | Policy seed: $EVAL_FIRST_POLICY_SEED | Greedy diagnostic: $GREEDY_EVAL"
echo "Device: $DEVICE | GPUs: ${SLURM_GPUS:-${CUDA_VISIBLE_DEVICES:-unknown}}"
echo "Output: ${OUTPUT_DIR}/${RUN_NAME}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export SDL_VIDEODRIVER=dummy

python scripts/utilities/train_periodic_ppo.py \
  --num-episodes "$NUM_EPISODES" \
  --seed "$SEED" \
  --learning-rate "$LEARNING_RATE" \
  --decision-clock "$DECISION_CLOCK" \
  --update-epochs "$UPDATE_EPOCHS" \
  --minibatch-decisions "$MINIBATCH_DECISIONS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-first-seed "$EVAL_FIRST_SEED" \
  --eval-first-policy-seed "$EVAL_FIRST_POLICY_SEED" \
  "${GREEDY_EVAL_ARGS[@]}" \
  --simulation "$SIMULATION" \
  --city "$CITY" \
  --horizon "$HORIZON" \
  --fleet "$FLEET" \
  --infrastructure "$INFRASTRUCTURE" \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --save-interval "$SAVE_INTERVAL"

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} finished."
