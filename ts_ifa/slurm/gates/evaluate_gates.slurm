#!/bin/bash
#SBATCH --job-name=tsifa_gates
#SBATCH --output=script_outputs/tsifa_gates_%j.out
#SBATCH --error=script_outputs/tsifa_gates_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

OUT_ROOT="outputs/results"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
DISTANCE_SPACES=("instance") # raw instance encoder
NEIGHBORS=(10)
RETRIEVAL_MODE="online"

GATE_ITERATIONS=300
GATE_LEARNING_RATE=0.03
GATE_DEPTH=4
SEED=1

echo "$(date -Is) job start kind=gate_families"
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/$SAVE_NAME"
        echo "$(date -Is) gates start dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"

        srun python -m ts_ifa.experiments.evaluate_baselines \
          --input-dir "$RUN_DIR" \
          --output-dir "$RUN_DIR/gates" \
          --family gates \
          --gate-iterations "$GATE_ITERATIONS" \
          --gate-learning-rate "$GATE_LEARNING_RATE" \
          --gate-depth "$GATE_DEPTH" \
          --seed "$SEED"

        echo "$(date -Is) gates done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
      done
    done
  done
done
echo "$(date -Is) job done kind=gate_families"
