#!/bin/bash
#SBATCH --job-name=tsifa_baselines
#SBATCH --output=script_outputs/tsifa_baselines_%j.out
#SBATCH --error=script_outputs/tsifa_baselines_%j.err
#SBATCH --time=04:00:00
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
L2=0.001
SEED=1

echo "$(date -Is) job start kind=baseline_families"
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/$SAVE_NAME"
        echo "$(date -Is) baseline start dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"

        srun python -m ts_ifa.experiments.evaluate_baselines \
          --input-dir "$RUN_DIR" \
          --output-dir "$RUN_DIR/baselines" \
          --family baselines \
          --l2 "$L2" \
          --fit-baselines-on-eval \
          --seed "$SEED"

        echo "$(date -Is) baseline done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
      done
    done
  done
done
echo "$(date -Is) job done kind=baseline_families"
