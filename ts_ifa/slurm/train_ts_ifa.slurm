#!/bin/bash
#SBATCH --job-name=tsifa_train
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tsifa_train_%j.out
#SBATCH --error=script_outputs/tsifa_train_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
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
SEED=1

EPOCHS=1000
BATCH_SIZE=256
LR=0.00001
WEIGHT_DECAY=0.0001
BETA=0.01
GAMMA=0.01

echo "$(date -Is) job start kind=ts_ifa_training"
for dataset in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/$SAVE_NAME"
        echo "$(date -Is) training start dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"

        srun python -m ts_ifa.experiments.train_ts_ifa \
          --input-dir "$RUN_DIR" \
          --output-dir "$RUN_DIR/ts_ifa" \
          --epochs "$EPOCHS" \
          --batch-size "$BATCH_SIZE" \
          --lr "$LR" \
          --weight-decay "$WEIGHT_DECAY" \
          --beta "$BETA" \
          --gamma "$GAMMA" \
          --normalization instance \
          --device gpu \
          --seed "$SEED"

        echo "$(date -Is) training done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
      done
    done
  done
done
echo "$(date -Is) job done kind=ts_ifa_training"
