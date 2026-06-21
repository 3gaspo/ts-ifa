#!/bin/bash
#SBATCH --job-name=tsifa_train
#SBATCH --gres=gpu:1
#SBATCH --output=tsifa_train_%j.out
#SBATCH --error=tsifa_train_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/ts_ifa"
CHRONOS_WEIGHTS_PATH="timetensor_oldest/src/timetensor/sota/chronos2/weights"
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"

DATASETS=("electricity") # "traffic" "solar" "exchange_rate"
SETTINGS=("168 24") # "672 168" "1344 336"

SPLITS="0.3,0.5,0.2"
TRAIN_STRIDE=24
EVAL_STRIDE=512
PERIOD=24
NEIGHBORS=10
MAX_TRAIN_WINDOWS=5000
SEED=1

EPOCHS=200
BATCH_SIZE=256
LR=0.001
WEIGHT_DECAY=0.0001
BETA=0.01
GAMMA=0.0

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future\"}"

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=(--drop-users "$ELECTRICITY_DROP_USERS")

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
    SAVE_NAME="chronos_chronos_euclidean_${NEIGHBORS}_online"
    RUN_DIR="$SETTING_OUT/$SAVE_NAME"

    srun python extraction.py \
      --csv "$DATASET_DIR" \
      --dataset-name "$dataset" \
      "${DATA_ARGS[@]}" \
      --lags "$L" \
      --horizon "$H" \
      --splits "$SPLITS" \
      --train-stride "$TRAIN_STRIDE" \
      --eval-stride "$EVAL_STRIDE" \
      --period "$PERIOD" \
      --neighbors "$NEIGHBORS" \
      --distance-space chronos \
      --distance-metric euclidean \
      --max-train-windows "$MAX_TRAIN_WINDOWS" \
      --online \
      --pool-representation \
      --model chronos \
      --model-kwargs "$CHRONOS_KWARGS" \
      --normalization instance \
      --device gpu \
      --output-dir "$SETTING_OUT" \
      --save-name "$SAVE_NAME" \
      --seed "$SEED"

    srun python train_ts_ifa.py \
      --input-dir "$RUN_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --beta "$BETA" \
      --gamma "$GAMMA" \
      --normalization instance \
      --device gpu \
      --seed "$SEED"
  done
done
