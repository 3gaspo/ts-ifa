#!/bin/bash
#SBATCH --job-name=tsifa_baselines
#SBATCH --gres=gpu:1
#SBATCH --output=tsifa_baselines_%j.out
#SBATCH --error=tsifa_baselines_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/baselines"
CHRONOS_WEIGHTS_PATH="timetensor_oldest/src/timetensor/sota/chronos2/weights"
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"

DATASETS=("electricity") # "traffic" "solar" "exchange_rate"
SETTINGS=("168 24") # "672 168" "1344 336"
MODELS=("persistence" "chronos")
DISTANCE_SPACES=("raw" "chronos")
NEIGHBORS=(1 3 10)

SPLITS="0.3,0.5,0.2"
TRAIN_STRIDE=24
EVAL_STRIDE=512
PERIOD=24
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future\"}"

for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=(--drop-users "$ELECTRICITY_DROP_USERS")

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"

    for model in "${MODELS[@]}"; do
      MODEL_ARGS=()
      [ "$model" = "chronos" ] && MODEL_ARGS+=(--model-kwargs "$CHRONOS_KWARGS")
      srun python experiment_univariate.py \
        --csv "$DATASET_DIR" \
        --dataset-name "$dataset" \
        "${DATA_ARGS[@]}" \
        --lags "$L" \
        --horizon "$H" \
        --splits "$SPLITS" \
        --eval-stride "$EVAL_STRIDE" \
        --model "$model" \
        --normalization instance \
        --device gpu \
        --output-dir "$SETTING_OUT/direct" \
        --save-name "$model" \
        --seed "$SEED" \
        "${MODEL_ARGS[@]}"
    done

    for space in "${DISTANCE_SPACES[@]}"; do
      MAX_TRAIN_WINDOWS=10000
      SPACE_ARGS=()
      if [ "$space" = "chronos" ]; then
        MAX_TRAIN_WINDOWS=5000
        SPACE_ARGS+=(--pool-representation)
      fi

      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_online"
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
          --neighbors "$neighbors" \
          --distance-space "$space" \
          --distance-metric euclidean \
          --max-train-windows "$MAX_TRAIN_WINDOWS" \
          --online \
          "${SPACE_ARGS[@]}" \
          --model chronos \
          --model-kwargs "$CHRONOS_KWARGS" \
          --normalization instance \
          --device gpu \
          --output-dir "$SETTING_OUT" \
          --save-name "$SAVE_NAME" \
          --seed "$SEED"

        srun python evaluate_baselines.py \
          --input-dir "$RUN_DIR" \
          --train-horizon-gate \
          --seed "$SEED"
      done
    done
  done
done
