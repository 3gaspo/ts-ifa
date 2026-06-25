#!/bin/bash
#SBATCH --job-name=tsifa_extract
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tsifa_extract_%j.out
#SBATCH --error=script_outputs/tsifa_extract_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/results"
CHRONOS_WEIGHTS_PATH="../weights/chronos2"
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
DIRECT_MODELS=("chronos")
DISTANCE_SPACES=("instance") # raw instance encoder
NEIGHBORS=(10)

SPLITS="0.3,0.35,0.15,0.2"
RETRIEVAL_MODE="online"
DATASTORE_STRIDE=24
TRAIN_QUERY_STRIDE=24
ORACLE_QUERY_STRIDE=24
EVAL_QUERY_STRIDE=128
PERIOD=24
MAX_STORE_WINDOWS=30000
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future\"}"

echo "$(date -Is) job start kind=extraction"
for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=(--drop-users "$ELECTRICITY_DROP_USERS")
  [ "$dataset" = "solar" ] && DATA_ARGS+=(--aggr sum --aggr-period h)

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
    echo "$(date -Is) setting start dataset=$dataset lags=$L horizon=$H"

    for model in "${DIRECT_MODELS[@]}"; do
      MODEL_ARGS=()
      [ "$model" = "chronos" ] && MODEL_ARGS+=(--model-kwargs "$CHRONOS_KWARGS")
      srun python -m ts_ifa.experiments.experiment_univariate \
        --csv "$DATASET_DIR" \
        --dataset-name "$dataset" \
        "${DATA_ARGS[@]}" \
        --lags "$L" \
        --horizon "$H" \
        --splits "$SPLITS" \
        --eval-stride "$EVAL_QUERY_STRIDE" \
        --model "$model" \
        --normalization instance \
        --device gpu \
        --output-dir "$SETTING_OUT/direct" \
        --save-name "$model" \
        --seed "$SEED" \
        "${MODEL_ARGS[@]}"
    done

    for space in "${DISTANCE_SPACES[@]}"; do
      SPACE_ARGS=()
      [ "$space" = "encoder" ] && SPACE_ARGS+=(--pool-representation)

      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        echo "$(date -Is) extraction start dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
        srun python -m ts_ifa.experiments.extraction \
          --csv "$DATASET_DIR" \
          --dataset-name "$dataset" \
          "${DATA_ARGS[@]}" \
          --lags "$L" \
          --horizon "$H" \
          --splits "$SPLITS" \
          --datastore-stride "$DATASTORE_STRIDE" \
          --train-stride "$TRAIN_QUERY_STRIDE" \
          --oracle-stride "$ORACLE_QUERY_STRIDE" \
          --eval-stride "$EVAL_QUERY_STRIDE" \
          --period "$PERIOD" \
          --neighbors "$neighbors" \
          --distance-space "$space" \
          --distance-metric euclidean \
          --max-store-windows "$MAX_STORE_WINDOWS" \
          --retrieval-mode "$RETRIEVAL_MODE" \
          "${SPACE_ARGS[@]}" \
          --model chronos \
          --model-kwargs "$CHRONOS_KWARGS" \
          --normalization instance \
          --device gpu \
          --output-dir "$SETTING_OUT" \
          --save-name "$SAVE_NAME" \
          --seed "$SEED"
        echo "$(date -Is) extraction done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
      done
    done
    echo "$(date -Is) setting done dataset=$dataset lags=$L horizon=$H"
  done
done
echo "$(date -Is) job done kind=extraction"
