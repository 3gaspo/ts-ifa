#!/bin/bash
#SBATCH --job-name=tsifa_baselines
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tsifa_baselines_%j.out
#SBATCH --error=script_outputs/tsifa_baselines_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

DATA_ROOT="../datasets"
OUT_ROOT="outputs/results"
CHRONOS_WEIGHTS_PATH="../weights/chronos2"
ELECTRICITY_DROP_USERS="57;106;127;182;298;6;113;143"

DATASETS=("electricity") # "traffic" "solar" "exchange_rate"
SETTINGS=("168 24") # "672 168" "1344 336"
MODELS=("chronos")
DISTANCE_SPACES=("raw")
NEIGHBORS=(3) #1 3 10)

SPLITS="0.3,0.35,0.15,0.2"
RETRIEVAL_MODE="online"
TRAIN_STRIDE=24
EVAL_STRIDE=512
PERIOD=24
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future\"}"

echo "$(date -Is) job start kind=baseline_sweep"
for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_ARGS=()
  [ "$dataset" = "electricity" ] && DATA_ARGS+=(--drop-users "$ELECTRICITY_DROP_USERS")

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
    echo "$(date -Is) setting start dataset=$dataset lags=$L horizon=$H"

    for model in "${MODELS[@]}"; do
      MODEL_ARGS=()
      [ "$model" = "chronos" ] && MODEL_ARGS+=(--model-kwargs "$CHRONOS_KWARGS")
      srun python -m ts_ifa.experiments.experiment_univariate \
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
      MAX_STORE_WINDOWS=30000
      SPACE_ARGS=()
      if [ "$space" = "chronos" ]; then
        MAX_STORE_WINDOWS=15000
        SPACE_ARGS+=(--pool-representation)
      fi

      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/$SAVE_NAME"

        srun python -m ts_ifa.experiments.extraction \
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

        srun python -m ts_ifa.experiments.evaluate_baselines \
          --input-dir "$RUN_DIR" \
          --train-horizon-gate \
          --seed "$SEED"
      done
    done
    echo "$(date -Is) setting done dataset=$dataset lags=$L horizon=$H"
  done
done
srun python -m ts_ifa.results_table "$OUT_ROOT" --metric mse --split eval --reference chronos
echo "$(date -Is) job done kind=baseline_sweep"
