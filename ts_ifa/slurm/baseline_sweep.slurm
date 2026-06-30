#!/bin/bash
#SBATCH --job-name=tsifa_base_sweep
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tsifa_base_sweep_%j.out
#SBATCH --error=script_outputs/tsifa_base_sweep_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

SHARED_ROOT="$(cd .. && pwd)"
DATA_ROOT="${SHARED_ROOT}/datasets"
OUT_ROOT="outputs/results"
CHRONOS_WEIGHTS_PATH="${SHARED_ROOT}/weights/chronos2"

DATASETS=("electricity" "solar")
SETTINGS=("168 24" "672 168")
DISTANCE_SPACES=("raw" "instance")
NEIGHBORS=(1 3 10)

SPLITS="0.3,0.35,0.15,0.2"
RETRIEVAL_MODE="online"
DATASTORE_STRIDE=24
TRAIN_QUERY_STRIDE=24
ORACLE_QUERY_STRIDE=24
EVAL_QUERY_STRIDE=128
PERIOD=24
MAX_STORE_WINDOWS=30000
L2=0.001
GATE_ITERATIONS=300
GATE_LEARNING_RATE=0.03
GATE_DEPTH=4
SEED=1

CHRONOS_KWARGS="{\"weights_path\":\"$CHRONOS_WEIGHTS_PATH\",\"device_map\":\"cuda\",\"context_mode\":\"future_included\"}"

echo "$(date -Is) job start kind=baseline_sweep"
for dataset in "${DATASETS[@]}"; do
  DATASET_DIR="$DATA_ROOT/$dataset"
  DATA_CONFIG="$DATASET_DIR/config.json"
  DATA_ARGS=()
  if [ ! -d "$DATASET_DIR" ]; then
    echo "$(date -Is) missing dataset directory dataset=$dataset path=$DATASET_DIR" >&2
    exit 1
  fi
  if [ ! -f "$DATA_CONFIG" ]; then
    echo "$(date -Is) missing dataset config dataset=$dataset path=$DATA_CONFIG" >&2
    exit 1
  fi
  DATA_ARGS+=(--dataset-config "$DATA_CONFIG")

  for setting in "${SETTINGS[@]}"; do
    read -r L H <<< "$setting"
    SETTING_OUT="$OUT_ROOT/$dataset/${L}_${H}"
    echo "$(date -Is) setting start dataset=$dataset lags=$L horizon=$H"

    if [ ! -f "$SETTING_OUT/direct/chronos/univariate_summary.json" ]; then
      echo "$(date -Is) direct start dataset=$dataset lags=$L horizon=$H model=chronos"
      srun python -m ts_ifa.experiments.experiment_univariate \
        --csv "$DATASET_DIR" \
        --dataset-name "$dataset" \
        "${DATA_ARGS[@]}" \
        --lags "$L" \
        --horizon "$H" \
        --splits "$SPLITS" \
        --eval-stride "$EVAL_QUERY_STRIDE" \
        --model chronos \
        --model-kwargs "$CHRONOS_KWARGS" \
        --normalization instance \
        --device gpu \
        --output-dir "$SETTING_OUT/direct" \
        --save-name chronos \
        --seed "$SEED"
      echo "$(date -Is) direct done dataset=$dataset lags=$L horizon=$H model=chronos"
    fi

    for space in "${DISTANCE_SPACES[@]}"; do
      for neighbors in "${NEIGHBORS[@]}"; do
        SAVE_NAME="chronos_${space}_euclidean_${neighbors}_${RETRIEVAL_MODE}"
        RUN_DIR="$SETTING_OUT/$SAVE_NAME"
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
          --model chronos \
          --model-kwargs "$CHRONOS_KWARGS" \
          --normalization instance \
          --device gpu \
          --output-dir "$SETTING_OUT" \
          --save-name "$SAVE_NAME" \
          --seed "$SEED"
        echo "$(date -Is) extraction done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"

        echo "$(date -Is) baselines start dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"
        srun python -m ts_ifa.experiments.evaluate_baselines \
          --input-dir "$RUN_DIR" \
          --output-dir "$RUN_DIR/baselines" \
          --family baselines \
          --l2 "$L2" \
          --fit-baselines-on-eval \
          --seed "$SEED"
        echo "$(date -Is) baselines done dataset=$dataset lags=$L horizon=$H run=$SAVE_NAME"

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
    echo "$(date -Is) setting done dataset=$dataset lags=$L horizon=$H"
  done
done
echo "$(date -Is) job done kind=baseline_sweep"
