#!/bin/bash
#SBATCH --job-name=tsifa_base
#SBATCH --gres=gpu:1
#SBATCH --output=script_outputs/tsifa_base_%j.out
#SBATCH --error=script_outputs/tsifa_base_%j.err
#SBATCH --time=23:00:00
#SBATCH --partition=h100
#SBATCH --mem=40000

set -euo pipefail

mkdir -p script_outputs
[ -f .venv/bin/activate ] && source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON_BIN=${PYTHON_BIN:-python}
SRUN=${SRUN:-srun}
DATA_ROOT=${DATA_ROOT:-../datasets}
OUT_ROOT=${OUT_ROOT:-outputs/slurm_baselines}
DATASETS=${DATASETS:-"electricity"}
SETTINGS=${SETTINGS:-"168:24"}
SPLITS=${SPLITS:-"0.6,0.2,0.2"}
SEED=${SEED:-1}
DEVICE=${DEVICE:-auto}

TARGET_COLS=${TARGET_COLS:-}
PAST_COVARIATE_COLS=${PAST_COVARIATE_COLS:-}
FUTURE_COVARIATE_COLS=${FUTURE_COVARIATE_COLS:-}
DATE_COL=${DATE_COL:-}
DROP_USERS=${DROP_USERS:-}
ELECTRICITY_DROP_USERS=${ELECTRICITY_DROP_USERS:-"57;106;127;182;298;6;113;143"}
RENAME_USERS=${RENAME_USERS:-false}
AGGR=${AGGR:-}
AGGR_PERIOD=${AGGR_PERIOD:-h}

RUN_DIRECT=${RUN_DIRECT:-true}
RUN_NEIGHBOR_BASELINES=${RUN_NEIGHBOR_BASELINES:-true}
RUN_EXTRACTION=${RUN_EXTRACTION:-true}
RUN_FEATURES=${RUN_FEATURES:-true}
RUN_ADAPTER_BASELINES=${RUN_ADAPTER_BASELINES:-true}
DIRECT_MODELS=${DIRECT_MODELS:-"persistence chronos"}
DIRECT_NORMALIZATION=${DIRECT_NORMALIZATION:-none}
EVAL_SPLITS=${EVAL_SPLITS:-"valid,test"}

FOUNDATION_MODELS=${FOUNDATION_MODELS:-"chronos"}
EXTRACTION_NORMALIZATION=${EXTRACTION_NORMALIZATION:-none}
NEIGHBORS_LIST=${NEIGHBORS_LIST:-"1 3 5 10"}
DISTANCE_SPACES=${DISTANCE_SPACES:-"raw model"}
DISTANCE_METRICS=${DISTANCE_METRICS:-"euclidean cosine"}
TRAIN_STRIDE=${TRAIN_STRIDE:-24}
EVAL_STRIDE=${EVAL_STRIDE:-24}
PERIOD=${PERIOD:-24}
MAX_TRAIN_WINDOWS=${MAX_TRAIN_WINDOWS:-}
SEARCH_CHUNK_SIZE=${SEARCH_CHUNK_SIZE:-512}
POOL_REPRESENTATION=${POOL_REPRESENTATION:-true}
ONLINE=${ONLINE:-false}
NO_ALIGN_PERIOD=${NO_ALIGN_PERIOD:-false}
NO_FEATURE_NORMALIZATION=${NO_FEATURE_NORMALIZATION:-false}
COMPUTE_EC=${COMPUTE_EC:-false}

BASELINE_L2=${BASELINE_L2:-1e-3}
ORACLE_TRAIN_FRACTION=${ORACLE_TRAIN_FRACTION:-0.15}
GATE_EPOCHS=${GATE_EPOCHS:-300}
GATE_LR=${GATE_LR:-1e-2}
GATE_WEIGHT_DECAY=${GATE_WEIGHT_DECAY:-1e-3}
TRAIN_HORIZON_GATE=${TRAIN_HORIZON_GATE:-false}

CHRONOS_CONTEXT_MODE=${CHRONOS_CONTEXT_MODE:-past_only}
CHRONOS_WEIGHTS_PATH=${CHRONOS_WEIGHTS_PATH:-}
CHRONOS_MODEL_KWARGS=${CHRONOS_MODEL_KWARGS:-}
PATCHTST_PRETRAINED_PATH=${PATCHTST_PRETRAINED_PATH:-}
LINEAR_PRETRAINED_PATH=${LINEAR_PRETRAINED_PATH:-}
ALLOW_UNTRAINED_PATCHTST=${ALLOW_UNTRAINED_PATCHTST:-false}
ALLOW_UNTRAINED_LINEAR=${ALLOW_UNTRAINED_LINEAR:-false}

truthy() {
  case "${1:-false}" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

run_cmd() {
  if [ -n "$SRUN" ]; then
    "$SRUN" "$@"
  else
    "$@"
  fi
}

dataset_path() {
  local dataset="$1"
  local candidate="${DATA_ROOT}/${dataset}"
  if [ -f "${candidate}.csv" ] && [ ! -d "$candidate" ]; then
    printf "%s\n" "${candidate}.csv"
  else
    printf "%s\n" "$candidate"
  fi
}

build_data_args() {
  local dataset="$1"
  DATA_ARGS=(--csv "$(dataset_path "$dataset")" --dataset-name "$dataset")
  [ -n "$TARGET_COLS" ] && DATA_ARGS+=(--target-cols "$TARGET_COLS")
  [ -n "$PAST_COVARIATE_COLS" ] && DATA_ARGS+=(--past-covariate-cols "$PAST_COVARIATE_COLS")
  [ -n "$FUTURE_COVARIATE_COLS" ] && DATA_ARGS+=(--future-covariate-cols "$FUTURE_COVARIATE_COLS")
  [ -n "$DATE_COL" ] && DATA_ARGS+=(--date-col "$DATE_COL")
  [ -n "$AGGR" ] && DATA_ARGS+=(--aggr "$AGGR" --aggr-period "$AGGR_PERIOD")
  local drop_users="$DROP_USERS"
  [ "$dataset" = "electricity" ] && [ -n "$ELECTRICITY_DROP_USERS" ] && drop_users="$ELECTRICITY_DROP_USERS"
  [ -n "$drop_users" ] && DATA_ARGS+=(--drop-users "$drop_users")
  truthy "$RENAME_USERS" && DATA_ARGS+=(--rename-users)
}

build_model_args() {
  local model="$1"
  MODEL_ARGS=()
  if [ "$model" = "chronos" ]; then
    if [ -n "$CHRONOS_MODEL_KWARGS" ]; then
      MODEL_ARGS+=(--model-kwargs "$CHRONOS_MODEL_KWARGS")
    elif [ -n "$CHRONOS_WEIGHTS_PATH" ]; then
      MODEL_ARGS+=(--model-kwargs "{\"context_mode\":\"${CHRONOS_CONTEXT_MODE}\",\"weights_path\":\"${CHRONOS_WEIGHTS_PATH}\"}")
    else
      MODEL_ARGS+=(--model-kwargs "{\"context_mode\":\"${CHRONOS_CONTEXT_MODE}\"}")
    fi
  elif [ "$model" = "patchtst" ] && [ -n "$PATCHTST_PRETRAINED_PATH" ]; then
    MODEL_ARGS+=(--pretrained-path "$PATCHTST_PRETRAINED_PATH")
  elif [ "$model" = "linear" ] && [ -n "$LINEAR_PRETRAINED_PATH" ]; then
    MODEL_ARGS+=(--pretrained-path "$LINEAR_PRETRAINED_PATH")
  fi
}

for dataset in $DATASETS; do
  build_data_args "$dataset"
  for setting in $SETTINGS; do
    IFS=: read -r L H <<< "$setting"
    echo "dataset=$dataset lags=$L horizon=$H"

    if truthy "$RUN_DIRECT"; then
      direct_out="${OUT_ROOT}/direct/${dataset}/${L}_${H}"
      for model in $DIRECT_MODELS; do
        if [ "$model" = "patchtst" ] && [ -z "$PATCHTST_PRETRAINED_PATH" ] && ! truthy "$ALLOW_UNTRAINED_PATCHTST"; then
          echo "Skipping patchtst direct baseline because PATCHTST_PRETRAINED_PATH is empty."
          continue
        fi
        if [ "$model" = "linear" ] && [ -z "$LINEAR_PRETRAINED_PATH" ] && ! truthy "$ALLOW_UNTRAINED_LINEAR"; then
          echo "Skipping linear direct baseline because LINEAR_PRETRAINED_PATH is empty."
          continue
        fi
        build_model_args "$model"
        run_cmd "$PYTHON_BIN" experiment_univariate.py \
          "${DATA_ARGS[@]}" \
          --lags "$L" \
          --horizon "$H" \
          --splits "$SPLITS" \
          --eval-stride "$EVAL_STRIDE" \
          --eval-splits "$EVAL_SPLITS" \
          --model "$model" \
          --normalization "$DIRECT_NORMALIZATION" \
          --device "$DEVICE" \
          --output-dir "$direct_out" \
          --save-name "$model" \
          --seed "$SEED" \
          "${MODEL_ARGS[@]}"
      done
    fi

    if truthy "$RUN_NEIGHBOR_BASELINES"; then
      extract_out="${OUT_ROOT}/neighbors/${dataset}/${L}_${H}"
      for model in $FOUNDATION_MODELS; do
        build_model_args "$model"
        for distance_space in $DISTANCE_SPACES; do
          for distance_metric in $DISTANCE_METRICS; do
            for neighbors in $NEIGHBORS_LIST; do
              save_name="${model}_${distance_space}_${distance_metric}_k${neighbors}"
              run_dir="${extract_out}/${save_name}"
              if truthy "$RUN_EXTRACTION"; then
                extraction_args=(
                  "${DATA_ARGS[@]}"
                  --lags "$L"
                  --horizon "$H"
                  --splits "$SPLITS"
                  --train-stride "$TRAIN_STRIDE"
                  --eval-stride "$EVAL_STRIDE"
                  --period "$PERIOD"
                  --neighbors "$neighbors"
                  --distance-space "$distance_space"
                  --distance-metric "$distance_metric"
                  --search-chunk-size "$SEARCH_CHUNK_SIZE"
                  --model "$model"
                  --normalization "$EXTRACTION_NORMALIZATION"
                  --device "$DEVICE"
                  --output-dir "$extract_out"
                  --save-name "$save_name"
                  --seed "$SEED"
                  "${MODEL_ARGS[@]}"
                )
                [ -n "$MAX_TRAIN_WINDOWS" ] && extraction_args+=(--max-train-windows "$MAX_TRAIN_WINDOWS")
                truthy "$POOL_REPRESENTATION" && extraction_args+=(--pool-representation)
                truthy "$ONLINE" && extraction_args+=(--online)
                truthy "$NO_ALIGN_PERIOD" && extraction_args+=(--no-align-period)
                truthy "$NO_FEATURE_NORMALIZATION" && extraction_args+=(--no-feature-normalization)
                truthy "$COMPUTE_EC" && extraction_args+=(--compute-ec)
                run_cmd "$PYTHON_BIN" extraction.py "${extraction_args[@]}"
              fi

              if [ -f "${run_dir}/train_prediction_payload.pt" ] && [ -f "${run_dir}/eval_prediction_payload.pt" ]; then
                truthy "$RUN_FEATURES" && run_cmd "$PYTHON_BIN" features.py --input-dir "$run_dir"
                if truthy "$RUN_ADAPTER_BASELINES"; then
                  baseline_args=(
                    --input-dir "$run_dir"
                    --l2 "$BASELINE_L2"
                    --oracle-train-fraction "$ORACLE_TRAIN_FRACTION"
                    --gate-epochs "$GATE_EPOCHS"
                    --gate-lr "$GATE_LR"
                    --gate-weight-decay "$GATE_WEIGHT_DECAY"
                    --seed "$SEED"
                  )
                  truthy "$TRAIN_HORIZON_GATE" && baseline_args+=(--train-horizon-gate)
                  run_cmd "$PYTHON_BIN" evaluate_baselines.py "${baseline_args[@]}"
                fi
              else
                echo "Missing payloads in ${run_dir}; skipping feature and adapter evaluation."
              fi
            done
          done
        done
      done
    fi
  done
done
