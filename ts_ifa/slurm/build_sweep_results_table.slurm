#!/bin/bash
#SBATCH --job-name=tsifa_sweep_tables
#SBATCH --output=script_outputs/tsifa_sweep_tables_%j.out
#SBATCH --error=script_outputs/tsifa_sweep_tables_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

OUT_ROOT="outputs/results"
OUTPUT_DIR="$OUT_ROOT/sweep_tables"

DATASETS=("electricity" "solar")
SETTINGS=("168_24" "672_168")
DISTANCE_SPACES=("raw" "instance")
NEIGHBORS=(1 3 10)
RETRIEVAL_MODE="online"

DATASET_ARG="$(IFS=,; echo "${DATASETS[*]}")"
SETTING_ARG="$(IFS=,; echo "${SETTINGS[*]}")"
SPACE_ARG="$(IFS=,; echo "${DISTANCE_SPACES[*]}")"
NEIGHBOR_ARG="$(IFS=,; echo "${NEIGHBORS[*]}")"

echo "$(date -Is) job start kind=sweep_results_tables datasets=$DATASET_ARG settings=$SETTING_ARG spaces=$SPACE_ARG neighbors=$NEIGHBOR_ARG mode=$RETRIEVAL_MODE"
srun python -m ts_ifa.sweep_results_table \
  "$OUT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --metric nmse \
  --split eval \
  --datasets "$DATASET_ARG" \
  --settings "$SETTING_ARG" \
  --spaces "$SPACE_ARG" \
  --neighbors "$NEIGHBOR_ARG" \
  --retrieval-mode "$RETRIEVAL_MODE" \
  --decimals 2
echo "$(date -Is) job done kind=sweep_results_tables output=$OUTPUT_DIR"
