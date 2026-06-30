#!/bin/bash
#SBATCH --job-name=tsifa_results
#SBATCH --output=script_outputs/tsifa_results_%j.out
#SBATCH --error=script_outputs/tsifa_results_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=h100-bis
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000

set -euo pipefail
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

OUT_ROOT="outputs/results"
RUN="chronos_instance_euclidean_10_online"
DATASET_ARGS=(
  --datasets electricity,solar
  --dataset-settings electricity=168_24,672_168
  --dataset-settings solar=168_24,672_168
)

MAIN_METHODS="chronos,$RUN/context_forecast,$RUN/horizon_knn_weighted,$RUN/horizon_knn_mean,$RUN/horizon_mix_scalar,$RUN/horizon_ridge_shared,$RUN/residual_knn_weighted,$RUN/residual_mix_scalar,$RUN/residual_ridge_shared,$RUN/residual_ridge_horizon,$RUN/full_ridge_horizon,$RUN/bayes_context_scalar,$RUN/bayes_context_horizon,$RUN/catboost_context_classifier_scalar,$RUN/catboost_context_classifier_horizon,$RUN/catboost_context_regressor_scalar,$RUN/catboost_context_regressor_horizon,$RUN/TS-IFA"
BASELINE_METHODS="chronos,$RUN/context_forecast,$RUN/horizon_knn_weighted,$RUN/horizon_knn_mean,$RUN/horizon_mix_scalar,$RUN/horizon_ridge_shared,$RUN/residual_knn_weighted,$RUN/residual_mix_scalar,$RUN/residual_ridge_shared,$RUN/residual_ridge_horizon,$RUN/full_ridge_horizon,$RUN/horizon_mix_scalar_eval_fit,$RUN/horizon_ridge_shared_eval_fit,$RUN/residual_mix_scalar_eval_fit,$RUN/residual_ridge_shared_eval_fit,$RUN/residual_ridge_horizon_eval_fit,$RUN/full_ridge_horizon_eval_fit"
GATE_METHODS="chronos,$RUN/context_forecast,$RUN/bayes_context_scalar,$RUN/bayes_context_horizon,$RUN/catboost_context_classifier_scalar,$RUN/catboost_context_classifier_horizon,$RUN/catboost_context_regressor_scalar,$RUN/catboost_context_regressor_horizon,$RUN/oracle_context_scalar,$RUN/oracle_context_horizon"
TS_IFA_METHODS="chronos,$RUN/TS-IFA"

echo "$(date -Is) job start kind=results_tables"
srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/results.tex" \
  --metric nmse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$MAIN_METHODS" \
  --reference chronos \
  --caption "Main nMSE forecasting results" \
  --label tab:results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/results_positive.tex" \
  --metric nmse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$MAIN_METHODS" \
  --reference chronos \
  --positive-only \
  --caption "Main nMSE forecasting results with positive overall improvement" \
  --label tab:results-positive

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/baselines_results.tex" \
  --metric nmse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$BASELINE_METHODS" \
  --reference chronos \
  --exclude-from-bold horizon_mix_scalar_eval_fit,horizon_ridge_shared_eval_fit,residual_mix_scalar_eval_fit,residual_ridge_shared_eval_fit,residual_ridge_horizon_eval_fit,full_ridge_horizon_eval_fit \
  --caption "Baseline nMSE results and optimistic T3-fitted bounds" \
  --label tab:baselines-results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/gates_results.tex" \
  --metric nmse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$GATE_METHODS" \
  --reference chronos \
  --caption "Gate nMSE results and ground-truth-informed oracles" \
  --label tab:gates-results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/ts_ifa_results.tex" \
  --metric nmse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$TS_IFA_METHODS" \
  --reference chronos \
  --caption "TS-IFA nMSE forecasting results" \
  --label tab:ts-ifa-results
echo "$(date -Is) job done kind=results_tables output=$OUT_ROOT"
