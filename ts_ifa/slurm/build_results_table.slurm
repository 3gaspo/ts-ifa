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

MAIN_METHODS="chronos,$RUN/context_conditioned,$RUN/neighbor_weighted_mean,$RUN/neighbor_unweighted_mean,$RUN/pred_plus_weighted_e,$RUN/mix_0_weighted,$RUN/mix_1_learned,$RUN/mix_2_full_horizon,$RUN/gated_context_classifier_scalar,$RUN/gated_context_classifier_horizon,$RUN/gated_context_regressor_scalar,$RUN/gated_context_regressor_horizon,$RUN/TS-IFA"
BASELINE_METHODS="chronos,$RUN/context_conditioned,$RUN/neighbor_weighted_mean,$RUN/neighbor_unweighted_mean,$RUN/pred_plus_weighted_e,$RUN/mix_0_weighted,$RUN/mix_1_learned,$RUN/mix_2_full_horizon,$RUN/mix_0_weighted_eval_fit,$RUN/mix_1_learned_eval_fit,$RUN/mix_2_full_horizon_eval_fit"
GATE_METHODS="chronos,$RUN/context_conditioned,$RUN/gated_context_classifier_scalar,$RUN/gated_context_classifier_horizon,$RUN/gated_context_regressor_scalar,$RUN/gated_context_regressor_horizon,$RUN/oracle_context_scalar,$RUN/oracle_context_horizon"
TS_IFA_METHODS="chronos,$RUN/TS-IFA"

echo "$(date -Is) job start kind=results_tables"
srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/results.tex" \
  --metric mse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$MAIN_METHODS" \
  --reference chronos \
  --caption "Main forecasting results" \
  --label tab:results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/baselines_results.tex" \
  --metric mse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$BASELINE_METHODS" \
  --reference chronos \
  --exclude-from-bold mix_0_weighted_eval_fit,mix_1_learned_eval_fit,mix_2_full_horizon_eval_fit \
  --caption "Baseline forecasting results and optimistic T3-fitted bounds" \
  --label tab:baselines-results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/gates_results.tex" \
  --metric mse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$GATE_METHODS" \
  --reference chronos \
  --caption "Gate forecasting results and ground-truth-informed oracles" \
  --label tab:gates-results

srun python -m ts_ifa.results_table \
  "$OUT_ROOT" \
  --output "$OUT_ROOT/ts_ifa_results.tex" \
  --metric mse \
  --split eval \
  "${DATASET_ARGS[@]}" \
  --methods "$TS_IFA_METHODS" \
  --reference chronos \
  --caption "TS-IFA forecasting results" \
  --label tab:ts-ifa-results
echo "$(date -Is) job done kind=results_tables output=$OUT_ROOT"
