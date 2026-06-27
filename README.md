# Lightweight Extraction

This repository contains Python tools for retrieval-based residual adaptation experiments in time-series forecasting. It includes data loading, forecasting baselines, neighbor retrieval, feature extraction, and diagnostics.

## File Map

- `ts_ifa/data/`: CSV loading, split helpers, aligned windows, representations, and exact KNN search.
- `ts_ifa/models/`: common forecast wrapper, persistence/linear baselines, Chronos, PatchTST, and the TS-IFA adapter.
- `ts_ifa/experiments/`: extraction, direct evaluation, payload baselines, feature analysis, TS-IFA training, and shared logging.
- `ts_ifa/visu/`: plotting helpers and the artifact-only retrieval dashboard notebook.
- `ts_ifa/slurm/`: cluster scripts for extraction, baselines, gates, TS-IFA training, and final LaTeX tables.
- `tests/smoke/`: tiny CSV fixture and load/inference checks for local or cluster sanity tests.

## Data Flow

Input data is a date-indexed CSV. When a dataset directory or CSV has a sibling
`config.json`, CSV-loading options such as `drop_users`, `aggr`, and
`aggr_period` are applied automatically; `--dataset-config` can point to a
different JSON file. Target user columns become `dataset.frame`; retrieved
windows from other users provide the same physical quantity as context during
inference. A window is represented as:

- `x`: `(users, 1, lags)`
- `y`: `(users, 1, horizon)`

`ts_ifa.experiments.experiment_univariate` loads a model and evaluates direct predictions on T3. `ts_ifa.experiments.extraction` builds query windows, aligned datastores, representations, neighbors, and separate T1/T2/T3 payloads. Baseline mixtures fit on T1, the context-conditioned gate fits on T2, and all final baseline metrics use T3. TS-IFA trains only on T1, evaluates T2 as a validation curve after every optimizer step, and saves final metrics on T3.

## Temporal Protocol

The default chronological ratios are `0.30,0.35,0.15,0.20`:

- T0: datastore reference period.
- T1: baseline-mixture training and TS-IFA training.
- T2: context-gate training and TS-IFA validation.
- T3: untouched final evaluation for direct baselines, mixtures, gates, and TS-IFA.

Retrieval defaults to L2 (`--distance-metric euclidean`) with `--distance-space instance`, which instance-normalizes every time-domain lookback independently. The other spaces are `raw`, for unnormalized lookbacks, and `encoder`, for the loaded forecasting model's representation. Encoder retrieval currently recomputes datastore representations for every query date and is therefore substantially more expensive; raw and instance retrieval do not call the model during neighbor search. Retrieval also defaults to `--retrieval-mode online`. Each query uses its most recent aligned history, capped by default to the number of aligned dates available in T0; this makes online datastore size comparable to fixed mode. `--retrieval-mode fixed` makes T1, T2, and T3 use only T0. `--datastore-stride` controls aligned retrieval datastore density and must be a multiple of `--period` while period alignment is enabled, so all store windows remain in the same phase as the query. `--train-stride`, `--oracle-stride`, and `--eval-stride` separately control T1, T2, and T3 query density. `--min-store-dates`, `--max-store-dates`, and `--max-store-windows` control capacity; `--full-online-history` removes the date cap, while `--store-start-date` and `--store-end-date` bound history explicitly. Query dates that do not satisfy the minimum datastore size are excluded.

The default TS-IFA adapter uses three single cross-attention blocks and two-layer MLPs with width 128. These dimensions are lightweight heuristic defaults, not tuned constants; all widths and attention sizes are exposed as CLI options. Its final adaptation wraps the learned mixture proposal in an explicit residual gate from the vanilla forecast, initialized near identity. Extraction and training logs report total and trainable parameter counts for the loaded forecaster and TS-IFA respectively; the TS-IFA counts are also saved in its checkpoint and `config.json`.

## Smoke Checks

Install the runtime dependencies used by the scripts (`torch`, `numpy`, `pandas`, `matplotlib`, and `einops`) in the environment where you run them.

Use the small fixture under `tests/smoke/` to validate loading and model construction:

```powershell
python tests/smoke/check_loads.py
python tests/smoke/check_loads.py --check-patchtst
python tests/smoke/check_loads.py --chronos-weights ../weights/chronos2
python tests/smoke/check_baseline_oracles.py
python tests/smoke/check_ts_ifa_training.py
python tests/smoke/check_retrieval_dashboard.py
```

These checks validate CSV parsing, window shapes, persistence inference, optional PatchTST construction, optional Chronos loading, the TS-IFA training path, and dashboard loading/metrics on synthetic payloads.

## Experiment Commands

Run a no-neighbor baseline:

```powershell
python -m ts_ifa.experiments.experiment_univariate --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model persistence --normalization none --eval-stride 24 --output-dir outputs/results --save-name electricity_persistence
```

Run neighbor extraction:

```powershell
python -m ts_ifa.experiments.extraction --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model chronos --model-kwargs '{"weights_path":"../weights/chronos2","context_mode":"future_included"}' --neighbors 5 --distance-space encoder --pool-representation --distance-metric cosine --datastore-stride 24 --train-stride 24 --oracle-stride 24 --eval-stride 24 --period 24 --output-dir outputs/results --save-name electricity_chronos_k5
```

Add `--compute-ec` only when you also need neighbor-context residuals in the payload; it adds extra model forwards.

Analyze an extraction run:

```powershell
python -m ts_ifa.experiments.features --input-dir outputs/results/electricity_chronos_k5
```

Train TS-IFA from extracted payloads:

```powershell
python -m ts_ifa.experiments.train_ts_ifa --input-dir outputs/results/electricity_chronos_k5 --epochs 10000 --batch-size 256 --lr 1e-5 --gamma 1e-2 --normalization instance
```

Evaluate ridge and neighbor baselines from completed payloads:

```powershell
python -m ts_ifa.experiments.evaluate_baselines --input-dir outputs/results/electricity_chronos_k5 --family baselines --fit-baselines-on-eval
```

Evaluate the four learned gates and the two target-aware oracles separately:

```powershell
python -m ts_ifa.experiments.evaluate_baselines --input-dir outputs/results/electricity_chronos_k5 --family gates
```

The two evaluator families write `baselines/visualization_payload.pt` and
`gates/visualization_payload.pt`. The latter contains gate scores and targets
for every extracted split. TS-IFA training writes
`ts_ifa/eval_predictions.pt` alongside its aggregate metrics. These compact
artifacts let visualization code inspect trained outputs without refitting models
or reconstructing experiment splits.

`--fit-baselines-on-eval` additionally fits the three trainable mixtures directly
on T3 and reports their in-sample scores with an `_eval_fit` suffix. These are
optimistic diagnostic bounds, not valid held-out results.

## Interactive Dashboard

Open `ts_ifa/visu/retrieval_dashboard.ipynb` after extraction and model-family
evaluation. Set `CLUSTER_RUN_DIR` to the extraction run directory when using the
cluster. On Colab, set `COLAB_RUN_DIR` and `COLAB_REPO_DIR`; the environment cell
mounts Google Drive and installs the project in editable, no-dependency mode.
The same cell installs `ipywidgets` only when it is missing from the active
notebook environment.

The notebook reads the saved extraction, baseline, and optional TS-IFA artifacts.
It can sample query/retrieval examples, display horizon-wise raw predictions and
error comparisons, and plot gate ROC curves and decision accuracy. It performs
no feature construction, split construction, model fitting, or model inference.

## SLURM Experiments

Edit the literal config block near the top of each script, especially
`DATASETS` and `SETTINGS`, then submit from the repository root. `SHARED_ROOT`
defaults to `..`, the parent folder that contains both `datasets/` and
`weights/`; Chronos weights are expected under
`${SHARED_ROOT}/weights/chronos2/`. Model results are written under
`outputs/results/`, while SLURM stdout and stderr are written under
`script_outputs/`.

First evaluate direct forecasts and create the shared retrieval payloads:

```bash
sbatch ts_ifa/slurm/extract_payloads.slurm
```

After extraction succeeds, the three model-family jobs can run concurrently:

```bash
sbatch ts_ifa/slurm/evaluate_baselines.slurm
sbatch ts_ifa/slurm/evaluate_gates.slurm
sbatch ts_ifa/slurm/train_ts_ifa.slurm
```

Finally, build the table only after all desired family jobs have completed:

```bash
sbatch ts_ifa/slurm/build_results_table.slurm
```

The current scripts run electricity and hourly-summed solar for L-H settings
168-24 and 672-168. Dataset-specific CSV options are read from
`${SHARED_ROOT}/datasets/<dataset>/config.json`; electricity drops the
configured legacy series and solar uses hourly `sum` aggregation. All runs use Chronos,
instance-normalized L2 retrieval,
10 neighbors, online retrieval, 24-step T1/T2 query strides, a 128-step T3
evaluation stride, and a 30,000-window datastore cap with a 24-step aligned
datastore stride. The extraction job is the only job that loads Chronos and builds T1/T2/T3
retrieval payloads. It also evaluates and plots the configured direct models on
T3. Downstream jobs only read those payloads and write to separate `baselines/`,
`gates/`, and `ts_ifa/` folders, so they do not overwrite one another.

Baseline mixtures fit on T1. Gates fit on T2 and are evaluated only on T3. Four
CatBoost gates are reported: binary improvement classifiers and signed
loss-improvement regressors, each in scalar and horizon-wise form. Classifiers
use balanced class weights and select context when the predicted improvement
probability exceeds 0.5. Regressors select context when the predicted
vanilla-minus-context loss is positive. The scalar models make one decision per
forecast; horizon-wise models make one decision per forecast step. The two true
oracles apply the corresponding decisions using T3 targets.

All learned gates share the existing retrieval features. Scalar gates receive
the signed context-minus-vanilla horizon mean and standard deviation, while
horizon gates receive the complete signed horizon vector. Both also receive 13
features covering signed neighbor differences, raw query/neighbor lookback
moments, same-user ratio, mean neighbor age, neighbor-weight concentration, and
mean retrieval distance. Ridge inputs are RMS-standardized without centering,
and their normal equations are averaged over observations so `--l2` has stable
strength across dataset units and payload sizes. The mix0 ridge coefficient is
projected to `[0, 1]` with `np.clip`, so the reported mix0 prediction is always
a convex interpolation between the vanilla forecast and the weighted-neighbor
forecast.

Before retrieved examples are given to the forecasting model, baselines, or TS-IFA, the neighbor lookback statistics transfer their lookbacks, horizons, and forecasts onto the query lookback's level and scale. Residuals receive the scale transform only, since their additive level cancels. TS-IFA then instance-normalizes all query-scale tensors with the query statistics by default and computes its prediction, vanilla-regularization, and residual-supervision losses in that normalized space. With `--normalization none`, losses are still scaled by the query lookback standard deviation. The TS-IFA job trains for 10000 epochs, samples random T1 date/user examples for one optimizer step per epoch, evaluates the full deterministic T2 validation payload every epoch, evaluates T3 once after training, and writes `ts_ifa/eval_metrics.json` plus `ts_ifa/training_nmse.pdf`.

The dedicated result job writes five nMSE tables:

- `results.tex`: held-out methods from all three families; excludes gate oracles
  and baselines fitted directly on T3.
- `results_positive.tex`: the same main-method selection, filtered to methods
  with positive overall improvement over Chronos.
- `baselines_results.tex`: baseline methods plus the optimistic T3-fitted mixture
  bounds, separated from regular methods and excluded from best-value bolding.
- `gates_results.tex`: all four learned gates plus the scalar and horizon-wise
  ground-truth-informed oracles.
- `ts_ifa_results.tex`: Chronos and TS-IFA.

The result loader combines direct
`univariate_summary.json`, adapter `baseline_metrics.json`, gate
`gate_metrics.json`, and
`ts_ifa/eval_metrics.json` artifacts. Retrieval-dependent columns are qualified
by retrieval setting so equally named baselines remain distinct. Table labels
are shortened by default: for example, `chronos_instance_euclidean_10_online/mix_1_learned`
is displayed as `IN_L2_10/mix1`, while an explicitly unnormalized run named
`chronos_raw_euclidean_10_fixed` is displayed as `raw_L2_10_fixed`. Run-specific
`vanilla` columns are hidden by default.

Generate or regenerate a table independently with:

```bash
python -m ts_ifa.results_table outputs/results \
  --metric nmse --split eval \
  --datasets electricity,traffic \
  --dataset-settings electricity=168_24,672_168 \
  --methods chronos,chronos_instance_euclidean_1_online/linear_mix \
  --reference chronos --decimals 2
```

By default, the best value per row is bold, each row has an explicit automatic
power-of-ten scale, and dataset, per-L-H, and overall percentage improvements
are shown. Each summary percentage is computed from the methods' average metric
values, not by averaging individual percentages. These can independently be disabled with `--no-bold`,
`--no-dataset-improvements`, `--no-setting-improvements`, and
`--no-overall-improvement`. Global `--settings`, repeatable dataset-specific
`--dataset-settings`, ordered `--methods`, `--higher-is-better`,
`--positive-only`, `--no-auto-scale`, `--scale-exponent`, repeatable
`--row-scale DATASET/L_H=EXPONENT`, `--caption`, `--label`, and `--output` are
also available. The scalar and horizon-wise true context oracles use T3 targets
to select between vanilla and context-conditioned predictions. Their columns
are automatically moved to the right behind a vertical rule and excluded from
best-value bolding. `--exclude-from-bold` applies the same treatment to other
method IDs or variant names, and `--long-method-names` restores artifact names.
The generated table uses the LaTeX `booktabs`, `multirow`, and `graphicx`
packages.

Experiment entry points surround each run with a shared separator, log their
identity once, then emit concise stage start/completion messages, throttled
training progress, output location, and total runtime. They do not print full
repeated configurations.

## Outputs

Write generated model artifacts under `outputs/results/`; reserve `outputs/hydra/` for Hydra job metadata. Typical files include `*_prediction_payload.pt`, `baseline_metrics.csv`, `visualization_payload.pt`, `ts_ifa.pt`, `eval_predictions.pt`, `training_nmse.pdf`, summary CSV/JSON files, and plots under `plots/`. Do not commit datasets, model weights, cluster outputs, or machine-specific paths.
