# Lightweight Extraction

This repository contains Python tools for retrieval-based residual adaptation experiments in time-series forecasting. It includes data loading, forecasting baselines, neighbor retrieval, feature extraction, and diagnostics.

## File Map

- `ts_ifa/data/`: CSV loading, split helpers, aligned windows, representations, and exact KNN search.
- `ts_ifa/models/`: common forecast wrapper, persistence/linear baselines, Chronos, PatchTST, and the TS-IFA adapter.
- `ts_ifa/experiments/`: extraction, direct evaluation, payload baselines, feature analysis, TS-IFA training, and shared logging.
- `ts_ifa/visu/`: plotting helpers and exploratory notebooks.
- `ts_ifa/compat/`: temporary compatibility imports for previous module names.
- `ts_ifa/slurm/baselines/`: cluster jobs for direct and payload baseline evaluation.
- `ts_ifa/slurm/training/`: cluster jobs for TS-IFA extraction, training, and evaluation.
- `tests/smoke/`: tiny CSV fixture and load/inference checks for local or cluster sanity tests.

## Data Flow

Input data is a date-indexed CSV. Target user columns become `dataset.frame`; optional past and future covariate columns become shared covariate tensors. A window is represented as:

- `x`: `(users, 1, lags)`
- `y`: `(users, 1, horizon)`
- past covariates: `(1, channels, lags)`
- future covariates: `(1, channels, horizon)`

`ts_ifa.experiments.experiment_univariate` loads a model and evaluates direct predictions on T3. `ts_ifa.experiments.extraction` builds query windows, aligned datastores, representations, neighbors, and separate T1/T2/T3 payloads. Baseline mixtures fit on T1, the context-conditioned gate fits on T2, and all final baseline metrics use T3. TS-IFA trains on T1+T2 without validation and evaluates once on T3.

## Temporal Protocol

The default chronological ratios are `0.30,0.35,0.15,0.20`:

- T0: datastore reference period.
- T1: baseline-mixture training and the first part of TS-IFA training.
- T2: context-gate training and the second part of TS-IFA training.
- T3: untouched final evaluation for direct baselines, mixtures, gates, and TS-IFA.

Retrieval defaults to L2 (`--distance-metric euclidean`) on time-domain lookbacks that are instance-normalized independently per window. This setting is labeled `IN_L2`; `raw_L2` is reserved for distances on unnormalized values. `--no-feature-normalization` selects the latter and is separate from the forecasting model's `--normalization` option. Retrieval also defaults to `--retrieval-mode online`. Each query uses its most recent aligned history, capped by default to the number of aligned dates available in T0; this makes online datastore size comparable to fixed mode. `--retrieval-mode fixed` makes T1, T2, and T3 use only T0. `--min-store-dates`, `--max-store-dates`, and `--max-store-windows` control capacity; `--full-online-history` removes the date cap, while `--store-start-date` and `--store-end-date` bound history explicitly. Query dates that do not satisfy the minimum datastore size are excluded.

The default TS-IFA adapter uses three single cross-attention blocks and two-layer MLPs with width 128. Extraction and training logs report total and trainable parameter counts for the loaded forecaster and TS-IFA respectively; the TS-IFA counts are also saved in its checkpoint and `config.json`.

## Smoke Checks

Install the runtime dependencies used by the scripts (`torch`, `numpy`, `pandas`, `matplotlib`, and `einops`) in the environment where you run them.

Use the small fixture under `tests/smoke/` to validate loading and model construction:

```powershell
python tests/smoke/check_loads.py
python tests/smoke/check_loads.py --check-patchtst
python tests/smoke/check_loads.py --chronos-weights ../weights/chronos2
python tests/smoke/check_baseline_oracles.py
python tests/smoke/check_ts_ifa_training.py
```

These checks validate CSV parsing, window shapes, persistence inference, optional PatchTST construction, optional Chronos loading, and the TS-IFA training path on synthetic payloads.

## Experiment Commands

Run a no-neighbor baseline:

```powershell
python -m ts_ifa.experiments.experiment_univariate --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model persistence --normalization none --eval-stride 24 --output-dir outputs/results --save-name electricity_persistence
```

Run neighbor extraction:

```powershell
python -m ts_ifa.experiments.extraction --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model chronos --model-kwargs '{"weights_path":"../weights/chronos2","context_mode":"past_only"}' --neighbors 5 --distance-space chronos --pool-representation --distance-metric cosine --train-stride 24 --eval-stride 24 --period 24 --output-dir outputs/results --save-name electricity_chronos_k5
```

Add `--compute-ec` only when you also need neighbor-context residuals in the payload; it adds extra model forwards.

Analyze an extraction run:

```powershell
python -m ts_ifa.experiments.features --input-dir outputs/results/electricity_chronos_k5
```

Train TS-IFA from extracted payloads:

```powershell
python -m ts_ifa.experiments.train_ts_ifa --input-dir outputs/results/electricity_chronos_k5 --epochs 20 --batch-size 256 --lr 1e-3 --normalization instance
```

Evaluate payload baselines:

```powershell
python -m ts_ifa.experiments.evaluate_baselines --input-dir outputs/results/electricity_chronos_k5 --train-horizon-gate
```

## SLURM Experiments

Edit the literal config block near the top of each script, especially `DATASETS` and `SETTINGS`, then submit from the repository root. Shared Chronos weights are expected under `../weights/chronos2/`. Model results are written under `outputs/results/`, while SLURM stdout and stderr are written under `script_outputs/`.

Evaluate direct forecasts and all payload baselines:

```bash
sbatch ts_ifa/slurm/baselines/evaluate_baselines.slurm
```

Submit TS-IFA extraction, training, and evaluation:

```bash
sbatch ts_ifa/slurm/training/train_ts_ifa.slurm
```

The baseline job loops over datasets, lag/horizon settings, retrieval spaces, and neighbor counts. It evaluates direct persistence/Chronos forecasts on T3, extracts Chronos payloads, fits ridge-regularized mixtures on T1, fits CatBoost regressors on T2 to predict vanilla-minus-context loss, and evaluates on T3. A gate selects the context forecast when its predicted loss advantage is positive. The scalar gate receives the signed context-minus-vanilla horizon mean and standard deviation, while horizon gates receive the complete signed horizon vector. Both share 13 retrieval features covering signed neighbor differences, raw query/neighbor lookback moments, same-user ratio, mean neighbor age, neighbor-weight concentration, and mean retrieval distance. Ridge inputs are RMS-standardized without centering and its normal equations are averaged over observations, so `--l2` has stable strength across dataset units and payload sizes. The current jobs cap raw and Chronos-representation datastores at 30,000 and 15,000 windows respectively and use an evaluation stride of 128.

Before retrieved examples are given to the forecasting model, baselines, or TS-IFA, the neighbor lookback statistics transfer their lookbacks, horizons, and forecasts onto the query lookback's level and scale. Residuals receive the scale transform only, since their additive level cancels. TS-IFA then optionally instance-normalizes all query-scale tensors with the query statistics. The TS-IFA job trains the adapter on T1+T2 with AdamW, uses an evaluation stride of 128, evaluates T3 once after training, and writes `ts_ifa/eval_metrics.json` plus `ts_ifa/training_nmse.pdf`.

Both jobs finish by running `ts_ifa.results_table` and write
`<OUT_ROOT>/results_mse.tex`. The result loader combines direct
`univariate_summary.json`, adapter `baseline_metrics.json`, and
`ts_ifa/eval_metrics.json` artifacts. Retrieval-dependent columns are qualified
by retrieval setting so equally named baselines remain distinct. Table labels
are shortened by default: for example, `chronos_in_euclidean_3_online/mix_1_learned`
is displayed as `IN_L2_3/mix1`, while an explicitly unnormalized run named
`chronos_raw_euclidean_3_fixed` is displayed as `raw_L2_3_fixed`. Run-specific
`vanilla` columns are hidden by default.

Generate or regenerate a table independently with:

```bash
python -m ts_ifa.results_table outputs/results \
  --metric mse --split eval \
  --datasets electricity,traffic \
  --dataset-settings electricity=168_24,672_168 \
  --methods chronos,chronos_in_euclidean_1_online/linear_mix \
  --reference chronos --decimals 2
```

By default, the best value per row is bold, each row has an explicit automatic
power-of-ten scale, and dataset, per-L-H, and overall percentage improvements
are shown. Each summary percentage is computed from the methods' average metric
values, not by averaging individual percentages. These can independently be disabled with `--no-bold`,
`--no-dataset-improvements`, `--no-setting-improvements`, and
`--no-overall-improvement`. Global `--settings`, repeatable dataset-specific
`--dataset-settings`, ordered `--methods`, `--higher-is-better`,
`--no-auto-scale`, `--scale-exponent`, repeatable
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

Write generated model artifacts under `outputs/results/`; reserve `outputs/hydra/` for Hydra job metadata. Typical files include `*_prediction_payload.pt`, `baseline_metrics.csv`, `ts_ifa.pt`, `training_nmse.pdf`, summary CSV/JSON files, and plots under `plots/`. Do not commit datasets, model weights, cluster outputs, or machine-specific paths.
