# Lightweight Extraction

This repository contains Python tools for retrieval-based residual adaptation experiments in time-series forecasting. It includes data loading, forecasting baselines, neighbor retrieval, feature extraction, and diagnostics.

## File Map

- `load_dataset_model.py`: CSV loading, target/covariate selection, split helpers, seeds, device resolution, and model construction.
- `models.py`: common `ForecastModel` wrapper, normalization (`none`, `instance`), persistence and linear baselines, and model registry.
- `chronos_model.py`: Chronos-2 wrapper for the current inference format.
- `patchtst.py`: PatchTST implementation with current-format inference hooks.
- `foundation_models.py`: compatibility imports only; new code should use `chronos_model.py` or `patchtst.py`.
- `neighbors.py`: aligned window generation, feature representations, and exact KNN search.
- `extraction.py`: neighbor extraction and prediction payload generation.
- `features.py`: feature tables and diagnostic plots from extraction payloads.
- `experiment_univariate.py`: no-neighbor baseline evaluation over all users.
- `visu/`: plotting helpers and notebooks.
- `tests/smoke/`: tiny CSV fixture and load/inference checks for local or cluster sanity tests.

## Data Flow

Input data is a date-indexed CSV. Target user columns become `dataset.frame`; optional past and future covariate columns become shared covariate tensors. A window is represented as:

- `x`: `(users, 1, lags)`
- `y`: `(users, 1, horizon)`
- past covariates: `(1, channels, lags)`
- future covariates: `(1, channels, horizon)`

`experiment_univariate.py` loads a model and evaluates direct predictions on selected splits. `extraction.py` builds query windows, builds an aligned datastore from earlier windows, computes features in `raw`, `fourier`, `model`, `chronos`, or `patchtst` space, searches nearest neighbors, and saves prediction payloads. `features.py` reads those payloads and produces flat feature summaries and plots.

## Smoke Checks

Use the small fixture under `tests/smoke/` to validate loading and model construction:

```powershell
python tests/smoke/check_loads.py
python tests/smoke/check_loads.py --check-patchtst
python tests/smoke/check_loads.py --chronos-weights path/to/chronos/weights
```

These checks validate CSV parsing, window shapes, persistence inference, optional PatchTST construction, and optional Chronos loading.

## Experiment Commands

Run a no-neighbor baseline:

```powershell
python experiment_univariate.py --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model persistence --normalization none --eval-stride 24 --output-dir outputs/extraction_univariate --save-name electricity_persistence
```

Run neighbor extraction:

```powershell
python extraction.py --csv ../datasets/electricity/electricity.csv --lags 168 --horizon 24 --model chronos --model-kwargs '{"weights_path":"path/to/chronos/weights","context_mode":"past_only"}' --neighbors 5 --distance-space chronos --pool-representation --distance-metric cosine --train-stride 24 --eval-stride 24 --period 24 --output-dir outputs/extraction_neighbors --save-name electricity_chronos_k5
```

Analyze an extraction run:

```powershell
python features.py --input-dir outputs/extraction_neighbors/electricity_chronos_k5
```

## Outputs

Write generated artifacts under `outputs/`. Typical files include `*_prediction_payload.pt`, summary CSV/JSON files, and plots under `plots/`. Do not commit datasets, model weights, cluster outputs, or machine-specific paths.
