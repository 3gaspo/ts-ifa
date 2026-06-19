# Repository Guidelines

## Project Structure & Module Organization

This repo implements baselines and TS-IFA from `latex_old/main.tex`; treat that LaTeX as prior design guidance that may evolve, and do not edit it unless asked. `timetensor_old/` contains reference code from the prior repo. Core code lives at the root: `load_dataset_model.py` loads CSVs and model configs, `models.py` owns the shared wrapper/registry, `chronos_model.py` and `patchtst.py` hold model-specific wrappers, `neighbors.py` builds aligned windows and KNN features, `extraction.py` runs neighbor extraction, `features.py` summarizes payloads, and `ts_ifa.py` plus `train_ts_ifa.py` expose the payload-based adapter. `visu/` contains plotting helpers and notebooks. `tests/smoke/` is for tiny load checks only. Files with `old` in the name are temporary comparison references.

## Data Flow & Experiment Scope

Input is a date-indexed CSV with target user columns and optional covariates. Windows use `(users, 1, lags)` inputs and `(users, 1, horizon)` targets. `experiment_univariate.py` evaluates direct baselines; `extraction.py` builds query/datastore windows, computes representations, searches neighbors, and saves payloads; `features.py` turns payloads into tables and plots. Full experiments run on a distant cluster from another PC, so keep local work focused on loading and shape checks.

## Build, Test, and Development Commands

There is no build step. Run smoke checks before handing off code:

```powershell
python tests/smoke/check_loads.py
python tests/smoke/check_loads.py --check-patchtst
python tests/smoke/check_ts_ifa_training.py
```

Use `--chronos-weights path/to/weights` only where Chronos dependencies and weights are installed. Example experiment commands are documented in `README.md`.

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, `snake_case` functions, and `CamelCase` classes. Prefer `pathlib.Path`, explicit tensor shape checks, deterministic seeds, and small functions matching existing module boundaries. Keep Chronos and PatchTST logic in their specific scripts; `foundation_models.py` is compatibility-only.

## Testing Guidelines

No full test suite exists. Add small fixtures under `tests/smoke/` when checking new load paths or model wrappers. Avoid committing cluster outputs, datasets, or weights.

## Git & Pull Request Guidelines

Git is handled by the repo owner. Do not create commits, branches, or history changes unless explicitly asked. For PR-ready notes, include commands run, dataset/model assumptions, changed outputs, and screenshots for plot/dashboard changes.
