# Repository Guidelines

## Project Structure & Module Organization

This repo implements baselines and TS-IFA from the LaTeX design notes; treat that LaTeX as prior design guidance that may evolve, and do not edit it unless asked. Active code and scripts are under `ts_ifa/`: `data/` owns loading and retrieval, `models/` contains forecast backbones and the adapter, `experiments/` contains runnable pipelines and logging, `visu/` contains plotting helpers and notebooks, and `slurm/` contains cluster jobs. `tests/smoke/` remains at repository root for tiny local checks.

## Data Flow & Experiment Scope

Input is a date-indexed CSV with target user columns and optional covariates. The default four-way protocol is T0 datastore (30%), T1 baseline training (35%), T2 context-gate training (15%), and T3 final evaluation (20%). Online retrieval is default and caps each query datastore to the aligned T0 capacity; fixed retrieval makes T1/T2/T3 fetch only from T0. Baseline mixtures fit T1, only the context-conditioned gate fits T2, TS-IFA fits T1+T2 without validation, and every final metric uses T3. Full experiments run on a distant cluster, so keep local work focused on loading and shape checks.

## SLURM Experiment Rules

Full experiments are launched from `ts_ifa/slurm/`, with separate subfolders by experiment type. Follow the established cluster style: include the complete `#SBATCH` header, write logs under `script_outputs/`, activate `.venv`, define configs directly near the top, loop over Bash `DATASETS` and `SETTINGS` arrays, and pass those values explicitly to Python entry points. Store model results under `outputs/results/`; Hydra job metadata, if introduced, belongs under `outputs/hydra/`. Experiment logging should identify the run once and report important stage start/completion, outputs, and runtime without repeatedly dumping provided metadata. Do not infer configs from path names, add `${VAR:-default}` environment fallbacks, or build generic launch frameworks. Keep each job limited to options used by this package and validate scripts with `bash -n`.

## Build, Test, and Development Commands

There is no build step. Run smoke checks before handing off code:

```powershell
python tests/smoke/check_loads.py
python tests/smoke/check_loads.py --check-patchtst
python tests/smoke/check_ts_ifa_training.py
```

Shared pretrained weights live beside the dataset folder at `../weights/`; Chronos uses `../weights/chronos2`. Use `--chronos-weights` only to override that default. Example experiment commands are documented in `README.md`.

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, `snake_case` functions, and `CamelCase` classes. Prefer `pathlib.Path`, explicit tensor shape checks, deterministic seeds, package-relative imports, and small functions matching existing subpackage boundaries. Keep Chronos and PatchTST logic in their specific model modules.

## Testing Guidelines

No full test suite exists. Add small fixtures under `tests/smoke/` when checking new load paths or model wrappers. Avoid committing cluster outputs, datasets, or weights.

## Git & Pull Request Guidelines

Git is handled by the repo owner. Do not create commits, branches, or history changes unless explicitly asked. For PR-ready notes, include commands run, dataset/model assumptions, changed outputs, and screenshots for plot/dashboard changes.
