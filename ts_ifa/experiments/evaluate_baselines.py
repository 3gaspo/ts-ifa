"""Evaluate baseline and gate families from extracted neighbor payloads."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from ..data.neighbors import neighbor_to_query_scale
from .runtime import log_experiment_separator, setup_logging


LOGGER = logging.getLogger(__name__)


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def flatten_payload(payload: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    x = payload[f"{prefix}_X_values"].float()
    x_c = payload[f"{prefix}_Xc_values"].float()
    y_c_raw = payload[f"{prefix}_Yc_values"].float()
    e_raw = payload[f"{prefix}_E_values"].float()
    pred_neighbors_raw = y_c_raw - e_raw
    y_c = neighbor_to_query_scale(x, x_c, y_c_raw)
    e = neighbor_to_query_scale(x, x_c, e_raw, residual=True)
    pred_neighbors = neighbor_to_query_scale(x, x_c, pred_neighbors_raw)
    query_t = payload[f"{prefix}_query_t"]
    query_user = payload[f"{prefix}_query_user_idx"]
    neighbor_t = payload[f"{prefix}_neighbor_t"]
    neighbor_user = payload[f"{prefix}_neighbor_user_idx"]
    return {
        "pred": rearrange(payload[f"{prefix}_preds"].float(), "date user horizon -> (date user) horizon").numpy(),
        "pred_c": rearrange(
            payload[f"{prefix}_preds_context"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "y": rearrange(payload[f"{prefix}_Y_values"].float(), "date user horizon -> (date user) horizon").numpy(),
        "x": rearrange(x, "date user lags -> (date user) lags").numpy(),
        "y_c": rearrange(y_c, "date user neighbor horizon -> (date user) neighbor horizon").numpy(),
        "e": rearrange(e, "date user neighbor horizon -> (date user) neighbor horizon").numpy(),
        "pred_neighbors": rearrange(
            pred_neighbors,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy(),
        "distance": rearrange(
            payload[f"{prefix}_distance_x_xc"].float(),
            "date user neighbor -> (date user) neighbor",
        ).numpy(),
        "query_t": rearrange(query_t, "date user -> (date user)").numpy(),
        "neighbor_lookback_mean": rearrange(
            x_c.mean(dim=-1).mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_mean_std": rearrange(
            x_c.mean(dim=-1).std(dim=-1, unbiased=False),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_std": rearrange(
            x_c.std(dim=-1, unbiased=False).mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_lookback_std_std": rearrange(
            x_c.std(dim=-1, unbiased=False).std(dim=-1, unbiased=False),
            "date user -> (date user)",
        ).numpy(),
        "same_user_ratio": rearrange(
            (neighbor_user == query_user.unsqueeze(-1)).float().mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
        "neighbor_age_mean": rearrange(
            (query_t.unsqueeze(-1) - neighbor_t).float().mean(dim=-1),
            "date user -> (date user)",
        ).numpy(),
    }


def distance_weights(arrays: dict[str, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    d = arrays["distance"].astype(np.float64)
    d_std = d.std(axis=-1, keepdims=True)
    d_norm = (d - d.min(axis=-1, keepdims=True)) / np.maximum(d_std, eps)
    return softmax_np(-d_norm, axis=-1)


def weighted_neighbor_horizon(arrays: dict[str, np.ndarray]) -> np.ndarray:
    w = distance_weights(arrays)
    return (w[:, :, None] * arrays["y_c"]).sum(axis=1)


def weighted_neighbor_residual(arrays: dict[str, np.ndarray]) -> np.ndarray:
    w = distance_weights(arrays)
    return (w[:, :, None] * arrays["e"]).sum(axis=1)


def ridge_no_intercept(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("cannot fit ridge regression without observations")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")
    # RMS-standardize without centering so zero coefficients still mean zero
    # correction. This makes the penalty invariant to dataset units while
    # retaining the vanilla forecast as the ridge anchor.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    feature_scale = np.sqrt(np.mean(x**2, axis=0))
    feature_scale = np.maximum(feature_scale, 1e-12)
    standardized = x / feature_scale
    xtx = (standardized.T @ standardized) / x.shape[0]
    xty = (standardized.T @ y) / x.shape[0]
    reg = float(l2) * np.eye(xtx.shape[0], dtype=np.float64)
    standardized_coef = np.linalg.solve(xtx + reg, xty)
    return standardized_coef / feature_scale


def fit_baseline_adapters(train: dict[str, np.ndarray], l2: float) -> dict[str, Any]:
    pred = train["pred"]
    y = train["y"]
    y_c = train["y_c"]
    pred_c = train["pred_c"]
    if y.shape[0] == 0:
        raise ValueError("cannot fit baseline adapters from an empty train payload")
    weighted = weighted_neighbor_horizon(train)
    residual_target = y - pred

    mix0_direction = weighted - pred
    lam = ridge_no_intercept(
        rearrange(mix0_direction, "sample horizon -> (sample horizon) 1"),
        rearrange(residual_target, "sample horizon -> (sample horizon)"),
        l2,
    )[0]
    lam = float(np.clip(lam, 0.0, 1.0))

    mix1_x = np.concatenate([pred[:, :, None], np.moveaxis(y_c, 1, 2)], axis=-1)
    mix1_coef = ridge_no_intercept(
        rearrange(mix1_x, "sample horizon feature -> (sample horizon) feature"),
        rearrange(residual_target, "sample horizon -> (sample horizon)"),
        l2,
    )

    horizon = y.shape[1]
    neighbors = y_c.shape[1]
    mix2_coef = np.zeros((horizon, neighbors + 2), dtype=np.float64)
    for h in range(horizon):
        x_h = np.concatenate(
            [
                pred[:, h : h + 1],
                y_c[:, :, h],
                pred_c[:, h : h + 1],
            ],
            axis=1,
        )
        mix2_coef[h] = ridge_no_intercept(x_h, residual_target[:, h], l2)

    return {
        "mix_0_lambda": lam,
        "mix_1_coef": mix1_coef,
        "mix_2_coef": mix2_coef,
    }


def predict_baseline_adapters(arrays: dict[str, np.ndarray], artifacts: dict[str, Any]) -> dict[str, np.ndarray]:
    pred = arrays["pred"]
    y_c = arrays["y_c"]
    pred_c = arrays["pred_c"]
    weighted = weighted_neighbor_horizon(arrays)
    unweighted = y_c.mean(axis=1)
    weighted_e = weighted_neighbor_residual(arrays)

    predictions: dict[str, np.ndarray] = {
        "vanilla": pred,
        "context_conditioned": pred_c,
        "neighbor_weighted_mean": weighted,
        "neighbor_unweighted_mean": unweighted,
        "pred_plus_weighted_e": pred + weighted_e,
        "mix_0_weighted": (1.0 - artifacts["mix_0_lambda"]) * pred + artifacts["mix_0_lambda"] * weighted,
    }

    mix1_coef = artifacts["mix_1_coef"]
    mix1_x = np.concatenate([pred[:, :, None], np.moveaxis(y_c, 1, 2)], axis=-1)
    correction = np.einsum("shf,f->sh", mix1_x, mix1_coef)
    predictions["mix_1_learned"] = pred + correction

    mix2_coef = artifacts["mix_2_coef"]
    full = np.empty_like(pred)
    for h in range(pred.shape[1]):
        x_h = np.concatenate([pred[:, h : h + 1], y_c[:, :, h], pred_c[:, h : h + 1]], axis=1)
        full[:, h] = pred[:, h] + x_h @ mix2_coef[h]
    predictions["mix_2_full_horizon"] = full
    return predictions


TRAINABLE_BASELINES = (
    "mix_0_weighted",
    "mix_1_learned",
    "mix_2_full_horizon",
)


def add_eval_fitted_baselines(
    predictions_by_split: dict[str, dict[str, np.ndarray]],
    eval_arrays: dict[str, np.ndarray],
    *,
    l2: float,
) -> dict[str, Any]:
    """Add explicitly optimistic T3 in-sample fits for trainable mixtures."""
    artifacts = fit_baseline_adapters(eval_arrays, l2)
    eval_predictions = predict_baseline_adapters(eval_arrays, artifacts)
    predictions_by_split["eval"].update(
        {
            f"{name}_eval_fit": eval_predictions[name]
            for name in TRAINABLE_BASELINES
        }
    )
    return artifacts


COMMON_GATE_FEATURE_NAMES = (
    "weighted_neighbor_minus_vanilla_mean",
    "weighted_neighbor_residual_mean",
    "query_mean",
    "query_std",
    "neighbor_lookback_means_mean_raw",
    "neighbor_lookback_means_std_raw",
    "neighbor_lookback_stds_mean_raw",
    "neighbor_lookback_stds_std_raw",
    "same_user_ratio",
    "neighbor_age_mean",
    "neighbor_weight_std",
    "neighbor_weight_max",
    "distance_mean",
)

SCALAR_GATE_FEATURE_NAMES = (
    "context_minus_vanilla_mean",
    "context_minus_vanilla_std",
    *COMMON_GATE_FEATURE_NAMES,
)


def horizon_gate_feature_names(horizon: int) -> tuple[str, ...]:
    return (
        *(f"context_minus_vanilla_h{index}" for index in range(horizon)),
        *COMMON_GATE_FEATURE_NAMES,
    )


def common_gate_features(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
    pred = arrays["pred"]
    x = arrays["x"]
    weights = distance_weights(arrays)
    weighted = weighted_neighbor_horizon(arrays)
    weighted_e = weighted_neighbor_residual(arrays)
    return [
        (weighted - pred).mean(axis=1),
        weighted_e.mean(axis=1),
        x.mean(axis=1),
        x.std(axis=1),
        arrays["neighbor_lookback_mean"],
        arrays["neighbor_lookback_mean_std"],
        arrays["neighbor_lookback_std"],
        arrays["neighbor_lookback_std_std"],
        arrays["same_user_ratio"],
        arrays["neighbor_age_mean"],
        weights.std(axis=1),
        weights.max(axis=1),
        arrays["distance"].mean(axis=1),
    ]


def scalar_gate_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    pred = arrays["pred"]
    pred_c = arrays["pred_c"]
    context_delta = pred_c - pred
    cols = [
        context_delta.mean(axis=1),
        context_delta.std(axis=1),
        *common_gate_features(arrays),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def horizon_gate_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    pred = arrays["pred"]
    context_delta = arrays["pred_c"] - pred
    common = np.stack(common_gate_features(arrays), axis=1)
    return np.concatenate([context_delta, common], axis=1).astype(np.float32)


def fit_loss_difference_regressor(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(y_np, dtype=np.float64).reshape(-1)
    if np.ptp(target) <= 1e-12:
        return {"constant": float(target.mean())}
    try:
        from catboost import CatBoostRegressor
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
    model = CatBoostRegressor(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_np, target)
    return {"regressor": model}


def fit_improvement_classifier(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
) -> dict[str, Any]:
    target = np.asarray(y_np, dtype=np.float64).reshape(-1) > 0.0
    if np.unique(target).size == 1:
        return {"constant": float(target[0]) - 0.5}
    try:
        from catboost import CatBoostClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency error
        raise ModuleNotFoundError(
            "CatBoost gates require the `catboost` project dependency. Run `uv sync`."
        ) from exc
    model = CatBoostClassifier(
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_np, target.astype(np.int8))
    return {"classifier": model}


def fit_gate(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
    objective: str = "regressor",
) -> list[dict[str, Any]]:
    if x_np.shape[0] == 0:
        raise ValueError("cannot train baseline gates from an empty oracle-train slice")
    targets = np.asarray(y_np)
    if targets.ndim == 1:
        targets = targets[:, None]
    if objective not in {"classifier", "regressor"}:
        raise ValueError(f"unknown gate objective {objective!r}")
    fit_one = (
        fit_improvement_classifier
        if objective == "classifier"
        else fit_loss_difference_regressor
    )
    return [
        fit_one(
            x_np,
            targets[:, output_idx],
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed + output_idx,
        )
        for output_idx in range(targets.shape[1])
    ]


def predict_gate(models: list[dict[str, Any]], features: np.ndarray) -> np.ndarray:
    columns = []
    for model in models:
        if "constant" in model:
            difference = np.full(features.shape[0], model["constant"], dtype=np.float64)
        elif "classifier" in model:
            # Center the positive-class probability so every gate uses zero as
            # the decision threshold and diagnostics remain directly comparable.
            difference = model["classifier"].predict_proba(features)[:, 1] - 0.5
        else:
            difference = model["regressor"].predict(features)
        columns.append(difference)
    return np.column_stack(columns)


def add_true_context_oracles(
    predictions: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
) -> None:
    """Add target-aware upper bounds for scalar and horizon context gates."""
    pred = arrays["pred"]
    pred_c = arrays["pred_c"]
    target = arrays["y"]
    base_loss = (target - pred) ** 2
    context_loss = (target - pred_c) ** 2
    use_context_scalar = context_loss.mean(axis=1, keepdims=True) < base_loss.mean(axis=1, keepdims=True)
    predictions["oracle_context_scalar"] = np.where(use_context_scalar, pred_c, pred)
    predictions["oracle_context_horizon"] = np.where(context_loss < base_loss, pred_c, pred)


def add_context_gate_predictions(
    base_predictions_by_split: dict[str, dict[str, np.ndarray]],
    oracle_arrays: dict[str, np.ndarray],
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
    dict[str, dict[str, np.ndarray]],
]:
    base_loss = (oracle_arrays["y"] - oracle_arrays["pred"]) ** 2
    context_loss = (oracle_arrays["y"] - oracle_arrays["pred_c"]) ** 2
    # Positive difference means the context forecast has smaller loss.
    train_targets = {
        "scalar": (base_loss - context_loss).mean(axis=1, keepdims=True),
        "horizon": base_loss - context_loss,
    }
    train_features = {
        "scalar": scalar_gate_features(oracle_arrays),
        "horizon": horizon_gate_features(oracle_arrays),
    }
    models: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for objective_index, objective in enumerate(("classifier", "regressor")):
        models[objective] = {}
        for shape_index, shape in enumerate(("scalar", "horizon")):
            models[objective][shape] = fit_gate(
                train_features[shape],
                train_targets[shape],
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                seed=seed + objective_index * 10_000 + shape_index * 1_000,
                objective=objective,
            )

    out: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    for split, arrays in arrays_by_split.items():
        split_predictions = dict(base_predictions_by_split[split])
        split_features = {
            "scalar": scalar_gate_features(arrays),
            "horizon": horizon_gate_features(arrays),
        }
        split_base_loss = (arrays["y"] - arrays["pred"]) ** 2
        split_context_loss = (arrays["y"] - arrays["pred_c"]) ** 2
        split_targets = {
            "scalar": (split_base_loss - split_context_loss).mean(axis=1),
            "horizon": split_base_loss - split_context_loss,
        }
        diagnostics[split] = {}
        for objective in ("classifier", "regressor"):
            for shape in ("scalar", "horizon"):
                score = predict_gate(models[objective][shape], split_features[shape])
                decision = score > 0.0
                if shape == "scalar":
                    decision = decision[:, :1]
                name = f"gated_context_{objective}_{shape}"
                split_predictions[name] = np.where(
                    decision,
                    arrays["pred_c"],
                    arrays["pred"],
                )
                diagnostics[split][f"{objective}_{shape}_score"] = (
                    score[:, 0] if shape == "scalar" else score
                )
                diagnostics[split][f"{objective}_{shape}_target"] = split_targets[shape]
        add_true_context_oracles(split_predictions, arrays)
        out[split] = split_predictions
    artifacts = {
        "backend": "catboost",
        "objectives": {
            "classifier": "context_improves_over_vanilla",
            "regressor": "vanilla_loss_minus_context_loss",
        },
        "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
        "horizon_feature_names": horizon_gate_feature_names(oracle_arrays["y"].shape[1]),
        "models": models,
    }
    return out, artifacts, diagnostics


def visualization_payload(
    predictions_by_split: dict[str, dict[str, np.ndarray]],
    gate_diagnostics: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Create a plotting payload without serialized estimators or duplicated inputs."""
    splits: dict[str, Any] = {}
    for split, predictions in predictions_by_split.items():
        splits[split] = {
            "predictions": {
                name: torch.as_tensor(value, dtype=torch.float32)
                for name, value in predictions.items()
            },
            "gate_diagnostics": {
                name: torch.as_tensor(value, dtype=torch.float32)
                for name, value in gate_diagnostics.get(split, {}).items()
            },
        }
    return {
        "format_version": 1,
        "description": "Precomputed baseline predictions and gate diagnostics for visualization.",
        "splits": splits,
    }


def evaluate_predictions(split: str, arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    y = arrays["y"]
    scale = np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
    vanilla_nmse = np.mean(((arrays["pred"] - y) / scale) ** 2)
    for name, pred in predictions.items():
        err = pred - y
        mse = np.mean(err**2)
        mae = np.mean(np.abs(err))
        nmse = np.mean((err / scale) ** 2)
        rows.append(
            {
                "split": split,
                "baseline": name,
                "mse": float(mse),
                "mae": float(mae),
                "nmse": float(nmse),
                "relative_nmse_improvement_pct": float(100.0 * (vanilla_nmse - nmse) / max(vanilla_nmse, 1e-12)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--family", choices=("all", "baselines", "gates"), default="all")
    parser.add_argument("--prefixes", default="train,oracle,eval")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument(
        "--fit-baselines-on-eval",
        action="store_true",
        help="Also report optimistic in-sample fits of trainable baselines on T3",
    )
    parser.add_argument("--gate-iterations", "--gate-epochs", dest="gate_iterations", type=int, default=300)
    parser.add_argument(
        "--gate-learning-rate",
        "--gate-lr",
        dest="gate_learning_rate",
        type=float,
        default=3e-2,
    )
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--train-horizon-gate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    started = perf_counter()
    input_dir = Path(args.input_dir).expanduser()
    default_subdir = {
        "all": "baseline_adapters",
        "baselines": "baselines",
        "gates": "gates",
    }[args.family]
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir / default_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("experiment start kind=%s input=%s", args.family, input_dir)

    prefixes = [part.strip() for part in args.prefixes.replace(";", ",").split(",") if part.strip()]
    LOGGER.info("payload load start")
    arrays_by_split = {
        prefix: flatten_payload(torch_load(input_dir / f"{prefix}_prediction_payload.pt"), prefix)
        for prefix in prefixes
    }
    missing = {"train", "oracle", "eval"} - set(arrays_by_split)
    if missing:
        raise ValueError(f"baseline evaluation requires train, oracle, and eval payloads; missing {sorted(missing)}")
    LOGGER.info("payload load done splits=%s", ",".join(prefixes))

    artifacts = None
    eval_fit_artifacts = None
    predictions_by_split: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in arrays_by_split
    }
    if args.family in {"all", "baselines"}:
        LOGGER.info("mixture fitting start")
        artifacts = fit_baseline_adapters(arrays_by_split["train"], args.l2)
        predictions_by_split = {
            split: predict_baseline_adapters(arrays, artifacts)
            for split, arrays in arrays_by_split.items()
        }
        if args.fit_baselines_on_eval:
            eval_fit_artifacts = add_eval_fitted_baselines(
                predictions_by_split,
                arrays_by_split["eval"],
                l2=args.l2,
            )
        LOGGER.info("mixture fitting done")

    gate_artifacts = None
    gate_diagnostics: dict[str, dict[str, np.ndarray]] = {
        split: {} for split in arrays_by_split
    }
    if args.family in {"all", "gates"}:
        LOGGER.info("context gate fitting start objectives=classifier,regressor shapes=scalar,horizon")
        predictions_by_split, gate_artifacts, gate_diagnostics = add_context_gate_predictions(
            predictions_by_split,
            arrays_by_split["oracle"],
            arrays_by_split,
            iterations=args.gate_iterations,
            learning_rate=args.gate_learning_rate,
            depth=args.gate_depth,
            seed=args.seed,
        )
        LOGGER.info("context gate fitting done")

    LOGGER.info("evaluation start split=eval")
    rows = evaluate_predictions("eval", arrays_by_split["eval"], predictions_by_split["eval"])
    LOGGER.info("evaluation done rows=%s", len(rows))
    frame = pd.DataFrame(rows)
    metrics_stem = "gate_metrics" if args.family == "gates" else "baseline_metrics"
    csv_path = output_dir / f"{metrics_stem}.csv"
    json_path = output_dir / f"{metrics_stem}.json"
    artifact_path = output_dir / ("gate_artifacts.pt" if args.family == "gates" else "baseline_artifacts.pt")
    visualization_path = output_dir / "visualization_payload.pt"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    saved_artifacts: dict[str, Any] = {"family": args.family}
    if artifacts is not None:
        saved_artifacts["mix_artifacts"] = artifacts
    if eval_fit_artifacts is not None:
        saved_artifacts["eval_fit_mix_artifacts"] = eval_fit_artifacts
    if gate_artifacts is not None:
        saved_artifacts["context_gate_artifacts"] = gate_artifacts
        saved_artifacts["gate_config"] = {
            "backend": "catboost",
            "objectives": ["classifier", "regressor"],
            "decision_threshold": 0.0,
            "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
            "horizon_feature_names": horizon_gate_feature_names(
                arrays_by_split["oracle"]["y"].shape[1]
            ),
            "iterations": args.gate_iterations,
            "learning_rate": args.gate_learning_rate,
            "depth": args.gate_depth,
            "shapes": ["scalar", "horizon"],
        }
    torch.save(saved_artifacts, artifact_path)
    torch.save(
        visualization_payload(predictions_by_split, gate_diagnostics),
        visualization_path,
    )
    LOGGER.info("outputs saved dir=%s", output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - started)
    log_experiment_separator(LOGGER)
    return {
        "csv": csv_path,
        "json": json_path,
        "artifacts": artifact_path,
        "visualization": visualization_path,
    }


if __name__ == "__main__":
    main()
