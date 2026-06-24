"""Evaluate notebook/LaTeX baselines from extracted neighbor payloads."""

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

from ..data.scaling import neighbor_to_query_scale
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


def fit_gate(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
) -> list[dict[str, Any]]:
    if x_np.shape[0] == 0:
        raise ValueError("cannot train baseline gates from an empty oracle-train slice")
    targets = np.asarray(y_np)
    if targets.ndim == 1:
        targets = targets[:, None]
    return [
        fit_loss_difference_regressor(
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
    train_horizon_gate: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    base_loss = (oracle_arrays["y"] - oracle_arrays["pred"]) ** 2
    context_loss = (oracle_arrays["y"] - oracle_arrays["pred_c"]) ** 2
    scalar_features_train = scalar_gate_features(oracle_arrays)
    # Positive difference means the context forecast has smaller loss.
    scalar_difference = (base_loss - context_loss).mean(axis=1, keepdims=True)
    scalar_model = fit_gate(
        scalar_features_train,
        scalar_difference,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed,
    )
    horizon_model = None
    if train_horizon_gate:
        horizon_difference = base_loss - context_loss
        horizon_features_train = horizon_gate_features(oracle_arrays)
        horizon_model = fit_gate(
            horizon_features_train,
            horizon_difference,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed + 1,
        )

    out: dict[str, dict[str, np.ndarray]] = {}
    for split, arrays in arrays_by_split.items():
        split_predictions = dict(base_predictions_by_split[split])
        scalar_features = scalar_gate_features(arrays)
        scalar_difference = predict_gate(scalar_model, scalar_features)
        split_predictions["gated_context_scalar"] = np.where(
            scalar_difference > 0.0,
            arrays["pred_c"],
            arrays["pred"],
        )
        if horizon_model is not None:
            horizon_features = horizon_gate_features(arrays)
            horizon_difference = predict_gate(horizon_model, horizon_features)
            split_predictions["gated_context_horizon"] = np.where(
                horizon_difference > 0.0,
                arrays["pred_c"],
                arrays["pred"],
            )
        add_true_context_oracles(split_predictions, arrays)
        out[split] = split_predictions
    artifacts = {
        "backend": "catboost",
        "objective": "vanilla_loss_minus_context_loss",
        "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
        "horizon_feature_names": horizon_gate_feature_names(oracle_arrays["y"].shape[1]),
        "scalar_models": scalar_model,
        "horizon_models": horizon_model,
    }
    return out, artifacts


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
    parser.add_argument("--prefixes", default="train,oracle,eval")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--gate-iterations", "--gate-epochs", dest="gate_iterations", type=int, default=300)
    parser.add_argument(
        "--gate-learning-rate",
        "--gate-lr",
        dest="gate_learning_rate",
        type=float,
        default=3e-2,
    )
    parser.add_argument("--gate-depth", type=int, default=4)
    parser.add_argument("--train-horizon-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    started = perf_counter()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir / "baseline_adapters"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("experiment start kind=baseline_adapters input=%s", input_dir)

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

    LOGGER.info("mixture fitting start")
    artifacts = fit_baseline_adapters(arrays_by_split["train"], args.l2)
    LOGGER.info("mixture fitting done")

    predictions_by_split_base = {
        split: predict_baseline_adapters(arrays, artifacts)
        for split, arrays in arrays_by_split.items()
    }
    LOGGER.info("context gate fitting start")
    predictions_by_split, gate_artifacts = add_context_gate_predictions(
        predictions_by_split_base,
        arrays_by_split["oracle"],
        arrays_by_split,
        iterations=args.gate_iterations,
        learning_rate=args.gate_learning_rate,
        depth=args.gate_depth,
        seed=args.seed,
        train_horizon_gate=args.train_horizon_gate,
    )
    LOGGER.info("context gate fitting done horizon=%s", args.train_horizon_gate)

    LOGGER.info("evaluation start split=eval")
    rows = evaluate_predictions("eval", arrays_by_split["eval"], predictions_by_split["eval"])
    LOGGER.info("evaluation done rows=%s", len(rows))
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "baseline_metrics.csv"
    json_path = output_dir / "baseline_metrics.json"
    artifact_path = output_dir / "baseline_artifacts.pt"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    torch.save(
        {
            "mix_artifacts": artifacts,
            "context_gate_artifacts": gate_artifacts,
            "gate_config": {
                "backend": "catboost",
                "objective": "vanilla_loss_minus_context_loss",
                "decision_threshold": 0.0,
                "scalar_feature_names": SCALAR_GATE_FEATURE_NAMES,
                "horizon_feature_names": horizon_gate_feature_names(
                    arrays_by_split["oracle"]["y"].shape[1]
                ),
                "iterations": args.gate_iterations,
                "learning_rate": args.gate_learning_rate,
                "depth": args.gate_depth,
                "train_horizon_gate": args.train_horizon_gate,
            },
        },
        artifact_path,
    )
    LOGGER.info("outputs saved dir=%s", output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - started)
    log_experiment_separator(LOGGER)
    return {"csv": csv_path, "json": json_path, "artifacts": artifact_path}


if __name__ == "__main__":
    main()
