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
    y_c = payload[f"{prefix}_Yc_values"].float()
    e = payload[f"{prefix}_E_values"].float()
    pred_neighbors = y_c - e
    return {
        "pred": rearrange(payload[f"{prefix}_preds"].float(), "date user horizon -> (date user) horizon").numpy(),
        "pred_c": rearrange(
            payload[f"{prefix}_preds_context"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "y": rearrange(payload[f"{prefix}_Y_values"].float(), "date user horizon -> (date user) horizon").numpy(),
        "x": rearrange(payload[f"{prefix}_X_values"].float(), "date user lags -> (date user) lags").numpy(),
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
        "query_t": rearrange(payload[f"{prefix}_query_t"], "date user -> (date user)").numpy(),
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
    # Average the normal equations before regularizing so that ``l2`` does
    # not become weaker merely because more windows or horizons are present.
    xtx = (x.T @ x) / x.shape[0]
    xty = (x.T @ y) / x.shape[0]
    reg = float(l2) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, xty)


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
    lam = np.mean(mix0_direction * residual_target) / (np.mean(mix0_direction**2) + float(l2))
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


def gate_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    pred = arrays["pred"]
    pred_c = arrays["pred_c"]
    y_c = arrays["y_c"]
    e = arrays["e"]
    distance = arrays["distance"]
    x = arrays["x"]
    weighted = weighted_neighbor_horizon(arrays)
    weighted_e = weighted_neighbor_residual(arrays)
    cols = [
        ((pred_c - pred) ** 2).mean(axis=1),
        ((weighted - pred) ** 2).mean(axis=1),
        (weighted_e**2).mean(axis=1),
        (e**2).mean(axis=(1, 2)),
        distance.mean(axis=1),
        distance.std(axis=1),
        x.mean(axis=1),
        x.std(axis=1),
        y_c.mean(axis=(1, 2)),
        y_c.std(axis=(1, 2)),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def fit_binary_gate(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    seed: int,
) -> dict[str, Any]:
    labels = np.asarray(y_np, dtype=np.int64).reshape(-1)
    unique = np.unique(labels)
    if len(unique) == 1:
        return {"constant": float(unique[0])}
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
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_np, labels)
    return {"classifier": model}


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
    labels = np.asarray(y_np)
    if labels.ndim == 1:
        labels = labels[:, None]
    return [
        fit_binary_gate(
            x_np,
            labels[:, output_idx],
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed + output_idx,
        )
        for output_idx in range(labels.shape[1])
    ]


def predict_gate(models: list[dict[str, Any]], features: np.ndarray) -> np.ndarray:
    columns = []
    for model in models:
        if "constant" in model:
            probability = np.full(features.shape[0], model["constant"], dtype=np.float64)
        else:
            probability = model["classifier"].predict_proba(features)[:, 1]
        columns.append(probability)
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
    features_train = gate_features(oracle_arrays)
    scalar_label = (context_loss.mean(axis=1) < base_loss.mean(axis=1)).astype(np.float32)[:, None]
    scalar_model = fit_gate(
        features_train,
        scalar_label,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        seed=seed,
    )
    horizon_model = None
    if train_horizon_gate:
        horizon_label = (context_loss < base_loss).astype(np.float32)
        horizon_model = fit_gate(
            features_train,
            horizon_label,
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            seed=seed + 1,
        )

    out: dict[str, dict[str, np.ndarray]] = {}
    for split, arrays in arrays_by_split.items():
        split_predictions = dict(base_predictions_by_split[split])
        features = gate_features(arrays)
        scalar_prob = predict_gate(scalar_model, features)
        split_predictions["gated_context_scalar"] = np.where(
            scalar_prob >= 0.5,
            arrays["pred_c"],
            arrays["pred"],
        )
        if horizon_model is not None:
            horizon_prob = predict_gate(horizon_model, features)
            split_predictions["gated_context_horizon"] = np.where(
                horizon_prob >= 0.5,
                arrays["pred_c"],
                arrays["pred"],
            )
        add_true_context_oracles(split_predictions, arrays)
        out[split] = split_predictions
    artifacts = {
        "backend": "catboost",
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
