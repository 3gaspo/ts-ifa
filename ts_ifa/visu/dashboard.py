"""Artifact-only helpers for the interactive retrieval dashboard notebook."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(Path(path), map_location="cpu")


def _flatten(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().reshape(-1, *value.shape[2:]).numpy()


def _flatten_optional(payload: dict[str, Any], key: str) -> np.ndarray | None:
    value = payload.get(key)
    if value is None or not torch.is_tensor(value):
        return None
    return _flatten(value)


def load_dashboard_data(run_dir: str | Path) -> dict[str, Any]:
    """Load extraction, baseline, and optional TS-IFA plotting artifacts."""
    root = Path(run_dir).expanduser()
    extracted: dict[str, dict[str, Any]] = {}
    for split in ("train", "oracle", "eval"):
        path = root / f"{split}_prediction_payload.pt"
        if path.exists():
            extracted[split] = torch_load(path)
    if not extracted:
        raise FileNotFoundError(f"No *_prediction_payload.pt files found under {root}")

    visualization_paths = [
        root / "baseline_adapters" / "visualization_payload.pt",
        root / "baselines" / "visualization_payload.pt",
        root / "gates" / "visualization_payload.pt",
    ]
    baseline: dict[str, Any] = {"splits": {}}
    for path in visualization_paths:
        if not path.exists():
            continue
        payload = torch_load(path)
        for split, split_payload in payload.get("splits", {}).items():
            merged = baseline["splits"].setdefault(
                split, {"predictions": {}, "gate_diagnostics": {}}
            )
            merged["predictions"].update(split_payload.get("predictions", {}))
            merged["gate_diagnostics"].update(split_payload.get("gate_diagnostics", {}))
    combined_visualization_path = next(
        (path for path in visualization_paths if path.exists()),
        root / "baselines" / "visualization_payload.pt",
    )
    ts_ifa_path = root / "ts_ifa" / "eval_predictions.pt"
    ts_ifa = torch_load(ts_ifa_path) if ts_ifa_path.exists() else {"predictions": {}}

    data = {
        "run_dir": root,
        "extracted": extracted,
        "baseline": baseline,
        "ts_ifa": ts_ifa,
        "paths": {
            "baseline": combined_visualization_path,
            "gates": root / "gates" / "visualization_payload.pt",
            "ts_ifa": ts_ifa_path,
        },
    }
    for split in extracted:
        split_arrays(data, split)  # validate alignment eagerly
    return data


def available_splits(data: dict[str, Any]) -> list[str]:
    return [name for name in ("train", "oracle", "eval") if name in data["extracted"]]


def split_arrays(data: dict[str, Any], split: str) -> dict[str, Any]:
    payload = data["extracted"][split]
    prefix = f"{split}_"
    x_tensor = payload[prefix + "X_values"].float()
    y_tensor = payload[prefix + "Y_values"].float()
    x_c_tensor = payload[prefix + "Xc_values"].float()
    y_c_tensor = payload[prefix + "Yc_values"].float()
    e_tensor = payload.get(prefix + "E_values")
    dates, users = x_tensor.shape[:2]
    arrays: dict[str, Any] = {
        "x": _flatten(x_tensor),
        "y": _flatten(y_tensor),
        "x_c": _flatten(x_c_tensor),
        "y_c": _flatten(y_c_tensor),
        "e": _flatten(e_tensor.float()) if torch.is_tensor(e_tensor) else None,
        "query_t": payload[prefix + "query_t"].reshape(-1).numpy(),
        "query_user_idx": payload[prefix + "query_user_idx"].reshape(-1).numpy(),
        "neighbor_t": _flatten(payload[prefix + "neighbor_t"]),
        "neighbor_user_idx": _flatten(payload[prefix + "neighbor_user_idx"]),
        "distance": _flatten_optional(payload, prefix + "distance_x_xc"),
        "dates": dates,
        "users": users,
        "datetimes": payload.get(prefix + "datetimes", []),
    }
    predictions = {
        "vanilla": _flatten(payload[prefix + "preds"]),
        "context_forecast": _flatten(payload[prefix + "preds_context"]),
    }
    baseline_split = data["baseline"].get("splits", {}).get(split, {})
    for name, value in baseline_split.get("predictions", {}).items():
        predictions[name] = np.asarray(value)
    if split == "eval":
        for name, value in data["ts_ifa"].get("predictions", {}).items():
            if torch.is_tensor(value) and value.numel() > 0:
                prediction = value.numpy()
                if prediction.shape[0] < dates * users:
                    padded = np.full((dates * users, prediction.shape[1]), np.nan, dtype=prediction.dtype)
                    padded[: prediction.shape[0]] = prediction
                    prediction = padded
                predictions[name] = prediction
    n_samples = dates * users
    invalid = {name: value.shape for name, value in predictions.items() if value.shape[0] != n_samples}
    if invalid:
        raise ValueError(f"Prediction alignment mismatch for {split}: {invalid}; expected {n_samples} samples")
    arrays["predictions"] = predictions
    arrays["gate_diagnostics"] = {
        name: np.asarray(value)
        for name, value in baseline_split.get("gate_diagnostics", {}).items()
    }
    return arrays


def _normalize_pair(lookback: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(lookback))
    std = max(float(np.std(lookback)), 1e-8)
    return (lookback - mean) / std, (future - mean) / std


def plot_query_example(
    data: dict[str, Any],
    split: str,
    sample_index: int,
    *,
    instance_normalized: bool,
    hide_axes: bool,
) -> plt.Figure:
    arrays = split_arrays(data, split)
    n_samples = len(arrays["x"])
    if not 0 <= sample_index < n_samples:
        raise IndexError(f"sample_index must be in [0, {n_samples})")
    x = arrays["x"][sample_index].copy()
    y = arrays["y"][sample_index].copy()
    x_c = arrays["x_c"][sample_index].copy()
    y_c = arrays["y_c"][sample_index].copy()
    if instance_normalized:
        x, y = _normalize_pair(x, y)
        for neighbor in range(len(x_c)):
            x_c[neighbor], y_c[neighbor] = _normalize_pair(x_c[neighbor], y_c[neighbor])

    lags, horizon = len(x), len(y)
    past_axis = np.arange(-lags, 0)
    future_axis = np.arange(horizon)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, max(len(x_c), 1)))
    for neighbor, color in enumerate(colors):
        label = (
            f"neighbor {neighbor + 1} "
            f"(user {int(arrays['neighbor_user_idx'][sample_index, neighbor])}, "
            f"t={int(arrays['neighbor_t'][sample_index, neighbor])})"
        )
        ax.plot(past_axis, x_c[neighbor], color=color, alpha=0.72, linewidth=1.2, label=label)
        ax.plot(
            np.r_[past_axis[-1], future_axis],
            np.r_[x_c[neighbor, -1], y_c[neighbor]],
            color=color,
            alpha=0.72,
            linewidth=1.2,
            linestyle="--",
        )
    ax.plot(past_axis, x, color="black", linewidth=2.6, label="query lookback")
    ax.plot(
        np.r_[past_axis[-1], future_axis],
        np.r_[x[-1], y],
        color="black",
        linewidth=2.6,
        linestyle="--",
        label="query future",
    )
    ax.axvline(-0.5, color="0.45", linewidth=1, linestyle=":")
    ax.set_xlabel("Time relative to forecast origin")
    ax.set_ylabel("Instance-normalized value" if instance_normalized else "Value")
    query_t = int(arrays["query_t"][sample_index])
    user = int(arrays["query_user_idx"][sample_index])
    ax.set_title(f"{split}: query user {user}, t={query_t}")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    if hide_axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def prediction_names(data: dict[str, Any], split: str) -> list[str]:
    return sorted(split_arrays(data, split)["predictions"])


SCALAR_FEATURE_ORDER = (
    "context_minus_vanilla_mean",
    "context_minus_vanilla_std",
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


def _distance_weights(distance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    shifted = -np.asarray(distance, dtype=np.float64)
    shifted = shifted - np.nanmax(shifted, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.nansum(exp, axis=1, keepdims=True), eps)


def _neighbor_to_query_scale_np(
    query_lookback: np.ndarray,
    neighbor_lookback: np.ndarray,
    neighbor_value: np.ndarray,
    *,
    residual: bool = False,
    eps: float = 1e-8,
) -> np.ndarray:
    query_mean = query_lookback.mean(axis=-1, keepdims=True)[:, None, :]
    query_std = np.maximum(query_lookback.std(axis=-1, keepdims=True), eps)[:, None, :]
    neighbor_mean = neighbor_lookback.mean(axis=-1, keepdims=True)
    neighbor_std = np.maximum(neighbor_lookback.std(axis=-1, keepdims=True), eps)
    if residual:
        return neighbor_value / neighbor_std * query_std
    return (neighbor_value - neighbor_mean) / neighbor_std * query_std + query_mean


def scalar_feature_values(data: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    x = np.asarray(arrays["x"], dtype=np.float64)
    x_c = np.asarray(arrays["x_c"], dtype=np.float64)
    pred = np.asarray(predictions["vanilla"], dtype=np.float64)
    pred_c = np.asarray(predictions["context_forecast"], dtype=np.float64)
    context_delta = pred_c - pred
    features: dict[str, np.ndarray] = {
        "context_minus_vanilla_mean": np.nanmean(context_delta, axis=1),
        "context_minus_vanilla_std": np.nanstd(context_delta, axis=1),
        "query_mean": np.nanmean(x, axis=1),
        "query_std": np.nanstd(x, axis=1),
        "neighbor_lookback_means_mean_raw": np.nanmean(np.nanmean(x_c, axis=-1), axis=1),
        "neighbor_lookback_means_std_raw": np.nanstd(np.nanmean(x_c, axis=-1), axis=1),
        "neighbor_lookback_stds_mean_raw": np.nanmean(np.nanstd(x_c, axis=-1), axis=1),
        "neighbor_lookback_stds_std_raw": np.nanstd(np.nanstd(x_c, axis=-1), axis=1),
        "same_user_ratio": np.nanmean(
            arrays["neighbor_user_idx"] == arrays["query_user_idx"][:, None],
            axis=1,
        ),
        "neighbor_age_mean": np.nanmean(
            arrays["query_t"][:, None] - arrays["neighbor_t"],
            axis=1,
        ),
    }

    distance = arrays.get("distance")
    if distance is not None:
        distance = np.asarray(distance, dtype=np.float64)
        weights = _distance_weights(distance)
        features.update(
            {
                "neighbor_weight_std": np.nanstd(weights, axis=1),
                "neighbor_weight_max": np.nanmax(weights, axis=1),
                "distance_mean": np.nanmean(distance, axis=1),
            }
        )
        if arrays.get("e") is not None:
            y_c_scaled = _neighbor_to_query_scale_np(x, x_c, np.asarray(arrays["y_c"], dtype=np.float64))
            e_scaled = _neighbor_to_query_scale_np(
                x,
                x_c,
                np.asarray(arrays["e"], dtype=np.float64),
                residual=True,
            )
            weighted = np.nansum(weights[:, :, None] * y_c_scaled, axis=1)
            weighted_e = np.nansum(weights[:, :, None] * e_scaled, axis=1)
            features.update(
                {
                    "weighted_neighbor_minus_vanilla_mean": np.nanmean(weighted - pred, axis=1),
                    "weighted_neighbor_residual_mean": np.nanmean(weighted_e, axis=1),
                }
            )

    return {name: features[name] for name in SCALAR_FEATURE_ORDER if name in features}


def scalar_feature_names(data: dict[str, Any], split: str) -> list[str]:
    return list(scalar_feature_values(data, split))


def _prediction_metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    lookback: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, str]:
    if metric == "difference":
        return prediction - target, "Difference"
    if metric == "mse":
        return (prediction - target) ** 2, "MSE"
    if metric == "nmse":
        scale = np.maximum(lookback.std(axis=1, keepdims=True), 1e-8)
        return ((prediction - target) / scale) ** 2, "nMSE"
    raise ValueError(f"Unknown metric: {metric}")


def _safe_relative_delta(selected: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denominator = np.where(np.abs(reference) > 1e-12, reference, np.nan)
    return (selected - reference) / denominator


def _format_metric_value(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if 1e-3 <= magnitude < 1e4:
        return f"{value:.4g}"
    return f"{value:.3e}"


def horizon_values(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> tuple[np.ndarray, str, float, float | None]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    selected = np.asarray(predictions[prediction_name], dtype=np.float64)
    reference = np.asarray(predictions[reference_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)

    selected_metric, metric_label = _prediction_metric_values(selected, target, x, metric)
    reference_metric, _ = _prediction_metric_values(reference, target, x, metric)
    selected_horizon = np.nanmean(selected_metric, axis=0)
    reference_horizon = np.nanmean(reference_metric, axis=0)
    if view == "direct":
        values = selected_horizon
        ylabel = metric_label
    elif view == "improvement":
        values = selected_horizon - reference_horizon
        ylabel = f"{metric_label} delta vs {reference_name}"
    elif view == "relative":
        values = _safe_relative_delta(selected_horizon, reference_horizon)
        ylabel = f"Relative {metric_label} delta vs {reference_name}"
    else:
        raise ValueError(f"Unknown view: {view}")

    average = float(np.nanmean(values))
    window_average = None
    if view == "relative":
        selected_window = np.nanmean(selected_metric, axis=1)
        reference_window = np.nanmean(reference_metric, axis=1)
        window_average = float(np.nanmean(_safe_relative_delta(selected_window, reference_window)))
    return values, ylabel, average, window_average


def window_metric_values(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> tuple[np.ndarray, str, float]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    selected = np.asarray(predictions[prediction_name], dtype=np.float64)
    reference = np.asarray(predictions[reference_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)

    selected_metric, metric_label = _prediction_metric_values(selected, target, x, metric)
    reference_metric, _ = _prediction_metric_values(reference, target, x, metric)
    selected_window = np.nanmean(selected_metric, axis=1)
    reference_window = np.nanmean(reference_metric, axis=1)
    if view == "direct":
        values = selected_window
        ylabel = metric_label
    elif view == "improvement":
        values = selected_window - reference_window
        ylabel = f"{metric_label} delta vs {reference_name}"
    elif view == "relative":
        values = _safe_relative_delta(selected_window, reference_window)
        ylabel = f"Relative {metric_label} delta vs {reference_name}"
    else:
        raise ValueError(f"Unknown view: {view}")
    return values, ylabel, float(np.nanmean(values))


def plot_window_metric_scatter(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
    scalar_feature_name: str,
    *,
    max_points: int = 5000,
) -> plt.Figure:
    feature_map = scalar_feature_values(data, split)
    if scalar_feature_name not in feature_map:
        raise KeyError(f"Unknown scalar feature {scalar_feature_name!r}")
    x_values = np.asarray(feature_map[scalar_feature_name], dtype=np.float64)
    y_values, ylabel, average = window_metric_values(
        data,
        split,
        prediction_name,
        reference_name,
        metric,
        view,
    )
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(len(x_values), size=max_points, replace=False)
        x_values = x_values[keep]
        y_values = y_values[keep]
    correlation = (
        float(np.corrcoef(x_values, y_values)[0, 1])
        if len(x_values) > 1 and np.std(x_values) > 0.0 and np.std(y_values) > 0.0
        else float("nan")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_values, y_values, s=10, alpha=0.35, linewidths=0)
    if view != "direct" or metric == "difference":
        ax.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel(scalar_feature_name)
    ax.set_ylabel(ylabel)
    title = f"{split}: {prediction_name} - {metric} ({view})"
    if view != "direct":
        title += f" vs {reference_name}"
    title += (
        f"\navg={_format_metric_value(average)}; "
        f"r={_format_metric_value(correlation)}; n={len(x_values):,}"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_horizon(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    view: str,
) -> plt.Figure:
    values, ylabel, average, window_average = horizon_values(
        data,
        split,
        prediction_name,
        reference_name,
        metric,
        view,
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=2.2)
    if view != "direct" or metric == "difference":
        ax.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    title = f"{split}: {prediction_name} - {metric} ({view})"
    if view != "direct":
        title += f" vs {reference_name}"
    if view == "relative":
        title += (
            f"\nhorizon avg={_format_metric_value(average)}; "
            f"window avg={_format_metric_value(float(window_average))}"
        )
    else:
        title += f"\navg={_format_metric_value(average)}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def gate_options(data: dict[str, Any], split: str) -> list[tuple[str, str]]:
    diagnostics = split_arrays(data, split)["gate_diagnostics"]
    options: list[tuple[str, str]] = []
    for objective in ("classifier", "regressor"):
        if f"{objective}_scalar_score" in diagnostics:
            options.append((f"{objective} scalar gate", f"{objective}_scalar"))
        horizon_key = f"{objective}_horizon_score"
        if horizon_key in diagnostics:
            options.append((f"{objective} horizon gate (all horizons)", f"{objective}_horizon_all"))
    # Backward-compatible options for artifacts produced before gate families
    # were split into classifier and regressor variants.
    if "scalar_score" in diagnostics:
        options.append(("scalar gate", "scalar"))
    if "horizon_score" in diagnostics:
        options.append(("horizon gate (all horizons)", "horizon_all"))
    return options


def _gate_score_target(data: dict[str, Any], split: str, gate_name: str) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = split_arrays(data, split)["gate_diagnostics"]
    if gate_name == "scalar":
        score = diagnostics["scalar_score"].reshape(-1)
        target = diagnostics["scalar_target"].reshape(-1)
    elif gate_name == "horizon_all":
        score = diagnostics["horizon_score"]
        target = diagnostics["horizon_target"]
    elif gate_name.startswith("horizon_"):
        horizon = int(gate_name.rsplit("_", 1)[1])
        score = diagnostics["horizon_score"][:, horizon]
        target = diagnostics["horizon_target"][:, horizon]
    elif gate_name.endswith("_scalar"):
        objective = gate_name.removesuffix("_scalar")
        score = diagnostics[f"{objective}_scalar_score"].reshape(-1)
        target = diagnostics[f"{objective}_scalar_target"].reshape(-1)
    elif gate_name.endswith("_horizon_all"):
        objective = gate_name.removesuffix("_horizon_all")
        score = diagnostics[f"{objective}_horizon_score"]
        target = diagnostics[f"{objective}_horizon_target"]
    elif "_horizon_" in gate_name:
        objective, horizon_text = gate_name.rsplit("_horizon_", 1)
        horizon = int(horizon_text)
        score = diagnostics[f"{objective}_horizon_score"][:, horizon]
        target = diagnostics[f"{objective}_horizon_target"][:, horizon]
    else:
        raise ValueError(gate_name)
    return np.asarray(score, dtype=np.float64), np.asarray(target, dtype=np.float64)


def gate_roc(
    data: dict[str, Any],
    split: str,
    gate_name: str,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    score, target = _gate_score_target(data, split, gate_name)
    score = score.reshape(-1)
    target = target.reshape(-1)
    finite = np.isfinite(score) & np.isfinite(target)
    score = np.asarray(score[finite], dtype=np.float64)
    label = np.asarray(target[finite] > 0.0, dtype=bool)
    accuracy = float(np.mean((score > 0.0) == label)) if len(label) else float("nan")
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan"), accuracy, len(label)
    order = np.argsort(-score, kind="stable")
    sorted_label = label[order]
    threshold_indices = np.r_[np.where(np.diff(score[order]))[0], len(score) - 1]
    tp = np.cumsum(sorted_label)[threshold_indices]
    fp = np.cumsum(~sorted_label)[threshold_indices]
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) * 0.5))
    return fpr, tpr, auc, accuracy, len(label)


def plot_gate_roc(data: dict[str, Any], split: str, gate_name: str) -> tuple[plt.Figure, dict[str, float]]:
    fpr, tpr, auc, accuracy, count = gate_roc(data, split, gate_name)
    fig, ax = plt.subplots(figsize=(6, 5))
    label = f"ROC (AUC={auc:.3f})" if np.isfinite(auc) else "ROC (one class only)"
    ax.plot(fpr, tpr, linewidth=2.2, label=label)
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(f"{split}: {gate_name}; accuracy={accuracy:.3f}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, {"auc": auc, "accuracy": accuracy, "count": float(count)}


def gate_prediction_names(data: dict[str, Any], split: str) -> list[str]:
    names = prediction_names(data, split)
    prefixes = ("bayes_context_", "catboost_context_", "oracle_context_")
    return [name for name in names if name.startswith(prefixes)]


def _gate_shape(name: str) -> str:
    return "horizon" if name.endswith("_horizon") else "scalar"


def _gate_right_percent(arrays: dict[str, Any], prediction_name: str) -> float:
    predictions = arrays["predictions"]
    prediction = np.asarray(predictions[prediction_name], dtype=np.float64)
    vanilla = np.asarray(predictions["vanilla"], dtype=np.float64)
    context = np.asarray(predictions["context_forecast"], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    base_loss = (vanilla - target) ** 2
    context_loss = (context - target) ** 2
    distance_to_context = np.abs(prediction - context)
    distance_to_vanilla = np.abs(prediction - vanilla)
    if _gate_shape(prediction_name) == "scalar":
        decision = np.nanmean(distance_to_context, axis=1) <= np.nanmean(distance_to_vanilla, axis=1)
        target_context = np.nanmean(context_loss, axis=1) < np.nanmean(base_loss, axis=1)
        non_tie = np.abs(np.nanmean(base_loss - context_loss, axis=1)) > 1e-12
    else:
        decision = distance_to_context <= distance_to_vanilla
        target_context = context_loss < base_loss
        non_tie = np.abs(base_loss - context_loss) > 1e-12
    finite = np.isfinite(decision) & np.isfinite(target_context) & non_tie
    return float(100.0 * np.mean(decision[finite] == target_context[finite])) if np.any(finite) else float("nan")


def _nmse_for_prediction(arrays: dict[str, Any], prediction_name: str) -> float:
    prediction = np.asarray(arrays["predictions"][prediction_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)
    values, _ = _prediction_metric_values(prediction, target, x, "nmse")
    return float(np.nanmean(values))


def gate_summary_rows(data: dict[str, Any], split: str) -> list[dict[str, float | str]]:
    arrays = split_arrays(data, split)
    rows: list[dict[str, float | str]] = []
    for name in gate_prediction_names(data, split):
        rows.append(
            {
                "name": name,
                "shape": _gate_shape(name),
                "right_pct": _gate_right_percent(arrays, name),
                "nmse": _nmse_for_prediction(arrays, name),
            }
        )
    return rows


def gate_summary_html(rows: list[dict[str, float | str]]) -> str:
    if not rows:
        return "<b>No gate or oracle predictions found.</b>"
    header = "<tr><th>gate/oracle</th><th>shape</th><th>% right</th><th>nMSE</th></tr>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{escape(str(row['name']))}</td>"
            f"<td>{escape(str(row['shape']))}</td>"
            f"<td>{_format_metric_value(float(row['right_pct']))}</td>"
            f"<td>{_format_metric_value(float(row['nmse']))}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 6px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}</style>"
        + header
        + "".join(body)
        + "</table>"
    )


def gate_threshold_sweep(
    data: dict[str, Any],
    split: str,
    gate_name: str,
    *,
    points: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = split_arrays(data, split)
    score, target = _gate_score_target(data, split, gate_name)
    score = np.asarray(score, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite_score = score[np.isfinite(score)]
    if finite_score.size == 0:
        raise ValueError(f"No finite gate scores for {gate_name}")
    thresholds = np.linspace(float(np.nanmin(finite_score)), float(np.nanmax(finite_score)), points)
    vanilla = np.asarray(arrays["predictions"]["vanilla"], dtype=np.float64)
    context = np.asarray(arrays["predictions"]["context_forecast"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)
    right_pct = np.empty_like(thresholds)
    nmse = np.empty_like(thresholds)
    target_label = target > 0.0
    finite_target = np.isfinite(score) & np.isfinite(target)
    scalar_gate = score.ndim == 1
    for index, threshold in enumerate(thresholds):
        decision = score > threshold
        if scalar_gate:
            right_mask = finite_target
            prediction = np.where(decision[:, None], context, vanilla)
        else:
            right_mask = finite_target
            prediction = np.where(decision, context, vanilla)
        right_pct[index] = (
            100.0 * np.mean(decision[right_mask] == target_label[right_mask])
            if np.any(right_mask)
            else np.nan
        )
        metric_values, _ = _prediction_metric_values(prediction, y, x, "nmse")
        nmse[index] = np.nanmean(metric_values)
    return thresholds, right_pct, nmse


def plot_gate_threshold_sweep(data: dict[str, Any], split: str, gate_name: str) -> plt.Figure:
    thresholds, right_pct, nmse = gate_threshold_sweep(data, split, gate_name)
    fig, (ax_right, ax_nmse) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_right.plot(thresholds, right_pct, linewidth=2.1)
    ax_right.axvline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_right.set_xlabel("Decision threshold")
    ax_right.set_ylabel("% right")
    ax_right.grid(True, alpha=0.25)
    ax_nmse.plot(thresholds, nmse, linewidth=2.1, color="tab:orange")
    ax_nmse.axvline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax_nmse.set_xlabel("Decision threshold")
    ax_nmse.set_ylabel("nMSE")
    ax_nmse.grid(True, alpha=0.25)
    fig.suptitle(f"{split}: {gate_name} threshold sweep")
    fig.tight_layout()
    return fig


def _preview_names(names: list[str], limit: int = 6) -> str:
    names = list(names)
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f", ... (+{len(names) - limit} more)"


def data_summary(data: dict[str, Any]) -> str:
    lines = [
        "Loaded splits: " + ", ".join(available_splits(data)),
        f"Baseline visualization payload: {data['paths']['baseline']} {data['paths']['baseline'].exists()}",
        f"TS-IFA prediction payload: {data['paths']['ts_ifa']} {data['paths']['ts_ifa'].exists()}",
    ]
    for split in available_splits(data):
        arrays = split_arrays(data, split)
        names = list(arrays["predictions"])
        lines.append(
            f"{split}: {len(arrays['x']):,} queries; "
            f"{len(names)} quantity options: {_preview_names(names)}"
        )
    return "\n".join(lines)


def _notebook_runtime() -> tuple[Any, Any, Any]:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    return widgets, clear_output, display


def _default_split(splits: list[str]) -> str:
    return "eval" if "eval" in splits else splits[0]


def _default_prediction_name(names: list[str]) -> str:
    return "vanilla" if "vanilla" in names else names[0]


def _default_scalar_feature(names: list[str]) -> str:
    return "distance_mean" if "distance_mean" in names else names[0]


def query_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    query_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    query_sample = widgets.Button(description="random query", icon="random", button_style="primary")
    query_normalized = widgets.ToggleButton(value=False, description="instance normalized", icon="exchange")
    query_hide_axes = widgets.ToggleButton(value=False, description="remove axes", icon="eye-slash")
    query_output = widgets.Output()
    query_rng = np.random.default_rng()
    query_state = {"index": 0}

    def draw_query(*_: Any) -> None:
        with query_output:
            clear_output(wait=True)
            fig = plot_query_example(
                data,
                query_split.value,
                query_state["index"],
                instance_normalized=query_normalized.value,
                hide_axes=query_hide_axes.value,
            )
            display(fig)
            plt.close(fig)

    def sample_query(*_: Any) -> None:
        count = len(split_arrays(data, query_split.value)["x"])
        query_state["index"] = int(query_rng.integers(count))
        draw_query()

    query_sample.on_click(sample_query)
    query_split.observe(sample_query, names="value")
    query_normalized.observe(draw_query, names="value")
    query_hide_axes.observe(draw_query, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([query_split, query_sample]),
            widgets.HBox([query_normalized, query_hide_axes]),
            query_output,
        ]
    )
    sample_query()
    return section


def window_scatter_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    scatter_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    scatter_names = prediction_names(data, scatter_split.value)
    scatter_features = scalar_feature_names(data, scatter_split.value)
    scatter_prediction = widgets.Dropdown(
        options=scatter_names,
        value=_default_prediction_name(scatter_names),
        description="quantity:",
    )
    scatter_reference = widgets.Dropdown(
        options=scatter_names,
        value=_default_prediction_name(scatter_names),
        description="relative to:",
    )
    scatter_reference_box = widgets.HBox([scatter_reference])
    scatter_metric = widgets.Dropdown(options=["mse", "nmse", "difference"], value="mse", description="metric:")
    scatter_view = widgets.Dropdown(
        options=["direct", "improvement", "relative"],
        value="direct",
        description="view:",
    )
    scatter_feature = widgets.Dropdown(
        options=scatter_features or [""],
        value=_default_scalar_feature(scatter_features) if scatter_features else "",
        description="x:",
    )
    scatter_help = widgets.HTML(
        value=(
            "<small>Each point is one query window. The y-axis first averages L_i,h over horizons "
            "for that window; relative view is then computed per window as "
            "(L_i(y')-L_i(y''))/L_i(y'').</small>"
        )
    )
    scatter_output = widgets.Output()

    def update_scatter_controls(*_: Any, redraw: bool = True) -> None:
        scatter_reference_box.layout.display = (
            "" if scatter_view.value in {"improvement", "relative"} else "none"
        )
        if redraw:
            draw_scatter()

    def draw_scatter(*_: Any) -> None:
        update_scatter_controls(redraw=False)
        with scatter_output:
            clear_output(wait=True)
            if not scatter_feature.value:
                print("No scalar features available for this split.")
                return
            fig = plot_window_metric_scatter(
                data,
                scatter_split.value,
                scatter_prediction.value,
                scatter_reference.value,
                scatter_metric.value,
                scatter_view.value,
                scatter_feature.value,
            )
            display(fig)
            plt.close(fig)

    def update_scatter_names(*_: Any) -> None:
        names = prediction_names(data, scatter_split.value)
        features = scalar_feature_names(data, scatter_split.value)
        scatter_prediction.options = names
        scatter_reference.options = names
        scatter_feature.options = features or [""]
        scatter_prediction.value = _default_prediction_name(names)
        scatter_reference.value = _default_prediction_name(names)
        scatter_feature.value = _default_scalar_feature(features) if features else ""
        draw_scatter()

    scatter_split.observe(update_scatter_names, names="value")
    for control in [scatter_prediction, scatter_reference, scatter_metric, scatter_view, scatter_feature]:
        control.observe(draw_scatter, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([scatter_split, scatter_prediction]),
            widgets.HBox([scatter_metric, scatter_view, scatter_feature]),
            scatter_reference_box,
            scatter_help,
            scatter_output,
        ]
    )
    draw_scatter()
    return section


def horizon_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    horizon_split = widgets.Dropdown(options=splits, value=_default_split(splits), description="split:")
    initial_names = prediction_names(data, horizon_split.value)
    horizon_prediction = widgets.Dropdown(
        options=initial_names,
        value=_default_prediction_name(initial_names),
        description="quantity:",
    )
    horizon_reference = widgets.Dropdown(
        options=initial_names,
        value=_default_prediction_name(initial_names),
        description="relative to:",
    )
    horizon_reference_box = widgets.HBox([horizon_reference])
    horizon_metric = widgets.Dropdown(options=["mse", "nmse", "difference"], value="mse", description="metric:")
    horizon_view = widgets.Dropdown(options=["direct", "improvement", "relative"], value="direct", description="view:")
    horizon_help = widgets.HTML(
        value=(
            "<small>Metrics are computed per sample and horizon against ground truth: "
            "mse=(y'-y)^2, nmse=((y'-y)/std(lookback))^2, difference=y'-y. "
            "direct plots mean_i L_i,h(y'), improvement plots mean_i L_i,h(y')-mean_i L_i,h(y''), "
            "and relative applies (A-B)/B after those horizon-wise means. "
            "The title reports the mean of the plotted horizon values; relative view also reports "
            "the per-window relative mean.</small>"
        )
    )
    horizon_output = widgets.Output()

    def update_horizon_controls(*_: Any, redraw: bool = True) -> None:
        horizon_reference_box.layout.display = "" if horizon_view.value in {"improvement", "relative"} else "none"
        if redraw:
            draw_horizon()

    def draw_horizon(*_: Any) -> None:
        update_horizon_controls(redraw=False)
        with horizon_output:
            clear_output(wait=True)
            fig = plot_horizon(
                data,
                horizon_split.value,
                horizon_prediction.value,
                horizon_reference.value,
                horizon_metric.value,
                horizon_view.value,
            )
            display(fig)
            plt.close(fig)

    def update_horizon_names(*_: Any) -> None:
        names = prediction_names(data, horizon_split.value)
        horizon_prediction.options = names
        horizon_reference.options = names
        horizon_prediction.value = _default_prediction_name(names)
        horizon_reference.value = _default_prediction_name(names)
        draw_horizon()

    horizon_split.observe(update_horizon_names, names="value")
    for control in [horizon_prediction, horizon_reference, horizon_metric, horizon_view]:
        control.observe(draw_horizon, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([horizon_split, horizon_prediction]),
            widgets.HBox([horizon_metric, horizon_view]),
            horizon_reference_box,
            horizon_help,
            horizon_output,
        ]
    )
    draw_horizon()
    return section


def gates_section(data: dict[str, Any]) -> Any:
    widgets, clear_output, display = _notebook_runtime()
    splits = available_splits(data)
    gate_splits = [split for split in splits if gate_summary_rows(data, split) or gate_options(data, split)]
    gate_split = widgets.Dropdown(
        options=gate_splits or [""],
        value=(gate_splits or [""])[0],
        description="split:",
    )
    initial_gate_options = gate_options(data, gate_split.value) if gate_split.value else [("no scored gate", "")]
    gate_choice = widgets.Dropdown(
        options=initial_gate_options or [("no scored gate", "")],
        description="gate:",
    )
    gate_table = widgets.HTML()
    roc_summary = widgets.HTML()
    roc_output = widgets.Output()
    threshold_output = widgets.Output()

    def refresh_gate_table() -> None:
        rows = gate_summary_rows(data, gate_split.value) if gate_split.value else []
        gate_table.value = gate_summary_html(rows)

    def draw_gates(*_: Any) -> None:
        refresh_gate_table()
        with roc_output:
            clear_output(wait=True)
            if not gate_split.value or not gate_choice.value:
                roc_summary.value = "<b>No saved gate diagnostics.</b> Run evaluate_baselines first."
            else:
                fig, metrics = plot_gate_roc(data, gate_split.value, gate_choice.value)
                auc_text = f"{metrics['auc']:.4f}" if np.isfinite(metrics["auc"]) else "undefined (one class)"
                roc_summary.value = (
                    f"<b>Threshold=0 % right:</b> {100 * metrics['accuracy']:.2f}% &nbsp; "
                    f"<b>ROC AUC:</b> {auc_text} &nbsp; "
                    f"<b>Decisions:</b> {int(metrics['count']):,}"
                )
                display(fig)
                plt.close(fig)
        with threshold_output:
            clear_output(wait=True)
            if gate_split.value and gate_choice.value:
                fig = plot_gate_threshold_sweep(data, gate_split.value, gate_choice.value)
                display(fig)
                plt.close(fig)

    def update_gate_options(*_: Any) -> None:
        options = gate_options(data, gate_split.value) if gate_split.value else [("no scored gate", "")]
        gate_choice.options = options or [("no scored gate", "")]
        draw_gates()

    gate_split.observe(update_gate_options, names="value")
    gate_choice.observe(draw_gates, names="value")
    section = widgets.VBox(
        [
            widgets.HBox([gate_split, gate_choice]),
            gate_table,
            roc_summary,
            roc_output,
            threshold_output,
        ]
    )
    draw_gates()
    return section
