"""Artifact-only helpers for the interactive retrieval dashboard notebook."""

from __future__ import annotations

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
    dates, users = x_tensor.shape[:2]
    arrays: dict[str, Any] = {
        "x": _flatten(x_tensor),
        "y": _flatten(y_tensor),
        "x_c": _flatten(x_c_tensor),
        "y_c": _flatten(y_c_tensor),
        "query_t": payload[prefix + "query_t"].reshape(-1).numpy(),
        "query_user_idx": payload[prefix + "query_user_idx"].reshape(-1).numpy(),
        "neighbor_t": _flatten(payload[prefix + "neighbor_t"]),
        "neighbor_user_idx": _flatten(payload[prefix + "neighbor_user_idx"]),
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
        ax.plot(future_axis, y_c[neighbor], color=color, alpha=0.72, linewidth=1.2, linestyle="--")
    ax.plot(past_axis, x, color="black", linewidth=2.6, label="query lookback")
    ax.plot(future_axis, y, color="black", linewidth=2.6, linestyle="--", label="query future")
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


def horizon_values(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    *,
    instance_normalized: bool,
) -> tuple[np.ndarray, str]:
    arrays = split_arrays(data, split)
    predictions = arrays["predictions"]
    selected = np.asarray(predictions[prediction_name], dtype=np.float64)
    reference = np.asarray(predictions[reference_name], dtype=np.float64)
    target = np.asarray(arrays["y"], dtype=np.float64)
    x = np.asarray(arrays["x"], dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    scale = np.maximum(x.std(axis=1, keepdims=True), 1e-8)
    if instance_normalized:
        selected = (selected - mean) / scale
        reference = (reference - mean) / scale
        target = (target - mean) / scale

    if metric == "raw prediction":
        return np.nanmean(selected, axis=0), "Mean prediction"
    if metric == "direct difference":
        return np.nanmean(selected - reference, axis=0), f"Mean {prediction_name} - {reference_name}"
    selected_sq = (selected - target) ** 2
    reference_sq = (reference - target) ** 2
    if metric == "mse":
        return np.nanmean(selected_sq, axis=0), "MSE"
    if metric == "nmse":
        if instance_normalized:
            return np.nanmean(selected_sq, axis=0), "nMSE"
        return np.nanmean(((selected - target) / scale) ** 2, axis=0), "nMSE"
    if metric == "relative mse":
        selected_mse = np.nanmean(selected_sq, axis=0)
        reference_mse = np.nanmean(reference_sq, axis=0)
        values = 100.0 * (reference_mse - selected_mse) / np.maximum(reference_mse, 1e-12)
        return values, f"MSE improvement vs {reference_name} (%)"
    raise ValueError(f"Unknown metric: {metric}")


def plot_horizon(
    data: dict[str, Any],
    split: str,
    prediction_name: str,
    reference_name: str,
    metric: str,
    *,
    instance_normalized: bool,
    hide_axes: bool,
) -> plt.Figure:
    values, ylabel = horizon_values(
        data,
        split,
        prediction_name,
        reference_name,
        metric,
        instance_normalized=instance_normalized,
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(np.arange(1, len(values) + 1), values, linewidth=2.2)
    if metric in {"direct difference", "relative mse"}:
        ax.axhline(0.0, color="0.4", linewidth=1, linestyle="--")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{split}: {prediction_name} — {metric}")
    ax.grid(True, alpha=0.25)
    if hide_axes:
        ax.axis("off")
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
            options.append((f"{objective} horizon gate (all)", f"{objective}_horizon_all"))
            options.extend(
                (f"{objective} horizon gate (h={index + 1})", f"{objective}_horizon_{index}")
                for index in range(diagnostics[horizon_key].shape[1])
            )
    # Backward-compatible options for artifacts produced before gate families
    # were split into classifier and regressor variants.
    if "scalar_score" in diagnostics:
        options.append(("scalar gate", "scalar"))
    if "horizon_score" in diagnostics:
        options.append(("horizon gate (all horizons)", "horizon_all"))
        options.extend(
            (f"horizon gate (h={index + 1})", f"horizon_{index}")
            for index in range(diagnostics["horizon_score"].shape[1])
        )
    return options


def gate_roc(
    data: dict[str, Any],
    split: str,
    gate_name: str,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    diagnostics = split_arrays(data, split)["gate_diagnostics"]
    if gate_name == "scalar":
        score = diagnostics["scalar_score"].reshape(-1)
        target = diagnostics["scalar_target"].reshape(-1)
    elif gate_name == "horizon_all":
        score = diagnostics["horizon_score"].reshape(-1)
        target = diagnostics["horizon_target"].reshape(-1)
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
        score = diagnostics[f"{objective}_horizon_score"].reshape(-1)
        target = diagnostics[f"{objective}_horizon_target"].reshape(-1)
    elif "_horizon_" in gate_name:
        objective, horizon_text = gate_name.rsplit("_horizon_", 1)
        horizon = int(horizon_text)
        score = diagnostics[f"{objective}_horizon_score"][:, horizon]
        target = diagnostics[f"{objective}_horizon_target"][:, horizon]
    else:
        raise ValueError(gate_name)
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
