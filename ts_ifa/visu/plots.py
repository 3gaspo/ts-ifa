"""Matplotlib-only visualizations for lightweight extraction outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def symlog(values, linthresh: float = 1.0):
    arr = np.asarray(values, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr / linthresh)) * linthresh


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_series(
    series: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    filename: str = "series.png",
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, values in series.items():
        ax.plot(np.asarray(values, dtype=float), label=str(name))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_prediction_example(
    lookback: Sequence[float],
    target: Sequence[float],
    prediction: Sequence[float],
    output_dir: str | Path,
    filename: str = "prediction_example.png",
    title: str | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(output_dir)
    x = np.asarray(lookback, dtype=float)
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    lags = len(x)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(lags), x, label="lookback")
    ax.plot(np.arange(lags, lags + len(y)), y, label="target")
    ax.plot(np.arange(lags, lags + len(p)), p, label="prediction")
    ax.axvline(lags - 1, color="black", linestyle=":", linewidth=1)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_user_error_scatter(
    frame: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "user_errors.png",
    *,
    mean_col: str = "mean_loss",
    std_col: str = "std_loss",
    title: str = "Per-user errors",
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(output_dir)
    df = frame[[mean_col, std_col]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(symlog(df[mean_col]), symlog(df[std_col]), s=24, alpha=0.8)
    ax.set_xlabel(f"symlog({mean_col})")
    ax.set_ylabel(f"symlog({std_col})")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_horizon_errors(
    losses_by_horizon: Sequence[float],
    output_dir: str | Path,
    filename: str = "horizon_errors.png",
    title: str = "Mean loss by horizon",
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(output_dir)
    values = np.asarray(losses_by_horizon, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(values)), values)
    ax.set_xlabel("horizon step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_error_distribution(
    losses: Sequence[float],
    output_dir: str | Path,
    filename: str = "loss_distribution.png",
    title: str = "Loss distribution",
) -> Path:
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(output_dir)
    values = np.asarray(losses, dtype=float)
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(symlog(values), bins=60)
    ax.set_xlabel("symlog(loss)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_feature_scatter(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_dir: str | Path,
    filename: str,
    *,
    title: str | None = None,
    symlog_axes: bool = True,
) -> Path | None:
    import matplotlib.pyplot as plt

    if x_col not in frame.columns or y_col not in frame.columns:
        return None
    out_dir = _ensure_dir(output_dir)
    df = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return None
    x = symlog(df[x_col]) if symlog_axes else df[x_col].to_numpy()
    y = symlog(df[y_col]) if symlog_axes else df[y_col].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=12, alpha=0.45)
    prefix = "symlog" if symlog_axes else ""
    ax.set_xlabel(f"{prefix}({x_col})" if prefix else x_col)
    ax.set_ylabel(f"{prefix}({y_col})" if prefix else y_col)
    ax.set_title(title or f"{y_col} vs {x_col}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
