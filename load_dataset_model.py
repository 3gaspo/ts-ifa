"""CSV dataset and pretrained-model loading for extraction experiments.

This module is intentionally small. It keeps only the pieces needed by the
neighbor retrieval scripts: a date-indexed CSV, deterministic window slicing,
optional global past/future covariates, and loading a pretrained forecaster.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from .models import load_model
except ImportError:  # pragma: no cover - direct script execution
    from models import load_model


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    name = "auto" if device is None else str(device).lower()
    if name in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if name in {"gpu", "cuda"}:
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cpu")
    return torch.device(name)


def _split_text(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return _split_text(value)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _select_columns(df: pd.DataFrame, columns: Sequence[Any] | str | None) -> list[str]:
    selected = []
    for item in _as_list(columns):
        if isinstance(item, int) or (isinstance(item, str) and item.lstrip("-").isdigit()):
            selected.append(str(df.columns[int(item)]))
        else:
            selected.append(str(item))
    missing = [col for col in selected if col not in df.columns]
    if missing:
        raise KeyError(f"CSV columns not found: {missing}")
    return selected


def _drop_users(df: pd.DataFrame, drop_users: Sequence[Any] | str | None) -> pd.DataFrame:
    columns = []
    for item in _as_list(drop_users):
        if isinstance(item, int) or (isinstance(item, str) and item.lstrip("-").isdigit()):
            idx = int(item)
            if idx < 0 or idx >= len(df.columns):
                raise IndexError(f"drop user index out of range: {idx}")
            columns.append(df.columns[idx])
        else:
            columns.append(str(item))
    return df.drop(columns=columns) if columns else df


def _aggregate(df: pd.DataFrame, aggr: str | None, period: str) -> pd.DataFrame:
    if aggr is None or str(aggr).lower() in {"", "none"}:
        return df
    name = str(aggr).lower()
    if name == "sum":
        return df.resample(period).sum()
    if name == "mean":
        return df.resample(period).mean()
    if name == "last":
        return df.resample(period).last()
    if name == "first":
        return df.resample(period).first()
    if name == "asfreq":
        return df.asfreq(period)
    raise ValueError(f"unknown aggregation {aggr!r}")


def resolve_csv_path(path: str | Path, dataset_name: str | None = None) -> Path:
    base = Path(path).expanduser()
    if base.suffix.lower() == ".csv":
        return base.resolve()
    if dataset_name:
        return (base / f"{dataset_name}.csv").resolve()
    matches = sorted(base.glob("*.csv"))
    if len(matches) == 1:
        return matches[0].resolve()
    raise ValueError("pass a CSV file or a directory with dataset_name")


@dataclass
class CsvTimeSeries:
    """Date x user values plus optional global covariates."""

    frame: pd.DataFrame
    past_covariates: pd.DataFrame | None = None
    future_covariates: pd.DataFrame | None = None

    @property
    def values(self) -> np.ndarray:
        return self.frame.to_numpy(dtype=np.float32)

    @property
    def datetimes(self) -> list[Any]:
        return list(self.frame.index)

    @property
    def user_names(self) -> list[str]:
        return [str(col) for col in self.frame.columns]

    @property
    def n_dates(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_users(self) -> int:
        return int(self.frame.shape[1])

    def validate_window(self, start: int, lags: int, horizon: int) -> None:
        stop = int(start) + int(lags) + int(horizon)
        if start < 0 or stop > self.n_dates:
            raise ValueError(
                f"window [{start}, {stop}) is outside dataset with {self.n_dates} dates"
            )

    def window_tensor(
        self,
        start: int,
        lags: int,
        horizon: int,
        *,
        users: Sequence[int] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``x`` and ``y`` shaped ``(users, 1, time)``."""
        self.validate_window(start, lags, horizon)
        values = self.values[start : start + lags + horizon]
        if users is not None:
            values = values[:, list(users)]
        arr = torch.as_tensor(values.T.copy(), dtype=torch.float32, device=device)
        x = arr[:, None, :lags]
        y = arr[:, None, lags:]
        return x, y

    def covariate_tensors(
        self,
        start: int,
        lags: int,
        horizon: int,
        *,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return optional global ``past`` and ``future`` covariates.

        Shapes are ``(1, channels, lags)`` and ``(1, channels, horizon)`` so
        models can share them across all users.
        """
        self.validate_window(start, lags, horizon)
        past = None
        future = None
        if self.past_covariates is not None:
            values = self.past_covariates.iloc[start : start + lags].to_numpy(
                dtype=np.float32
            )
            past = torch.as_tensor(
                values.T[None, :, :].copy(),
                dtype=torch.float32,
                device=device,
            )
        if self.future_covariates is not None:
            values = self.future_covariates.iloc[
                start + lags : start + lags + horizon
            ].to_numpy(dtype=np.float32)
            future = torch.as_tensor(
                values.T[None, :, :].copy(),
                dtype=torch.float32,
                device=device,
            )
        return past, future


def load_csv_dataset(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    target_cols: Sequence[Any] | str | None = None,
    past_covariate_cols: Sequence[Any] | str | None = None,
    future_covariate_cols: Sequence[Any] | str | None = None,
    date_col: str | None = None,
    drop_users: Sequence[Any] | str | None = None,
    rename_users: bool = False,
    aggr: str | None = None,
    aggr_period: str = "h",
) -> CsvTimeSeries:
    csv_path = resolve_csv_path(path, dataset_name)
    if date_col:
        raw = pd.read_csv(csv_path, parse_dates=[date_col])
        raw = raw.set_index(date_col)
    else:
        raw = pd.read_csv(csv_path, index_col=0)
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            pass

    raw = _aggregate(raw, aggr, aggr_period)
    raw = raw.dropna(axis=0, how="any")

    past_cols = _select_columns(raw, past_covariate_cols)
    future_cols = _select_columns(raw, future_covariate_cols)
    cov_cols = set(past_cols + future_cols)
    if target_cols is None:
        value_cols = [col for col in raw.columns if col not in cov_cols]
    else:
        value_cols = _select_columns(raw, target_cols)

    values = _drop_users(raw[value_cols].copy(), drop_users)
    if rename_users:
        values.columns = [f"user_{idx}" for idx in range(values.shape[1])]

    past = raw[past_cols].copy() if past_cols else None
    future = raw[future_cols].copy() if future_cols else None
    if values.empty:
        raise ValueError("dataset has no target columns after filtering")
    return CsvTimeSeries(values, past_covariates=past, future_covariates=future)


def parse_ratios(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        ratios = [float(part) for part in _split_text(value)]
    else:
        ratios = [float(part) for part in value]
    if len(ratios) != 3:
        raise ValueError("split ratios must contain exactly three values: T0,T1,T2")
    total = sum(ratios)
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    return ratios


def split_bounds(n_dates: int, ratios: str | Sequence[float]) -> tuple[int, int, int]:
    r0, r1, _ = parse_ratios(ratios)
    t0_end = int(r0 * n_dates)
    t1_end = int((r0 + r1) * n_dates)
    return t0_end, t1_end, int(n_dates)


def load_json_kwargs(text_or_path: str | None) -> dict[str, Any]:
    if not text_or_path:
        return {}
    text = str(text_or_path)
    path = Path(text).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(text)


def load_pretrained_model(
    name: str,
    *,
    lags: int,
    horizon: int,
    dim: int = 1,
    normalization: str | None = "none",
    pretrained_path: str | Path | None = None,
    device: str | torch.device | None = "auto",
    model_kwargs: dict[str, Any] | None = None,
) -> torch.nn.Module:
    model = load_model(
        name,
        lags=lags,
        dim=dim,
        horizon=horizon,
        normalization=normalization,
        pretrained_path=pretrained_path,
        **(model_kwargs or {}),
    )
    return model.to(resolve_device(device)).eval()


def run_dir(output_dir: str | Path, save_name: str) -> Path:
    path = Path(output_dir).expanduser() / str(save_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path.resolve()
