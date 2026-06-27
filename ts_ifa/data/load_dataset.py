"""CSV dataset loading and shared experiment helpers."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from einops import rearrange


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def _column_names(columns: Sequence[Any] | str | None) -> list[str]:
    return [str(item) for item in _as_list(columns)]


DATASET_CONFIG_KEYS = {
    "target_cols",
    "date_col",
    "drop_users",
    "rename_users",
    "aggr",
    "aggr_period",
}


def _dataset_config_path(
    path: str | Path,
    dataset_config: str | Path | None = None,
) -> tuple[Path, bool]:
    if dataset_config not in {None, ""}:
        config_path = Path(dataset_config).expanduser()
        return (config_path / "config.json" if config_path.is_dir() else config_path), True
    base = Path(path).expanduser()
    directory = base.parent if base.suffix.lower() == ".csv" else base
    return directory / "config.json", False


def _dataset_config_options(raw: Mapping[str, Any]) -> dict[str, Any]:
    options = {key: raw[key] for key in DATASET_CONFIG_KEYS if key in raw}
    scoped = raw.get("ts_ifa")
    if scoped is not None:
        if not isinstance(scoped, Mapping):
            raise ValueError("dataset config field 'ts_ifa' must be an object")
        options.update(scoped)
    return options


def load_dataset_config(
    path: str | Path,
    dataset_config: str | Path | None = None,
) -> dict[str, Any]:
    config_path, explicit = _dataset_config_path(path, dataset_config)
    if not config_path.exists():
        if explicit:
            raise FileNotFoundError(config_path)
        return {}
    if config_path.suffix.lower() != ".json":
        raise ValueError(f"dataset config must be JSON, got {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"dataset config must contain a JSON object: {config_path}")
    options = _dataset_config_options(raw)
    LOGGER.info("loaded dataset config path=%s keys=%s", config_path, sorted(options))
    return options


def _configured_value(explicit: Any, configured: Any, default: Any = None) -> Any:
    if explicit is not None:
        return explicit
    if configured is not None:
        return configured
    return default


def _drop_users(df: pd.DataFrame, drop_users: Sequence[Any] | str | None) -> pd.DataFrame:
    columns = []
    for item in _as_list(drop_users):
        item_text = str(item)
        if item in df.columns:
            columns.append(item)
        elif item_text in df.columns:
            columns.append(item_text)
        elif isinstance(item, int) or item_text.lstrip("-").isdigit():
            idx = int(item)
            if idx < 0 or idx >= len(df.columns):
                raise IndexError(f"drop user index out of range: {idx}")
            columns.append(df.columns[idx])
        else:
            columns.append(item_text)
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
    """Date x user values for target-only forecasting experiments."""

    frame: pd.DataFrame

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
        arr = torch.as_tensor(
            rearrange(values, "time user -> user time").copy(),
            dtype=torch.float32,
            device=device,
        )
        x = arr[:, None, :lags]
        y = arr[:, None, lags:]
        return x, y


def load_csv_dataset(
    path: str | Path,
    *,
    dataset_name: str | None = None,
    target_cols: Sequence[Any] | str | None = None,
    date_col: str | None = None,
    drop_users: Sequence[Any] | str | None = None,
    rename_users: bool | None = None,
    aggr: str | None = None,
    aggr_period: str | None = None,
    dataset_config: str | Path | None = None,
) -> CsvTimeSeries:
    config = load_dataset_config(path, dataset_config)
    target_cols = _configured_value(target_cols, config.get("target_cols"))
    date_col = _configured_value(date_col, config.get("date_col"))
    drop_users = _configured_value(drop_users, config.get("drop_users"))
    rename_users = bool(_configured_value(rename_users, config.get("rename_users"), False))
    aggr = _configured_value(aggr, config.get("aggr"))
    aggr_period = str(_configured_value(aggr_period, config.get("aggr_period"), "h"))

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

    value_cols = list(raw.columns) if target_cols is None else _column_names(target_cols)
    missing = [col for col in value_cols if col not in raw.columns]
    if missing:
        raise KeyError(f"CSV target columns not found: {missing}")

    values = _drop_users(raw[value_cols].copy(), drop_users)
    if rename_users:
        values.columns = [f"user_{idx}" for idx in range(values.shape[1])]
    if values.empty:
        raise ValueError("dataset has no target columns after filtering")
    return CsvTimeSeries(values)


def parse_ratios(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        ratios = [float(part) for part in _split_text(value)]
    else:
        ratios = [float(part) for part in value]
    if len(ratios) != 4:
        raise ValueError("split ratios must contain exactly four values: T0,T1,T2,T3")
    total = sum(ratios)
    if not np.isclose(total, 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    return ratios


def split_bounds(n_dates: int, ratios: str | Sequence[float]) -> tuple[int, int, int, int]:
    r0, r1, r2, _ = parse_ratios(ratios)
    t0_end = int(round(r0 * n_dates))
    t1_end = int(round((r0 + r1) * n_dates))
    t2_end = int(round((r0 + r1 + r2) * n_dates))
    return t0_end, t1_end, t2_end, int(n_dates)


def load_json_kwargs(text_or_path: str | None) -> dict[str, Any]:
    if not text_or_path:
        return {}
    text = str(text_or_path)
    path = Path(text).expanduser()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(text)


def run_dir(output_dir: str | Path, save_name: str) -> Path:
    path = Path(output_dir).expanduser() / str(save_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path.resolve()
