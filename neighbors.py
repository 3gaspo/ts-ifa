"""Aligned datastore and exact nearest-neighbor utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .load_dataset_model import CsvTimeSeries

DistanceMetric = Literal["euclidean", "cosine", "pearson"]


def fourier_features(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    centered = values - values.mean(axis=1, keepdims=True)
    scaled = centered / (values.std(axis=1, keepdims=True) + eps)
    return np.abs(np.fft.fft(scaled, axis=1)).astype(np.float32)


def normalize_windows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return ((values - mean) / (std + eps)).astype(np.float32)


@dataclass
class WindowBatch:
    """Flattened windows in user-major, date-minor order."""

    dates: np.ndarray
    features: np.ndarray
    windows: torch.Tensor
    n_users: int
    lags: int
    horizon: int

    @property
    def n_dates(self) -> int:
        return int(len(self.dates))

    def decode_indices(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_dates == 0:
            raise ValueError("cannot decode indices against an empty store")
        idx = torch.as_tensor(indices, dtype=torch.long)
        user_idx = idx // self.n_dates
        date_pos = idx % self.n_dates
        store_dates = torch.as_tensor(self.dates, dtype=torch.long)
        return user_idx, store_dates[date_pos]

    def select_windows(self, indices: np.ndarray) -> torch.Tensor:
        flat = torch.as_tensor(indices, dtype=torch.long)
        return self.windows[flat]


def period_eval_dates(
    period_start: int,
    period_end: int,
    *,
    n_dates: int,
    lags: int,
    horizon: int,
    stride: int,
) -> np.ndarray:
    """Return query starts fully contained in ``[period_start, period_end)``."""
    max_start = n_dates - (lags + horizon)
    last = min(period_end - (lags + horizon), max_start)
    if last < period_start:
        return np.array([], dtype=np.int64)
    return np.arange(period_start, last + 1, int(stride), dtype=np.int64)


def _trim_dates(
    dates: np.ndarray,
    *,
    max_train_windows: int | None,
    n_users: int,
) -> np.ndarray:
    if max_train_windows is None or len(dates) == 0:
        return dates
    allowed_steps = int(max_train_windows) // int(n_users)
    if allowed_steps <= 0:
        return np.array([], dtype=np.int64)
    return dates[-allowed_steps:]


def aligned_store_dates(
    query_t: int,
    *,
    lags: int,
    horizon: int,
    train_stride: int,
    n_users: int,
    period: int,
    store_start: int,
    store_end: int,
    online: bool = False,
    align_period: bool = True,
    max_train_windows: int | None = None,
) -> np.ndarray:
    """Return datastore start dates aligned to the query phase.

    In fixed mode the store is ``[store_start, store_end)``. In online mode it
    uses all complete history ending before the query window.
    """
    if online:
        last_valid_store = int(query_t) - (int(lags) + int(horizon))
        if last_valid_store < 0:
            return np.array([], dtype=np.int64)
        first = 0
        last = last_valid_store
    else:
        last = int(store_end) - (int(lags) + int(horizon))
        if last < store_start:
            return np.array([], dtype=np.int64)
        first = int(store_start)

    if align_period:
        if period <= 0:
            raise ValueError("period must be positive when align_period=True")
        first = first + ((int(query_t) - first) % int(period))
        last = last - ((last - first) % int(period))
        if last < first:
            return np.array([], dtype=np.int64)

    dates = np.arange(first, last + 1, int(train_stride), dtype=np.int64)
    return _trim_dates(dates, max_train_windows=max_train_windows, n_users=n_users)


def build_window_batch(
    dataset: CsvTimeSeries,
    start_dates: np.ndarray,
    *,
    lags: int,
    horizon: int,
    distance_space: str = "raw",
    model: torch.nn.Module | None = None,
    device: str | torch.device | None = None,
    normalize: bool = True,
    pool_representation: bool = False,
) -> WindowBatch:
    """Build flattened features and raw windows for deterministic start dates."""
    start_dates = np.asarray(start_dates, dtype=np.int64)
    if len(start_dates) == 0:
        return WindowBatch(
            dates=start_dates,
            features=np.empty((0, int(lags)), dtype=np.float32),
            windows=torch.empty((0, int(lags) + int(horizon)), dtype=torch.float32),
            n_users=dataset.n_users,
            lags=int(lags),
            horizon=int(horizon),
        )
    max_stop = int(start_dates.max()) + int(lags) + int(horizon)
    if max_stop > dataset.n_dates:
        raise ValueError("requested window dates exceed dataset length")

    value_indices = start_dates[:, None] + np.arange(int(lags) + int(horizon))
    raw = dataset.values[value_indices]  # (dates, lags+horizon, users)
    windows = raw.transpose(2, 0, 1).reshape(-1, int(lags) + int(horizon))
    lookbacks = windows[:, : int(lags)]
    feature_source = normalize_windows(lookbacks) if normalize else lookbacks.astype(np.float32)

    space = str(distance_space).lower()
    if space == "fourier":
        features = fourier_features(feature_source)
    elif space in {"chronos", "patchtst", "model", "representation"}:
        if model is None:
            raise ValueError(f"distance_space={distance_space!r} requires a model")
        if not hasattr(model, "representation"):
            raise AttributeError("model does not expose representation()")
        x = torch.as_tensor(feature_source, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.inference_mode():
            reps = model.representation(x, pool=pool_representation)
        features = reps.detach().cpu().numpy().astype(np.float32)
    elif space == "raw":
        features = np.ascontiguousarray(feature_source, dtype=np.float32)
    else:
        raise ValueError(f"unknown distance_space={distance_space!r}")

    return WindowBatch(
        dates=start_dates,
        features=np.ascontiguousarray(features, dtype=np.float32),
        windows=torch.as_tensor(windows, dtype=torch.float32),
        n_users=dataset.n_users,
        lags=int(lags),
        horizon=int(horizon),
    )


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, eps)


def _metric_ready(values: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if metric == "euclidean":
        return arr
    if metric == "cosine":
        return _normalize_rows(arr)
    if metric == "pearson":
        centered = arr - arr.mean(axis=1, keepdims=True)
        return _normalize_rows(centered)
    raise ValueError(f"unknown distance metric {metric!r}")


def search_neighbors(
    query_features: np.ndarray,
    store_features: np.ndarray,
    *,
    k: int,
    metric: DistanceMetric = "euclidean",
    chunk_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact KNN search with bounded query chunking.

    Returns ``(distances, indices)`` shaped ``(n_query, k)``.
    """
    k = int(k)
    n_query = int(query_features.shape[0])
    n_store = int(store_features.shape[0])
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
        )
    if n_store < k:
        raise ValueError(f"datastore has {n_store} windows, fewer than k={k}")

    metric = str(metric).lower()  # type: ignore[assignment]
    query = _metric_ready(query_features, metric)  # type: ignore[arg-type]
    store = _metric_ready(store_features, metric)  # type: ignore[arg-type]

    all_distances = np.empty((n_query, k), dtype=np.float32)
    all_indices = np.empty((n_query, k), dtype=np.int64)
    for start in range(0, n_query, int(chunk_size)):
        stop = min(start + int(chunk_size), n_query)
        q = query[start:stop]
        if metric == "euclidean":
            q2 = (q * q).sum(axis=1, keepdims=True)
            s2 = (store * store).sum(axis=1, keepdims=True).T
            distances = np.sqrt(np.maximum(q2 + s2 - 2.0 * q @ store.T, 0.0))
        else:
            distances = 1.0 - q @ store.T
        top = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        top_dist = np.take_along_axis(distances, top, axis=1)
        order = np.argsort(top_dist, axis=1)
        all_distances[start:stop] = np.take_along_axis(top_dist, order, axis=1)
        all_indices[start:stop] = np.take_along_axis(top, order, axis=1)
    return all_distances, all_indices
