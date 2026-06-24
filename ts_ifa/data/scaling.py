"""Scale-transfer helpers for retrieved time-series examples."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _mean_std(value: Any, eps: float) -> tuple[Any, Any]:
    if isinstance(value, np.ndarray):
        mean = value.mean(axis=-1, keepdims=True)
        std = np.maximum(value.std(axis=-1, keepdims=True), float(eps))
    elif torch.is_tensor(value):
        mean = value.mean(dim=-1, keepdim=True)
        std = value.std(dim=-1, keepdim=True, unbiased=False).clamp_min(float(eps))
    else:
        raise TypeError(f"expected a NumPy array or torch tensor, got {type(value).__name__}")
    return mean, std


def neighbor_to_query_scale(
    query_lookback: Any,
    neighbor_lookback: Any,
    neighbor_value: Any,
    *,
    residual: bool = False,
    eps: float = 1e-8,
) -> Any:
    """Express a neighbor tensor in the query's lookback scale.

    ``neighbor_value`` must have the same leading dimensions as
    ``neighbor_lookback``. Ordinary values receive both the neighbor-to-query
    location and scale transform. Residuals receive only the scale transform
    because their additive locations already cancel.
    """
    if query_lookback.ndim + 1 != neighbor_lookback.ndim:
        raise ValueError("neighbor lookbacks must add exactly one neighbor dimension")
    if neighbor_value.shape[:-1] != neighbor_lookback.shape[:-1]:
        raise ValueError("neighbor values and lookbacks must share leading dimensions")

    query_mean, query_std = _mean_std(query_lookback, eps)
    neighbor_mean, neighbor_std = _mean_std(neighbor_lookback, eps)
    query_mean = query_mean[..., None, :]
    query_std = query_std[..., None, :]

    if residual:
        return neighbor_value / neighbor_std * query_std
    return (neighbor_value - neighbor_mean) / neighbor_std * query_std + query_mean
