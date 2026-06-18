"""Minimal forecasting model wrapper for extraction experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class NoNormalization(nn.Module):
    name = "none"

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[()]]:
        return x, ()

    def inverse(self, y: torch.Tensor, state: tuple[()]) -> torch.Tensor:
        del state
        return y


class InstanceNormalization(nn.Module):
    name = "instance"

    def __init__(self, eps: float = 1e-8, center: str = "mean"):
        super().__init__()
        self.eps = float(eps)
        self.center = str(center)

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.center == "last":
            mean = x[..., -1:].detach()
        elif self.center == "mean":
            mean = x.mean(dim=-1, keepdim=True).detach()
        else:
            raise ValueError("instance center must be 'mean' or 'last'")
        std = x.std(dim=-1, keepdim=True, unbiased=False).detach()
        return (x - mean) / (std + self.eps), (mean, std)

    def inverse(
        self,
        y: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mean, std = state
        return y * (std + self.eps) + mean


def build_normalization(name: str | None, **kwargs: Any) -> nn.Module:
    if name is None or str(name).lower() in {"", "none", "identity"}:
        return NoNormalization()
    if str(name).lower() == "instance":
        return InstanceNormalization(**kwargs)
    raise ValueError("lightweight extraction only supports normalization='none' or 'instance'")


class ForecastModel(nn.Module):
    """Compose one optional normalization with a base forecaster."""

    def __init__(self, base_model: nn.Module, normalization: nn.Module | None = None):
        super().__init__()
        self.base_model = base_model
        self.normalization = normalization or NoNormalization()
        self.lags = int(getattr(base_model, "lags"))
        self.dim = int(getattr(base_model, "dim", 1))
        self.horizon = int(getattr(base_model, "horizon"))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        x_norm, state = self.normalization.normalize(x)
        pred = self.base_model(
            x_norm,
            context=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )
        return self.normalization.inverse(pred, state)

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not hasattr(self.base_model, "representation"):
            raise AttributeError(f"{self.base_model.__class__.__name__} has no representation()")
        x_norm, _ = self.normalization.normalize(x)
        return self.base_model.representation(
            x_norm,
            context=context,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )


class Persistence(nn.Module):
    def __init__(self, lags: int, dim: int = 1, horizon: int | None = None, **kwargs: Any):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return x[..., -1:].repeat_interleave(self.horizon, dim=-1)


class Linear(nn.Module):
    def __init__(self, lags: int, dim: int = 1, horizon: int | None = None, **kwargs: Any):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.linear = nn.Linear(self.lags * self.dim, self.horizon * self.dim)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        y = self.linear(x.reshape(x.shape[0], self.lags * self.dim))
        return y.reshape(x.shape[0], self.dim, self.horizon)


def _state_dict_from_file(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu")
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        return payload
    raise TypeError(f"expected a state-dict-like payload in {path}")


def _load_pretrained(model: ForecastModel, path: str | Path) -> None:
    state = _state_dict_from_file(path)
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        pass
    model.base_model.load_state_dict(state)


def load_model(
    name: str,
    *,
    lags: int,
    dim: int = 1,
    horizon: int,
    normalization: str | None = "none",
    pretrained_path: str | Path | None = None,
    normalization_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ForecastModel:
    """Load a minimal extraction forecaster.

    Built-ins are ``persistence``, ``linear``, ``patchtst``, and ``chronos``.
    """
    from .foundation_models import Chronos, PatchTST

    key = str(name).lower()
    registry = {
        "persistence": Persistence,
        "repeat_last": Persistence,
        "linear": Linear,
        "patchtst": PatchTST,
        "patch": PatchTST,
        "chronos": Chronos,
    }
    if key not in registry:
        raise ValueError(f"unknown extraction model {name!r}")
    base = registry[key](lags=lags, dim=dim, horizon=horizon, **kwargs)
    model = ForecastModel(
        base,
        normalization=build_normalization(normalization, **(normalization_kwargs or {})),
    )
    if pretrained_path is not None:
        _load_pretrained(model, pretrained_path)
    return model
