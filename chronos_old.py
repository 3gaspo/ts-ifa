"""Chronos-2 forecasting wrapper.

The module keeps Chronos optional: importing ``timetensors`` should not require
the heavy Chronos dependency, but constructing this model does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _import_chronos():
    try:
        from chronos import BaseChronosPipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chronos support requires the optional dependency "
            "`chronos-forecasting`. Install the SOTA extras or add it to the "
            "environment before using model.name=chronos."
        ) from exc
    return BaseChronosPipeline


def _existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _default_weights_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    return _existing_path(
        Path(__file__).resolve().parent / "chronos2" / "weights",
        repo_root / "timetensors_old" / "src" / "timetensor" / "sota" / "chronos2" / "weights",
    )


class Chronos(nn.Module):
    """Frozen Chronos-2 forecaster with TimeTensor covariate support.

    Inputs and outputs follow the package contract:
    ``(batch, dim, lags) -> (batch, dim, horizon)``.
    """

    def __init__(
        self,
        lags: int,
        dim: int = 1,
        horizon: int | None = None,
        *,
        context_mode: str = "structured",
        cross_learning: bool = False,
        device_map: str = "cuda",
        weights_path: str | Path | None = None,
        local_files_only: bool = True,
        shared_context: bool = False,
        frozen: bool = True,
        quantile_index: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("Chronos requires horizon")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.context_mode = str(context_mode)
        self.cross_learning = bool(cross_learning)
        self.shared_context = bool(shared_context)
        self.quantile_index = quantile_index

        model_path = Path(weights_path).expanduser().resolve() if weights_path is not None else _default_weights_path()
        if model_path is None:
            raise FileNotFoundError(
                "Chronos weights were not found. Pass model.kwargs.weights_path "
                "or place weights under timetensors/models/sota/chronos2/weights."
            )

        base_pipeline = _import_chronos()
        self.pipeline = base_pipeline.from_pretrained(
            str(model_path),
            device_map=device_map,
            local_files_only=bool(local_files_only),
        )

        if frozen:
            model = getattr(self.pipeline, "model", None)
            if model is not None:
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if x.ndim != 3:
            raise ValueError(f"expected x with shape (batch, dim, lags), got {tuple(x.shape)}")
        if x.shape[-1] != self.lags:
            raise ValueError(f"expected lags={self.lags}, got {x.shape[-1]}")
        inputs = self._prepare_inputs(x, covariates)
        predictions = self.pipeline.predict(
            inputs=inputs,
            prediction_length=self.horizon,
            cross_learning=self.cross_learning,
        )
        output = []
        for prediction in predictions:
            q_index = prediction.shape[1] // 2 if self.quantile_index is None else int(self.quantile_index)
            output.append(prediction[:, q_index, :])
        return torch.stack(output, dim=0).to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None = None,
        *,
        pool: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if x.shape[-1] != self.lags:
            raise ValueError(f"expected lags={self.lags}, got {x.shape[-1]}")
        if not hasattr(self.pipeline, "embed"):
            raise AttributeError("loaded Chronos pipeline does not expose embed()")
        embeddings, _ = self.pipeline.embed(inputs=self._prepare_inputs(x, covariates))
        raw = torch.stack(embeddings, dim=0).to(device=x.device, dtype=x.dtype)
        if pool:
            return raw.mean(dim=1).mean(dim=1)
        return raw.flatten(start_dim=1)

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        covariates: dict[str, torch.Tensor | None] | None,
    ) -> list[dict[str, Any]]:
        past, future = self._select_context(covariates)
        batch_size, _, lags = x.shape
        inputs = []
        for batch_index in range(batch_size):
            item: dict[str, Any] = {"target": x[batch_index].detach().cpu()}
            past_covariates = self._series_dict(
                past,
                batch_index=batch_index,
                batch_size=batch_size,
                prefix="past",
            )
            future_covariates = {}
            if future is not None:
                future_past = future[..., :lags]
                future_future = future[..., lags : lags + self.horizon]
                past_covariates.update(
                    self._series_dict(
                        future_past,
                        batch_index=batch_index,
                        batch_size=batch_size,
                        prefix="future",
                    )
                )
                future_covariates = self._series_dict(
                    future_future,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    prefix="future",
                )
            if past_covariates:
                item["past_covariates"] = past_covariates
            if future_covariates:
                item["future_covariates"] = future_covariates
            inputs.append(item)
        return inputs

    def _select_context(
        self,
        covariates: dict[str, torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if covariates is None:
            return None, None
        past = covariates.get("past")
        future = covariates.get("future")
        if self.context_mode == "structured":
            joined_future = None
            if future is not None:
                future_prefix = self._future_prefix(past, future)
                joined_future = torch.cat([future_prefix, future], dim=-1)
            return past, joined_future
        if self.context_mode == "past_only":
            return past, None
        if self.context_mode == "future_included":
            if future is None:
                return None, None
            future_prefix = self._future_prefix(past, future)
            return None, torch.cat([future_prefix, future], dim=-1)
        raise ValueError(f"unknown context_mode={self.context_mode!r}")

    def _future_prefix(
        self,
        past: torch.Tensor | None,
        future: torch.Tensor,
    ) -> torch.Tensor:
        if past is not None and past.shape[:-1] == future.shape[:-1]:
            return past
        return torch.zeros(
            *future.shape[:-1],
            self.lags,
            device=future.device,
            dtype=future.dtype,
        )

    def _series_dict(
        self,
        value: torch.Tensor | None,
        *,
        batch_index: int,
        batch_size: int,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        if value is None:
            return {}
        context_batch, channels, _ = value.shape
        if not self.shared_context and context_batch == batch_size:
            return {
                f"{prefix}_{channel}": value[batch_index, channel, :].detach().cpu()
                for channel in range(channels)
            }
        return {
            f"{prefix}_{context_index}_{channel}": value[context_index, channel, :].detach().cpu()
            for context_index in range(context_batch)
            for channel in range(channels)
        }
