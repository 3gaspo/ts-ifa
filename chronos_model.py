"""Chronos-2 wrapper for the lightweight extraction scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat


def _import_chronos():
    try:
        from chronos import BaseChronosPipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Chronos extraction requires `chronos-forecasting` and local weights."
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


def _default_chronos_weights() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    return _existing_path(
        repo_root / "timetensors" / "models" / "sota" / "chronos2" / "weights",
        repo_root / "timetensors_old" / "src" / "timetensor" / "sota" / "chronos2" / "weights",
    )


def _broadcast(value: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] == 1:
        return repeat(value, "1 channel time -> batch channel time", batch=batch_size)
    return value


def _cat(parts: list[torch.Tensor | None]) -> torch.Tensor | None:
    present = [part for part in parts if part is not None]
    if not present:
        return None
    packed, _ = pack(present, "batch * time")
    return packed


class Chronos(nn.Module):
    """Thin Chronos-2 wrapper for extraction.

    ``context`` is a tensor shaped ``(batch, channels, lags)`` or
    ``(batch, channels, lags + horizon)``. Additional global covariates can be
    supplied as ``past_covariates`` and ``future_covariates``.
    """

    def __init__(
        self,
        lags: int,
        dim: int = 1,
        horizon: int | None = None,
        *,
        context_mode: str = "past_only",
        cross_learning: bool = False,
        shared_context: bool = False,
        weights_path: str | Path | None = None,
        device_map: str = "cuda",
        local_files_only: bool = True,
        frozen: bool = True,
        quantile_index: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        del kwargs
        if horizon is None:
            raise ValueError("horizon is required")
        self.lags = int(lags)
        self.dim = int(dim)
        self.horizon = int(horizon)
        self.context_mode = str(context_mode)
        self.cross_learning = bool(cross_learning)
        self.shared_context = bool(shared_context)
        self.quantile_index = quantile_index

        model_path = Path(weights_path).expanduser().resolve() if weights_path else _default_chronos_weights()
        if model_path is None:
            raise FileNotFoundError("Chronos weights were not found; pass weights_path")
        pipeline_cls = _import_chronos()
        self.pipeline = pipeline_cls.from_pretrained(
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

    def _context_parts(
        self,
        context: torch.Tensor | None,
        batch_size: int,
        past_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        context_past = None
        context_future = None
        if context is not None:
            if context.shape[-1] < self.lags:
                raise ValueError(f"context length must be at least {self.lags}")
            context_past = context[..., : self.lags]
            if self.context_mode in {"future", "future_included", "structured"}:
                if context.shape[-1] < self.lags + self.horizon:
                    raise ValueError("future context requires lags + horizon values")
                context_future = context[..., self.lags : self.lags + self.horizon]
            elif self.context_mode != "past_only":
                raise ValueError(f"unknown context_mode={self.context_mode!r}")

        past_covariates = _broadcast(past_covariates, batch_size)
        future_covariates = _broadcast(future_covariates, batch_size)
        past = _cat([context_past, past_covariates])
        future = _cat([context_future, future_covariates])
        return past, future

    def _prepare_inputs(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        past_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
    ) -> list[dict[str, Any]]:
        batch_size, _, _ = x.shape
        past, future = self._context_parts(
            context,
            batch_size,
            past_covariates,
            future_covariates,
        )
        inputs = []
        for batch_index in range(batch_size):
            item: dict[str, Any] = {"target": x[batch_index].detach().cpu()}
            past_dict = self._series_dict(past, batch_index, batch_size, "past")
            future_dict = self._series_dict(future, batch_index, batch_size, "future")
            if past_dict:
                item["past_covariates"] = past_dict
            if future_dict:
                item["future_covariates"] = future_dict
            inputs.append(item)
        return inputs

    def _series_dict(
        self,
        value: torch.Tensor | None,
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
            f"{prefix}_{item}_{channel}": value[item, channel, :].detach().cpu()
            for item in range(context_batch)
            for channel in range(channels)
        }

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        inputs = self._prepare_inputs(x, context, past_covariates, future_covariates)
        predictions = self.pipeline.predict(
            inputs=inputs,
            prediction_length=self.horizon,
            cross_learning=self.cross_learning,
        )
        rows = []
        for pred in predictions:
            q_index = pred.shape[1] // 2 if self.quantile_index is None else int(self.quantile_index)
            rows.append(pred[:, q_index, :])
        return rearrange(rows, "batch dim horizon -> batch dim horizon").to(
            device=x.device,
            dtype=x.dtype,
        )

    @torch.no_grad()
    def representation(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        past_covariates: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        pool: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if not hasattr(self.pipeline, "embed"):
            raise AttributeError("loaded Chronos pipeline does not expose embed()")
        embeddings, _ = self.pipeline.embed(
            inputs=self._prepare_inputs(x, context, past_covariates, future_covariates)
        )
        raw = rearrange(embeddings, "batch ... -> batch ...").to(device=x.device, dtype=x.dtype)
        if pool:
            return raw.mean(dim=1).mean(dim=1)
        return rearrange(raw, "batch ... -> batch (...)")
