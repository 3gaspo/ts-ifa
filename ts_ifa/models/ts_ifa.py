"""TS-IFA: Time Series Informed Forecasting Adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import pack, rearrange


def mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class CrossAttentionBlock(nn.Module):
    """Projected multi-head cross-attention followed by norm and feed-forward layers."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        *,
        heads: int = 4,
        attn_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = int(heads)
        self.attn_dim = int(attn_dim)
        self.model_dim = self.heads * self.attn_dim
        self.query_projection = nn.Linear(query_dim, self.model_dim)
        self.key_projection = nn.Linear(key_dim, self.model_dim)
        self.value_projection = nn.Linear(value_dim, self.model_dim)
        self.query_norm = nn.LayerNorm(self.model_dim)
        self.key_norm = nn.LayerNorm(self.model_dim)
        self.value_norm = nn.LayerNorm(self.model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(self.model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.model_dim, 4 * self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.model_dim, self.model_dim),
            nn.Dropout(dropout),
        )
        self.feed_forward_norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Linear(self.model_dim, output_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(
            self.query_norm(self.query_projection(query)),
            "batch dim -> batch 1 dim",
        )
        k = self.key_norm(self.key_projection(keys))
        v = self.value_norm(self.value_projection(values))
        attended, weights = self.attention(
            q,
            k,
            v,
            need_weights=True,
            average_attn_weights=True,
        )
        hidden = self.attention_norm(q + attended)
        hidden = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        output = self.output_projection(rearrange(hidden, "batch 1 dim -> batch dim"))
        weights = rearrange(weights, "batch 1 items -> batch items")
        return output, weights


@dataclass(frozen=True)
class TSIFAConfig:
    lags: int
    horizon: int
    neighbors: int
    residual_heads: int = 4
    memory_heads: int = 4
    mixture_heads: int = 4
    residual_attn_dim: int = 32
    memory_attn_dim: int = 32
    mixture_attn_dim: int = 32
    residual_hidden: int = 128
    memory_hidden: int = 128
    mixture_hidden: int = 128
    mixture_key_dim: int = 64
    mixture_gate_init: float = -6.0
    dropout: float = 0.0


class TimeSeriesInformedForecastingAdapter(nn.Module):
    """TS-IFA adapter trained from extracted neighbor payloads."""

    def __init__(self, config: TSIFAConfig):
        super().__init__()
        if config.neighbors <= 0:
            raise ValueError("TimeSeriesInformedForecastingAdapter requires neighbors > 0")
        self.config = config
        lags = int(config.lags)
        horizon = int(config.horizon)
        mixture_key_dim = int(config.mixture_key_dim)

        self.residual_attention = CrossAttentionBlock(
            query_dim=lags + horizon,
            key_dim=lags + horizon,
            value_dim=horizon,
            output_dim=horizon,
            heads=config.residual_heads,
            attn_dim=config.residual_attn_dim,
            dropout=config.dropout,
        )
        self.residual_head = mlp(
            horizon,
            config.residual_hidden,
            horizon,
            dropout=config.dropout,
        )

        self.memory_attention = CrossAttentionBlock(
            query_dim=lags,
            key_dim=lags,
            value_dim=horizon,
            output_dim=horizon,
            heads=config.memory_heads,
            attn_dim=config.memory_attn_dim,
            dropout=config.dropout,
        )
        self.memory_head = mlp(
            2 * horizon,
            config.memory_hidden,
            horizon,
            dropout=config.dropout,
        )

        self.query_mlp = mlp(
            lags + horizon,
            config.mixture_hidden,
            mixture_key_dim,
            dropout=config.dropout,
        )
        self.candidate_mlp = mlp(
            horizon,
            config.mixture_hidden,
            mixture_key_dim,
            dropout=config.dropout,
        )
        self.mixture_attention = CrossAttentionBlock(
            query_dim=mixture_key_dim,
            key_dim=mixture_key_dim,
            value_dim=horizon,
            output_dim=horizon,
            heads=config.mixture_heads,
            attn_dim=config.mixture_attn_dim,
            dropout=config.dropout,
        )
        self.mixture_gate = mlp(
            lags + 2 * horizon,
            config.mixture_hidden,
            horizon,
            dropout=config.dropout,
        )
        gate_output = self.mixture_gate[-1]
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, float(config.mixture_gate_init))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        x_c = batch["x_c"]
        y_c = batch["y_c"]
        pred = batch["pred"]
        pred_context = batch["pred_context"]
        pred_neighbors = batch["pred_neighbors"]
        residual_c = batch["residual_c"]

        m, _ = pack([x, pred], "batch *")
        m_c, _ = pack([x_c, pred_neighbors], "batch neighbor *")

        z_r, residual_weights = self.residual_attention(m, m_c, residual_c)
        residual_delta = self.residual_head(z_r)
        y_r = pred + residual_delta

        z_m, memory_weights = self.memory_attention(x, x_c, y_c)
        memory_input, _ = pack([pred, z_m], "batch *")
        y_m = self.memory_head(memory_input)

        candidates = rearrange(
            [pred, pred_context, y_r, y_m],
            "candidate batch horizon -> batch candidate horizon",
        )
        z_x = self.query_mlp(m)
        z_y = self.candidate_mlp(
            rearrange(candidates, "batch candidate horizon -> (batch candidate) horizon")
        )
        z_y = rearrange(
            z_y,
            "(batch candidate) dim -> batch candidate dim",
            candidate=4,
        )
        mixture_candidate, mixture_weights = self.mixture_attention(z_x, z_y, candidates)
        gate_input, _ = pack([x, pred, mixture_candidate], "batch *")
        mixture_gate = torch.sigmoid(self.mixture_gate(gate_input))
        prediction = pred + mixture_gate * (mixture_candidate - pred)

        return {
            "prediction": prediction,
            "residual_prediction": y_r,
            "memory_prediction": y_m,
            "mixture_candidate": mixture_candidate,
            "mixture_gate": mixture_gate,
            "residual_delta": residual_delta,
            "residual_weights": residual_weights,
            "memory_weights": memory_weights,
            "mixture_weights": mixture_weights,
        }


ProposedModelConfig = TSIFAConfig
