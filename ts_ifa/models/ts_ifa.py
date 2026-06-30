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
            2 * horizon,
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

        self.mixture_attention = CrossAttentionBlock(
            query_dim=lags,
            key_dim=lags,
            value_dim=lags,
            output_dim=lags,
            heads=config.mixture_heads,
            attn_dim=config.mixture_attn_dim,
            dropout=config.dropout,
        )
        self.mixture_logits = mlp(
            lags,
            config.mixture_hidden,
            4 * horizon,
            dropout=config.dropout,
        )
        logits_output = self.mixture_logits[-1]
        nn.init.zeros_(logits_output.weight)
        with torch.no_grad():
            logits_output.bias.zero_()
            logits_bias = logits_output.bias.view(4, horizon)
            logits_bias[1:].fill_(float(config.mixture_gate_init))

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
        residual_input, _ = pack([pred, z_r], "batch *")
        residual_delta = self.residual_head(residual_input)
        y_r = pred + residual_delta

        z_m, memory_weights = self.memory_attention(x, x_c, y_c)
        memory_input, _ = pack([pred, z_m], "batch *")
        memory_delta = self.memory_head(memory_input)
        y_m = pred + memory_delta

        candidates = rearrange(
            [pred, pred_context, y_r, y_m],
            "candidate batch horizon -> batch candidate horizon",
        )
        z_mix, retrieval_mixture_weights = self.mixture_attention(x, x_c, x_c)
        mixture_logits = rearrange(
            self.mixture_logits(z_mix),
            "batch (candidate horizon) -> batch candidate horizon",
            candidate=4,
            horizon=pred.shape[-1],
        )
        mixture_weights = torch.softmax(mixture_logits, dim=1)
        prediction = (mixture_weights * candidates).sum(dim=1)

        return {
            "prediction": prediction,
            "residual_prediction": y_r,
            "memory_prediction": y_m,
            "mixture_candidate": prediction,
            "mixture_gate": mixture_weights[:, 1:].sum(dim=1),
            "residual_delta": residual_delta,
            "memory_delta": memory_delta,
            "residual_weights": residual_weights,
            "memory_weights": memory_weights,
            "mixture_weights": mixture_weights,
            "retrieval_mixture_weights": retrieval_mixture_weights,
        }


ProposedModelConfig = TSIFAConfig
