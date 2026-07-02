"""Smoke-test the TS-IFA training path on synthetic payloads."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from einops import repeat


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.experiments.train_ts_ifa import (  # noqa: E402
    PredictionPayloadDataset,
    loss_components,
    main,
    prepare_batch,
)
from ts_ifa.models.ts_ifa import TSIFAConfig, TimeSeriesInformedForecastingAdapter  # noqa: E402


def make_payload(prefix: str) -> dict[str, torch.Tensor]:
    n_dates, n_users, neighbors, lags, horizon = 5, 3, 2, 6, 2
    x = torch.randn(n_dates, n_users, lags)
    x_c = torch.randn(n_dates, n_users, neighbors, lags)
    y = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    y = y + 0.1 * torch.randn(n_dates, n_users, horizon)
    y_c = repeat(
        x_c[..., -1],
        "date user neighbor -> date user neighbor horizon",
        horizon=horizon,
    )
    y_c = y_c + 0.1 * torch.randn(
        n_dates,
        n_users,
        neighbors,
        horizon,
    )
    preds = repeat(x[..., -1], "date user -> date user horizon", horizon=horizon)
    preds_context = preds + 0.05 * torch.randn_like(preds)
    pred_neighbors = repeat(
        x_c[..., -1],
        "date user neighbor -> date user neighbor horizon",
        horizon=horizon,
    )
    return {
        f"{prefix}_preds": preds,
        f"{prefix}_preds_context": preds_context,
        f"{prefix}_E_values": y_c - pred_neighbors,
        f"{prefix}_X_values": x,
        f"{prefix}_Xc_values": x_c,
        f"{prefix}_Y_values": y,
        f"{prefix}_Yc_values": y_c,
    }


def check_query_scale_transfer() -> None:
    payload = {
        "train_preds": torch.tensor([[[7.0]]]),
        "train_preds_context": torch.tensor([[[7.0]]]),
        "train_E_values": torch.tensor([[[[2.0]]]]),
        "train_X_values": torch.tensor([[[3.0, 7.0]]]),
        "train_Xc_values": torch.tensor([[[[8.0, 12.0]]]]),
        "train_Y_values": torch.tensor([[[7.0]]]),
        "train_Yc_values": torch.tensor([[[[14.0]]]]),
    }
    dataset = PredictionPayloadDataset(payload, prefix="train")
    np_like = {
        "x_c": torch.tensor([[[3.0, 7.0]]]),
        "y_c": torch.tensor([[[9.0]]]),
        "pred_neighbors": torch.tensor([[[7.0]]]),
        "residual_c": torch.tensor([[[2.0]]]),
    }
    for name, expected in np_like.items():
        torch.testing.assert_close(dataset.tensors[name], expected)

    batch, _ = prepare_batch(dataset.tensors, normalization="instance", eps=1e-8)
    torch.testing.assert_close(batch["x_c"], torch.tensor([[[-1.0, 1.0]]]))
    torch.testing.assert_close(batch["y_c"], torch.tensor([[[2.0]]]))
    torch.testing.assert_close(batch["pred_neighbors"], torch.tensor([[[1.0]]]))
    torch.testing.assert_close(batch["residual_c"], torch.tensor([[[1.0]]]))


def check_identity_initialized_mixture() -> None:
    torch.manual_seed(1)
    config = TSIFAConfig(
        lags=6,
        horizon=2,
        neighbors=2,
        residual_heads=2,
        memory_heads=2,
        mixture_heads=2,
        residual_attn_dim=8,
        memory_attn_dim=8,
        mixture_attn_dim=8,
        residual_hidden=16,
        memory_hidden=16,
        mixture_hidden=16,
        mixture_key_dim=16,
    )
    model = TimeSeriesInformedForecastingAdapter(config)
    batch = {
        "x": torch.randn(4, 6),
        "x_c": torch.randn(4, 2, 6),
        "y_c": torch.randn(4, 2, 2),
        "pred": torch.randn(4, 2),
        "pred_context": torch.randn(4, 2),
        "pred_neighbors": torch.randn(4, 2, 2),
        "residual_c": torch.randn(4, 2, 2),
    }
    outputs = model(batch)
    expected_logits = torch.zeros_like(outputs["mixture_weights"])
    expected_logits[:, 1:, :] = config.mixture_gate_init
    expected_weights = torch.softmax(expected_logits, dim=1)
    torch.testing.assert_close(outputs["mixture_weights"], expected_weights)
    torch.testing.assert_close(outputs["residual_delta"], torch.zeros_like(outputs["residual_delta"]))
    torch.testing.assert_close(outputs["memory_delta"], torch.zeros_like(outputs["memory_delta"]))
    torch.testing.assert_close(outputs["residual_prediction"], batch["pred"])
    torch.testing.assert_close(outputs["memory_prediction"], batch["pred"])
    torch.testing.assert_close(outputs["memory_prediction"], batch["pred"] + outputs["memory_delta"])
    torch.testing.assert_close(
        outputs["prediction"],
        (outputs["mixture_weights"] * torch.stack(
            [
                batch["pred"],
                batch["pred_context"],
                outputs["residual_prediction"],
                outputs["memory_prediction"],
            ],
            dim=1,
        )).sum(dim=1),
    )


def check_memory_loss_component() -> None:
    outputs = {
        "prediction": torch.tensor([[1.0, 3.0]]),
        "residual_delta": torch.tensor([[0.0, 2.0]]),
        "memory_delta": torch.tensor([[1.0, 0.0]]),
    }
    batch = {
        "y": torch.tensor([[2.0, 4.0]]),
        "pred": torch.tensor([[1.0, 1.0]]),
    }
    state = {"loss_scale": torch.ones((1, 1))}
    losses = loss_components(outputs, batch, state, beta=0.5, gamma=0.25)
    residual_target = batch["y"] - batch["pred"]
    expected_prediction = (outputs["prediction"] - batch["y"]).pow(2).mean()
    expected_regularization = (outputs["prediction"] - batch["pred"]).pow(2).mean()
    expected_residual = (outputs["residual_delta"] - residual_target).pow(2).mean()
    expected_memory = (outputs["memory_delta"] - residual_target).pow(2).mean()
    expected_total = (
        expected_prediction
        + 0.5 * expected_regularization
        + 0.25 * (expected_residual + expected_memory)
    )
    torch.testing.assert_close(losses["prediction"], expected_prediction)
    torch.testing.assert_close(losses["regularization"], expected_regularization)
    torch.testing.assert_close(losses["residual"], expected_residual)
    torch.testing.assert_close(losses["memory"], expected_memory)
    torch.testing.assert_close(losses["loss"], expected_total)


def run() -> None:
    check_query_scale_transfer()
    check_identity_initialized_mixture()
    check_memory_loss_component()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_path = base / "train_prediction_payload.pt"
        oracle_path = base / "oracle_prediction_payload.pt"
        eval_path = base / "eval_prediction_payload.pt"
        out = base / "ts_ifa"
        torch.save(make_payload("train"), train_path)
        torch.save(make_payload("oracle"), oracle_path)
        torch.save(make_payload("eval"), eval_path)
        old_argv = sys.argv
        try:
            sys.argv = [
                "ts_ifa.experiments.train_ts_ifa",
                "--train-payload",
                str(train_path),
                "--eval-payload",
                str(eval_path),
                "--oracle-payload",
                str(oracle_path),
                "--output-dir",
                str(out),
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--residual-heads",
                "2",
                "--memory-heads",
                "2",
                "--mixture-heads",
                "2",
                "--residual-attn-dim",
                "8",
                "--memory-attn-dim",
                "8",
                "--mixture-attn-dim",
                "8",
                "--residual-hidden",
                "16",
                "--memory-hidden",
                "16",
                "--mixture-hidden",
                "16",
                "--mixture-key-dim",
                "16",
            ]
            paths = main()
        finally:
            sys.argv = old_argv
        for path in paths.values():
            assert Path(path).exists(), path
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        assert config["parameters"]["total"] > 0
        assert config["parameters"]["trainable"] == config["parameters"]["total"]
        training_history = json.loads(paths["history"].read_text(encoding="utf-8"))
        history = training_history["history"]
        train_steps = training_history["train_steps"]
        assert "valid_adapted_nmse" in history[0]
        assert "train_nmse" in train_steps[0]
        assert config["training"]["train_split"] == "T1"
        assert config["training"]["validation_split"] == "T2"
    print("TS-IFA training smoke checks passed")


if __name__ == "__main__":
    run()
