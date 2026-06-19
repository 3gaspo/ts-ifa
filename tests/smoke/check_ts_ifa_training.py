"""Smoke-test the TS-IFA training path on synthetic payloads."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from einops import repeat


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_ts_ifa import main  # noqa: E402


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


def run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        train_path = base / "train_prediction_payload.pt"
        eval_path = base / "eval_prediction_payload.pt"
        out = base / "ts_ifa"
        torch.save(make_payload("train"), train_path)
        torch.save(make_payload("eval"), eval_path)
        old_argv = sys.argv
        try:
            sys.argv = [
                "train_ts_ifa.py",
                "--train-payload",
                str(train_path),
                "--eval-payload",
                str(eval_path),
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
    print("TS-IFA training smoke checks passed")


if __name__ == "__main__":
    run()
