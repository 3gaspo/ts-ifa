"""Smoke-test artifact loading and dashboard calculations without a notebook kernel."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.visu.dashboard import (  # noqa: E402
    gate_roc,
    horizon_values,
    load_dashboard_data,
    plot_query_example,
    prediction_names,
    split_arrays,
)
from ts_ifa.experiments.evaluate_baselines import visualization_payload  # noqa: E402


def extraction_payload(prefix: str) -> dict[str, torch.Tensor]:
    dates, users, neighbors, lags, horizon = 3, 2, 2, 4, 3
    x = torch.arange(dates * users * lags, dtype=torch.float32).reshape(dates, users, lags)
    x_c = torch.stack([x - 2.0, x + 2.0], dim=2)
    y = x[..., -1:].repeat(1, 1, horizon) + torch.arange(horizon)
    y_c = x_c[..., -1:].repeat(1, 1, 1, horizon) + torch.arange(horizon)
    query_t = torch.arange(10, 10 + dates).unsqueeze(1).repeat(1, users)
    query_user = torch.arange(users).unsqueeze(0).repeat(dates, 1)
    return {
        f"{prefix}_X_values": x,
        f"{prefix}_Y_values": y,
        f"{prefix}_Xc_values": x_c,
        f"{prefix}_Yc_values": y_c,
        f"{prefix}_preds": y + 1.0,
        f"{prefix}_preds_context": y + 0.25,
        f"{prefix}_query_t": query_t,
        f"{prefix}_query_user_idx": query_user,
        f"{prefix}_neighbor_t": query_t.unsqueeze(-1).repeat(1, 1, neighbors) - 1,
        f"{prefix}_neighbor_user_idx": query_user.unsqueeze(-1).repeat(1, 1, neighbors),
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payloads = {split: extraction_payload(split) for split in ("train", "oracle", "eval")}
        for split, payload in payloads.items():
            torch.save(payload, root / f"{split}_prediction_payload.pt")

        baseline_predictions = {}
        gate_predictions = {}
        diagnostics_by_split = {}
        for split, payload in payloads.items():
            target = payload[f"{split}_Y_values"].reshape(-1, 3)
            score = torch.linspace(-1.0, 1.0, len(target))
            baseline_predictions[split] = {
                "vanilla": target + 1.0,
                "context_conditioned": target + 0.25,
                "neighbor_weighted_mean": target + 0.5,
            }
            gate_predictions[split] = {
                "gated_context_classifier_scalar": target + 0.2,
                "oracle_context_scalar": target + 0.1,
            }
            diagnostics_by_split[split] = {
                "classifier_scalar_score": score,
                "classifier_scalar_target": score,
            }
        baseline_dir = root / "baselines"
        baseline_dir.mkdir()
        torch.save(
            visualization_payload(baseline_predictions, {}),
            baseline_dir / "visualization_payload.pt",
        )
        gate_dir = root / "gates"
        gate_dir.mkdir()
        torch.save(
            visualization_payload(gate_predictions, diagnostics_by_split),
            gate_dir / "visualization_payload.pt",
        )

        ts_ifa_dir = root / "ts_ifa"
        ts_ifa_dir.mkdir()
        target = payloads["eval"]["eval_Y_values"].reshape(-1, 3)
        torch.save(
            {"split": "eval", "predictions": {"ts_ifa": target + 0.05}},
            ts_ifa_dir / "eval_predictions.pt",
        )

        data = load_dashboard_data(root)
        arrays = split_arrays(data, "eval")
        assert arrays["x"].shape == (6, 4)
        assert "neighbor_weighted_mean" in prediction_names(data, "eval")
        assert "gated_context_classifier_scalar" in prediction_names(data, "eval")
        assert "ts_ifa" in prediction_names(data, "eval")

        values, _ = horizon_values(
            data,
            "eval",
            "context_conditioned",
            "vanilla",
            "relative mse",
            instance_normalized=False,
        )
        assert values.shape == (3,)
        assert (values > 0).all()

        _, _, auc, accuracy, count = gate_roc(data, "eval", "classifier_scalar")
        assert auc == 1.0
        assert accuracy == 1.0
        assert count == 6

        figure = plot_query_example(
            data,
            "eval",
            0,
            instance_normalized=True,
            hide_axes=False,
        )
        plt.close(figure)
    print("retrieval dashboard smoke check passed")


if __name__ == "__main__":
    main()
