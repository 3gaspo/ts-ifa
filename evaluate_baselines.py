"""Evaluate notebook/LaTeX baselines from extracted neighbor payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def flatten_payload(payload: dict[str, Any], prefix: str) -> dict[str, np.ndarray]:
    y_c = payload[f"{prefix}_Yc_values"].float()
    e = payload[f"{prefix}_E_values"].float()
    pred_neighbors = y_c - e
    return {
        "pred": rearrange(payload[f"{prefix}_preds"].float(), "date user horizon -> (date user) horizon").numpy(),
        "pred_c": rearrange(
            payload[f"{prefix}_preds_context"].float(),
            "date user horizon -> (date user) horizon",
        ).numpy(),
        "y": rearrange(payload[f"{prefix}_Y_values"].float(), "date user horizon -> (date user) horizon").numpy(),
        "x": rearrange(payload[f"{prefix}_X_values"].float(), "date user lags -> (date user) lags").numpy(),
        "y_c": rearrange(y_c, "date user neighbor horizon -> (date user) neighbor horizon").numpy(),
        "e": rearrange(e, "date user neighbor horizon -> (date user) neighbor horizon").numpy(),
        "pred_neighbors": rearrange(
            pred_neighbors,
            "date user neighbor horizon -> (date user) neighbor horizon",
        ).numpy(),
        "distance": rearrange(
            payload[f"{prefix}_distance_x_xc"].float(),
            "date user neighbor -> (date user) neighbor",
        ).numpy(),
        "query_t": rearrange(payload[f"{prefix}_query_t"], "date user -> (date user)").numpy(),
    }


def split_train_arrays(arrays: dict[str, np.ndarray], oracle_fraction: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    query_t = arrays["query_t"]
    unique_dates = np.unique(query_t)
    if len(unique_dates) < 2:
        return arrays, arrays
    n_oracle_dates = max(1, int(np.ceil(len(unique_dates) * float(oracle_fraction))))
    n_oracle_dates = min(n_oracle_dates, len(unique_dates) - 1)
    oracle_start = unique_dates[-n_oracle_dates]
    fit_mask = query_t < oracle_start
    gate_mask = ~fit_mask
    return subset_arrays(arrays, fit_mask), subset_arrays(arrays, gate_mask)


def subset_arrays(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] if value.shape[0] == mask.shape[0] else value for key, value in arrays.items()}


def distance_weights(arrays: dict[str, np.ndarray], eps: float = 1e-8) -> np.ndarray:
    d = arrays["distance"].astype(np.float64)
    d_std = d.std(axis=-1, keepdims=True)
    d_norm = (d - d.min(axis=-1, keepdims=True)) / np.maximum(d_std, eps)
    return softmax_np(-d_norm, axis=-1)


def weighted_neighbor_horizon(arrays: dict[str, np.ndarray]) -> np.ndarray:
    w = distance_weights(arrays)
    return (w[:, :, None] * arrays["y_c"]).sum(axis=1)


def weighted_neighbor_residual(arrays: dict[str, np.ndarray]) -> np.ndarray:
    w = distance_weights(arrays)
    return (w[:, :, None] * arrays["e"]).sum(axis=1)


def ridge_no_intercept(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xtx = x.T @ x
    reg = float(l2) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def fit_baseline_adapters(train: dict[str, np.ndarray], l2: float) -> dict[str, Any]:
    pred = train["pred"]
    y = train["y"]
    y_c = train["y_c"]
    pred_c = train["pred_c"]
    if y.shape[0] == 0:
        raise ValueError("cannot fit baseline adapters from an empty train payload")
    weighted = weighted_neighbor_horizon(train)
    residual_target = y - pred

    mix0_direction = weighted - pred
    lam = np.mean(mix0_direction * residual_target) / (np.mean(mix0_direction**2) + float(l2))
    lam = float(np.clip(lam, 0.0, 1.0))

    mix1_x = np.concatenate([pred[:, :, None], np.moveaxis(y_c, 1, 2)], axis=-1)
    mix1_coef = ridge_no_intercept(
        rearrange(mix1_x, "sample horizon feature -> (sample horizon) feature"),
        rearrange(residual_target, "sample horizon -> (sample horizon)"),
        l2,
    )

    horizon = y.shape[1]
    neighbors = y_c.shape[1]
    mix2_coef = np.zeros((horizon, neighbors + 2), dtype=np.float64)
    for h in range(horizon):
        x_h = np.concatenate(
            [
                pred[:, h : h + 1],
                y_c[:, :, h],
                pred_c[:, h : h + 1],
            ],
            axis=1,
        )
        mix2_coef[h] = ridge_no_intercept(x_h, residual_target[:, h], l2)

    return {
        "mix_0_lambda": lam,
        "mix_1_coef": mix1_coef,
        "mix_2_coef": mix2_coef,
    }


def predict_baseline_adapters(arrays: dict[str, np.ndarray], artifacts: dict[str, Any]) -> dict[str, np.ndarray]:
    pred = arrays["pred"]
    y_c = arrays["y_c"]
    pred_c = arrays["pred_c"]
    weighted = weighted_neighbor_horizon(arrays)
    unweighted = y_c.mean(axis=1)
    weighted_e = weighted_neighbor_residual(arrays)

    predictions: dict[str, np.ndarray] = {
        "vanilla": pred,
        "context_conditioned": pred_c,
        "neighbor_weighted_mean": weighted,
        "neighbor_unweighted_mean": unweighted,
        "pred_plus_weighted_e": pred + weighted_e,
        "mix_0_weighted": (1.0 - artifacts["mix_0_lambda"]) * pred + artifacts["mix_0_lambda"] * weighted,
    }

    mix1_coef = artifacts["mix_1_coef"]
    mix1_x = np.concatenate([pred[:, :, None], np.moveaxis(y_c, 1, 2)], axis=-1)
    correction = np.einsum("shf,f->sh", mix1_x, mix1_coef)
    predictions["mix_1_learned"] = pred + correction

    mix2_coef = artifacts["mix_2_coef"]
    full = np.empty_like(pred)
    for h in range(pred.shape[1]):
        x_h = np.concatenate([pred[:, h : h + 1], y_c[:, :, h], pred_c[:, h : h + 1]], axis=1)
        full[:, h] = pred[:, h] + x_h @ mix2_coef[h]
    predictions["mix_2_full_horizon"] = full
    return predictions


def gate_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    pred = arrays["pred"]
    pred_c = arrays["pred_c"]
    y_c = arrays["y_c"]
    e = arrays["e"]
    distance = arrays["distance"]
    x = arrays["x"]
    weighted = weighted_neighbor_horizon(arrays)
    weighted_e = weighted_neighbor_residual(arrays)
    cols = [
        ((pred_c - pred) ** 2).mean(axis=1),
        ((weighted - pred) ** 2).mean(axis=1),
        (weighted_e**2).mean(axis=1),
        (e**2).mean(axis=(1, 2)),
        distance.mean(axis=1),
        distance.std(axis=1),
        x.mean(axis=1),
        x.std(axis=1),
        y_c.mean(axis=(1, 2)),
        y_c.std(axis=(1, 2)),
    ]
    return np.stack(cols, axis=1).astype(np.float32)


class TorchGate(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_gate(x_np: np.ndarray, y_np: np.ndarray, *, epochs: int, lr: float, weight_decay: float, seed: int) -> TorchGate:
    if x_np.shape[0] == 0:
        raise ValueError("cannot train baseline gates from an empty oracle-train slice")
    torch.manual_seed(int(seed))
    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(y_np, dtype=torch.float32)
    model = TorchGate(x.shape[1], y.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        optimizer.step()
    return model


def predict_gate(model: TorchGate, features: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        logits = model(torch.as_tensor(features, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def candidate_names(predictions: dict[str, np.ndarray]) -> list[str]:
    return [name for name in predictions if name != "vanilla"]


def add_gate_predictions(
    base_predictions_by_split: dict[str, dict[str, np.ndarray]],
    train_gate_predictions: dict[str, np.ndarray],
    train_gate: dict[str, np.ndarray],
    arrays_by_split: dict[str, dict[str, np.ndarray]],
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_horizon_gate: bool,
) -> dict[str, dict[str, np.ndarray]]:
    base_loss = (train_gate["y"] - train_gate["pred"]) ** 2
    features_train = gate_features(train_gate)

    scalar_models = {}
    horizon_models = {}
    for offset, name in enumerate(candidate_names(train_gate_predictions)):
        candidate_loss = (train_gate["y"] - train_gate_predictions[name]) ** 2
        scalar_label = (candidate_loss.mean(axis=1) < base_loss.mean(axis=1)).astype(np.float32)[:, None]
        scalar_models[name] = fit_gate(
            features_train,
            scalar_label,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed + offset,
        )
        if train_horizon_gate:
            horizon_label = (candidate_loss < base_loss).astype(np.float32)
            horizon_models[name] = fit_gate(
                features_train,
                horizon_label,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed + 1000 + offset,
            )

    out: dict[str, dict[str, np.ndarray]] = {}
    for split, arrays in arrays_by_split.items():
        split_predictions = dict(base_predictions_by_split[split])
        features = gate_features(arrays)
        for name, scalar_model in scalar_models.items():
            scalar_prob = predict_gate(scalar_model, features)
            gated = np.where(
                scalar_prob >= 0.5,
                base_predictions_by_split[split][name],
                arrays["pred"],
            )
            split_predictions[f"oracle(scalar)_{name}"] = gated
            if name == "context_conditioned":
                split_predictions["gated_context_scalar"] = gated
        for name, horizon_model in horizon_models.items():
            horizon_prob = predict_gate(horizon_model, features)
            gated = np.where(
                horizon_prob >= 0.5,
                base_predictions_by_split[split][name],
                arrays["pred"],
            )
            split_predictions[f"oracle(horizon)_{name}"] = gated
            if name == "context_conditioned":
                split_predictions["gated_context_horizon"] = gated
        out[split] = split_predictions
    return out


def evaluate_predictions(split: str, arrays: dict[str, np.ndarray], predictions: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    y = arrays["y"]
    scale = np.maximum(arrays["x"].std(axis=1, keepdims=True), 1e-8)
    vanilla_nmse = np.mean(((arrays["pred"] - y) / scale) ** 2)
    for name, pred in predictions.items():
        err = pred - y
        mse = np.mean(err**2)
        mae = np.mean(np.abs(err))
        nmse = np.mean((err / scale) ** 2)
        rows.append(
            {
                "split": split,
                "baseline": name,
                "mse": float(mse),
                "mae": float(mae),
                "nmse": float(nmse),
                "relative_nmse_improvement_pct": float(100.0 * (vanilla_nmse - nmse) / max(vanilla_nmse, 1e-12)),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prefixes", default="train,eval")
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--oracle-train-fraction", type=float, default=0.15)
    parser.add_argument("--gate-epochs", type=int, default=300)
    parser.add_argument("--gate-lr", type=float, default=1e-2)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-3)
    parser.add_argument("--train-horizon-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else input_dir / "baseline_adapters"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [part.strip() for part in args.prefixes.replace(";", ",").split(",") if part.strip()]
    arrays_by_split = {
        prefix: flatten_payload(torch_load(input_dir / f"{prefix}_prediction_payload.pt"), prefix)
        for prefix in prefixes
    }
    if "train" not in arrays_by_split:
        raise ValueError("baseline fitting requires a train payload")

    fit_train, gate_train = split_train_arrays(arrays_by_split["train"], args.oracle_train_fraction)
    artifacts = fit_baseline_adapters(fit_train, args.l2)

    predictions_by_split_base = {
        split: predict_baseline_adapters(arrays, artifacts)
        for split, arrays in arrays_by_split.items()
    }
    train_gate_predictions = predict_baseline_adapters(gate_train, artifacts)
    predictions_by_split = add_gate_predictions(
        predictions_by_split_base,
        train_gate_predictions,
        gate_train,
        arrays_by_split,
        epochs=args.gate_epochs,
        lr=args.gate_lr,
        weight_decay=args.gate_weight_decay,
        seed=args.seed,
        train_horizon_gate=args.train_horizon_gate,
    )

    rows = []
    for split, arrays in arrays_by_split.items():
        rows.extend(evaluate_predictions(split, arrays, predictions_by_split[split]))
    frame = pd.DataFrame(rows)
    csv_path = output_dir / "baseline_metrics.csv"
    json_path = output_dir / "baseline_metrics.json"
    artifact_path = output_dir / "baseline_artifacts.pt"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    torch.save(
        {
            "mix_artifacts": artifacts,
            "gate_config": {
                "epochs": args.gate_epochs,
                "lr": args.gate_lr,
                "weight_decay": args.gate_weight_decay,
                "train_horizon_gate": args.train_horizon_gate,
            },
        },
        artifact_path,
    )
    return {"csv": csv_path, "json": json_path, "artifacts": artifact_path}


if __name__ == "__main__":
    main()
