"""Feature extraction from neighbor prediction payloads."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat

from ..visu import plot_feature_scatter
from .runtime import log_experiment_separator, setup_logging


LOGGER = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flat(value: Any) -> np.ndarray:
    return rearrange(_to_numpy(value), "... -> (...)")


def mse_by_window(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2).mean(dim=-1)


def _optional_mean(payload: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    arr = _to_numpy(payload[key])
    if arr.ndim >= 3:
        return rearrange(arr.mean(axis=tuple(range(2, arr.ndim))), "... -> (...)")
    return rearrange(arr, "... -> (...)")


def _optional_std(payload: dict[str, Any], key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    arr = _to_numpy(payload[key])
    if arr.ndim >= 3:
        return rearrange(arr.std(axis=tuple(range(2, arr.ndim))), "... -> (...)")
    return np.zeros(rearrange(arr, "... -> (...)").shape, dtype=np.float32)


def compute_split_features(
    prediction_payload: dict[str, Any],
    *,
    prefix: str,
    feature_payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one row per ``(date, user)`` query window."""
    features = feature_payload or {}
    preds = prediction_payload[f"{prefix}_preds"].float()
    preds_context = prediction_payload[f"{prefix}_preds_context"].float()
    target = prediction_payload[f"{prefix}_Y_values"].float()

    base_loss = mse_by_window(preds, target)
    context_loss = mse_by_window(preds_context, target)
    oracle_loss = torch.minimum(base_loss, context_loss)
    improvement = 100.0 * (base_loss - context_loss) / base_loss.clamp_min(1e-8)
    oracle_improvement = 100.0 * (base_loss - oracle_loss) / base_loss.clamp_min(1e-8)

    n_eval, n_users = base_loss.shape
    dates = _to_numpy(prediction_payload[f"{prefix}_dates"]).astype(np.int64)
    datetimes = prediction_payload.get(f"{prefix}_datetimes")
    if datetimes is None:
        datetimes = [str(value) for value in dates]

    frame = pd.DataFrame(
        {
            "split": prefix,
            "date_index": repeat(dates, "date -> (date user)", user=n_users),
            "datetime": repeat(np.asarray(datetimes, dtype=object), "date -> (date user)", user=n_users),
            "user_idx": repeat(np.arange(n_users), "user -> (date user)", date=n_eval),
            "base_loss": _flat(base_loss),
            "context_loss": _flat(context_loss),
            "oracle_loss": _flat(oracle_loss),
            "context_better": _flat(context_loss < base_loss).astype(bool),
            "improvement_pct": _flat(improvement),
            "oracle_improvement_pct": _flat(oracle_improvement),
            "pred_context_mse": _flat((preds - preds_context).pow(2).mean(dim=-1)),
        }
    )

    distance = _optional_mean(prediction_payload, f"{prefix}_distance_x_xc")
    if distance is not None:
        frame["distance_mean"] = distance
        frame["distance_std"] = _optional_std(prediction_payload, f"{prefix}_distance_x_xc")

    for key in ("store_date_count", "store_window_count"):
        values = _optional_mean(prediction_payload, f"{prefix}_{key}")
        if values is not None:
            frame[key] = values

    neighbor_user = prediction_payload.get(f"{prefix}_neighbor_user_idx")
    if neighbor_user is not None:
        users = repeat(np.arange(n_users), "user -> date user 1", date=n_eval)
        neighbor_arr = _to_numpy(neighbor_user)
        if neighbor_arr.shape[-1] > 0:
            frame["same_user_neighbor_frac"] = rearrange(
                (neighbor_arr == users).mean(axis=-1),
                "... -> (...)",
            )

    neighbor_t = prediction_payload.get(f"{prefix}_neighbor_t")
    query_t = prediction_payload.get(f"{prefix}_query_t")
    if neighbor_t is not None and query_t is not None and _to_numpy(neighbor_t).shape[-1] > 0:
        delta = rearrange(_to_numpy(query_t), "date user -> date user 1") - _to_numpy(neighbor_t)
        frame["neighbor_delta_mean"] = rearrange(delta.mean(axis=-1), "... -> (...)")
        frame["neighbor_delta_std"] = rearrange(delta.std(axis=-1), "... -> (...)")

    for key in (
        "mu_x",
        "sigma_x",
        "mu_xc_mean",
        "sigma_xc_mean",
        "mu_xc_std",
        "sigma_xc_std",
        "loss_pred_yc_mean",
        "loss_neighbor_residual_mean",
        "loss_neighbor_context_residual_mean",
    ):
        full_key = f"{prefix}_{key}"
        if full_key in features:
            frame[key] = _flat(features[full_key])

    if "mu_x" in frame and "mu_xc_mean" in frame:
        frame["mu_diff"] = frame["mu_x"] - frame["mu_xc_mean"]
    if "sigma_x" in frame and "sigma_xc_mean" in frame:
        frame["sigma_diff"] = frame["sigma_x"] - frame["sigma_xc_mean"]
    return frame


def load_split_payloads(input_dir: str | Path, prefix: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    base = Path(input_dir).expanduser()
    pred_path = base / f"{prefix}_prediction_payload.pt"
    feat_path = base / f"{prefix}_features_payload.pt"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    prediction = torch.load(pred_path, map_location="cpu", weights_only=False)
    feature = torch.load(feat_path, map_location="cpu", weights_only=False) if feat_path.exists() else None
    return prediction, feature


def build_feature_table(input_dir: str | Path, prefixes: list[str]) -> pd.DataFrame:
    frames = []
    for prefix in prefixes:
        prediction, feature = load_split_payloads(input_dir, prefix)
        frames.append(compute_split_features(prediction, prefix=prefix, feature_payload=feature))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_feature_outputs(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "neighbor_features.csv"
    frame.to_csv(csv_path, index=False)

    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    pairs = [
        ("distance_mean", "improvement_pct", "improvement_vs_distance.png"),
        ("loss_pred_yc_mean", "improvement_pct", "improvement_vs_pred_yc.png"),
        ("loss_neighbor_residual_mean", "improvement_pct", "improvement_vs_neighbor_residual.png"),
        ("mu_diff", "improvement_pct", "improvement_vs_mu_diff.png"),
        ("sigma_diff", "improvement_pct", "improvement_vs_sigma_diff.png"),
        ("base_loss", "context_loss", "context_vs_base_loss.png"),
    ]
    for x_col, y_col, filename in pairs:
        plot_feature_scatter(frame, x_col, y_col, plot_dir, filename)
    return {"csv": csv_path, "plots": plot_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory with *_prediction_payload.pt files")
    parser.add_argument("--output-dir", default=None, help="Defaults to <input-dir>/feature_analysis")
    parser.add_argument("--prefixes", default="train,oracle,eval", help="Comma/semicolon-separated split prefixes")
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    started = perf_counter()
    prefixes = [part.strip() for part in args.prefixes.replace(";", ",").split(",") if part.strip()]
    output_dir = args.output_dir or str(Path(args.input_dir) / "feature_analysis")
    LOGGER.info("experiment start kind=feature_analysis input=%s", args.input_dir)
    LOGGER.info("feature table start splits=%s", ",".join(prefixes))
    frame = build_feature_table(args.input_dir, prefixes)
    LOGGER.info("feature table done rows=%s columns=%s", len(frame), len(frame.columns))
    outputs = save_feature_outputs(frame, output_dir)
    LOGGER.info("outputs saved dir=%s", output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - started)
    log_experiment_separator(LOGGER)
    return outputs


if __name__ == "__main__":
    main()
