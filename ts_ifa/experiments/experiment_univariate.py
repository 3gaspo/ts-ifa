"""Evaluate a pretrained model without neighbors on all users."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import rearrange

from ..data.load_dataset_model import (
    load_csv_dataset,
    load_json_kwargs,
    load_pretrained_model,
    resolve_device,
    run_dir,
    set_seed,
    split_bounds,
)
from ..data.neighbors import period_eval_dates
from ..visu import (
    plot_error_distribution,
    plot_horizon_errors,
    plot_prediction_example,
    plot_user_error_scatter,
)
from .runtime import setup_logging


LOGGER = logging.getLogger(__name__)


def _loss_tensors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lookback: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    err = prediction - target
    std = lookback.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return {
        "mse": err.pow(2).mean(dim=(1, 2)),
        "mae": err.abs().mean(dim=(1, 2)),
        "nmse": (err / std).pow(2).mean(dim=(1, 2)),
        "horizon_mse": err.pow(2).mean(dim=1),
    }


def evaluate_split(
    dataset,
    model,
    *,
    split_name: str,
    dates: np.ndarray,
    lags: int,
    horizon: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    rows = []
    horizon_losses = []
    first_payload = None
    with torch.inference_mode():
        for t in dates:
            x, y = dataset.window_tensor(int(t), lags, horizon, device=device)
            past_cov, future_cov = dataset.covariate_tensors(int(t), lags, horizon, device=device)
            pred = model(x, past_covariates=past_cov, future_covariates=future_cov)
            losses = _loss_tensors(pred, y, x)
            horizon_losses.append(losses["horizon_mse"].detach().cpu())
            if first_payload is None:
                first_payload = {
                    "date": int(t),
                    "x": x[0, 0].detach().cpu(),
                    "y": y[0, 0].detach().cpu(),
                    "pred": pred[0, 0].detach().cpu(),
                }
            for user_idx, user_name in enumerate(dataset.user_names):
                rows.append(
                    {
                        "split": split_name,
                        "date_index": int(t),
                        "datetime": str(dataset.datetimes[int(t)]),
                        "user_idx": int(user_idx),
                        "user_name": user_name,
                        "mse": float(losses["mse"][user_idx].detach().cpu()),
                        "mae": float(losses["mae"][user_idx].detach().cpu()),
                        "nmse": float(losses["nmse"][user_idx].detach().cpu()),
                    }
                )
    payload = {
        "dates": torch.as_tensor(dates, dtype=torch.long),
        "horizon_mse": rearrange(horizon_losses, "batch user horizon -> batch user horizon")
        if horizon_losses
        else torch.empty(0),
        "example": first_payload,
    }
    return pd.DataFrame(rows), payload


def summarize_losses(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split, split_df in frame.groupby("split"):
        summary[split] = {}
        for metric in ("mse", "mae", "nmse"):
            per_user = split_df.groupby("user_idx")[metric].mean()
            tail_count = max(1, int(np.ceil(0.1 * len(per_user))))
            summary[split][metric] = {
                "mean": float(split_df[metric].mean()),
                "std": float(split_df[metric].std(ddof=0)),
                "w10_user_mean": float(np.sort(per_user.to_numpy())[-tail_count:].mean()),
            }
    return summary


def save_plots(frame: pd.DataFrame, payloads: dict[str, Any], output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for split, split_df in frame.groupby("split"):
        per_user = split_df.groupby("user_idx")["nmse"].agg(["mean", "std"]).reset_index()
        per_user = per_user.rename(columns={"mean": "mean_loss", "std": "std_loss"})
        plot_user_error_scatter(
            per_user,
            plot_dir,
            f"{split}_user_nmse.png",
            title=f"{split} per-user nMSE",
        )
        plot_error_distribution(
            split_df["nmse"].to_numpy(),
            plot_dir,
            f"{split}_nmse_distribution.png",
            title=f"{split} nMSE distribution",
        )
        horizon = payloads[split]["horizon_mse"]
        if torch.is_tensor(horizon) and horizon.numel() > 0:
            plot_horizon_errors(
                horizon.mean(dim=(0, 1)).numpy(),
                plot_dir,
                f"{split}_horizon_mse.png",
                title=f"{split} mean MSE by horizon",
            )
        example = payloads[split].get("example")
        if example:
            plot_prediction_example(
                example["x"].numpy(),
                example["y"].numpy(),
                example["pred"].numpy(),
                plot_dir,
                f"{split}_prediction_example.png",
                title=f"{split} example prediction",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV file or dataset directory")
    parser.add_argument("--dataset-name", default=None, help="CSV stem when --csv is a directory")
    parser.add_argument("--target-cols", default=None)
    parser.add_argument("--past-covariate-cols", default=None)
    parser.add_argument("--future-covariate-cols", default=None)
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--drop-users", default=None)
    parser.add_argument("--rename-users", action="store_true")
    parser.add_argument("--aggr", default=None)
    parser.add_argument("--aggr-period", default="h")
    parser.add_argument("--model", default="persistence")
    parser.add_argument("--model-kwargs", default=None, help="JSON string or path")
    parser.add_argument("--pretrained-path", default=None)
    parser.add_argument("--normalization", default="none", choices=["none", "instance"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lags", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--splits", default="0.3,0.35,0.15,0.2")
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument(
        "--eval-splits",
        default="eval",
        help="Any of train,oracle,eval separated by comma/semicolon",
    )
    parser.add_argument("--output-dir", default="outputs/results")
    parser.add_argument("--save-name", default="univariate")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    started = perf_counter()
    dataset_name = args.dataset_name or Path(args.csv).stem
    LOGGER.info(
        "experiment start kind=univariate dataset=%s model=%s lags=%s horizon=%s",
        dataset_name,
        args.model,
        args.lags,
        args.horizon,
    )
    set_seed(args.seed)
    LOGGER.info("dataset load start")
    dataset = load_csv_dataset(
        args.csv,
        dataset_name=args.dataset_name,
        target_cols=args.target_cols,
        past_covariate_cols=args.past_covariate_cols,
        future_covariate_cols=args.future_covariate_cols,
        date_col=args.date_col,
        drop_users=args.drop_users,
        rename_users=args.rename_users,
        aggr=args.aggr,
        aggr_period=args.aggr_period,
    )
    LOGGER.info("dataset load done dates=%s users=%s", dataset.n_dates, dataset.n_users)
    t0_end, t1_end, t2_end, t3_end = split_bounds(dataset.n_dates, args.splits)
    split_ranges = {
        "train": (t0_end, t1_end),
        "oracle": (t1_end, t2_end),
        "eval": (t2_end, t3_end),
    }
    selected = [part.strip() for part in args.eval_splits.replace(";", ",").split(",") if part.strip()]
    LOGGER.info("model load start")
    model = load_pretrained_model(
        args.model,
        lags=args.lags,
        horizon=args.horizon,
        dim=1,
        normalization=args.normalization,
        pretrained_path=args.pretrained_path,
        device=args.device,
        model_kwargs=load_json_kwargs(args.model_kwargs),
    )
    device = resolve_device(args.device)
    LOGGER.info("model load done device=%s", device)
    out = run_dir(args.output_dir, args.save_name)

    frames = []
    payloads: dict[str, Any] = {}
    for split in selected:
        if split not in split_ranges:
            raise ValueError(f"unknown eval split {split!r}")
        start, end = split_ranges[split]
        dates = period_eval_dates(
            start,
            end,
            n_dates=dataset.n_dates,
            lags=args.lags,
            horizon=args.horizon,
            stride=args.eval_stride,
        )
        LOGGER.info("evaluation start split=%s queries=%s", split, len(dates))
        frame, payload = evaluate_split(
            dataset,
            model,
            split_name=split,
            dates=dates,
            lags=args.lags,
            horizon=args.horizon,
            device=device,
        )
        frames.append(frame)
        payloads[split] = payload
        LOGGER.info("evaluation done split=%s", split)

    losses = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    csv_path = out / "univariate_losses.csv"
    losses.to_csv(csv_path, index=False)
    summary = summarize_losses(losses)
    summary_path = out / "univariate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    payload_path = out / "univariate_payload.pt"
    torch.save(payloads, payload_path)
    if not losses.empty:
        save_plots(losses, payloads, out)
    LOGGER.info("outputs saved dir=%s", out)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - started)
    return {"losses": csv_path, "summary": summary_path, "payload": payload_path}


if __name__ == "__main__":
    main()
