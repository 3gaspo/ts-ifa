"""Run aligned neighbor extraction for a pretrained forecaster."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange

try:
    from .load_dataset_model import (
        load_csv_dataset,
        load_json_kwargs,
        load_pretrained_model,
        resolve_device,
        run_dir,
        set_seed,
        split_bounds,
    )
    from .neighbors import (
        aligned_store_dates,
        build_window_batch,
        period_eval_dates,
        search_neighbors,
    )
    from .visu import plot_series
except ImportError:  # pragma: no cover - direct script execution
    from load_dataset_model import (
        load_csv_dataset,
        load_json_kwargs,
        load_pretrained_model,
        resolve_device,
        run_dir,
        set_seed,
        split_bounds,
    )
    from neighbors import (
        aligned_store_dates,
        build_window_batch,
        period_eval_dates,
        search_neighbors,
    )
    from visu import plot_series


def _empty_neighbor_tensors(
    n_users: int,
    n_neighbors: int,
    lags: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    return {
        "E": torch.zeros((n_users, n_neighbors, horizon), dtype=torch.float32),
        "Xc": torch.zeros((n_users, n_neighbors, lags), dtype=torch.float32),
        "Yc": torch.zeros((n_users, n_neighbors, horizon), dtype=torch.float32),
        "mu_xc": torch.zeros((n_users, n_neighbors), dtype=torch.float32),
        "sigma_xc": torch.zeros((n_users, n_neighbors), dtype=torch.float32),
        "distance": torch.zeros((n_users, n_neighbors), dtype=torch.float32),
        "neighbor_t": torch.zeros((n_users, n_neighbors), dtype=torch.long),
        "neighbor_user": torch.zeros((n_users, n_neighbors), dtype=torch.long),
    }


def _payload_size_gb(payload: dict[str, Any]) -> float:
    total = 0
    for value in payload.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    return total / 1024**3


def _predict(
    model,
    x: torch.Tensor,
    *,
    context: torch.Tensor | None,
    past_covariates: torch.Tensor | None,
    future_covariates: torch.Tensor | None,
) -> torch.Tensor:
    prediction = model(
        x,
        context=context,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    ).detach().cpu()
    return rearrange(prediction, "user 1 horizon -> user horizon")


def extract_period(
    *,
    dataset,
    model,
    prefix: str,
    eval_dates: np.ndarray,
    store_start: int,
    store_end: int,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_eval = int(len(eval_dates))
    n_users = dataset.n_users
    k = int(args.neighbors)
    lags = int(args.lags)
    horizon = int(args.horizon)

    preds = torch.empty((n_eval, n_users, horizon), dtype=torch.float32)
    preds_context = torch.empty_like(preds)
    e_values = torch.empty((n_eval, n_users, k, horizon), dtype=torch.float32)
    ec_values = (
        torch.empty((n_eval, n_users, k, horizon), dtype=torch.float32)
        if args.compute_ec
        else None
    )
    x_values = torch.empty((n_eval, n_users, lags), dtype=torch.float32)
    xc_values = torch.empty((n_eval, n_users, k, lags), dtype=torch.float32)
    y_values = torch.empty((n_eval, n_users, horizon), dtype=torch.float32)
    yc_values = torch.empty((n_eval, n_users, k, horizon), dtype=torch.float32)
    mu_x = torch.empty((n_eval, n_users), dtype=torch.float32)
    sigma_x = torch.empty((n_eval, n_users), dtype=torch.float32)
    mu_xc = torch.empty((n_eval, n_users, k), dtype=torch.float32)
    sigma_xc = torch.empty((n_eval, n_users, k), dtype=torch.float32)
    distance_x_xc = torch.empty((n_eval, n_users, k), dtype=torch.float32)
    query_t = torch.empty((n_eval, n_users), dtype=torch.long)
    query_user_idx = torch.empty((n_eval, n_users), dtype=torch.long)
    neighbor_t = torch.empty((n_eval, n_users, k), dtype=torch.long)
    neighbor_user_idx = torch.empty((n_eval, n_users, k), dtype=torch.long)

    for i, t_raw in enumerate(eval_dates):
        t = int(t_raw)
        query = build_window_batch(
            dataset,
            np.asarray([t], dtype=np.int64),
            lags=lags,
            horizon=horizon,
            distance_space=args.distance_space,
            model=model,
            device=device,
            normalize=not args.no_feature_normalization,
            pool_representation=args.pool_representation,
        )
        x = rearrange(query.windows[:, :lags], "user lags -> user 1 lags").to(device)
        y = query.windows[:, lags:]
        past_cov, future_cov = dataset.covariate_tensors(t, lags, horizon, device=device)
        pred = _predict(
            model,
            x,
            context=None,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

        x_values[i] = rearrange(x.detach().cpu(), "user 1 lags -> user lags")
        y_values[i] = y
        preds[i] = pred
        mu_x[i] = rearrange(x.detach().cpu().mean(dim=-1), "user 1 -> user")
        sigma_x[i] = rearrange(
            x.detach().cpu().std(dim=-1, unbiased=False),
            "user 1 -> user",
        )
        query_t[i] = t
        query_user_idx[i] = torch.arange(n_users, dtype=torch.long)

        if k == 0:
            empty = _empty_neighbor_tensors(n_users, k, lags, horizon)
            preds_context[i] = pred
            e_values[i] = empty["E"]
            if ec_values is not None:
                ec_values[i] = empty["E"]
            xc_values[i] = empty["Xc"]
            yc_values[i] = empty["Yc"]
            mu_xc[i] = empty["mu_xc"]
            sigma_xc[i] = empty["sigma_xc"]
            distance_x_xc[i] = empty["distance"]
            neighbor_t[i] = empty["neighbor_t"]
            neighbor_user_idx[i] = empty["neighbor_user"]
            continue

        store_dates = aligned_store_dates(
            t,
            lags=lags,
            horizon=horizon,
            train_stride=args.train_stride,
            n_users=n_users,
            period=args.period,
            store_start=store_start,
            store_end=store_end,
            online=args.online,
            align_period=not args.no_align_period,
            max_train_windows=args.max_train_windows,
        )
        store = build_window_batch(
            dataset,
            store_dates,
            lags=lags,
            horizon=horizon,
            distance_space=args.distance_space,
            model=model,
            device=device,
            normalize=not args.no_feature_normalization,
            pool_representation=args.pool_representation,
        )
        distances, indices = search_neighbors(
            query.features,
            store.features,
            k=k,
            metric=args.distance_metric,
            chunk_size=args.search_chunk_size,
        )
        xy_c = store.select_windows(indices)
        x_c = xy_c[:, :, :lags]
        y_c = xy_c[:, :, lags:]
        context = xy_c.to(device)
        pred_context = _predict(
            model,
            x,
            context=context,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

        x_c_flat = rearrange(x_c, "user neighbor lags -> (user neighbor) 1 lags").to(device)
        pred_neighbors = rearrange(
            model(x_c_flat).detach().cpu(),
            "(user neighbor) 1 horizon -> user neighbor horizon",
            user=n_users,
            neighbor=k,
        )
        if ec_values is not None:
            if k > 1:
                pred_neighbors_context = torch.empty_like(pred_neighbors)
                for neighbor_idx in range(k):
                    mask = torch.ones(k, dtype=torch.bool)
                    mask[neighbor_idx] = False
                    neighbor_context_pred = (
                        model(
                            x_c[:, neighbor_idx : neighbor_idx + 1, :].to(device),
                            context=xy_c[:, mask, :].to(device),
                        )
                        .detach()
                        .cpu()
                    )
                    pred_neighbors_context[:, neighbor_idx] = rearrange(
                        neighbor_context_pred,
                        "user 1 horizon -> user horizon",
                    )
            else:
                pred_neighbors_context = pred_neighbors.clone()
            ec_values[i] = y_c - pred_neighbors_context
        neighbor_users, neighbor_dates = store.decode_indices(indices)

        preds_context[i] = pred_context
        e_values[i] = y_c - pred_neighbors
        xc_values[i] = x_c
        yc_values[i] = y_c
        mu_xc[i] = x_c.mean(dim=-1)
        sigma_xc[i] = x_c.std(dim=-1, unbiased=False)
        distance_x_xc[i] = torch.as_tensor(distances, dtype=torch.float32)
        neighbor_t[i] = neighbor_dates
        neighbor_user_idx[i] = neighbor_users

        if args.verbose and (i == 0 or (i + 1) % 25 == 0 or i + 1 == n_eval):
            print(f"{prefix}: processed {i + 1}/{n_eval} dates")

    prediction_payload = {
        f"{prefix}_dates": torch.as_tensor(eval_dates, dtype=torch.long),
        f"{prefix}_datetimes": [str(dataset.datetimes[int(t)]) for t in eval_dates],
        f"{prefix}_preds": preds,
        f"{prefix}_preds_context": preds_context,
        f"{prefix}_E_values": e_values,
        f"{prefix}_X_values": x_values,
        f"{prefix}_Xc_values": xc_values,
        f"{prefix}_Y_values": y_values,
        f"{prefix}_Yc_values": yc_values,
        f"{prefix}_distance_x_xc": distance_x_xc,
        f"{prefix}_query_t": query_t,
        f"{prefix}_query_user_idx": query_user_idx,
        f"{prefix}_neighbor_t": neighbor_t,
        f"{prefix}_neighbor_user_idx": neighbor_user_idx,
        f"{prefix}_retrieval_period": torch.tensor(args.period, dtype=torch.long),
    }
    if ec_values is not None:
        prediction_payload[f"{prefix}_Ec_values"] = ec_values
    if k > 0:
        features_payload = {
            f"{prefix}_mu_x": mu_x,
            f"{prefix}_sigma_x": sigma_x,
            f"{prefix}_mu_xc_mean": mu_xc.mean(dim=-1),
            f"{prefix}_sigma_xc_mean": sigma_xc.mean(dim=-1),
            f"{prefix}_mu_xc_std": mu_xc.std(dim=-1, unbiased=False),
            f"{prefix}_sigma_xc_std": sigma_xc.std(dim=-1, unbiased=False),
            f"{prefix}_loss_pred_pred_c": (preds - preds_context).pow(2).mean(dim=-1),
            f"{prefix}_loss_pred_yc_mean": (
                (
                    rearrange(preds, "date user horizon -> date user 1 horizon")
                    - yc_values
                )
                .pow(2)
                .mean(dim=-1)
                .mean(dim=-1)
            ),
            f"{prefix}_loss_neighbor_residual_mean": e_values.pow(2).mean(dim=-1).mean(dim=-1),
        }
        if ec_values is not None:
            features_payload[f"{prefix}_loss_neighbor_context_residual_mean"] = (
                ec_values.pow(2).mean(dim=-1).mean(dim=-1)
            )
    else:
        features_payload = {
            f"{prefix}_mu_x": mu_x,
            f"{prefix}_sigma_x": sigma_x,
            f"{prefix}_loss_pred_pred_c": (preds - preds_context).pow(2).mean(dim=-1),
        }

    torch.save(prediction_payload, output_dir / f"{prefix}_prediction_payload.pt")
    torch.save(features_payload, output_dir / f"{prefix}_features_payload.pt")
    if args.verbose:
        print(
            f"{prefix}: prediction payload {_payload_size_gb(prediction_payload):.3f} GB, "
            f"feature payload {_payload_size_gb(features_payload):.3f} GB"
        )
    return prediction_payload, features_payload


def plot_neighbor_example(
    *,
    dataset,
    model,
    eval_dates: np.ndarray,
    store_start: int,
    store_end: int,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> None:
    if args.neighbors <= 0 or len(eval_dates) == 0:
        return
    rng = np.random.default_rng(args.seed)
    t = int(eval_dates[int(rng.integers(0, len(eval_dates)))])
    user_idx = int(rng.integers(0, dataset.n_users))
    query = build_window_batch(
        dataset,
        np.asarray([t], dtype=np.int64),
        lags=args.lags,
        horizon=args.horizon,
        distance_space=args.distance_space,
        model=model,
        device=device,
        normalize=not args.no_feature_normalization,
        pool_representation=args.pool_representation,
    )
    store_dates = aligned_store_dates(
        t,
        lags=args.lags,
        horizon=args.horizon,
        train_stride=args.train_stride,
        n_users=dataset.n_users,
        period=args.period,
        store_start=store_start,
        store_end=store_end,
        online=args.online,
        align_period=not args.no_align_period,
        max_train_windows=args.max_train_windows,
    )
    store = build_window_batch(
        dataset,
        store_dates,
        lags=args.lags,
        horizon=args.horizon,
        distance_space=args.distance_space,
        model=model,
        device=device,
        normalize=not args.no_feature_normalization,
        pool_representation=args.pool_representation,
    )
    _, indices = search_neighbors(
        query.features,
        store.features,
        k=args.neighbors,
        metric=args.distance_metric,
        chunk_size=args.search_chunk_size,
    )
    xy_c = store.select_windows(indices)
    series = {"target": query.windows[user_idx, : args.lags].numpy()}
    for neighbor_idx in range(args.neighbors):
        series[f"neighbor_{neighbor_idx + 1}"] = xy_c[user_idx, neighbor_idx, : args.lags].numpy()
    plot_series(
        series,
        output_dir / "plots",
        "neighbors.png",
        f"Neighbors for user {user_idx} at {dataset.datetimes[t]}",
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
    parser.add_argument("--splits", default="0.6,0.2,0.2")
    parser.add_argument("--train-stride", type=int, default=24)
    parser.add_argument("--eval-stride", type=int, default=24)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--neighbors", type=int, default=0)
    parser.add_argument("--distance-space", default="raw", choices=["raw", "fourier", "chronos", "patchtst", "model", "representation"])
    parser.add_argument("--distance-metric", default="euclidean", choices=["euclidean", "cosine", "pearson"])
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--no-align-period", action="store_true")
    parser.add_argument("--no-feature-normalization", action="store_true")
    parser.add_argument("--pool-representation", action="store_true")
    parser.add_argument("--compute-ec", action="store_true", help="Also save neighbor-context residuals Ec")
    parser.add_argument("--search-chunk-size", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs/extraction_neighbors")
    parser.add_argument("--save-name", default="neighbors")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Path]:
    args = parse_args()
    set_seed(args.seed)
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
    device = resolve_device(args.device)
    model = load_pretrained_model(
        args.model,
        lags=args.lags,
        horizon=args.horizon,
        dim=1,
        normalization=args.normalization,
        pretrained_path=args.pretrained_path,
        device=device,
        model_kwargs=load_json_kwargs(args.model_kwargs),
    )
    out = run_dir(args.output_dir, args.save_name)
    t0_end, t1_end, t2_end = split_bounds(dataset.n_dates, args.splits)
    train_eval_dates = period_eval_dates(
        t0_end,
        t1_end,
        n_dates=dataset.n_dates,
        lags=args.lags,
        horizon=args.horizon,
        stride=args.eval_stride,
    )
    eval_eval_dates = period_eval_dates(
        t1_end,
        t2_end,
        n_dates=dataset.n_dates,
        lags=args.lags,
        horizon=args.horizon,
        stride=args.eval_stride,
    )
    if args.verbose:
        print(f"dataset: dates={dataset.n_dates}, users={dataset.n_users}")
        print(f"T0=[0,{t0_end}), T1=[{t0_end},{t1_end}), T2=[{t1_end},{t2_end})")
        print(f"train queries={len(train_eval_dates)}, eval queries={len(eval_eval_dates)}")

    extract_period(
        dataset=dataset,
        model=model,
        prefix="train",
        eval_dates=train_eval_dates,
        store_start=0,
        store_end=t0_end,
        args=args,
        output_dir=out,
        device=device,
    )
    extract_period(
        dataset=dataset,
        model=model,
        prefix="eval",
        eval_dates=eval_eval_dates,
        store_start=0,
        store_end=t1_end,
        args=args,
        output_dir=out,
        device=device,
    )
    plot_neighbor_example(
        dataset=dataset,
        model=model,
        eval_dates=eval_eval_dates,
        store_start=0,
        store_end=t1_end,
        args=args,
        output_dir=out,
        device=device,
    )
    return {
        "run_dir": out,
        "train_prediction": out / "train_prediction_payload.pt",
        "eval_prediction": out / "eval_prediction_payload.pt",
    }


if __name__ == "__main__":
    main()
