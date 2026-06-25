"""Train TS-IFA from extraction payloads."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from ..data.load_dataset_model import resolve_device, set_seed
from ..data.scaling import neighbor_to_query_scale
from ..models.models import parameter_counts
from ..models.ts_ifa import TSIFAConfig, TimeSeriesInformedForecastingAdapter
from .runtime import log_experiment_separator, setup_logging


LOGGER = logging.getLogger(__name__)


def torch_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(Path(path), map_location="cpu")


def flatten_time_user(value: torch.Tensor) -> torch.Tensor:
    return rearrange(value, "date user ... -> (date user) ...").float()


class PredictionPayloadDataset(Dataset):
    """Flatten ``(date, user, ...)`` payload tensors into examples."""

    required = (
        "preds",
        "preds_context",
        "E_values",
        "X_values",
        "Xc_values",
        "Y_values",
        "Yc_values",
    )

    def __init__(
        self,
        payload: dict[str, Any],
        *,
        prefix: str,
        max_samples: int | None = None,
    ):
        self.prefix = prefix
        missing = [f"{prefix}_{name}" for name in self.required if f"{prefix}_{name}" not in payload]
        if missing:
            raise KeyError(f"payload is missing required keys: {missing}")

        x = payload[f"{prefix}_X_values"].float()
        x_c_raw = payload[f"{prefix}_Xc_values"].float()
        x_c = neighbor_to_query_scale(x, x_c_raw, x_c_raw)
        if x_c.shape[2] <= 0:
            raise ValueError("TS-IFA training requires payloads extracted with neighbors > 0")

        y_c_raw = payload[f"{prefix}_Yc_values"].float()
        residual_c_raw = payload[f"{prefix}_E_values"].float()
        pred_neighbors_raw = y_c_raw - residual_c_raw
        y_c = neighbor_to_query_scale(x, x_c_raw, y_c_raw)
        residual_c = neighbor_to_query_scale(x, x_c_raw, residual_c_raw, residual=True)
        pred_neighbors = neighbor_to_query_scale(x, x_c_raw, pred_neighbors_raw)

        self.tensors = {
            "x": flatten_time_user(x),
            "x_c": flatten_time_user(x_c),
            "y": flatten_time_user(payload[f"{prefix}_Y_values"]),
            "y_c": flatten_time_user(y_c),
            "pred": flatten_time_user(payload[f"{prefix}_preds"]),
            "pred_context": flatten_time_user(payload[f"{prefix}_preds_context"]),
            "pred_neighbors": flatten_time_user(pred_neighbors),
            "residual_c": flatten_time_user(residual_c),
        }
        n_examples = self.tensors["x"].shape[0]
        if max_samples is not None:
            n_examples = min(n_examples, int(max_samples))
            self.tensors = {key: value[:n_examples] for key, value in self.tensors.items()}

        self.lags = int(self.tensors["x"].shape[-1])
        self.horizon = int(self.tensors["y"].shape[-1])
        self.neighbors = int(self.tensors["x_c"].shape[1])

    def __len__(self) -> int:
        return int(self.tensors["x"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}

    @classmethod
    def concatenate(
        cls,
        datasets: list["PredictionPayloadDataset"],
        *,
        max_samples: int | None = None,
    ) -> "PredictionPayloadDataset":
        if not datasets:
            raise ValueError("at least one training payload is required")
        reference = datasets[0]
        for dataset in datasets[1:]:
            if (dataset.lags, dataset.horizon, dataset.neighbors) != (
                reference.lags,
                reference.horizon,
                reference.neighbors,
            ):
                raise ValueError("train and oracle payload shapes are incompatible")
        combined = cls.__new__(cls)
        combined.prefix = "+".join(dataset.prefix for dataset in datasets)
        combined.tensors = {
            key: torch.cat([dataset.tensors[key] for dataset in datasets], dim=0)
            for key in reference.tensors
        }
        if max_samples is not None:
            combined.tensors = {
                key: value[: int(max_samples)]
                for key, value in combined.tensors.items()
            }
        combined.lags = reference.lags
        combined.horizon = reference.horizon
        combined.neighbors = reference.neighbors
        return combined


def query_stats(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return mean, std


def prepare_batch(
    raw: dict[str, torch.Tensor],
    *,
    normalization: str,
    eps: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    q_mean, q_std = query_stats(raw["x"], eps)
    neighbor_mean = q_mean.unsqueeze(-2)
    neighbor_std = q_std.unsqueeze(-2)
    if normalization == "instance":
        batch = {
            "x": (raw["x"] - q_mean) / q_std,
            "y": (raw["y"] - q_mean) / q_std,
            "pred": (raw["pred"] - q_mean) / q_std,
            "pred_context": (raw["pred_context"] - q_mean) / q_std,
            "x_c": (raw["x_c"] - neighbor_mean) / neighbor_std,
            "y_c": (raw["y_c"] - neighbor_mean) / neighbor_std,
            "pred_neighbors": (raw["pred_neighbors"] - neighbor_mean) / neighbor_std,
        }
        batch["residual_c"] = raw["residual_c"] / neighbor_std
        loss_scale = torch.ones_like(q_std)
    elif normalization == "none":
        batch = dict(raw)
        loss_scale = q_std
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return batch, {"mean": q_mean, "std": q_std, "loss_scale": loss_scale}


def denormalize(
    value: torch.Tensor,
    state: dict[str, torch.Tensor],
    normalization: str,
) -> torch.Tensor:
    if normalization == "instance":
        return value * state["std"] + state["mean"]
    return value


def nmse_mean(pred: torch.Tensor, target: torch.Tensor, lookback: torch.Tensor, eps: float) -> torch.Tensor:
    scale = lookback.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return ((pred - target) / scale).pow(2).mean(dim=-1)


def loss_components(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    beta: float,
    gamma: float,
) -> dict[str, torch.Tensor]:
    scale = state["loss_scale"]
    prediction = outputs["prediction"]
    pred_loss = ((prediction - batch["y"]) / scale).pow(2).mean()
    reg_loss = ((prediction - batch["pred"]) / scale).pow(2).mean()
    residual_target = batch["y"] - batch["pred"]
    residual_loss = ((outputs["residual_delta"] - residual_target) / scale).pow(2).mean()
    total = pred_loss + float(beta) * reg_loss + float(gamma) * residual_loss
    return {
        "loss": total,
        "prediction": pred_loss,
        "regularization": reg_loss,
        "residual": residual_loss,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(
    model: TimeSeriesInformedForecastingAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    normalization: str,
    eps: float,
    prediction_output: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.eval()
    sums = {
        "adapted_nmse": 0.0,
        "vanilla_nmse": 0.0,
        "context_nmse": 0.0,
        "residual_branch_nmse": 0.0,
        "memory_branch_nmse": 0.0,
        "adapted_mse": 0.0,
        "adapted_mae": 0.0,
    }
    prediction_batches: dict[str, list[torch.Tensor]] = {
        "ts_ifa": [],
        "ts_ifa_residual_branch": [],
        "ts_ifa_memory_branch": [],
    }
    count = 0
    with torch.inference_mode():
        for raw_cpu in loader:
            raw = move_batch(raw_cpu, device)
            batch, state = prepare_batch(raw, normalization=normalization, eps=eps)
            outputs = model(batch)
            adapted = denormalize(outputs["prediction"], state, normalization)
            residual = denormalize(outputs["residual_prediction"], state, normalization)
            memory = denormalize(outputs["memory_prediction"], state, normalization)
            y = raw["y"]
            n = y.shape[0]

            if prediction_output is not None:
                prediction_batches["ts_ifa"].append(adapted.detach().cpu())
                prediction_batches["ts_ifa_residual_branch"].append(residual.detach().cpu())
                prediction_batches["ts_ifa_memory_branch"].append(memory.detach().cpu())

            sums["adapted_nmse"] += nmse_mean(adapted, y, raw["x"], eps).sum().item()
            sums["vanilla_nmse"] += nmse_mean(raw["pred"], y, raw["x"], eps).sum().item()
            sums["context_nmse"] += nmse_mean(raw["pred_context"], y, raw["x"], eps).sum().item()
            sums["residual_branch_nmse"] += nmse_mean(residual, y, raw["x"], eps).sum().item()
            sums["memory_branch_nmse"] += nmse_mean(memory, y, raw["x"], eps).sum().item()
            sums["adapted_mse"] += F.mse_loss(adapted, y, reduction="sum").item() / y.shape[-1]
            sums["adapted_mae"] += F.l1_loss(adapted, y, reduction="sum").item() / y.shape[-1]
            count += n
    if prediction_output is not None:
        prediction_output.update(
            {
                name: torch.cat(batches, dim=0) if batches else torch.empty(0)
                for name, batches in prediction_batches.items()
            }
        )
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_loss_curve(history: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_nmse = [row.get("train_nmse", row.get("train_prediction")) for row in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_nmse, label="train nMSE", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("nMSE")
    ax.set_title("TS-IFA training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=None, help="Directory with train/oracle/eval prediction payloads")
    parser.add_argument("--train-payload", default=None)
    parser.add_argument("--oracle-payload", default=None)
    parser.add_argument("--eval-payload", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-2, help="Penalty toward vanilla prediction")
    parser.add_argument("--gamma", type=float, default=0.0, help="Residual branch supervision weight")
    parser.add_argument("--normalization", default="instance", choices=["instance", "none"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--residual-heads", type=int, default=4)
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--mixture-heads", type=int, default=4)
    parser.add_argument("--residual-attn-dim", type=int, default=32)
    parser.add_argument("--memory-attn-dim", type=int, default=32)
    parser.add_argument("--mixture-attn-dim", type=int, default=32)
    parser.add_argument("--residual-hidden", type=int, default=128)
    parser.add_argument("--memory-hidden", type=int, default=128)
    parser.add_argument("--mixture-hidden", type=int, default=128)
    parser.add_argument("--mixture-key-dim", type=int, default=64)
    parser.add_argument("--mixture-gate-init", type=float, default=-6.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    base = Path(args.input_dir).expanduser() if args.input_dir else None
    train_payload = Path(args.train_payload).expanduser() if args.train_payload else None
    oracle_payload = Path(args.oracle_payload).expanduser() if args.oracle_payload else None
    eval_payload = Path(args.eval_payload).expanduser() if args.eval_payload else None
    if train_payload is None:
        if base is None:
            raise ValueError("pass --input-dir or --train-payload")
        train_payload = base / "train_prediction_payload.pt"
    if oracle_payload is None:
        if base is None:
            raise ValueError("pass --input-dir or --oracle-payload")
        oracle_payload = base / "oracle_prediction_payload.pt"
    if eval_payload is None and base is not None:
        candidate = base / "eval_prediction_payload.pt"
        eval_payload = candidate if candidate.exists() else None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    elif base is not None:
        output_dir = base / "ts_ifa"
    else:
        output_dir = train_payload.parent / "ts_ifa"
    output_dir.mkdir(parents=True, exist_ok=True)
    return train_payload, oracle_payload, eval_payload, output_dir


def main() -> dict[str, Path]:
    args = parse_args()
    setup_logging()
    log_experiment_separator(LOGGER)
    experiment_start = perf_counter()
    set_seed(args.seed)
    train_payload_path, oracle_payload_path, eval_payload_path, output_dir = resolve_paths(args)
    LOGGER.info(
        "experiment start kind=ts_ifa_train input=%s epochs=%s batch_size=%s",
        train_payload_path.parent,
        args.epochs,
        args.batch_size,
    )
    LOGGER.info("payload load start")
    train_dataset = PredictionPayloadDataset.concatenate(
        [
            PredictionPayloadDataset(torch_load(train_payload_path), prefix="train"),
            PredictionPayloadDataset(torch_load(oracle_payload_path), prefix="oracle"),
        ],
        max_samples=args.max_train_samples,
    )
    eval_dataset = None
    if eval_payload_path is not None and eval_payload_path.exists():
        eval_dataset = PredictionPayloadDataset(
            torch_load(eval_payload_path),
            prefix="eval",
            max_samples=args.max_eval_samples,
        )
    LOGGER.info(
        "payload load done train_samples=%s eval_samples=%s",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )

    eval_batch_size = args.eval_batch_size or args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    config = TSIFAConfig(
        lags=train_dataset.lags,
        horizon=train_dataset.horizon,
        neighbors=train_dataset.neighbors,
        residual_heads=args.residual_heads,
        memory_heads=args.memory_heads,
        mixture_heads=args.mixture_heads,
        residual_attn_dim=args.residual_attn_dim,
        memory_attn_dim=args.memory_attn_dim,
        mixture_attn_dim=args.mixture_attn_dim,
        residual_hidden=args.residual_hidden,
        memory_hidden=args.memory_hidden,
        mixture_hidden=args.mixture_hidden,
        mixture_key_dim=args.mixture_key_dim,
        mixture_gate_init=args.mixture_gate_init,
        dropout=args.dropout,
    )
    device = resolve_device(args.device)
    model = TimeSeriesInformedForecastingAdapter(config).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    LOGGER.info(
        "model ready name=TS-IFA device=%s parameters_total=%s parameters_trainable=%s",
        device,
        f"{total_parameters:,}",
        f"{trainable_parameters:,}",
    )

    history: list[dict[str, Any]] = []
    eps = 1e-8
    start_time = perf_counter()
    log_every = max(1, args.epochs // 20)
    LOGGER.info("training start")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "prediction": 0.0, "regularization": 0.0, "residual": 0.0}
        seen = 0
        for raw_cpu in train_loader:
            raw = move_batch(raw_cpu, device)
            batch, state = prepare_batch(raw, normalization=args.normalization, eps=eps)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            losses = loss_components(
                outputs,
                batch,
                state,
                beta=args.beta,
                gamma=args.gamma,
            )
            losses["loss"].backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = raw["x"].shape[0]
            seen += batch_size
            for key in totals:
                totals[key] += losses[key].detach().item() * batch_size

        row: dict[str, Any] = {
            "epoch": epoch,
            **{f"train_{key}": value / max(seen, 1) for key, value in totals.items()},
        }
        row["train_nmse"] = row["train_prediction"]
        history.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % log_every == 0:
            LOGGER.info(
                "training progress epoch=%s/%s train_nmse=%.6f",
                epoch,
                args.epochs,
                row["train_nmse"],
            )
    LOGGER.info("training done seconds=%.2f", perf_counter() - start_time)

    final_eval = {}
    eval_predictions: dict[str, torch.Tensor] = {}
    if eval_loader is not None:
        LOGGER.info("evaluation start")
        final_eval = evaluate(
            model,
            eval_loader,
            device=device,
            normalization=args.normalization,
            eps=eps,
            prediction_output=eval_predictions,
        )
        LOGGER.info("evaluation done adapted_nmse=%.6f", final_eval["adapted_nmse"])

    checkpoint_path = output_dir / "ts_ifa.pt"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "eval_metrics.json"
    predictions_path = output_dir / "eval_predictions.pt"
    config_path = output_dir / "config.json"
    plot_path = output_dir / "training_nmse.pdf"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "model_name": "TS-IFA",
            "parameter_counts": {
                "total": total_parameters,
                "trainable": trainable_parameters,
            },
            "normalization": args.normalization,
            "train_payloads": [str(train_payload_path), str(oracle_payload_path)],
            "eval_payload": str(eval_payload_path) if eval_payload_path else None,
            "epochs": args.epochs,
        },
        checkpoint_path,
    )
    save_json(history_path, {"history": history})
    save_json(metrics_path, final_eval)
    torch.save(
        {
            "format_version": 1,
            "split": "eval",
            "predictions": eval_predictions,
        },
        predictions_path,
    )
    plot_loss_curve(history, plot_path)
    save_json(
        config_path,
        {
            "name": "TS-IFA",
            "model": asdict(config),
            "parameters": {
                "total": total_parameters,
                "trainable": trainable_parameters,
            },
            "training": {
                "optimizer": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "loss": "nMSE",
                "normalization": args.normalization,
                "beta": args.beta,
                "gamma": args.gamma,
                "seconds": perf_counter() - start_time,
            },
        },
    )
    LOGGER.info("outputs saved dir=%s", output_dir)
    LOGGER.info("experiment done seconds=%.2f", perf_counter() - experiment_start)
    log_experiment_separator(LOGGER)
    return {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "metrics": metrics_path,
        "predictions": predictions_path,
        "config": config_path,
        "plot": plot_path,
    }


if __name__ == "__main__":
    main()
