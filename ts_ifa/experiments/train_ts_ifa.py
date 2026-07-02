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

from ..data.load_dataset import set_seed
from ..data.neighbors import neighbor_to_query_scale
from ..models.models import parameter_counts, resolve_device
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
        self.n_dates = int(x.shape[0])
        self.n_users = int(x.shape[1])
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


class RandomPredictionPayloadDataset(Dataset):
    """Draw random examples from T1 so one epoch is one optimizer step."""

    def __init__(self, source: PredictionPayloadDataset, *, virtual_size: int):
        if len(source) == 0:
            raise ValueError("cannot sample from an empty training payload")
        if int(virtual_size) <= 0:
            raise ValueError("virtual_size must be positive")
        self.source = source
        self.virtual_size = int(virtual_size)
        self.lags = source.lags
        self.horizon = source.horizon
        self.neighbors = source.neighbors

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        if len(self.source) == self.source.n_dates * self.source.n_users:
            date_index = int(torch.randint(self.source.n_dates, ()).item())
            user_index = int(torch.randint(self.source.n_users, ()).item())
            source_index = date_index * self.source.n_users + user_index
        else:
            source_index = int(torch.randint(len(self.source), ()).item())
        return self.source[source_index]


def log_scale_diagnostics(name: str, dataset: PredictionPayloadDataset | None) -> None:
    if dataset is None:
        return
    scale = dataset.tensors["x"].std(dim=-1, unbiased=False).float()
    if scale.numel() == 0:
        LOGGER.info("payload scale split=%s samples=0", name)
        return
    quantiles = torch.quantile(
        scale,
        torch.tensor([0.0, 0.001, 0.01, 0.05, 0.1, 0.5], dtype=torch.float32),
    )
    LOGGER.info(
        "payload scale split=%s samples=%s std_min=%.6g std_q001=%.6g std_q01=%.6g std_q05=%.6g std_q10=%.6g std_median=%.6g below_1e-8=%s below_1e-6=%s below_1e-3=%s",
        name,
        len(dataset),
        float(quantiles[0]),
        float(quantiles[1]),
        float(quantiles[2]),
        float(quantiles[3]),
        float(quantiles[4]),
        float(quantiles[5]),
        int((scale < 1e-8).sum().item()),
        int((scale < 1e-6).sum().item()),
        int((scale < 1e-3).sum().item()),
    )


def ensure_compatible(
    reference: PredictionPayloadDataset,
    candidate: PredictionPayloadDataset | None,
    *,
    name: str,
) -> None:
    if candidate is None:
        return
    if (candidate.lags, candidate.horizon, candidate.neighbors) != (
        reference.lags,
        reference.horizon,
        reference.neighbors,
    ):
        raise ValueError(f"{name} payload shape is incompatible with train payload")


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
    memory_loss = ((outputs["memory_delta"] - residual_target) / scale).pow(2).mean()
    branch_loss = residual_loss + memory_loss
    total = pred_loss + float(beta) * reg_loss + float(gamma) * branch_loss
    return {
        "loss": total,
        "prediction": pred_loss,
        "regularization": reg_loss,
        "residual": residual_loss,
        "memory": memory_loss,
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


def resolve_step_frequency(value: int | None, *, default: int, name: str) -> int:
    frequency = default if value is None else int(value)
    if frequency <= 0:
        raise ValueError(f"{name} must be positive")
    return frequency


def plot_loss_curve(
    history: list[dict[str, Any]],
    output_path: Path,
    *,
    train_steps: list[dict[str, Any]] | None = None,
    plot_step_train_loss: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return

    steps = [row.get("step", row["epoch"]) for row in history]
    train_batch_nmse = [
        row.get("train_batch_nmse", row.get("train_nmse", row.get("train_prediction")))
        for row in history
    ]
    valid_nmse = [row.get("valid_adapted_nmse") for row in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    if plot_step_train_loss and train_steps:
        step_x = [row["step"] for row in train_steps]
        step_y = [row.get("train_nmse", row.get("train_prediction")) for row in train_steps]
        ax.plot(step_x, step_y, label="train step nMSE", linewidth=1, alpha=0.35)
    ax.plot(steps, train_batch_nmse, marker="o", label="train interval nMSE", linewidth=2)
    if any(value is not None for value in valid_nmse):
        ax.plot(steps, valid_nmse, marker="o", label="T2 validation nMSE", linewidth=2)
    ax.set_xlabel("Step")
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
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument(
        "--valid-eval-freq",
        type=int,
        default=None,
        help="Run full T2 validation every N optimizer steps. Defaults to one epoch.",
    )
    parser.add_argument(
        "--logging-eval-freq",
        type=int,
        default=None,
        help="Print the latest train interval and T2 validation metrics every N optimizer steps. Defaults to one epoch.",
    )
    parser.add_argument(
        "--plot-step-train-loss",
        action="store_true",
        help="Include noisy per-step train nMSE in training_nmse.pdf. Disabled by default.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-2, help="Penalty toward vanilla prediction")
    parser.add_argument("--gamma", type=float, default=1e-2, help="Branch delta supervision weight")
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
    parser.add_argument("--mixture-key-dim", type=int, default=64, help="Deprecated; kept for old launch configs")
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
    train_dataset = PredictionPayloadDataset(
        torch_load(train_payload_path),
        prefix="train",
        max_samples=args.max_train_samples,
    )
    valid_dataset = PredictionPayloadDataset(torch_load(oracle_payload_path), prefix="oracle")
    eval_dataset = None
    if eval_payload_path is not None and eval_payload_path.exists():
        eval_dataset = PredictionPayloadDataset(
            torch_load(eval_payload_path),
            prefix="eval",
            max_samples=args.max_eval_samples,
        )
    ensure_compatible(train_dataset, valid_dataset, name="validation")
    ensure_compatible(train_dataset, eval_dataset, name="evaluation")
    LOGGER.info(
        "payload load done train_samples=%s valid_samples=%s eval_samples=%s",
        len(train_dataset),
        len(valid_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
    )
    log_scale_diagnostics("train", train_dataset)
    log_scale_diagnostics("valid", valid_dataset)
    log_scale_diagnostics("eval", eval_dataset)

    eval_batch_size = args.eval_batch_size or args.batch_size
    random_train_dataset = RandomPredictionPayloadDataset(
        train_dataset,
        virtual_size=args.batch_size,
    )
    train_loader = DataLoader(
        random_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
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
    train_steps: list[dict[str, Any]] = []
    eps = 1e-8
    start_time = perf_counter()
    steps_per_epoch = max(1, len(train_loader))
    total_steps = int(args.epochs) * steps_per_epoch
    valid_eval_freq = resolve_step_frequency(
        args.valid_eval_freq,
        default=steps_per_epoch,
        name="valid_eval_freq",
    )
    logging_eval_freq = resolve_step_frequency(
        args.logging_eval_freq,
        default=steps_per_epoch,
        name="logging_eval_freq",
    )
    if logging_eval_freq % valid_eval_freq != 0:
        raise ValueError("logging_eval_freq must be a multiple of valid_eval_freq")
    LOGGER.info(
        "training start epochs=%s steps_per_epoch=%s total_steps=%s valid_eval_freq=%s logging_eval_freq=%s",
        args.epochs,
        steps_per_epoch,
        total_steps,
        valid_eval_freq,
        logging_eval_freq,
    )
    recent_totals = {
        "loss": 0.0,
        "prediction": 0.0,
        "regularization": 0.0,
        "residual": 0.0,
        "memory": 0.0,
    }
    recent_seen = 0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for raw_cpu in train_loader:
            step += 1
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
            step_row = {
                "epoch": epoch,
                "step": step,
                **{f"train_{key}": losses[key].detach().item() for key in recent_totals},
            }
            step_row["train_nmse"] = step_row["train_prediction"]
            train_steps.append(step_row)
            recent_seen += batch_size
            for key in recent_totals:
                recent_totals[key] += losses[key].detach().item() * batch_size

            should_valid_eval = step % valid_eval_freq == 0
            should_logging_eval = step % logging_eval_freq == 0
            if not (should_valid_eval or should_logging_eval):
                continue

            row: dict[str, Any] = {
                "epoch": epoch,
                "step": step,
                **{
                    f"train_batch_{key}": value / max(recent_seen, 1)
                    for key, value in recent_totals.items()
                },
            }
            row["train_nmse"] = row["train_batch_prediction"]
            row["train_batch_nmse"] = row["train_batch_prediction"]

            valid_metrics = evaluate(
                model,
                valid_loader,
                device=device,
                normalization=args.normalization,
                eps=eps,
            )
            row.update({f"valid_{key}": value for key, value in valid_metrics.items()})

            if should_logging_eval:
                LOGGER.info(
                    "training progress step=%s/%s epoch=%s/%s train_interval_nmse=%.6f valid_nmse=%.6f",
                    step,
                    total_steps,
                    epoch,
                    args.epochs,
                    row["train_batch_nmse"],
                    row["valid_adapted_nmse"],
                )

            history.append(row)
            recent_totals = {key: 0.0 for key in recent_totals}
            recent_seen = 0
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
            "train_payload": str(train_payload_path),
            "validation_payload": str(oracle_payload_path),
            "eval_payload": str(eval_payload_path) if eval_payload_path else None,
            "epochs": args.epochs,
            "steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "valid_eval_freq": valid_eval_freq,
            "logging_eval_freq": logging_eval_freq,
        },
        checkpoint_path,
    )
    save_json(history_path, {"history": history, "train_steps": train_steps})
    save_json(metrics_path, final_eval)
    torch.save(
        {
            "format_version": 1,
            "split": "eval",
            "predictions": eval_predictions,
        },
        predictions_path,
    )
    plot_loss_curve(
        history,
        plot_path,
        train_steps=train_steps,
        plot_step_train_loss=args.plot_step_train_loss,
    )
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
                "loss": "normalized_mse",
                "normalization": args.normalization,
                "beta": args.beta,
                "gamma": args.gamma,
                "gamma_components": ["residual_delta", "memory_delta"],
                "train_split": "T1",
                "validation_split": "T2",
                "final_eval_split": "T3",
                "random_epoch_size": args.batch_size,
                "steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "valid_eval_freq": valid_eval_freq,
                "logging_eval_freq": logging_eval_freq,
                "plot_step_train_loss": bool(args.plot_step_train_loss),
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
