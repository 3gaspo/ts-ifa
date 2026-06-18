"""Small local/cluster smoke checks for data and model loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_dataset_model import load_csv_dataset, load_pretrained_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=Path(__file__).with_name("tiny_timeseries.csv"))
    parser.add_argument("--check-patchtst", action="store_true")
    parser.add_argument("--chronos-weights", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_csv_dataset(
        args.csv,
        date_col="date",
        target_cols="user_a,user_b",
        future_covariate_cols="temp",
    )
    assert dataset.n_dates == 12
    assert dataset.n_users == 2

    lags = 4
    horizon = 2
    x, y = dataset.window_tensor(0, lags, horizon)
    past_cov, future_cov = dataset.covariate_tensors(0, lags, horizon)
    assert x.shape == (2, 1, lags)
    assert y.shape == (2, 1, horizon)
    assert past_cov is None
    assert future_cov is not None and future_cov.shape == (1, 1, horizon)

    persistence = load_pretrained_model(
        "persistence",
        lags=lags,
        horizon=horizon,
        device="cpu",
    )
    pred = persistence(x, future_covariates=future_cov)
    assert pred.shape == y.shape

    if args.check_patchtst:
        patchtst = load_pretrained_model(
            "patchtst",
            lags=lags,
            horizon=horizon,
            device="cpu",
            model_kwargs={"patch_len": 2, "stride": 1, "n_heads": 4},
        )
        assert patchtst(x).shape == y.shape
        assert patchtst.representation(x).shape[0] == x.shape[0]

    if args.chronos_weights:
        chronos = load_pretrained_model(
            "chronos",
            lags=lags,
            horizon=horizon,
            device="cpu",
            model_kwargs={
                "weights_path": args.chronos_weights,
                "device_map": "cpu",
                "context_mode": "past_only",
            },
        )
        assert chronos(x).shape == y.shape

    print("smoke checks passed")


if __name__ == "__main__":
    main()
