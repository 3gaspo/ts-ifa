"""Smoke-check target-aware context oracle baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.experiments.evaluate_baselines import (  # noqa: E402
    add_true_context_oracles,
    fit_gate,
    predict_gate,
    ridge_no_intercept,
)


def main() -> None:
    arrays = {
        "pred": np.asarray([[0.0, 2.0], [10.0, 10.0]], dtype=np.float32),
        "pred_c": np.asarray([[1.0, 3.0], [8.0, 12.0]], dtype=np.float32),
        "y": np.asarray([[1.0, 2.0], [9.0, 10.0]], dtype=np.float32),
    }
    predictions: dict[str, np.ndarray] = {}
    add_true_context_oracles(predictions, arrays)
    np.testing.assert_array_equal(
        predictions["oracle_context_scalar"],
        np.asarray([[0.0, 2.0], [10.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        predictions["oracle_context_horizon"],
        np.asarray([[1.0, 2.0], [10.0, 10.0]], dtype=np.float32),
    )
    coefficient = ridge_no_intercept(
        np.ones((2, 1), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        l2=1.0,
    )
    np.testing.assert_allclose(coefficient, np.asarray([0.5]))

    gate_x = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    gate_y = np.asarray([[0], [0], [1], [1]], dtype=np.float32)
    gate = fit_gate(
        gate_x,
        gate_y,
        iterations=20,
        learning_rate=0.1,
        depth=2,
        seed=1,
    )
    probabilities = predict_gate(gate, gate_x)
    assert probabilities.shape == gate_y.shape
    assert probabilities[:2].mean() < probabilities[2:].mean()
    print("baseline oracle checks passed")


if __name__ == "__main__":
    main()
