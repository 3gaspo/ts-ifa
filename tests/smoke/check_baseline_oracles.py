"""Smoke-check target-aware context oracle baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.experiments.evaluate_baselines import add_true_context_oracles  # noqa: E402


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
    print("baseline oracle checks passed")


if __name__ == "__main__":
    main()
