"""Smoke-check target-aware context oracle baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.data.neighbors import neighbor_to_query_scale  # noqa: E402
from ts_ifa.experiments.evaluate_baselines import (  # noqa: E402
    add_context_gate_predictions,
    add_eval_fitted_baselines,
    add_true_context_oracles,
    fit_gate,
    flatten_payload,
    horizon_gate_feature_names,
    horizon_gate_features,
    predict_gate,
    ridge_no_intercept,
    scalar_gate_features,
)


def has_catboost() -> bool:
    try:
        import catboost  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


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

    ridge_x = np.asarray(
        [[1.0, 3.0], [2.0, 1.0], [4.0, 2.0], [3.0, 5.0]],
        dtype=np.float64,
    )
    ridge_y = np.asarray([2.0, -1.0, 3.0, 4.0], dtype=np.float64)
    coefficient = ridge_no_intercept(ridge_x, ridge_y, l2=0.4)
    rescaled_coefficient = ridge_no_intercept(100.0 * ridge_x, 100.0 * ridge_y, l2=0.4)
    np.testing.assert_allclose(coefficient, rescaled_coefficient)

    query = np.asarray([[3.0, 7.0]], dtype=np.float32)
    neighbor = np.asarray([[[8.0, 12.0]]], dtype=np.float32)
    horizon = np.asarray([[[14.0, 16.0]]], dtype=np.float32)
    residual = np.asarray([[[2.0, 4.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        neighbor_to_query_scale(query, neighbor, horizon),
        np.asarray([[[9.0, 11.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        neighbor_to_query_scale(query, neighbor, residual, residual=True),
        np.asarray([[[2.0, 4.0]]], dtype=np.float32),
    )

    payload = {
        "train_preds": torch.tensor([[[7.0, 7.0]]]),
        "train_preds_context": torch.tensor([[[8.0, 8.0]]]),
        "train_X_values": torch.tensor([[[3.0, 7.0]]]),
        "train_Xc_values": torch.tensor([[[[8.0, 12.0]]]]),
        "train_Y_values": torch.tensor([[[9.0, 11.0]]]),
        "train_Yc_values": torch.tensor([[[[14.0, 16.0]]]]),
        "train_E_values": torch.tensor([[[[2.0, 4.0]]]]),
        "train_distance_x_xc": torch.tensor([[[0.5]]]),
        "train_query_t": torch.tensor([[42]]),
        "train_query_user_idx": torch.tensor([[3]]),
        "train_neighbor_t": torch.tensor([[[30]]]),
        "train_neighbor_user_idx": torch.tensor([[[3]]]),
    }
    flattened = flatten_payload(payload, "train")
    np.testing.assert_allclose(flattened["y_c"], np.asarray([[[9.0, 11.0]]]))
    np.testing.assert_allclose(flattened["e"], np.asarray([[[2.0, 4.0]]]))
    np.testing.assert_allclose(flattened["pred_neighbors"], np.asarray([[[7.0, 7.0]]]))
    np.testing.assert_allclose(flattened["neighbor_lookback_mean"], np.asarray([10.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_mean_std"], np.asarray([0.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_std"], np.asarray([2.0]))
    np.testing.assert_allclose(flattened["neighbor_lookback_std_std"], np.asarray([0.0]))
    np.testing.assert_allclose(flattened["same_user_ratio"], np.asarray([1.0]))
    np.testing.assert_allclose(flattened["neighbor_age_mean"], np.asarray([12.0]))
    scalar_features = scalar_gate_features(flattened)
    horizon_features = horizon_gate_features(flattened)
    np.testing.assert_allclose(scalar_features[0, :4], np.asarray([1.0, 0.0, 3.0, 3.0]))
    np.testing.assert_allclose(horizon_features[0, :4], np.asarray([1.0, 1.0, 3.0, 3.0]))
    assert scalar_features.shape[1] == 2 + 13
    assert horizon_features.shape[1] == 2 + 13
    assert horizon_features.shape[1] == len(horizon_gate_feature_names(2))

    gate_predictions, gate_artifacts, gate_diagnostics = add_context_gate_predictions(
        {split: {} for split in ("train", "oracle", "eval")},
        flattened,
        {split: flattened for split in ("train", "oracle", "eval")},
        iterations=1,
        learning_rate=0.1,
        depth=1,
        seed=1,
    )
    assert set(gate_predictions["eval"]) == {
        "gated_context_classifier_scalar",
        "gated_context_classifier_horizon",
        "gated_context_regressor_scalar",
        "gated_context_regressor_horizon",
        "oracle_context_scalar",
        "oracle_context_horizon",
    }
    assert set(gate_artifacts["models"]) == {"classifier", "regressor"}
    assert set(gate_diagnostics["eval"]) == {
        "classifier_scalar_score",
        "classifier_scalar_target",
        "classifier_horizon_score",
        "classifier_horizon_target",
        "regressor_scalar_score",
        "regressor_scalar_target",
        "regressor_horizon_score",
        "regressor_horizon_target",
    }

    baseline_predictions = {"eval": {}}
    eval_fit_artifacts = add_eval_fitted_baselines(
        baseline_predictions,
        flattened,
        l2=1e-3,
    )
    assert set(baseline_predictions["eval"]) == {
        "mix_0_weighted_eval_fit",
        "mix_1_learned_eval_fit",
        "mix_2_full_horizon_eval_fit",
    }
    assert set(eval_fit_artifacts) == {"mix_0_lambda", "mix_1_coef", "mix_2_coef"}

    gate_x = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    gate_y = np.asarray([[-4.0], [-1.0], [1.0], [4.0]], dtype=np.float32)
    if has_catboost():
        gate = fit_gate(
            gate_x,
            gate_y,
            iterations=50,
            learning_rate=0.1,
            depth=2,
            seed=1,
        )
        differences = predict_gate(gate, gate_x)
        assert differences.shape == gate_y.shape
        assert differences[:2].mean() < 0.0 < differences[2:].mean()

        classifier = fit_gate(
            gate_x,
            gate_y,
            iterations=50,
            learning_rate=0.1,
            depth=2,
            seed=1,
            objective="classifier",
        )
        classifier_scores = predict_gate(classifier, gate_x)
        assert classifier_scores.shape == gate_y.shape
        assert classifier_scores[:2].mean() < 0.0 < classifier_scores[2:].mean()

    constant_classifier = fit_gate(
        gate_x,
        np.ones_like(gate_y),
        iterations=1,
        learning_rate=0.1,
        depth=1,
        seed=1,
        objective="classifier",
    )
    np.testing.assert_array_equal(
        predict_gate(constant_classifier, gate_x),
        np.full_like(gate_y, 0.5, dtype=np.float64),
    )
    print("baseline oracle checks passed")


if __name__ == "__main__":
    main()
