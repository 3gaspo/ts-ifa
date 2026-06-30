"""Smoke-check retrieval-sweep LaTeX table generation."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.sweep_results_table import generate_sweep_results_tables


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for dataset, offset in [("electricity", 0.0), ("solar", 0.2)]:
            setting = root / dataset / "168_24"
            _write(
                setting / "direct" / "chronos" / "univariate_summary.json",
                {"eval": {"nmse": {"mean": 1.0 + offset}, "mse": {"mean": 0.01}}},
            )
            for run, ridge, bayes, ts_ifa in [
                ("chronos_raw_euclidean_1_online", 0.92 + offset, 0.88 + offset, 0.84 + offset),
                ("chronos_instance_euclidean_3_online", 0.72 + offset, 0.62 + offset, 0.58 + offset),
            ]:
                _write(
                    setting / run / "baselines" / "baseline_metrics.json",
                    [
                        {"split": "eval", "baseline": "context_forecast", "nmse": 0.95 + offset, "mse": 0.009},
                        {"split": "eval", "baseline": "horizon_ridge_shared", "nmse": ridge, "mse": 0.007},
                        {
                            "split": "eval",
                            "baseline": "residual_ridge_horizon_eval_fit",
                            "nmse": 0.1 + offset,
                            "mse": 0.001,
                        },
                    ],
                )
                _write(
                    setting / run / "gates" / "gate_metrics.json",
                    [
                        {"split": "eval", "baseline": "bayes_context_scalar", "nmse": bayes, "mse": 0.006},
                        {
                            "split": "eval",
                            "baseline": "catboost_context_classifier_scalar",
                            "nmse": bayes + 0.05,
                            "mse": 0.0065,
                        },
                        {"split": "eval", "baseline": "oracle_context_horizon", "nmse": 0.4 + offset, "mse": 0.004},
                    ],
                )
                _write(
                    setting / run / "ts_ifa" / "eval_metrics.json",
                    {"adapted_nmse": ts_ifa, "adapted_mse": 0.005, "vanilla_nmse": 1.0 + offset},
                )

        outputs = generate_sweep_results_tables(
            root,
            root / "tables",
            datasets=["electricity", "solar"],
            settings=["168_24"],
            spaces=["raw", "instance"],
            neighbors=[1, 3],
        )
        names = {output.name for output in outputs}
        assert names == {"baselines_results.tex", "gates_results.tex", "ts_ifa_results.tex"}

        baselines = (root / "tables" / "baselines_results.tex").read_text(encoding="utf-8")
        assert baselines.count(r"\begin{table}") == 2
        assert r"raw\_L2\_1/Y-ridge" in baselines
        assert r"IN\_L2\_3/Y-ridge" in baselines
        assert r"R-ridge-h-fit-T3" in baselines
        assert r"\textit{Improvement}" not in baselines
        assert "Overall improvement" not in baselines
        assert r"raw\_L2\_1" in baselines
        assert r"\textbf{0.82}" in baselines

        gates = (root / "tables" / "gates_results.tex").read_text(encoding="utf-8")
        assert r"bayes-s & 0.98 & -- & -- & \textbf{0.72}" in gates
        assert r"cb-cls-s" in gates

        ts_ifa = (root / "tables" / "ts_ifa_results.tex").read_text(encoding="utf-8")
        assert r"TS-IFA & 0.94 & -- & -- & \textbf{0.68}" in ts_ifa
        assert ts_ifa.count(r"\begin{table}") == 2

    print("sweep results table checks passed")


if __name__ == "__main__":
    main()
