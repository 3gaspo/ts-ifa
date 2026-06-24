"""Smoke-check TS-IFA result discovery and LaTeX rendering."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_ifa.results_table import discover_results, generate_results_table


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        setting = root / "electricity" / "168_24"
        _write(setting / "direct" / "chronos" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 0.0012}, "nmse": {"mean": 0.4}}})
        run = "chronos_instance_euclidean_3_online"
        _write(setting / run / "baselines" / "baseline_metrics.json",
               [{"split": "eval", "baseline": "vanilla", "mse": 0.0012, "mae": 0.03, "nmse": 0.4},
                {"split": "eval", "baseline": "mix_1_learned", "mse": 0.0009, "mae": 0.02, "nmse": 0.3},
                {"split": "eval", "baseline": "mix_1_learned_eval_fit", "mse": 0.0005,
                 "mae": 0.015, "nmse": 0.18}])
        _write(setting / run / "gates" / "gate_metrics.json",
               [{"split": "eval", "baseline": "gated_context_classifier_scalar", "mse": 0.0007,
                 "mae": 0.018, "nmse": 0.22},
                {"split": "eval", "baseline": "gated_context_regressor_horizon", "mse": 0.0006,
                 "mae": 0.016, "nmse": 0.2},
                {"split": "eval", "baseline": "oracle_context_scalar", "mse": 0.0002, "mae": 0.01, "nmse": 0.1},
                {"split": "eval", "baseline": "oracle_context_horizon", "mse": 0.0001, "mae": 0.005, "nmse": 0.05}])
        _write(setting / run / "ts_ifa" / "eval_metrics.json",
               {"adapted_mse": 0.0008, "adapted_mae": 0.018, "adapted_nmse": 0.25,
                "vanilla_mse": 0.0012, "vanilla_nmse": 0.4})

        records = discover_results(root)
        methods = {record.method for record in records if record.metric == "mse"}
        assert methods == {
            "chronos",
            f"{run}/vanilla",
            f"{run}/mix_1_learned",
            f"{run}/mix_1_learned_eval_fit",
            f"{run}/gated_context_classifier_scalar",
            f"{run}/gated_context_regressor_horizon",
            f"{run}/oracle_context_scalar",
            f"{run}/oracle_context_horizon",
            f"{run}/TS-IFA",
        }
        output = generate_results_table(
            root,
            methods=["chronos", f"{run}/mix_1_learned", f"{run}/TS-IFA",
                     f"{run}/oracle_context_scalar", f"{run}/oracle_context_horizon"],
            reference="chronos",
        )
        latex = output.read_text(encoding="utf-8")
        assert r"$\times 10^{-3}$" in latex
        assert r"\textbf{0.80}" in latex
        assert "33.33\\%" in latex
        assert r"IN\_L2\_3/TS-IFA" in latex
        assert "online" not in latex
        assert r"\begin{tabular}{llcrrr|rr}" in latex
        assert r"\textbf{0.10}" not in latex

        default_output = generate_results_table(root, output=root / "default.tex", datasets=["electricity"])
        default_latex = default_output.read_text(encoding="utf-8")
        assert "vanilla" not in default_latex
        assert r"IN\_L2\_3/oracle-s" in default_latex
        assert r"IN\_L2\_3/gate-cls-s" in default_latex
        assert r"IN\_L2\_3/gate-reg-h" in default_latex

        baseline_output = generate_results_table(
            root,
            output=root / "baselines.tex",
            methods=["chronos", f"{run}/mix_1_learned", f"{run}/mix_1_learned_eval_fit"],
            reference="chronos",
            excluded_from_bold=["mix_1_learned_eval_fit"],
        )
        baseline_latex = baseline_output.read_text(encoding="utf-8")
        assert r"IN\_L2\_3/mix1-fit-T3" in baseline_latex
        assert r"\begin{tabular}{llcrr|r}" in baseline_latex

        fixed_run = "chronos_raw_euclidean_3_fixed"
        _write(setting / fixed_run / "baseline_adapters" / "baseline_metrics.json",
               [{"split": "eval", "baseline": "mix_0_weighted", "mse": 0.001, "mae": 0.02, "nmse": 0.35}])
        fixed_output = generate_results_table(root, output=root / "fixed.tex", datasets=["electricity"])
        assert r"raw\_L2\_3\_fixed/mix0" in fixed_output.read_text(encoding="utf-8")

        _write(root / "toy" / "1_1" / "direct" / "reference" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 1.0}}})
        _write(root / "toy" / "1_1" / "direct" / "candidate" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 0.5}}})
        _write(root / "toy" / "2_1" / "direct" / "reference" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 9.0}}})
        _write(root / "toy" / "2_1" / "direct" / "candidate" / "univariate_summary.json",
               {"eval": {"mse": {"mean": 8.1}}})
        averaged_output = generate_results_table(
            root,
            output=root / "averaged.tex",
            datasets=["toy"],
            methods=["reference", "candidate"],
            reference="reference",
            setting_improvements=False,
        )
        averaged_latex = averaged_output.read_text(encoding="utf-8")
        assert "14.00\\%" in averaged_latex
        assert "30.00\\%" not in averaged_latex

    print("results table checks passed")


if __name__ == "__main__":
    main()
