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
        _write(setting / "chronos_raw_1" / "baseline_adapters" / "baseline_metrics.json",
               [{"split": "eval", "baseline": "linear_mix", "mse": 0.0009, "mae": 0.02, "nmse": 0.3}])
        _write(setting / "chronos_raw_1" / "ts_ifa" / "eval_metrics.json",
               {"adapted_mse": 0.0008, "adapted_mae": 0.018, "adapted_nmse": 0.25,
                "vanilla_nmse": 0.4})

        records = discover_results(root)
        methods = {record.method for record in records if record.metric == "mse"}
        assert methods == {"chronos", "chronos_raw_1/linear_mix", "chronos_raw_1/TS-IFA"}
        output = generate_results_table(
            root,
            methods=["chronos", "chronos_raw_1/linear_mix", "chronos_raw_1/TS-IFA"],
            reference="chronos",
        )
        latex = output.read_text(encoding="utf-8")
        assert r"$\times 10^{-3}$" in latex
        assert r"\textbf{0.80}" in latex
        assert "33.33\\%" in latex
        assert r"chronos\_raw\_1/TS-IFA" in latex

    print("results table checks passed")


if __name__ == "__main__":
    main()
