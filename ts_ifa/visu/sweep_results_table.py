"""Build sweep LaTeX tables across retrieval spaces and neighbor counts."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .results_table import (
    Result,
    _latex,
    _method_label,
    _parse_dataset_settings,
    _short_run_name,
    _split_names,
    build_table,
    discover_results,
)


BASELINE_VARIANTS = (
    "context_forecast",
    "horizon_knn_weighted",
    "horizon_knn_mean",
    "horizon_mix_scalar",
    "horizon_ridge_shared",
    "residual_knn_weighted",
    "residual_mix_scalar",
    "residual_ridge_shared",
    "residual_ridge_horizon",
    "full_ridge_horizon",
    "horizon_mix_scalar_eval_fit",
    "horizon_ridge_shared_eval_fit",
    "residual_mix_scalar_eval_fit",
    "residual_ridge_shared_eval_fit",
    "residual_ridge_horizon_eval_fit",
    "full_ridge_horizon_eval_fit",
)

GATE_VARIANTS = (
    "context_forecast",
    "bayes_context_scalar",
    "bayes_context_horizon",
    "catboost_context_classifier_scalar",
    "catboost_context_classifier_horizon",
    "catboost_context_regressor_scalar",
    "catboost_context_regressor_horizon",
    "oracle_context_scalar",
    "oracle_context_horizon",
)

TS_IFA_VARIANTS = ("TS-IFA",)

EVAL_FIT_VARIANTS = tuple(variant for variant in BASELINE_VARIANTS if variant.endswith("_eval_fit"))


@dataclass(frozen=True)
class Family:
    name: str
    variants: tuple[str, ...]
    output_name: str
    caption: str
    label: str
    exclude_from_bold: tuple[str, ...] = ()
    include_chronos: bool = True


FAMILIES = (
    Family(
        "baselines",
        BASELINE_VARIANTS,
        "baselines_results.tex",
        "Baseline nMSE results across retrieval settings",
        "tab:baselines-results",
        EVAL_FIT_VARIANTS,
    ),
    Family(
        "gates",
        GATE_VARIANTS,
        "gates_results.tex",
        "Gate nMSE results across retrieval settings",
        "tab:gates-results",
    ),
    Family(
        "ts_ifa",
        TS_IFA_VARIANTS,
        "ts_ifa_results.tex",
        "TS-IFA nMSE results across retrieval settings",
        "tab:ts-ifa-results",
    ),
)


def _run_name(space: str, neighbors: int, retrieval_mode: str) -> str:
    return f"chronos_{space}_euclidean_{neighbors}_{retrieval_mode}"


def _run_names(spaces: Sequence[str], neighbors: Sequence[int], retrieval_mode: str) -> list[str]:
    return [_run_name(space, k, retrieval_mode) for space in spaces for k in neighbors]


def _methods_for_family(family: Family, runs: Sequence[str]) -> list[str]:
    methods = ["chronos"] if family.include_chronos else []
    methods.extend(f"{run}/{variant}" for run in runs for variant in family.variants)
    return methods


def _filters_match(
    result: Result,
    dataset_order: Sequence[str] | None,
    setting_filter: set[str],
    dataset_settings: Mapping[str, set[str]],
) -> bool:
    if dataset_order is not None and result.dataset not in dataset_order:
        return False
    if result.dataset in dataset_settings:
        return result.setting in dataset_settings[result.dataset]
    return not setting_filter or result.setting in setting_filter


def _average_metric(
    results: Sequence[Result],
    *,
    method: str,
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
) -> float:
    dataset_order = list(datasets) if datasets else None
    setting_filter = set(settings or ())
    values = [
        result.value
        for result in results
        if result.method == method
        and result.metric.casefold() == metric.casefold()
        and result.split.casefold() == split.casefold()
        and _filters_match(result, dataset_order, setting_filter, dataset_settings)
        and math.isfinite(result.value)
    ]
    return sum(values) / len(values) if values else math.nan


def _matrix_row_label(variant: str) -> str:
    if variant == "chronos":
        return "chronos"
    return _method_label(f"run/{variant}", True).rsplit("/", 1)[-1]


def _relative_improvement(reference: float, value: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or reference == 0:
        return math.nan
    direction = 1.0 if lower_is_better else -1.0
    return direction * (reference - value) / abs(reference) * 100.0


def _matrix_cell(value: float, improvement: float, decimals: int, bold: bool) -> str:
    if not math.isfinite(value) or not math.isfinite(improvement):
        return "--"
    top = f"{improvement:.{decimals}f}" + r"\%"
    if bold:
        top = rf"\textbf{{{top}}}"
    bottom = rf"{{\scriptsize {value:.{decimals}f}}}"
    return rf"\begin{{tabular}}{{@{{}}c@{{}}}}{top}\\{bottom}\end{{tabular}}"


def build_matrix_table(
    results: Sequence[Result],
    *,
    variants: Sequence[str],
    runs: Sequence[str],
    metric: str = "nmse",
    split: str = "eval",
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    decimals: int = 2,
    lower_is_better: bool = True,
    caption: str | None = None,
    label: str = "tab:sweep-matrix",
    include_chronos: bool = True,
    excluded_from_bold: Sequence[str] | None = None,
) -> str:
    """Render the neo-seminar matrix: models as rows, retrieval settings as columns."""
    dataset_settings = dataset_settings or {}
    del include_chronos
    row_variants = list(variants)
    reference = _average_metric(
        results,
        method="chronos",
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
    )
    values: dict[tuple[str, str], float] = {}
    for variant in row_variants:
        for run in runs:
            method = f"{run}/{variant}"
            values[(variant, run)] = _average_metric(
                results,
                method=method,
                metric=metric,
                split=split,
                datasets=datasets,
                settings=settings,
                dataset_settings=dataset_settings,
            )

    excluded_selectors = set(excluded_from_bold or ())
    eligible_variants = {
        variant
        for variant in row_variants
        if not variant.startswith("oracle_") and variant not in excluded_selectors
    }
    finite = [
        improvement
        for (variant, _), value in values.items()
        if variant in eligible_variants
        and math.isfinite(improvement := _relative_improvement(reference, value, lower_is_better))
    ]
    best = max(finite) if finite else None
    column_spec = "l" + "c" * len(runs)
    caption_text = caption or f"Average {metric.upper()} by retrieval setting."
    caption_separator = " " if caption_text.rstrip().endswith((".", "?", "!")) else ". "
    reference_text = (
        f"{caption_separator}Direct Chronos {metric.upper()}: {reference:.{decimals}f}."
        if math.isfinite(reference)
        else f"{caption_separator}Direct Chronos {metric.upper()}: unavailable."
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_latex(caption_text + reference_text)}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(_latex(_short_run_name(run)) for run in runs) + r" \\",
        r"\midrule",
    ]
    for variant in row_variants:
        cells = []
        for run in runs:
            value = values[(variant, run)]
            improvement = _relative_improvement(reference, value, lower_is_better)
            is_best = (
                variant in eligible_variants
                and best is not None
                and math.isclose(improvement, best, rel_tol=1e-12, abs_tol=1e-15)
            )
            cells.append(_matrix_cell(value, improvement, decimals, is_best))
        lines.append(" & ".join([_latex(_matrix_row_label(variant)), *cells]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", rf"\label{{{_latex(label)}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _write_family_table(
    results: Sequence[Result],
    output_dir: Path,
    family: Family,
    *,
    runs: Sequence[str],
    metric: str,
    split: str,
    datasets: Sequence[str] | None,
    settings: Sequence[str] | None,
    dataset_settings: Mapping[str, set[str]],
    decimals: int,
    lower_is_better: bool,
) -> Path:
    methods = _methods_for_family(family, runs)
    regular_caption = family.caption + " by dataset and horizon setting"
    matrix_caption = family.caption + ", averaged over selected datasets and horizon settings"
    regular = build_table(
        results,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
        methods=methods,
        reference="chronos",
        decimals=decimals,
        lower_is_better=lower_is_better,
        dataset_improvements=False,
        setting_improvements=False,
        overall_improvement=False,
        caption=regular_caption,
        label=family.label,
        excluded_from_bold=family.exclude_from_bold,
    )
    matrix = build_matrix_table(
        results,
        variants=family.variants,
        runs=runs,
        metric=metric,
        split=split,
        datasets=datasets,
        settings=settings,
        dataset_settings=dataset_settings,
        decimals=decimals,
        lower_is_better=lower_is_better,
        caption=matrix_caption,
        label=f"{family.label}-matrix",
        include_chronos=family.include_chronos,
        excluded_from_bold=family.exclude_from_bold,
    )
    output = output_dir / family.output_name
    output.write_text(regular + "\n" + matrix, encoding="utf-8")
    return output


def generate_sweep_results_tables(
    experiment_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    datasets: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    dataset_settings: Mapping[str, set[str]] | None = None,
    spaces: Sequence[str] = ("raw", "instance"),
    neighbors: Sequence[int] = (1, 3, 10),
    retrieval_mode: str = "online",
    metric: str = "nmse",
    split: str = "eval",
    decimals: int = 2,
    lower_is_better: bool = True,
) -> list[Path]:
    root = Path(experiment_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve() if output_dir else root / "sweep_tables"
    destination.mkdir(parents=True, exist_ok=True)
    records = discover_results(root)
    runs = _run_names(spaces, neighbors, retrieval_mode)
    return [
        _write_family_table(
            records,
            destination,
            family,
            runs=runs,
            metric=metric,
            split=split,
            datasets=datasets,
            settings=settings,
            dataset_settings=dataset_settings or {},
            decimals=decimals,
            lower_is_better=lower_is_better,
        )
        for family in FAMILIES
    ]


def _parse_neighbors(value: str | Sequence[str] | None) -> list[int]:
    if value is None:
        return [1, 3, 10]
    return [int(item) for item in _split_names(value)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric", default="nmse")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--spaces", default="raw,instance")
    parser.add_argument("--neighbors", default="1,3,10")
    parser.add_argument("--retrieval-mode", default="online")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    outputs = generate_sweep_results_tables(
        args.experiment_dir,
        args.output_dir,
        metric=args.metric,
        split=args.split,
        datasets=_split_names(args.datasets),
        settings=_split_names(args.settings),
        dataset_settings=_parse_dataset_settings(args.dataset_settings),
        spaces=_split_names(args.spaces) or ("raw", "instance"),
        neighbors=_parse_neighbors(args.neighbors),
        retrieval_mode=args.retrieval_mode,
        decimals=args.decimals,
        lower_is_better=not args.higher_is_better,
    )
    for output in outputs:
        print(f"LaTeX table written to {output}")
    return outputs


if __name__ == "__main__":
    main()
