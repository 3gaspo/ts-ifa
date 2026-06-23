"""Build publication-ready LaTeX tables from a TS-IFA experiment folder.

The loader understands direct ``univariate_summary.json`` results, adapter
``baseline_metrics.json`` results, and ``ts_ifa/eval_metrics.json`` results.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Result:
    dataset: str
    setting: str
    method: str
    split: str
    metric: str
    value: float
    path: Path


def _setting_ancestor(path: Path, root: Path) -> tuple[str, str] | None:
    relative = path.relative_to(root)
    parts = relative.parts
    for index in range(len(parts) - 1, 0, -1):
        if re.fullmatch(r"\d+[_-]\d+", parts[index]):
            return parts[index - 1], parts[index]
    return None


def _relative_run(path: Path, root: Path, setting: str) -> str:
    parts = path.relative_to(root).parts
    index = parts.index(setting)
    return parts[index + 1] if index + 1 < len(parts) - 1 else path.parent.name


def discover_results(experiment_dir: str | Path) -> list[Result]:
    """Discover direct, baseline, and TS-IFA metrics below a result root."""
    root = Path(experiment_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root}")
    results: list[Result] = []

    for path in sorted(root.rglob("univariate_summary.json")):
        identity = _setting_ancestor(path, root)
        if identity is None:
            continue
        dataset, setting = identity
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split, metrics in payload.items():
            if not isinstance(metrics, Mapping):
                continue
            for metric, summary in metrics.items():
                value = summary.get("mean") if isinstance(summary, Mapping) else summary
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = math.nan
                results.append(Result(dataset, setting, path.parent.name, str(split), str(metric), value, path))

    for path in sorted(root.rglob("baseline_metrics.json")):
        identity = _setting_ancestor(path, root)
        if identity is None:
            continue
        dataset, setting = identity
        run = _relative_run(path, root, setting)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload if isinstance(payload, list) else ():
            if not isinstance(row, Mapping) or "baseline" not in row:
                continue
            for metric in ("mse", "mae", "nmse"):
                if metric in row:
                    results.append(
                        Result(dataset, setting, f"{run}/{row['baseline']}", str(row.get("split", "eval")), metric,
                               float(row[metric]), path)
                    )

    for path in sorted(root.rglob("eval_metrics.json")):
        identity = _setting_ancestor(path, root)
        if identity is None:
            continue
        dataset, setting = identity
        run = _relative_run(path, root, setting)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            continue
        for key, value in payload.items():
            match = re.fullmatch(r"(.+)_(mse|mae|nmse)", str(key).lower())
            if match is None:
                continue
            variant, metric = match.groups()
            method = f"{run}/TS-IFA" if variant == "adapted" else f"{run}/{variant}"
            results.append(Result(dataset, setting, method, "eval", metric, float(value), path))
    return results


def _split_names(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    values = re.split(r"[,;]", value) if isinstance(value, str) else [str(item) for item in value]
    return [item.strip() for item in values if item.strip()]


def _setting_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"[_-]", value))


def _parse_dataset_settings(values: Iterable[str] | None) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"dataset setting must be DATASET=L_H[,L_H], got {item!r}")
        dataset, settings = item.split("=", 1)
        selected.setdefault(dataset.strip(), set()).update(_split_names(settings) or ())
    return selected


def _parse_scale_exponents(values: Iterable[str] | None) -> dict[tuple[str, str], int]:
    exponents: dict[tuple[str, str], int] = {}
    for item in values or ():
        if "=" not in item or "/" not in item.split("=", 1)[0]:
            raise ValueError(f"scale must be DATASET/L_H=EXPONENT, got {item!r}")
        row, exponent = item.split("=", 1)
        dataset, setting = row.split("/", 1)
        exponents[(dataset.strip(), setting.strip())] = int(exponent)
    return exponents


def _latex(text: Any) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(text))


def _latex_setting(setting: str) -> str:
    return "--".join(_latex(part) for part in re.split(r"[_-]", setting))


_METHOD_LABELS = {
    "context_conditioned": "context",
    "neighbor_weighted_mean": "kNN-w",
    "neighbor_unweighted_mean": "kNN",
    "pred_plus_weighted_e": "residual",
    "mix_0_weighted": "mix0",
    "mix_1_learned": "mix1",
    "mix_2_full_horizon": "mix2",
    "gated_context_scalar": "gate-s",
    "gated_context_horizon": "gate-h",
    "oracle_context_scalar": "oracle-s",
    "oracle_context_horizon": "oracle-h",
}


def _short_run_name(run: str) -> str:
    match = re.fullmatch(
        r"chronos_(raw|fourier|chronos|patchtst|model|representation)_(euclidean|cosine|pearson)_(\d+)_(online|fixed)",
        run,
    )
    if match is not None:
        space, metric, neighbors, mode = match.groups()
        metric = "L2" if metric == "euclidean" else metric
        parts = [space, metric, neighbors]
        if mode == "fixed":
            parts.append("fixed")
        return "_".join(parts)
    short = run.removeprefix("chronos_").replace("_euclidean_", "_L2_")
    return short.removesuffix("_online")


def _method_label(method: str, short_names: bool) -> str:
    if not short_names or "/" not in method:
        return method
    run, variant = method.rsplit("/", 1)
    return f"{_short_run_name(run)}/{_METHOD_LABELS.get(variant, variant)}"


def _method_selected(method: str, selectors: set[str]) -> bool:
    return method in selectors or method.rsplit("/", 1)[-1] in selectors


def _auto_exponent(values: Sequence[float], lower_is_better: bool) -> int:
    del lower_is_better
    finite = [abs(value) for value in values if math.isfinite(value) and value != 0]
    if not finite:
        return 0
    finite.sort()
    middle = len(finite) // 2
    anchor = finite[middle] if len(finite) % 2 else (finite[middle - 1] + finite[middle]) / 2.0
    return math.floor(math.log10(anchor))


def _improvement(reference: float, current: float, lower_is_better: bool) -> float:
    if not math.isfinite(reference) or not math.isfinite(current) or reference == 0:
        return math.nan
    return (1.0 if lower_is_better else -1.0) * (reference - current) / abs(reference) * 100.0


def _improvements_of_averages(rows: Sequence[Mapping[str, float]], methods: Sequence[str], reference: str,
                              lower_is_better: bool) -> dict[str, float]:
    averages = {}
    for method in methods:
        finite = [row.get(method, math.nan) for row in rows]
        finite = [value for value in finite if math.isfinite(value)]
        averages[method] = sum(finite) / len(finite) if finite else math.nan
    reference_average = averages.get(reference, math.nan)
    return {
        method: _improvement(reference_average, averages[method], lower_is_better)
        for method in methods
    }


def _format_cells(values: Mapping[str, float], methods: Sequence[str], decimals: int, *, lower_is_better: bool,
                  bold: bool, divisor: float = 1.0, percent: bool = False,
                  bold_methods: Sequence[str] | None = None) -> list[str]:
    eligible = set(methods if bold_methods is None else bold_methods)
    finite = [values.get(method, math.nan) for method in methods if method in eligible]
    finite = [value for value in finite if math.isfinite(value)]
    best = (min(finite) if lower_is_better else max(finite)) if finite else None
    cells = []
    for method in methods:
        raw = values.get(method, math.nan)
        if not math.isfinite(raw):
            cells.append("--")
            continue
        cell = f"{raw / divisor:.{decimals}f}" + (r"\%" if percent else "")
        if bold and method in eligible and best is not None and math.isclose(raw, best, rel_tol=1e-12, abs_tol=1e-15):
            cell = rf"\textbf{{{cell}}}"
        cells.append(cell)
    return cells


def build_table(results: Sequence[Result], *, metric: str = "mse", split: str = "eval",
                datasets: Sequence[str] | None = None, settings: Sequence[str] | None = None,
                dataset_settings: Mapping[str, set[str]] | None = None, methods: Sequence[str] | None = None,
                reference: str | None = None, decimals: int = 2, lower_is_better: bool = True,
                bold: bool = True, dataset_improvements: bool = True, setting_improvements: bool = True,
                overall_improvement: bool = True, auto_scale: bool = True, scale_exponent: int | None = None,
                scale_exponents: Mapping[tuple[str, str], int] | None = None, caption: str | None = None,
                label: str = "tab:results", excluded_from_bold: Sequence[str] | None = None,
                short_names: bool = True) -> str:
    """Render selected records as a complete LaTeX table environment."""
    filtered = [result for result in results if result.metric.casefold() == metric.casefold()
                and result.split.casefold() == split.casefold()]
    dataset_order = list(datasets) if datasets else sorted({result.dataset for result in filtered}, key=str.casefold)
    filtered = [result for result in filtered if result.dataset in set(dataset_order)]
    global_settings, per_dataset = set(settings or ()), dataset_settings or {}
    if global_settings or per_dataset:
        filtered = [
            result for result in filtered
            if (result.setting in per_dataset[result.dataset] if result.dataset in per_dataset
                else not global_settings or result.setting in global_settings)
        ]
    if methods:
        method_order = list(methods)
    else:
        method_order = sorted(
            {result.method for result in filtered if result.method.rsplit("/", 1)[-1] != "vanilla"},
            key=str.casefold,
        )
    excluded_selectors = set(excluded_from_bold or ())
    excluded_methods = [
        method for method in method_order
        if method.rsplit("/", 1)[-1].startswith("oracle_")
        or _method_selected(method, excluded_selectors)
    ]
    regular_methods = [method for method in method_order if method not in set(excluded_methods)]
    method_order = [*regular_methods, *excluded_methods]
    filtered = [result for result in filtered if result.method in set(method_order)]
    if not filtered:
        raise ValueError(f"no results match metric={metric!r}, split={split!r}, and the selected filters")
    reference = reference or method_order[0]
    if reference not in method_order:
        raise ValueError(f"reference {reference!r} is not in selected methods {method_order}")

    grouped: dict[tuple[str, str, str], list[float]] = {}
    for result in filtered:
        grouped.setdefault((result.dataset, result.setting, result.method), []).append(result.value)
    table: dict[tuple[str, str], dict[str, float]] = {}
    for (dataset, setting, method), values in grouped.items():
        finite = [value for value in values if math.isfinite(value)]
        table.setdefault((dataset, setting), {})[method] = sum(finite) / len(finite) if finite else math.nan
    dataset_order = [dataset for dataset in dataset_order if any(key[0] == dataset for key in table)]
    settings_by_dataset = {dataset: sorted((key[1] for key in table if key[0] == dataset), key=_setting_key)
                           for dataset in dataset_order}
    observed_settings = sorted({setting for _, setting in table}, key=_setting_key)
    exponent_overrides = scale_exponents or {}

    column_spec = "llc" + "r" * len(regular_methods)
    if excluded_methods:
        column_spec += "|" + "r" * len(excluded_methods)
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{{_latex(caption or f'{metric.upper()} results on {split}.')}}}",
             r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{column_spec}}}", r"\toprule",
             "Dataset & $L$--$H$ & Scale & "
             + " & ".join(_latex(_method_label(method, short_names)) for method in method_order) + r" \\",
             r"\midrule"]
    for dataset_index, dataset in enumerate(dataset_order):
        row_settings = settings_by_dataset[dataset]
        for setting_index, setting in enumerate(row_settings):
            row = table[(dataset, setting)]
            scale_values = [row[reference]] if math.isfinite(row.get(reference, math.nan)) else list(row.values())
            exponent = (exponent_overrides[(dataset, setting)] if (dataset, setting) in exponent_overrides
                        else scale_exponent if scale_exponent is not None
                        else _auto_exponent(scale_values, lower_is_better) if auto_scale else 0)
            dataset_cell = rf"\multirow{{{len(row_settings)}}}{{*}}{{{_latex(dataset)}}}" if setting_index == 0 else ""
            cells = _format_cells(row, method_order, decimals, lower_is_better=lower_is_better, bold=bold,
                                  divisor=10.0**exponent, bold_methods=regular_methods)
            lines.append(" & ".join([dataset_cell, _latex_setting(setting),
                                      rf"$\times 10^{{{exponent}}}$", *cells]) + r" \\")
        if dataset_improvements:
            improvements = _improvements_of_averages([table[(dataset, setting)] for setting in row_settings],
                                                      method_order, reference, lower_is_better)
            cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                                  percent=True, bold_methods=regular_methods)
            lines.append(" & ".join(["", r"\textit{Improvement}", "", *cells]) + r" \\")
        if dataset_index < len(dataset_order) - 1:
            lines.append(r"\midrule")
    if setting_improvements:
        lines.extend([r"\midrule", r"\multicolumn{%d}{l}{\textit{Improvements by setting}} \\" % (3 + len(method_order))])
        for setting in observed_settings:
            rows = [row for (_, row_setting), row in table.items() if row_setting == setting]
            improvements = _improvements_of_averages(rows, method_order, reference, lower_is_better)
            cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                                  percent=True, bold_methods=regular_methods)
            lines.append(" & ".join(["", _latex_setting(setting), "", *cells]) + r" \\")
    if overall_improvement:
        improvements = _improvements_of_averages(list(table.values()), method_order, reference, lower_is_better)
        cells = _format_cells(improvements, method_order, decimals, lower_is_better=False, bold=bold,
                              percent=True, bold_methods=regular_methods)
        lines.extend([r"\midrule", " & ".join([r"\multicolumn{2}{l}{Overall improvement}", "", *cells]) + r" \\"])
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", rf"\label{{{_latex(label)}}}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def generate_results_table(experiment_dir: str | Path, output: str | Path | None = None, **kwargs: Any) -> Path:
    root = Path(experiment_dir).expanduser().resolve()
    default_name = f"results_{str(kwargs.get('metric', 'mse')).lower()}.tex"
    destination = Path(output).expanduser().resolve() if output else root / default_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_table(discover_results(root), **kwargs), encoding="utf-8")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metric", default="mse")
    parser.add_argument("--split", default="eval")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--dataset-settings", action="append", default=[], metavar="DATASET=L_H,L_H")
    parser.add_argument("--methods", default=None, help="Comma/semicolon-separated ordered columns")
    parser.add_argument("--reference", default=None)
    parser.add_argument(
        "--exclude-from-bold",
        default=None,
        help="Comma/semicolon-separated method IDs or variant names to move right and exclude from bolding",
    )
    parser.add_argument("--long-method-names", action="store_true")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--no-bold", action="store_true")
    parser.add_argument("--no-dataset-improvements", action="store_true")
    parser.add_argument("--no-setting-improvements", action="store_true")
    parser.add_argument("--no-overall-improvement", action="store_true")
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument("--scale-exponent", type=int, default=None)
    parser.add_argument("--row-scale", action="append", default=[], metavar="DATASET/L_H=EXPONENT")
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:results")
    args = parser.parse_args(argv)
    if args.decimals < 0:
        parser.error("--decimals must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    output = generate_results_table(
        args.experiment_dir, args.output, metric=args.metric, split=args.split,
        datasets=_split_names(args.datasets), settings=_split_names(args.settings),
        dataset_settings=_parse_dataset_settings(args.dataset_settings), methods=_split_names(args.methods),
        reference=args.reference, decimals=args.decimals, lower_is_better=not args.higher_is_better,
        bold=not args.no_bold, dataset_improvements=not args.no_dataset_improvements,
        setting_improvements=not args.no_setting_improvements, overall_improvement=not args.no_overall_improvement,
        auto_scale=not args.no_auto_scale, scale_exponent=args.scale_exponent,
        scale_exponents=_parse_scale_exponents(args.row_scale), caption=args.caption, label=args.label,
        excluded_from_bold=_split_names(args.exclude_from_bold), short_names=not args.long_method_names,
    )
    print(f"LaTeX table written to {output}")
    return output


if __name__ == "__main__":
    main()
