#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / f"matplotlib-cache-urban-embedding-{os.getpid()}"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distribution plots and summary statistics for scalar CSV columns."
    )
    parser.add_argument("csv_path", type=Path, help="CSV file to inspect.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Logical dataset name used for titles and output directory naming.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: data/channel_distribution_report/<dataset-name>",
    )
    parser.add_argument(
        "--docs-readme",
        type=Path,
        default=None,
        help="Optional Markdown report path under docs/.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=None,
        help="Explicit value columns. If omitted, columns are selected by year range.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Inclusive start year when inferring columns from column names.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Inclusive end year when inferring columns from column names.",
    )
    parser.add_argument(
        "--year-regex",
        type=str,
        default=r"(19|20)\d{2}",
        help="Regex used to extract years from column names.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Histogram bin count. Default: 60.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG DPI. Default: 150.",
    )
    return parser.parse_args()


def infer_dataset_name(csv_path: Path, dataset_name: str | None) -> str:
    if dataset_name:
        return dataset_name
    return csv_path.stem


def extract_year(column_name: str, year_pattern: re.Pattern[str]) -> int | None:
    match = year_pattern.search(str(column_name))
    if match is None:
        return None
    return int(match.group(0))


def select_value_columns(
    df: pd.DataFrame,
    *,
    explicit_columns: list[str] | None,
    start_year: int | None,
    end_year: int | None,
    year_regex: str,
) -> list[tuple[str, str]]:
    if explicit_columns:
        missing = [column for column in explicit_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        return [(column, column) for column in explicit_columns]

    if start_year is None or end_year is None:
        raise ValueError("Either --columns or both --start-year and --end-year are required.")

    year_pattern = re.compile(year_regex)
    selected: list[tuple[int, str, str]] = []
    for column in df.columns:
        year = extract_year(column, year_pattern)
        if year is None or not (start_year <= year <= end_year):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        selected.append((year, str(year), column))

    selected.sort(key=lambda item: item[0])
    deduped: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for _, label, column in selected:
        if label in seen_labels:
            raise ValueError(f"Found multiple columns for inferred year label {label!r}. Use --columns explicitly.")
        seen_labels.add(label)
        deduped.append((label, column))
    return deduped


def build_year_summary_rows(df: pd.DataFrame, selected_columns: list[tuple[str, str]], dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = int(len(df))
    for label, column in selected_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        values = numeric[np.isfinite(numeric.to_numpy(dtype=float, copy=False))]
        values = values.to_numpy(dtype=float, copy=False)
        nan_count = int(total_rows - values.size)
        if values.size == 0:
            rows.append(
                {
                    "dataset": dataset_name,
                    "year_label": label,
                    "column_name": column,
                    "total_rows": total_rows,
                    "valid_count": 0,
                    "nan_count": nan_count,
                    "min": np.nan,
                    "p01": np.nan,
                    "p05": np.nan,
                    "p25": np.nan,
                    "p50": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "mean": np.nan,
                    "std": np.nan,
                }
            )
            continue

        quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
        rows.append(
            {
                "dataset": dataset_name,
                "year_label": label,
                "column_name": column,
                "total_rows": total_rows,
                "valid_count": int(values.size),
                "nan_count": nan_count,
                "min": float(values.min()),
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p25": float(quantiles[2]),
                "p50": float(quantiles[3]),
                "p75": float(quantiles[4]),
                "p95": float(quantiles[5]),
                "p99": float(quantiles[6]),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def _concat_selected_values(df: pd.DataFrame, selected_columns: list[tuple[str, str]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for _, column in selected_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, copy=False))].to_numpy(dtype=float, copy=False)
        if finite.size:
            values.append(finite)
    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate(values, axis=0)


def build_aggregate_summary_row(
    df: pd.DataFrame,
    selected_columns: list[tuple[str, str]],
    dataset_name: str,
) -> pd.DataFrame:
    values = _concat_selected_values(df, selected_columns)
    total_rows = int(len(df))
    total_cells = total_rows * len(selected_columns)
    nan_count = int(total_cells - values.size)
    if values.size == 0:
        payload = {
            "dataset": dataset_name,
            "year_range": f"{selected_columns[0][0]}-{selected_columns[-1][0]}",
            "selected_year_count": len(selected_columns),
            "total_rows": total_rows,
            "total_cells": total_cells,
            "valid_count": 0,
            "nan_count": nan_count,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
        return pd.DataFrame([payload])

    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    payload = {
        "dataset": dataset_name,
        "year_range": f"{selected_columns[0][0]}-{selected_columns[-1][0]}",
        "selected_year_count": len(selected_columns),
        "total_rows": total_rows,
        "total_cells": total_cells,
        "valid_count": int(values.size),
        "nan_count": nan_count,
        "min": float(values.min()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }
    return pd.DataFrame([payload])


def infer_entity_column(df: pd.DataFrame) -> tuple[str | None, str]:
    if "ID" in df.columns and ("City" in df.columns or "UC_NM_MN" in df.columns):
        return "ID", "cities"
    if "ID" in df.columns:
        return "ID", "entities"
    if "City" in df.columns:
        return "City", "cities"
    if "UC_NM_MN" in df.columns:
        return "UC_NM_MN", "cities"
    return None, "rows"


def singularize(label: str) -> str:
    irregular = {
        "cities": "city",
        "entities": "entity",
        "rows": "row",
    }
    if label in irregular:
        return irregular[label]
    if label.endswith("s"):
        return label[:-1]
    return label


def build_coverage_summary(
    df: pd.DataFrame,
    selected_columns: list[tuple[str, str]],
    dataset_name: str,
) -> pd.DataFrame:
    valid_matrix = pd.DataFrame(
        {
            label: pd.to_numeric(df[column], errors="coerce").notna()
            for label, column in selected_columns
        }
    )
    valid_counts = valid_matrix.sum(axis=1)
    entity_column, entity_label = infer_entity_column(df)

    if entity_column is None:
        total_entities = int(len(df))
        any_valid_entities = int((valid_counts > 0).sum())
        all_valid_entities = int((valid_counts == len(selected_columns)).sum())
    else:
        tmp = pd.DataFrame(
            {
                "entity": df[entity_column],
                "valid_count": valid_counts,
            }
        )
        grouped = tmp.groupby("entity", dropna=False)["valid_count"].max()
        total_entities = int(grouped.shape[0])
        any_valid_entities = int((grouped > 0).sum())
        all_valid_entities = int((grouped == len(selected_columns)).sum())
        valid_counts = grouped

    payload = {
        "dataset": dataset_name,
        "year_range": f"{selected_columns[0][0]}-{selected_columns[-1][0]}",
        "selected_year_count": len(selected_columns),
        "entity_label": entity_label,
        "entity_column": entity_column,
        "total_entities": total_entities,
        "entities_with_any_valid": any_valid_entities,
        "entities_with_all_valid": all_valid_entities,
        "mean_valid_years_per_entity": float(valid_counts.mean()) if len(valid_counts) else np.nan,
        "median_valid_years_per_entity": float(valid_counts.median()) if len(valid_counts) else np.nan,
        "max_valid_years_per_entity": int(valid_counts.max()) if len(valid_counts) else 0,
    }
    return pd.DataFrame([payload])


def plot_histograms(
    df: pd.DataFrame,
    selected_columns: list[tuple[str, str]],
    output_path: Path,
    dataset_name: str,
    bins: int,
    dpi: int,
) -> None:
    values = _concat_selected_values(df, selected_columns)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if values.size == 0:
        ax.set_title(f"{dataset_name}: no valid values")
        ax.axis("off")
    else:
        ax.hist(values, bins=bins, color="#3A86FF", edgecolor="white", alpha=0.85)
        ax.set_title(
            f"{dataset_name}: pooled histogram for {selected_columns[0][0]}-{selected_columns[-1][0]}  "
            f"(n={values.size})"
        )
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot(
    df: pd.DataFrame,
    selected_columns: list[tuple[str, str]],
    output_path: Path,
    dataset_name: str,
    dpi: int,
) -> None:
    labels: list[str] = []
    data: list[np.ndarray] = []
    for label, column in selected_columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        values = numeric[np.isfinite(numeric.to_numpy(dtype=float, copy=False))].to_numpy(dtype=float, copy=False)
        if values.size == 0:
            continue
        labels.append(label)
        data.append(values)

    if not data:
        raise ValueError("No valid numeric data available for boxplot.")

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 5.5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title(f"{dataset_name}: distribution by year")
    ax.set_xlabel("year")
    ax.set_ylabel("value")
    ax.grid(axis="y", alpha=0.2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    *,
    readme_path: Path,
    dataset_name: str,
    csv_path: Path,
    selected_columns: list[tuple[str, str]],
    summary_path: Path,
    year_summary_path: Path,
    coverage_path: Path,
    histogram_path: Path,
    boxplot_path: Path,
    total_rows: int,
    coverage_row: pd.Series,
) -> None:
    entity_label = str(coverage_row["entity_label"])
    lines = [
        f"# Scalar Distribution Report: {dataset_name}",
        "",
        f"- Source CSV: `{csv_path.resolve()}`",
        f"- Total rows: `{total_rows}`",
        f"- Selected columns: `{', '.join(column for _, column in selected_columns)}`",
        f"- Year labels: `{', '.join(label for label, _ in selected_columns)}`",
        f"- Aggregated summary CSV: `{summary_path.resolve()}`",
        f"- Per-year summary CSV: `{year_summary_path.resolve()}`",
        f"- Coverage CSV: `{coverage_path.resolve()}`",
        f"- Aggregated histogram: `{histogram_path.resolve()}`",
        f"- Per-year boxplot: `{boxplot_path.resolve()}`",
        "",
        "## Notes",
        "",
        "- Values are read directly from the CSV and coerced with `pandas.to_numeric(errors=\"coerce\")`.",
        "- Non-finite values are excluded from the plotted and summarized distributions.",
        "- The main histogram pools all selected years together into one combined distribution.",
        "- The boxplot is kept only as a supplemental cross-year comparison view.",
        "- Coverage is computed at the entity level when an identifier column such as `City`, `UC_NM_MN`, or `ID` exists.",
        "",
        "## Coverage",
        "",
        f"- Total {entity_label}: `{int(coverage_row['total_entities'])}`",
        f"- {entity_label.capitalize()} with at least one valid selected-year value: `{int(coverage_row['entities_with_any_valid'])}`",
        f"- {entity_label.capitalize()} with all selected years valid: `{int(coverage_row['entities_with_all_valid'])}`",
        f"- Mean valid years per {singularize(entity_label)}: `{float(coverage_row['mean_valid_years_per_entity']):.3f}`",
        "",
    ]
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    dataset_name = infer_dataset_name(csv_path, args.dataset_name)
    out_dir = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (Path("data/channel_distribution_report") / dataset_name).resolve()
    )

    df = pd.read_csv(csv_path)
    selected_columns = select_value_columns(
        df,
        explicit_columns=args.columns,
        start_year=args.start_year,
        end_year=args.end_year,
        year_regex=args.year_regex,
    )
    if not selected_columns:
        raise ValueError("No columns matched the requested selection.")

    summary_df = build_aggregate_summary_row(df, selected_columns, dataset_name)
    year_summary_df = build_year_summary_rows(df, selected_columns, dataset_name)
    coverage_df = build_coverage_summary(df, selected_columns, dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    year_summary_path = out_dir / "year_summary.csv"
    coverage_path = out_dir / "coverage.csv"
    histogram_path = out_dir / "histograms.png"
    boxplot_path = out_dir / "boxplot.png"

    summary_df.to_csv(summary_path, index=False)
    year_summary_df.to_csv(year_summary_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    plot_histograms(df, selected_columns, histogram_path, dataset_name, args.bins, args.dpi)
    plot_boxplot(df, selected_columns, boxplot_path, dataset_name, args.dpi)

    if args.docs_readme is not None:
        write_readme(
            readme_path=args.docs_readme.expanduser().resolve(),
            dataset_name=dataset_name,
            csv_path=csv_path,
            selected_columns=selected_columns,
            summary_path=summary_path,
            year_summary_path=year_summary_path,
            coverage_path=coverage_path,
            histogram_path=histogram_path,
            boxplot_path=boxplot_path,
            total_rows=len(df),
            coverage_row=coverage_df.iloc[0],
        )

    print(f"dataset={dataset_name}")
    print(f"summary={summary_path}")
    print(f"year_summary={year_summary_path}")
    print(f"coverage={coverage_path}")
    print(f"histograms={histogram_path}")
    print(f"boxplot={boxplot_path}")
    if args.docs_readme is not None:
        print(f"readme={args.docs_readme.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
