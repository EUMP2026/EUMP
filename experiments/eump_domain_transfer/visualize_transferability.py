#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = EXPERIMENT_DIR / "outputs" / "eump_domain_transfer_mae_v3_city_classes_multi_transfer"

MODEL_ORDER = ("swin", "scalar_mlp", "aef_mlp")
TASK_ORDER = ("lst", "precip", "pm25", "traffic")
SCENARIO_ORDER = (
    "climate_acd_to_b",
    "gdp_nonpoor_to_poor",
    "income_hic_lic_lmic_to_umic",
    "climate_abc_to_d",
    "climate_abd_to_c",
    "climate_bcd_to_a",
)
MODEL_LABELS = {
    "swin": "Swin",
    "scalar_mlp": "Scalar MLP",
    "aef_mlp": "AEF MLP",
}
TASK_LABELS = {
    "lst": "LST",
    "precip": "Precip",
    "pm25": "PM2.5",
    "traffic": "Traffic",
}
SCENARIO_LABELS = {
    "climate_acd_to_b": "Climate: A+C+D -> B",
    "gdp_nonpoor_to_poor": "GDP class: non-poor -> Poor",
    "income_hic_lic_lmic_to_umic": "Income: HIC+LIC+LMIC -> UMIC",
    "climate_abc_to_d": "Climate: A+B+C -> D",
    "climate_abd_to_c": "Climate: A+B+D -> C",
    "climate_bcd_to_a": "Climate: B+C+D -> A",
}
MODEL_COLORS = {
    "swin": "#2563eb",
    "scalar_mlp": "#059669",
    "aef_mlp": "#d97706",
}
REQUIRED_COLUMNS = (
    "scenario",
    "train_domain",
    "eval_domain",
    "task",
    "target",
    "scope",
    "rmse",
    "mae",
    "r2",
    "count",
    "city_count",
    "row_count",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transferability comparisons for EUMP domain-transfer results."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Domain transfer run directory. Default: "
            "experiments/eump_domain_transfer/outputs/"
            "eump_domain_transfer_mae_v3_city_classes_multi_transfer."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional explicit summary CSV path. Default: <run-dir>/summary.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Default: <run-dir>/figures.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_ORDER),
        help="Comma-separated model names to include. Default: swin,scalar_mlp,aef_mlp.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI. Default: 180.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format. Default: png.",
    )
    return parser.parse_args(argv)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def comma_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not items:
        raise ValueError("--models must select at least one model")
    unknown = sorted(set(items) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unsupported model(s): {', '.join(unknown)}. Supported: {', '.join(MODEL_ORDER)}")
    return tuple(model for model in MODEL_ORDER if model in items)


def setup_matplotlib():
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.environ.get("TMPDIR", "/tmp")) / f"matplotlib-cache-eump-domain-transfer-{os.getpid()}"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    return plt


def ordered_unique(values: Iterable[str], order: tuple[str, ...] = ()) -> list[str]:
    seen = {str(value) for value in values}
    ordered = [value for value in order if value in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def comparable_scope(row) -> str | None:
    model = str(row["model"])
    task = str(row["task"])
    scope = str(row["scope"])
    if model == "swin":
        expected = "city_mean" if task == "lst" else "global"
        return expected if scope == expected else None
    return "global" if scope == "global" else None


def read_summary(summary_csv: Path, *, selected_models: tuple[str, ...]) -> tuple[object, object]:
    import numpy as np
    import pandas as pd

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")
    df = pd.read_csv(summary_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"summary.csv is missing required column(s): {', '.join(missing)}")
    if "model" not in df.columns:
        df["model"] = "swin"
    if "status" not in df.columns:
        df["status"] = "ok"
    df["model"] = df["model"].fillna("swin").astype(str)
    for column in ("rmse", "mae", "r2", "count", "city_count", "row_count"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["model"].isin(selected_models)].copy()
    df = df[df["task"].isin(TASK_ORDER)].copy()
    df = df[(df["status"].fillna("ok") == "ok") & (df["count"].fillna(0) > 0)].copy()
    df = df[np.isfinite(df["rmse"])].copy()
    if df.empty:
        raise ValueError("No valid rows remained after filtering summary.csv")

    df["is_source_eval"] = df["eval_domain"].astype(str).str.startswith("source:")
    df["scope_for_compare"] = df.apply(comparable_scope, axis=1)
    comparable = df[df["scope_for_compare"].notna()].copy()
    if comparable.empty:
        raise ValueError("No comparable model rows found. Expected Swin city_mean/global and MLP global rows.")
    comparable["model_label"] = comparable["model"].map(MODEL_LABELS).fillna(comparable["model"])
    comparable["task_label"] = comparable["task"].map(TASK_LABELS).fillna(comparable["task"])
    comparable["scenario_label"] = comparable["scenario"].map(SCENARIO_LABELS).fillna(comparable["scenario"])
    comparable["eval_label"] = comparable["eval_domain"].astype(str).str.replace("source:", "source:", regex=False)
    comparable["comparison_key"] = (
        comparable["scenario"].astype(str)
        + " | "
        + comparable["task"].astype(str)
        + " | "
        + comparable["eval_domain"].astype(str)
    )
    return df, comparable


def build_transfer_table(comparable):
    import numpy as np
    import pandas as pd

    keys = ["model", "scenario", "task"]
    source = comparable[comparable["is_source_eval"]].copy()
    target = comparable[~comparable["is_source_eval"]].copy()
    if source.empty:
        transfer = target.copy()
        for column in ("source_rmse", "source_r2", "rmse_ratio", "rmse_delta", "r2_drop"):
            transfer[column] = np.nan
        return transfer

    source_metrics = (
        source.groupby(keys, as_index=False)
        .agg(
            source_rmse=("rmse", "mean"),
            source_mae=("mae", "mean"),
            source_r2=("r2", "mean"),
            source_count=("count", "sum"),
        )
    )
    transfer = target.merge(source_metrics, on=keys, how="left")
    transfer["rmse_ratio"] = transfer["rmse"] / transfer["source_rmse"].where(transfer["source_rmse"] > 0)
    transfer["mae_ratio"] = transfer["mae"] / transfer["source_mae"].where(transfer["source_mae"] > 0)
    transfer["rmse_delta"] = transfer["rmse"] - transfer["source_rmse"]
    transfer["r2_drop"] = transfer["source_r2"] - transfer["r2"]
    transfer["comparison"] = (
        transfer["scenario"].map(SCENARIO_LABELS).fillna(transfer["scenario"])
        + "\n"
        + transfer["task"].map(TASK_LABELS).fillna(transfer["task"])
        + " -> "
        + transfer["eval_domain"].astype(str)
    )
    return transfer


def build_rank_tables(transfer):
    import pandas as pd

    rank_rows = transfer.dropna(subset=["rmse", "r2"]).copy()
    if rank_rows.empty:
        return rank_rows, pd.DataFrame()
    group_keys = ["scenario", "task", "eval_domain"]
    rank_rows["available_model_count"] = rank_rows.groupby(group_keys)["model"].transform("nunique")
    rank_rows = rank_rows[rank_rows["available_model_count"] >= 2].copy()
    if rank_rows.empty:
        return rank_rows, pd.DataFrame()
    rank_rows["rmse_rank"] = rank_rows.groupby(group_keys)["rmse"].rank(method="average", ascending=True)
    rank_rows["r2_rank"] = rank_rows.groupby(group_keys)["r2"].rank(method="average", ascending=False)
    best_rmse = rank_rows["rmse_rank"] == 1.0
    best_r2 = rank_rows["r2_rank"] == 1.0
    rank_rows["best_rmse"] = best_rmse.astype(int)
    rank_rows["best_r2"] = best_r2.astype(int)
    summary = (
        rank_rows.groupby("model", as_index=False)
        .agg(
            comparisons=("comparison_key", "count"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_rmse_ratio=("rmse_ratio", "mean"),
            mean_r2_drop=("r2_drop", "mean"),
            mean_rmse_rank=("rmse_rank", "mean"),
            mean_r2_rank=("r2_rank", "mean"),
            best_rmse_count=("best_rmse", "sum"),
            best_r2_count=("best_r2", "sum"),
        )
    )
    summary["model_label"] = summary["model"].map(MODEL_LABELS).fillna(summary["model"])
    return rank_rows, summary


def save_table(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def grouped_bar(ax, data, *, x_col: str, y_col: str, models: tuple[str, ...], title: str, ylabel: str) -> None:
    import numpy as np

    x_values = ordered_unique(data[x_col].astype(str))
    if not x_values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    x = np.arange(len(x_values), dtype=float)
    width = 0.78 / max(1, len(models))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2.0) * width
    for offset, model in zip(offsets, models):
        model_rows = data[data["model"] == model]
        values = []
        for x_value in x_values:
            subset = model_rows[model_rows[x_col].astype(str) == x_value]
            values.append(float(subset[y_col].mean()) if not subset.empty else math.nan)
        ax.bar(
            x + offset,
            values,
            width=width,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#6b7280"),
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=25, ha="right")
    ax.margins(x=0.02)


def plot_metric_grid(plt, transfer, *, metric: str, ylabel: str, out_path: Path, models: tuple[str, ...], dpi: int) -> None:
    scenarios = ordered_unique(transfer["scenario"], SCENARIO_ORDER)
    tasks = ordered_unique(transfer["task"], TASK_ORDER)
    fig, axes = plt.subplots(
        len(tasks),
        len(scenarios),
        figsize=(5.4 * max(1, len(scenarios)), 3.4 * max(1, len(tasks))),
        squeeze=False,
        constrained_layout=True,
    )
    for row_idx, task in enumerate(tasks):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx][col_idx]
            subset = transfer[(transfer["task"] == task) & (transfer["scenario"] == scenario)]
            title = f"{TASK_LABELS.get(task, task)} | {SCENARIO_LABELS.get(scenario, scenario)}"
            grouped_bar(ax, subset, x_col="eval_domain", y_col=metric, models=models, title=title, ylabel=ylabel)
            if row_idx == 0 and col_idx == len(scenarios) - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(f"Target-domain {ylabel} by model", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    plt,
    table,
    *,
    value_col: str,
    out_path: Path,
    title: str,
    colorbar_label: str,
    fmt: str,
    cmap: str,
    dpi: int,
) -> None:
    import numpy as np

    if table.empty or value_col not in table.columns:
        return
    pivot = table.pivot_table(index="model", columns="comparison", values=value_col, aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.reindex([model for model in MODEL_ORDER if model in pivot.index])
    values = pivot.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig_width = max(8.0, 0.65 * max(1, values.shape[1]))
    fig_height = max(2.7, 0.75 * max(1, values.shape[0]) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#f3f4f6")
    image = ax.imshow(masked, aspect="auto", cmap=colormap)
    ax.set_title(title)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([MODEL_LABELS.get(model, model) for model in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if math.isfinite(value):
                ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(image, ax=ax, shrink=0.82)
    cbar.set_label(colorbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rank_summary(plt, rank_summary, *, out_path: Path, models: tuple[str, ...], dpi: int) -> None:
    import numpy as np

    if rank_summary.empty:
        return
    rows = rank_summary.set_index("model").reindex([model for model in models if model in set(rank_summary["model"])])
    x = np.arange(len(rows), dtype=float)
    labels = [MODEL_LABELS.get(model, model) for model in rows.index]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
    axes[0].bar(x, rows["mean_rmse_rank"], color=[MODEL_COLORS.get(model, "#6b7280") for model in rows.index])
    axes[0].set_title("Mean RMSE rank")
    axes[0].set_ylabel("Lower is better")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1].bar(x, rows["mean_rmse_ratio"], color=[MODEL_COLORS.get(model, "#6b7280") for model in rows.index])
    axes[1].axhline(1.0, color="#374151", linewidth=1.0, linestyle="--")
    axes[1].set_title("Mean target/source RMSE")
    axes[1].set_ylabel("Lower is better")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    axes[2].bar(x, rows["best_rmse_count"], color=[MODEL_COLORS.get(model, "#6b7280") for model in rows.index])
    axes[2].set_title("Best RMSE count")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")

    fig.suptitle("Cross-domain transferability summary", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_summary_by_scenario(plt, transfer, *, out_path: Path, models: tuple[str, ...], dpi: int) -> None:
    import numpy as np

    if transfer.empty:
        return
    summary = (
        transfer.groupby(["scenario", "model"], as_index=False)
        .agg(mean_rmse_ratio=("rmse_ratio", "mean"), mean_r2_drop=("r2_drop", "mean"))
    )
    scenarios = ordered_unique(summary["scenario"], SCENARIO_ORDER)
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.4 * max(1, len(scenarios)), 3.8),
        squeeze=False,
        constrained_layout=True,
    )
    for idx, scenario in enumerate(scenarios):
        ax = axes[0][idx]
        subset = summary[summary["scenario"] == scenario]
        x = np.arange(len(models), dtype=float)
        ratios = [
            float(subset[subset["model"] == model]["mean_rmse_ratio"].mean())
            if not subset[subset["model"] == model].empty
            else math.nan
            for model in models
        ]
        ax.bar(x, ratios, color=[MODEL_COLORS.get(model, "#6b7280") for model in models])
        ax.axhline(1.0, color="#374151", linewidth=1.0, linestyle="--")
        ax.set_title(SCENARIO_LABELS.get(scenario, scenario))
        ax.set_ylabel("Mean target/source RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(model, model) for model in models], rotation=20, ha="right")
    fig.suptitle("Transfer degradation by scenario", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df, columns: list[str], *, float_digits: int = 4) -> list[str]:
    if df.empty:
        return ["No rows available."]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append("nan" if math.isnan(value) else f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    *,
    out_path: Path,
    summary_csv: Path,
    comparable,
    transfer,
    rank_rows,
    rank_summary,
    figure_paths: list[Path],
) -> None:
    lines: list[str] = [
        "# EUMP Domain Transferability Report",
        "",
        f"- Source summary: `{summary_csv}`",
        f"- Comparable metric rows: {len(comparable)}",
        f"- Target-domain transfer rows: {len(transfer)}",
        "- Cross-model comparison scope: Swin `city_mean` for LST, Swin `global` for scalar Precip/PM2.5/Traffic, MLP/AEF `global`.",
        "- Lower RMSE, lower target/source RMSE ratio, lower RMSE rank, and smaller R2 drop indicate stronger transferability.",
        "",
    ]
    if not rank_summary.empty:
        table = rank_summary.copy()
        table = table.sort_values(["mean_rmse_rank", "mean_rmse_ratio"])
        table["model"] = table["model"].map(MODEL_LABELS).fillna(table["model"])
        lines.extend(["## Overall Rank", ""])
        lines.extend(
            markdown_table(
                table,
                [
                    "model",
                    "comparisons",
                    "mean_rmse_rank",
                    "mean_rmse_ratio",
                    "mean_r2_drop",
                    "best_rmse_count",
                ],
            )
        )
        lines.append("")

    if not transfer.empty:
        scenario_summary = (
            transfer.groupby(["scenario", "model"], as_index=False)
            .agg(
                comparisons=("comparison_key", "count"),
                mean_rmse_ratio=("rmse_ratio", "mean"),
                mean_r2_drop=("r2_drop", "mean"),
                mean_rmse=("rmse", "mean"),
                mean_r2=("r2", "mean"),
            )
            .sort_values(["scenario", "mean_rmse_ratio"])
        )
        scenario_summary["scenario"] = scenario_summary["scenario"].map(SCENARIO_LABELS).fillna(scenario_summary["scenario"])
        scenario_summary["model"] = scenario_summary["model"].map(MODEL_LABELS).fillna(scenario_summary["model"])
        lines.extend(["## Scenario Summary", ""])
        lines.extend(
            markdown_table(
                scenario_summary,
                ["scenario", "model", "comparisons", "mean_rmse_ratio", "mean_r2_drop", "mean_rmse", "mean_r2"],
            )
        )
        lines.append("")

    complete_count = 0
    if not rank_rows.empty:
        complete_count = int((rank_rows.groupby(["scenario", "task", "eval_domain"])["model"].nunique() == 3).sum())
    lines.extend(
        [
            "## Notes",
            "",
            f"- Complete three-model comparison groups: {complete_count}",
            "- If a method is missing for a target domain, it is omitted from ranks for that group.",
            "- `source:*` rows are used only to compute transfer degradation, not as target-domain comparison rows.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in figure_paths:
        lines.append(f"- `{path}`")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve_path(args.run_dir)
    summary_csv = resolve_path(args.summary_csv) if args.summary_csv is not None else run_dir / "summary.csv"
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else run_dir / "figures"
    selected_models = comma_list(args.models)
    plt = setup_matplotlib()

    raw, comparable = read_summary(summary_csv, selected_models=selected_models)
    transfer = build_transfer_table(comparable)
    rank_rows, rank_summary = build_rank_tables(transfer)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_table(raw, out_dir / "valid_metric_rows.csv")
    save_table(comparable, out_dir / "comparable_metrics.csv")
    save_table(transfer, out_dir / "transferability_table.csv")
    save_table(rank_rows, out_dir / "transferability_rank_rows.csv")
    save_table(rank_summary, out_dir / "transferability_rank_summary.csv")

    suffix = args.format
    figure_paths: list[Path] = []
    rmse_path = out_dir / f"target_domain_rmse_by_model.{suffix}"
    r2_path = out_dir / f"target_domain_r2_by_model.{suffix}"
    ratio_path = out_dir / f"transfer_rmse_ratio_heatmap.{suffix}"
    r2_drop_path = out_dir / f"transfer_r2_drop_heatmap.{suffix}"
    rank_path = out_dir / f"transferability_rank_summary.{suffix}"
    scenario_path = out_dir / f"transfer_degradation_by_scenario.{suffix}"

    plot_metric_grid(plt, transfer, metric="rmse", ylabel="RMSE", out_path=rmse_path, models=selected_models, dpi=args.dpi)
    figure_paths.append(rmse_path)
    plot_metric_grid(plt, transfer, metric="r2", ylabel="R2", out_path=r2_path, models=selected_models, dpi=args.dpi)
    figure_paths.append(r2_path)
    plot_heatmap(
        plt,
        transfer,
        value_col="rmse_ratio",
        out_path=ratio_path,
        title="Target/source RMSE ratio",
        colorbar_label="RMSE ratio",
        fmt=".2f",
        cmap="YlOrRd",
        dpi=args.dpi,
    )
    if ratio_path.exists():
        figure_paths.append(ratio_path)
    plot_heatmap(
        plt,
        transfer,
        value_col="r2_drop",
        out_path=r2_drop_path,
        title="R2 drop from source to target",
        colorbar_label="source R2 - target R2",
        fmt=".2f",
        cmap="PuRd",
        dpi=args.dpi,
    )
    if r2_drop_path.exists():
        figure_paths.append(r2_drop_path)
    plot_rank_summary(plt, rank_summary, out_path=rank_path, models=selected_models, dpi=args.dpi)
    if rank_path.exists():
        figure_paths.append(rank_path)
    plot_transfer_summary_by_scenario(plt, transfer, out_path=scenario_path, models=selected_models, dpi=args.dpi)
    if scenario_path.exists():
        figure_paths.append(scenario_path)

    report_path = out_dir / "transferability_report.md"
    write_report(
        out_path=report_path,
        summary_csv=summary_csv,
        comparable=comparable,
        transfer=transfer,
        rank_rows=rank_rows,
        rank_summary=rank_summary,
        figure_paths=figure_paths,
    )

    print(f"Read summary: {summary_csv}")
    print(f"Comparable rows: {len(comparable)}")
    print(f"Transfer rows: {len(transfer)}")
    print(f"Saved figures and tables: {out_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
