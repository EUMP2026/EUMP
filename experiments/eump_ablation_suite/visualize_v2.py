#!/usr/bin/env python
"""Visualize ablation suite v2 results from summary.csv.

Usage:
    python experiments/eump_ablation_suite/visualize_v2.py [--csv PATH] [--output DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "experiments/eump_ablation_suite/outputs/eump_ablation_v2_full/summary.csv"
DEFAULT_OUTPUT = REPO_ROOT / "experiments/eump_ablation_suite/outputs/eump_ablation_v2_full/figures"

TASK_DISPLAY = {"lst": "LST", "precip": "Precip", "traffic": "Traffic", "pm25": "PM2.5"}
METRIC_LABEL = {"rmse": "RMSE", "r2": "R²"}

GROUPS = [
    {
        "name": "Baselines & Architecture",
        "variants": ["mae_v3", "noaux", "scratch_init", "swin_t", "swin_l_unfrozen", "swin_unet", "convnext_unet", "simclr", "moco"],
        "color": "#2c7bb6",
        "marker": "o",
    },
    {
        "name": "Mask Ratio (center, no aux)",
        "variants": [
            "center1_mask0_noaux",
            "center1_mask20_noaux",
            "center1_mask40_noaux",
            "center1_mask60_noaux",
            "center1_mask80_noaux",
        ],
        "color": "#d7191c",
        "marker": "s",
    },
    {
        "name": "Input Groups",
        "variants": ["group_1", "group_2", "group_3", "group_4", "group_5", "group_6"],
        "color": "#1a9641",
        "marker": "^",
    },
]

VARIANT_SHORT = {
    "mae_v3": "MAE v3",
    "noaux": "No Aux",
    "scratch_init": "Scratch",
    "swin_t": "Swin-T",
    "swin_l_unfrozen": "One-Stage",
    "swin_unet": "Swin-UNet",
    "convnext_unet": "ConvNeXt-UNet",
    "simclr": "SimCLR",
    "moco": "MoCo",
    "center1_mask0_noaux": "mask=0%",
    "center1_mask20_noaux": "mask=20%",
    "center1_mask40_noaux": "mask=40%",
    "center1_mask60_noaux": "mask=60%",
    "center1_mask80_noaux": "mask=80%",
    "group_1": "G1",
    "group_2": "G2",
    "group_3": "G3",
    "group_4": "G4",
    "group_5": "G5",
    "group_6": "G6",
}


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "test"].copy()
    lst_mask = (df["task"] == "lst") & (df["scope"] != "city_mean")
    df = df[~lst_mask]
    return df


def _get_variant_value(df: pd.DataFrame, variant: str, task: str, col: str):
    mask = (df["variant"] == variant) & (df["task"] == task)
    rows = df[mask]
    if rows.empty:
        return np.nan
    return rows.iloc[0][col]


def plot_task(df: pd.DataFrame, task: str, metric: str, output_dir: Path) -> None:
    task_df = df[df["task"] == task]
    if task_df.empty:
        return

    higher_better = metric == "r2"
    std_col = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(12, 4.5))
    offset = 0
    tick_positions = []
    tick_labels = []
    group_spans = []

    for gi, group in enumerate(GROUPS):
        variants_in_data = [v for v in group["variants"] if v in task_df["variant"].values]
        if not variants_in_data:
            continue

        start_x = offset
        xs, ys, errs = [], [], []
        for v in variants_in_data:
            val = _get_variant_value(task_df, v, task, metric)
            std = _get_variant_value(task_df, v, task, std_col)
            if np.isnan(val):
                continue
            xs.append(offset)
            ys.append(val)
            errs.append(std if not np.isnan(std) else 0.0)
            tick_positions.append(offset)
            tick_labels.append(VARIANT_SHORT.get(v, v))
            offset += 1

        if not xs:
            continue

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        errs_arr = np.array(errs)

        ax.plot(xs_arr, ys_arr, color=group["color"], marker=group["marker"],
                linewidth=1.5, markersize=7, label=group["name"], zorder=3)
        ax.fill_between(xs_arr, ys_arr - errs_arr, ys_arr + errs_arr,
                        color=group["color"], alpha=0.12, zorder=2)

        best_idx = int(np.argmax(ys_arr) if higher_better else np.argmin(ys_arr))
        ax.plot(xs_arr[best_idx], ys_arr[best_idx], marker="*",
                color=group["color"], markersize=16, zorder=5,
                markeredgecolor="black", markeredgewidth=0.6)

        group_spans.append((start_x, offset - 1, group["name"]))
        offset += 1

    for start, end, _ in group_spans[:-1]:
        sep_x = end + 0.5
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(f"{TASK_DISPLAY.get(task, task)} — {METRIC_LABEL.get(metric, metric)} (test set)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    vals = df[df["task"] == task][metric].dropna()
    if not vals.empty:
        margin = (vals.max() - vals.min()) * 0.15
        ax.set_ylim(vals.min() - margin, vals.max() + margin)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{task}_{metric}_test.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_overview_table(df: pd.DataFrame, output_dir: Path) -> None:
    tasks = [t for t in ["lst", "precip", "traffic", "pm25"] if t in df["task"].unique()]
    all_variants = []
    for group in GROUPS:
        for v in group["variants"]:
            if v in df["variant"].values:
                all_variants.append((group["name"], v))

    rows = []
    for gname, v in all_variants:
        entry = {"group": gname, "variant": VARIANT_SHORT.get(v, v)}
        for task in tasks:
            rmse = _get_variant_value(df, v, task, "rmse")
            r2 = _get_variant_value(df, v, task, "r2")
            entry[f"{task}_rmse"] = f"{rmse:.4f}" if not np.isnan(rmse) else ""
            entry[f"{task}_r2"] = f"{r2:.3f}" if not np.isnan(r2) else ""
        rows.append(entry)

    result_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "overview_test.csv", index=False)
    print(f"  overview_test.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: {args.csv} not found")
        raise SystemExit(1)

    df = load_and_filter(args.csv)
    if df.empty:
        print("No test-set results found in summary.csv")
        raise SystemExit(0)

    print(f"Loaded {len(df)} test-set rows from {args.csv.name}")
    tasks = [t for t in ["lst", "precip", "traffic", "pm25"] if t in df["task"].unique()]

    for task in tasks:
        for metric in ("rmse", "r2"):
            plot_task(df, task, metric, args.output)

    plot_overview_table(df, args.output)
    print(f"\nFigures: {args.output}")


if __name__ == "__main__":
    main()
