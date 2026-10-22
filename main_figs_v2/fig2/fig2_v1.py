"""Main figure 2 (v2): downstream prediction quality scatter grid.

3 model rows x 4 task columns density scatter of observed vs predicted
(physical units), read entirely from ``fig2_data.npz`` (produced by
``collect_data.py`` on the server). No repo / server dependencies -- runs
anywhere with numpy + scipy + matplotlib.

Rows:    EUMP (frozen) / Supervised NN (one-stage) / Hand (scalar MLP)
Columns: LST / Precip / TCI / PM2.5
Each cell: KDE-coloured scatter, 1:1 line, OLS fit, R2 / RMSE annotation.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress, gaussian_kde

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -------- style --------
plt.rcParams.update({"font.family": "Arial", "font.size": 7})
_DENSITY_COLORS = [
    (0.00, 0.19, 0.57),  # deep blue (low density)
    (0.45, 0.68, 0.85),  # light blue
    (0.95, 0.95, 0.95),  # white
    (0.92, 0.52, 0.36),  # light red
    (0.70, 0.00, 0.00),  # deep red (high density)
]
SC_CMAP = LinearSegmentedColormap.from_list("temp_change", _DENSITY_COLORS, N=256)

# Column units for axis labels.
TASK_UNITS = {"LST": "K", "Precip": "mm", "TCI": "", "PM2.5": "ug/m3"}

DATA_FILE = "fig2_data.npz"
OUT_FILE = "fig2.png"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load_cells():
    d = np.load(DATA_FILE)
    metas = json.loads(str(d["meta_json"]))
    tasks = [str(x) for x in d["task_labels"]]      # display order, columns
    models = [str(x) for x in d["model_labels"]]    # display order, rows
    task_keys = [str(x) for x in d["tasks"]]
    model_keys = [str(x) for x in d["models"]]
    by_cell = {(m["task"], m["model"]): m for m in metas}
    arrays = {
        (m["task"], m["model"]): (
            np.asarray(d[f"{m['task']}__{m['model']}__y"], dtype=float),
            np.asarray(d[f"{m['task']}__{m['model']}__yhat"], dtype=float),
        )
        for m in metas
    }
    return tasks, models, task_keys, model_keys, by_cell, arrays


def cell_finite(y, yhat):
    """Trim to equal length and drop non-finite pairs."""
    n = min(len(y), len(yhat))
    y, yhat = y[:n], yhat[:n]
    mask = np.isfinite(y) & np.isfinite(yhat)
    return y[mask], yhat[mask]


def axis_limits(y, yhat):
    """Padded [lo, hi] spanning both observed and predicted."""
    dmin = float(min(y.min(), yhat.min()))
    dmax = float(max(y.max(), yhat.max()))
    span = (dmax - dmin) if dmax > dmin else 1.0
    return dmin - 0.1 * span, dmax + 0.1 * span


def column_ticks(lo, hi, nbins=5):
    """One shared tick set for both axes of every panel in a column.

    Same count and values on x and y -- the panels are square (xlim==ylim),
    so identical ticks make the 1:1 line land on matching gridlines.
    """
    ticks = MaxNLocator(nbins=nbins, prune=None).tick_values(lo, hi)
    return ticks[(ticks >= lo) & (ticks <= hi)]


def draw_cell(ax, y, yhat, *, title_r2, title_rmse, lo=None, hi=None, ticks=None):
    y, yhat = cell_finite(y, yhat)

    # Axis limits default to this cell's own range, but callers pass shared
    # per-column limits so every model row of a task uses the same axes
    # (otherwise a model that overshoots stretches its own panel and the
    # clouds no longer line up down the column).
    if lo is None or hi is None:
        lo, hi = axis_limits(y, yhat)

    # OLS fit and density colouring (guard tiny / degenerate samples).
    x_dense = np.linspace(lo, hi, 200)
    try:
        slope, intercept, *_ = linregress(y, yhat)
        y_fit = slope * x_dense + intercept
    except Exception:
        y_fit = None
    try:
        z = gaussian_kde(np.vstack([y, yhat]))(np.vstack([y, yhat]))
        order = z.argsort()
    except Exception:
        z = np.zeros_like(y)
        order = np.arange(len(y))

    ax.scatter(
        y[order], yhat[order], c=z[order], s=4, cmap=SC_CMAP,
        edgecolor="w", linewidth=0.01, alpha=0.6,
    )
    ax.plot(x_dense, x_dense, "--", color="k", lw=1)
    if y_fit is not None:
        ax.plot(x_dense, y_fit, "-", color="r", lw=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    ax.set_aspect("equal", "box")
    ax.text(0.04, 0.98, f"R²={title_r2:.3f}", transform=ax.transAxes, fontsize=5.5, va="top")
    ax.text(0.04, 0.90, f"RMSE={title_rmse:.3g}", transform=ax.transAxes, fontsize=5.5, va="top")
    ax.tick_params(labelsize=5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_linewidth(0.5)


def main():
    tasks, models, task_keys, model_keys, by_cell, arrays = load_cells()
    # Normalize row labels at render time so a stale npz (baked before the
    # rename) still displays the current names without a server-side regen.
    label_remap = {"EUMP-E2E": "Supervised NN"}
    models = [label_remap.get(m, m) for m in models]
    ncols, nrows = len(tasks), len(models)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(18 / 2.54, 13.6 / 2.54),
        dpi=600,
    )
    axes = np.atleast_2d(axes)

    # Shared axis limits per task column: span every model row so the observed
    # clouds align down the column regardless of which model overshoots.
    col_limits: dict[str, tuple[float, float]] = {}
    for task_key in task_keys:
        los, his = [], []
        for model_key in model_keys:
            if (task_key, model_key) not in arrays:
                continue
            y, yhat = cell_finite(*arrays[(task_key, model_key)])
            if not len(y):
                continue
            lo, hi = axis_limits(y, yhat)
            los.append(lo)
            his.append(hi)
        if los:
            col_limits[task_key] = (min(los), max(his))

    # One shared tick set per column (identical count + values on x and y).
    col_ticks = {tk: column_ticks(lo, hi) for tk, (lo, hi) in col_limits.items()}

    for r, (model_label, model_key) in enumerate(zip(models, model_keys)):
        for c, (task_label, task_key) in enumerate(zip(tasks, task_keys)):
            ax = axes[r, c]
            cell = by_cell.get((task_key, model_key))
            if cell is None:
                ax.set_visible(False)
                continue
            y, yhat = arrays[(task_key, model_key)]
            lo, hi = col_limits.get(task_key, (None, None))
            draw_cell(ax, y, yhat, title_r2=float(cell["r2"]), title_rmse=float(cell["rmse"]),
                      lo=lo, hi=hi, ticks=col_ticks.get(task_key))

            # Column titles (task) on top row.
            if r == 0:
                ax.set_title(task_label, fontsize=7, weight="bold", pad=4)
            # Row labels (model) on left column; "Predicted" axis label too.
            if c == 0:
                ax.text(-0.32, 0.5, model_label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=7, weight="bold")
                ax.set_ylabel("Predicted", fontsize=6)
            # Y-tick numbers stay on every panel: each column has its own scale
            # (LST K vs Precip mm vs TCI vs PM2.5), so blanking columns 2-4 would
            # invite reading column 1's ticks against the wrong axis.
            # X labels only on bottom row.
            if r == nrows - 1:
                unit = TASK_UNITS.get(task_label, "")
                ax.set_xlabel(f"Observed ({unit})" if unit else "Observed", fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.subplots_adjust(hspace=0.15, wspace=0.21, left=0.09, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight")
    print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()
