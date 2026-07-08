"""Main figure 2 (v2): downstream prediction quality scatter grid.

3 model rows x 4 task columns density scatter of observed vs predicted
(physical units), read entirely from ``fig2_data.npz`` (produced by
``collect_data.py`` on the server). No repo / server dependencies -- runs
anywhere with numpy + scipy + matplotlib.

Rows:    EUMP (frozen) / NN (one-stage) / Hand (scalar MLP)
Columns: LST / Precip / TCI / PM2.5
Each cell: KDE-coloured scatter, 1:1 line, OLS fit, R2 / RMSE annotation.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress, gaussian_kde

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -------- style (matches main_figs_v1/fig2) --------
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


def draw_cell(ax, y, yhat, *, title_r2, title_rmse):
    n = min(len(y), len(yhat))
    y, yhat = y[:n], yhat[:n]
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]

    dmin = float(min(y.min(), yhat.min()))
    dmax = float(max(y.max(), yhat.max()))
    span = (dmax - dmin) if dmax > dmin else 1.0
    lo, hi = dmin - 0.1 * span, dmax + 0.1 * span

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
    ax.set_aspect("equal", "box")
    ax.text(0.04, 0.98, f"R²={title_r2:.3f}", transform=ax.transAxes, fontsize=5.5, va="top")
    ax.text(0.04, 0.90, f"RMSE={title_rmse:.3g}", transform=ax.transAxes, fontsize=5.5, va="top")
    ax.tick_params(labelsize=5)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_linewidth(0.5)


# %% -------------------------------------------------------------------------
# Diagnostic cell: EUMP observation / bias / bias-% distributions
# ---------------------------------------------------------------------------
# Before the main scatter grid, summarise the EUMP prediction-error structure
# across the four downstream tasks (LST, Precip, TCI, PM2.5). For each task:
#   * histogram of observations (obs)
#   * histogram of bias            = prediction - observation
#   * histogram of bias percentage = (prediction - observation) / observation * 100
#   * histogram of |bias| %        = |prediction - observation| / |observation| * 100
# and report mean / median / std of bias, bias_pct and |bias|_pct
# (obs stats included too).
# Runs standalone in the VS Code Interactive Window.

_DIAG_MODEL_KEY = "eump"
_DIAG_TASKS = [("lst", "LST"), ("precip", "Precip"), ("tci", "TCI"), ("pm25", "PM2.5")]
_BIAS_PCT_EPS = 1e-6   # guard obs ~ 0 (precip has exact zeros) before dividing


def _summ(x):
    """mean / median / std over finite values."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan)
    return dict(n=int(x.size), mean=float(np.mean(x)),
                median=float(np.median(x)), std=float(np.std(x)))


def diagnose_eump_errors():
    _, _, _, _, _, arrays = load_cells()
    stats_rows = []
    nrows, ncols = len(_DIAG_TASKS), 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.2))
    axes = np.atleast_2d(axes)

    for r, (task_key, task_label) in enumerate(_DIAG_TASKS):
        y, yhat = arrays[(task_key, _DIAG_MODEL_KEY)]
        y = np.asarray(y, dtype=float)
        yhat = np.asarray(yhat, dtype=float)
        n = min(y.size, yhat.size)
        y, yhat = y[:n], yhat[:n]
        finite = np.isfinite(y) & np.isfinite(yhat)
        y, yhat = y[finite], yhat[finite]

        bias = yhat - y
        nz = np.abs(y) > _BIAS_PCT_EPS
        bias_pct = np.full_like(bias, np.nan)
        bias_pct[nz] = (yhat[nz] - y[nz]) / y[nz] * 100.0   # percent
        abs_bias_pct = np.full_like(bias, np.nan)           # |bias| / |obs| (%)
        abs_bias_pct[nz] = np.abs(yhat[nz] - y[nz]) / np.abs(y[nz]) * 100.0

        s_obs, s_bias = _summ(y), _summ(bias)
        s_pct, s_abs = _summ(bias_pct), _summ(abs_bias_pct)
        unit = TASK_UNITS.get(task_label, "")
        stats_rows.append((task_label, unit, s_obs, s_bias, s_pct, s_abs, int(np.sum(~nz))))

        # --- col 0: observations -------------------------------------------
        ax = axes[r, 0]
        ax.hist(y, bins=40, color="#3a8bbf", alpha=0.85)
        ax.set_ylabel(f"{task_label}\ncount", fontsize=8)
        if r == 0:
            ax.set_title("Observations", fontsize=9)
        ax.text(0.97, 0.95,
                f"mean={s_obs['mean']:.4g}\nmed={s_obs['median']:.4g}\nstd={s_obs['std']:.4g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlabel(f"obs ({unit})" if unit else "obs", fontsize=7)

        # --- col 1: bias = pred - obs --------------------------------------
        ax = axes[r, 1]
        ax.hist(bias, bins=40, color="#c0504d", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        if r == 0:
            ax.set_title("Bias (pred - obs)", fontsize=9)
        ax.text(0.97, 0.95,
                f"mean={s_bias['mean']:.4g}\nmed={s_bias['median']:.4g}\nstd={s_bias['std']:.4g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlabel(f"bias ({unit})" if unit else "bias", fontsize=7)

        # --- col 2: bias percentage = (pred - obs)/obs * 100 ---------------
        ax = axes[r, 2]
        bp = bias_pct[np.isfinite(bias_pct)]
        # clip display to robust [1, 99] pct range; stats above use full data
        if bp.size:
            lo, hi = np.percentile(bp, [1, 99])
            bp_disp = bp[(bp >= lo) & (bp <= hi)]
        else:
            bp_disp = bp
        ax.hist(bp_disp, bins=40, color="#9b59b6", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8, ls="--")
        if r == 0:
            ax.set_title("Bias %  (pred-obs)/obs", fontsize=9)
        ax.text(0.97, 0.95,
                f"mean={s_pct['mean']:.4g}%\nmed={s_pct['median']:.4g}%\nstd={s_pct['std']:.4g}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlabel("bias (%)", fontsize=7)

        # --- col 3: absolute relative bias = |pred - obs|/|obs| * 100 ------
        ax = axes[r, 3]
        ab = abs_bias_pct[np.isfinite(abs_bias_pct)]
        # clip display to robust [0, 99] pct range; stats above use full data
        if ab.size:
            hi = np.percentile(ab, 99)
            ab_disp = ab[ab <= hi]
        else:
            ab_disp = ab
        ax.hist(ab_disp, bins=40, color="#27ae60", alpha=0.85)
        if r == 0:
            ax.set_title("|Bias| %  |pred-obs|/|obs|", fontsize=9)
        ax.text(0.97, 0.95,
                f"mean={s_abs['mean']:.4g}%\nmed={s_abs['median']:.4g}%\nstd={s_abs['std']:.4g}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.set_xlabel("|bias| (%)", fontsize=7)

    fig.suptitle("EUMP prediction-error diagnostics", fontsize=11, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig("fig2_eump_error_diagnostics.png", dpi=300, bbox_inches="tight")
    print("Saved: fig2_eump_error_diagnostics.png")

    # --- console summary table ---------------------------------------------
    print("\nEUMP error summary  (bias = pred - obs ; pct in % ; |pct| = |bias|/|obs|)")
    print(f"{'task':6s} {'n':>5s} {'obs_mean':>9s} {'bias_mean':>9s} {'bias_med':>9s} "
          f"{'bias_std':>9s} {'pct_mean':>8s} {'pct_med':>8s} {'pct_std':>8s} "
          f"{'|pct|_mean':>10s} {'|pct|_med':>9s} {'|pct|_std':>9s} {'obs0':>5s}")
    for label, unit, s_obs, s_bias, s_pct, s_abs, nzero in stats_rows:
        print(f"{label:6s} {s_bias['n']:5d} {s_obs['mean']:9.4g} {s_bias['mean']:9.4g} "
              f"{s_bias['median']:9.4g} {s_bias['std']:9.4g} {s_pct['mean']:8.4g} "
              f"{s_pct['median']:8.4g} {s_pct['std']:8.4g} {s_abs['mean']:10.4g} "
              f"{s_abs['median']:9.4g} {s_abs['std']:9.4g} {nzero:5d}")
    return stats_rows


_diag_stats = diagnose_eump_errors()


# %% -------------------------------------------------------------------------
# Main figure 2 scatter grid
# ---------------------------------------------------------------------------
def main():
    tasks, models, task_keys, model_keys, by_cell, arrays = load_cells()
    # Normalize row labels at render time so a stale npz (baked before the
    # rename) still displays the current names without a server-side regen.
    label_remap = {"EUMP-E2E": "NN"}
    models = [label_remap.get(m, m) for m in models]
    ncols, nrows = len(tasks), len(models)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2 / 2.54, nrows * 4.2 / 2.54),
        dpi=600,
    )
    axes = np.atleast_2d(axes)

    for r, (model_label, model_key) in enumerate(zip(models, model_keys)):
        for c, (task_label, task_key) in enumerate(zip(tasks, task_keys)):
            ax = axes[r, c]
            cell = by_cell.get((task_key, model_key))
            if cell is None:
                ax.set_visible(False)
                continue
            y, yhat = arrays[(task_key, model_key)]
            draw_cell(ax, y, yhat, title_r2=float(cell["r2"]), title_rmse=float(cell["rmse"]))

            # Column titles (task) on top row.
            if r == 0:
                ax.set_title(task_label, fontsize=7, weight="bold", pad=4)
            # Row labels (model) on left column.
            if c == 0:
                ax.text(-0.32, 0.5, model_label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=7, weight="bold")
                ax.set_ylabel("Predicted", fontsize=6)
            else:
                ax.set_yticklabels([])
            # X labels only on bottom row.
            if r == nrows - 1:
                unit = TASK_UNITS.get(task_label, "")
                ax.set_xlabel(f"Observed ({unit})" if unit else "Observed", fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.subplots_adjust(hspace=0.18, wspace=0.08, left=0.09, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(OUT_FILE, dpi=600, bbox_inches="tight")
    print("Saved:", OUT_FILE)


if __name__ == "__main__":
    main()
