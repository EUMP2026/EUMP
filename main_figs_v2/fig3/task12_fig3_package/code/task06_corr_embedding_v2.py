"""
Task 06 — v2 redesign of corr_plot_fig1.

Objective
    Re-plot the embedding-to-variable max-|r| summary (fig1 only) with
    a top-aligned year-count row and a tidy dim-list block per indicator.

Layout summary
    * Top row at y ≈ 0.95 — "y=<n_years>" annotations (lowercase y),
      one per dynamic indicator, all on the same horizontal line.
    * Middle band — error bars (dynamic, mean ± std across years) or
      diamond markers (static), both at matched visual size.
    * Lower band — per-indicator dim block:
        - "D<idx>" labels listed top-to-bottom by frequency descending,
        - left-aligned within each block,
        - horizontally centered just above the indicator's x-tick,
        - decorated with a short vertical line on the block's left edge.

Inputs (precomputed by task06_corr_embedding.py; relative to package root)
    - data/task06_all/corr_dynamic.csv          (max |r|, var × year)
    - data/task06_all/corr_dynamic_bestdim.csv  (best dim, var × year)
    - data/task06_all/corr_static.csv           (abs_r, best_dim)

Outputs
    - output/task06_all/corr_plot_fig1_v2.png
"""

# %% [1] Setup
from collections import Counter
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Package root: parent of code/ when run as a script; falls back to cwd in
# an interactive window (start the interactive session from the package root).
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if Path.cwd().name != 'code' else Path.cwd().parent
DATA_DIR = ROOT / 'data' / 'task06_all'
OUT_DIR  = ROOT / 'output' / 'task06_all'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (csv_var, display_label, kind)
FIG1_VARS = [
    ('NDVI',             'NDVI',          'dynamic'),
    ('NL',               'NL',            'dynamic'),
    ('POP_mean',         'POP_density',   'dynamic'),
    ('POP_sum',          'POP_total',     'dynamic'),
    ('GAIA',             'ISA',           'dynamic'),
    ('GDP_mean',         'GDP_density',   'dynamic'),
    ('GDP_sum',          'GDP_total',     'dynamic'),
    ('Electricity_mean', 'ELEC_density',  'dynamic'),
    ('Electricity_sum',  'ELEC_total',    'dynamic'),
    ('TLML',             'TA',            'dynamic'),
    ('PRECTOTCORR',      'PREC',          'dynamic'),
    ('SPEED',            'Wind Speed',    'dynamic'),
    ('BF',               'BF',            'static'),
    ('BH_mean',          'BH',            'static'),
    ('DEM',              'Elevation',     'static'),
    ('zone_area_km2',    'AREA',          'static'),
]

C_DYNAMIC = '#1565C0'
C_STATIC  = '#EF5350'

# Marker visual size — matched between dynamic (Line2D markersize)
# and static (scatter s = ms**2 so the diamond reads at the same size).
MARKER_MS   = 11
SCATTER_S   = MARKER_MS ** 2

# Error bar style — thicker / more visible
ELINEWIDTH  = 2.2
CAPSIZE     = 10
CAPTHICK    = 2.8

# Year-count row (y=N) — single shared y position, just under top border
Y_N_LABEL   = 0.98

# Dim block layout
DIM_ROW_TOP   = 0.36       # y of the top dim row
DIM_ROW_STEP  = 0.06       # vertical gap between dim rows
DIM_X_OFFSET  = -0.14      # x offset of dim text left edge from column center
LINE_X_OFFSET = -0.22      # x offset of decorative vertical line (wider gap to "D")

plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.grid']  = False

# %% [2] Load precomputed correlation tables
corr_dyn     = pd.read_csv(DATA_DIR / 'corr_dynamic.csv',         index_col='variable')
corr_dyn_dim = pd.read_csv(DATA_DIR / 'corr_dynamic_bestdim.csv', index_col='variable')
corr_stat    = pd.read_csv(DATA_DIR / 'corr_static.csv',          index_col='variable')

print(f'corr_dynamic.csv : {corr_dyn.shape}    vars={list(corr_dyn.index)}')
print(f'corr_static.csv  : {corr_stat.shape}   vars={list(corr_stat.index)}')

# %% [3] Per-column dim lists (sorted by frequency desc, then dim asc)
def dims_for_dynamic(var):
    if var not in corr_dyn_dim.index:
        return []
    dims = corr_dyn_dim.loc[var].dropna().astype(int).tolist()
    if not dims:
        return []
    counts = Counter(dims)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in ordered]

def dims_for_static(var):
    if var not in corr_stat.index:
        return []
    return [int(corr_stat.loc[var, 'best_dim'])]

# %% [4] Draw dim block (left-aligned text + vertical decoration line)
def draw_dim_block(ax, xi, dims):
    if not dims:
        return
    n_rows = len(dims)
    pad    = 0.025
    y_top  = DIM_ROW_TOP + pad
    y_bot  = DIM_ROW_TOP - (n_rows - 1) * DIM_ROW_STEP - pad

    # decorative vertical line on the left of the block
    ax.plot([xi + LINE_X_OFFSET, xi + LINE_X_OFFSET],
            [y_bot, y_top],
            color='#888888', lw=0.9, solid_capstyle='round', zorder=4)

    for row_i, d in enumerate(dims):
        y = DIM_ROW_TOP - row_i * DIM_ROW_STEP
        ax.text(xi + DIM_X_OFFSET, y, f'D{d}',
                ha='left', va='center',
                fontsize=9, color='black')

# %% [5] Main plot
def make_fig_v2(rows, title=None, figsize=(15, 6.2)):
    n_vars = len(rows)
    xlim   = (-0.7, n_vars - 0.3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.6, zorder=0)

    for xi, (var, label, kind) in enumerate(rows):
        if kind == 'static':
            if var in corr_stat.index:
                val = float(corr_stat.loc[var, 'abs_r'])
                ax.scatter([xi], [val], color=C_STATIC,
                           s=SCATTER_S, marker='D', zorder=5,
                           edgecolors='white', linewidths=0.8)
            dims = dims_for_static(var)
        else:
            if var in corr_dyn.index:
                vals = corr_dyn.loc[var].dropna().values
                if len(vals):
                    mean = float(np.mean(vals))
                    std  = float(np.std(vals, ddof=0))
                    ax.errorbar([xi], [mean], yerr=[std],
                                fmt='o', color=C_DYNAMIC, ecolor=C_DYNAMIC,
                                elinewidth=ELINEWIDTH,
                                capsize=CAPSIZE, capthick=CAPTHICK,
                                markersize=MARKER_MS,
                                markerfacecolor=C_DYNAMIC,
                                markeredgecolor='white', markeredgewidth=0.8,
                                zorder=5)
                    # y=<n_years> at the shared top row
                    ax.text(xi, Y_N_LABEL, f'y={len(vals)}',
                            ha='center', va='center', fontsize=8.5,
                            color='black')
            dims = dims_for_dynamic(var)

        draw_dim_block(ax, xi, dims)

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([r[1] for r in rows], rotation=40, ha='right', fontsize=9.5)
    ax.set_ylabel('Max |R| over 1536 dims', fontsize=11)
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 1.0)

    ax.legend(
        handles=[
            mlines.Line2D([], [], marker='o', color=C_DYNAMIC,
                          linestyle='None', markersize=MARKER_MS * 0.8,
                          label='Dynamic'),
            mlines.Line2D([], [], marker='D', color='w',
                          markerfacecolor=C_STATIC,
                          markersize=MARKER_MS * 0.9,
                          label='Static'),
        ],
        loc='upper right', fontsize=10, framealpha=0.85,
    )
    fig.tight_layout()
    return fig


fig = make_fig_v2(FIG1_VARS, title=None)
out_path = OUT_DIR / 'corr_plot_fig1_v2.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out_path}')
