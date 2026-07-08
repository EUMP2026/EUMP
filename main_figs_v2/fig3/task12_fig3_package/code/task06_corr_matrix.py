# %%
"""
Task 06 — Embedding-to-variable coefficient matrix heatmap.

Objective
    Compute the full Pearson r matrix between every physical variable used
    in corr_plot_fig1_v2.png (y-axis, 16 variables) and all 1536 embedding
    dimensions (x-axis), then plot it as a diverging-colormap heatmap in
    the style of fig2_v2.py (02-urban-height project).

Method
    - Dynamic variables: use the LATEST year with sufficient matched data
      (e.g., 2020 if available; GAIA → 2018, Electricity → 2019, ...).
    - Static variables: use the same prescribed embedding years as
      task06_corr_embedding.py (BF → 2019, BH_mean → 2012, DEM/AREA → 2020).
    - Strict city-year match between embeddings and variables (inner join
      on city ID for the chosen year).
    - Vectorised Pearson r over all 1536 dims at once.

Inputs (relative to the package root)
    - data/embd/eump_reconstruction_best_checkpoint_embeddings_2003-2020_pooled_260503_v4.npz
    - data/zonal_merged_2020.csv

Outputs
    - output/task06_all/corr_matrix_16x1536.npy   (r matrix, row order below)
    - output/task06_all/corr_matrix_meta.csv      (variable, label, kind, year, n)
    - output/task06_all/corr_matrix_fig.png
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# %% [1] Paths and config
# Package root: parent of code/ when run as a script; falls back to cwd in
# an interactive window (start the interactive session from the package root).
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if Path.cwd().name != "code" else Path.cwd().parent
MERGED_PATH = ROOT / "data" / "zonal_merged_2020.csv"
EMBD_PATH   = (
    ROOT / "data" / "embd"
    / "eump_reconstruction_best_checkpoint_embeddings_2003-2020_pooled_260503_v4.npz"
)
OUT_DIR = ROOT / "output" / "task06_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_YEARS = list(range(2003, 2021))
MIN_N     = 30

# (csv_var, display_label, kind) — same order/labels as corr_plot_fig1_v2.png
MATRIX_VARS = [
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

# Static variables — embedding year prescribed (same as task06_corr_embedding.py)
STATIC_EMB_YEAR = {
    'zone_area_km2': 2020,
    'BF':            2012,
    'BH_mean':       2012,
    'DEM':           2020,
}

# Style (fig2_v2.py)
FONT = {'title': 7, 'label': 6, 'tick': 5}
CMAP = LinearSegmentedColormap.from_list(
    'teal_orange', ['#249c98', '#e0e0e0', '#f0785b'])

plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.grid']  = False

print("merged :", MERGED_PATH)
print("embd   :", EMBD_PATH)

# %% [2] Load embeddings, index by year
with np.load(EMBD_PATH, allow_pickle=False) as data:
    embeddings = data["embeddings"].astype("float32", copy=False)
    scene_ids  = data["scene_ids"].astype(str).astype("int64")
    emb_years  = data["years"].astype("int32", copy=False)

n_dims   = embeddings.shape[1]
emb_cols = [f"emb_{i:04d}" for i in range(n_dims)]
emb_df   = pd.DataFrame(embeddings, columns=emb_cols)
emb_df.insert(0, "ID",   scene_ids)
emb_df.insert(1, "year", emb_years)

emb_by_year = {yr: grp.drop(columns="year").reset_index(drop=True)
               for yr, grp in emb_df.groupby("year")}
print(f"Embeddings: {emb_df.shape}  dims={n_dims}  years={sorted(emb_by_year)}")

# %% [3] Load merged city table
city_wide = pd.read_csv(MERGED_PATH, low_memory=False)
city_wide["ID"] = city_wide["ID"].astype("int64")
all_cols     = city_wide.columns.tolist()
emb_year_set = set(EMB_YEARS)
print(f"City table: {city_wide.shape}")


def get_overlap(var):
    """Return [(col, year), ...] for var overlapping with embedding years, asc year."""
    pat = re.compile(r"^" + re.escape(var) + r"_(\d{4})$")
    hits = sorted(
        [(c, int(m.group(1))) for c in all_cols if (m := pat.match(c))],
        key=lambda x: x[1],
    )
    return [(c, y) for c, y in hits if y in emb_year_set]


# %% [4] Full-vector Pearson r helper
def full_corr(X, y):
    """Return r vector (n_dims,) and sample size n; NaN where undefined."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    n = X.shape[0]
    if n < MIN_N:
        return np.full(X.shape[1], np.nan), n

    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()

    x_norm = np.sqrt((Xc * Xc).sum(axis=0))
    y_norm = np.sqrt((yc * yc).sum())
    if y_norm == 0:
        return np.full(X.shape[1], np.nan), n

    x_norm_safe = np.where(x_norm == 0, np.nan, x_norm)
    r = (Xc.T @ yc) / (x_norm_safe * y_norm)
    return r, n


# %% [5] Build the 16 x 1536 matrix
corr_rows = []
meta_rows = []

for var, label, kind in MATRIX_VARS:
    if kind == 'static':
        emb_yr = STATIC_EMB_YEAR[var]
        col    = var
        if col not in city_wide.columns or emb_yr not in emb_by_year:
            print(f"  SKIP {var}")
            corr_rows.append(np.full(n_dims, np.nan))
            meta_rows.append({"variable": var, "label": label, "kind": kind,
                              "year": emb_yr, "n": 0})
            continue
        sub = city_wide[["ID", col]].dropna(subset=[col])
        df  = sub.merge(emb_by_year[emb_yr], on="ID", how="inner")
        r, n = full_corr(df[emb_cols].values, df[col].values)
        used_year = emb_yr
    else:
        # latest year with sufficient matched data
        overlap = get_overlap(var)
        r, n, used_year = np.full(n_dims, np.nan), 0, None
        for col, year in reversed(overlap):          # latest first
            if year not in emb_by_year:
                continue
            sub = city_wide[["ID", col]].dropna(subset=[col])
            df  = sub.merge(emb_by_year[year], on="ID", how="inner")
            if len(df) < MIN_N:
                continue
            r_try, n_try = full_corr(df[emb_cols].values, df[col].values)
            if np.all(np.isnan(r_try)):
                continue
            r, n, used_year = r_try, n_try, year
            break

    corr_rows.append(r)
    meta_rows.append({"variable": var, "label": label, "kind": kind,
                      "year": used_year, "n": n})
    mean_abs = np.nanmean(np.abs(r))
    print(f"  {label:<14} kind={kind:<7} year={used_year}  n={n:5d}  "
          f"mean|r|={mean_abs:.3f}  max|r|={np.nanmax(np.abs(r)):.3f}")

corr_matrix = np.vstack(corr_rows)          # (16, 1536)
meta_df     = pd.DataFrame(meta_rows)

np.save(OUT_DIR / "corr_matrix_16x1536.npy", corr_matrix)
meta_df.to_csv(OUT_DIR / "corr_matrix_meta.csv", index=False)
print("\nSaved corr_matrix_16x1536.npy:", corr_matrix.shape)
print("Saved corr_matrix_meta.csv")

# %% [6] Heatmap — fig2_v2.py style
n_rows = corr_matrix.shape[0]

# y labels with max |r| in parentheses
row_max_abs = np.nanmax(np.abs(corr_matrix), axis=1)
ylabels = [f"{m['label']} ({v:.2f})"
           for m, v in zip(meta_rows, row_max_abs)]

vmax = float(np.nanmax(np.abs(corr_matrix)))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

cm_to_inch = 1.0 / 2.54
figsize = (18.0 * cm_to_inch, 8.0 * cm_to_inch)

fig, ax = plt.subplots(figsize=figsize, facecolor='white',
                       constrained_layout=True)
im = ax.imshow(corr_matrix, aspect='auto', cmap=CMAP, norm=norm,
               interpolation='nearest', origin='upper')

ax.set_ylabel('Physical variables', fontsize=FONT['label'])
ax.set_xlabel('Embeddings', fontsize=FONT['label'])
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(ylabels, fontsize=FONT['tick'])

n_x_ticks   = 4
x_positions = np.linspace(0, corr_matrix.shape[1] - 1, n_x_ticks, dtype=int)
ax.set_xticks(x_positions)
ax.set_xticklabels(x_positions, fontsize=FONT['tick'])

ax.tick_params(axis='both', which='both', length=0)
for r_line in np.arange(0.5, n_rows - 0.5, 1.0):
    ax.axhline(r_line, color='#dddddd', linewidth=0.8)
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_linewidth(0.6)

cax = inset_axes(ax, width='1.5%', height='100%',
                 bbox_transform=ax.transAxes, loc='right',
                 bbox_to_anchor=(0.03, 0., 1, 1), borderpad=0)
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=FONT['tick'], length=2, width=0.6)
cb.set_label('Correlation', fontsize=FONT['label'])
for spine in cb.ax.spines.values():
    spine.set_linewidth(0.6)

out_png = OUT_DIR / "corr_matrix_fig.png"
fig.savefig(out_png, dpi=600, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("Saved:", out_png)
