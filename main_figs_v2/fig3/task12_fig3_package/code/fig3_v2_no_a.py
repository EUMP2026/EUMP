# -*- coding: utf-8 -*-
"""Fig. 3 v2 — composite figure without panel a.

Layout (3 rows x 2 cols):
    a  (row0, span 2)  top-|R| summary (errorbars / diamonds + dim blocks)
    b  (row1, left)    ENV physical clustering map  (NDVI, NL, log_POP)
    c  (row1, right)   ENV embedding clustering map
    d  (row2, left)    FORM physical clustering map (BF, BH_mean, BH_QCD)
    e  (row2, right)   FORM embedding clustering map

Self-contained: reuses the validated logic of the four task06/task11 component
scripts, reads the single verified embedding npz, and renders everything into
one figure without importing or calling fig3_v2.py. Map basemap uses the local
world-borders shapefile (no online coastline download); ocean is white as in the
original main_figs_v1/fig3.
"""

from collections import Counter
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------------
# Paths (package root = parent of code/)
# ----------------------------------------------------------------------------
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if Path.cwd().name != "code" else Path.cwd().parent

EMBD_PATH   = ROOT / "data" / "embd" / \
    "eump_reconstruction_best_checkpoint_embeddings_2003-2020_pooled_260503_v4.npz"
MERGED_CSV  = ROOT / "data" / "zonal_merged_2020.csv"
FORM_CSV    = ROOT / "data" / "urban_form_clusters.csv"
TASK06_DIR  = ROOT / "data" / "task06_all"
COUNTRY_SHP = ROOT / "data" / "shp" / "TM_WORLD_BORDERS-0.3.shp"
GHSL_SHP    = ROOT / "data" / "shp" / "GHSL_filtered_v1.shp"
OUT_DIR     = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG     = OUT_DIR / "fig3_v2_no_a.png"

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def _register_arial():
    """Use Arial if installed; else register a bundled Arial.ttf if present."""
    from matplotlib import font_manager as fm
    if any(f.name == "Arial" for f in fm.fontManager.ttflist):
        return "Arial"
    for cand in ("/home/hy/.config/Ultralytics/Arial.ttf",
                 str(Path.home() / ".config/Ultralytics/Arial.ttf")):
        if Path(cand).exists():
            fm.fontManager.addfont(cand)
            return fm.FontProperties(fname=cand).get_name()
    return "DejaVu Sans"


plt.rcParams.update({"font.family": _register_arial()})
# Cluster palette (task11 v3/v4).
PALETTE = ["#1976D2", "#43A047", "#FB8C00", "#C62828"]
K = 4
SEED = 42
TOP_N = 10

PANEL_FS = 7           # panel-letter font size (kept within the 5-7 range)
FONT = {"label": 7, "tick": 6}

# ----------------------------------------------------------------------------
# Summary panel config (from task06_corr_embedding_v2.py)
# ----------------------------------------------------------------------------
FIG1_VARS = [
    ("NDVI",             "NDVI",          "dynamic"),
    ("NL",               "NL",            "dynamic"),
    ("POP_mean",         "POP_density",   "dynamic"),
    ("POP_sum",          "POP_total",     "dynamic"),
    ("GAIA",             "ISA",           "dynamic"),
    ("GDP_mean",         "GDP_density",   "dynamic"),
    ("GDP_sum",          "GDP_total",     "dynamic"),
    ("Electricity_mean", "ELEC_density",  "dynamic"),
    ("Electricity_sum",  "ELEC_total",    "dynamic"),
    ("TLML",             "TA",            "dynamic"),
    ("PRECTOTCORR",      "PREC",          "dynamic"),
    ("SPEED",            "Wind Speed",    "dynamic"),
    ("BF",               "BF",            "static"),
    ("BH_mean",          "BH",            "static"),
    ("DEM",              "Elevation",     "static"),
    ("zone_area_km2",    "AREA",          "static"),
]
C_DYNAMIC = "#1565C0"
C_STATIC  = "#EF5350"
MARKER_MS = 5.5
SCATTER_S = MARKER_MS ** 2
ELINEWIDTH, CAPSIZE, CAPTHICK = 1.4, 4, 1.6
Y_N_LABEL    = 0.98
DIM_ROW_TOP  = 0.36
DIM_ROW_STEP = 0.075
DIM_X_OFFSET = -0.14
LINE_X_OFFSET = -0.22
DIM_BOX_X_PAD = 0.02
DIM_BOX_CHAR_WIDTH = 0.14
DIM_BOX_Y_PAD = 0.034
SHARED_DIM_STYLES = {
    "nl_isa": {
        "dims": frozenset({238, 676, 967}),
        "face": "#DCEFED",
        "edge": "#74ADA8",
        "text": "#145F5A",
    },
    "scale": {
        "dims": frozenset({1236}),
        "face": "#ECE8F4",
        "edge": "#A28AC6",
        "text": "#5C447F",
    },
}


def load_embeddings():
    """Load the single npz once; return (emb_by_year dict, emb_cols, n_dims)."""
    with np.load(EMBD_PATH, allow_pickle=False) as data:
        emb = data["embeddings"].astype("float32", copy=False)
        sid = data["scene_ids"].astype(str).astype("int64")
        yrs = data["years"].astype("int32", copy=False)
    n_dims = emb.shape[1]
    emb_cols = [f"emb_{i:04d}" for i in range(n_dims)]
    df = pd.DataFrame(emb, columns=emb_cols)
    df.insert(0, "ID", sid)
    df.insert(1, "year", yrs)
    by_year = {int(y): g.drop(columns="year").reset_index(drop=True)
               for y, g in df.groupby("year")}
    return by_year, emb_cols, n_dims


def _dim_highlight_key(dim):
    for key, style in SHARED_DIM_STYLES.items():
        if dim in style["dims"]:
            return key
    return None


# ============================================================================
# Summary panel — top-|R| summary (errorbars / diamonds + dim blocks)
# ============================================================================
def draw_summary_panel(ax):
    corr_dyn     = pd.read_csv(TASK06_DIR / "corr_dynamic.csv",         index_col="variable")
    corr_dyn_dim = pd.read_csv(TASK06_DIR / "corr_dynamic_bestdim.csv", index_col="variable")
    corr_stat    = pd.read_csv(TASK06_DIR / "corr_static.csv",          index_col="variable")

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
        return [int(corr_stat.loc[var, "best_dim"])]

    def draw_dim_block(xi, dims):
        if not dims:
            return
        dims = dims[:4]                      # cap rows so blocks stay compact
        y_positions = [DIM_ROW_TOP - row_i * DIM_ROW_STEP
                       for row_i in range(len(dims))]
        y_top = y_positions[0] + DIM_BOX_Y_PAD
        y_bot = y_positions[-1] - DIM_BOX_Y_PAD
        ax.plot([xi + LINE_X_OFFSET, xi + LINE_X_OFFSET], [y_bot, y_top],
                color="#888888", lw=0.8, solid_capstyle="round", zorder=4)

        row_i = 0
        while row_i < len(dims):
            dim = dims[row_i]
            key = _dim_highlight_key(dim)
            if key is None:
                row_i += 1
                continue
            start = row_i
            while row_i + 1 < len(dims) and _dim_highlight_key(dims[row_i + 1]) == key:
                row_i += 1
            style = SHARED_DIM_STYLES[key]
            y0 = y_positions[row_i] - DIM_BOX_Y_PAD
            y1 = y_positions[start] + DIM_BOX_Y_PAD
            labels = [f"D{dims[i]}" for i in range(start, row_i + 1)]
            box_width = max(len(label) for label in labels) * DIM_BOX_CHAR_WIDTH
            box = FancyBboxPatch(
                (xi + DIM_X_OFFSET - DIM_BOX_X_PAD, y0),
                box_width,
                y1 - y0,
                boxstyle="round,pad=0.005,rounding_size=0.018",
                transform=ax.transData,
                facecolor=style["face"],
                edgecolor=style["edge"],
                linewidth=0.45,
                alpha=0.95,
                zorder=3.5,
                clip_on=False,
            )
            ax.add_patch(box)
            row_i += 1

        for row_i, d in enumerate(dims):
            y = y_positions[row_i]
            key = _dim_highlight_key(d)
            color = SHARED_DIM_STYLES[key]["text"] if key else "black"
            weight = "bold" if key else "normal"
            ax.text(xi + DIM_X_OFFSET, y, f"D{d}", ha="left", va="center",
                    fontsize=6, color=color, fontweight=weight, zorder=5)

    rows = FIG1_VARS
    n_vars = len(rows)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    for xi, (var, label, kind) in enumerate(rows):
        if kind == "static":
            if var in corr_stat.index:
                val = float(corr_stat.loc[var, "abs_r"])
                ax.scatter([xi], [val], color=C_STATIC, s=SCATTER_S, marker="D",
                           zorder=5, edgecolors="white", linewidths=0.6)
            draw_dim_block(xi, dims_for_static(var))
        else:
            if var in corr_dyn.index:
                vals = corr_dyn.loc[var].dropna().values
                if len(vals):
                    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=0))
                    ax.errorbar([xi], [mean], yerr=[std], fmt="o", color=C_DYNAMIC,
                                ecolor=C_DYNAMIC, elinewidth=ELINEWIDTH, capsize=CAPSIZE,
                                capthick=CAPTHICK, markersize=MARKER_MS,
                                markerfacecolor=C_DYNAMIC, markeredgecolor="white",
                                markeredgewidth=0.6, zorder=5)
                    ax.text(xi, Y_N_LABEL, f"yr={len(vals)}", ha="center", va="center",
                            fontsize=6, color="black")
            draw_dim_block(xi, dims_for_dynamic(var))

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels([r[1] for r in rows], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Max |r| over 1536 dims", fontsize=FONT["label"])
    ax.set_xlim(-0.7, n_vars - 0.3)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="y", labelsize=FONT["tick"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        handles=[
            mlines.Line2D([], [], marker="o", color=C_DYNAMIC, linestyle="None",
                          markersize=MARKER_MS * 0.8, label="Dynamic"),
            mlines.Line2D([], [], marker="D", color="w", markerfacecolor=C_STATIC,
                          markersize=MARKER_MS * 0.9, label="Static"),
        ],
        loc="upper right", fontsize=7, framealpha=0.85,
    )


# ============================================================================
# Map panels — clustering (env / form) + world maps
# ============================================================================
ENV_NAMES = {
    0: "C1: Dim, Green, Mid-pop",
    1: "C2: Bright, Green, Low-pop",
    2: "C3: Bright, Semi-green, High-pop",
    3: "C4: Bright, Barren, Varied-pop",
}
FORM_NAMES = {
    0: "C1: Sprawling Low-rise",
    1: "C2: Compact Low-rise",
    2: "C3: Moderate Mid-rise",
    3: "C4: Heterogeneous High-rise",
}
NL_TIERS  = [(40.0, "Dim"), (58.0, "Moderate"), (np.inf, "Bright")]
POP_TIERS = [(2.77, "Low"), (3.58, "Medium"), (np.inf, "High")]


def _tier_rank(val, tiers):
    for r, (thr, _) in enumerate(tiers):
        if val < thr:
            return r
    return len(tiers) - 1


def _cluster_pair(df, indices, emb_cols, kind):
    """KMeans physical + embedding (top-10/index union), Hungarian-aligned.

    Returns (df with 'cluster'/'emb_cluster', metrics dict).
    """
    E = df[emb_cols].values.astype("float64")
    E_z = (E - E.mean(0)) / E.std(0, ddof=0)
    M = df[indices].values.astype("float64")
    M_z = (M - M.mean(0)) / M.std(0, ddof=0)
    R = (E_z.T @ M_z) / E_z.shape[0]
    corr_df = pd.DataFrame(R, index=emb_cols, columns=indices)
    selected = []
    for idx in indices:
        for d in corr_df[idx].abs().sort_values(ascending=False).index[:TOP_N]:
            if d not in selected:
                selected.append(d)

    idx_scaler = StandardScaler()
    Xi = idx_scaler.fit_transform(df[indices].values)
    km_idx = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Xi)
    centers_raw = idx_scaler.inverse_transform(km_idx.cluster_centers_)

    if kind == "env":
        # NDVI, NL, log_POP  → sort NL↑ then POP↑, then swap the two bright clusters
        nl_ranks = np.array([_tier_rank(c[1], NL_TIERS) for c in centers_raw])
        pop_ranks = np.array([_tier_rank(c[2], POP_TIERS) for c in centers_raw])
        sort_order = np.lexsort((pop_ranks, nl_ranks)).copy()
        sort_order[[1, 2]] = sort_order[[2, 1]]
    else:
        # BF, BH_mean, BH_QCD → sort BH↑ then BF↑, swap the two low-rise clusters
        sort_order = np.lexsort((centers_raw[:, 0], centers_raw[:, 1])).copy()
        sort_order[[0, 1]] = sort_order[[1, 0]]
    remap_idx = {int(old): int(new) for new, old in enumerate(sort_order)}
    df = df.copy()
    df["cluster"] = [remap_idx[int(l)] for l in km_idx.labels_]
    y_idx = df["cluster"].values

    Xe = StandardScaler().fit_transform(df[selected].values)
    km_emb = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Xe)
    cm_raw = confusion_matrix(y_idx, km_emb.labels_, labels=list(range(K)))
    _, col_ind = linear_sum_assignment(-cm_raw)
    remap = {int(raw): int(i) for i, raw in enumerate(col_ind)}
    df["emb_cluster"] = [remap[int(l)] for l in km_emb.labels_]
    y_emb = df["emb_cluster"].values

    metrics = {
        "n_dims": len(selected),
    }
    return df, metrics


def prepare_env(emb_by_year, emb_cols):
    raw = pd.read_csv(MERGED_CSV, usecols=["ID", "NDVI_2020", "NL_2020", "POP_mean_2020"])
    raw = raw.dropna(subset=["NDVI_2020", "NL_2020", "POP_mean_2020"]).copy()
    raw["log_POP"] = np.log1p(raw["POP_mean_2020"])
    raw["ID"] = raw["ID"].astype("int64")
    df = raw.merge(emb_by_year[2020], on="ID", how="inner").copy()
    return _cluster_pair(df, ["NDVI_2020", "NL_2020", "log_POP"], emb_cols, "env")


def prepare_form(emb_by_year, emb_cols):
    src = pd.read_csv(FORM_CSV)
    src["ID"] = src["ID"].astype("int64")
    df = src.merge(emb_by_year[2012], on="ID", how="inner").dropna(
        subset=["BF", "BH_mean", "BH_QCD"]).copy()
    return _cluster_pair(df, ["BF", "BH_mean", "BH_QCD"], emb_cols, "form")


# Geo basemap (loaded once, reused for all four maps).
_DATA_CRS = ccrs.PlateCarree()
_PROJ = ccrs.Robinson(central_longitude=0)


def load_basemap():
    country = gpd.read_file(COUNTRY_SHP)
    ghsl = gpd.read_file(GHSL_SHP)[["ID", "GCPNT_LON", "GCPNT_LAT"]]
    ghsl["ID"] = ghsl["ID"].astype("int64")
    return country, ghsl


def draw_cluster_map(ax, df, col, ghsl, country_feat, *, names, title,
                     show_legend):
    ax.set_extent([-180, 180, -60, 78], crs=_DATA_CRS)
    ax.set_facecolor("white")                       # ocean = white (orig fig3)
    ax.add_feature(country_feat, zorder=1)
    dm = df.merge(ghsl, on="ID", how="left").dropna(subset=["GCPNT_LON"])
    n_plotted = 0
    for cid in range(K):
        mk = dm[col] == cid
        if not mk.any():
            continue
        ax.scatter(dm.loc[mk, "GCPNT_LON"], dm.loc[mk, "GCPNT_LAT"],
                   transform=_DATA_CRS, s=3.5, alpha=0.8, color=PALETTE[cid],
                   edgecolors="none", zorder=5)
        n_plotted += int(mk.sum())
    ax.set_title(title, fontsize=7, fontweight="bold", pad=5, linespacing=1.25)
    if show_legend:
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=PALETTE[c], markersize=6,
                          label=names[c]) for c in range(K)]
        # Single horizontal row spanning both maps in this row. The left map's
        # axes span 0..1; the pair midpoint sits just past its right edge.
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(1.015, 0),
                  fontsize=5.5, framealpha=0.0, ncol=K, borderpad=0.3,
                  handletextpad=0.3, labelspacing=0.25, columnspacing=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def panel_letter(ax, letter):
    ax.text(0.005, 1.0, letter, transform=ax.transAxes, fontsize=PANEL_FS,
            fontweight="bold", va="bottom", ha="left")


def build_figure(env_df, env_m, form_df, form_m, ghsl, country_feat, *, out_png):
    """Render the composite figure without panel a."""
    cm_to_in = 1.0 / 2.54
    LEFT, RIGHT = 0.10, 0.97

    fig = plt.figure(figsize=(18.0 * cm_to_in, 16 * cm_to_in), facecolor="white")
    TOP, BOTTOM = 0.965, 0.05
    SPLIT = 0.7        # boundary between panel a and the map block
    GAP = 0.11
    map_hspace = 0.18
    top_gs = fig.add_gridspec(1, 1, left=LEFT, right=RIGHT,
                              top=TOP, bottom=SPLIT)
    ax_a = fig.add_subplot(top_gs[0])
    letters = ("a", "b", "c", "d", "e")

    maps_gs = fig.add_gridspec(2, 2, left=LEFT, right=RIGHT,
                               top=SPLIT - GAP, bottom=BOTTOM,
                               hspace=map_hspace, wspace=0.03)

    # a — top dims
    draw_summary_panel(ax_a)
    panel_letter(ax_a, letters[0])

    # b / c — env maps (maps row 0)
    env_emb_title = f"Embedding cluster ({env_m['n_dims']} dims, yr 2020)"
    ax_c = fig.add_subplot(maps_gs[0, 0], projection=_PROJ)
    draw_cluster_map(ax_c, env_df, "cluster", ghsl, country_feat,
                     names=ENV_NAMES, title="Index cluster (NDVI, NL, POP)",
                     show_legend=True)
    panel_letter(ax_c, letters[1])
    ax_d = fig.add_subplot(maps_gs[0, 1], projection=_PROJ)
    draw_cluster_map(ax_d, env_df, "emb_cluster", ghsl, country_feat,
                     names=ENV_NAMES, title=env_emb_title,
                     show_legend=False)
    panel_letter(ax_d, letters[2])

    # d / e — form maps (maps row 1)
    form_emb_title = f"Embedding cluster ({form_m['n_dims']} dims, yr 2012)"
    ax_e = fig.add_subplot(maps_gs[1, 0], projection=_PROJ)
    draw_cluster_map(ax_e, form_df, "cluster", ghsl, country_feat,
                     names=FORM_NAMES, title="Index cluster (BF, BH_mean, BH_QCD)",
                     show_legend=True)
    panel_letter(ax_e, letters[3])
    ax_f = fig.add_subplot(maps_gs[1, 1], projection=_PROJ)
    draw_cluster_map(ax_f, form_df, "emb_cluster", ghsl, country_feat,
                     names=FORM_NAMES, title=form_emb_title,
                     show_legend=False)
    panel_letter(ax_f, letters[4])

    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out_png)


def prepare_figure_inputs():
    emb_by_year, emb_cols, n_dims = load_embeddings()
    print(f"Embeddings: years={sorted(emb_by_year)} dims={n_dims}")

    env_df, env_m = prepare_env(emb_by_year, emb_cols)
    print(f"env: n_dims={env_m['n_dims']}")
    form_df, form_m = prepare_form(emb_by_year, emb_cols)
    print(f"form: n_dims={form_m['n_dims']}")
    country, ghsl = load_basemap()
    country_feat = ShapelyFeature(country.geometry, _DATA_CRS, facecolor="#f0f0f0",
                                  edgecolor="#aaaaaa", linewidth=0.2)
    return env_df, env_m, form_df, form_m, ghsl, country_feat


def main():
    build_figure(*prepare_figure_inputs(), out_png=OUT_PNG)


if __name__ == "__main__":
    main()
