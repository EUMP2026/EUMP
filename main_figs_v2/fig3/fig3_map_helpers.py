# -*- coding: utf-8 -*-
"""Fig. 3 v2 — held-out-city embedding-interpretation figure.

Layout (3 rows x 2 cols):
    a  (row0, span 2)  top-|R| summary (errorbars / diamonds + dim blocks)
    b  (row1, left)    ENV physical clustering map  (NDVI, NL, log_POP)
    c  (row1, right)   ENV embedding clustering map
    d  (row2, left)    FORM physical clustering map (BF, BH_mean, BH_QCD)
    e  (row2, right)   FORM embedding clustering map

Objective: render the held-out test-city assignments from the discovery-fitted
Task 11 analyses together with the held-out correlation summary in fig3.py.
Inputs: packaged Task 11 held-out label/selection/metric CSVs and local
world/city shapefiles.
The four maps share each typology's test-city cohort. No fitting takes place
in this figure script. Ocean is white as in the original figure.
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature

# ----------------------------------------------------------------------------
# Paths (all inputs are packaged beside this module)
# ----------------------------------------------------------------------------

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

TASK11_DIR  = ROOT / "data" / "task11_heldout"
COUNTRY_SHP = ROOT / "data" / "shp" / "world_map_v2.shp"
GHSL_SHP    = ROOT / "data" / "shp" / "GHSL_filtered_v1.shp"

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
def _register_arial():
    """Use installed Arial, otherwise Matplotlib-provided DejaVu Sans."""
    from matplotlib import font_manager as fm
    if any(f.name == "Arial" for f in fm.fontManager.ttflist):
        return "Arial"
    return "DejaVu Sans"


plt.rcParams.update({"font.family": _register_arial(),
                     "svg.fonttype": "none", "pdf.fonttype": 42})
# Cluster palette (matching the discovery-fitted Task 11 scripts).
PALETTE = ["#1976D2", "#43A047", "#FB8C00", "#C62828"]
K = 4

PANEL_FS = 7           # panel-letter font size (kept within the 5-7 range)
FONT = {"label": 7, "tick": 6}

# User-adjustable display controls.
PANEL_A_XLABEL_ROTATION_DEG = 25  # smaller angles make panel-a labels shorter
MAP_POINT_SIZE = 3 # 3.5               # city-marker area in panels b-e (points^2)

# ----------------------------------------------------------------------------
# Summary panel config (from task06_corr_embedding_v2.py)
# ----------------------------------------------------------------------------
FIG1_VARS = [
    ("NDVI",             "NDVI",          "dynamic"),
    ("NL",               "NL",            "dynamic"),
    ("POP_mean",         r"POP$_{\mathrm{den}}$",    "dynamic"),
    ("POP_sum",          r"POP$_{\mathrm{tot}}$",    "dynamic"),
    ("GAIA",             "ISA",           "dynamic"),
    ("GDP_mean",         r"GDP$_{\mathrm{den}}$",    "dynamic"),
    ("GDP_sum",          r"GDP$_{\mathrm{tot}}$",    "dynamic"),
    ("Electricity_mean", r"Elec$_{\mathrm{den}}$",   "dynamic"),
    ("Electricity_sum",  r"Elec$_{\mathrm{tot}}$",   "dynamic"),
    ("TLML",             r"Ta$_{\mathrm{bk}}$",      "dynamic"),
    ("PRECTOTCORR",      r"Precip$_{\mathrm{bk}}$",  "dynamic"),
    ("SPEED",            r"Wind$_{\mathrm{bk}}$",    "dynamic"),
    ("BF",               "BF",            "static"),
    ("BH_mean",          "BH",            "static"),
    ("DEM",              "Elevation",     "static"),
    ("zone_area_km2",    "Area",          "static"),
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
DIM_BOX_Y_PAD = 0.034


# ============================================================================
# Summary panel — top-|R| summary (errorbars / diamonds + dim blocks)
# ============================================================================
# Panel-a configuration used by fig3.py


# ============================================================================
# Map panels — clustering (env / form) + world maps
# ============================================================================
# Load discovery-fitted held-out cluster assignments
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
def load_test_clusters(labels_name, selected_name, metrics_name, expected_n,
                       expected_dims):
    """Read held-out labels from a frozen discovery-fitted Task 11 analysis."""
    labels = pd.read_csv(TASK11_DIR / labels_name)
    selected = pd.read_csv(TASK11_DIR / selected_name)
    metrics = pd.read_csv(TASK11_DIR / metrics_name)

    if labels["ID"].duplicated().any():
        raise ValueError(f"Duplicate city IDs in {labels_name}.")
    if not labels["split"].isin(["train", "val", "test"]).all():
        raise ValueError(f"Unexpected source split in {labels_name}.")
    test = labels.loc[labels["evaluation_split"] == "test"].copy()
    if not test["split"].eq("test").all():
        raise ValueError(f"The test rows in {labels_name} do not match the source split.")
    test = test.dropna(subset=["cluster", "emb_cluster"]).copy()
    if len(test) != expected_n:
        raise ValueError(f"Expected {expected_n} complete test cities in {labels_name}; "
                         f"found {len(test)}.")
    test["ID"] = test["ID"].astype("int64")
    for col in ("cluster", "emb_cluster"):
        values = test[col].to_numpy()
        if not np.isin(values, np.arange(K)).all():
            raise ValueError(f"Unexpected {col} labels in {labels_name}.")
        test[col] = test[col].astype("int64")

    n_dims = selected["dim"].nunique()
    if n_dims != expected_dims:
        raise ValueError(f"Expected {expected_dims} unique embedding dimensions "
                         f"in {selected_name}; found {n_dims}.")
    held_out_metrics = metrics.loc[metrics["split"] == "held_out"]
    if len(held_out_metrics) != 1:
        raise ValueError(f"Expected one held-out metric row in {metrics_name}.")
    accuracy = float((test["cluster"] == test["emb_cluster"]).mean())
    saved = held_out_metrics.iloc[0]
    if int(saved["n"]) != len(test) or not np.isclose(accuracy, saved["accuracy"]):
        raise ValueError(f"Held-out labels and metrics disagree in {labels_name}.")
    return test, {"n_dims": n_dims, "n": len(test), "accuracy": accuracy}


def prepare_env():
    return load_test_clusters(
        "urban_env_clusters_v4.csv", "selected_dims_v4.csv",
        "train_test_consistency_v4.csv", expected_n=162, expected_dims=30,
    )


def prepare_form():
    return load_test_clusters(
        "urban_form_clusters_v5.csv", "selected_dims_v5.csv",
        "train_test_consistency_v5.csv", expected_n=161, expected_dims=27,
    )


# Geo basemap (loaded once, reused for all four maps).
# Draw held-out maps
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
    # Rasterize the detailed country polygons in vector exports to keep the
    # PDF/SVG practical; city points, text, and panel a remain vector objects.
    ax.add_feature(country_feat, zorder=1, rasterized=True)
    dm = df.merge(ghsl, on="ID", how="left").dropna(subset=["GCPNT_LON"])
    n_plotted = 0
    for cid in range(K):
        mk = dm[col] == cid
        if not mk.any():
            continue
        ax.scatter(dm.loc[mk, "GCPNT_LON"], dm.loc[mk, "GCPNT_LAT"],
                   transform=_DATA_CRS, s=MAP_POINT_SIZE, alpha=0.8,
                   color=PALETTE[cid],
                   edgecolors="none", zorder=5)
        n_plotted += int(mk.sum())
    if n_plotted != len(df):
        raise ValueError(f"Only {n_plotted}/{len(df)} test cities were mapped in {title}.")
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




def prepare_figure_inputs():
    env_df, env_m = prepare_env()
    print(f"environment test: n={env_m['n']}, "
          f"n_dims={env_m['n_dims']}, OA={env_m['accuracy']:.4f}")
    form_df, form_m = prepare_form()
    print(f"urban-form test: n={form_m['n']}, "
          f"n_dims={form_m['n_dims']}, OA={form_m['accuracy']:.4f}")
    country, ghsl = load_basemap()
    country_feat = ShapelyFeature(country.geometry, _DATA_CRS, facecolor="#f0f0f0",
                                  edgecolor="#aaaaaa", linewidth=0.2)
    return env_df, env_m, form_df, form_m, ghsl, country_feat
