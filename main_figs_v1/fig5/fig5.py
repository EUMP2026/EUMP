# -*- coding: utf-8 -*-

import os
import sys
import warnings
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm, colors
from scipy.stats import linregress, t
import cartopy.crs as ccrs

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Setup ---
CURRENT_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
warnings.filterwarnings("ignore")

# --- Plotting Configuration ---

# Define RGB values (normalized to 0–1)
colors_b = [
    (6/255, 50/255, 108/255),
    (7/255, 81/255, 156/255),
    (33/255, 113/255, 181/255),
    (66/255, 146/255, 199/255),
    (107/255, 173/255, 215/255),
    (159/255, 203/255, 226/255),
    (198/255, 220/255, 238/255),
    (222/255, 236/255, 247/255),
    (249/255, 251/255, 252/255),
]

# Create custom colormap
# blue_cmap = ListedColormap(colors_b, name="custom_blue")
blue_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_b)

# --- Load full colormap ---
cmap_full = cm.get_cmap('RdBu_r')
# --- Extract the half (cooling, i.e., blue) part ---
# 'RdBu_r' goes from red → white → blue, so the first half is blue
n = 256
half_colors = cmap_full(np.linspace(0, 0.5, n))
blue_cmap = colors.ListedColormap(half_colors, name='RdBu_r_cooling')

plt.rcParams.update({
    "axes.linewidth": 0.3,
    "xtick.major.width": 0.3,
    "ytick.major.width": 0.3,
    "grid.linewidth": 0.3,
    "lines.linewidth": 0.3,
    "font.size": 6,
})

FONT = {"title": 7, "label": 7, "tick": 6}
MARKER = {"size": 3, "alpha": 0.8}
# MARKER = {"size": 1, "alpha": 1}
LINE = {"coast": 0.1, "country": 0.1}

# Custom colormap
COOL2WARM_DARK = mcolors.LinearSegmentedColormap.from_list(
    "cool2warm_dark",
    ["#08306b", "#2171b5", "#deebf7", "#f7f7f7"],
    N=256,
)

# --- File Paths ---
SHAPEFILES = {
    "country": "./country_shp/TM_WORLD_BORDERS-0.3.shp",
    "urban": "./urban_shp/GHSL_filtered_v1.shp",
}
DATA_FILES = {
    "main": "./data_2020_2000steps.csv",
    "climate_meta": "./CityPoints_withClimate.csv",
    "climate_bk": "clim_values.csv",
    "imerg": "IMERG_all.csv",
}

# --- Load & Prepare Data ---
def load_and_merge_data() -> gpd.GeoDataFrame:
    # Load base data
    # country_shp = gpd.read_file(SHAPEFILES["country"])
    urban_shp = gpd.read_file(SHAPEFILES["urban"])[
        ["ID", "geometry", "CTR_MN_NM", "CTR_MN_ISO", "UC_NM_MN", "UC_NM_LST", "AREA", "P15"]
    ]
    urban_pts = urban_shp.copy()
    urban_pts["geometry"] = urban_pts.geometry.centroid

    # Ensure consistent ID type
    for df in [urban_pts]:
        df["ID"] = df["ID"].astype(str)

    # Main dataset
    df_main = pd.read_csv(DATA_FILES["main"])
    df_main["ID"] = df_main["ID"].astype(str)
    merged = urban_pts.merge(df_main, on="ID", how="left")

    # Climate metadata
    climate_meta = pd.read_csv(DATA_FILES["climate_meta"], usecols=["ID", "climates_f"], encoding="utf-8")
    climate_meta["ID"] = climate_meta["ID"].astype(str)
    merged = merged.merge(climate_meta, on="ID", how="left")

    # IMERG data (convert mm/hr → mm/yr)
    imerg = pd.read_csv(DATA_FILES["imerg"])[["ID", "IMERG_max", "IMERG_mean", "IMERG_min", "IMERG_range"]]
    imerg["ID"] = imerg["ID"].astype(str)
    merged = merged.merge(imerg, on="ID", how="left")
    merged[["IMERG_max", "IMERG_mean", "IMERG_min", "IMERG_range"]] *= 24 * 365

    # Derived columns
    merged["urban_pop0_log10"] = np.log10(merged["urban_pop0"])
    merged["AREA_log10"] = np.log10(merged["AREA"])
    merged["UHI"] = merged["urban_lst0"] - merged["rural_lst0"]
    merged["delta_congestion"] = merged["congestion1"] - merged["congestion0"]
    merged["delta_ndvi_diff"] = merged["urban_delta_ndvi"] - merged["cooling_urban_delta_ndvi"]
    merged["f_diff"] = merged["f_lst_n1_p1"] - merged["f_lst_n0_p0"]
    merged["sensitivity"] = merged["urban_delta_lst"] / merged["urban_delta_ndvi"] * 0.1
    merged["sensitivity_cooling"] = merged["cooling_urban_delta_lst"] / merged["cooling_urban_delta_ndvi"] * 0.1
    merged["sensitivity_diff"] = merged["sensitivity"] - merged["sensitivity_cooling"]

    return merged


# --- Plotting Utilities ---
def add_gridlabels_only(ax, fontsize: int = FONT["tick"]):
    gl = ax.gridlines(draw_labels=True, linewidth=0, color="none", alpha=0, linestyle="")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": fontsize, "color": "black"}
    return gl


def plot_points(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    ax,
    cmap=COOL2WARM_DARK,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    draw_cbar: bool = True,
):
    if column not in gdf.columns:
        raise KeyError(f"Column '{column}' not found.")

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    gdf.plot(
        column=column,
        cmap=cmap,
        norm=norm,
        markersize=2,
        alpha=1,

        ax=ax,
        transform=ccrs.PlateCarree(),
    )

    # Borders
    country_shp = gpd.read_file(SHAPEFILES["country"])
    country_shp.plot(lw=LINE["country"], edgecolor="grey", facecolor="none", ax=ax, transform=ccrs.PlateCarree())
    ax.coastlines(lw=LINE["coast"], edgecolor="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    if draw_cbar:
        cax = inset_axes(
            ax,
            width="6%",
            height="120%",
            loc="lower left",
            bbox_to_anchor=(0.1, 0.12, 0.5, 0.4),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        cbar = ax.get_figure().colorbar(sm, cax=cax, orientation="vertical")
        cbar.ax.tick_params(labelsize=FONT["tick"], length=1.5, width=0.3, pad=1)
        cbar.outline.set_linewidth(0.3)

    return ax, sm


def create_binned_boxplot(
    ax,
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: np.ndarray,
    labels: list,
    xlabel: str,
    ylabel: str,
    y_filter_threshold: Optional[float] = None,
    ylabel_on: bool = True, 
):
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if y_filter_threshold is not None:
        mask &= y_data > y_filter_threshold
    x, y = x_data[mask], y_data[mask]

    bin_idx = np.digitize(x, bins) - 1
    df = pd.DataFrame({"bin": bin_idx, "y": y})
    df = df[(df["bin"] >= 0) & (df["bin"] < len(bins) - 1)]

    box_data = [df[df["bin"] == i]["y"].values for i in range(len(bins) - 1)]
    p10 = [np.percentile(b, 10) if len(b) > 0 else np.nan for b in box_data]
    counts = [np.sum((x >= bins[i]) & (x < bins[i + 1])) for i in range(len(bins) - 1)]
    
    # print(box_data)
    
    colors = {
        "box": "#D0E1F3",
        "median": "#E74C3C", 
        "line": "#1f4e9f",#"#E74C3C",
        "flier": "#85929E",
        "outline": "#34495E",
    }

    bp = ax.boxplot(
        box_data,
        positions=np.arange(len(labels)),
        patch_artist=True,
        widths=0.6,
        boxprops=dict(facecolor=colors["box"], color=colors["outline"], linewidth=0.6),
        medianprops=dict(color=colors["median"], linewidth=1),
        whiskerprops=dict(color=colors["outline"], linewidth=0.6),
        capprops=dict(color=colors["outline"], linewidth=0.6),
        flierprops=dict(marker="o", markersize=1.5, markerfacecolor=colors["flier"], alpha=0.6, markeredgecolor="none"),
    )

    ax.plot(np.arange(len(labels)), p10, color=colors["line"], lw=1, label="10th percentile", marker="o", markersize=1.5)

    ylim = ax.get_ylim()
    offset = (ylim[1] - ylim[0]) * 0.02
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, ylim[1] + offset, f"n={count}", ha="center", va="bottom", fontsize=FONT["tick"], color=colors["outline"])

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=FONT["tick"])
    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    if ylabel_on:
        ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.legend(loc="lower right", fontsize=FONT["tick"], frameon=False, facecolor="white", edgecolor="gray", framealpha=0.6)


def plot_scatter_with_ols(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ax,
    xlabel: str,
    ylabel: str,
    text_loc: Tuple[float, float] = (0.95, 0.95),
    ylabel_on: bool = True, 
):
    x, y = df[x_col].values, df[y_col].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    res = linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 200)
    yy = res.slope * xx + res.intercept
    
    # --- Confidence interval ---
    n = len(x)
    y_pred = res.slope * x + res.intercept
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))  # standard error of regression
    
    # critical t value for confidence level
    ci: float = 0.99
    t_val = t.ppf((1 + ci) / 2., n - 2)

    mean_x = np.mean(x)
    se_line = s_err * np.sqrt(1/n + (xx - mean_x)**2 / np.sum((x - mean_x)**2))
    ci_upper = yy + t_val * se_line
    ci_lower = yy - t_val * se_line

    ax.scatter(
        x, y,
        s=MARKER["size"] * 2,           # make markers larger
        facecolor="#D0E1F3",               # gray fill
        edgecolor="k",              # black edge
        linewidth=0.2,                  # thin edge
        alpha=MARKER["alpha"]
    )
    
    ax.plot(
        xx, yy,
        lw=1.5,
        color="#E74C3C",
        label="OLS fit"
    )
    
    ax.fill_between(
        xx, ci_lower, ci_upper,
        color="#E74C3C",
        alpha=0.2,
        edgecolor="none",               # no edge for fill band
        label=f"{int(ci*100)}% CI"
    )
      
    txt = rf"$y = {res.slope:.2f}x {res.intercept:+.2f}$" + "\n" + rf"$R^2 = {res.rvalue**2:.2f}$"
    ax.text(
        text_loc[0],
        text_loc[1],
        txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
    )

    ax.set_xlabel(xlabel, fontsize=FONT["label"])
    if ylabel_on:
        ax.set_ylabel(ylabel, fontsize=FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])

# --- Main Figure ---
def plot_with_csv(save_path: str = "./fig4.png"):
    merged_gdf = load_and_merge_data()

    fig = plt.figure(figsize=(18 / 2.54, 18 / 2.54), dpi=600)
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[3.2, 1.5, 1.5], hspace=0.25, wspace=0.2)

    # ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.Robinson(central_longitude=0))
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Panel a: Global map
    plot_points(
        merged_gdf,
        column="sensitivity_cooling",
        title="Cooling efficiency for single goal",
        ax=ax1,
        vmin=-4,
        vmax=0,
        cmap=blue_cmap,
    )
    ax1.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())
    ax1.text(0.02, 0.98, "a", transform=ax1.transAxes, fontsize=FONT["title"], fontweight="bold", va="top", ha="left")

    # Panel b: Precipitation vs sensitivity
    x1 = merged_gdf["IMERG_mean"] / 1000
    y1 = merged_gdf["sensitivity_cooling"]
    

    bins1 = np.array([0, 500, 1000, 1500, 2000, 5000])/1000
    labels1 = [f"{int(bins1[i]*1000)}-{int(bins1[i+1]*1000)}" for i in range(len(bins1) - 2)]
    labels1.append(f">{int(bins1[-2]*1000)}")
    create_binned_boxplot(ax2, x1, y1, bins1, labels1, "Precipitation (mm)", "Sensitivity", y_filter_threshold=-5)    
    
    ax2.text(0.02, 0.98, "b", transform=ax2.transAxes, fontsize=FONT["title"], fontweight="bold", va="top", ha="left")

    # Panel c: Urban population vs sensitivity
    x2 = merged_gdf["urban_pop0_log10"]
    y2 = merged_gdf["sensitivity_cooling"]
    bins2 = np.array([0, 4.5, 5, 5.5, 6, 7])
    labels2 = [f"{bins2[i]:.1f}-{bins2[i+1]:.1f}" for i in range(len(bins2) - 1)]
    create_binned_boxplot(ax3, x2, y2, bins2, labels2, "Urban Population ($\log_{10} N$)", "Sensitivity", y_filter_threshold=-5)
    ax3.text(0.02, 0.98, "c", transform=ax3.transAxes, fontsize=FONT["title"], fontweight="bold", va="top", ha="left")

    # Panel d: Cooling ΔNDVI vs ΔLST
    plot_scatter_with_ols(merged_gdf, "cooling_urban_delta_ndvi", "cooling_urban_delta_lst", ax4, "Single-goal ΔNDVI", "Single-goal ΔLST")
    ax4.text(0.02, 0.98, "d", transform=ax4.transAxes, fontsize=FONT["title"], fontweight="bold", va="top", ha="left")

    # Panel e: Dual-goal ΔNDVI vs ΔLST
    plot_scatter_with_ols(merged_gdf, "urban_delta_ndvi", "urban_delta_lst", ax5, "Dual-goal ΔNDVI", "Dual-goal ΔLST")
    ax5.text(0.02, 0.98, "e", transform=ax5.transAxes, fontsize=FONT["title"], fontweight="bold", va="top", ha="left")

    fig.savefig(save_path, bbox_inches="tight", dpi=600)
    print(f"Saved figure to: {save_path}")

    return fig, (ax1, ax2, ax3, ax4, ax5)


if __name__ == "__main__":
    plot_with_csv(save_path="./fig5.png")
    
merged_gdf = load_and_merge_data()
plt.hist(merged_gdf['sensitivity_cooling'].values,bins=100)
plt.show()
