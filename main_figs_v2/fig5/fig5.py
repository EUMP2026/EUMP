# -*- coding: utf-8 -*-
"""Render manuscript Fig. 5 from the packaged per-city summary tables.

The figure contains the global cooling-efficiency map, climate/economic group
comparisons, and remote-buffer LST contrasts. All inputs are under ``data/``
and the 600-dpi ``fig5.png`` is written beside this script.
"""
# Imports
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats

import cartopy.crs as ccrs
import geopandas as gpd
from cartopy.feature import ShapelyFeature

# ---------------------------------------------------------------------------
# Paths (resolve relative to this script)
# ---------------------------------------------------------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
OUT_PNG = ROOT / "fig5.png"

CE_MAP_CSV = DATA_DIR / "ce_eump_per_city.csv"
CITY_PTS_CSV = DATA_DIR / "ghsl_city_points.csv"
GROUP_CSV = DATA_DIR / "per_city_ce_embd_groups_v2.csv"
COOLING_CSV = DATA_DIR / "city_cooling_summary.csv"
COUNTRY_SHP = DATA_DIR / "shp" / "world_map_v2.shp"

# User-adjustable marker size for the city points in panel a.
PANEL_A_POINT_SIZE = 9

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def _register_arial():
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
FONT = {"label": 7, "tick": 6, "title": 7.5, "panel": 9, "annot": 6}
CM = 1.0 / 2.54

# Palette for the three panels. Panel a uses the cooling half of RdBu_r.
PAL = {
    "terracotta": "#CA755B",
    "apricot": "#EBBD8A",
    "cream": "#F5EBD5",
    "blue": "#76A8D5",
    "steel": "#93A9D6",
    "pale_blue": "#BDDCFD",
    "ink": "#263337",
    "axis": "#4B5558",
    "grid": "#E2E7E4",
    "zero": "#8B9692",
}

GCOLORS = {"Low": PAL["cream"], "High": PAL["apricot"]}  # b
SIZE_COLORS = {"small": PAL["pale_blue"], "large": PAL["blue"]}  # c
BOX_EDGE = "#4C5757"
AXIS_GREY = PAL["axis"]
GRID_GREY = PAL["grid"]
ZERO_GREY = PAL["zero"]

# Shared boxplot rendering: centre line = median; black dot = mean.
MEDIAN_KW = dict(color=PAL["ink"], linewidth=1.15)


def _style_box(bp, facecolors, alpha=0.95):
    for patch, c in zip(bp["boxes"], facecolors):
        patch.set_facecolor(c); patch.set_alpha(alpha)
        patch.set_edgecolor(BOX_EDGE); patch.set_linewidth(0.6)
    for el in ("whiskers", "caps"):
        for ln in bp[el]:
            ln.set_color(BOX_EDGE); ln.set_linewidth(0.6)


def _mean_dot(ax, x, val):
    ax.plot(x, val, "o", color=PAL["ink"], ms=2.5, zorder=4)


def _label_mean_below(ax, x, val, ylim, fmt="{:.2f}"):
    """Mean caption pinned just above the x-axis at a shared height."""
    y = ylim[0] + 0.03 * (ylim[1] - ylim[0])
    ax.text(x, y, "mean=" + fmt.format(val), ha="center", va="bottom",
            fontsize=FONT["annot"], color=AXIS_GREY)


_PROJ = ccrs.Robinson(central_longitude=0)
_DATA_CRS = ccrs.PlateCarree()


def star(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def panel_letter(fig, cell, letter):
    """Place a bold panel letter at the top-left of a gridspec cell (fig coords,
    so it is unaffected by per-axis aspect/centering)."""
    bb = cell.get_position(fig)
    fig.text(bb.x0 - 0.012, bb.y1 + 0.008, letter, fontsize=FONT["panel"],
             fontweight="bold", va="bottom", ha="left")


def _thin_spines(ax, sides=("top", "right", "left", "bottom"), lw=0.6):
    for s in sides:
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color(AXIS_GREY)


def _v1_panel_a_cmap():
    """Cooling palette for panel a: the cold half of ``RdBu_r``."""
    cmap_full = plt.get_cmap("RdBu_r")
    half_colors = cmap_full(np.linspace(0, 0.5, 256))
    return mcolors.ListedColormap(half_colors, name="RdBu_r_cooling")


# ===========================================================================
# Panel a — global cooling-efficiency dot map
# ===========================================================================
def draw_panel_a(ax, fig):
    ce_map = pd.read_csv(CE_MAP_CSV)[["scene_id", "ce_01"]]
    pts = pd.read_csv(CITY_PTS_CSV)
    pts["ID"] = pts["ID"].astype(int)
    mp = ce_map.merge(pts, left_on="scene_id", right_on="ID", how="left")
    mp = mp.dropna(subset=["ce_01", "GCPNT_LON", "GCPNT_LAT"])

    # Keep the v2 data range, but use the v1 panel-a color ramp.
    blue_cmap = _v1_panel_a_cmap()
    vmin = float(np.floor(np.nanpercentile(mp["ce_01"], 2))) * 0.95
    vmax = 0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # White ocean and grey land from the packaged world-borders shapefile.
    ax.set_extent([-180, 180, -60, 78], crs=_DATA_CRS)
    ax.set_facecolor("white")
    country = gpd.read_file(COUNTRY_SHP)
    land = ShapelyFeature(country.geometry, _DATA_CRS, facecolor="#f0f0f0",
                          edgecolor="#aaaaaa", linewidth=0.2)
    ax.add_feature(land, zorder=1)

    ax.scatter(mp["GCPNT_LON"], mp["GCPNT_LAT"], transform=_DATA_CRS,
               c=mp["ce_01"].clip(vmin, vmax), cmap=blue_cmap, norm=norm,
               s=PANEL_A_POINT_SIZE, alpha=1, edgecolors="grey",
               linewidths=0.1, zorder=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    cax = inset_axes(ax, width="1.4%", height="34%", loc="lower left",
                     bbox_to_anchor=(0.14, 0.10, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0)
    sm = ScalarMappable(norm=norm, cmap=blue_cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical", extend="min")
    cb.set_label("CE (K per NDVI unit)", fontsize=FONT["annot"])
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=FONT["tick"], length=2, width=0.4)
    for spine in cb.ax.spines.values():
        spine.set_linewidth(0.4)


# ===========================================================================
# Panel b — embedding CE: extreme-group contrast  (dry/wet, low/high-GDP)
# ===========================================================================
def _mw(low, high):
    if len(low) < 3 or len(high) < 3:
        return np.nan
    _, p = stats.mannwhitneyu(low, high, alternative="two-sided")
    return p


def draw_panel_b(axes):
    m = pd.read_csv(GROUP_CSV)
    drivers = [
        dict(key="imerg_summer", div=1.0, lo="Dry", hi="Wet", unit="mm"),
        dict(key="gdp_mean", div=1e6, lo="Low-GDP", hi="High-GDP", unit="M Int$"),
    ]
    y_all = m["ce"].dropna()
    # Extra headroom for the significance bracket and footroom for mean captions.
    lo = min(float(y_all.quantile(0.01)), -12.0)
    hi = max(float(y_all.quantile(0.99)), 4.0)
    span0 = hi - lo
    ylim = (lo - 0.10 * span0, hi + 0.16 * span0)
    span = ylim[1] - ylim[0]

    for col, d in enumerate(drivers):
        ax = axes[col]
        k = d["key"]
        q25, q75 = m[k].quantile([0.25, 0.75])
        data = [m.loc[m[f"grp_{k}"] == g, "ce"].dropna().values for g in ("Low", "High")]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                        showfliers=False, medianprops=MEDIAN_KW)
        _style_box(bp, (GCOLORS["Low"], GCOLORS["High"]))
        for i, dv in enumerate(data):
            mean_val = np.mean(dv)
            _mean_dot(ax, i, mean_val)
            _label_mean_below(ax, i, mean_val, ylim)
        p = _mw(data[0], data[1])
        y_top = ylim[1] - 0.10 * span
        ax.plot([0, 0, 1, 1], [y_top - 0.025 * span, y_top, y_top, y_top - 0.025 * span],
                lw=0.8, color=AXIS_GREY)
        ax.text(0.5, y_top + 0.01 * span, star(p), ha="center", va="bottom",
                fontsize=FONT["title"])
        ax.axhline(0, color=ZERO_GREY, lw=0.6, ls="--")
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(*ylim)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"{d['lo']}\n(<{q25/d['div']:.0f} {d['unit']})",
             f"{d['hi']}\n(>{q75/d['div']:.0f} {d['unit']})"],
            fontsize=FONT["tick"])
        ax.tick_params(axis="y", labelsize=FONT["tick"])
        ax.set_title("Summer PRE" if col == 0 else "Mean GDP",
                     fontsize=FONT["title"] - 0.5)
        _thin_spines(ax)
        if col == 1:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel("CE (K per NDVI unit)", fontsize=FONT["label"])


# ===========================================================================
# Panel c — remote-buffer ΔLST by city size  (5 km buffer, city+buffer)
# ===========================================================================
def draw_panel_c(axes):
    RATIO_MIN = 0.75
    df = pd.read_csv(COOLING_CSV)
    df5 = df.loc[df["buffer_5km_ratio"] >= RATIO_MIN - 1e-9].copy().reset_index(drop=True)
    umf = df5["urban_mask_fraction"]
    p25, p75 = float(np.quantile(umf, 0.25)), float(np.quantile(umf, 0.75))
    df5["size_group"] = umf.apply(lambda u: "small" if u < p25 else "large" if u > p75 else "medium")

    scn_titles = {"buffer_5km": "5 km buffer", "both_5km": "city + buffer"}
    groups = ["small", "large"]

    # small-vs-large Mann-Whitney significance per scenario (star only)
    sig = {}
    for scn in scn_titles:
        col = f"dLST_{scn}_K"
        s = df5.loc[df5.size_group == "small", col].dropna().values
        l = df5.loc[df5.size_group == "large", col].dropna().values
        sig[scn] = star(stats.mannwhitneyu(s, l, alternative="two-sided")[1])

    # Shared y-range with footroom for mean captions and bracket headroom.
    cols = [f"dLST_{scn}_K" for scn in scn_titles]
    ymax = max(np.quantile(df5.loc[df5.size_group == g, c].dropna(), 0.99)
               for c in cols for g in groups)
    ymin = min(np.quantile(df5.loc[df5.size_group == g, c].dropna(), 0.01)
               for c in cols for g in groups)
    span0 = ymax - ymin
    ylim = (ymin + 0.10 * span0, ymax + 0.21 * span0)
    span = ylim[1] - ylim[0]

    for ax, (scn, stitle) in zip(axes, scn_titles.items()):
        col = f"dLST_{scn}_K"
        data, labels = [], []
        for g in groups:
            v = df5.loc[df5.size_group == g, col].dropna().values
            data.append(v); labels.append(g)
        bp = ax.boxplot(data, labels=labels, showfliers=False,
                        patch_artist=True, widths=0.5,
                        medianprops=MEDIAN_KW)
        _style_box(bp, [SIZE_COLORS[g] for g in groups])
        for j, v in enumerate(data):
            mean_val = np.mean(v)
            _mean_dot(ax, j + 1, mean_val)
            _label_mean_below(ax, j + 1, mean_val, ylim, fmt="{:.3f}")
        # small-vs-large significance bracket (star only, like panel b)
        y_br = ymax + 0.10 * span0
        ax.plot([1, 1, 2, 2],
                [y_br - 0.025 * span, y_br, y_br, y_br - 0.025 * span],
                lw=0.8, color=AXIS_GREY, clip_on=False)
        ax.text(1.5, y_br + 0.01 * span, sig[scn], ha="center", va="bottom",
                fontsize=FONT["title"])
        ax.axhline(0, color=ZERO_GREY, lw=0.6, ls="--")
        ax.set_ylim(*ylim)
        ax.set_title(stitle, fontsize=FONT["title"] - 0.5)
        ax.tick_params(axis="both", labelsize=FONT["tick"])
        ax.grid(axis="y", ls=":", color=GRID_GREY, alpha=0.9, linewidth=0.5)
        _thin_spines(ax)
    axes[0].set_ylabel("ΔLST (K)", fontsize=FONT["label"])
    axes[1].tick_params(labelleft=False)


def main():
    # Assemble the three-panel layout.
    fig = plt.figure(figsize=(18 * CM, 13.6 * CM), facecolor="white")
    gs_top = fig.add_gridspec(1, 1, left=0.06, right=0.96, top=0.975, bottom=0.482)
    ax_a = fig.add_subplot(gs_top[0], projection=_PROJ)
    draw_panel_a(ax_a, fig)
    panel_letter(fig, gs_top[0], "a")

    gs_bot = fig.add_gridspec(1, 2, left=0.085, right=0.97, top=0.394, bottom=0.10,
                              wspace=0.2)
    gs_b = gs_bot[0, 0].subgridspec(1, 2, wspace=0.08)
    ax_b = [fig.add_subplot(gs_b[0]), fig.add_subplot(gs_b[1])]
    ax_b[1].sharey(ax_b[0])
    draw_panel_b(ax_b)
    panel_letter(fig, gs_bot[0, 0], "b")

    gs_c = gs_bot[0, 1].subgridspec(1, 2, wspace=0.08)
    ax_c = [fig.add_subplot(gs_c[0]), fig.add_subplot(gs_c[1])]
    ax_c[1].sharey(ax_c[0])
    draw_panel_c(ax_c)
    panel_letter(fig, gs_bot[0, 1], "c")

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    print("Saved:", OUT_PNG.as_posix())
    plt.close(fig)


if __name__ == "__main__":
    main()
