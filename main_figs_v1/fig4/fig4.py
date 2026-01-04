from pathlib import Path
import pickle, sys, types
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch

from scipy.signal import savgol_filter

class RemapNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map NumPy 2.x internal path back to 1.x
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

# ----------------------
# Constants / Config
# ----------------------
NDVI_MEAN = 15601.670767568561
NDVI_STD = 3199.6713321801726
LST_STD = 382.0982046523313 * 0.02
POP_STD = 23.045971452623522
POP_MEAN = 8.28806420616652

FONTS = {"label": 6, "tick": 5, "title": 7}
COLORS = {"dual_lst": "#37579B", "single_lst": "#C39029", "congestion": "#4C752D"}
COLORS = {"dual_lst": "#2172B4", "single_lst": "#69ADD4", "congestion": "#A06CA8"}
COLORS = {"dual_lst": "#94C594", "single_lst": "#9999FF", "congestion": "#FF4C4C"}

CITY_INFO = {
    782: {"city": "Beijing", "country": "China"},
    339: {"city": "Paris", "country": "France"},
    216: {"city": "New York", "country": "United States"},
    475: {"city": "Johannesburg", "country": "South Africa"},
    261: {"city": "Sao Paulo", "country": "Brazil"},
}

# Define RGB colors (normalized 0–1)
colors = [
    (209/255, 228/255, 212/255),  # light green (low NDVI)
    (148/255, 193/255, 157/255),
    (103/255, 170/255, 114/255),
    (64/255, 141/255, 73/255),
    (7/255, 107/255, 54/255),
    (14/255, 105/255, 58/255),    # dark green (high NDVI)
]

# Create continuous colormap
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_green_r", colors)

lst_cmap = 'RdBu_r'

color_ndvi = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b']
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_green", color_ndvi)

color_pop = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
pop_cmap = LinearSegmentedColormap.from_list('custom_cmap', color_pop)
pop_cmap = 'RdBu_r'


# ----------------------
# Utilities
# ----------------------

def half_cmap(name: str = "coolwarm", positive: bool = True) -> mcolors.Colormap:
    cmap = plt.cm.get_cmap(name)
    rng = np.linspace(0.5, 1.0, 256) if positive else np.linspace(0.0, 0.5, 256)
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_{'pos' if positive else 'neg'}", cmap(rng))


def safe_minmax(arrs: List[np.ndarray]) -> Tuple[float, float]:
    """Return (vmin, vmax) across arrays ignoring NaNs. Default (0,1) if empty."""
    mins, maxs = [], []
    for a in arrs:
        if a is None:
            continue
        vals = a[~np.isnan(a)]
        if vals.size:
            mins.append(float(np.nanmin(vals)))
            maxs.append(float(np.nanmax(vals)))
    return (0.0, 1.0) if not mins else (min(mins), max(maxs))


def _compute_diff(adj, orig, scale: float, mask: np.ndarray) -> np.ndarray:
    if adj is None or orig is None:
        return np.full_like(mask, np.nan, dtype=float)
    diff = (np.array(adj) - np.array(orig)) * scale
    diff[mask] = np.nan
    return diff

# ----------------------
# Data preparation
# ----------------------

def prepare_differences(data: List[Dict[str, Any]]):
    """Compute arrays for plotting and return them plus global bounds."""
    lists = {k: [] for k in ("target_lst", "cooling_ndvi", "cooling_lst", "ndvi", "lst", "pop")}
    pop_sum = []

    for item in data:
        ocean = item["ocean_mask"]

        # diffs: NDVI normalized to 1e4, LST scaled by LST_STD
        d_ndvi = (item["adjusted_ndvi"] - item["original_ndvi"]) * NDVI_STD / 1e4
        d_lst = (item["adjusted_lst"] - item["original_lst"]) * LST_STD
        d_ndvi[ocean] = d_lst[ocean] = np.nan

        lists["ndvi"].append(d_ndvi)
        lists["lst"].append(d_lst)

        # population
        pop_sum.append(np.nansum(item["original_pop"][~ocean] * POP_STD + POP_MEAN))
        d_pop = (item["adjusted_pop"] - item["original_pop"]) * POP_STD
        d_pop[ocean] = np.nan
        lists["pop"].append(d_pop)

        # target LST (if provided)
        t_raw = item.get("target_lst")
        if t_raw is None:
            lists["target_lst"].append(np.full_like(d_lst, np.nan))
        else:
            t = (np.array(t_raw) - item["original_lst"]) * LST_STD
            t[ocean] = np.nan
            lists["target_lst"].append(t)

        # cooling diffs
        c_ndvi = _compute_diff(item.get("cooling_adjusted_ndvi"), item.get("original_ndvi"), NDVI_STD / 1e4, ocean)
        c_lst = _compute_diff(item.get("cooling_adjusted_lst"), item.get("cooling_original_lst"), LST_STD, ocean)
        lists["cooling_ndvi"].append(c_ndvi)
        lists["cooling_lst"].append(c_lst)

    # global vmins/vmaxs
    v_target = safe_minmax(lists["target_lst"])
    v_cndvi = safe_minmax(lists["cooling_ndvi"])
    v_clst = safe_minmax(lists["cooling_lst"])
    v_ndvi = safe_minmax(lists["ndvi"])
    v_lst = safe_minmax(lists["lst"])
    v_pop = safe_minmax(lists["pop"])

    v_ndvi_pair = (min(v_ndvi[0], v_cndvi[0]), max(v_ndvi[1], v_cndvi[1]))
    v_lst_pair = (min(v_lst[0], v_clst[0]), max(v_lst[1], v_clst[1]))

    vbounds = {
        "target": v_target,
        "cndvi": v_cndvi,
        "clst": v_clst,
        "ndvi_pair": v_ndvi_pair,
        "lst_pair": v_lst_pair,
        "pop": v_pop,
    }

    return (lists["target_lst"], lists["cooling_ndvi"], lists["cooling_lst"],
            lists["ndvi"], lists["lst"], lists["pop"], pop_sum, vbounds)

# ----------------------
# Plot helpers
# ----------------------

def annotate(ax, text: str):
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=FONTS["tick"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.3))


def add_city_labels(fig, axes, data, city_info):
    for idx, item in enumerate(data):
        ax = axes[idx, 0]
        pos = ax.get_position()
        center_x = pos.x0 + pos.width / 2
        fig.text(center_x, pos.y0 + pos.height + 0.01, city_info[item["ID"]]["city"],
                 ha="center", va="bottom", fontsize=FONTS["label"], fontweight="bold")


def add_group_boxes(fig, axes, n_rows: int):
    groups = [[0], [1, 2], [3, 4, 5]]
    labels = ["Target", "Cooling only", "Cooling & Release traffic congestion"]
    pad, dx = 0.005, [0.0, 0.0003, 0.0008]

    for gidx, cols in enumerate(groups):
        xs, ys, x1s, y1s = [], [], [], []
        for col in cols:
            for r in range(n_rows):
                p = axes[r, col].get_position()
                xs.append(p.x0 + dx[gidx])
                ys.append(p.y0)
                x1s.append(p.x1 + dx[gidx])
                y1s.append(p.y1)
        x0, y0 = min(xs) - pad, min(ys) - pad
        x1, y1 = max(x1s) + pad, max(y1s) + pad
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                         fill=False, linestyle="-", linewidth=0.4, edgecolor="black", zorder=20)
        fig.patches.append(rect)
        fig.text(x0 - pad - 0.025, y0 + (y1 - y0) / 2, labels[gidx], ha="center", va="center",
                 fontsize=FONTS["title"], fontweight="bold", rotation=90, transform=fig.transFigure)


def add_colorbars(fig, axes, vbounds: Dict[str, Tuple[float, float]], cmaps: List[mcolors.Colormap], labels: List[str]):
    # place colorbars to the right of right-most panel
    n_rows, n_cols = axes.shape
    right_base = axes[-1, -1].get_position().x1
    gap, width = 0.015, 0.012

    for j in range(len(cmaps)):
        pos = axes[0, j].get_position()
        cax = fig.add_axes([right_base + gap, pos.y0 + 0.03 * pos.height, width, 0.94 * pos.height])
        norm = mcolors.Normalize(vmin=vbounds[j][0], vmax=vbounds[j][1])
        sm = cm.ScalarMappable(norm=norm, cmap=cmaps[j])
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label(labels[j], fontsize=FONTS["label"], rotation=90, labelpad=6,fontweight='bold')
        cbar.ax.tick_params(labelsize=FONTS["tick"], length=2.5, width=0.3, pad=2)
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(0.3)
        cbar.formatter = ticker.FormatStrFormatter("%.1f")
        cbar.update_ticks()

# ----------------------
# Main grid plotting
# ----------------------

def plot_main_grids(axes, data, arrs_list, pop_sum, vmax_ndvi, vmin_pop, vmax_pop):
    vmax_lst = 3
    vmax_ndvi = 0.45
    vmax_pop = 30
    
    col_vbounds = [(-vmax_lst, vmax_lst), (0, vmax_ndvi), (-1*vmax_lst, vmax_lst), (0, vmax_ndvi), (-vmax_pop, vmax_pop), (-vmax_lst, vmax_lst)]
    col_cmaps = [lst_cmap, ndvi_cmap, lst_cmap, ndvi_cmap, pop_cmap, lst_cmap]
    col_labels = ["ΔLST* (K)", "ΔNDVI", "ΔLST (K)", "ΔNDVI", "ΔPOP (%)", "ΔLST (K)"]

    target_cooling_list = []

    for i, (item, arrs) in enumerate(zip(data, arrs_list)):
        mask = item["mask"].squeeze().astype(bool)
        for j, (arr, cmap, (vmin, vmax)) in enumerate(zip(arrs, col_cmaps, col_vbounds)):
            ax = axes[i, j]
            
            # -----------------------------------------------------------------
            # CHANGE: Add labels a, b, c for the three main blocks in the grid
            # -----------------------------------------------------------------
            if i == 0:
                label_text = None
                # Block 1: Target (Row 0)
                if j == 0:
                    label_text = "a"
                # Block 2: Cooling only (Row 1 start)
                elif j == 1:
                    label_text = "b"
                # Block 3: Congestion (Row 3 start)
                elif j == 3:
                    label_text = "c"
                
                if label_text:
                    ax.text(-0.35, 1.0, label_text, transform=ax.transAxes, 
                            fontsize=FONTS["title"], fontweight="bold", va="top", ha="left")

            ax.imshow(np.nan_to_num(arr, nan=0.0), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(mask, levels=[0.5], colors="#333333", linestyles=[(0, (8, 4))], linewidths=0.4, alpha=0.65, zorder=5)
            ax.axis("off")

            # annotations
            if j in (0, 2, 5):
                val = np.nanmean(arr[mask])
                if j == 0:
                    lines = [f"ΔLST*: {val:.2f}"]
                    target_cooling_list.append(val)
                else:
                    lines = [f"ΔLST: {val:.2f}"]                    
                if j == 0:
                    lines.append("ΔTCI*: -0.2")
                elif j == 5:
                    cong = item.get("adjusted_congestion", 0) - item.get("original_congestion", 0)
                    lines.append(f"ΔTCI: {cong:.3f}")
                annotate(ax, "\n".join(lines))
            elif j in (1, 3):
                annotate(ax, f"ΔNDVI: {np.nanmean(arr[mask]):.2f}")
            elif j == 4:
                urban = np.nansum(arr[mask])
                rural = np.nansum(arr[~mask])
                total = pop_sum[i] if pop_sum[i] != 0 else 1.0
                annotate(ax, f"ΔPop: {urban/total*100:.2f}%")

    return col_cmaps, col_vbounds, col_labels, target_cooling_list

# ----------------------
# Trajectory row
# ----------------------

def plot_trajectory_row(axes, traj_items, target_cooling_list):
    for j, (ax, item) in enumerate(zip(axes, traj_items)):
        # prepare series
        ndvi0, lst0 = item["urban_ndvi0"], item["urban_lst0"]
        ndvi1, lst1 = item["urban_ndvi1"], item["urban_lst1"]
        ndvi_ds = np.array(item["delta_ndvi_traj"] + [ndvi1 - ndvi0]) / 1e4
        lst_ds = np.array(item["delta_lst_traj"] + [lst1 - lst0])
        con_ds = np.array(item["delta_congestion_traj"] + [item.get("congestion1", 0) - item.get("congestion0", 0)])
        
        target_cooling = item["urban_target_lst"] - item["urban_lst0"]

        cndvi0, clst0 = item.get("cooling_urban_ndvi0", 0), item.get("cooling_urban_lst0", 0)
        cndvi1, clst1 = item.get("cooling_urban_ndvi1", 0), item.get("cooling_urban_lst1", 0)
        ndvi_ds1 = np.array(item.get("cooling_delta_ndvi_traj", []) + [cndvi1 - cndvi0]) / 1e4
        lst_ds1 = np.array(item.get("cooling_delta_lst_traj", []) + [clst1 - clst0])

        ax2 = ax.twinx()
        
        # --- Smooth the time series ---       
        ndvi_ds_smooth = savgol_filter(ndvi_ds, window_length=21, polyorder=3)
        ndvi_ds1_smooth = savgol_filter(ndvi_ds1, window_length=21, polyorder=3)
        lst_ds_smooth = savgol_filter(lst_ds, window_length=21, polyorder=3)
        lst_ds1_smooth = savgol_filter(lst_ds1, window_length=21, polyorder=3)
        con_ds_smooth = savgol_filter(con_ds, window_length=21, polyorder=3)

        # Create index-based x-axis (Time steps)
        steps_dual = np.arange(len(lst_ds_smooth))
        steps_single = np.arange(len(lst_ds1_smooth))

        # --- Plot lines using steps as X ---
        l1, = ax.plot(steps_dual, lst_ds_smooth, color=COLORS["dual_lst"], linewidth=2, alpha=0.8,)
        l2, = ax.plot(steps_single, lst_ds1_smooth, color=COLORS["single_lst"], linewidth=2, alpha=0.8,)       
        l3, = ax2.plot(steps_dual, con_ds_smooth, color=COLORS["congestion"], linewidth=1.5, alpha=0.8,)

        # --- Add horizontal dashed line (e.g., baseline = 0) ---
        ax.axhline(y=target_cooling_list[j], color="k", linestyle="--", linewidth=0.6, alpha=0.8)
        
        ax.set_ylim(-1.4, 0)
        
        # Update X-axis label and format
        ax.set_xlabel("Time steps", fontsize=FONTS["label"], fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax.set_ylabel("ΔLST (K)", fontsize=FONTS["label"],fontweight="bold")
        ax.tick_params(axis="both", labelsize=FONTS["tick"], width=0.3)
        
        ax2.set_ylim(-0.07, 0)
        ax2.set_ylabel("ΔTCI", fontsize=FONTS["label"], color=COLORS["congestion"],fontweight="bold") 
        ax2.tick_params(axis="y", labelsize=FONTS["tick"], width=0.3, labelcolor=COLORS["congestion"]) 

        for sp in list(ax.spines.values()) + list(ax2.spines.values()):
            sp.set_linewidth(0.3)
            
        # --- Manage y-axis ticks and labels ---
        if j == 0:
            # ✅ First panel: keep left y-axis label/ticks, hide right
            ax2.set_ylabel("")  # remove right label
            ax2.tick_params(axis="y", labelright=False, right=False)
        elif j == len(axes) - 1:
            # ✅ Last panel: keep right y-axis label/ticks, hide left
            ax.set_ylabel("")  # remove left label
            ax.tick_params(axis="y", labelleft=False, left=False)
        else:
            # ✅ Middle panels: hide both y-axis labels and ticks
            ax.set_ylabel("")
            ax2.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False, left=False)
            ax2.tick_params(axis="y", labelright=False, right=False)

        # show fewer xticks for readability
        for i, label in enumerate(ax.get_xticklabels()):
            label.set_visible(i % 2 == 1)

        if j == 0:
            # -------------------------------------------------------------
            # CHANGE: Label 'd' for the trajectory row
            # -------------------------------------------------------------
            ax.text(-0.4, 0.9, "d", transform=ax.transAxes, fontsize=FONTS["title"], fontweight="bold")
            
            ax.legend([l2, l1, l3], ["ΔLST-Single", "ΔLST-Dual", "ΔTCI-Dual"], loc="upper right",
                      fontsize=FONTS["tick"], frameon=False, framealpha=0.4, facecolor="white", edgecolor="gray", handlelength=2)

# ----------------------
# Main figure
# ----------------------

def make_combined_figure(data: List[Dict[str, Any]], traj_data: List[Dict[str, Any]], save_path: str = "./fig4.png"):

    (target_lst, cooling_ndvi, cooling_lst, ndvi, lst, pop, pop_sum, vbounds) = prepare_differences(data)

    vmin_ndvi_pair, vmax_ndvi_pair = vbounds["ndvi_pair"]
    vmin_pop, vmax_pop = vbounds["pop"]

    # trajectory items ordered like data
    traj_map = {t["ID"]: t for t in traj_data}
    traj_items = [traj_map[item["ID"]] for item in data]

    n_cities = len(data)
    n_panels = 6

    fig = plt.figure(figsize=(16 / 2.54, 21 / 2.54))

    # 定义高度比例，插入间隔 (Gap)
    gap_size = 0.15
    height_ratios = [1, gap_size, 1, 1, gap_size, 1, 1, 1]
    
    gs_main = GridSpec(nrows=8, ncols=n_cities, figure=fig,
                       left=0.02, right=0.92, top=1, bottom=0.17, 
                       hspace=0.02, wspace=0.04, 
                       height_ratios=height_ratios)

    gs_traj = GridSpec(nrows=1, ncols=n_cities, figure=fig,
                       left=0.02, right=0.92, top=0.15, bottom=0, hspace=0.02, wspace=0.04)

    # 建立逻辑行到物理行的映射
    row_map = [0, 2, 3, 5, 6, 7]

    axes_main = np.array([[fig.add_subplot(gs_main[row_map[p], c]) for p in range(n_panels)] for c in range(n_cities)])
    axes_traj = [fig.add_subplot(gs_traj[0, j]) for j in range(n_cities)]

    arrs_list = [[target_lst[i], cooling_ndvi[i], cooling_lst[i], ndvi[i], pop[i], lst[i]] for i in range(n_cities)]

    col_cmaps, col_vbounds, col_labels, target_cooling_list  = plot_main_grids(
        axes_main, data, arrs_list, pop_sum, vmax_ndvi=vmax_ndvi_pair, vmin_pop=vmin_pop, vmax_pop=vmax_pop
    )

    add_city_labels(fig, axes_main, data, CITY_INFO)
    add_group_boxes(fig, axes_main, n_cities)
    add_colorbars(fig, axes_main, col_vbounds, col_cmaps, col_labels)
    plot_trajectory_row(axes_traj, traj_items, target_cooling_list)

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved figure to: {save_path}")

# ----------------------
# Script entry
# ----------------------
if __name__ == "__main__":
    
    with open("data_list_5cities_with_traj_lager_tv.pkl", "rb") as f:
        data = pickle.load(f)
        
    with open("data_list_1-11_pop_weighted.pkl", "rb") as f:
        traj_data = pickle.load(f)
        
    # filter/order according to CITY_INFO
    data = [item for item in data if item["ID"] in CITY_INFO]
    ordered_ids = list(CITY_INFO.keys())
    data.sort(key=lambda x: ordered_ids.index(x["ID"]))

    make_combined_figure(data, traj_data, save_path="./fig4.png")