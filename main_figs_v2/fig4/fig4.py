import argparse
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

EXPERIMENT_DIR = Path(__file__).resolve().parent


class RemapNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map NumPy 2.x internal path back to 1.x
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


# ----------------------
# Constants / Config
# ----------------------
# Fallback values only used if an older/incomplete planning package is missing
# per-item stats. Current run.py exports the required stats in each city result.
LEGACY_NDVI_STD = 3199.6713321801726
LEGACY_LST_STD = 382.0982046523313 * 0.02
LEGACY_POP_STD = 23.045971452623522
LEGACY_POP_MEAN = 8.28806420616652

FONTS = {"label": 6, "tick": 5, "title": 7}

CITY_INFO = {
    782: {"city": "Beijing", "country": "China"},
    339: {"city": "Paris", "country": "France"},
    216: {"city": "New York", "country": "United States"},
    475: {"city": "Johannesburg", "country": "South Africa"},
    261: {"city": "Sao Paulo", "country": "Brazil"},
}

DEFAULT_DATA_FILE = "planning_data.pkl"
POP_RESAMPLE_AREA_SCALE = (250.0 / 92.77) ** 2

# Manually tuned for the current five-city planning data package. These ranges keep the
# same variable comparable across cooling-only and joint rows while avoiding the
# very pale maps caused by a few extreme pixels.
FIG4_NDVI_VMAX = 0.2
FIG4_LST_ABS_LIMIT = 2.5
FIG4_POP_LINEAR_ABS_LIMIT = 280.0
FIG4_POP_LOG_ABS_LIMIT = 4.0

lst_cmap = "RdBu_r"

color_ndvi = [
    "#f7fcf5",
    "#e5f5e0",
    "#c7e9c0",
    "#a1d99b",
    "#74c476",
    "#41ab5d",
    "#238b45",
    "#006d2c",
    "#00441b",
]
ndvi_cmap = LinearSegmentedColormap.from_list("ndvi_green", color_ndvi)

pop_cmap = "RdBu_r"


# ----------------------
# Utilities
# ----------------------

def _manual_positive_bounds(vmax: float, *, fallback: float = 0.1) -> tuple[float, float]:
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return (0.0, float(vmax))


def _manual_symmetric_bounds(limit: float, *, fallback: float = 1.0) -> tuple[float, float]:
    if not np.isfinite(limit) or limit <= 0:
        limit = fallback
    limit = float(limit)
    return (-limit, limit)


def _compute_diff(adj, orig, scale: float, mask: np.ndarray) -> np.ndarray:
    if adj is None or orig is None:
        return np.full_like(mask, np.nan, dtype=float)
    diff = (np.array(adj) - np.array(orig)) * scale
    diff[mask] = np.nan
    return diff


def _scalar(item: dict[str, Any], key: str, default: float) -> float:
    if key not in item:
        return default
    arr = np.asarray(item[key])
    if arr.size == 0:
        return default
    try:
        value = float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _text(item: dict[str, Any], key: str, default: str = "") -> str:
    if key not in item:
        return default
    arr = np.asarray(item[key])
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def _ndvi_divisor(item: dict[str, Any]) -> float:
    return _scalar(item, "ndvi_display_divisor", 1e4)


def _population_physical(values: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    pop_std = _scalar(item, "pop_std", LEGACY_POP_STD)
    pop_mean = _scalar(item, "pop_mean", LEGACY_POP_MEAN)
    pop_mode = _text(item, "pop_source_mode", "")
    physical = np.asarray(values) * pop_std + pop_mean
    if "log" in pop_mode.lower():
        physical = np.power(10.0, physical) - 1.0
    return physical


# ----------------------
# Data preparation
# ----------------------

def prepare_differences(data: list[dict[str, Any]], pop_display: str = "linear"):
    """Compute arrays for plotting and return them plus global bounds."""
    if pop_display not in {"linear", "log"}:
        raise ValueError(f"Unsupported pop_display={pop_display!r}; expected 'linear' or 'log'.")
    lists = {k: [] for k in ("cooling_ndvi", "cooling_lst", "ndvi", "lst", "pop")}
    city_pop_delta_pct = []

    for item in data:
        ocean = item["ocean_mask"]
        city_mask = item["mask"].squeeze().astype(bool)
        lst_scale = _scalar(item, "lst_std", LEGACY_LST_STD)
        ndvi_scale = _scalar(item, "ndvi_std", LEGACY_NDVI_STD)
        ndvi_divisor = _ndvi_divisor(item)

        d_ndvi = (item["adjusted_ndvi"] - item["original_ndvi"]) * ndvi_scale / ndvi_divisor
        d_lst = (item["adjusted_lst"] - item["original_lst"]) * lst_scale
        d_ndvi[ocean] = d_lst[ocean] = np.nan

        lists["ndvi"].append(d_ndvi)
        lists["lst"].append(d_lst)

        # population
        original_pop_physical = (
            _population_physical(item["original_pop"], item) * POP_RESAMPLE_AREA_SCALE
        )
        adjusted_pop_physical = (
            _population_physical(item["adjusted_pop"], item) * POP_RESAMPLE_AREA_SCALE
        )
        total_city_pop = np.nansum(original_pop_physical[city_mask])
        d_pop_linear = adjusted_pop_physical - original_pop_physical
        city_pop_delta = np.nansum(d_pop_linear[city_mask])
        city_pop_delta_pct.append(
            float(city_pop_delta / total_city_pop * 100.0)
            if total_city_pop != 0
            else float("nan")
        )
        if pop_display == "log":
            d_pop = np.sign(d_pop_linear) * np.log10(np.abs(d_pop_linear) + 1.0)
        else:
            d_pop = d_pop_linear
        d_pop[ocean] = np.nan
        lists["pop"].append(d_pop)

        # cooling diffs
        c_ndvi = _compute_diff(
            item.get("cooling_adjusted_ndvi"),
            item.get("original_ndvi"),
            ndvi_scale / ndvi_divisor,
            ocean,
        )
        c_lst = _compute_diff(
            item.get("cooling_adjusted_lst"),
            item.get("cooling_original_lst"),
            lst_scale,
            ocean,
        )
        lists["cooling_ndvi"].append(c_ndvi)
        lists["cooling_lst"].append(c_lst)

    pop_limit = (
        FIG4_POP_LOG_ABS_LIMIT if pop_display == "log" else FIG4_POP_LINEAR_ABS_LIMIT
    )
    vbounds = {
        "ndvi_pair": _manual_positive_bounds(FIG4_NDVI_VMAX, fallback=0.1),
        "lst_pair": _manual_symmetric_bounds(FIG4_LST_ABS_LIMIT, fallback=1.0),
        "pop": _manual_symmetric_bounds(pop_limit, fallback=1.0),
    }

    return (lists["cooling_ndvi"], lists["cooling_lst"], lists["ndvi"],
            lists["lst"], lists["pop"], city_pop_delta_pct, vbounds)

# ----------------------
# Plot helpers
# ----------------------

def annotate(ax, text: str):
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=FONTS["tick"], bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.3))


# Fixed 0-TCI_BAR_MAX_PCT % scale for the per-city TCI progress bars in the
# dedicated bottom row. The traffic reduction target is a uniform 20%, so the
# full bar length encodes the complete 20% goal and the filled length encodes
# the achieved drop pct = (original - adjusted)/original*100 toward that goal.
TCI_BAR_MAX_PCT = 20.0
# Progress-bar palette: a light "remaining-to-target" track plus a neutral
# steel-blue achieved fill that stays distinct from the NDVI/LST colormaps.
TCI_TRACK_COLOR = "#E6E6E6"   # light grey track (full 20% target)
TCI_FILL_COLOR = "#3D5A80"    # achieved reduction fill (steel blue)
# Horizontal padding (in pct units) on each side of the 0-20 bar so adjacent
# cities' bars keep clear separation inside their cells.
TCI_BAR_PAD = 2


def draw_tci_progress_bars(bar_axes, data):
    """Draw one horizontal TCI-reduction progress bar per city in the dedicated
    bottom row of block b.

    The full light track is the uniform 20% reduction target; the steel-blue
    fill is the achieved drop pct toward that target, labelled on the fill. A
    short black tick marks the 20% goal at the right end. The absolute ΔTCI is
    centered under each bar. Bars use square corners and side padding so
    neighbouring cities never touch.
    """
    bar_h = 0.35           # bar thickness in axes data coords
    y = 0.64               # vertical center of the bar

    for ax, item in zip(bar_axes, data):
        ax.set_axis_off()
        original = _scalar(item, "original_congestion", float("nan"))
        adjusted = _scalar(item, "adjusted_congestion", float("nan"))
        if not np.isfinite(original) or original <= 0:
            continue
        abs_drop = original - adjusted
        drop_pct = abs_drop / original * 100.0
        frac = float(np.clip(drop_pct, 0.0, TCI_BAR_MAX_PCT))

        ax.set_xlim(-TCI_BAR_PAD, TCI_BAR_MAX_PCT + TCI_BAR_PAD)
        ax.set_ylim(0.0, 1.0)

        # full-target track (square corners via rectangle)
        ax.barh(y, TCI_BAR_MAX_PCT, height=bar_h, left=0.0, align="center",
                color=TCI_TRACK_COLOR, edgecolor="none", zorder=1)
        # achieved fill toward the 20% target
        if frac > 0:
            ax.barh(y, frac, height=bar_h, left=0.0, align="center",
                    color=TCI_FILL_COLOR, edgecolor="none", zorder=2)

        # achieved percentage label: white on the fill when wide enough, else
        # black just past the fill end.
        label = f"{drop_pct:.1f}%"
        if frac >= 4.5:
            ax.text(frac / 2.0, y, label, ha="center", va="center",
                    fontsize=FONTS["tick"], color="white", fontweight="bold", zorder=3)
        else:
            ax.text(frac + 0.6, y, label, ha="left", va="center",
                    fontsize=FONTS["tick"], color="black", fontweight="bold", zorder=3)

        # short black tick at the 20% goal (right end of the track)
        ax.plot([TCI_BAR_MAX_PCT, TCI_BAR_MAX_PCT],
                [y - bar_h / 2 - 0.08, y + bar_h / 2 + 0.08],
                color="black", lw=0.6, solid_capstyle="butt", zorder=3)
        # absolute ΔTCI caption, centered under the bar
        ax.text(TCI_BAR_MAX_PCT / 2.0, y - bar_h / 2 - 0.18, f"ΔTCI",
                ha="center", va="top", fontsize=FONTS["tick"], color="black")


def add_city_labels(fig, axes, data, city_info):
    for idx, item in enumerate(data):
        ax = axes[idx, 0]
        pos = ax.get_position()
        center_x = pos.x0 + pos.width / 2
        city_id = int(item["ID"])
        label = city_info.get(city_id, {}).get("city", f"City {city_id}")
        fig.text(center_x, pos.y0 + pos.height + 0.01, label,
                 ha="center", va="bottom", fontsize=FONTS["label"], fontweight="bold")


def add_group_boxes(fig, axes, n_rows: int, bar_axes=None):
    groups = [[0, 1], [2, 3, 4]]
    labels = ["Cooling only", "Cooling & Release traffic congestion"]
    pad, dx = 0.005, [0.0003, 0.0008]

    for gidx, cols in enumerate(groups):
        xs, ys, x1s, y1s = [], [], [], []
        for col in cols:
            for r in range(n_rows):
                p = axes[r, col].get_position()
                xs.append(p.x0 + dx[gidx])
                ys.append(p.y0)
                x1s.append(p.x1 + dx[gidx])
                y1s.append(p.y1)
        # Block b (second group) extends down to enclose the TCI progress-bar
        # strip so the bars read as part of the joint-intervention block.
        if gidx == 1 and bar_axes:
            for ax in bar_axes:
                p = ax.get_position()
                ys.append(p.y0)
                y1s.append(p.y1)
        x0, y0 = min(xs) - pad, min(ys) - pad
        x1, y1 = max(x1s) + pad, max(y1s) + pad
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                         fill=False, linestyle="-", linewidth=0.4, edgecolor="black", zorder=20)
        fig.patches.append(rect)
        fig.text(x0 - pad - 0.025, y0 + (y1 - y0) / 2, labels[gidx], ha="center", va="center",
                 fontsize=FONTS["title"], fontweight="bold", rotation=90, transform=fig.transFigure)


def add_colorbars(
    fig,
    axes,
    vbounds: list[tuple[float, float]],
    cmaps: list[mcolors.Colormap],
    labels: list[str],
):
    # place colorbars to the right of right-most panel
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
        span = abs(float(vbounds[j][1]) - float(vbounds[j][0]))
        if span < 0.1:
            fmt = "%.3f"
        elif span < 1.0:
            fmt = "%.2f"
        elif span < 10.0:
            fmt = "%.1f"
        else:
            fmt = "%.0f"
        cbar.formatter = ticker.FormatStrFormatter(fmt)
        cbar.update_ticks()

# ----------------------
# Main grid plotting
# ----------------------

def plot_main_grids(axes, data, arrs_list, city_pop_delta_pct, vbounds, pop_display):
    ndvi_bounds = vbounds["ndvi_pair"]
    lst_bounds = vbounds["lst_pair"]
    pop_bounds = vbounds["pop"]
    pop_label = (
        "sign·log10(|ΔPOP|+1)"
        if pop_display == "log"
        else "ΔPOP"
    )

    col_vbounds = [
        ndvi_bounds,
        lst_bounds,
        ndvi_bounds,
        pop_bounds,
        lst_bounds,
    ]
    col_cmaps = [ndvi_cmap, lst_cmap, ndvi_cmap, pop_cmap, lst_cmap]
    col_labels = ["ΔNDVI", "ΔLST (K)", "ΔNDVI", pop_label, "ΔLST (K)"]

    for i, (item, arrs) in enumerate(zip(data, arrs_list)):
        mask = item["mask"].squeeze().astype(bool)
        for j, (arr, cmap, (vmin, vmax)) in enumerate(zip(arrs, col_cmaps, col_vbounds)):
            ax = axes[i, j]

            if i == 0:
                label_text = None
                if j == 0:
                    label_text = "a"
                elif j == 2:
                    label_text = "b"

                if label_text:
                    ax.text(-0.35, 1.0, label_text, transform=ax.transAxes,
                            fontsize=FONTS["title"], fontweight="bold", va="top", ha="left")

            ax.imshow(np.nan_to_num(arr, nan=0.0), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(mask, levels=[0.5], colors="#333333", linestyles=[(0, (8, 4))], linewidths=0.4, alpha=0.65, zorder=5)
            ax.axis("off")

            # annotations
            if j in (1, 4):
                val = np.nanmean(arr[mask])
                annotate(ax, f"ΔLST: {val:.2f}")
            elif j in (0, 2):
                annotate(ax, f"ΔNDVI: {np.nanmean(arr[mask]):.2f}")
            elif j == 3:
                pct = city_pop_delta_pct[i]
                annotate(ax, f"ΔPop: {pct:.2f}%" if np.isfinite(pct) else "ΔPop: n/a")

    return col_cmaps, col_vbounds, col_labels

# ----------------------
# Main figure
# ----------------------

def make_combined_figure(
    data: list[dict[str, Any]],
    save_path: str = "./fig4.png",
    pop_display: str = "linear",
):
    if not data:
        raise ValueError("No Fig4 spatial data to plot.")

    (cooling_ndvi, cooling_lst, ndvi, lst, pop, city_pop_delta_pct, vbounds) = (
        prepare_differences(data, pop_display=pop_display)
    )

    n_cities = len(data)
    n_panels = 5

    fig = plt.figure(figsize=(16 / 2.54, 16.8 / 2.54))

    gap_size = 0.15
    # rows 0-1: block a maps; row 2: inter-block gap; rows 3-5: block b maps;
    # row 6: small gap; row 7: dedicated per-city TCI progress-bar strip.
    bar_gap_size = 0.05
    bar_row_size = 0.30
    height_ratios = [1, 1, gap_size, 1, 1, 1, bar_gap_size, bar_row_size]

    gs_main = GridSpec(nrows=8, ncols=n_cities, figure=fig,
                       left=0.02, right=0.92, top=0.98, bottom=0.02,
                       hspace=0.02, wspace=0.04,
                       height_ratios=height_ratios)

    row_map = [0, 1, 3, 4, 5]

    axes_main = np.array([[fig.add_subplot(gs_main[row_map[p], c]) for p in range(n_panels)] for c in range(n_cities)])

    # dedicated TCI progress-bar row at the very bottom (one cell per city)
    bar_axes = [fig.add_subplot(gs_main[7, c]) for c in range(n_cities)]

    arrs_list = [
        [cooling_ndvi[i], cooling_lst[i], ndvi[i], pop[i], lst[i]]
        for i in range(n_cities)
    ]

    col_cmaps, col_vbounds, col_labels = plot_main_grids(
        axes_main,
        data,
        arrs_list,
        city_pop_delta_pct,
        vbounds,
        pop_display,
    )

    add_city_labels(fig, axes_main, data, CITY_INFO)
    draw_tci_progress_bars(bar_axes, data)
    add_group_boxes(fig, axes_main, n_cities, bar_axes=bar_axes)
    add_colorbars(fig, axes_main, col_vbounds, col_cmaps, col_labels)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {save_path}")

# ----------------------
# Script entry
# ----------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render planning Fig4 from planning_data.pkl.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=EXPERIMENT_DIR,
        help="Directory containing the planning data package.",
    )
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--cities", default=None, help="Comma-separated city IDs to plot/order.")
    parser.add_argument(
        "--pop-display",
        choices=("linear", "log"),
        default="linear",
        help="Display POP deltas as physical linear values or signed log10(|delta|+1).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    if argv is None:
        argv = sys.argv[1:]
        if "ipykernel" in sys.modules:
            filtered_argv = []
            skip_next = False
            for arg in argv:
                if skip_next:
                    skip_next = False
                    continue
                if arg in ("-f", "--f"):
                    skip_next = True
                    continue
                if arg.startswith(("-f=", "--f=")):
                    continue
                filtered_argv.append(arg)
            argv = filtered_argv
    return parser.parse_args(argv)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return RemapNumpyUnpickler(f).load()


def _array_field(data: dict[str, Any], key: str, *, context: str) -> np.ndarray:
    if key not in data:
        raise ValueError(f"{context} is missing required field {key!r}.")
    return np.asarray(data[key]).squeeze()


def _spatial_item_from_planning_item(item: dict[str, Any]) -> dict[str, Any]:
    city_id = int(item["city_id"])
    if "joint" not in item or "cooling_only" not in item:
        raise ValueError(
            f"City {city_id} is missing joint or cooling_only data. Fig4 requires "
            "planning_data.pkl generated with --mode both."
        )

    joint = item["joint"]
    cooling = item["cooling_only"]
    if not isinstance(joint, dict) or not isinstance(cooling, dict):
        raise ValueError(f"City {city_id} joint/cooling_only data must be dictionaries.")

    joint_context = f"city {city_id} joint data"
    cooling_context = f"city {city_id} cooling_only data"
    city_mask = _array_field(joint, "city_mask", context=joint_context).astype(bool)
    ocean_mask = _array_field(joint, "ocean_mask", context=joint_context).astype(bool)

    spatial_item: dict[str, Any] = {
        "ID": city_id,
        "mask": city_mask.astype(np.float32),
        "ocean_mask": ocean_mask,
        "original_ndvi": _array_field(joint, "original_ndvi", context=joint_context),
        "adjusted_ndvi": _array_field(joint, "adjusted_ndvi", context=joint_context),
        "original_pop": _array_field(joint, "original_pop", context=joint_context),
        "adjusted_pop": _array_field(joint, "adjusted_pop", context=joint_context),
        "original_lst": _array_field(joint, "original_lst", context=joint_context),
        "adjusted_lst": _array_field(joint, "adjusted_lst", context=joint_context),
        "original_congestion": _scalar(joint, "original_traffic", float("nan")),
        "adjusted_congestion": _scalar(joint, "adjusted_traffic", float("nan")),
        "lst_std": _scalar(joint, "lst_std", LEGACY_LST_STD),
        "ndvi_std": _scalar(joint, "ndvi_std", LEGACY_NDVI_STD),
        "ndvi_display_divisor": 1.0,
        "pop_mean": _scalar(joint, "pop_mean", LEGACY_POP_MEAN),
        "pop_std": _scalar(joint, "pop_std", LEGACY_POP_STD),
        "pop_source_mode": _text(joint, "pop_source_mode", ""),
        "cooling_adjusted_ndvi": _array_field(
            cooling, "adjusted_ndvi", context=cooling_context
        ),
        "cooling_original_lst": _array_field(cooling, "original_lst", context=cooling_context),
        "cooling_adjusted_lst": _array_field(cooling, "adjusted_lst", context=cooling_context),
    }

    return spatial_item


def _load_planning_package(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Planning data package not found: {path}. Re-run planning_exp/run.py "
            "with --mode both to generate planning_data.pkl."
        )
    package = _load_pickle(path)
    if not isinstance(package, dict):
        raise ValueError(
            f"Invalid planning data package: {path} must contain a dict with "
            "'schema_version' and 'city_data'."
        )
    if package.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported planning data schema_version={package.get('schema_version')!r} "
            f"in {path}; expected 1."
        )
    city_data = package.get("city_data")
    if not isinstance(city_data, list):
        raise ValueError(
            f"Invalid planning data package: {path} is missing list field 'city_data'."
        )

    data: list[dict[str, Any]] = []
    for item in city_data:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid city_data item in {path}: expected dict.")
        spatial_item = _spatial_item_from_planning_item(item)
        data.append(spatial_item)
    return data


def _filter_and_order(data: list[dict[str, Any]], cities: list[int] | None) -> list[dict[str, Any]]:
    if cities is None:
        return data
    order = {int(city_id): idx for idx, city_id in enumerate(cities)}
    filtered = [item for item in data if int(item["ID"]) in order]
    filtered.sort(key=lambda item: order[int(item["ID"])])
    return filtered


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir
    data_path = input_dir / args.data_file
    output_path = args.output or (input_dir / "fig4.png")

    data = _load_planning_package(data_path)
    cities = None
    if args.cities:
        cities = [int(part.strip()) for part in args.cities.split(",") if part.strip()]
    data = _filter_and_order(data, cities)
    make_combined_figure(
        data,
        save_path=str(output_path),
        pop_display=args.pop_display,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
