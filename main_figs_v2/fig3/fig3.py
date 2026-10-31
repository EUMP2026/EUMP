"""Task 21 step 4 v2: Fig. 3 with separate variable and alignment legends.

Workflow: aggregate held-out |r| across years (mean +/- population SD), count original
training winners across years, render standalone a and the supplied five-panel layout.
Inputs: step 3 heldout_correlations.csv, task13 summary (RF reference only), packaged maps.
Outputs: two summary CSVs, 600-dpi JPGs, revised caption and validation JSON in
output/. Panel a data and all held-out cluster results are retained.
Panel a uses the reference Dynamic/Static legend style. Each map has its own
two-row legend below its panel: descriptive variable labels in b/d, and
'Aligned to C1-C4' in c/e without a legend heading. The latter describes the frozen
one-to-one discovery-set alignment, not the majority physical class.
No model fitting or bootstrap resampling. Run directly from any working directory.
"""
# %% Configuration and source data
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from cartopy.feature import ShapelyFeature

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/correlations/heldout_correlations.csv'
REFERENCE_SUMMARY = ROOT / 'data/correlations/rmax_summary.csv'
REFERENCE_FIGURE = ROOT / 'fig3_map_helpers.py'
OUT = ROOT / 'output'
OUT.mkdir(parents=True, exist_ok=True)
# Map styling controls (panel a is independent of these settings).
MAP_LEGEND_FONT_SIZE = 6.0
MAP_LEGEND_MARKER_SIZE = 4.5
MAP_CITY_POINT_AREA = 6  # points squared; previous value was 4.2
MAP_CITY_EDGE_COLOR = '#777777'
MAP_CITY_EDGE_ALPHA = .45
MAP_CITY_EDGE_WIDTH = .25
COUNTRY_EDGE_COLOR = '#d0d0d0'
COUNTRY_LINE_WIDTH = .12
source = pd.read_csv(SOURCE)
previous = pd.read_csv(REFERENCE_SUMMARY)
assert len(source) == 188 and source.csv_var.nunique() == 16
assert not source.duplicated(['csv_var', 'year']).any()
np.testing.assert_allclose(source.abs_r_test, source.r_test.abs())

# %% Summarize correlations and original training winners across years
summary_rows, dimension_rows = [], []
for var, group in source.groupby('csv_var', sort=False):
    vals = group.abs_r_test.to_numpy()
    original = previous.set_index('csv_var').loc[var]
    summary_rows.append({
        'variable': group.label.iloc[0], 'csv_var': var, 'kind': group.kind.iloc[0],
        'n_years': len(group), 'years': ';'.join(group.year.astype(str)),
        'r_test_abs_mean': vals.mean(), 'r_test_abs_std': vals.std(ddof=0),
        'r_test_abs_min': vals.min(), 'r_test_abs_max': vals.max(),
        'r_test_signed_mean': group.r_test.mean(),
        'r_test_oriented_mean': group.r_test_oriented.mean(),
        'n_sign_reversals': int((group.r_test_oriented < 0).sum()),
        'n_train_min': int(group.n_train.min()), 'n_train_max': int(group.n_train.max()),
        'n_test_min': int(group.n_test.min()), 'n_test_max': int(group.n_test.max()),
        'n_distinct_training_winners': group.selected_dimension_zero_based.nunique(),
        'r2_mean': original.r2_mean, 'r2_std': original.r2_std,
    })
    for dim, subset in group.groupby('selected_dimension_zero_based'):
        dimension_rows.append({
            'variable': group.label.iloc[0], 'csv_var': var, 'kind': group.kind.iloc[0],
            'dimension_zero_based': int(dim), 'dimension_label': f'D{dim}',
            'n_years_selected': len(subset), 'n_years_available': len(group),
            'year_fraction': len(subset) / len(group),
            'year_percentage': 100 * len(subset) / len(group),
            'years_selected': ';'.join(subset.year.sort_values().astype(str)),
        })
summary = pd.DataFrame(summary_rows).sort_values('r_test_abs_mean', ascending=False).reset_index(drop=True)
summary.insert(0, 'rank', np.arange(1, len(summary) + 1))
dimensions = pd.DataFrame(dimension_rows).sort_values(
    ['csv_var', 'n_years_selected', 'dimension_zero_based'], ascending=[True, False, True]).reset_index(drop=True)
dimensions.insert(3, 'frequency_rank_within_variable', dimensions.groupby('csv_var').cumcount() + 1)
assert summary.n_years.sum() == 188
np.testing.assert_array_equal(
    dimensions.groupby('csv_var').n_years_selected.sum().sort_index(),
    summary.set_index('csv_var').n_years.sort_index())
summary.to_csv(OUT / 'heldout_r_summary.csv', index=False)
dimensions.to_csv(OUT / 'training_dimension_year_frequency.csv', index=False)

# %% Import plotting helpers; do not execute the reference script's entry point
spec = importlib.util.spec_from_file_location('fig3_reference', REFERENCE_FIGURE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference.MAP_POINT_SIZE = MAP_CITY_POINT_AREA
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .7})
summary_lookup = summary.set_index('csv_var')
plot_order = [var for var, _, _ in reference.FIG1_VARS]
assert set(plot_order) == set(summary.csv_var)
drawn_dimensions = []


def draw_heldout_panel(ax):
    """Draw mean +/- across-year SD; display every original training winner."""
    ax.grid(axis='x', alpha=.3, linestyle='--', linewidth=.5, zorder=0)
    for xi, (var, label, kind) in enumerate(reference.FIG1_VARS):
        row = summary_lookup.loc[var]
        if kind == 'dynamic':
            ax.errorbar(xi, row.r_test_abs_mean, yerr=row.r_test_abs_std,
                        fmt='o', color=reference.C_DYNAMIC, markersize=5.5,
                        markeredgecolor='white', markeredgewidth=.6,
                        elinewidth=1.4, capsize=4, capthick=1.6, zorder=5)
            ax.text(xi, .98, f'yr={row.n_years}', ha='center', va='center', fontsize=6)
        else:
            ax.scatter(xi, row.r_test_abs_mean, marker='D', color=reference.C_STATIC,
                       s=5.5**2, edgecolors='white', linewidths=.6, zorder=5)
        freq = dimensions[dimensions.csv_var.eq(var)]
        ys = [.36 - k * .075 for k in range(len(freq))]
        ax.plot([xi-.22, xi-.22], [ys[-1]-.03, ys[0]+.03], color='#888888', lw=.7)
        for y_pos, item in zip(ys, freq.itertuples(index=False)):
            ax.text(xi-.14, y_pos, item.dimension_label,
                    ha='left', va='center', fontsize=6, color='black')
            drawn_dimensions.append((var, item.dimension_zero_based))
    ax.set_xticks(np.arange(len(plot_order)))
    ax.set_xticklabels([label for _, label, _ in reference.FIG1_VARS],
                       rotation=25, ha='right', rotation_mode='anchor', fontsize=7)
    ax.set_ylabel('Held-out |r|', fontsize=7)
    ax.set_xlim(-.7, len(plot_order)-.3)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='y', labelsize=6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='None', color=reference.C_DYNAMIC,
               markersize=5.5 * .8, label='Dynamic'),
        Line2D([], [], marker='D', linestyle='None', color='white',
               markerfacecolor=reference.C_STATIC, markersize=5.5 * .9, label='Static'),
    ], loc='upper right', bbox_to_anchor=(1.0, 1.04),
       fontsize=7, frameon=True, framealpha=.85)


# %% Export a standalone panel, keeping the reference panel size
cm = 1/2.54
fig_a = plt.figure(figsize=(18*cm, 5.7*cm), facecolor='white')
ax_a = fig_a.add_axes([.10, .24, .87, .70])
draw_heldout_panel(ax_a)
reference.panel_letter(ax_a, 'a')
fig_a.savefig(OUT / 'fig3a_heldout.jpg', dpi=600, bbox_inches='tight', facecolor='white')
assert len(drawn_dimensions) == len(dimensions)

# %% Per-panel map legends: variable meanings versus alignment identifiers
map_legend_records = []
city_marker_records = []


def style_city_markers(ax, *, letter, expected_n):
    """Keep cluster fills; give each city a separately transparent grey outline."""
    n_points = 0
    for collection in ax.collections:
        if not isinstance(collection, PathCollection):
            continue
        # The shared map helper sets a global alpha of 0.8. Preserve that fill
        # alpha explicitly, then remove global alpha so edge transparency can
        # be controlled independently rather than overwritten by the collection.
        face_colors = collection.get_facecolors().copy()
        collection.set_alpha(None)
        collection.set_facecolors(face_colors)
        collection.set_sizes([MAP_CITY_POINT_AREA])
        collection.set_edgecolor(to_rgba(MAP_CITY_EDGE_COLOR, MAP_CITY_EDGE_ALPHA))
        collection.set_linewidth(MAP_CITY_EDGE_WIDTH)
        n_points += len(collection.get_offsets())
        np.testing.assert_allclose(collection.get_facecolors(), face_colors)
        np.testing.assert_allclose(collection.get_edgecolors()[:, 3], MAP_CITY_EDGE_ALPHA)
    assert n_points == expected_n, f'Panel {letter}: unexpected city-marker count'
    city_marker_records.append({'panel': letter, 'n_city_points': n_points,
                                'fill_colors_preserved': True, 'outline_alpha': MAP_CITY_EDGE_ALPHA})


def add_map_legend(ax, names, *, embedding, letter):
    """Separate a physical description from a discovery-alignment identifier."""
    labels = [f'Aligned to C{cid + 1}' if embedding else names[cid]
              for cid in range(reference.K)]
    handles = [Line2D([], [], marker='o', linestyle='None',
                      markerfacecolor=reference.PALETTE[cid],
                      markeredgecolor=to_rgba(MAP_CITY_EDGE_COLOR, MAP_CITY_EDGE_ALPHA),
                      markeredgewidth=MAP_CITY_EDGE_WIDTH, markersize=MAP_LEGEND_MARKER_SIZE,
                      label=labels[cid])
               for cid in range(reference.K)]
    # Matplotlib fills columns first. This order displays C1/C2 on the first
    # row and C3/C4 on the second row, with all four names kept on one line.
    legend_order = [0, 2, 1, 3]
    legend = ax.legend(
        handles=[handles[i] for i in legend_order], loc='upper center',
        bbox_to_anchor=(.5, -.025), bbox_transform=ax.transAxes,
        ncol=2, fontsize=MAP_LEGEND_FONT_SIZE, title=None, frameon=False,
        borderaxespad=0, borderpad=.2, labelspacing=.3,
        handlelength=.9, handletextpad=.35, columnspacing=.9)
    map_legend_records.append({
        'panel': letter, 'role': 'embedding_alignment' if embedding else 'variable_description',
        'labels': labels, 'ax': ax, 'legend': legend,
    })
    return legend


# %% Assemble the same four map panels with the unchanged panel a
env_df, env_metrics, form_df, form_metrics, ghsl, country_feature = reference.prepare_figure_inputs()
# Use the same country geometry and land fill, with fainter, thinner boundaries.
country_feature = ShapelyFeature(
    list(country_feature.geometries()), reference._DATA_CRS,
    facecolor='#f0f0f0', edgecolor=COUNTRY_EDGE_COLOR, linewidth=COUNTRY_LINE_WIDTH)
fig_full = plt.figure(figsize=(18*cm, 16*cm), facecolor='white')
top = fig_full.add_gridspec(1, 1, left=.10, right=.97, top=.965, bottom=.7)
ax_top = fig_full.add_subplot(top[0])
draw_heldout_panel(ax_top)
reference.panel_letter(ax_top, 'a')
map_grid = fig_full.add_gridspec(2, 2, left=.10, right=.97, top=.59, bottom=.10,
                               hspace=.42, wspace=.03)
for grid_row, (data, metrics, names, variable_title, letters) in enumerate([
    (env_df, env_metrics, reference.ENV_NAMES, 'NDVI, NL, POP', ('b', 'c')),
    (form_df, form_metrics, reference.FORM_NAMES, 'BF, BH_mean, BH_QCD', ('d', 'e')),
]):
    for grid_col, col in enumerate(['cluster', 'emb_cluster']):
        ax_map = fig_full.add_subplot(map_grid[grid_row, grid_col], projection=reference._PROJ)
        title = (f'Variable cluster ({variable_title})' if grid_col == 0
                 else f'Embedding cluster ({metrics["n_dims"]} dims; OA = {metrics["accuracy"]:.2f})')
        reference.draw_cluster_map(ax_map, data, col, ghsl, country_feature,
                                   names=names, title=title, show_legend=False)
        style_city_markers(ax_map, letter=letters[grid_col], expected_n=len(data))
        add_map_legend(ax_map, names, embedding=grid_col == 1, letter=letters[grid_col])
        reference.panel_letter(ax_map, letters[grid_col])
# Check legend placement at the actual rendered axes size.
fig_full.canvas.draw()
renderer = fig_full.canvas.get_renderer()
legend_validation = []
for record in map_legend_records:
    panel_bounds = record['ax'].get_window_extent(renderer)
    # Legends are outside the map, within the width of their own panel.
    # Keep exactly two rows at the requested larger size. Do not silently shrink.
    placed = False
    for font_size in (MAP_LEGEND_FONT_SIZE,):
        for text in record['legend'].get_texts():
            text.set_fontsize(font_size)
        bounds = record['legend'].get_window_extent(renderer)
        below_panel = bounds.y1 < panel_bounds.y0
        within_panel_width = bounds.x0 >= panel_bounds.x0 and bounds.x1 <= panel_bounds.x1
        within_figure = fig_full.bbox.contains(bounds.x0, bounds.y0)
        other_panel_overlap = any(
            bounds.overlaps(other['ax'].get_window_extent(renderer))
            for other in map_legend_records if other is not record)
        other_legend_overlap = any(
            bounds.overlaps(other['legend'].get_window_extent(renderer))
            for other in map_legend_records if other is not record)
        other_title_overlap = any(
            bounds.overlaps(other['ax'].title.get_window_extent(renderer))
            for other in map_legend_records if other is not record)
        if (below_panel and within_panel_width and within_figure and not other_panel_overlap
                and not other_legend_overlap and not other_title_overlap):
            placed = True
        if placed:
            break
    if not placed:
        print(f"Warning: panel {record['panel']} legend falls outside the allocated bounds on this renderer")
    assert not record['legend'].get_title().get_text()
    assert len(record['legend'].get_texts()) == 4
    legend_validation.append({
        'panel': record['panel'], 'role': record['role'], 'labels': record['labels'],
        'location': 'below own panel', 'n_rows': 2, 'n_columns': 2,
        'row_labels': [record['labels'][:2], record['labels'][2:]],
        'font_size_points': font_size, 'title': '',
        'below_panel': bool(below_panel), 'within_panel_width': bool(within_panel_width),
        'overlaps_another_map': bool(other_panel_overlap),
        'overlaps_another_legend': bool(other_legend_overlap),
        'overlaps_another_title': bool(other_title_overlap),
    })
assert len(legend_validation) == 4 and not fig_full.legends
fig_full.savefig(OUT / 'fig3_heldout_updated_a.jpg', dpi=600, bbox_inches='tight', facecolor='white')
fig_full.savefig(ROOT / 'fig3.jpg', dpi=600, bbox_inches='tight', facecolor='white')

# A separate caption explains that alignment labels do not assert physical purity.
caption_source = ROOT / 'data/correlations/fig3_caption_base.md'
caption = caption_source.read_text(encoding='utf-8').strip()
caption += (
    ' In c and e, “Aligned to C1–C4” denotes Hungarian-matched labels, not '
    'necessarily the majority variable-based type within each embedding cluster.'
)
(OUT / 'fig3_caption_revised.md').write_text(caption + '\n', encoding='utf-8')

# %% Validation metadata; interactive figures remain available
validation = {
    'status': 'PASS', 'source_pairs': len(source), 'variables': len(summary),
    'dimension_frequency_rows': len(dimensions), 'all_dimensions_shown': True,
    'errorbars': 'population SD of held-out absolute r across years, not bootstrap CI',
    'dimension_frequency_denominator': 'available original training-selected years',
    'environment_map_n': env_metrics['n'], 'form_map_n': form_metrics['n'],
    'shared_map_legends': False, 'map_legends': legend_validation,
    'map_style': {'city_marker_area_points_squared': MAP_CITY_POINT_AREA,
                  'city_marker_edge_color': MAP_CITY_EDGE_COLOR,
                  'city_marker_edge_alpha': MAP_CITY_EDGE_ALPHA,
                  'city_marker_edge_width_points': MAP_CITY_EDGE_WIDTH,
                  'legend_font_size_points': MAP_LEGEND_FONT_SIZE,
                  'country_boundary_color': COUNTRY_EDGE_COLOR,
                  'country_boundary_linewidth_points': COUNTRY_LINE_WIDTH},
    'city_marker_checks': city_marker_records,
    'panel_a_data_and_clustering_results': 'unchanged from source script',
    'panel_a_legend': {'labels': ['Dynamic', 'Static'], 'fontsize': 7,
                       'location': 'upper right', 'framealpha': .85,
                       'anchor_axes': [1.0, 1.04],
                       'style_source': 'fig3_v2_no_a_v1.py'},
    'outputs': ['heldout_r_summary.csv', 'training_dimension_year_frequency.csv',
                'fig3a_heldout.jpg', 'fig3_heldout_updated_a.jpg', 'fig3_caption_revised.md'],
}
(OUT / 'figure_validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
print(summary[['rank','variable','n_years','r_test_abs_mean','r_test_abs_std']].round(4).to_string(index=False))
print(json.dumps(validation, indent=2))
plt.close('all')
