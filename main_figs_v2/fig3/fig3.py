"""Render the held-out-city interpretation panels for manuscript Fig. 3.

The script reads only the packaged correlation, clustering, and shapefile data
under ``data/``. It performs no model fitting and writes ``fig3.jpg`` beside
the script at 600 dpi.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/correlations/heldout_correlations.csv'
REFERENCE_SUMMARY = ROOT / 'data/correlations/rmax_summary.csv'
REFERENCE_FIGURE = ROOT / 'fig3_map_helpers.py'
OUT_JPG = ROOT / 'fig3.jpg'


def main():
    source = pd.read_csv(SOURCE)
    previous = pd.read_csv(REFERENCE_SUMMARY)
    assert len(source) == 188 and source.csv_var.nunique() == 16
    assert not source.duplicated(['csv_var', 'year']).any()
    np.testing.assert_allclose(source.abs_r_test, source.r_test.abs())

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

    spec = importlib.util.spec_from_file_location('fig3_reference', REFERENCE_FIGURE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
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
                   markersize=4.4, label='Dynamic: mean +/- SD'),
            Line2D([], [], marker='D', linestyle='None', color=reference.C_STATIC,
                   markersize=4.4, label='Static'),
        ], loc='upper right', fontsize=6, frameon=False, handletextpad=.4)


    cm = 1/2.54
    env_df, env_metrics, form_df, form_metrics, ghsl, country_feature = reference.prepare_figure_inputs()
    fig_full = plt.figure(figsize=(18*cm, 16*cm), facecolor='white')
    top = fig_full.add_gridspec(1, 1, left=.10, right=.97, top=.965, bottom=.7)
    ax_top = fig_full.add_subplot(top[0])
    draw_heldout_panel(ax_top)
    reference.panel_letter(ax_top, 'a')
    map_grid = fig_full.add_gridspec(2, 2, left=.10, right=.97, top=.59, bottom=.05,
                                   hspace=.18, wspace=.03)
    for grid_row, (data, metrics, names, variable_title, letters) in enumerate([
        (env_df, env_metrics, reference.ENV_NAMES, 'NDVI, NL, POP', ('b', 'c')),
        (form_df, form_metrics, reference.FORM_NAMES, 'BF, BH_mean, BH_QCD', ('d', 'e')),
    ]):
        for grid_col, col in enumerate(['cluster', 'emb_cluster']):
            ax_map = fig_full.add_subplot(map_grid[grid_row, grid_col], projection=reference._PROJ)
            title = (f'Variable cluster ({variable_title}; test n = {metrics["n"]})' if grid_col == 0
                     else f'Embedding cluster ({metrics["n_dims"]} dims; test OA = {metrics["accuracy"]:.2f})')
            reference.draw_cluster_map(ax_map, data, col, ghsl, country_feature,
                                       names=names, title=title, show_legend=grid_col == 0)
            reference.panel_letter(ax_map, letters[grid_col])
    fig_full.savefig(OUT_JPG, dpi=600, bbox_inches='tight', facecolor='white')
    assert len(drawn_dimensions) == len(dimensions)
    print(f'Saved {OUT_JPG}')
    plt.close('all')


if __name__ == "__main__":
    main()
