"""
Task 11 — urban-form clustering v4: 10 dims per variable (10×3 = 30 emb dims).

Objective
    Based on v2, lock the dim-per-variable count to 10. Each of BF, BH_mean,
    BH_QCD picks its top-10 |R| embedding dims (year 2012); the union (dedup)
    forms the embedding feature set (nominal 30, fewer after dedup). Cluster
    on those dims (k=4) and compare with the physical-index clustering.

Fixed setup (same as v2)
    - Indices : BF, BH_mean, BH_QCD   (k = 4, emb year 2012)
    - Physical clusters sorted BH↑ then BF↑, with C1/C2 swapped so the
      sparser low-rise is C1; custom names applied.

Method
    1. KMeans k=4 on standardised (BF, BH_mean, BH_QCD) → physical labels.
    2. Pearson R per index vs the 2012 embedding; top-10 dims each → union.
    3. KMeans k=4 on the standardised emb dims; Hungarian-align to physical.
    4. Consistency (ARI/NMI/Acc/Sil), purity, confusion, global map,
       per-cluster boxplots (index vs emb), and the physical-cluster
       scatter+boxplot view.

Inputs (relative to the package root)
    - data/urban_form_clusters.csv  (BF, BH_mean, BH_QCD)
    - data/embd/...260503_v4.npz    (1536-d emb, year 2012)
    - data/shp/GHSL_filtered_v1.shp + data/shp/TM_WORLD_BORDERS-0.3.shp

Outputs (output/task11_best_clusters_form_v4/)
    - selected_dims_v4.csv          : 10 dims/variable + R
    - task11_v4_confusion.png       : counts + row-normalised confusion
    - task11_v4_purity.csv          : per-cluster purity
    - task11_v4_global_map.png      : index (left) vs embedding (right)
    - task11_v4_var_boxplots.png    : per-cluster boxplots, index vs emb
    - task11_v4_index_scatter_box.png : physical k=4 scatter + boxplots
    - task11_v4_cluster_stats.csv   : per-cluster mean/median/std
    - urban_form_clusters_v4.csv    : per-city labels
"""

# %% [1] Setup
import os
os.environ['USE_PYGEOS'] = '0'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import gridspec
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, silhouette_score,
)
from scipy.optimize import linear_sum_assignment

# Package root: parent of code/ when run as a script; falls back to cwd in
# an interactive window (start the interactive session from the package root).
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd() if Path.cwd().name != 'code' else Path.cwd().parent
EMBD_PATH   = ROOT / 'data' / 'embd' / \
              'eump_reconstruction_best_checkpoint_embeddings_2003-2020_pooled_260503_v4.npz'
CLUSTER_CSV = ROOT / 'data' / 'urban_form_clusters.csv'
COUNTRY_SHP = ROOT / 'data' / 'shp' / 'TM_WORLD_BORDERS-0.3.shp'
GHSL_SHP    = ROOT / 'data' / 'shp' / 'GHSL_filtered_v1.shp'
OUT_DIR     = ROOT / 'output' / 'task11_best_clusters_form_v4'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_YEAR = 2012
K        = 4
SEED     = 42
TOP_N    = 10                                  # dims per variable → 10×3 = 30

INDICES = ['BF', 'BH_mean', 'BH_QCD']

# Tier thresholds (for the auto names, later overridden by custom names)
BH_TIERS  = [(5.0,  'Flat'),   (7.5,  'Low-rise'), (10.5, 'Mid-rise'), (np.inf, 'High-rise')]
BF_TIERS  = [(10.0, 'Sparse'), (17.0, 'Medium'),                       (np.inf, 'Dense')]
QCD_TIERS = [(0.30, 'Uniform'),(0.45, 'Mixed'),                        (np.inf, 'Heterogeneous')]

PALETTE = ['#1976D2', '#43A047', '#FB8C00', '#C62828']

# %% [2] Load metrics + embeddings (year=2012)
src = pd.read_csv(CLUSTER_CSV)
src['ID'] = src['ID'].astype('int64')

with np.load(EMBD_PATH, allow_pickle=False) as f:
    emb = f['embeddings'].astype('float32', copy=False)
    sid = f['scene_ids'].astype(str).astype('int64')
    yrs = f['years'].astype('int32', copy=False)
mask     = yrs == EMB_YEAR
emb_cols = [f'emb_{i:04d}' for i in range(emb.shape[1])]
emb_df   = pd.DataFrame(emb[mask], columns=emb_cols)
emb_df.insert(0, 'ID', sid[mask])

df = src.merge(emb_df, on='ID', how='inner').dropna(subset=INDICES).copy()
print(f'Working set: {len(df)} cities')

# %% [3] Pearson R per index → ranked dims; select top-10 each
E   = df[emb_cols].values.astype('float64')
E_z = (E - E.mean(0)) / E.std(0, ddof=0)
M   = df[INDICES].values.astype('float64')
M_z = (M - M.mean(0)) / M.std(0, ddof=0)
R   = (E_z.T @ M_z) / E_z.shape[0]                     # (1536, 3)
corr_df = pd.DataFrame(R, index=emb_cols, columns=INDICES)

ranked_per_index = {idx: corr_df[idx].abs().sort_values(ascending=False).index.tolist()
                    for idx in INDICES}

selected, sel_rows = [], []
for idx in INDICES:
    print(f'\n{idx} — top-{TOP_N} |R|:')
    for rank, d in enumerate(ranked_per_index[idx][:TOP_N]):
        r = corr_df.loc[d, idx]
        print(f'  #{rank+1:>2}  {d}  R={r:+.4f}')
        sel_rows.append({'index': idx, 'rank': rank + 1, 'dim': d,
                         'R': round(float(r), 4), 'in_final_set': d not in selected})
        if d not in selected:
            selected.append(d)
print(f'\nFinal embedding feature set: {len(selected)} dims (nominal {TOP_N}×{len(INDICES)}={TOP_N*len(INDICES)})')
pd.DataFrame(sel_rows).to_csv(OUT_DIR / 'selected_dims_v4.csv', index=False)
print('Saved: selected_dims_v4.csv')

# %% [4] Physical (index) clustering — k=4, C1/C2 swapped, custom names
def _tier_rank(val, tiers):
    for r, (thr, _) in enumerate(tiers):
        if val < thr:
            return r
    return len(tiers) - 1

def _tier_label(val, tiers):
    return tiers[_tier_rank(val, tiers)][1]

idx_scaler = StandardScaler()
Xi = idx_scaler.fit_transform(df[INDICES].values)
km_idx = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Xi)

centers_raw = idx_scaler.inverse_transform(km_idx.cluster_centers_)
sort_order  = np.lexsort((centers_raw[:, 0], centers_raw[:, 1]))   # BH primary, BF secondary
# within the two low-rise clusters, put the sparser one first (C1=Sprawling, C2=Compact)
sort_order  = sort_order.copy()
sort_order[[0, 1]] = sort_order[[1, 0]]
remap_idx   = {int(old): int(new) for new, old in enumerate(sort_order)}
df['cluster'] = [remap_idx[int(l)] for l in km_idx.labels_]

centers_df = pd.DataFrame(centers_raw[sort_order], columns=INDICES)
centers_df.index = range(K)
centers_df['n_cities'] = [int((df['cluster'] == i).sum()) for i in range(K)]
CUSTOM_TYPE_NAMES = {
    0: 'C1: Sprawling Low-rise',
    1: 'C2: Compact Low-rise',
    2: 'C3: Moderate Mid-rise',
    3: 'C4: Heterogeneous High-rise',
}
centers_df['type'] = [CUSTOM_TYPE_NAMES[i] for i in range(K)]
print('\nPhysical clusters (k=4):')
print(centers_df[['n_cities', 'BF', 'BH_mean', 'BH_QCD', 'type']].round(3).to_string())
y_idx = df['cluster'].values

# %% [5] Embedding clustering on the ~30 dims; Hungarian-align to physical
Xe = StandardScaler().fit_transform(df[selected].values)
km_emb = KMeans(n_clusters=K, n_init=50, random_state=SEED).fit(Xe)

cm_raw = confusion_matrix(y_idx, km_emb.labels_, labels=list(range(K)))
_, col_ind = linear_sum_assignment(-cm_raw)
remap = {int(raw): int(idx) for idx, raw in enumerate(col_ind)}
df['emb_cluster'] = [remap[int(l)] for l in km_emb.labels_]
y_emb = df['emb_cluster'].values

cm      = confusion_matrix(y_idx, y_emb, labels=list(range(K)))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
ari = adjusted_rand_score(y_idx, y_emb)
nmi = normalized_mutual_info_score(y_idx, y_emb)
acc = cm.diagonal().sum() / cm.sum()
sil = silhouette_score(Xe, y_emb)
print(f'\nConsistency ({len(selected)} dims): '
      f'ARI={ari:.4f}  NMI={nmi:.4f}  Acc={acc:.4f}  Sil(emb)={sil:.4f}')

print('\nPer-cluster purity (index → dominant emb cluster):')
print(f'  {"Type":<32} {"n":>5}  {"purity":>7}  dominant')
purity_rows = []
for cid in range(K):
    sub = pd.Series(y_emb[y_idx == cid])
    purity   = sub.value_counts().iloc[0] / len(sub)
    dominant = int(sub.value_counts().index[0])
    print(f'  {centers_df.loc[cid, "type"]:<32} {len(sub):>5}  {purity:>7.2f}  C{dominant+1}')
    purity_rows.append({'cluster': cid, 'type': centers_df.loc[cid, 'type'],
                        'n': int(len(sub)), 'purity': round(float(purity), 3),
                        'dominant_emb_cluster': dominant + 1})
pd.DataFrame(purity_rows).to_csv(OUT_DIR / 'task11_v4_purity.csv', index=False)
print('Saved: task11_v4_purity.csv')

# %% [6] Confusion matrix figure
tick_short = [centers_df.loc[i, 'type'].split(': ', 1)[1] for i in range(K)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, mat, ttl, fmt in [(axes[0], cm, 'Counts', 'd'),
                          (axes[1], cm_norm, 'Row-normalised', '.2f')]:
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    ax.set_xticks(range(K)); ax.set_xticklabels([f'E{i+1}' for i in range(K)], fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(tick_short, fontsize=8)
    ax.set_xlabel('Emb cluster (aligned)', fontsize=10)
    ax.set_ylabel('Index cluster (true)', fontsize=10)
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(K):
        for j in range(K):
            v = mat[i, j]
            ax.text(j, i, f'{v:{fmt}}', ha='center', va='center', fontsize=9,
                    color='white' if v > mat.max() * 0.6 else 'black')
plt.suptitle(f'v4 (10 dims/var → {len(selected)} emb dims): Index vs Embedding KMeans (k=4)\n'
             f'ARI={ari:.3f}  NMI={nmi:.3f}  Acc={acc:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'task11_v4_confusion.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v4_confusion.png')
plt.show()

# %% [7] Global map — index (left) vs embedding (right)
ghsl_gdf = gpd.read_file(GHSL_SHP)[['ID', 'GCPNT_LON', 'GCPNT_LAT']]
ghsl_gdf['ID'] = ghsl_gdf['ID'].astype('int64')
country_gdf = gpd.read_file(COUNTRY_SHP)
df_map = df.merge(ghsl_gdf, on='ID', how='left').dropna(subset=['GCPNT_LON'])
print(f'Map cities: {len(df_map)}')

PROJ     = ccrs.Robinson(central_longitude=0)
DATA_CRS = ccrs.PlateCarree()
country_feat = ShapelyFeature(country_gdf.geometry, DATA_CRS,
                              facecolor='#f0f0f0', edgecolor='#aaaaaa', linewidth=0.2)

fig = plt.figure(figsize=(18, 6), facecolor='white')
gs  = gridspec.GridSpec(1, 2, wspace=0.06)
panels = [
    ('cluster',     'Index cluster (BF, BH_mean, BH_QCD)', 'a'),
    ('emb_cluster', f'Embedding cluster — 10 dims/var ({len(selected)} dims, yr 2012)\n'
                    f'ARI={ari:.3f}  NMI={nmi:.3f}  Acc={acc:.3f}', 'b'),
]
for i, (col, title, panel_label) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i], projection=PROJ)
    ax.set_extent([-180, 180, -65, 78], crs=DATA_CRS)
    ax.set_facecolor('#d8ecf5')
    ax.add_feature(country_feat, zorder=1)
    ax.coastlines(linewidth=0.35, color='#555', zorder=3)
    n_plotted = 0
    for cid in range(K):
        mk = df_map[col] == cid
        if mk.sum() == 0:
            continue
        ax.scatter(df_map.loc[mk, 'GCPNT_LON'], df_map.loc[mk, 'GCPNT_LAT'],
                   transform=DATA_CRS, s=14, alpha=0.85,
                   color=PALETTE[cid], edgecolors='none', zorder=5)
        n_plotted += int(mk.sum())
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6, linespacing=1.4)
    ax.text(0.01, 0.97, panel_label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
    if i == 0:
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[cid],
                          markersize=8, label=centers_df.loc[cid, 'type']) for cid in range(K)]
        ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02),
                  fontsize=8, framealpha=0.85, ncol=2, borderpad=0.5,
                  handletextpad=0.4, labelspacing=0.3, columnspacing=0.9)
    ax.text(0.99, 0.02, f'n = {n_plotted}', transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom', color='#555',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none'))
plt.suptitle('Urban-form clustering v4 — Index (left) vs Embedding 10×3 (right)',
             fontsize=13, fontweight='bold', y=1.04)
fig.savefig(OUT_DIR / 'task11_v4_global_map.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: task11_v4_global_map.png')
plt.show()

# %% [8] Per-cluster boxplots — index vs embedding side by side
VAR_LABELS = {'BF': 'BF (%)', 'BH_mean': 'BH_mean (m)', 'BH_QCD': 'BH_QCD'}
LABEL_SYSTEMS = [
    ('Index clustering',     'cluster',     -0.20, ''),
    ('Embedding 10×3',       'emb_cluster',  0.20, '///'),
]
BOX_W = 0.34
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
for ax, var in zip(axes, INDICES):
    for sys_name, col, offset, hatch in LABEL_SYSTEMS:
        data = [df[df[col] == cid][var].values for cid in range(K)]
        bp = ax.boxplot(data, positions=np.arange(K) + offset, widths=BOX_W,
                        patch_artist=True, showmeans=True,
                        medianprops=dict(color='black', lw=1.3),
                        meanprops=dict(marker='D', markerfacecolor='red',
                                       markeredgecolor='red', markersize=4),
                        flierprops=dict(marker='.', ms=3, alpha=0.4))
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color); patch.set_alpha(0.75); patch.set_hatch(hatch)
    ax.set_xticks(range(K)); ax.set_xticklabels([f'C{i+1}' for i in range(K)], fontsize=10)
    ax.set_ylabel(VAR_LABELS[var], fontsize=11)
    ax.set_title(var, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
legend_handles = [
    Patch(facecolor='gray', alpha=0.75, label='Index clustering'),
    Patch(facecolor='gray', alpha=0.75, hatch='///', label='Embedding 10×3'),
]
fig.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.9)
plt.suptitle('Per-cluster variable distribution — index vs embedding (side by side)\n'
             'left box = index, right (hatched) = embedding 10×3  |  red ◆ = mean',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / 'task11_v4_var_boxplots.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v4_var_boxplots.png')
plt.show()

# %% [9] Physical k=4 clustering view — pairwise scatter + per-cluster boxplots
AX_LAB = {'BF': 'BF (%)', 'BH_mean': 'BH_mean (m)', 'BH_QCD': 'BH_QCD'}
PAIRS  = [('BF', 'BH_mean'), ('BF', 'BH_QCD'), ('BH_mean', 'BH_QCD')]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, (xc, yc) in zip(axes[0], PAIRS):
    for cid in range(K):
        m = df['cluster'] == cid
        ax.scatter(df.loc[m, xc], df.loc[m, yc], s=16, alpha=0.55, color=PALETTE[cid])
    for cid in range(K):
        ax.scatter(centers_df.loc[cid, xc], centers_df.loc[cid, yc], s=230, marker='*',
                   color=PALETTE[cid], edgecolors='black', linewidth=0.8, zorder=5)
    ax.set_xlabel(AX_LAB[xc], fontsize=11); ax.set_ylabel(AX_LAB[yc], fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
for ax, var in zip(axes[1], INDICES):
    data = [df[df['cluster'] == cid][var].values for cid in range(K)]
    bp = ax.boxplot(data, positions=range(K), widths=0.6, patch_artist=True,
                    showmeans=True, medianprops=dict(color='black', lw=1.3),
                    meanprops=dict(marker='D', markerfacecolor='red',
                                   markeredgecolor='red', markersize=5),
                    flierprops=dict(marker='.', ms=3, alpha=0.4))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks(range(K)); ax.set_xticklabels([f'C{i+1}' for i in range(K)], fontsize=10)
    ax.set_ylabel(AX_LAB[var], fontsize=11); ax.set_title(var, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[i],
                  markersize=9, label=centers_df.loc[i, 'type']) for i in range(K)]
fig.legend(handles=handles, loc='upper center', ncol=K, bbox_to_anchor=(0.5, 1.0),
           fontsize=9, framealpha=0.85)
plt.suptitle('Physical k=4 clustering (BF, BH_mean, BH_QCD)\n'
             'pairwise scatter (top, ★ = centroid)  +  per-cluster boxplots (bottom)',
             fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(OUT_DIR / 'task11_v4_index_scatter_box.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v4_index_scatter_box.png')
plt.show()

# %% [10] Per-cluster stats + export labels
stat_rows = []
for sys_name, col, _o, _h in LABEL_SYSTEMS:
    for cid in range(K):
        sub = df[df[col] == cid]
        row = {'method': sys_name, 'cluster': f'C{cid+1}', 'n': len(sub)}
        for var in INDICES:
            row[f'{var}_mean']   = round(sub[var].mean(), 3)
            row[f'{var}_median'] = round(sub[var].median(), 3)
            row[f'{var}_std']    = round(sub[var].std(), 3)
        stat_rows.append(row)
stat_df = pd.DataFrame(stat_rows)
print('\n=== Per-cluster variable statistics ===')
print(stat_df.to_string(index=False))
stat_df.to_csv(OUT_DIR / 'task11_v4_cluster_stats.csv', index=False)
print('Saved: task11_v4_cluster_stats.csv')

type_map = dict(zip(range(K), centers_df['type']))
out = df[['ID'] + INDICES + ['cluster', 'emb_cluster']].copy()
out['cluster_type']     = out['cluster'].map(type_map)
out['emb_cluster_type'] = out['emb_cluster'].map(type_map)
out.to_csv(OUT_DIR / 'urban_form_clusters_v4.csv', index=False)
print(f'Saved: urban_form_clusters_v4.csv ({len(out)} rows)')
