"""
Task 11 — Urban-environment clustering, v3: 10 dims per index (10×3 = 30).

Objective
    Based on env v2 (which swept N), lock the dim-per-index count to 10.
    Each of NDVI_2020, NL_2020, log_POP picks its top-10 |R| embedding dims
    (year 2020); the union (dedup) forms the embedding feature set (nominal
    30, fewer after dedup). Cluster on those dims (k=4) and compare with the
    physical-index clustering.

Fixed setup
    - Indices : NDVI_2020, NL_2020, log_POP   (k = 4, emb year 2020)
    - Physical clusters sorted NL↑ then POP↑, tier-named (NL · density · Veg).

Method
    1. KMeans k=4 on standardised indices → physical labels.
    2. Pearson R per index vs the 2020 embedding; top-10 dims each → union.
    3. KMeans k=4 on the standardised emb dims; Hungarian-align to physical.
    4. Consistency (ARI/NMI/Acc/Sil), purity, confusion, global map,
       per-cluster boxplots (index vs emb).

Inputs (relative to the package root)
    - data/zonal_merged_2020.csv   (NDVI_2020, NL_2020, POP_mean_2020)
    - data/embd/...260503_v4.npz   (1536-d emb, year 2020)
    - data/shp/GHSL_filtered_v1.shp + data/shp/TM_WORLD_BORDERS-0.3.shp

Outputs (output/task11_corr3dim_env_v3/)
    - selected_dims_v3.csv          : 10 dims/index + R
    - task11_v3_confusion.png       : counts + row-normalised confusion
    - task11_v3_purity.csv          : per-cluster purity
    - task11_v3_global_map.png      : index (left) vs embedding (right)
    - task11_v3_var_boxplots.png    : per-cluster boxplots, index vs emb
    - task11_v3_cluster_stats.csv   : per-cluster mean/median/std
    - urban_env_clusters_v3.csv     : per-city labels
    - task11_v3_index_scatter_box.png : physical k=4 clustering shown as
                                      pairwise scatter + per-cluster boxplots
                                      (to aid group naming)
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
EMBD_PATH = ROOT / 'data' / 'embd' / \
            'eump_reconstruction_best_checkpoint_embeddings_2003-2020_pooled_260503_v4.npz'
CSV_PATH  = ROOT / 'data' / 'zonal_merged_2020.csv'
COUNTRY_SHP = ROOT / 'data' / 'shp' / 'TM_WORLD_BORDERS-0.3.shp'
GHSL_SHP    = ROOT / 'data' / 'shp' / 'GHSL_filtered_v1.shp'
OUT_DIR   = ROOT / 'output' / 'task11_corr3dim_env_v3'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_YEAR = 2020
K        = 4
SEED     = 42
TOP_N    = 10                                  # dims per index → 10×3 = 30 (deduped)

RAW_COLS = ['NDVI_2020', 'NL_2020', 'POP_mean_2020']
INDICES  = ['NDVI_2020', 'NL_2020', 'log_POP']

NL_TIERS   = [(40.0, 'Dim'), (58.0, 'Moderate'), (np.inf, 'Bright')]
POP_TIERS  = [(2.77, 'Low'), (3.58, 'Medium'),   (np.inf, 'High')]
NDVI_TIERS = [(0.35, 'Low'), (0.55, 'Medium'),   (np.inf, 'High')]

PALETTE = ['#1976D2', '#43A047', '#FB8C00', '#C62828']

# %% [2] Load metrics + embeddings (year=2020)
df_raw = pd.read_csv(CSV_PATH, usecols=['ID', 'split'] + RAW_COLS).dropna(subset=RAW_COLS).copy()
df_raw['log_POP'] = np.log1p(df_raw['POP_mean_2020'])
df_raw['ID'] = df_raw['ID'].astype('int64')

with np.load(EMBD_PATH, allow_pickle=False) as f:
    emb = f['embeddings'].astype('float32', copy=False)
    sid = f['scene_ids'].astype(str).astype('int64')
    yrs = f['years'].astype('int32', copy=False)
mask     = yrs == EMB_YEAR
emb_cols = [f'emb_{i:04d}' for i in range(emb.shape[1])]
emb_df   = pd.DataFrame(emb[mask], columns=emb_cols)
emb_df.insert(0, 'ID', sid[mask])

df = df_raw.merge(emb_df, on='ID', how='inner').copy()
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
pd.DataFrame(sel_rows).to_csv(OUT_DIR / 'selected_dims_v3.csv', index=False)
print('Saved: selected_dims_v3.csv')

# %% [4] Physical (index) clustering — k=4, sorted NL↑ then POP↑, tier-named
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

centers_raw = idx_scaler.inverse_transform(km_idx.cluster_centers_)   # [NDVI, NL, log_POP]
nl_ranks    = np.array([_tier_rank(c[1], NL_TIERS)  for c in centers_raw])
pop_ranks   = np.array([_tier_rank(c[2], POP_TIERS) for c in centers_raw])
sort_order  = np.lexsort((pop_ranks, nl_ranks))                       # NL primary, POP secondary
# manual reorder so the sequence reads: C1 dim-green, C2 bright-green-less-pop,
# C3 bright-populous, C4 bright-less-veg  (swap the two bright clusters by POP)
sort_order  = sort_order.copy()
sort_order[[1, 2]] = sort_order[[2, 1]]
remap_idx   = {int(old): int(new) for new, old in enumerate(sort_order)}
df['cluster'] = [remap_idx[int(l)] for l in km_idx.labels_]

centers_df = pd.DataFrame(centers_raw[sort_order], columns=INDICES)
centers_df.index = range(K)
centers_df['n_cities'] = [int((df['cluster'] == i).sum()) for i in range(K)]
centers_df['NDVI_tier'] = centers_df['NDVI_2020'].apply(lambda v: _tier_label(v, NDVI_TIERS))
centers_df['NL_tier']   = centers_df['NL_2020'].apply(lambda v: _tier_label(v, NL_TIERS))
centers_df['POP_tier']  = centers_df['log_POP'].apply(lambda v: _tier_label(v, POP_TIERS))
centers_df['type'] = [
    f'C{i+1}: {centers_df.loc[i, "NL_tier"]} NL · '
    f'{centers_df.loc[i, "POP_tier"]}-density · '
    f'{centers_df.loc[i, "NDVI_tier"]} Veg'
    for i in range(K)
]

# Custom data-driven names (clusters sorted NL↑ then POP↑):
#   C1 dim+high NDVI, C2 bright+high POP, C3 bright+high NDVI+low POP, C4 bright+low NDVI
# names: 3 words = (night-lights, vegetation, population) in fixed order
CUSTOM_TYPE_NAMES = {
    0: 'C1: Dim, Green, Mid-pop',
    1: 'C2: Bright, Green, Low-pop',
    2: 'C3: Bright, Semi-green, High-pop',
    3: 'C4: Bright, Barren, Varied-pop',
}
centers_df['type'] = [CUSTOM_TYPE_NAMES[i] for i in range(K)]

print('\nPhysical clusters (NDVI, NL, POP):')
print(centers_df[['n_cities', 'NDVI_2020', 'NL_2020', 'log_POP', 'type']].round(3).to_string())
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
print(f'  {"Type":<42} {"n":>5}  {"purity":>7}  dominant')
purity_rows = []
for cid in range(K):
    sub = pd.Series(y_emb[y_idx == cid])
    purity   = sub.value_counts().iloc[0] / len(sub)
    dominant = int(sub.value_counts().index[0])
    print(f'  {centers_df.loc[cid, "type"]:<42} {len(sub):>5}  {purity:>7.2f}  C{dominant+1}')
    purity_rows.append({'cluster': cid, 'type': centers_df.loc[cid, 'type'],
                        'n': int(len(sub)), 'purity': round(float(purity), 3),
                        'dominant_emb_cluster': dominant + 1})
pd.DataFrame(purity_rows).to_csv(OUT_DIR / 'task11_v3_purity.csv', index=False)
print('Saved: task11_v3_purity.csv')

# %% [6] Confusion matrix figure
tick_short = [centers_df.loc[i, 'type'].split(': ', 1)[1] for i in range(K)]
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, mat, ttl, fmt in [(axes[0], cm, 'Counts', 'd'),
                          (axes[1], cm_norm, 'Row-normalised', '.2f')]:
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    ax.set_xticks(range(K)); ax.set_xticklabels([f'E{i+1}' for i in range(K)], fontsize=9)
    ax.set_yticks(range(K)); ax.set_yticklabels(tick_short, fontsize=7.5)
    ax.set_xlabel('Emb cluster (aligned)', fontsize=10)
    ax.set_ylabel('Env-index cluster (true)', fontsize=10)
    ax.set_title(ttl, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(K):
        for j in range(K):
            v = mat[i, j]
            ax.text(j, i, f'{v:{fmt}}', ha='center', va='center', fontsize=9,
                    color='white' if v > mat.max() * 0.6 else 'black')
plt.suptitle(f'env v3 (10 dims/index → {len(selected)} emb dims): Index vs Embedding KMeans (k=4)\n'
             f'ARI={ari:.3f}  NMI={nmi:.3f}  Acc={acc:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(OUT_DIR / 'task11_v3_confusion.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v3_confusion.png')
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
    ('cluster',     'Index cluster (NDVI, NL, POP)', 'a'),
    ('emb_cluster', f'Embedding cluster — 10 dims/index ({len(selected)} dims, yr 2020)\n'
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
plt.suptitle('Urban-environment clustering v3 — Index (left) vs Embedding 10×3 (right)',
             fontsize=13, fontweight='bold', y=1.04)
fig.savefig(OUT_DIR / 'task11_v3_global_map.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved: task11_v3_global_map.png')
plt.show()

# %% [8] Per-cluster boxplots — index vs embedding side by side
VAR_LABELS = {'NDVI_2020': 'NDVI', 'NL_2020': 'NL', 'log_POP': 'log(POP)'}
LABEL_SYSTEMS = [
    ('Index clustering',  'cluster',     -0.20, ''),
    ('Embedding 10×3',    'emb_cluster',  0.20, '///'),
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
    ax.set_title(VAR_LABELS[var], fontsize=12, fontweight='bold')
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
fig.savefig(OUT_DIR / 'task11_v3_var_boxplots.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v3_var_boxplots.png')
plt.show()

# %% [9] Per-cluster stats + export labels
stat_rows = []
for sys_name, col, _o, _h in LABEL_SYSTEMS:
    for cid in range(K):
        sub = df[df[col] == cid]
        row = {'method': sys_name, 'cluster': f'C{cid+1}', 'n': len(sub)}
        for var in INDICES:
            row[f'{VAR_LABELS[var]}_mean']   = round(sub[var].mean(), 3)
            row[f'{VAR_LABELS[var]}_median'] = round(sub[var].median(), 3)
            row[f'{VAR_LABELS[var]}_std']    = round(sub[var].std(), 3)
        stat_rows.append(row)
stat_df = pd.DataFrame(stat_rows)
print('\n=== Per-cluster variable statistics ===')
print(stat_df.to_string(index=False))
stat_df.to_csv(OUT_DIR / 'task11_v3_cluster_stats.csv', index=False)
print('Saved: task11_v3_cluster_stats.csv')

type_map = dict(zip(range(K), centers_df['type']))
out = df[['ID', 'split'] + INDICES + ['cluster', 'emb_cluster']].copy()
out['cluster_type']     = out['cluster'].map(type_map)
out['emb_cluster_type'] = out['emb_cluster'].map(type_map)
out.to_csv(OUT_DIR / 'urban_env_clusters_v3.csv', index=False)
print(f'Saved: urban_env_clusters_v3.csv ({len(out)} rows)')

# %% [10] Physical k=4 clustering distribution — scatter + boxplot (naming aid)
# Visualise the index clustering itself (NDVI, NL, log_POP) so each cluster's
# position in the data can guide a more descriptive group name.
AX_LAB = {'NDVI_2020': 'NDVI (2020)', 'NL_2020': 'Night Lights (2020)',
          'log_POP': 'log(POP density)'}
PAIRS  = [('NL_2020', 'log_POP'), ('NL_2020', 'NDVI_2020'), ('log_POP', 'NDVI_2020')]

# naming-aid table: cluster mean + its percentile in the global distribution
print('\n=== Naming aid — cluster mean (global percentile) ===')
hdr = f'  {"Cluster":<10}' + ''.join(f'{AX_LAB[v]:>22}' for v in INDICES)
print(hdr)
for cid in range(K):
    sub = df[df['cluster'] == cid]
    cells = []
    for v in INDICES:
        m   = sub[v].mean()
        pct = (df[v] < m).mean() * 100
        cells.append(f'{m:8.2f} ({pct:2.0f}pct)')
    print(f'  C{cid+1:<9}' + ''.join(f'{c:>22}' for c in cells)
          + f'   {centers_df.loc[cid, "type"]}')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
# top row — pairwise scatter, coloured by cluster, ★ = centroid
for ax, (xc, yc) in zip(axes[0], PAIRS):
    for cid in range(K):
        m = df['cluster'] == cid
        ax.scatter(df.loc[m, xc], df.loc[m, yc], s=16, alpha=0.55, color=PALETTE[cid])
    for cid in range(K):
        ax.scatter(centers_df.loc[cid, xc], centers_df.loc[cid, yc], s=230, marker='*',
                   color=PALETTE[cid], edgecolors='black', linewidth=0.8, zorder=5)
    ax.set_xlabel(AX_LAB[xc], fontsize=11); ax.set_ylabel(AX_LAB[yc], fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
# bottom row — per-cluster boxplot of each variable
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
    ax.set_ylabel(AX_LAB[var], fontsize=11); ax.set_title(VAR_LABELS[var], fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[i],
                  markersize=9, label=centers_df.loc[i, 'type']) for i in range(K)]
fig.legend(handles=handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02),
           fontsize=9, framealpha=0.85)
plt.suptitle('Physical k=4 clustering (NDVI, NL, log_POP)\n'
             'pairwise scatter (top, ★ = centroid)  +  per-cluster boxplots (bottom)',
             fontsize=13, fontweight='bold', y=1.08)
plt.tight_layout()
fig.savefig(OUT_DIR / 'task11_v3_index_scatter_box.png', dpi=150, bbox_inches='tight')
print('Saved: task11_v3_index_scatter_box.png')
plt.show()
