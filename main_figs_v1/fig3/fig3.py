# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import gridspec
import re

# Geo / Mapping
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------ USER CONFIG ------------------
DATA = {
    'corr_matrix': './corr_matrix.npy',
    'country_shp': './country_shp/TM_WORLD_BORDERS-0.3.shp',
    'urban_shp': './urban_shp/GHSL_filtered_v1.shp',
    'pos_csv': './pos_2020.csv',
    'hand_csv': './2020_hand_features_with_id.csv',
    'embd_npy': './embds_mean.npy'
}

# Font sizes
FONT = {'title': 7, 'label': 6, 'tick': 5, 'legend': 5}

# Output file
OUT_PNG = './fig3.png'

# Colormap for Heatmap
COLOR_SCHEME = 1  # 1 or 2

if COLOR_SCHEME == 1:
    # A: Teal-Orange
    CMAP_HEAT = LinearSegmentedColormap.from_list(
        'teal_orange_discrete',
        ['#249c98', '#e0e0e0', '#f0785b'],
        N=12 
    )
elif COLOR_SCHEME == 2:
    # B: Spectral Style
    colors_spectral = [
        '#9e0142',
        '#d53e4f',
        '#f46d43',
        '#fdae61',
        '#fee08b',
        '#ffffbf',
        '#e6f598',
        '#abdda4',
        '#66c2a5',
        '#3288bd',
        '#5e4fa2'
    ]
    CMAP_HEAT = LinearSegmentedColormap.from_list(
        'spectral_custom',
        colors_spectral,
        N=12
    )
else:
    CMAP_HEAT = LinearSegmentedColormap.from_list(
        'teal_orange',
        ['#249c98', '#e0e0e0', '#f0785b']
    )

# ------------------ Utility Functions ------------------

def load_corr_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Correlation matrix not found: {path}")
    return np.load(path)

def two_slope_norm_from(array):
    vmax = np.nanmax(np.abs(array))
    vmax = float(vmax) if vmax and vmax > 0 else 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

def k_means_cities(df, features, k=5):
    """Perform k-means clustering on selected predictors."""
    df_out = df.copy()
    X = df_out[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters

def prepare_data(data_paths):
    """Load and merge all data sources."""
    # 1. Load Vector/CSV
    country_shp = gpd.read_file(data_paths['country_shp'])
    urban_shp = gpd.read_file(data_paths['urban_shp'])
    pos_df = pd.read_csv(data_paths['pos_csv'])
    hand_df = pd.read_csv(data_paths['hand_csv'])
    
    # 2. Load Embeddings
    if not os.path.exists(data_paths['embd_npy']):
        raise FileNotFoundError(f"Embeddings not found: {data_paths['embd_npy']}")
        
    embds = np.load(data_paths['embd_npy'], mmap_mode='r')
    # Mean over the last two axes
    # embds = embds.mean(axis=(2, 3))
    embd_cols = [f"emb_{i}" for i in range(embds.shape[1])]
    
    # Create Embd DataFrame
    ids = np.arange(1, embds.shape[0] + 1)
    embd_df = pd.DataFrame(embds, columns=embd_cols)
    embd_df.insert(0, "ID", ids)
    embd_df['ID'] = embd_df['ID'].astype(int)
    
    # 3. Process IDs and Merge
    urban_pts = urban_shp.copy()
    urban_pts['geometry'] = urban_pts.geometry.centroid
    urban_pts['ID'] = urban_pts['ID'].astype(int)
    urban_pts = urban_pts[['ID','geometry']]

    pos_df = pos_df[['ID', 'x', 'y','z']]
    pos_df['ID'] = pos_df['ID'].astype(int)
    
    hand_df['ID'] = hand_df['ID'].astype(int)
    
    # Merge all
    merged = urban_pts.merge(pos_df, on='ID', how='left') \
                      .merge(hand_df, on='ID', how='left') \
                      .merge(embd_df, on='ID', how='left')
                      
    return merged, embd_cols, country_shp

# ------------------ Main Plotting ------------------

def make_combined_figure():
    merged_df, embd_cols, country_shp = prepare_data(DATA)
    corr = load_corr_matrix(DATA['corr_matrix'])
    # --- Prepare Clustering Data ---
    
    # 1. Hand Features Clustering
    hand_features = ['NDVI','nl','pop','water','BH','BF','DEM','tlml','prectotcorr','speed']
    c1 = k_means_cities(merged_df, hand_features, k=5)
    
    # Mapping for Hand
    mapping1 = {1: 0, 3: 1, 0: 2, 2: 3, 4: 4}
    c1_new = np.vectorize(mapping1.get)(c1)
    merged_df['cluster_hand'] = c1_new
    
    # 2. Embedding Features Clustering
    c2 = k_means_cities(merged_df, embd_cols, k=5)
    
    # Mapping for Embd
    mapping2 = {4: 0, 3: 1, 1: 2, 2: 3, 0: 4}
    c2_new = np.vectorize(mapping2.get)(c2)
    merged_df['cluster_embd'] = c2_new

    # --- Setup Figure ---
    cm_to_inch = 1.0 / 2.54
    
    # Width 18cm, Height 12.5cm creates a compact, balanced layout.
    figsize = (18.0 * cm_to_inch, 12 * cm_to_inch)
    
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5], width_ratios=[1, 1], figure=fig, hspace=0.0)

    # ==========================
    # Panel A: Heatmap (Top)
    # ==========================
    ax_heat = fig.add_subplot(gs[0, :])
    
    # Process correlation matrix sorting
    k = 0
    sort_indices = np.argsort(corr[k, :])
    corr = corr[:, sort_indices]
    
    n_rows, n_cols = corr.shape
    feature_names_raw = ["NDVI","NL","POP","Water","BH","BF","DEM","Ta","Precip","Wind"]
    feature_abs_mean = np.mean(np.abs(corr), axis=1)
    feature_names = [f"{name} ({val:.2f})" for name, val in zip(feature_names_raw, feature_abs_mean)]
    
    norm = two_slope_norm_from(corr)
    im = ax_heat.imshow(corr, aspect='auto', cmap=CMAP_HEAT, norm=norm, interpolation='nearest', origin='upper')
    
    ax_heat.set_ylabel('Handcraft predictors', fontsize=FONT['label'])
    ax_heat.set_xlabel('Embeddings', fontsize=FONT['label'])
    ax_heat.set_yticks(np.arange(n_rows))
    ax_heat.set_yticklabels(feature_names[:n_rows], fontsize=FONT['tick'])
    
    n_x_ticks = 4
    x_positions = np.linspace(0, corr.shape[1]-1, n_x_ticks, dtype=int)
    ax_heat.set_xticks(x_positions)
    ax_heat.set_xticklabels(x_positions, fontsize=FONT['tick'])
    
    for r in np.arange(0.5, corr.shape[0]-0.5, 1.0):
        ax_heat.axhline(r, color='#dddddd', linewidth=0.8)
    
    # Panel Label A
    ax_heat.text(0.00, 1.02, 'a', transform=ax_heat.transAxes, fontsize=FONT['title'], fontweight='bold', va='bottom', ha='left')
    
    cax = inset_axes(ax_heat, width='1.5%', height='100%', bbox_transform=ax_heat.transAxes, loc='right', bbox_to_anchor=(0.03, 0., 1, 1), borderpad=0)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FONT['tick'], length=2, width=0.6)
    cb.set_label('Correlation', fontsize=FONT['label'])

    # ==========================
    # Panel B & C: Cluster Maps (Bottom)
    # ==========================
    
    def plot_cluster_map(ax, cluster_col, title, panel_label):
        ax.set_global()
        ax.set_extent([-180, 180, -70, 70], crs=ccrs.PlateCarree())
        # ax.set_aspect('auto') 
        ax.coastlines(linewidth=0.3)
        ax.add_feature(cfeature.BORDERS, linewidth=0.1)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        
        lons = merged_df.geometry.x.values
        lats = merged_df.geometry.y.values
        clusters = merged_df[cluster_col].values
        
        sc = ax.scatter(
            lons, lats,
            c=clusters,
            s=4,
            cmap="tab10",
            transform=ccrs.PlateCarree(),
            edgecolor="black",
            linewidth=0.1,
            alpha=0.9,
            # vmin=0, vmax=9 
        )
        
        handles, raw_labels = sc.legend_elements()
        clean_labels = []
        for lbl in raw_labels:
            num = re.findall(r'\d+', lbl)
            clean_labels.append(f"{int(num[0])}" if num else lbl)
            
        ax.legend(
            handles[:5], clean_labels[:5],
            title="Clusters",
            loc="lower left",
            fontsize=FONT['legend'],
            title_fontsize=FONT['legend'],
            frameon=False,
            bbox_to_anchor=(0.1, 0.02),
            borderaxespad=0
        )
        
        ax.set_title(title, fontsize=FONT['title'])
        # Panel Label B/C
        ax.text(0.0, 1.05, panel_label, transform=ax.transAxes, fontsize=FONT['title'], fontweight='bold', va='bottom', ha='left')
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # --- Map 1: Hand Features ---
    ax_map1 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=0))
    plot_cluster_map(ax_map1, 'cluster_hand', 'Handcrafted predictors', 'b')

    # --- Map 2: Embeddings ---
    ax_map2 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=0))
    plot_cluster_map(ax_map2, 'cluster_embd', 'Embeddings', 'c')

    # Tight layout control
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, hspace=0.18, wspace=0.05)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
    print('Saved Combined Figure:', OUT_PNG)

if __name__ == '__main__':
    make_combined_figure()
