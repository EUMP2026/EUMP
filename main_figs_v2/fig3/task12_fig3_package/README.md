# Fig. 3 package — embedding-vs-physical-variable analyses

Self-contained code + data bundle for the four panels feeding main-text Fig. 3
(1076 GHSL urban centres, 1536-dim annual urban embeddings, 2003–2020).

## Folder layout

```
task12_fig3_package/
├── code/
│   ├── fig3_v2.py                       # full Fig. 3 composite
│   ├── fig3_v2_no_a.py                  # Fig. 3 composite without panel a
│   ├── fig3_v2_no_a_v1.py               # default Fig. 3 composite without panel a
│   ├── task06_corr_embedding_v2.py     # max-|R| summary plot (16 variables)
│   ├── task06_corr_matrix.py           # 16 × 1536 Pearson-r heatmap
│   ├── task11_corr_top_dims_env_v3.py  # env clustering: index vs embedding (k=4)
│   └── task11_best_clusters_form_v4.py # form clustering: index vs embedding (k=4)
├── data/
│   ├── zonal_merged_2020.csv           # per-city physical variables (wide, per year)
│   ├── urban_form_clusters.csv         # per-city BF / BH_mean / BH_QCD
│   ├── task06_all/                     # precomputed correlation tables
│   │   ├── corr_dynamic.csv            #   max |r| per variable × year
│   │   ├── corr_dynamic_bestdim.csv    #   best dim per variable × year
│   │   └── corr_static.csv             #   abs_r / r_signed / best_dim (static vars)
│   ├── embd/
│   │   └── eump_..._260503_v4.npz      # embeddings (19368 × 1536; keys:
│   │                                   #   embeddings, scene_ids, years)
│   └── shp/
│       ├── GHSL_filtered_v1.*          # city centroids (ID, GCPNT_LON/LAT)
│       └── TM_WORLD_BORDERS-0.3.*      # country borders (basemap)
└── output/                             # created/filled by the scripts
```

## How to run

All paths inside the scripts are relative to the package root (resolved from
the script location via `__file__`). From the package root, the default Fig. 3
command is:

```
python code/fig3_v2_no_a_v1.py
```

Additional figure/task scripts can be run as needed:

```
python code/fig3_v2.py
python code/fig3_v2_no_a.py
python code/task06_corr_embedding_v2.py
python code/task06_corr_matrix.py
python code/task11_corr_top_dims_env_v3.py
python code/task11_best_clusters_form_v4.py
```

`fig3_v2_no_a_v1.py` writes `output/fig3_v2_no_a_v1.png` and is the default
version used for the main Fig. 3 package.
`fig3_v2.py` writes `output/fig3_v2.png`, the full version with panel a.
`fig3_v2_no_a.py` writes `output/fig3_v2_no_a.png`, the version without panel a.
The embedding-cluster map titles intentionally report only the selected
embedding dimension count/year, without ARI/NMI/Acc metrics in the map title.
The two Fig. 3 scripts are fully independent: neither imports nor calls the
other, and each can be run on its own.
The summary panel highlights repeated dimensions: D238/D676/D967 share one
continuous NL/ISA background, while D1236 marks the shared size/total dimension.

The task scripts are also organised as `# %%` cells for the VS Code Interactive
Window; in that case start the interactive session with the package root as the
working directory. The task scripts are independent of each other — any subset
can be run in any order. Results land in `output/<task name>/`.

## Dependencies

Python ≥ 3.10 with: numpy, pandas, matplotlib, scikit-learn, scipy,
geopandas, cartopy (the last two only for the two task11 clustering scripts).

## Notes

- Embedding–variable matching is a strict inner join on city ID for the
  chosen year (dynamic variables: latest available year; static variables:
  prescribed year — BF/BH 2012, DEM/AREA 2020).
- `data/task06_all/*.csv` were precomputed by `task06_corr_embedding.py`
  (not included); `task06_corr_embedding_v2.py` only re-plots them.
