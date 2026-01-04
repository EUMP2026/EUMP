import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.stats import linregress, gaussian_kde
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity as ssim
from matplotlib.cm import ScalarMappable

class RemapNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map NumPy 2.x internal path back to 1.x
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

#%%
# -------- config --------
VMIN = 295
VMAX = 315
LST_MEAN = 15285.875487117819 * 0.02
LST_STD = 382.0982046523313 * 0.02
plt.rcParams.update({"font.family": "Arial", "font.size": 7})
SC_CMAP = LinearSegmentedColormap.from_list("custom", ['#5A6CCA', "#9FC2DE"], 256)
SC_CMAP = 'viridis'
SC_CMAP = 'RdBu_r'

# Define custom red–white–blue colormap
colors = [
    (0.00, 0.19, 0.57),  # deep blue (cooling)
    (0.45, 0.68, 0.85),  # light blue
    (0.95, 0.95, 0.95),  # white (neutral)
    (0.92, 0.52, 0.36),  # light red
    (0.70, 0.00, 0.00)   # deep red (warming)
]

SC_CMAP = LinearSegmentedColormap.from_list("temp_change", colors, N=256)

# -------- utilities --------
def rmse(a, b): 
    return np.sqrt(np.mean((a - b) ** 2))


# -------- load --------
npz = np.load("images_test.npz")
gt_all = npz["gt"] * LST_STD + LST_MEAN
hand_all = npz["mlp"] * LST_STD + LST_MEAN
embd_all = npz["deep"] * LST_STD + LST_MEAN 


with open("scatter_plot_data_v4.pkl", "rb") as f:
    # scatter = pickle.load(f)
    scatter = RemapNumpyUnpickler(f).load()

# city id list and selected
# city_id_list = [20, 1059, 221, 1009, 5, 168, 423, 51, 94, 785, 495, 216, 782, 968, 652, 339, 548, 475, 5, 142, 261]
test_city_id_list = [72, 568, 211, 59, 183, 446, 79, 495, 216]
selected_cities = [446, 59, 79]
selected_citie_names = ['Tirana (Albania)','Spokane (USA)','Tegucigalpa (Honduras)']

def city_idx(cid):
    return test_city_id_list.index(cid)

sel_idx = [city_idx(c) for c in selected_cities]

all_vals = []
for i in sel_idx:
    vals = np.concatenate([gt_all[i].ravel(), hand_all[i].ravel(), embd_all[i].ravel()])
    vals = vals[~np.isnan(vals)]
    all_vals.append(vals)

# 合并成一个一维数组
all_vals = np.concatenate(all_vals)

# 绘制直方图
# plt.figure(figsize=(8, 5))
# plt.hist(all_vals, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
# plt.title('Histogram of All Values (gt + hand + embd)')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
# exit(0)

# -------- figure layout --------
# 修改 GridSpec 为 6 行 5 列，最后一列用于 color bar
fig = plt.figure(figsize=(15 / 2.54, 20 / 2.54), dpi=600)
gs = GridSpec(5, 4, figure=fig, 
              height_ratios=[1, 1, 0.05, 1, 1],
              width_ratios=[1, 1, 1, 0.05],  # 前4列为图像/散点，最后1列窄条用于 colorbar
              hspace=0.15, wspace=0.04)

# keys/titles for scatter (keep same order)
keys = ['lst_embd', 'precip_embd', 'congestion_embd', 'lst_hand', 'precip_hand', 'congestion_hand']
# titles = ['a: LST EUMP', 'b: UHI EUMP', 'c: Precip EUMP', 'd: Congestion EUMP', 'e: LST hand', 'f: UHI hand', 'g: Precip hand', 'h: Congestion hand']
titles = ['LST_EUMP', 'Precip_EUMP', 'TCI_EUMP', 'LST_hand', 'Precip_hand', 'TCI_hand']

# -------- top: 2x4 scatter plots --------
for i, (k, t) in enumerate(zip(keys, titles)):
    ax = fig.add_subplot(gs[i // 3, i % 3]) 
    entry = scatter[k]
    y = np.asarray(entry['y'])
    yhat = np.asarray(entry['y_hat'])
    n = min(len(y), len(yhat))
    y, yhat = y[:n], yhat[:n]
    R2 = r2_score(y, yhat)
    RMSE = rmse(y, yhat)
    dmin, dmax = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
    span = (dmax - dmin) if dmax > dmin else 1.0
    lo, hi = dmin - 0.1 * span, dmax + 0.1 * span

    # fit & density
    slope, intercept, r, p, se = linregress(y, yhat)
    x_dense = np.linspace(lo, hi, 200)
    y_fit = slope * x_dense + intercept

    z = gaussian_kde(np.vstack([y, yhat]))(np.vstack([y, yhat]))

    order = z.argsort()
    ax.scatter(y[order], yhat[order], c=z[order], s=4, cmap=SC_CMAP, edgecolor='w', linewidth=0.01, alpha=0.5)
    ax.plot(x_dense, y_fit, '-', color='r', lw=1)
    ax.plot(x_dense, x_dense, '--', color='k', lw=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', 'box')
    ax.text(0.04, 0.98, t, transform=ax.transAxes, fontsize=6, weight='bold', va='top')
    ax.text(0.04, 0.82, f"R²={R2:.3f}", transform=ax.transAxes, fontsize=5)
    ax.text(0.04, 0.74, f"RMSE={RMSE:.3f}", transform=ax.transAxes, fontsize=5)
    ax.tick_params(labelsize=5)
    if i < 3: 
        ax.set_xticklabels([]) 
    else: 
        ax.set_xlabel('Observed', fontsize=6)
    if i % 3 != 0: 
        ax.set_yticklabels([])
    else: 
        ax.set_ylabel('Predicted', fontsize=6)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

# -------- bottom: 3x4 images (rows 3,4,5 in GridSpec) --------
row_names = ['Observed', 'Predicted']
for r in range(2):
    for c, idx in enumerate(sel_idx):
        ax = fig.add_subplot(gs[3 + r, c])  # 图像只占前4列（c=0~3）
        if r == 0:
            img = gt_all[idx]
        else:
            img = embd_all[idx]
        
        im = ax.imshow(img, vmin=VMIN, vmax=VMAX, cmap=SC_CMAP, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        # column title on bottom row
        if r == 1:
            ax.set_xlabel(f"{selected_citie_names[c]}", fontsize=7, labelpad=8)

        # leftmost column: label rows
        if c == 0:
            ax.text(-0.15, 0.5, row_names[r], transform=ax.transAxes, rotation=90,
                    va='center', ha='center', fontsize=6)

        # compute and annotate SSIM only for the two prediction rows
        if r in (1, 2):
            gt_img = gt_all[idx].astype(float)
            pred_img = img.astype(float)

            # ssim for 2D arrays; data_range from actual vmax-vmin
            # dr = float(np.nanmax(gt_img) - np.nanmin(gt_img)) or 1.0
            # s = ssim(gt_img, pred_img, data_range=dr)
            # ax.text(0.02, 0.96, f"SSIM={s:.3f}", transform=ax.transAxes, fontsize=6,
            #         color='white', ha='left', va='top', bbox=dict(facecolor='black', alpha=0.45, pad=1))

# -------- add shared color bar for image panels --------
# 创建一个与图像使用相同归一化和 colormap 的 ScalarMappable（不实际绘图）
norm = plt.Normalize(vmin=VMIN, vmax=VMAX)
sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(SC_CMAP))
sm.set_array([])

# 在 GridSpec 最后一列（第5列，索引-1），从第3行到底部添加 colorbar
cax = fig.add_subplot(gs[-1:, -1])
cbar = fig.colorbar(sm, cax=cax, ticks=np.linspace(VMIN, VMAX, 5))
cbar.set_ticks(np.arange(VMIN,VMAX+5,5), labels=np.arange(VMIN,VMAX+5,5))
cbar.ax.tick_params(labelsize=5)

# Add custom text on the top of colorbar
cax.text(
    2, 1.05,               # (x, y) in normalized axis coordinates
    'LST (K)',               # text string
    transform=cax.transAxes, # coordinate system relative to cax
    ha='center', va='bottom',
    fontsize=7,
    fontweight='bold'         # <-- make it bold
)

# -------- add panel labels --------
fig.text(0.08, 0.88, 'a', fontsize=9, fontweight='bold', ha='left', va='top')  # for scatter panels (upper block)
fig.text(0.08, 0.47, 'b', fontsize=9, fontweight='bold', ha='left', va='top')  # for image panels (lower block)

# Save
out = "fig2.png"
fig.savefig(out, dpi=600, bbox_inches='tight')
print("Saved:", out)