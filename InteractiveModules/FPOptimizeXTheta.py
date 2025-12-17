import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

cmap = cm.get_cmap('viridis').copy()
cmap.set_under("white")

# Load parquet once
# parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
# parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/40deg_12.9kG_total_cut.parquet"
parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_total_cut.parquet"

df = pl.read_parquet(parquet_path)
df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))
df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))

S = 42.8625  # mm

def compute_xf(H, alpha_deg):
    alpha = np.radians(alpha_deg)
    tga = np.tan(alpha)
    ctga = 1 / tga

    H_mm = H * S
    xf = ((df2["X2"] * S / np.sqrt(1 + tga**2)) - df2["Xdiff"] * H_mm) / \
         ((S / np.sqrt(1 + tga**2)) - (df2["Xdiff"] / np.sqrt(1 + ctga**2)))

    return xf.to_numpy()

def compute_theta():
    theta = np.arctan(df2["Xdiff"] / S)
    return theta.to_numpy()

# Initial parameters
H_init = 0.0
alpha_init = 0.0
bins = 600

# Compute initial xf, theta
xf_init = compute_xf(H_init, alpha_init)
theta = compute_theta()

# Precompute fixed bin edges
hist_vals, bin_edges = np.histogram(xf_init, bins=bins)

# Prepare 2-panel figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
plt.subplots_adjust(left=0.18, bottom=0.25, hspace=0.35)

fp_bins = np.linspace(-300, 300, 600)
theta_bins = np.linspace(0.35, 0.9, 600)

# --- Top panel: Xf 1D histogram ---
line1, = ax1.plot(
    bin_edges[:-1],
    hist_vals,
    drawstyle='steps-post',
    color='blue',
    label=f"H={H_init:.2f}"
)

ax1.set_ylabel("Counts")
ax1.set_xlim(bin_edges[0], bin_edges[-1])
ax1.set_ylim(0, hist_vals.max() * 1.1)
ax1.set_title("Xf Spectrum")

# --- Bottom panel: 2D θ vs Xf histogram ---
heatmap = ax2.hist2d(
    xf_init, theta,
    bins=[bin_edges, theta_bins],
    cmap=cmap,
    vmin=1
)

ax2.set_xlabel("Xf")
ax2.set_ylabel("Theta (rad)")
ax2.set_title("Theta vs Xf (2D Histogram)")

# Keep a reference to update 2D histogram
quadmesh = heatmap[3]
# quadmesh.set_under("white")

# --- Sliders ---
axAlpha = plt.axes([0.18, 0.15, 0.65, 0.03])
sAlpha = Slider(axAlpha, "Alpha", -10.0, 10.0, valinit=alpha_init)

axH = plt.axes([0.18, 0.10, 0.65, 0.03])
sH = Slider(axH, "H", -5.0, 5.0, valinit=H_init)

# Update function
def update(val):
    H = sH.val
    alpha = sAlpha.val

    # Recompute xf
    xf = compute_xf(H, alpha)

    # --- Update 1D histogram ---
    hist_vals, _ = np.histogram(xf, bins=bin_edges)
    line1.set_ydata(hist_vals)
    ax1.set_title(f"Xf Spectrum (Alpha={alpha:.4f}°, H={H:.4f})")
    ax1.set_ylim(0, hist_vals.max() * 1.1)

    # --- Update 2D histogram ---
    # Remove old heatmap
    # for coll in quadmesh.collections:
    #     coll.remove()
    global quadmesh
    quadmesh.remove()  # remove previous QuadMesh

    heatmap_new = ax2.hist2d(
        xf,
        theta,
        bins=[bin_edges, theta_bins],
        cmap=cmap,
        vmin=1
    )
    quadmesh = heatmap_new[3]  # update reference
    # quadmesh.set_under("white")


    fig.canvas.draw_idle()

sAlpha.on_changed(update)
sH.on_changed(update)

plt.show()
