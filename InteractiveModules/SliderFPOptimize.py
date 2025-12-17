import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# Load and filter the parquet once
parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
df = pl.read_parquet(parquet_path)
df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))
df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))

S = 42.8625  # mm

def compute_xf(H, alpha_deg):
    alpha = np.radians(alpha_deg)
    tga = np.tan(alpha)
    ctga = 1 / tga
    
    H_mm = H * S
    xf = ((df2['X2'] * S / np.sqrt(1 + tga**2)) - df2['Xdiff'] * H_mm) / \
         ((S / np.sqrt(1 + tga**2)) - (df2['Xdiff'] / np.sqrt(1 + ctga**2)))
    return xf.to_numpy()

# --- Initial parameters ---
H_init = 0.5       # fraction
alpha_init = 0.0   # degrees
bins = 600

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(left=0.2, bottom=0.35)

# Initial plot
xf = compute_xf(H_init, alpha_init)
hist_vals, bin_edges, _ = ax.hist(xf, bins=bins, histtype='step', color='blue', label=f"H={H_init:.2f}")
hist_obj = ax.patches
ax.set_xlabel("Xf")
ax.set_ylabel("Counts")
ax.set_title(f"Alpha={alpha_init:.1f}°, H={H_init:.2f}")
ax.legend()

# --- Sliders ---
axAlpha = plt.axes([0.2, 0.2, 0.65, 0.03])
sAlpha = Slider(axAlpha, 'Alpha', 0.0, 20.0, valinit=alpha_init)

axH = plt.axes([0.2, 0.15, 0.65, 0.03])
sH = Slider(axH, 'H', -5.0, 5.0, valinit=H_init)

# --- Update function ---
def update(val):
    H = sH.val
    alpha = sAlpha.val
    
    ax.cla()
    xf = compute_xf(H, alpha)
    ax.hist(xf, bins=bins, histtype='step', color='blue', label=f"H={H:.2f}")
    ax.set_xlabel("Xf")
    ax.set_ylabel("Counts")
    ax.set_title(f"Alpha={alpha:.1f}°, H={H:.2f}")
    ax.legend()
    fig.canvas.draw_idle()

sAlpha.on_changed(update)
sH.on_changed(update)

plt.show()