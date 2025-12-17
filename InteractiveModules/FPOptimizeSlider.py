import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load and filter the parquet once
parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/40deg_12.9kG_total_cut.parquet"
df = pl.read_parquet(parquet_path)
df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))
df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))

S = 42.8625  # mm

# Function to compute xavg for given H and alpha
def compute_xf(H, alpha_deg):
    alpha = np.radians(alpha_deg)
    tga = np.tan(alpha)
    ctga = 1 / tga

    H_mm = H * S
    xf = ((df2["X2"] * S / np.sqrt(1 + tga**2)) - df2["Xdiff"] * H_mm) / \
         ((S / np.sqrt(1 + tga**2)) - (df2["Xdiff"] / np.sqrt(1 + ctga**2)))
    return xf.to_numpy()


# Initial parameters
H_init = 0.5
alpha_init = 0.0
bins = 600

# Compute initial xf
xf_init = compute_xf(H_init, alpha_init)

# Precompute bin edges ONCE so the histogram stays fixed
hist_vals, bin_edges = np.histogram(xf_init, bins=bins)

# Figure
fig, ax = plt.subplots(figsize=(8,5))
plt.subplots_adjust(left=0.2, bottom=0.35)

# Draw initial histogram manually
# (Needed so we can update it later without clearing the axes)
line, = ax.plot(
    bin_edges[:-1],
    hist_vals,
    drawstyle='steps-post',
    color='blue',
    label=f"H={H_init:.2f}"
)

ax.set_xlabel("Xf")
ax.set_ylabel("Counts")
ax.set_title(f"Alpha={alpha_init:.1f}°, H={H_init:.2f}")
ax.legend()

# FIXED AXIS LIMITS — lock them
ax.set_xlim(bin_edges[0], bin_edges[-1])
ax.set_ylim(0, hist_vals.max() * 1.1)

# Sliders
axAlpha = plt.axes([0.2, 0.2, 0.65, 0.03])
sAlpha = Slider(axAlpha, "Alpha", -20.0, 20.0, valinit=alpha_init)

axH = plt.axes([0.2, 0.15, 0.65, 0.03])
sH = Slider(axH, "H", -10.0, 10.0, valinit=H_init)

# Update function
def update(val):
    H = sH.val
    alpha = sAlpha.val

    # Recompute xf
    xf = compute_xf(H, alpha)

    # Recompute histogram counts USING SAME BIN EDGES
    hist_vals, _ = np.histogram(xf, bins=bin_edges)

    # Update the line data (NOT the axes)
    line.set_ydata(hist_vals)

    # Update title
    ax.set_title(f"Alpha={alpha:.1f}°, H={H:.2f}")

    # Keep y-limits fixed OR adjust dynamically (your choice)
    # ax.set_ylim(0, max(hist_vals) * 1.1)

    fig.canvas.draw_idle()

sAlpha.on_changed(update)
sH.on_changed(update)

plt.show()
