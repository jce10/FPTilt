#!/usr/bin/env python3

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from scipy.optimize import curve_fit

# --- Gaussian function ---
def gauss(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2)) + C

# --- Double Gaussian function ---
def double_gauss(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    """Sum of two Gaussians + constant background."""
    g1 = A1 * np.exp(-(x - mu1)**2 / (2*sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2*sigma2**2))
    return g1 + g2 + C


# --- Load CSV spectrum as histogram ---
def load_csv_spectrum(path, bins=800, range=(-400, 400)):
    df = pl.read_csv(path)
    xf = df["xf"].to_numpy()
    hist_vals, bin_edges = np.histogram(xf, bins=bins, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist_vals

# --- Load all files in run directory ---
def load_run_data(outputs_root, run_name):
    run_dir = Path(outputs_root) / run_name
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    files = sorted(run_dir.glob("*.csv"))
    if not files:
        raise ValueError("No CSV files found in directory.")
    return files


# --- Interactive peak picker ---
def pick_peak_interactive(x, y):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y, lw=1)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    ax.set_title("Click on the peak you want to track")
    ax.set_xlabel("xf")
    ax.set_ylabel("Counts")

    clicked = {}
    def onclick(event):
        if event.inaxes is not None:
            clicked['x'] = event.xdata
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return clicked.get('x', None)

# --- Fit Gaussian and compute chi² ---
def fit_gaussian(x, y, x_peak, window=20):
    idx = np.abs(x - x_peak).argmin()
    # idx = idx + np.argmax(y[max(0, idx-5):min(len(y), idx+6)]) - 5  # recenter on local max ±5 bins
    lo, hi = max(0, idx-window), min(len(x), idx+window)
    x_fit, y_fit = x[lo:hi], y[lo:hi]

    p0 = [y_fit.max(), x_peak, (x[1]-x[0])*5, y_fit.min()]
    try:
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=p0)
        A, mu, sigma, C = popt
        y_model = gauss(x_fit, *popt)
        chi2 = np.sum((y_fit - y_model)**2 / np.maximum(y_model,1))  # avoid div by zero
        return popt, chi2
    except:
        return [np.nan]*4, np.nan
    

# --- Fit Double Gaussian and compute chi² ---
def fit_double_gaussian(x, y, x_peak, window=20):
    idx = np.abs(x - x_peak).argmin()
    lo, hi = max(0, idx-window), min(len(x), idx+window)
    x_fit, y_fit = x[lo:hi], y[lo:hi]

    # Initial guesses
    A1 = y_fit.max()
    mu1 = x_peak
    sigma1 = (x[1]-x[0]) * 5
    A2 = A1 * 0.3        # shoulder smaller amplitude
    mu2 = x_peak + sigma1  # shoulder slightly shifted
    sigma2 = sigma1
    C = y_fit.min()

    p0 = [A1, mu1, sigma1, A2, mu2, sigma2, C]

    try:
        popt, pcov = curve_fit(double_gauss, x_fit, y_fit, p0=p0)
        y_model = double_gauss(x_fit, *popt)
        chi2 = np.sum((y_fit - y_model)**2 / np.maximum(y_model, 1))
        return popt, chi2
    except Exception as e:
        print("Double Gaussian fit failed:", e)
        return [np.nan]*7, np.nan


# --- Cross-correlation to find peak in subsequent spectra ---
def cross_correlate_peak(x_ref, y_ref, x_target, y_target):
    corr = np.correlate(y_target - np.mean(y_target), y_ref - np.mean(y_ref), mode='full')
    shift_idx = np.argmax(corr) - (len(y_ref)-1)
    dx = x_target[1]-x_target[0]
    x_peak_shifted = x_ref[np.argmax(y_ref)] + shift_idx*dx
    return x_peak_shifted


# --- Main analysis function ---
def analyze_run(outputs_root, run_name):
    files = load_run_data(outputs_root, run_name)

    # --- Pick peak on first spectrum ---
    x0, y0 = load_csv_spectrum(files[0])
    x_click = pick_peak_interactive(x0, y0)
    if x_click is None:
        print("No peak selected. Exiting.")
        return

    # --- Prepare arrays ---
    H_vals = []
    A_vals = []
    mu_vals = []
    sigma_vals = []
    C_vals = []
    chi2_vals = []
    chi2_red_vals = []
    y_fit_list = []  # store y-values of the fit window for reduced chi²

    # --- Loop through all spectra ---
    for f in files:
        x, y = load_csv_spectrum(f)

        # extract H value from filename (assumes "0.1H_..." pattern)
        H_val = float(f.stem.split("H")[0])
        H_vals.append(H_val)

        # for first file, use clicked x; for others, cross-correlate
        if f == files[0]:
            x_peak = x_click
        else:
            x_peak = cross_correlate_peak(x0, y0, x, y)

        # fit Gaussian
        popt, chi2 = fit_gaussian(x, y, x_peak)
        A, mu, sigma, C = popt

        # Store fit parameters
        A_vals.append(A)
        mu_vals.append(mu)
        sigma_vals.append(sigma)
        C_vals.append(C)
        chi2_vals.append(chi2)

        # Store y-values of the fit window for reduced chi² calculation
        idx = np.abs(x - x_peak).argmin()
        lo, hi = max(0, idx-20), min(len(x), idx+20)
        y_fit_list.append(y[lo:hi])

    # --- Compute reduced chi² ---
    for i, chi2 in enumerate(chi2_vals):
        nu = max(len(y_fit_list[i]) - 4, 1)  # degrees of freedom = N - number of fit params
        chi2_red_vals.append(chi2 / nu)

    # --- Sort by H ---
    sorted_data = sorted(zip(H_vals, A_vals, mu_vals, sigma_vals, C_vals, chi2_vals, chi2_red_vals))
    H_vals, A_vals, mu_vals, sigma_vals, C_vals, chi2_vals, chi2_red_vals = map(list, zip(*sorted_data))

    # --- Parabola fit for sigma vs H ---
    coeffs = np.polyfit(H_vals, sigma_vals, 3)
    H_fit = np.linspace(min(H_vals), max(H_vals), 200)
    sigma_fit = np.polyval(coeffs, H_fit)
    fit_text = f"σ(H) fit: {coeffs[0]:.3f}*H³ + {coeffs[1]:.3f}*H² + {coeffs[2]:.3f}*H + {coeffs[3]:.3f}"

# --- Linear fit for centroid vs H ---
    coeffs_mu = np.polyfit(H_vals, mu_vals, 1)   # returns [m, b]
    m, b = coeffs_mu

    # build a plotting x-range
    H_lin = np.linspace(min(H_vals), max(H_vals), 200)
    mu_lin = np.polyval(coeffs_mu, H_lin)
    fit_text_mu = f"μ(H) fit: {m:.6f}*H + {b:.6f}"

    # angle (signed); use abs() only if you want magnitude
    alpha_rad = np.arctan(m)
    alpha_deg = np.degrees(alpha_rad)

    # useful H quantities
    # H_at_mu0 = -b / m if m != 0 else np.nan        # H where μ(H)=0
    H_at_mu0 = b * np.cos(alpha_rad)  
    H_mu0_units = H_at_mu0 * 42.8625  # convert to mm


    # print to terminal
    print("\n--- Centroid (linear) fit ---")
    print(f"m (slope) = {m:.4f}")
    print(f"b (intercept) = {b:.4f}")
    print(f"Angle alpha = {alpha_rad:.4f} rad")
    print(f"Angle alpha = {alpha_deg:.4f} deg")
    print(f"Optimal H value fraction : {H_at_mu0:.4f}")
    print(f"Optimal H value (mm units): {H_mu0_units:.4f} mm")


    # --- Identify extrema for vertical lines ---
    H_max_A = H_vals[np.argmax(A_vals)]
    H_max_chi2 = H_vals[np.argmax(chi2_vals)]
    H_min_sigma = H_vals[np.argmin(sigma_vals)]

    # --- Multi-panel plot ---
    fig, axes = plt.subplots(3, 2, figsize=(14,10), constrained_layout=True)
    axes = axes.flatten()

    axes[0].plot(H_vals, A_vals, 'o-'); axes[0].set_title("Amplitude vs H"); axes[0].set_xlabel("H"); axes[0].set_ylabel("A")
    axes[0].axvline(H_max_A, color='blue', linestyle='--', label=f'Max A at H={H_max_A:.2f}'); axes[0].legend()

    # axes[1].plot(H_vals, mu_vals, 'o-'); axes[1].set_title("Centroid vs H"); axes[1].set_xlabel("H"); axes[1].set_ylabel("μ")
    axes[1].plot(H_vals, mu_vals, 'o', label='data')
    axes[1].plot(H_lin, mu_lin, '-', label=fit_text_mu)
    axes[1].set_title("Centroid vs H")
    axes[1].set_xlabel("H")
    axes[1].set_ylabel("μ")
    axes[1].legend()

    axes[2].plot(H_vals, sigma_vals, 'o-', label="data"); axes[2].plot(H_fit, sigma_fit, '-', label=fit_text)
    axes[2].set_title("Width vs H"); axes[2].set_xlabel("H"); axes[2].set_ylabel("σ")
    axes[2].axvline(H_min_sigma, color='green', linestyle='--', label=f'Min σ at H={H_min_sigma:.2f}'); axes[2].legend()

    axes[3].plot(H_vals, C_vals, 'o-'); axes[3].set_title("Background vs H"); axes[3].set_xlabel("H"); axes[3].set_ylabel("C")

    # axes[4].plot(H_vals, chi2_vals, 'o-'); axes[4].set_title("Chi² vs H"); axes[4].set_xlabel("H"); axes[4].set_ylabel("Chi²")
    # axes[4].axvline(H_max_chi2, color='orange', linestyle='--', label=f'Max χ² at H={H_max_chi2:.2f}'); axes[4].legend()
    axes[4].set_visible(False)  # hide unused subplot

    # axes[5].plot(H_vals, chi2_red_vals, 'o-'); axes[5].set_title("Reduced Chi² vs H"); axes[5].set_xlabel("H"); axes[5].set_ylabel("χ²_red")
    # axes[5].axvline(H_max_chi2, color='orange', linestyle='--', label=f'Max χ² at H={H_max_chi2:.2f}'); axes[5].legend()
    axes[5].set_visible(False)  # hide unused subplot

    plt.suptitle(f"Gaussian Fit Parameters vs H for run {run_name}")
    plt.show()

    return {
        "H": H_vals,
        "A": A_vals,
        "mu": mu_vals,
        "sigma": sigma_vals,
        "C": C_vals,
        "chi2": chi2_vals,
        "chi2_red": chi2_red_vals,
        "sigma_fit": (H_fit, sigma_fit),
    }



# =========================
# Script entry point
# =========================
if __name__ == "__main__":

    # 🟢 YOU control this for each new run
    # run_name = "15deg_138kG_a0"
    run_name = "15deg_138kG_total_a0"

    # corresponding parquet file
    outputs_root = "/home/jce18b/Programs/FPTilt/outputs"
    
    analyze_run(outputs_root, run_name)


