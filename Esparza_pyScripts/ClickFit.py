#!/usr/bin/env python3
"""
ClickFit.py

script to pick a peak from a run of xf histograms, fit a Gaussian to it,
and report the fit parameters.

i mostly used this for testing purposes, but it might be useful later.

J.C. Esparza
November 2025

"""


import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import Cursor


# ==========================================================
# Simple Gaussian function
# ==========================================================
def gauss(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2)) + C


# ==========================================================
# Load all CSVs from a run directory
# ==========================================================
def load_run_data(outputs_root, run_name):
    run_dir = Path(outputs_root) / run_name

    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")

    files = sorted(run_dir.glob("*.csv"))
    if not files:
        raise ValueError("No CSV files found in directory.")

    print(f"Found {len(files)} CSV files in: {run_dir}\n")

    return files


# ==========================================================
# Interactive Peak Picker
# ==========================================================
def pick_peak_interactive(x, y):
    """
    Display a plot, wait for a mouse click, return x-click position.
    """

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y, lw=1)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    ax.set_title("Click on the peak you want to fit")
    ax.set_xlabel("xf")
    ax.set_ylabel("Counts")
    

    clicked = {}

    def onclick(event):
        if event.inaxes is not None:
            clicked['x'] = event.xdata
            print(f"\nClicked at x = {event.xdata:.3f}")
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return clicked.get('x', None)


# ==========================================================
# Fit a Gaussian around chosen region
# ==========================================================
def fit_peak(x, y, x_click, window=20, show_plot=True):
    """
    Fit a Gaussian around a clicked peak.
    window = number of bins to include on each side of click point.
    """

    # Find index closest to click
    idx = (np.abs(x - x_click)).argmin()

    # region around the click
    lo = max(0, idx - window)
    hi = min(len(x), idx + window)

    x_fit = x[lo:hi]
    y_fit = y[lo:hi]

    # initial guesses
    A0 = y_fit.max()
    mu0 = x_click
    sigma0 = (x[1] - x[0]) * 5  # approx 5 bins
    C0 = y_fit.min()

    p0 = [A0, mu0, sigma0, C0]

    try:
        popt, pcov = curve_fit(gauss, x_fit, y_fit, p0=p0)
        A, mu, sigma, C = popt

        print("\n===== Peak Fit Results =====")
        print(f"μ (peak position): {mu:.4f}")
        print(f"σ (width):         {sigma:.4f}")
        print(f"A (height):        {A:.1f}")
        print(f"Background C:      {C:.1f}")
        print("===========================\n")

        # ===============================
        # PLOTTING THE FIT (easy to comment later)
        # ===============================
        if show_plot:
            fig, ax = plt.subplots(figsize=(10,6))
            cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

            # --- Plot the FULL histogram ---
            ax.plot(x, y, 'k-', lw=1, label="Full spectrum")

            # --- Highlight the fitting window ---
            ax.axvspan(x_fit[0], x_fit[-1], color='orange', alpha=0.15,
                       label="Fit window")

            # --- Plot the Gaussian fit (smooth) ---
            x_smooth = np.linspace(x_fit[0], x_fit[-1], 400)
            y_smooth = gauss(x_smooth, *popt)
            ax.plot(x_smooth, y_smooth, 'r-', lw=2, label="Gaussian fit")

            # --- Peak marker ---
            ax.axvline(mu, color='r', linestyle='--', alpha=0.7,
                       label=f"μ = {mu:.3f}")

            ax.set_title("Gaussian Fit on Full Spectrum")
            ax.set_xlabel("xf")
            ax.set_ylabel("Counts")
            ax.legend()
            plt.show()


        return popt, (x_fit, y_fit)

    except Exception as e:
        print("\n❌ Gaussian fit failed:", e)
        return None, None



# ==========================================================
# MAIN FUNCTION
# ==========================================================
def pick_peak_from_run(outputs_root, run_name):

    # 1) Load all results
    files = load_run_data(outputs_root, run_name)

    # 2) Use the FIRST FILE for peak selection
    first = files[0]
    print(f"\nUsing first file:\n  {first.name}\n")

    df = pl.read_csv(first)
    y = df["xf"].to_numpy()

    # create histogram so you get a spectrum-like shape
    nBins = 600
    hist, edges = np.histogram(y, bins=nBins)
    x = (edges[:-1] + edges[1:]) / 2

    # 3) Ask user to click a peak
    x_click = pick_peak_interactive(x, hist)

    if x_click is None:
        print("No peak selected.")
        return

    # 4) Fit the peak around the click
    popt, _ = fit_peak(x, hist, x_click)

    return popt


# ==========================================================
# SCRIPT ENTRY POINT
# ==========================================================
if __name__ == "__main__":


    # Change these as needed
    outputs_root = "/home/jce18b/Programs/FPTilt/outputs"

    # Change this to reflect your input parameters
    run_name = "15deg_138kG_a0"

    pick_peak_from_run(outputs_root, run_name)