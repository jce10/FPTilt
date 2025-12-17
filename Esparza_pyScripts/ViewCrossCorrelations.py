#!/usr/bin/env python3

"""
ViewCrossCorrelations.py

preview cross-correlation peak alignments for a run of xf histograms.
you can verify that the cross-correlation is working as expected before
proceeding to fit the correlated peaks or apply shifts to the data.

J.C. Esparza
November 2025

"""

import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

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
    return files

# ==========================================================
# Interactive Peak Picker
# ==========================================================
def pick_peak_interactive(x, y):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, y, lw=1)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    ax.set_title("Click on the peak you want to align")
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
# Extract a window around a chosen peak
# ==========================================================
def extract_window(x, y, center, width):
    mask = (x > center - width) & (x < center + width)
    return x[mask], y[mask]

# 11/21/2025 - tested and working
# ==========================================================
# Cross-correlation alignment
# ==========================================================
def cross_correlate_peaks(x_ref, y_ref, x_target, y_target):
    corr = np.correlate(y_target - np.mean(y_target),
                        y_ref - np.mean(y_ref),
                        mode='full')
    shift_bins = corr.argmax() - (len(y_target) - 1)
    x_shifted = x_target - shift_bins * (x_target[1] - x_target[0])
    idx = np.argmin(np.abs(x_shifted - x_ref[np.argmax(y_ref)]))
    return x_target[idx]

# ==========================================================
# Load CSV and make histogram
# ==========================================================
def load_csv_hist(path, bins=600, range=(-300,300)):
    df = pl.read_csv(path)
    xf = df["xf"].to_numpy()
    hist, edges = np.histogram(xf, bins=bins, range=range)
    x = (edges[:-1] + edges[1:]) / 2
    return x, hist

# ==========================================================
# Main alignment & preview function
# ==========================================================
def preview_cross_correlation(outputs_root, run_name):
    files = load_run_data(outputs_root, run_name)
    n_files = len(files)

    # --- Step 1: Interactive pick on first spectrum ---
    first_file = files[0]
    print(f"\nUsing first file:\n  {first_file.name}\n")
    x_ref, y_ref = load_csv_hist(first_file)
    x_click = pick_peak_interactive(x_ref, y_ref)
    if x_click is None:
        print("No peak selected. Exiting.")
        return

    idx_ref = np.argmin(np.abs(x_ref - x_click))
    print(f"Reference peak index: {idx_ref}, x = {x_ref[idx_ref]:.3f}")

    # --- Step 2: Cross-correlate remaining spectra ---
    x_peaks = [x_ref[idx_ref]]  # reference peak
    spectra = [(x_ref, y_ref)]
    for f in files[1:]:
        x_t, y_t = load_csv_hist(f)
        peak_x = cross_correlate_peaks(x_ref, y_ref, x_t, y_t)


        x_peaks.append(peak_x)
        spectra.append((x_t, y_t))

    # --- Step 3: Plot grid with correlated peaks in chunks of 9 (3x3) ---
    n_cols = 3
    max_per_fig = n_cols**2

    for start_idx in range(0, n_files, max_per_fig):
        end_idx = min(start_idx + max_per_fig, n_files)
        chunk_spectra = spectra[start_idx:end_idx]
        chunk_peaks = x_peaks[start_idx:end_idx]
        chunk_files = files[start_idx:end_idx]

        n_chunk = len(chunk_spectra)
        n_rows = int(np.ceil(n_chunk / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), constrained_layout=True)
        axes = axes.flatten() if n_chunk > 1 else [axes]

        for i, ((x, y), peak_x) in enumerate(zip(chunk_spectra, chunk_peaks)):
            axes[i].step(x, y, where='mid', lw=1)
            idx = np.argmin(np.abs(x - peak_x))
            # idx = idx + np.argmax(y[max(0, idx-5):min(len(y), idx+6)]) - 5  # recenter on local max ±5 bins
            axes[i].plot(x[idx], y[idx], 'ro', markersize=5)
            axes[i].set_title(chunk_files[i].name, fontsize=10)
            axes[i].tick_params(axis='both', which='major', labelsize=8)
            Cursor(axes[i], useblit=True, color='red', linewidth=1)

        # Turn off unused axes
        for j in range(n_chunk, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Cross-Correlation Peak Alignment: {run_name} (files {start_idx+1}-{end_idx})")
        plt.show()

    return x_peaks, spectra

# ==========================================================
# Script entry
# ==========================================================
if __name__ == "__main__":

    # change these as needed
    outputs_root = "/home/jce18b/Programs/FPTilt/outputs"
    run_name = "15deg_138kG_total_a0"

    peaks, spectra = preview_cross_correlation(outputs_root, run_name)

