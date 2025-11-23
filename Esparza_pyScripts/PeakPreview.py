#!/usr/bin/env python3

"""
PeakPreview.py

testing script to preview peak alignments across a run of xf histograms
don't use this script for anything serious, it's just for quick visualization.
for proper peak alignment and fitting, use ViewCrossCorrelations.py and CrossCorrelate.py

"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.widgets import Cursor

def load_run_data(outputs_root, run_name):
    run_dir = Path(outputs_root) / run_name
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    files = sorted(run_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in directory: {run_dir}")
    return files

def load_csv_spectrum(path, bins=200, range=(-300,300)):
    df = pl.read_csv(path)
    xf = df["xf"].to_numpy()
    hist_vals, bin_edges = np.histogram(xf, bins=bins, range=range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist_vals

def correlate_peak_alignment(run_name, outputs_dir=".", bins=600, hist_range=(-300,300)):
    files = load_run_data(outputs_dir, run_name)
    n_files = len(files)

    n_cols = 3
    n_rows = int(np.ceil(n_files / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows), constrained_layout=True)
    axes = axes.flatten() if n_files > 1 else [axes]

    spectra = []
    for i, f in enumerate(files):
        x, y = load_csv_spectrum(f, bins=bins, range=hist_range)
        axes[i].step(x, y, where='mid', lw=1)
        axes[i].set_title(f"{f.name}")
        spectra.append((x, y))

    plt.suptitle(f"Run: {run_name}")

    # Select peak on the first spectrum
    print("Click on a peak in the FIRST subplot to select it...")
    coords = plt.ginput(1)
    if not coords:
        print("No point clicked. Exiting...")
        return
    peak_x, _ = coords[0]
    print(f"Reference peak at x={peak_x:.2f}")

    # Find the closest bin in first spectrum
    x0, y0 = spectra[0]
    idx0 = np.argmin(np.abs(x0 - peak_x))
    ref_peak_idx = idx0

    # Cross-correlate all other spectra with first spectrum
    correlated_peaks = [x0[ref_peak_idx]]  # first peak

    for i in range(1, n_files):
        xi, yi = spectra[i]
        # Use numpy.correlate (normalized)
        corr = np.correlate(yi - np.mean(yi), y0 - np.mean(y0), mode='full')
        shift_idx = corr.argmax() - (len(yi)-1)  # index shift
        correlated_peak_x = xi[ref_peak_idx - shift_idx] if 0 <= ref_peak_idx - shift_idx < len(xi) else xi[0]
        correlated_peaks.append(correlated_peak_x)

    # Mark correlated peaks
    for i, (x, y) in enumerate(spectra):
        axes[i].plot(correlated_peaks[i], y[np.argmin(np.abs(x - correlated_peaks[i]))], 'ro', markersize=5)
        cursor = Cursor(axes[i], useblit=True, color='red', linewidth=1)

    plt.show()

    return spectra, correlated_peaks

if __name__ == "__main__":
    
    outputs_root = "/home/jce18b/Programs/FPTilt/outputs"
    run_name = "15deg_138kG_a0"
    spectra, peaks = correlate_peak_alignment(run_name, outputs_dir=outputs_root)


