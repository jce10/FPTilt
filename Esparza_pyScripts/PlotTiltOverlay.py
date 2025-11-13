#!/usr/bin/env python3

import numpy as np
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

def plotHists(inFiles, bins):

    nFiles = len(inFiles)

    # load all files and get H values from filenames
    filesArr = []
    Hs = []
    for f in inFiles:
        # skip the first line header in CSV
        data = np.loadtxt(f, delimiter=',', skiprows=1)
        filesArr.append(data)
        H_val, _ = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", f.name)
        Hs.append(float(H_val))

    # default hist range (-300, 300)
    xlow, xhigh = -300, 300

    # automatic bins if bins < 0
    nBins = bins if bins > 0 else int(xhigh - xlow)

    # make a color map to differentiate each H
    colors = cm.viridis(np.linspace(0, 1, nFiles))

    fig, ax = plt.subplots(figsize=(8, 5))
    histos = []

    for i, data in enumerate(filesArr):
        hist_vals, bin_edges = np.histogram(data, bins=nBins, range=(xlow, xhigh))
        ax.step(bin_edges[:-1], hist_vals, where='mid', color=colors[i], label=f"H={Hs[i]}")
        histos.append((hist_vals, bin_edges))

    ax.set_xlabel("Xavg (mm)")
    ax.set_ylabel("Counts")
    ax.set_title("Xavg for Different H Values")
    ax.legend()
    plt.show()

    return fig, histos


def main(file_dir, nbinsx):
    # convert directory or list of files to a list of Path objects
    dir_path = Path(file_dir)
    files = sorted([f for f in dir_path.glob("*.csv")])
    return plotHists(files, nbinsx)


if __name__ == '__main__':
    output_dir = "/home/jce18b/Programs/FPTilt/outputs"
    nbinsx = 6000

    main(output_dir, nbinsx)


