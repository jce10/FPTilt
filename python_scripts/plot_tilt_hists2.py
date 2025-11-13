#!/usr/bin/env python3

import numpy as np
import re
import argparse
import matplotlib.pyplot as plt

# to plot histograms with matplotlib (step style)
def plotHists(inFiles, bins):

    nFiles = len(inFiles)

    # load all files and get H values from filenames
    filesArr = []
    Hs = []
    for f in inFiles:
        data = np.loadtxt(f)
        filesArr.append(data)
        H_val, _ = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", f.name)
        Hs.append(float(H_val))

    # default hist range (-300, 300)
    xlows = [-300] * nFiles
    xhighs = [300] * nFiles

    # create a figure and axes
    fig, axes = plt.subplots(nFiles, 1, figsize=(6, 3*nFiles), constrained_layout=True)
    if nFiles == 1:
        axes = [axes]  # make it iterable

    histos = []

    for i, ax in enumerate(axes):
        data = filesArr[i]
        low = xlows[i]
        high = xhighs[i]

        # automatic bins if bins < 0
        nBins = bins if bins > 0 else int(high - low)
        hist_vals, bin_edges = np.histogram(data, bins=nBins, range=(low, high))
        
        # plot as step histogram
        ax.step(bin_edges[:-1], hist_vals, where='mid', color='blue')
        ax.set_title(f"H = {Hs[i]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Counts")

        histos.append((hist_vals, bin_edges))

    plt.show()

    return fig, histos


#####################################################
###                 MAIN FUNCTION                 ###
#####################################################
def main(files, nbinsx):
    results = plotHists(files, nbinsx)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot peak histograms in Python (step style)")
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+', help="text files of x values")
    parser.add_argument("-bin", "--nbinsx", type=int, default=-10, help="number of bins; if negative, auto bins")

    args = parser.parse_args()
    results = main(**vars(args))

