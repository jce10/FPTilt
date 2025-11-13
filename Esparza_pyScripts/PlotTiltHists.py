from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

def plotHists(csv_files, bins):
    nFiles = len(csv_files)
    filesArr = []
    Hs = []

    for fpath in csv_files:
        data = np.loadtxt(fpath, delimiter=',', skiprows=1)
        filesArr.append(data)

        # extract H from filename
        H_val, _ = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", fpath.name)
        Hs.append(float(H_val))

    xlows = [-300] * nFiles
    xhighs = [300] * nFiles

    fig, axes = plt.subplots(nFiles, 1, figsize=(6, 3*nFiles), constrained_layout=True)
    if nFiles == 1:
        axes = [axes]

    histos = []

    for i, ax in enumerate(axes):
        data = filesArr[i]
        low, high = xlows[i], xhighs[i]
        nBins = bins if bins > 0 else int(high - low)

        hist_vals, bin_edges = np.histogram(data, bins=nBins, range=(low, high))
        ax.step(bin_edges[:-1], hist_vals, where='mid', color='blue')
        ax.set_title(f"H = {Hs[i]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Counts")
        histos.append((hist_vals, bin_edges))

    plt.show()
    return fig, histos


def main(outdir, nbinsx=-10):
    outdir = Path(outdir)
    if not outdir.is_dir():
        raise ValueError(f"{outdir} is not a valid directory")

    csv_files = sorted(outdir.glob("*.csv"))  # get all CSVs in directory
    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in {outdir}")

    return plotHists(csv_files, nbinsx)


if __name__ == '__main__':
    output_dir = "/home/jce18b/Programs/FPTilt/outputs"
    nbinsx = 6000

    main(output_dir, nbinsx)


