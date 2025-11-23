
import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.widgets import Cursor


def load_csv_spectrum(path):
    """Load single-column CSV and return indices as x, values as y."""
    # data = np.loadtxt(path, delimiter=",", skiprows=1)
    # x = np.arange(len(data))  # index of each data point
    # y = data                  # actual xf values


    df = pl.read_csv(path)
    y = df["xf"].to_numpy()

    # create histogram so you get a spectrum-like shape
    nBins = 600
    hist, edges = np.histogram(y, bins=nBins)
    x = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, hist, lw=1)
    cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

    ax.set_title("Click on the peak you want to fit")
    ax.set_xlabel("xf")
    ax.set_ylabel("Counts")

    plt.show()




    return x, y


if __name__ == "__main__":
    outputs = "/home/jce18b/Programs/FPTilt/outputs"
    run_name = "15deg_138kG_a0"  
    x, y = load_csv_spectrum(path=outputs+"/"+run_name+"/0.0000H_0.0_degrees_xavg_tilt.csv")
    print("Loaded spectrum from:", run_name)
    print(y)