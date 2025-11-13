#!/usr/bin/env python3

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import csv
import os

# to calculate the incident position of a particle
# on the focal plane according to the xf equation (eq. 5) 
# in https://doi.org/10.1016/0029-554X(75)90121-4

def kinematicFP(h, alpha, S, parquetfile):

    # Load and filter the parquet file
    df = pl.read_parquet(parquetfile)
    df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))
    # print(df_filtered.columns)

    # add new column to dataframe that is X2-X1 difference
    df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))

    # Compute tangent and cotangent of alpha
    # MAKE SURE alpha IS IN RADIANS
    tga = np.tan(alpha)
    ctga = 1 / tga

    # Compute xf as a new column in Polars
    df_result = df2.with_columns(
        (((pl.col("X2") * S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") * h)) /
         ((S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") / np.sqrt(1 + ctga**2)))).alias("xf")
    )

    # Round xf to 12 digits (Polars supports rounding)
    df_result = df_result.with_columns(pl.col("xf").round(12))

    return df_result



#####################################################
###                                               ###
###                 MAIN FUNCTION                 ###
###                                               ###
#####################################################
# to control what the whole program does
def calc_FP(parquet_path, hLow, hHigh, sp, alpha_deg, outdir="."):

    # Check if file exists
    inFile = Path(parquet_path)
    if not inFile.is_file():
        sys.exit(f"File does not exist: {parquet_path}")

    # Distance between wires (from Gordon's Wire_Dist)
    S = 42.8625  # mm

    # Convert alpha to radians
    alpha = np.radians(alpha_deg)

    # Array of H values
    Hs = np.arange(hLow, hHigh + sp, sp)
    relHs = Hs * S  # Convert fractional H to real distances

    for i, H in enumerate(relHs):
        # Compute focal plane using Polars-based function
        df_result = kinematicFP(H, alpha, S, parquet_path)

        # Save xf column to CSV
        out_filename = f"{Hs[i]:.4f}H_{alpha_deg:.1f}_degrees_xavg_tilt.csv"
        output_path = os.path.join(outdir, out_filename)
        df_result.select("xf").write_csv(output_path)



if __name__ == '__main__':

    parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
    out_dir = "/home/jce18b/Programs/FPTilt/outputs"
    hLow = 0.0
    hHigh = 2.0
    sp = 0.1
    alpha = 0.0  # degrees

    calc_FP(parquet_path, hLow, hHigh, sp, alpha, out_dir)