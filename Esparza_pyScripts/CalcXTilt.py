# #!/usr/bin/env python3
"""CalcXTilt.py

script to calculate the x-tilt for a run of xf histograms.

"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import csv
import os


# ==========================================================
#  Kinematic focal plane calculation
# ==========================================================

def kinematicFP(h, alpha, S, parquetfile):

    # Load and filter input parquet
    df = pl.read_parquet(parquetfile)
    df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))

    # Add Xdiff column
    df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))

    # Compute tangent and cotangent of alpha
    tga = np.tan(alpha)
    ctga = 1 / tga

    # Compute xf using Polars
    df_result = df2.with_columns(
        (((pl.col("X2") * S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") * h)) /
         ((S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") / np.sqrt(1 + ctga**2)))).alias("xf")
    )

    # Round xf for numerical cleanliness
    df_result = df_result.with_columns(pl.col("xf").round(12))

    return df_result


# ==========================================================
#  MAIN FUNCTION
# ==========================================================
def calc_FP(parquet_path, hLow, hHigh, sp, alpha_deg, outputs_root, run_name):
    """
    Generates xf CSV files for a range of H values.
    Outputs are written to:  outputs_root/run_name/
    """

    # Verify input file exists
    inFile = Path(parquet_path)
    if not inFile.is_file():
        sys.exit(f"File does not exist: {parquet_path}")

    # Prepare output directory
    run_dir = Path(outputs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Output directory created (or already exists): {run_dir}\n")

    # Distance between wires (mm)
    S = 42.8625

    # Convert alpha to radians
    alpha = np.radians(alpha_deg)

    # Generate array of H values (fractional)
    Hs = np.arange(hLow, hHigh + sp, sp)
    relHs = Hs * S  # Convert to actual distances

    print(f"Generating {len(Hs)} files...")
    print(f"H range: {hLow} → {hHigh} (step = {sp})\n")

    # Main loop
    for i, H in enumerate(relHs):
        df_result = kinematicFP(H, alpha, S, parquet_path)

        out_filename = f"{Hs[i]:.4f}H_{alpha_deg:.1f}deg_xavg.csv"
        output_path = run_dir / out_filename

        df_result.select("xf").write_csv(output_path)

        print(f"  ✔ Written: {output_path.name}")

    print("\n🎉 All xf files generated!\n")


# ==========================================================
#  COMMAND-LINE ENTRY
# ==========================================================
if __name__ == '__main__':

    
    # 🟢 YOU control this for each new run
    run_name = "15deg_138kG_a0"


    # corresponding parquet file
    # parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/40deg_12.9kG_total_cut.parquet"
    parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
    outputs_root = "/home/jce18b/Programs/FPTilt/outputs"



    # H-scan parameters
    hLow  = 0.0
    hHigh = 5.0
    sp    = 0.1
    alpha = 0.0  # degrees

    calc_FP(parquet_path, hLow, hHigh, sp, alpha, outputs_root, run_name)
