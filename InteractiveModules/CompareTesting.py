import polars as pl
import matplotlib.pyplot as plt
import numpy as np

"""
CompareXavg.py

comparing Xavg histograms from Event Builder and calculating it manually 
per Shapira et al. 1985  https://doi.org/10.1016/0029-554X(75)90121-4

"""


def compare_xavg(H, alpha_deg, S, parquetfile):
    
    # Load Event Builder parquet and compute histogram for Xavg
    df = pl.read_parquet(parquetfile)
    # filtered data frame -- removed invalid rows    
    df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6) & (pl.col("Xavg") != -1e6))

    # create an Xdiff column
    df2 = df_filtered.with_columns((pl.col("X2") - pl.col("X1")).alias("Xdiff"))
    

    # Compute tangent and cotangent of alpha
    
    alpha = np.radians(alpha_deg)
    tga = np.tan(alpha)
    ctga = 1 / tga

    print("S =", S)
    print("tga, ctga =", tga, ctga)
    print("Xdiff min,max:", df2["Xdiff"].min(), df2["Xdiff"].max())
    denom = (S / np.sqrt(1 + tga**2)) - (df2["Xdiff"] / np.sqrt(1 + ctga**2))
    print("Denominator min,max:", denom.min(), denom.max())


    # Compute xf using Polars
    df_xf = df2.with_columns(
        (((pl.col("X2") * S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") * H)) /
         ((S / np.sqrt(1 + tga**2)) - (pl.col("Xdiff") / np.sqrt(1 + ctga**2)))).alias("xf")
    )
    # Round xf for numerical cleanliness
    df_xf = df_xf.with_columns(pl.col("xf").round(12))

    # xavg_vals = df_xf["Xavg"].to_numpy()
    # xf_vals   = df_xf["xf"].to_numpy()
    # print("Xavg min,max:", xavg_vals.min(), xavg_vals.max())
    # print("xf   min,max:", xf_vals.min(), xf_vals.max())
    # print("Xavg unique sample:", np.unique(xavg_vals)[:10])
    # print("xf   unique sample:", np.unique(xf_vals)[:10])

    # Compute histograms
    hist_vals_xavg, bin_edges_xavg = np.histogram(df_xf["Xavg"].to_numpy(), bins=600)
    hist_vals_xf, bin_edges_xf     = np.histogram(df_xf["xf"].to_numpy(), bins=600)

    # Compute bin centers
    bin_centers_xavg = (bin_edges_xavg[:-1] + bin_edges_xavg[1:]) / 2
    bin_centers_xf   = (bin_edges_xf[:-1]   + bin_edges_xf[1:])   / 2

    # Plot
    plt.figure(figsize=(9,6))
    plt.step(bin_centers_xavg, hist_vals_xavg, where='mid', color='blue', label="Event Builder Xavg")
    plt.step(bin_centers_xf,   hist_vals_xf,   where='mid', color='orange', label="Kinematic xf", linestyle='--')
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.title("Xavg Histogram Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()


    return df_xf


# ==========================================================
#  COMMAND-LINE ENTRY
# ==========================================================
if __name__ == '__main__':

    # 🟢 YOU control this for each new run
    run_name = "15deg_138kG_a0"


    # corresponding parquet file
    # parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/40deg_12.9kG_total_cut.parquet"
    parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
    output_root = "/home/jce18b/Programs/FPTilt/outputs"



    # H-scan parameters
    H = 1.18     # mm
    alpha = 1.04 # degrees
    S = 42.8625  # mm

    xf = compare_xavg(H, alpha, S, parquet_path)
    print(xf["xf"])
    # print(xf["Xavg"])