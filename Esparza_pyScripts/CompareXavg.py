import polars as pl
import matplotlib.pyplot as plt
import numpy as np

"""
CompareXavg.py

comparing Xavg histograms from Event Builder and calculating it manually 
per Shapira et al. 1985  https://doi.org/10.1016/0029-554X(75)90121-4

"""


def compare_xavg(H, alpha_deg, S, parquetfile, output_path=None):
    
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

    # print("S =", S)
    # print("tga, ctga =", tga, ctga)
    # print("Xdiff min,max:", df2["Xdiff"].min(), df2["Xdiff"].max()) # sanity check

    denom = (S / np.sqrt(1 + tga**2)) - (df2["Xdiff"] / np.sqrt(1 + ctga**2))

    # print("Denominator min,max:", denom.min(), denom.max())


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
    # print("xf   unique sample:", np.unique(xf_vals)[:10]) # sanity check

    # Compute histograms
    hist_vals_xavg, bin_edges_xavg = np.histogram(df_xf["Xavg"].to_numpy(), bins=600)
    hist_vals_xf, bin_edges_xf     = np.histogram(df_xf["xf"].to_numpy(), bins=600)

    hist_vals_xavg = hist_vals_xavg / hist_vals_xavg.max()
    hist_vals_xf   = hist_vals_xf / hist_vals_xf.max()


    # Compute bin centers
    bin_centers_xavg = (bin_edges_xavg[:-1] + bin_edges_xavg[1:]) / 2
    bin_centers_xf   = (bin_edges_xf[:-1]   + bin_edges_xf[1:])   / 2

    # Plot
    plt.figure(figsize=(9,6))
    plt.step(bin_centers_xavg, hist_vals_xavg, where='mid', color='blue', label="Event Builder Xavg")
    plt.step(bin_centers_xf,   hist_vals_xf,   where='mid', color='orange', label="Kinematic Xf")
    plt.xlabel("Position (mm)")
    plt.ylabel("Counts")
    plt.title("Xavg Histogram Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Export DataFrame if output path is provided
    if output_path is not None:
        df_xf.write_parquet(output_path)
        print(f"Exported DataFrame to {output_path}")


    return df_xf


# ==========================================================
#  COMMAND-LINE ENTRY
# ==========================================================
if __name__ == '__main__':

    # 🟢 YOU control this for each new run
    # run_name = "15deg_138kG_a0"
    run_name = " 13C_40deg_test1"


    # corresponding parquet file
    # parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/40deg_12.9kG_total_cut.parquet"
    # parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_300s_cut.parquet"
    parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_total_cut.parquet"
    output_root = "/home/jce18b/Programs/FPTilt/outputs"



    # H-scan parameters @ 15 deg
    H = 1.2508    # mm
    alpha = 0.1767 # degrees
    S = 42.8625  # mm
    H_mm = H * S


    # # H-scan parameters @ 40 deg
    # H = 1.9887    # mm
    # alpha = 0.2593 # degrees
    # S = 42.8625  # mm
    # H_mm = H * S

    # xf = compare_xavg(H_mm, alpha, S, parquet_path, output_path=output_root + f"/{run_name}_H{H:.2f}_a{alpha:.2f}_xavg.parquet")
    # xf = compare_xavg(H_mm, alpha, S, parquet_path, output_path=output_root + f"/{run_name}_H{H:.2f}_a{alpha:.2f}_xavg.parquet")
    xf = compare_xavg(H_mm, alpha, S, parquet_path, output_path=None)
    
    # print(xf["xf"])
    # print(xf["Xavg"])
