import polars as pl
import matplotlib.pyplot as plt
import numpy as np


def kinematicFP(h, alpha, S, parquetfile):
    
    # load X1 and X2 from the parquet file
    df = pl.read_parquet(parquetfile)
    x1 = df['X1'].to_numpy()
    x2 = df['X2'].to_numpy()

    # tangent and cotangent of alpha
    tga = np.sin(alpha) / np.cos(alpha)
    ctga = np.cos(alpha) / np.sin(alpha)

    num = (x2*S/np.sqrt(1 + tga**2)) - (x2 - x1)*h
    deno = (S/np.sqrt(1 + tga**2)) - ((x2 - x1)/np.sqrt(1 + ctga**2))

    xf = num / deno

    # rounding xf to 12 digits
    rounded_xf = np.array([float('{:.12e}'.format(val)) for val in xf])

    return rounded_xf


# Load parquet files
def load_parquet(parquet_path):
    df = pl.read_parquet(parquet_path)
    df_filtered = df.filter((pl.col("X1") != -1e6) & (pl.col("X2") != -1e6))
    df_xavg = df.filter(pl.col("Xavg") != -1e6)
    return df_filtered, df_xavg