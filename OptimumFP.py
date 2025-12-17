import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import time


parquet_path = "/home/jce18b/Esparza_SPS/2025_06_13C_campaign/built/15deg_13.85kG_total_cut.parquet"
df = pl.read_parquet(parquet_path)
# print(df.get_columns)

D = 42.8625  # distance between front and rear delay lines in mm


# create a 2D histogram with columns ScintLeftEnergy and AnodeBackEnergy with bins 512x512 and range 0-4096 with lognormed colorbar
# create aa filter statement that says if x1 and x2 are not the default flag value of -1e6, then plot the histogram
df = df.filter(pl.col('ScintLeftEnergy') != -1e6)\
    .filter(pl.col('AnodeBackEnergy') != -1e6)\
    .filter(pl.col('X1') != -1e6)\
    .filter(pl.col('X2') != -1e6)
z_list = []
x_list = []
# print(df.get_columns)

# i want to add a new column to the dataframe that is: (df['X2']-df['X1']) + df['X1']
df = df.with_columns( ( (df['X2']-df['X1'])).alias('XDiff') )


# write a for loop that loops over all values in the column XDiff
for i in range (10000):
    for j in range (400):
        z = -50 + 100/400 * j
        x = (z/D + 0.5) * df['XDiff'][i] + df['X1'][i]
        z_list.append(z)
        x_list.append(x)
plt.hist2d(x_list,z_list, bins=(600,100), range=((-300,300),(-80,80)), norm=colors.LogNorm())
plt.show()

# plt.hist2d(df['ScintLeftEnergy'], df['AnodeBackEnergy'], bins=(512,512), range=((0,4096),(0,4096)), norm=colors.LogNorm())
# plt.show()


# 40 deg
# df = df.filter((pl.col('ScintLeftEnergy') > 595) & (pl.col('ScintLeftEnergy') < 1200))\
#     .filter((pl.col('AnodeBackEnergy') > 450) & (pl.col('AnodeBackEnergy') < 2000))
# plt.hist2d(df['ScintLeftEnergy'], df['AnodeBackEnergy'], bins=(512,512), range=((0,4096),(0,4096)), norm=colors.LogNorm())
# plt.show()

# 10 deg
# df = df.filter((pl.col('ScintLeftEnergy') > 395) & (pl.col('ScintLeftEnergy') < 750))\
#     .filter((pl.col('AnodeBackEnergy') > 317) & (pl.col('AnodeBackEnergy') < 1600))
# plt.hist2d(df['ScintLeftEnergy'], df['AnodeBackEnergy'], bins=(512,512), range=((0,4096),(0,4096)), norm=colors.LogNorm())
# plt.show()

# z_list = []
# x_list = []

# N = min(len(df), 10000)   # <-- new safety guard
# for i in range(N):
#     for j in range(400):
#         z = -50 + 100/400 * j
#         x = (z/D + 0.5) * df['XDiff'][i] + df['X1'][i]
#         z_list.append(z)
#         x_list.append(x)
# plt.hist2d(x_list,z_list, bins=(600,100), range=((-300,300),(-80,80)), norm=colors.LogNorm())
# plt.show()
# plt.hist(df['Xavg'], bins=600, range=(-300,300), histtype='step')

# import polars as pl
# import numpy as np
# import matplotlib.colors as colors
# import matplotlib.pyplot as plt

# # load & filter (your existing filters)
# df = pl.read_parquet("run_409.parquet")
# df = df.filter(pl.col('ScintLeftEnergy') != -1e6)\
#        .filter(pl.col('AnodeBackEnergy') != -1e6)\
#        .filter(pl.col('X1') != -1e6)\
#        .filter(pl.col('X2') != -1e6)

# # parameters
# D = 42.8625          # distance between front & rear delay lines
# n_z = 400            # number of samples along each track
# z_vals = np.linspace(-50, 50, n_z)   # sample positions (same as legacy code)

# # Pull X1, X2 into numpy arrays
# X1 = df['X1'].to_numpy()
# X2 = df['X2'].to_numpy()

# # If you only want first N events (legacy used 10000), choose safely:
# N = min(len(X1), 10000)

# X1 = X1[:N]
# X2 = X2[:N]

# # slopes and intercepts (vector)
# m = (X2 - X1) / D               # shape (N,)
# b = 0.5 * (X1 + X2)             # shape (N,)

# # Build grids: shape (N, n_z)
# # Broadcasting: b[:,None] is (N,1), z_vals[None,:] is (1,n_z)
# Xgrid = b[:, None] + m[:, None] * z_vals[None, :]
# Zgrid = np.tile(z_vals, (N, 1))   # (N, n_z)

# # Flatten for hist2d
# Xflat = Xgrid.ravel()
# Zflat = Zgrid.ravel()

# plt.hist2d(Xflat, Zflat, bins=(600, 100),
#            range=((-300, 300), (-50, 50)),
#            norm=colors.LogNorm())
# plt.xlabel("X (mm)")
# plt.ylabel("z (mm)")
# plt.title(f"Projected tracks (N={N}, samples per track={n_z})")
# plt.show()
