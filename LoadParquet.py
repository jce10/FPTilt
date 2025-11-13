import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# to calculate the incident position of a particle
# on the focal plane according to the xf equation (eq. 5) 
# in https://doi.org/10.1016/0029-554X(75)90121-4
def kinematicFP(h, alpha, S, parquetfile):
    
    # load X1 and X2 from the parquet file
    df = pd.read_parquet(parquetfile, columns=['X1', 'X2'])
    x1 = df['X1'].values
    x2 = df['X2'].values

    # tangent and cotangent of alpha
    tga = np.sin(alpha) / np.cos(alpha)
    ctga = np.cos(alpha) / np.sin(alpha)

    num = (x2*S/np.sqrt(1 + tga**2)) - (x2 - x1)*h
    deno = (S/np.sqrt(1 + tga**2)) - ((x2 - x1)/np.sqrt(1 + ctga**2))

    xf = num / deno

    # rounding xf to 12 digits
    rounded_xf = np.array([float('{:.12e}'.format(val)) for val in xf])

    return rounded_xf