import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ------------------------
# 1. Define a double Gaussian
# ------------------------
def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2 + C


# ------------------------
# 2. Make synthetic data
# ------------------------
np.random.seed(0)

x = np.linspace(0, 200, 800)

# true parameters
A1_true = 120
mu1_true = 90
sigma1_true = 4

A2_true = 60
mu2_true = 103
sigma2_true = 6

C_true = 8

y_true = double_gaussian(x, A1_true, mu1_true, sigma1_true,
                            A2_true, mu2_true, sigma2_true,
                            C_true)

# add Poisson noise (simulate counts)
y_noise = np.random.poisson(y_true)

# ------------------------
# 3. Fit it
# ------------------------
# initial guesses
p0 = [100, 88, 3,      # A1, mu1, sigma1
      50, 105, 5,      # A2, mu2, sigma2
      5]               # C

# parameter bounds (keeps fit stable)
bounds = (
    [0, 80, 1,   0, 90, 1,   0],         # lower
    [500, 100, 20,  300, 120, 20,  50]   # upper
)

popt, pcov = curve_fit(double_gaussian, x, y_noise, p0=p0, bounds=bounds)

y_fit = double_gaussian(x, *popt)

print("Fitted parameters:\n", popt)

# ------------------------
# 4. Plot
# ------------------------
plt.figure(figsize=(10,6))
plt.plot(x, y_noise, '.', label="Noisy Spectrum", alpha=0.4)
plt.plot(x, y_true, '--', label="True Model")
plt.plot(x, y_fit, '-', label="Double Gaussian Fit", lw=2)

plt.legend()
plt.xlabel("Channel")
plt.ylabel("Counts")
plt.title("Double Gaussian Fit Demo")
plt.show()
