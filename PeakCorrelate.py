import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Create a clean reference peak
# -----------------------------
x = np.linspace(-5, 5, 200)
reference_peak = np.exp(-(x**2) / (2 * 0.5**2))  # Gaussian

# -----------------------------
# 2) Create a shifted + noisy spectrum
# -----------------------------
true_shift = 23  # shift in bins
noise = 0.10 * np.random.randn(len(reference_peak))

spectrum = np.zeros_like(reference_peak)
spectrum[true_shift:] = reference_peak[:-true_shift]  # apply shift
spectrum += noise  # add some noise

# -----------------------------
# 3) Compute cross-correlation
# -----------------------------
corr = np.correlate(spectrum, reference_peak, mode='same')

# find index of maximum correlation
recovered_shift = np.argmax(corr) - len(reference_peak)//2

print("True shift:      ", true_shift)
print("Recovered shift: ", recovered_shift)

# -----------------------------
# 4) Plot results
# -----------------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

axs[0].plot(reference_peak)
axs[0].set_title("Reference Peak")

axs[1].plot(spectrum)
axs[1].set_title(f"Spectrum (shifted by {true_shift} bins + noise)")

axs[2].plot(corr)
axs[2].set_title("Cross-Correlation Signal")
axs[2].axvline(np.argmax(corr), color='r', linestyle='--')

plt.tight_layout()
plt.show()
