import numpy as np
import matplotlib.pyplot as plt

A = 10
omega = np.pi / 3
phi = 0

num_samples = 1000

n = np.arange(num_samples)

signal = A * np.sin(omega * n + phi)

mean = 0
variance = 4

standard_deviation = np.sqrt(variance)

noise = np.random.normal(mean, standard_deviation, num_samples)

noisy_signal = signal + noise

N = 21
window_size = 2 * N + 1

filtered_signal = np.zeros(num_samples)
for n_idx in range(num_samples):
    total = 0.0
    for k in range(0, 2 * N + 1):
        if 0 <= n_idx - k < num_samples:
            total += noisy_signal[n_idx - k]
    filtered_signal[n_idx] = (1 / window_size) * total

plt.figure(figsize=(12, 8))

# Signal sạch
plt.subplot(3, 1, 1)
plt.plot(n[:50], signal[:50], marker="o", markersize=3)
plt.title("1. Original Signal s[n]")
plt.ylabel("Amplitude")
plt.grid(True)

# Signal bị nhiễu
plt.subplot(3, 1, 2)
plt.plot(
    n[:50],
    noisy_signal[:50],
    marker="o",
    markersize=3,
    color="orange",
)
plt.title("2. Noisy Signal x[n] = s[n] + g[n]")
plt.ylabel("Amplitude")
plt.grid(True)

# Signal sau khi lọc trung bình trượt đối xứng
plt.subplot(3, 1, 3)
plt.plot(
    n[:50],
    filtered_signal[:50],
    marker="o",
    markersize=3,
    color="green",
)
plt.title(f"3. Moving Average Filter y[n] (N={N}, Window Size={window_size})")
plt.xlabel("Sample n")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()
