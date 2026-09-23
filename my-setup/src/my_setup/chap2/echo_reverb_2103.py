from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

# y[n] = x[n] + a y[n - D]
#
# = x[n] + a x[n - D] + a^2 y[n - 2D]
#
# = x[n] + a x[n - D] + a^2 x[n - 2D]
#   + a^3 y[n - 3D]
#
# = ...


a: float = 0.7
T: float = 0.1

wav_path = Path(__file__).parent / "a.wav"
x, Fs = sf.read(wav_path)

D = round(T * Fs)

print(f"Fs = {Fs}")
print(f"D = {D} samples")
print(f"Delay = {D / Fs:.3f} seconds")

y = np.zeros_like(x)

for n in range(len(x)):
    y[n] = x[n]

    if n >= D:
        y[n] += a * y[n - D]

# Plot input and output

n = np.arange(len(x))

# Plot
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(n, x)
plt.xlabel("n")
plt.ylabel("x[n]")
plt.title("Input Signal")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(n, y)
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Output Signal")
plt.grid()

plt.tight_layout()

# Show plot without blocking
plt.show(block=False)

# Play input
print("Playing input...")
sd.play(x, Fs)
sd.wait()

input()

# Play output
print("Playing output...")
sd.play(y, Fs)
sd.wait()

# Keep plot open
input()
