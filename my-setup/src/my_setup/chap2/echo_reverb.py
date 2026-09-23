# cài đặt hệ thống có 3 tiếng vọng
# input: a sound signal
# output: y[n] = x[n] + a * x[n - D] + a^2 * x[n - 2D] + a^3 * x[n - 3D]
# h[n] = s[n] + a * s[n - D] + a^2 * s[n - 2D] + a^3 * s[n - 3D] = [1, 0, 0, 0, ...,a, 0, 0, ..., a^2, ... 0, a^3] (h[0], ..., h[D], ..., h[2D], ..., h[3D])
# hệ có 2 tham số: a: hệ số suy giảm = 0.6; D: độ trễ = T.Fs (T: time delay = 100ms) với D là tròn

from pathlib import Path
import numpy as np
import soundfile as sf

a = 0.7
T = 0.1  # 100 ms

wav_path = Path(__file__).parent / "a.wav"
x, Fs = sf.read(wav_path)

print("Fs =", Fs)

D = round(T * Fs)

print("D =", D, "samples")

h = np.zeros(3 * D + 1)

h[0] = 1
h[D] = a
h[2 * D] = a**2
h[3 * D] = a**3

y = np.convolve(x, h)

sf.write("output_echo.wav", y, Fs)

print("Done!")
