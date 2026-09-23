from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import lfilter, convolve
import matplotlib.pyplot as plt

wav_path = Path(__file__).parent / "a.wav"
x, fs = sf.read(wav_path)

x = x.astype(float)

h = np.array([1, 2, 3]) / 6

b = h
a = np.array([1])


def apply_filter(x, b, a):
    return lfilter(b, a, x)


def apply_convolution(x, h):
    y_convolve = convolve(x, h, mode="full")
    return y_convolve[:len(x)]


def apply_khong_thu_vien(x, h):
    y_manual = np.zeros(len(x))

    m = len(h)

    for n in range(len(x)):
        for k in range(m):
            if n - k >= 0:
                y_manual[n] += h[k] * x[n - k]

    return y_manual


def so_sanh(x, h, b, a):
    y1 = apply_filter(x, b, a)
    y2 = apply_convolution(x, h)
    y3 = apply_khong_thu_vien(x, h)

    print("apply filter: ", y1)
    print("apply_convolution: ", y2)
    print("apply_khong_thu_vien: ", y3)

    plt.plot(y1[:100], label="apply_filter", linewidth=2.5)
    plt.plot(y2[:100], label="apply_convolution", linestyle="--")
    plt.plot(y3[:100], label="apply_khong_thu_vien", linestyle=":")
    plt.title("So sánh kết quả của 3 phương pháp lọc")
    plt.xlabel("Mẫu (Sample)")
    plt.ylabel("Biên độ (Amplitude)")
    plt.legend()
    plt.grid(True)
    plt.show()

so_sanh(x, h, b, a)
