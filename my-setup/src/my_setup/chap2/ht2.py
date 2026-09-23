from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf


def giam_fs_2_lan(x):
    length = len(x) // 2
    y = np.zeros(length)
    for n in range(length):
        y[n] = x[2 * n]
    return y


def tang_fs_2_lan(x):
    length = len(x)
    y = np.zeros(2 * length)
    for n in range(2 * length):
        if n % 2 == 0:
            y[n] = x[n // 2]
        else:
            left = (n - 1) // 2
            right = (n + 1) // 2
            if right < length:
                y[n] = 0.5 * (x[left] + x[right])
            else:
                y[n] = x[left]
    return y


wav_path = Path(__file__).parent / "16k.wav"
x, fs = sf.read(wav_path)

x1_a, fs1_a = giam_fs_2_lan(x), 8000
x2_a, fs2_a = giam_fs_2_lan(x1_a), 4000

xb, fs_b = x2_a, 4000
x1_b, fs1_b = tang_fs_2_lan(xb), 8000
x2_b, fs2_b = tang_fs_2_lan(x1_b), 16000

for sig, f in [(x, fs), (x1_a, fs1_a), (x2_a, fs2_a), (xb, fs_b), (x1_b, fs1_b), (x2_b, fs2_b)]:
    sd.play(sig, f)
    sd.wait()
