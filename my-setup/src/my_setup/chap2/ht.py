from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf


def giam_fs_2_lan(x):
    length_out = len(x) // 2
    y = np.zeros(length_out)
    for n in range(length_out):
        y[n] = x[2 * n]
    return y


def tang_fs_2_lan(x):
    length_in = len(x)
    y = np.zeros(2 * length_in)
    for n in range(2 * length_in):
        if n % 2 == 0:
            y[n] = x[n // 2]
        else:
            left_idx = (n - 1) // 2
            right_idx = (n + 1) // 2
            if right_idx < length_in:
                y[n] = 0.5 * (x[left_idx] + x[right_idx])
            else:
                y[n] = x[left_idx]
    return y


wav_path = Path(__file__).parent / "16k.wav"
x, fs = sf.read(wav_path)

x1_a = giam_fs_2_lan(x)
fs1_a = 8000

x2_a = giam_fs_2_lan(x1_a)
fs2_a = 4000

sd.play(x, fs)
sd.wait()

sd.play(x1_a, fs1_a)
sd.wait()

sd.play(x2_a, fs2_a)
sd.wait()

xb = x2_a
fs_b = 4000

x1_b = tang_fs_2_lan(xb)
fs1_b = 8000

x2_b = tang_fs_2_lan(x1_b)
fs2_b = 16000

sd.play(xb, fs_b)
sd.wait()

sd.play(x1_b, fs1_b)
sd.wait()

sd.play(x2_b, fs2_b)
sd.wait()
