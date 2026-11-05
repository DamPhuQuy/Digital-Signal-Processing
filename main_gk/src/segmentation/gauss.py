import numpy as np


def find_threshold_gaussian(
    ste: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, float, float]:
    silence_values = []
    speech_values = []

    for i in range(len(ste)):
        if labels[i] == 0:
            silence_values.append(ste[i])
        else:
            speech_values.append(ste[i])

    mean_sil = float(np.mean(silence_values)) if len(silence_values) > 0 else 0.001
    std_sil = float(np.std(silence_values)) if len(silence_values) > 0 else 0.001

    mean_sp = float(np.mean(speech_values)) if len(speech_values) > 0 else 0.1
    std_sp = float(np.std(speech_values)) if len(speech_values) > 0 else 0.05

    threshold = (
        (mean_sil * std_sp + mean_sp * std_sil) / (std_sp + std_sil)
        if (std_sp + std_sil) != 0
        else (mean_sil + mean_sp) / 2
    )

    return float(threshold), mean_sp, std_sp, mean_sil, std_sil

