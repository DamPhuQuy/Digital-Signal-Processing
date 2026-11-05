import numpy as np


def estimate_f0_frame(
    frame: np.ndarray,
    fs: int,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    voicing_thresh: float = 0.35,
    eps: float = 1e-10,
) -> float:
    x = frame.astype(np.float64) - np.mean(frame)
    if np.max(np.abs(x)) < 1e-6:
        return float(np.nan)

    min_lag = int(round(fs / f0_max))
    max_lag = int(round(fs / f0_min))

    corr = np.correlate(x, x, mode="full")
    corr = corr[len(corr) // 2:]  # Chỉ giữ độ trễ lag >= 0

    max_lag_local = min(max_lag, len(corr) - 1)
    if max_lag_local <= min_lag:
        return float(np.nan)

    search_zone = corr[min_lag:max_lag_local]
    peak_idx = int(np.argmax(search_zone)) + min_lag
    confidence = corr[peak_idx] / (corr[0] + eps)

    if confidence >= voicing_thresh and peak_idx > 0:
        return float(fs / peak_idx)

    return float(np.nan)


def estimate_f0_track(
    signal: np.ndarray,
    fs: int,
    timestamps: np.ndarray,
    frame_len: int,
    f0_min: float = 60.0,
    f0_max: float = 400.0,
    voicing_thresh: float = 0.35,
) -> np.ndarray:
    f0_hz = np.full(len(timestamps), np.nan)
    x = signal.astype(np.float64)
    total_samples = len(x)

    for i, t in enumerate(timestamps):
        start = int(round(t * fs - frame_len / 2.0))
        end = start + frame_len
        if start < 0 or end > total_samples:
            frame = np.zeros(frame_len)
            s_valid = max(0, start)
            e_valid = min(total_samples, end)
            offset = max(0, -start)
            frame[offset:offset + (e_valid - s_valid)] = x[s_valid:e_valid]
        else:
            frame = x[start:end]

        f0_hz[i] = estimate_f0_frame(
            frame=frame,
            fs=fs,
            f0_min=f0_min,
            f0_max=f0_max,
            voicing_thresh=voicing_thresh,
        )

    return f0_hz
