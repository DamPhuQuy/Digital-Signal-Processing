import numpy as np
from src.features.framing import apply_framing
from src.models import AudioSignal, FrameSignal, ShortTimeFeatures


def compute_ste(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return np.sum(frames**2)
    return np.sum(frames**2, axis=1)

def normalize_minmax(feature: np.ndarray) -> np.ndarray:
    f_min = np.min(feature)
    f_max = np.max(feature)
    denom = f_max - f_min

    if denom > 0:
        return (feature - f_min) / denom
    return np.zeros_like(feature)


def extract_features(
    signal: FrameSignal | AudioSignal | np.ndarray,
    frame_size_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window_type: str = "hamming",
) -> ShortTimeFeatures:
    if isinstance(signal, FrameSignal):
        frames = signal.frames
    else:
        framed = apply_framing(
            signal,
            frame_size_ms=frame_size_ms,
            frame_shift_ms=frame_shift_ms,
            window_type=window_type,
        )
        frames = framed.frames

    ste_raw: np.ndarray = compute_ste(frames)
    ste_norm: np.ndarray = normalize_minmax(ste_raw)
    log_ste: np.ndarray = np.log(ste_raw + 1e-12)

    return ShortTimeFeatures(
        ste_raw=ste_raw,
        ste_norm=ste_norm,
        log_ste=log_ste,
    )
