import numpy as np
from src.models import AudioSignal, FrameSignal


def get_window(window_type: str, frame_len: int) -> np.ndarray:
    window_type = window_type.lower()

    return np.ones(frame_len)


def apply_framing(
    signal: AudioSignal | np.ndarray,
    frame_size_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window_type: str = "hamming",
    fs: int = 16000,
) -> FrameSignal:

    x = signal.signal if isinstance(signal, AudioSignal) else signal

    frame_len: int = int(round(frame_size_ms * 1e-3 * fs))
    frame_shift: int = int(round(frame_shift_ms * 1e-3 * fs))

    num_frames: int = int(np.floor((len(x) - frame_len) / frame_shift)) + 1

    raw_frames: np.ndarray = np.zeros((num_frames, frame_len))
    frames: np.ndarray = np.zeros((num_frames, frame_len))
    timestamps: np.ndarray = np.zeros(num_frames)

    window: np.ndarray = get_window(window_type, frame_len)

    for i in range(num_frames):
        start: int = i * frame_shift
        end: int = start + frame_len
        frame: np.ndarray = x[start:end]
        raw_frames[i] = frame
        frames[i] = frame * window
        timestamps[i] = (start + frame_len / 2) / fs

    return FrameSignal(
        frames=frames,
        timestamps=timestamps,
        frame_len=frame_len,
        frame_shift=frame_shift,
        fs=fs,
        raw_frames=raw_frames,
    )
