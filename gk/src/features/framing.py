"""Module framing.py: Phân khung và áp dụng hàm cửa sổ (Hamming, Hann, Rectangular).

Thuộc Khối 2 trong quy trình xử lý tín hiệu:
- Chia tín hiệu thành các khung thời gian ngắn (mặc định: 25 ms, bước dịch 10 ms -> overlap 60%).
- Nhân với hàm cửa sổ Hamming w[n] để làm mượt 2 biên, triệt tiêu hiện tượng rò rỉ phổ.
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np

from src.models import AudioSample, AudioSignal, FramedSignal


def get_window(window_type: str, frame_len: int) -> np.ndarray:
    """Tạo mảng trọng số của hàm cửa sổ theo công thức toán học rời rạc."""
    if frame_len <= 0:
        raise ValueError(f"Độ dài khung phải là số nguyên dương, nhận: {frame_len}")

    m = np.arange(frame_len, dtype=np.float64)
    win_type = window_type.lower()

    if win_type in ("rectangular", "rect"):
        return np.ones(frame_len, dtype=np.float64)

    if frame_len == 1:
        return np.ones(1, dtype=np.float64)

    if win_type == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * m / (frame_len - 1))

    if win_type == "hann":
        return 0.50 * (1.0 - np.cos(2.0 * np.pi * m / (frame_len - 1)))

    raise ValueError(f"Không hỗ trợ loại cửa sổ: {window_type}. Chọn 'hamming', 'rectangular', 'hann'.")


def apply_framing(
    signal: AudioSignal | np.ndarray,
    frame_size_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window_type: str = "hamming",
    fs: Optional[int] = None,
) -> FramedSignal:
    """Phân chia tín hiệu thành ma trận các khung ngắn hạn và nhân cửa sổ."""
    if isinstance(signal, AudioSignal):
        sig = signal.signal
        sampling_rate = signal.fs
    else:
        sig = signal
        sampling_rate = fs if fs is not None else 16000

    frame_len = int(round(frame_size_ms * 1e-3 * sampling_rate))
    frame_shift = int(round(frame_shift_ms * 1e-3 * sampling_rate))

    if frame_len <= 0 or frame_shift <= 0:
        raise ValueError("Chiều dài khung và bước dịch khung phải lớn hơn 0")

    sig_len = len(sig)
    if sig_len < frame_len:
        padded_sig = np.pad(sig, (0, frame_len - sig_len), mode="constant")
        num_frames = 1
    else:
        num_frames = 1 + int(np.floor((sig_len - frame_len) / frame_shift))
        padded_sig = sig

    window = get_window(window_type, frame_len)

    frames = np.empty((num_frames, frame_len), dtype=np.float64)
    raw_frames = np.empty((num_frames, frame_len), dtype=np.float64)
    timestamps = np.empty(num_frames, dtype=np.float64)

    for i in range(num_frames):
        start_idx = i * frame_shift
        end_idx = start_idx + frame_len
        frame = padded_sig[start_idx:end_idx]
        raw_frames[i, :] = frame
        frames[i, :] = frame * window
        timestamps[i] = (start_idx + frame_len / 2.0) / sampling_rate

    return FramedSignal(
        frames=frames,
        timestamps=timestamps,
        frame_len=frame_len,
        frame_shift=frame_shift,
        window_type=window_type,
        fs=sampling_rate,
        raw_frames=raw_frames,
    )


def frame_signal(
    signal: AudioSignal | np.ndarray,
    fs: Optional[int] = None,
    frame_size_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    window_type: str = "hamming",
) -> Tuple[np.ndarray, np.ndarray] | FramedSignal:
    """Hàm wrapper tương thích ngược: trả về (frames, timestamps) nếu đầu vào là np.ndarray."""
    framed = apply_framing(signal, frame_size_ms=frame_size_ms, frame_shift_ms=frame_shift_ms, window_type=window_type, fs=fs)
    if isinstance(signal, np.ndarray):
        return framed.frames, framed.timestamps
    return framed
