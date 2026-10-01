"""Module energy.py: Trích xuất và chuẩn hóa đặc trưng năng lượng ngắn hạn (STE, MA, ZCR).

Thuộc Khối 3 trong quy trình xử lý tín hiệu:
- Tính Năng lượng ngắn hạn (Short-Time Energy - STE).
- Tính Độ lớn trung bình ngắn hạn (Magnitude Average - MA).
- Tính Tốc độ qua điểm không (Zero-Crossing Rate - ZCR).
- Chuẩn hóa Min-Max các đặc trưng về đoạn [0.0, 1.0].
"""

from __future__ import annotations

import numpy as np

from src.models import AcousticFeatures, FramedSignal, ShortTimeFeatures


def compute_short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Tính năng lượng ngắn hạn (Short-Time Energy - STE): E[i] = sum(x_i[m]^2)."""
    return np.sum(frames ** 2, axis=1)


compute_ste = compute_short_time_energy


def compute_magnitude_average(frames: np.ndarray) -> np.ndarray:
    """Tính độ lớn trung bình ngắn hạn (Magnitude Average - MA): MA[i] = sum(|x_i[m]|)."""
    return np.sum(np.abs(frames), axis=1)


compute_ma = compute_magnitude_average


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    """Tính Tốc độ qua điểm không (Zero-Crossing Rate - ZCR) cho từng khung."""
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)

    signs = np.sign(frames)
    signs[signs == 0] = 1.0
    diffs = np.abs(signs[:, 1:] - signs[:, :-1])
    zcr = np.sum(diffs, axis=1) / (2.0 * (frames.shape[1] - 1))
    return zcr


compute_zero_crossing_rate = compute_zcr


def normalize_minmax(feature: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Chuẩn hóa vector đặc trưng về đoạn [0.0, 1.0] bằng Min-Max Scaling."""
    f_min = float(np.min(feature))
    f_max = float(np.max(feature))
    denom = f_max - f_min
    if denom <= 1e-12:
        return np.zeros_like(feature, dtype=np.float64)
    return (feature - f_min) / denom


normalize_feature = normalize_minmax


def extract_features(framed: FramedSignal) -> ShortTimeFeatures:
    """Trích xuất và chuẩn hóa đặc trưng năng lượng ngắn hạn (Short-Time Energy - STE)."""
    ste_raw = compute_short_time_energy(framed.frames)
    ste_norm = normalize_minmax(ste_raw)

    return ShortTimeFeatures(
        ste_raw=ste_raw,
        ste_norm=ste_norm,
    )


extract_short_time_features = extract_features
extract_acoustic_features = extract_features
