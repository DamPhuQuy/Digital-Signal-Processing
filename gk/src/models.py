"""Module models.py: Các cấu trúc dữ liệu miền (Domain Data Models) cho VAD Pipeline.

Đóng gói dữ liệu giữa 5 khối xử lý bằng dataclass có định kiểu nghiêm ngặt (type hints),
thay thế hoàn toàn các dictionary phi cấu trúc nhằm tăng độ tin cậy và tính rõ ràng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AudioSignal:
    """Khối 1 Output: Biểu diễn một tín hiệu âm thanh và nhãn chuẩn (Ground Truth).

    Thuộc tính:
        name (str): Tên định danh file (ví dụ: 'phone_F1', 'studio_M2').
        wav_path (Path): Đường dẫn tới file âm thanh .wav.
        fs (int): Tần số lấy mẫu của tín hiệu (Hz).
        signal (np.ndarray): Mảng 1D biên độ tín hiệu đã chuẩn hóa về dải [-1.0, 1.0].
        duration_sec: float: Thời lượng của tín hiệu tính bằng giây.
        is_phone (bool): True nếu thuộc môi trường điện thoại (SNR thấp), False nếu là Studio.
        gt_segments (List[Tuple[float, float, str]]): Danh sách các đoạn chuẩn (start, end, label).
        gt_boundaries (List[float]): Danh sách các mốc thời gian biên chuẩn (giây).
        lab_path (Optional[Path]): Đường dẫn tới file nhãn .lab (nếu có).
    """

    name: str
    wav_path: Path
    fs: int
    signal: np.ndarray
    duration_sec: float
    is_phone: bool
    gt_segments: List[Tuple[float, float, str]] = field(default_factory=list)
    gt_boundaries: List[float] = field(default_factory=list)
    lab_path: Optional[Path] = None

    @property
    def environment_name(self) -> str:
        """Tên môi trường thu âm."""
        return "Phone" if self.is_phone else "Studio"

    @property
    def ground_truth_segments(self) -> List[Tuple[float, float, str]]:
        return self.gt_segments

    @property
    def ground_truth_boundaries(self) -> List[float]:
        return self.gt_boundaries

    @property
    def file_path(self) -> Path:
        return self.wav_path

    @property
    def is_female(self) -> bool:
        return "_f" in self.name.lower() or "f" in self.name.lower()


# Tương thích ngược
AudioSample = AudioSignal


@dataclass(frozen=True)
class FramedSignal:
    """Khối 2 Output: Biểu diễn tín hiệu sau khi phân khung và nhân hàm cửa sổ.

    Thuộc tính:
        frames (np.ndarray): Ma trận kích thước (num_frames, frame_len) chứa các khung tín hiệu.
        timestamps (np.ndarray): Mảng 1D chứa mốc thời gian tại tâm mỗi khung (giây).
        frame_len (int): Chiều dài mỗi khung tính bằng số mẫu (N).
        frame_shift (int): Bước dịch giữa 2 khung liên tiếp tính bằng số mẫu (M).
        window_type (str): Tên loại cửa sổ đã áp dụng ('hamming', 'rectangular', 'hann').
        fs (int): Tần số lấy mẫu gốc (Hz).
    """

    frames: np.ndarray
    timestamps: np.ndarray
    frame_len: int
    frame_shift: int
    window_type: str
    fs: int
    raw_frames: Optional[np.ndarray] = None

    @property
    def num_frames(self) -> int:
        """Tổng số khung."""
        return int(len(self.timestamps))

    @property
    def overlap_ratio(self) -> float:
        """Tỷ lệ chồng lấp giữa 2 khung liên tiếp."""
        return (self.frame_len - self.frame_shift) / float(self.frame_len)


@dataclass(frozen=True)
class ShortTimeFeatures:
    """Khối 3 Output: Chứa các đặc trưng năng lượng ngắn hạn (Short-Time Features: STE & MA).

    Thuộc tính:
        ste_raw (np.ndarray): Năng lượng ngắn hạn (Short-Time Energy) gốc.
        ste_norm (np.ndarray): STE đã chuẩn hóa Min-Max về khoảng [0.0, 1.0].
        ma_raw (np.ndarray): Độ lớn trung bình ngắn hạn (Magnitude Average) gốc.
        ma_norm (np.ndarray): MA đã chuẩn hóa Min-Max về khoảng [0.0, 1.0].
    """

    ste_raw: np.ndarray
    ste_norm: np.ndarray
    ma_raw: Optional[np.ndarray] = None
    ma_norm: Optional[np.ndarray] = None


# Tương thích ngược
AcousticFeatures = ShortTimeFeatures


@dataclass(frozen=True)
class Thresholds:
    """Bảng lưu trữ các bộ ngưỡng tối ưu tìm được từ tập huấn luyện cho 3 thuật toán."""

    binary_global: float
    binary_phone: float
    binary_studio: float
    histogram_global: float
    gaussian_global: float
    mean_speech: float = 0.0
    std_speech: float = 0.0
    mean_silence: float = 0.0
    std_silence: float = 0.0

    def get_threshold(self, algorithm: str, is_phone: bool = False, use_adaptive: bool = True) -> float:
        """Lấy ngưỡng phù hợp với thuật toán và môi trường.

        Tham số:
            algorithm: Thuật toán cần lấy ngưỡng ('binary', 'histogram', 'gaussian').
            is_phone: True nếu là môi trường Phone, False nếu Studio.
            use_adaptive: True nếu dùng ngưỡng thích nghi riêng theo môi trường SNR,
                          False nếu dùng ngưỡng chung toàn cục (dùng chung cho cả 4 file).
        """
        algo = algorithm.lower()
        if algo == "binary":
            if use_adaptive:
                return self.binary_phone if is_phone else self.binary_studio
            return self.binary_global
        if algo == "histogram":
            return self.histogram_global
        if algo == "gaussian":
            return self.gaussian_global
        raise ValueError(f"Thuật toán không hỗ trợ: {algorithm}. Chọn: 'binary', 'histogram', 'gaussian'.")


# Tương thích ngược
VADThresholds = Thresholds


@dataclass(frozen=True)
class SegmentationResult:
    """Khối 4 Output: Kết quả phân đoạn tiếng nói và khoảng lặng của một tín hiệu âm thanh."""

    signal_name: str
    algorithm: str
    threshold_used: float
    raw_decisions: np.ndarray
    filtered_decisions: np.ndarray
    predicted_boundaries: List[float]
    gt_boundaries: List[float]
    timestamps: np.ndarray
    matched_pairs: List[Tuple[float, float]] = field(default_factory=list)
    segments: List[Tuple[float, float, str]] = field(default_factory=list)

    def __init__(
        self,
        signal_name: str = "",
        algorithm: str = "",
        threshold_used: float = 0.0,
        raw_decisions: Optional[np.ndarray] = None,
        filtered_decisions: Optional[np.ndarray] = None,
        predicted_boundaries: Optional[List[float]] = None,
        gt_boundaries: Optional[List[float]] = None,
        timestamps: Optional[np.ndarray] = None,
        matched_pairs: Optional[List[Tuple[float, float]]] = None,
        sample_name: Optional[str] = None,
        ground_truth_boundaries: Optional[List[float]] = None,
        segments: Optional[List[Tuple[float, float, str]]] = None,
        **kwargs,
    ) -> None:
        name = sample_name if sample_name is not None else signal_name
        gb = ground_truth_boundaries if ground_truth_boundaries is not None else gt_boundaries
        object.__setattr__(self, "signal_name", name)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "threshold_used", threshold_used)
        object.__setattr__(self, "raw_decisions", raw_decisions if raw_decisions is not None else np.array([]))
        object.__setattr__(self, "filtered_decisions", filtered_decisions if filtered_decisions is not None else np.array([]))
        object.__setattr__(self, "predicted_boundaries", predicted_boundaries if predicted_boundaries is not None else [])
        object.__setattr__(self, "gt_boundaries", gb if gb is not None else [])
        object.__setattr__(self, "timestamps", timestamps if timestamps is not None else np.array([]))
        object.__setattr__(self, "matched_pairs", matched_pairs if matched_pairs is not None else [])
        object.__setattr__(self, "segments", segments if segments is not None else [])

    @property
    def sample_name(self) -> str:
        """Bí danh tương thích ngược."""
        return self.signal_name

    @property
    def ground_truth_boundaries(self) -> List[float]:
        return self.gt_boundaries


# Tương thích ngược
VADResult = SegmentationResult


@dataclass(frozen=True)
class EvaluationMetrics:
    """Khối 5 Output: Các chỉ số đánh giá định lượng sai số phân đoạn."""

    signal_name: str
    environment: str
    algorithm: str
    threshold: float
    mae_ms: float
    rmse_ms: float
    accuracy: float
    fer: float
    precision: float
    recall: float
    f1: float
    predicted_boundaries: List[float] = field(default_factory=list)
    gt_boundaries: List[float] = field(default_factory=list)

    def __init__(
        self,
        signal_name: str = "",
        environment: str = "",
        algorithm: str = "",
        threshold: float = 0.0,
        mae_ms: float = 0.0,
        rmse_ms: float = 0.0,
        accuracy: float = 0.0,
        fer: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0,
        predicted_boundaries: Optional[List[float]] = None,
        gt_boundaries: Optional[List[float]] = None,
        sample_name: Optional[str] = None,
    ) -> None:
        name = sample_name if sample_name is not None else signal_name
        object.__setattr__(self, "signal_name", name)
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "mae_ms", mae_ms)
        object.__setattr__(self, "rmse_ms", rmse_ms)
        object.__setattr__(self, "accuracy", accuracy)
        object.__setattr__(self, "fer", fer)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "recall", recall)
        object.__setattr__(self, "f1", f1)
        object.__setattr__(self, "predicted_boundaries", predicted_boundaries if predicted_boundaries is not None else [])
        object.__setattr__(self, "gt_boundaries", gt_boundaries if gt_boundaries is not None else [])

    @property
    def sample_name(self) -> str:
        """Bí danh tương thích ngược."""
        return self.signal_name


# Tương thích ngược
SegmentationMetrics = EvaluationMetrics
