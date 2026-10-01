"""Module pipeline.py: Lớp điều phối SpeechSegmenter (Pipeline Orchestrator).

Xâu chuỗi 5 khối chức năng vòng đời xử lý tín hiệu DSP:
  Khối 1: Preprocessing (Tiền xử lý tín hiệu: Nạp âm thanh & Ground Truth, phân khung 25ms, cửa sổ Hamming)
    ↓
  Khối 2: Feature Extraction (Trích xuất đặc trưng: Năng lượng ngắn hạn STE, chuẩn hóa Min-Max [0, 1])
    ↓
  Khối 3: Threshold Calibration (Huấn luyện ngưỡng: Tìm ngưỡng tối ưu T trên tập TRAIN)
    ↓
  Khối 4: Inference & Post-processing (Phân đoạn VAD trên TEST, lọc khoảng lặng < 200ms, trích xuất biên)
    ↓
  Khối 5: Evaluation & Visualization (Đánh giá định lượng MAE/RMSE/FER/F1, vẽ đồ thị tổng hợp 2x2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    AcousticFeatures,
    AudioSample,
    AudioSignal,
    EvaluationMetrics,
    FramedSignal,
    SegmentationResult,
    ShortTimeFeatures,
    Thresholds,
    VADResult,
    VADThresholds,
)
from src.audio import (
    create_frame_labels,
    load_dataset,
)
from src.features import (
    apply_framing,
    extract_features,
)
from src.segmentation import (
    find_all_thresholds,
    segment_signal,
)
from src.evaluate import (
    evaluate_segmentation,
)


class SpeechSegmenter:
    """Bộ phân đoạn tín hiệu tiếng nói và khoảng lặng (Speech/Silence Discrimination)."""

    def __init__(
        self,
        frame_size_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        window_type: str = "hamming",
        min_silence_ms: float = 200.0,
    ) -> None:
        """Khởi tạo cấu hình phân đoạn tín hiệu.

        Tham số:
            frame_size_ms (float): Độ dài khung (mặc định: 25.0 ms).
            frame_shift_ms (float): Bước dịch khung (mặc định: 10.0 ms).
            window_type (str): Hàm cửa sổ (mặc định: 'hamming').
            min_silence_ms (float): Ngưỡng lọc khoảng lặng tối thiểu (mặc định: 200.0 ms).
        """
        self.frame_size_ms = frame_size_ms
        self.frame_shift_ms = frame_shift_ms
        self.window_type = window_type
        self.min_silence_ms = min_silence_ms
        self.thresholds: Optional[Thresholds] = None

    def extract_features(
        self,
        signal: AudioSignal,
    ) -> Tuple[FramedSignal, ShortTimeFeatures, np.ndarray]:
        """Thực thi phân khung và trích xuất đặc trưng STE/MA cho một tín hiệu âm thanh.

        Tham số:
            signal (AudioSignal): Tín hiệu âm thanh đầu vào.

        Trả về:
            Tuple[FramedSignal, ShortTimeFeatures, np.ndarray]:
                - framed: Tín hiệu sau khi phân khung và nhân cửa sổ.
                - features: Đặc trưng STE và MA đã chuẩn hóa.
                - labels: Nhãn nhị phân mức khung từ Ground Truth (nếu có).
        """
        # Khối 2: Framing & Windowing
        framed = apply_framing(
            signal,
            frame_size_ms=self.frame_size_ms,
            frame_shift_ms=self.frame_shift_ms,
            window_type=self.window_type,
        )

        # Khối 3: Feature Extraction (STE / MA)
        features = extract_features(framed)

        # Tạo nhãn mức khung từ Ground Truth
        labels = create_frame_labels(signal, framed.timestamps)

        return framed, features, labels

    transform_features = extract_features

    def find_thresholds(self, train_signals: List[AudioSignal]) -> Thresholds:
        """Xác định bộ ngưỡng tối ưu trên tập huấn luyện cho cả 3 thuật toán.

        Tham số:
            train_signals (List[AudioSignal]): Danh sách các tín hiệu tập train.

        Trả về:
            Thresholds: Bảng ngưỡng tối ưu tìm được.
        """
        features_list: List[ShortTimeFeatures] = []
        labels_list: List[np.ndarray] = []

        for sig in train_signals:
            _, feats, lbls = self.extract_features(sig)
            features_list.append(feats)
            labels_list.append(lbls)

        # Khối 4: Tìm ngưỡng
        self.thresholds = find_all_thresholds(
            train_signals=train_signals,
            train_features=features_list,
            train_labels=labels_list,
        )
        return self.thresholds

    fit = find_thresholds

    def segment_signal(
        self,
        signal: AudioSignal,
        algorithm: str = "binary",
        custom_threshold: Optional[float] = None,
        use_adaptive: bool = True,
    ) -> Tuple[FramedSignal, ShortTimeFeatures, SegmentationResult]:
        """Thực thi phân đoạn tiếng nói và khoảng lặng cho một tín hiệu âm thanh.

        Tham số:
            signal (AudioSignal): Tín hiệu âm thanh cần phân đoạn.
            algorithm (str): Thuật toán ('binary', 'histogram', 'gaussian').
            custom_threshold (Optional[float]): Ngưỡng tự định nghĩa nếu không dùng ngưỡng học được.
            use_adaptive (bool): True nếu dùng ngưỡng thích nghi riêng theo môi trường Phone/Studio,
                                 False nếu dùng ngưỡng chung toàn cục.

        Trả về:
            Tuple[FramedSignal, ShortTimeFeatures, SegmentationResult]: Dữ liệu trung gian và kết quả phân đoạn.
        """
        framed, features, _ = self.extract_features(signal)

        # Xác định ngưỡng áp dụng
        if custom_threshold is not None:
            th = custom_threshold
        elif self.thresholds is not None:
            th = self.thresholds.get_threshold(algorithm, is_phone=signal.is_phone, use_adaptive=use_adaptive)
        else:
            raise ValueError("Chưa xác định ngưỡng. Hãy gọi .find_thresholds() hoặc truyền custom_threshold.")

        # Khối 4: Cắt ngưỡng, lọc 200ms và trích xuất biên
        seg_res = segment_signal(
            signal=signal,
            features=features,
            framed=framed,
            threshold=th,
            algorithm=algorithm,
        )
        return framed, features, seg_res

    segment = segment_signal
    predict_sample = segment_signal

    def evaluate_signal(
        self,
        signal: AudioSignal,
        algorithm: str = "binary",
        custom_threshold: Optional[float] = None,
        use_adaptive: bool = True,
    ) -> Tuple[FramedSignal, ShortTimeFeatures, SegmentationResult, EvaluationMetrics]:
        """Thực thi phân đoạn và tính toán sai số định lượng."""
        framed, features, seg_res = self.segment_signal(
            signal,
            algorithm=algorithm,
            custom_threshold=custom_threshold,
            use_adaptive=use_adaptive,
        )
        metrics = evaluate_segmentation(signal, seg_res, framed.timestamps)
        return framed, features, seg_res, metrics

    evaluate_sample = evaluate_signal

    def evaluate_dataset(
        self,
        test_signals: List[AudioSignal],
        algorithm: str = "binary",
        use_adaptive: bool = True,
    ) -> List[Tuple[AudioSignal, FramedSignal, ShortTimeFeatures, SegmentationResult, EvaluationMetrics]]:
        """Đánh giá toàn bộ tập kiểm thử, trả về danh sách đầy đủ thông tin để hiển thị."""
        eval_records = []
        for sig in test_signals:
            framed, features, seg_res, metrics = self.evaluate_signal(
                sig,
                algorithm=algorithm,
                use_adaptive=use_adaptive,
            )
            eval_records.append((sig, framed, features, seg_res, metrics))
        return eval_records



# Tương thích ngược
VADPipeline = SpeechSegmenter
