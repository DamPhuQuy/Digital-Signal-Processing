"""Module segmenter.py: Áp dụng ngưỡng, lọc khoảng lặng < 200ms và trích xuất biên (Khối 4).

Thuộc Khối 4 trong quy trình xử lý tín hiệu:
- Áp dụng ngưỡng phân loại nhị phân sơ bộ cho từng khung.
- Hậu xử lý (Post-processing): Lọc và gộp khoảng lặng < 200 ms (quy tắc bắt buộc của Giảng viên).
- Trích xuất các mốc biên thời gian chuyển tiếp (State Transitions).
- Đóng gói toàn bộ kết quả phân đoạn thành SegmentationResult.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from src.models import (
    AcousticFeatures,
    AudioSample,
    AudioSignal,
    FramedSignal,
    SegmentationResult,
    ShortTimeFeatures,
    VADResult,
)


def apply_threshold(feature: np.ndarray, threshold: float) -> np.ndarray:
    """Phân loại nhị phân sơ bộ từng khung thành tiếng nói (1) hoặc khoảng lặng (0)."""
    return (feature >= threshold).astype(int)


def filter_short_silences(
    decisions: np.ndarray,
    timestamps: np.ndarray,
    min_silence_ms: float = 200.0,
    min_silence_duration_ms: Optional[float] = None,
) -> np.ndarray:
    """Quy tắc nghiệp vụ bắt buộc: Loại bỏ các khoảng lặng giả định có độ dài < 200 ms.

    Nếu một đoạn khoảng lặng (0) xen giữa 2 đoạn tiếng nói (1) có thời lượng ngắn hơn
    ngưỡng tối thiểu 200 ms, đoạn đó sẽ được chuyển thành tiếng nói (1) để tránh xé vụn âm tiết.
    """
    if min_silence_duration_ms is not None:
        min_silence_ms = min_silence_duration_ms

    filtered = decisions.copy()
    num_frames = len(filtered)
    if num_frames == 0:
        return filtered

    in_silence = False
    sil_start_idx = 0

    for i in range(num_frames):
        if filtered[i] == 0:
            if not in_silence:
                in_silence = True
                sil_start_idx = i
        else:
            if in_silence:
                in_silence = False
                sil_end_idx = i - 1

                # Khoảng lặng ở đầu hoặc cuối tín hiệu không bị gộp vào tiếng nói
                if sil_start_idx == 0:
                    continue

                duration_sec = timestamps[sil_end_idx] - timestamps[sil_start_idx]
                if duration_sec < (min_silence_ms * 1e-3):
                    filtered[sil_start_idx : sil_end_idx + 1] = 1

    return filtered


def extract_boundaries(decisions: np.ndarray, timestamps: np.ndarray) -> List[float]:
    """Trích xuất danh sách các mốc thời gian biên (chuyển tiếp giữa speech và silence)."""
    boundaries: List[float] = []
    if len(decisions) < 2:
        return boundaries

    for i in range(1, len(decisions)):
        if decisions[i] != decisions[i - 1]:
            b_time = float(round((timestamps[i] + timestamps[i - 1]) / 2.0, 2))
            boundaries.append(b_time)

    return boundaries


def segment_signal(
    signal: AudioSignal,
    features: ShortTimeFeatures,
    framed: FramedSignal,
    threshold: float,
    algorithm: str = "binary",
    min_silence_ms: float = 200.0,
) -> SegmentationResult:
    """Thực hiện toàn bộ quy trình phân đoạn: cắt ngưỡng, lọc < 200ms, trích xuất biên."""
    raw_decisions = apply_threshold(features.ste_norm, threshold)
    filtered_decisions = filter_short_silences(raw_decisions, framed.timestamps, min_silence_ms=min_silence_ms)
    predicted_boundaries = extract_boundaries(filtered_decisions, framed.timestamps)

    # Xây dựng danh sách các đoạn (segments)
    segments = []
    num_frames = len(filtered_decisions)
    if num_frames > 0:
        curr_label = "speech" if filtered_decisions[0] == 1 else "silence"
        start_t = 0.0

        for i in range(1, num_frames):
            lbl = "speech" if filtered_decisions[i] == 1 else "silence"
            if lbl != curr_label:
                end_t = float(round(framed.timestamps[i], 2))
                segments.append((start_t, end_t, curr_label))
                curr_label = lbl
                start_t = end_t

        end_t = float(round(len(signal.signal) / signal.fs, 2))
        segments.append((start_t, end_t, curr_label))

    return SegmentationResult(
        signal_name=signal.name,
        algorithm=algorithm,
        threshold_used=threshold,
        raw_decisions=raw_decisions,
        filtered_decisions=filtered_decisions,
        predicted_boundaries=predicted_boundaries,
        segments=segments,
        ground_truth_boundaries=signal.ground_truth_boundaries,
        timestamps=framed.timestamps,
    )


segment_audio = segment_signal
segment_sample = segment_signal
