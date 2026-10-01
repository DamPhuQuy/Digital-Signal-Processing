"""Package src.vad: Thuật toán tìm ngưỡng VAD và phân đoạn tiếng nói (Voice Activity Detection)."""

from src.segmentation.thresholds import (
    calculate_thresholds,
    calibrate_all_thresholds,
    estimate_threshold_binary_search,
    estimate_threshold_gaussian,
    estimate_threshold_histogram,
    find_all_thresholds,
    find_threshold_binary_search,
    find_threshold_gaussian,
    find_threshold_histogram,
)
from src.segmentation.segmenter import (
    apply_threshold,
    extract_boundaries,
    filter_short_silences,
    segment_audio,
    segment_sample,
    segment_signal,
)

__all__ = [
    "find_threshold_binary_search",
    "estimate_threshold_binary_search",
    "find_threshold_histogram",
    "estimate_threshold_histogram",
    "find_threshold_gaussian",
    "estimate_threshold_gaussian",
    "find_all_thresholds",
    "calculate_thresholds",
    "calibrate_all_thresholds",
    "apply_threshold",
    "filter_short_silences",
    "extract_boundaries",
    "segment_signal",
    "segment_audio",
    "segment_sample",
]
