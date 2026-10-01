"""Package src: Hệ thống phân đoạn tiếng nói và khoảng lặng (VAD).

Được tổ chức chuẩn mực theo 5 khối chức năng xử lý tín hiệu DSP:
  • Khối 1: Preprocessing (src.audio, src.features.framing): Nạp WAV & Ground Truth, phân khung 25ms, cửa sổ Hamming.
  • Khối 2: Feature Extraction (src.features.energy): Năng lượng ngắn hạn STE, chuẩn hóa Min-Max [0, 1].
  • Khối 3: Threshold Calibration (src.segmentation.thresholds): Tìm ngưỡng tối ưu T trên TRAIN (Binary Search, Histogram, Gaussian).
  • Khối 4: VAD Inference & Post-processing (src.segmentation.segmenter): Phân đoạn trên TEST, lọc silence < 200ms, trích xuất biên.
  • Khối 5: Evaluation & Visualization (src.evaluate): Đo lường sai số định lượng MAE/RMSE/FER/F1, vẽ đồ thị tổng hợp 2x2.

Điều phối toàn diện:
  src.pipeline.SpeechSegmenter: Bộ phân đoạn kết nối xuyên suốt quy trình 5 khối & 8 bước tuần tự.
"""

from src.models import (
    AcousticFeatures,
    AudioSample,
    AudioSignal,
    EvaluationMetrics,
    FramedSignal,
    SegmentationMetrics,
    SegmentationResult,
    ShortTimeFeatures,
    Thresholds,
    VADResult,
    VADThresholds,
)
from src.pipeline import SpeechSegmenter, VADPipeline

from src.audio import (
    create_frame_labels,
    get_ground_truth_segments,
    load_audio,
    load_audio_dataset,
    load_audio_sample,
    load_audio_signal,
    load_dataset,
    load_dataset_samples,
    load_wav,
    parse_lab,
    parse_lab_file,
)
from src.features import (
    apply_framing,
    compute_ma,
    compute_magnitude_average,
    compute_short_time_energy,
    compute_ste,
    compute_zcr,
    compute_zero_crossing_rate,
    extract_acoustic_features,
    extract_features,
    extract_short_time_features,
    frame_signal,
    get_window,
    normalize_feature,
    normalize_minmax,
)
from src.segmentation import (
    apply_threshold,
    calculate_thresholds,
    calibrate_all_thresholds,
    estimate_threshold_binary_search,
    estimate_threshold_gaussian,
    estimate_threshold_histogram,
    extract_boundaries,
    filter_short_silences,
    find_all_thresholds,
    find_threshold_binary_search,
    find_threshold_gaussian,
    find_threshold_histogram,
    segment_audio,
    segment_sample,
    segment_signal,
)
from src.evaluate import (
    compute_boundary_errors,
    compute_frame_metrics,
    evaluate_segmentation,
    evaluate_vad_result,
    get_screen_dimensions,
    match_boundaries,
    plot_combined_segmentation_figure,
    plot_combined_vad_figure,
    plot_segmentation_figure,
    plot_vad_result,
    plot_vad_sample_figure,
    position_figure_at_corner,
)

# Tương thích ngược với các tên package cũ nếu có
import sys
import src.audio as audio
import src.features as features
import src.segmentation as segmentation
import src.evaluate as evaluate

# Aliases
preprocess = features
train = segmentation
inference = segmentation
vad = segmentation
sys.modules['src.vad'] = segmentation

__all__ = [
    # Data Models
    "AudioSignal",
    "FramedSignal",
    "ShortTimeFeatures",
    "Thresholds",
    "SegmentationResult",
    "EvaluationMetrics",
    "SegmentationMetrics",
    "AudioSample",
    "AcousticFeatures",
    "VADThresholds",
    "VADResult",
    # Segmenter / Pipeline
    "SpeechSegmenter",
    "VADPipeline",
    # Audio
    "load_audio_signal",
    "load_audio",
    "load_audio_sample",
    "load_dataset",
    "load_audio_dataset",
    "load_dataset_samples",
    "parse_lab_file",
    "parse_lab",
    "get_ground_truth_segments",
    "load_wav",
    "create_frame_labels",
    # Features
    "get_window",
    "apply_framing",
    "frame_signal",
    "compute_short_time_energy",
    "compute_magnitude_average",
    "compute_ste",
    "compute_ma",
    "compute_zcr",
    "compute_zero_crossing_rate",
    "normalize_minmax",
    "normalize_feature",
    "extract_features",
    "extract_short_time_features",
    "extract_acoustic_features",
    # VAD
    "find_threshold_binary_search",
    "find_threshold_histogram",
    "find_threshold_gaussian",
    "find_all_thresholds",
    "calculate_thresholds",
    "estimate_threshold_binary_search",
    "estimate_threshold_histogram",
    "estimate_threshold_gaussian",
    "calibrate_all_thresholds",
    "apply_threshold",
    "filter_short_silences",
    "extract_boundaries",
    "segment_signal",
    "segment_audio",
    "segment_sample",
    # Evaluate
    "match_boundaries",
    "compute_boundary_errors",
    "compute_frame_metrics",
    "evaluate_segmentation",
    "evaluate_vad_result",
    "get_screen_dimensions",
    "position_figure_at_corner",
    "plot_segmentation_figure",
    "plot_vad_sample_figure",
    "plot_vad_result",
    "plot_combined_segmentation_figure",
    "plot_combined_vad_figure",
]
