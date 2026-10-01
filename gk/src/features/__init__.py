"""Package src.features: Phân khung, áp dụng cửa sổ và trích xuất đặc trưng năng lượng."""

from src.features.framing import (
    apply_framing,
    frame_signal,
    get_window,
)
from src.features.energy import (
    compute_ma,
    compute_magnitude_average,
    compute_short_time_energy,
    compute_ste,
    compute_zcr,
    compute_zero_crossing_rate,
    extract_acoustic_features,
    extract_features,
    extract_short_time_features,
    normalize_feature,
    normalize_minmax,
)

__all__ = [
    "apply_framing",
    "frame_signal",
    "get_window",
    "compute_short_time_energy",
    "compute_ste",
    "compute_magnitude_average",
    "compute_ma",
    "compute_zcr",
    "compute_zero_crossing_rate",
    "normalize_minmax",
    "normalize_feature",
    "extract_features",
    "extract_short_time_features",
    "extract_acoustic_features",
]
