from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class AudioSignal:
    name: str
    wav_path: Path
    fs: int
    signal: np.ndarray
    duration_sec: float
    is_phone: bool
    gt_segments: list[tuple[float, float]]
    gt_boundaries: list[float]


@dataclass
class FrameSignal:
    frames: np.ndarray
    timestamps: np.ndarray
    frame_len: int
    frame_shift: int
    fs: int
    raw_frames: np.ndarray


@dataclass
class ShortTimeFeatures:
    ste_raw: np.ndarray
    ste_norm: np.ndarray  # preprocessed min-max
    log_ste: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Threshold:
    binary_threshold: float
    binary_threshold_phone: float
    binary_threshold_studio: float

    histogram_threshold: float
    histogram_threshold_phone: float
    histogram_threshold_studio: float

    gaussian_threshold: float
    gaussian_threshold_phone: float
    gaussian_threshold_studio: float

    mean_speech: float = 0.0
    std_speech: float = 0.0
    mean_silence: float = 0.0
    std_silence: float = 0.0

    def get_threshold(
        self, algorithm: str, is_phone: bool, use_common: bool = False
    ) -> float:
        if use_common:
            if algorithm == "binary":
                return self.binary_threshold
            elif algorithm == "histogram":
                return self.histogram_threshold
            elif algorithm == "gaussian":
                return self.gaussian_threshold
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        if algorithm == "binary":
            return (
                self.binary_threshold_phone
                if is_phone
                else self.binary_threshold_studio
            )
        elif algorithm == "histogram":
            return (
                self.histogram_threshold_phone
                if is_phone
                else self.histogram_threshold_studio
            )
        elif algorithm == "gaussian":
            return (
                self.gaussian_threshold_phone
                if is_phone
                else self.gaussian_threshold_studio
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


@dataclass()
class SegmentationResult:
    signal_name: str
    algorithm: str
    threshold_used: float

    timestamps: np.ndarray
    raw_decisions: np.ndarray
    filtered_decisions: np.ndarray

    predicted_boundaries: list[float]
    segments: list[tuple[float, float, str]]  # [(start, end, "speech")]


@dataclass
class EvaluationMetrics:
    signal_name: str
    environment: str
    algorithm: str
    threshold: float
    mae_ms: float
    rmse_ms: float
    predicted_boundaries: list[float]
    gt_boundaries: list[float]
    snr_db: float = 0.0
