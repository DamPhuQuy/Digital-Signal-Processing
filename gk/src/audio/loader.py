"""Package src.audio: Xử lý nạp dữ liệu âm thanh và phân tích nhãn Ground Truth.

Cung cấp các tiện ích:
- Đọc file WAV và chuẩn hóa biên độ [-1.0, 1.0].
- Nạp tập dữ liệu huấn luyện và kiểm thử (AudioSignal).
- Phân tích file nhãn .lab (gộp voiced và unvoiced thành speech).
- Trích xuất các mốc biên ranh giới chuẩn.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
from scipy.io import wavfile

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

from src.models import AudioSample, AudioSignal


def parse_lab(lab_path: Path | str) -> List[Tuple[float, float, str]]:
    """Đọc trực tiếp file nhãn .lab và trả về danh sách các đoạn (start, end, label) gốc."""
    path = Path(lab_path)
    if not path.is_file():
        return []

    segments: List[Tuple[float, float, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    start_t = float(parts[0])
                    end_t = float(parts[1])
                    label_str = parts[2].lower()
                except ValueError:
                    continue
                segments.append((start_t, end_t, label_str))
    return segments


def parse_lab_file(lab_path: Path | str) -> Tuple[List[Tuple[float, float, str]], List[float]]:
    """Đọc file nhãn .lab và trích xuất danh sách đoạn cùng các mốc thời gian biên."""
    raw_segments = parse_lab(lab_path)
    if not raw_segments:
        return [], []

    unified_segments = []
    for start_t, end_t, label_str in raw_segments:
        unified_label = "silence" if label_str == "sil" else "speech"
        unified_segments.append((start_t, end_t, unified_label))

    # Hợp nhất các đoạn liên tiếp có cùng nhãn sau khi gộp v và uv
    merged_segments: List[Tuple[float, float, str]] = []
    for seg in unified_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev_start, prev_end, prev_label = merged_segments[-1]
            curr_start, curr_end, curr_label = seg
            if prev_label == curr_label and np.isclose(prev_end, curr_start, atol=1e-3):
                merged_segments[-1] = (prev_start, curr_end, prev_label)
            else:
                merged_segments.append(seg)

    boundaries = [float(round(seg[0], 2)) for seg in merged_segments[1:]]
    return merged_segments, boundaries


def get_ground_truth_segments(lab_path: Path | str) -> Tuple[List[Tuple[float, float, str]], List[float]]:
    """Lấy danh sách đoạn chuẩn và các mốc biên thời gian từ file .lab."""
    return parse_lab_file(lab_path)


def load_wav(file_path: Path | str, normalize: bool = True) -> Tuple[int, np.ndarray]:
    """Đọc file WAV và trả về (fs, signal_array)."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file âm thanh: {path}")

    fs, data = wavfile.read(str(path))

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    signal = data.astype(np.float64)

    if normalize:
        max_abs = float(np.max(np.abs(signal)))
        if max_abs > 0:
            signal = signal / max_abs

    return int(fs), signal


def load_audio_signal(wav_path: Path | str, lab_path: Path | str | None = None) -> AudioSignal:
    """Nạp file âm thanh và file nhãn .lab tương ứng thành thực thể AudioSignal."""
    w_path = Path(wav_path)
    l_path = Path(lab_path) if lab_path else w_path.with_suffix(".lab")

    fs, signal = load_wav(w_path, normalize=True)
    sample_name = w_path.stem
    is_phone = "phone" in sample_name.lower()
    segments, boundaries = parse_lab_file(l_path)

    return AudioSignal(
        name=sample_name,
        wav_path=w_path,
        fs=fs,
        signal=signal,
        duration_sec=float(len(signal) / fs),
        is_phone=is_phone,
        gt_segments=segments,
        gt_boundaries=boundaries,
        lab_path=l_path if l_path.is_file() else None,
    )


load_audio_sample = load_audio_signal
load_audio = load_audio_signal


def load_dataset(dir_path: Path | str) -> List[AudioSignal]:
    """Nạp toàn bộ các file .wav và .lab trong thư mục thành danh sách AudioSignal."""
    p = Path(dir_path)
    if not p.is_dir():
        raise NotADirectoryError(f"Thư mục không tồn tại: {p}")

    wav_files = sorted(p.glob("*.wav"))
    signals: List[AudioSignal] = []
    for w in wav_files:
        signals.append(load_audio_signal(w))
    return signals


load_audio_dataset = load_dataset
load_dataset_samples = load_dataset


def create_frame_labels(signal: AudioSignal, timestamps: np.ndarray) -> np.ndarray:
    """Gán nhãn chuẩn Ground Truth (1: speech, 0: silence) cho từng khung thời gian."""
    labels = np.zeros(len(timestamps), dtype=int)
    for start_t, end_t, seg_label in signal.ground_truth_segments:
        if seg_label == "speech":
            mask = (timestamps >= start_t) & (timestamps <= end_t)
            labels[mask] = 1
    return labels
