"""Module thresholds.py: Cài đặt 03 thuật toán tìm ngưỡng phân đoạn tối ưu (Khối 4).

Thuộc Khối 4 trong quy trình xử lý tín hiệu:
- Thuật toán 1: Tìm kiếm nhị phân (Binary Search / Ternary Search).
- Thuật toán 2: Phân tích biểu đồ tần suất 2 đỉnh (Histogram Valley Detection).
- Thuật toán 3: Thống kê phân phối chuẩn (Gaussian Statistics & Bayes Equal-std rule).
- Xác định bộ ngưỡng toàn cục (Global) và thích nghi theo môi trường (Phone / Studio).
"""

from __future__ import annotations

from typing import List, Tuple, Union
import numpy as np

from src.models import (
    AcousticFeatures,
    AudioSample,
    AudioSignal,
    ShortTimeFeatures,
    Thresholds,
    VADThresholds,
)


def find_threshold_binary_search(
    features_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    t_min: float = 0.0001,
    t_max: float = 0.05,
    num_steps: int = 25,
) -> float:
    """Thuật toán 1: Tìm ngưỡng tối ưu bằng phương pháp tìm kiếm nhị phân / ternary search."""
    def evaluate_loss(threshold: float) -> float:
        total_error = 0.0
        total_frames = 0
        for feat, lbl in zip(features_list, labels_list):
            pred = (feat >= threshold).astype(int)
            total_error += float(np.sum(pred != lbl))
            total_frames += len(lbl)
        return total_error / max(total_frames, 1)

    low = float(t_min)
    high = float(t_max)
    best_th = (low + high) / 2.0
    best_loss = evaluate_loss(best_th)

    for _ in range(num_steps):
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0

        loss1 = evaluate_loss(mid1)
        loss2 = evaluate_loss(mid2)

        if loss1 < best_loss:
            best_loss = loss1
            best_th = mid1
        if loss2 < best_loss:
            best_loss = loss2
            best_th = mid2

        if loss1 < loss2:
            high = mid2
        else:
            low = mid1

    return float(best_th)


estimate_threshold_binary_search = find_threshold_binary_search


def find_threshold_histogram(
    features_list_or_data: Union[np.ndarray, List[np.ndarray]],
    num_bins: int = 100,
    range_max: float = 0.5,
) -> float:
    """Thuật toán 2: Tìm ngưỡng tối ưu tại đáy thung lũng (Valley) giữa 2 đỉnh Histogram."""
    if isinstance(features_list_or_data, np.ndarray):
        concatenated = features_list_or_data
    else:
        concatenated = np.concatenate(features_list_or_data)

    counts, bin_edges = np.histogram(concatenated, bins=num_bins, range=(0.0, range_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peaks = []
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            peaks.append(i)

    if len(peaks) >= 2:
        p1 = peaks[0]
        p2 = peaks[-1]
        valley_idx = p1 + int(np.argmin(counts[p1:p2+1]))
        return float(bin_centers[valley_idx])
    elif len(peaks) == 1:
        p1 = peaks[0]
        valley_idx = min(len(bin_centers) - 1, p1 + 5)
        return float(bin_centers[valley_idx])
    else:
        return float(np.median(concatenated))


estimate_threshold_histogram = find_threshold_histogram


def find_threshold_gaussian(
    speech_features_or_list: Union[np.ndarray, List[np.ndarray]],
    silence_features_or_labels: Union[np.ndarray, List[np.ndarray]],
) -> Union[float, Tuple[float, float, float, float, float]]:
    """Thuật toán 3: Thống kê tham số Gauss của Speech/Silence và tìm ngưỡng tối ưu Bayes."""
    if isinstance(speech_features_or_list, np.ndarray) and isinstance(silence_features_or_labels, np.ndarray):
        sp_feats = speech_features_or_list
        sil_feats = silence_features_or_labels
        mean_sil = float(np.mean(sil_feats)) if len(sil_feats) > 0 else 0.001
        std_sil = float(np.std(sil_feats)) if len(sil_feats) > 0 else 0.001
        mean_sp = float(np.mean(sp_feats)) if len(sp_feats) > 0 else 0.1
        std_sp = float(np.std(sp_feats)) if len(sp_feats) > 0 else 0.05
        if np.isclose(std_sil + std_sp, 0.0):
            return float((mean_sil + mean_sp) / 2.0)
        return float((mean_sil * std_sp + mean_sp * std_sil) / (std_sil + std_sp))

    all_feats = np.concatenate(speech_features_or_list)
    all_lbls = np.concatenate(silence_features_or_labels)

    sil_feats = all_feats[all_lbls == 0]
    sp_feats = all_feats[all_lbls == 1]

    mean_sil = float(np.mean(sil_feats)) if len(sil_feats) > 0 else 0.001
    std_sil = float(np.std(sil_feats)) if len(sil_feats) > 0 else 0.001

    mean_sp = float(np.mean(sp_feats)) if len(sp_feats) > 0 else 0.1
    std_sp = float(np.std(sp_feats)) if len(sp_feats) > 0 else 0.05

    if np.isclose(std_sil + std_sp, 0.0):
        t_opt = (mean_sil + mean_sp) / 2.0
    else:
        t_opt = (mean_sil * std_sp + mean_sp * std_sil) / (std_sil + std_sp)

    return float(t_opt), mean_sil, std_sil, mean_sp, std_sp


estimate_threshold_gaussian = find_threshold_gaussian


def find_all_thresholds(
    train_signals: List[AudioSignal],
    train_features: List[ShortTimeFeatures],
    train_labels: List[np.ndarray],
) -> Thresholds:
    """Xác định toàn bộ bảng ngưỡng tối ưu cho 3 thuật toán và các môi trường Phone/Studio."""
    ste_list = [f.ste_norm for f in train_features]

    # Ngưỡng toàn cục (Global)
    t_binary_global = find_threshold_binary_search(ste_list, train_labels)
    t_hist_global = find_threshold_histogram(ste_list)
    t_gauss_global, m_sil, s_sil, m_sp, s_sp = find_threshold_gaussian(ste_list, train_labels)

    # Tách theo môi trường Phone vs Studio
    phone_indices = [i for i, s in enumerate(train_signals) if s.is_phone]
    studio_indices = [i for i, s in enumerate(train_signals) if not s.is_phone]

    phone_ste = [ste_list[i] for i in phone_indices]
    phone_labels = [train_labels[i] for i in phone_indices]
    t_binary_phone = find_threshold_binary_search(phone_ste, phone_labels) if phone_ste else t_binary_global

    studio_ste = [ste_list[i] for i in studio_indices]
    studio_labels = [train_labels[i] for i in studio_indices]
    t_binary_studio = find_threshold_binary_search(studio_ste, studio_labels) if studio_ste else t_binary_global

    return Thresholds(
        binary_global=t_binary_global,
        binary_phone=t_binary_phone,
        binary_studio=t_binary_studio,
        histogram_global=t_hist_global,
        gaussian_global=t_gauss_global,
        mean_silence=m_sil,
        std_silence=s_sil,
        mean_speech=m_sp,
        std_speech=s_sp,
    )


calculate_thresholds = find_all_thresholds
calibrate_all_thresholds = find_all_thresholds
