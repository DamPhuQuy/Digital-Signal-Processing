import numpy as np
from src.audio.labels import create_frame_labels
from src.features.extraction import extract_features
from src.features.framing import apply_framing
from src.models import AudioSignal, Threshold
from src.segmentation.binary import find_threshold_binary
from src.segmentation.gauss import find_threshold_gaussian
from src.segmentation.histogram import find_threshold_histogram


def _get_signals_data(signals: list[AudioSignal]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trích xuất ste_norm, log_ste và ground truth nhãn khung từ danh sách tín hiệu."""
    ste_norm_list = []
    log_ste_list = []
    lbls_list = []

    for sig in signals:
        framed = apply_framing(sig)
        feats = extract_features(framed)
        labels = create_frame_labels(framed.timestamps, sig.gt_segments)
        ste_norm_list.append(feats.ste_norm)
        log_ste_list.append(feats.log_ste)
        lbls_list.append(labels)

    if len(ste_norm_list) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.concatenate(ste_norm_list), np.concatenate(log_ste_list), np.concatenate(lbls_list)


def calibrate_thresholds(train_signals: list[AudioSignal]) -> Threshold:
    """Hiệu chuẩn các bộ ngưỡng (Binary, Histogram, Gaussian) từ tập tín hiệu huấn luyện."""
    # 1. Trích xuất đặc trưng toàn cục từ tất cả các file âm thanh huấn luyện
    ste_all, log_ste_all, lbls_all = _get_signals_data(train_signals)

    bin_global = find_threshold_binary(ste_all, lbls_all)
    hist_global = find_threshold_histogram(log_ste_all)
    gauss_global, m_sp, s_sp, m_sil, s_sil = find_threshold_gaussian(ste_all, lbls_all)

    # 2. Phân loại danh sách tín hiệu huấn luyện theo môi trường Phone và Studio
    phone_signals = []
    studio_signals = []
    for sig in train_signals:
        if sig.is_phone:
            phone_signals.append(sig)
        else:
            studio_signals.append(sig)

    # 3. Tính toán ngưỡng riêng cho môi trường Phone
    if len(phone_signals) > 0:
        ste_phone, log_ste_phone, lbls_phone = _get_signals_data(phone_signals)
        bin_phone = find_threshold_binary(ste_phone, lbls_phone)
        hist_phone = find_threshold_histogram(log_ste_phone)
        gauss_phone, _, _, _, _ = find_threshold_gaussian(ste_phone, lbls_phone)
    else:
        bin_phone = bin_global
        hist_phone = hist_global
        gauss_phone = gauss_global

    # 4. Tính toán ngưỡng riêng cho môi trường Studio
    if len(studio_signals) > 0:
        ste_studio, log_ste_studio, lbls_studio = _get_signals_data(studio_signals)
        bin_studio = find_threshold_binary(ste_studio, lbls_studio)
        hist_studio = find_threshold_histogram(log_ste_studio)
        gauss_studio, _, _, _, _ = find_threshold_gaussian(ste_studio, lbls_studio)
    else:
        bin_studio = bin_global
        hist_studio = hist_global
        gauss_studio = gauss_global

    return Threshold(
        binary_threshold=bin_global,
        binary_threshold_phone=bin_phone,
        binary_threshold_studio=bin_studio,
        histogram_threshold=hist_global,
        histogram_threshold_phone=hist_phone,
        histogram_threshold_studio=hist_studio,
        gaussian_threshold=gauss_global,
        gaussian_threshold_phone=gauss_phone,
        gaussian_threshold_studio=gauss_studio,
        mean_speech=m_sp,
        std_speech=s_sp,
        mean_silence=m_sil,
        std_silence=s_sil,
    )
