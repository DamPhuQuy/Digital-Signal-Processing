"""Thuật toán phân đoạn tiếng nói / khoảng lặng dựa trên Histogram (Giannakopoulos 2014).

Cài đặt 100% bằng NumPy và built-in Python theo đúng yêu cầu đề bài.
Không sử dụng bất kỳ hàm xử lý tín hiệu nào từ thư viện bên ngoài.
"""

from typing import Tuple
import numpy as np


def build_histogram(feature_values: np.ndarray, num_bins: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Tạo histogram các giá trị đặc trưng thành các khoảng (bins) đều nhau.

    Tham số:
        feature_values (np.ndarray): Mảng 1 chiều chứa giá trị đặc trưng (ví dụ log-STE).
        num_bins (int): Số lượng khoảng chia (mặc định 40 bins theo nghiên cứu).

    Trả về:
        Tuple[np.ndarray, np.ndarray]:
            - hist: Số lượng mẫu rơi vào từng bin (mảng int64 kích thước num_bins).
            - bin_centers: Tọa độ tâm của từng bin trên miền giá trị đặc trưng.
    """
    # 1. Kiểm tra tính hợp lệ của tham số đầu vào
    values = np.asarray(feature_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("feature_values phải là mảng 1 chiều không rỗng.")
    if num_bins <= 0:
        raise ValueError("num_bins phải là số nguyên dương lớn hơn 0.")

    # 2. Xác định cận min, max của đặc trưng
    xmin = float(np.min(values))
    xmax = float(np.max(values))

    # Xử lý trường hợp đặc biệt: tất cả giá trị bằng nhau
    if xmin == xmax:
        hist = np.zeros(num_bins, dtype=np.int64)
        hist[0] = values.size
        return hist, np.full(num_bins, xmin, dtype=np.float64)

    # 3. Tính độ rộng của mỗi bin và khởi tạo mảng đếm tần suất
    bin_width = (xmax - xmin) / num_bins
    hist = np.zeros(num_bins, dtype=np.int64)

    # 4. Gom các mẫu vào từng bin tương ứng
    for val in values:
        if val == xmax:
            idx = num_bins - 1
        else:
            idx = int(np.floor((val - xmin) / bin_width))
        idx = max(0, min(idx, num_bins - 1))
        hist[idx] += 1

    # 5. Tính tọa độ tâm của từng bin
    bin_centers = xmin + (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_width
    return hist, bin_centers


def smooth_histogram(histogram: np.ndarray, window_size: int = 1) -> np.ndarray:
    """Làm mượt histogram bằng bộ lọc trung bình trượt (Moving Average).

    Tham số:
        histogram (np.ndarray): Mảng số lượng phần tử của các bin.
        window_size (int): Kích thước cửa sổ trượt (số nguyên dương lẻ, 1 = không làm mượt).

    Trả về:
        np.ndarray: Mảng histogram sau khi làm mượt.
    """
    # 1. Kiểm tra tham số cửa sổ trượt
    counts = np.asarray(histogram, dtype=np.float64)
    if window_size <= 1:
        return counts.copy()
    if window_size % 2 == 0:
        raise ValueError("window_size phải là số lẻ dương.")

    # 2. Áp dụng trung bình trượt có xử lý biên không cần padding
    radius = window_size // 2
    n = len(counts)
    smoothed = np.empty(n, dtype=np.float64)

    for i in range(n):
        start_idx = max(0, i - radius)
        end_idx = min(n, i + radius + 1)
        smoothed[i] = np.mean(counts[start_idx:end_idx])

    return smoothed


def find_local_maxima(histogram: np.ndarray) -> np.ndarray:
    """Tìm tất cả các chỉ số bin là cực đại địa phương (lớn hơn 2 lân cận kề bên).

    Tham số:
        histogram (np.ndarray): Mảng tần suất histogram 1 chiều.

    Trả về:
        np.ndarray: Mảng chứa các chỉ số bin đạt cực đại địa phương.
    """
    # 1. Kiểm tra kích thước dữ liệu
    counts = np.asarray(histogram, dtype=np.float64)
    if len(counts) < 3:
        return np.array([], dtype=np.int64)

    # 2. Duyệt qua các điểm nội bộ để tìm điểm lớn hơn 2 điểm xung quanh
    peaks = []
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
            peaks.append(i)

    return np.asarray(peaks, dtype=np.int64)


def select_two_peaks(
    histogram: np.ndarray,
    bin_centers: np.ndarray,
    peak_indices: np.ndarray,
    min_peak_distance: int = 2,
    min_relative_height: float = 0.1,
) -> Tuple[float, float]:
    """Chọn ra 2 đỉnh đặc trưng M1 (Silence) và M2 (Speech) tối ưu nhất từ các cực đại.

    Tham số:
        histogram (np.ndarray): Mảng tần suất histogram.
        bin_centers (np.ndarray): Tọa độ tâm các bin.
        peak_indices (np.ndarray): Các chỉ số bin cực đại tìm được.
        min_peak_distance (int): Khoảng cách tối thiểu giữa 2 đỉnh (đơn vị bin).
        min_relative_height (float): Ngưỡng chiều cao tương đối so với đỉnh cao nhất.

    Trả về:
        Tuple[float, float]: Giá trị đặc trưng của (M1, M2) với M1 < M2.
    """
    counts = np.asarray(histogram, dtype=np.float64)
    centers = np.asarray(bin_centers, dtype=np.float64)
    indices = [int(p) for p in peak_indices]

    # 1. Trường hợp có ít hơn 2 đỉnh: fallback chọn 2 vị trí phân tán theo bách phân vị
    if len(indices) < 2:
        idx1 = len(counts) // 4
        idx2 = (len(counts) * 3) // 4
        return float(centers[idx1]), float(centers[idx2])

    # 2. Lọc các đỉnh đủ cao theo tỷ lệ so với đỉnh cao nhất
    max_height = float(max(counts[p] for p in indices))
    valid_peaks = [p for p in indices if counts[p] >= min_relative_height * max_height]

    if len(valid_peaks) < 2:
        valid_peaks = sorted(indices, key=lambda p: counts[p], reverse=True)[:2]
        valid_peaks.sort()

    # 3. Tìm cặp đỉnh (left, right) thỏa mãn khoảng cách tối thiểu có tổng độ cao lớn nhất
    best_pair = None
    best_score = -1.0

    for i in range(len(valid_peaks)):
        for j in range(i + 1, len(valid_peaks)):
            left = valid_peaks[i]
            right = valid_peaks[j]
            dist = right - left
            if dist >= min_peak_distance:
                score = (counts[left] + counts[right]) + 0.01 * dist
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)

    # 4. Nếu không có cặp nào cách nhau >= min_peak_distance, lấy 2 đỉnh cao nhất
    if best_pair is None:
        sorted_by_height = sorted(valid_peaks, key=lambda p: counts[p], reverse=True)[:2]
        sorted_by_height.sort()
        best_pair = (sorted_by_height[0], sorted_by_height[1])

    m1_val = float(centers[best_pair[0]])
    m2_val = float(centers[best_pair[1]])
    return min(m1_val, m2_val), max(m1_val, m2_val)


def calculate_threshold(m1: float, m2: float, weight: float = 2.0) -> float:
    """Tính toán ngưỡng phân đoạn T theo công thức Giannakopoulos (2014).

        T = (W * M1 + M2) / (W + 1)

    Tham số:
        m1 (float): Vị trí đỉnh khoảng lặng (Silence mode).
        m2 (float): Vị trí đỉnh tiếng nói (Speech mode).
        weight (float): Trọng số ưu tiên (mặc định W = 2.0).

    Trả về:
        float: Giá trị ngưỡng phân tách tối ưu T.
    """
    if weight <= 0:
        raise ValueError("Trọng số weight phải lớn hơn 0.")
    if m1 >= m2:
        # Đảm bảo m1 < m2
        m1, m2 = min(m1, m2), max(m1, m2)

    return float((weight * m1 + m2) / (weight + 1.0))


def find_threshold_histogram(
    feature_values: np.ndarray,
    num_bins: int = 40,
    smooth_window: int = 1,
    weight: float = 2.0,
    min_peak_distance: int = 2,
    min_relative_height: float = 0.1,
) -> float:
    """Hàm giao diện chính: Tính toán ngưỡng Histogram cho một mảng giá trị đặc trưng.

    Tham số:
        feature_values (np.ndarray): Mảng 1 chiều chứa giá trị đặc trưng (khuyến nghị logSTE).
        num_bins (int): Số lượng bin (mặc định 40).
        smooth_window (int): Cửa sổ làm mượt (mặc định 1 - không mượt).
        weight (float): Trọng số W (mặc định 2.0).
        min_peak_distance (int): Khoảng cách tối thiểu giữa 2 đỉnh.
        min_relative_height (float): Tỷ lệ chiều cao tối thiểu của đỉnh.

    Trả về:
        float: Giá trị ngưỡng T tối ưu.
    """
    if len(feature_values) == 0:
        return 0.0

    # 1. Tạo histogram
    hist, centers = build_histogram(feature_values, num_bins=num_bins)

    # 2. Làm mượt histogram
    smoothed = smooth_histogram(hist, window_size=smooth_window)

    # 3. Tìm các cực đại địa phương
    peaks = find_local_maxima(smoothed)

    # 4. Chọn 2 đỉnh M1 (silence) và M2 (speech)
    m1, m2 = select_two_peaks(
        smoothed,
        centers,
        peaks,
        min_peak_distance=min_peak_distance,
        min_relative_height=min_relative_height,
    )

    # 5. Tính ngưỡng phân đoạn T theo công thức Giannakopoulos
    threshold = calculate_threshold(m1, m2, weight=weight)
    return threshold
