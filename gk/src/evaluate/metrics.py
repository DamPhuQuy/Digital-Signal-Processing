"""Evaluation Metrics Module for VAD.

Cung cấp các hàm đo lường và tính toán định lượng:
1. Sai số biên thời gian: MAE và RMSE (đơn vị: ms) giữa biên dự đoán và biên Ground Truth.
2. Các chỉ số phân loại mức khung: Accuracy, FER (Frame Error Rate), Precision, Recall, F1-Score.
3. Hàm tổng hợp đánh giá một mẫu âm thanh: evaluate_vad_result -> EvaluationMetrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.models import (
    AudioSample,
    AudioSignal,
    EvaluationMetrics,
    SegmentationResult,
    VADResult,
)


def match_boundaries(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
    max_tolerance_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """Ghép cặp các mốc biên tìm được với các mốc biên chuẩn Ground Truth gần nhất.

    Tham số:
        pred_boundaries (List[float]): Danh sách mốc biên do thuật toán tìm được (giây).
        gt_boundaries (List[float]): Danh sách mốc biên Ground Truth từ file .lab (giây).
        max_tolerance_sec (float): Dung sai tối đa để coi 2 biên là cùng một sự kiện.

    Trả về:
        List[Tuple[float, float]]: Danh sách các cặp (t_pred, t_gt) ghép được.
    """
    if not gt_boundaries or not pred_boundaries:
        return []

    # Nếu số lượng biên bằng nhau (lý tưởng), ghép theo thứ tự thời gian
    if len(pred_boundaries) == len(gt_boundaries):
        return list(zip(pred_boundaries, gt_boundaries))

    # Nếu số lượng biên khác nhau, ghép cặp với biên GT gần nhất trong dải tolerance
    matched_pairs: List[Tuple[float, float]] = []
    used_pred_indices = set()

    for gt_t in gt_boundaries:
        best_diff = float("inf")
        best_idx = -1
        for idx, pred_t in enumerate(pred_boundaries):
            if idx in used_pred_indices:
                continue
            diff = abs(pred_t - gt_t)
            if diff < best_diff and diff <= max_tolerance_sec:
                best_diff = diff
                best_idx = idx

        if best_idx != -1:
            matched_pairs.append((pred_boundaries[best_idx], gt_t))
            used_pred_indices.add(best_idx)

    return matched_pairs


def compute_boundary_errors(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Tính sai số định lượng MAE và RMSE giữa biên dự đoán và biên chuẩn (đơn vị: ms).

    Công thức:
        MAE = 1/M * sum(|t_pred - t_gt|) * 1000  (ms)
        RMSE = sqrt(1/M * sum((t_pred - t_gt)^2)) * 1000  (ms)

    Trả về:
        Tuple[float, float, List[Tuple[float, float]]]: (MAE ms, RMSE ms, matched_pairs).
    """
    pairs = match_boundaries(pred_boundaries, gt_boundaries)
    if not pairs:
        return float("nan"), float("nan"), []

    diffs_ms = np.array([abs(p - g) * 1000.0 for p, g in pairs], dtype=np.float64)
    mae = float(np.mean(diffs_ms))
    rmse = float(np.sqrt(np.mean(diffs_ms ** 2)))

    return mae, rmse, pairs


def compute_frame_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số đánh giá phân loại ở mức khung (Frame-level)."""
    if len(y_pred) != len(y_true) or len(y_true) == 0:
        return {"accuracy": 0.0, "fer": 1.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_p = y_pred.astype(int)
    y_t = y_true.astype(int)

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    tn = int(np.sum((y_p == 0) & (y_t == 0)))

    total = len(y_t)
    accuracy = (tp + tn) / total
    fer = (fp + fn) / total

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "fer": float(fer),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def evaluate_segmentation(
    signal: AudioSignal,
    result: SegmentationResult,
    timestamps: np.ndarray,
) -> EvaluationMetrics:
    """Đánh giá toàn diện kết quả phân đoạn của một tín hiệu âm thanh."""
    # 1. Tính sai số biên (MAE & RMSE ms)
    mae_ms, rmse_ms, matched_pairs = compute_boundary_errors(
        result.predicted_boundaries,
        signal.gt_boundaries,
    )

    # 2. Tạo nhãn nhị phân chuẩn mức khung
    y_true = np.zeros(len(timestamps), dtype=int)
    for start_t, end_t, label in signal.gt_segments:
        if label == "speech":
            y_true[(timestamps >= start_t) & (timestamps <= end_t)] = 1

    # 3. Tính chỉ số phân loại mức khung
    f_metrics = compute_frame_metrics(result.filtered_decisions, y_true)

    return EvaluationMetrics(
        signal_name=signal.name,
        environment=signal.environment_name,
        algorithm=result.algorithm,
        threshold=result.threshold_used,
        mae_ms=mae_ms,
        rmse_ms=rmse_ms,
        accuracy=f_metrics["accuracy"],
        fer=f_metrics["fer"],
        precision=f_metrics["precision"],
        recall=f_metrics["recall"],
        f1=f_metrics["f1"],
        predicted_boundaries=result.predicted_boundaries,
        gt_boundaries=signal.gt_boundaries,
    )


# Bí danh tương thích ngược
evaluate_vad_result = evaluate_segmentation
