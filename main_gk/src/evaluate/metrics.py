import numpy as np
from src.models import AudioSignal, EvaluationMetrics, SegmentationResult


def match_boundaries(
    predicted_boundaries: list[float],
    gt_boundaries: list[float],
) -> list[tuple[float, float]]:
    pred_sorted: list[float] = sorted(predicted_boundaries)
    gt_sorted: list[float] = sorted(gt_boundaries)

    pairs: list[tuple[float, float]] = []
    num_pred: int = len(pred_sorted)
    num_gt: int = len(gt_sorted)

    if num_pred == 0 or num_gt == 0:
        return pairs

    if num_pred == num_gt:
        # Trường hợp số biên dự đoán và biên chuẩn bằng nhau: ghép 1-1 tuần tự
        for i in range(num_gt):
            pairs.append((pred_sorted[i], gt_sorted[i]))
    else:
        # Trường hợp số lượng biên lệch nhau: với mỗi biên chuẩn, tìm biên dự đoán gần nhất
        for i in range(num_gt):
            target_gt: float = gt_sorted[i]
            best_pred: float = pred_sorted[0]
            min_dist: float = abs(best_pred - target_gt)

            for j in range(1, num_pred):
                dist: float = abs(pred_sorted[j] - target_gt)
                if dist < min_dist:
                    min_dist = dist
                    best_pred = pred_sorted[j]

            pairs.append((best_pred, target_gt))

    return pairs


def compute_mae(
    predicted_boundaries: list[float],
    gt_boundaries: list[float],
) -> float:
    if len(predicted_boundaries) == 0 or len(gt_boundaries) == 0:
        return 0.0

    pairs: list[tuple[float, float]] = match_boundaries(predicted_boundaries, gt_boundaries)
    num_pairs: int = len(pairs)
    if num_pairs == 0:
        return 0.0

    total_abs_error_sec: float = 0.0
    for i in range(num_pairs):
        pred_t: float = pairs[i][0]
        gt_t: float = pairs[i][1]
        abs_diff: float = abs(pred_t - gt_t)
        total_abs_error_sec = total_abs_error_sec + abs_diff

    mae_sec: float = total_abs_error_sec / num_pairs
    mae_ms: float = mae_sec * 1000.0
    return float(mae_ms)


def compute_rmse(
    predicted_boundaries: list[float],
    gt_boundaries: list[float],
) -> float:
    if len(predicted_boundaries) == 0 or len(gt_boundaries) == 0:
        return 0.0

    pairs: list[tuple[float, float]] = match_boundaries(predicted_boundaries, gt_boundaries)
    num_pairs: int = len(pairs)
    if num_pairs == 0:
        return 0.0

    total_sq_error_sec: float = 0.0
    for i in range(num_pairs):
        pred_t: float = pairs[i][0]
        gt_t: float = pairs[i][1]
        diff: float = pred_t - gt_t
        total_sq_error_sec = total_sq_error_sec + (diff * diff)

    mean_sq_error_sec: float = total_sq_error_sec / num_pairs
    rmse_sec: float = float(np.sqrt(mean_sq_error_sec))
    rmse_ms: float = rmse_sec * 1000.0
    return float(rmse_ms)


def estimate_snr_db(signal: AudioSignal, eps: float = 1e-10) -> float:
    x = signal.signal.astype(np.float64)
    sr = signal.fs
    total_len = len(x)
    is_speech = np.zeros(total_len, dtype=bool)
    for s, e in signal.gt_segments:
        i0 = max(0, int(round(s * sr)))
        i1 = min(total_len, int(round(e * sr)))
        is_speech[i0:i1] = True

    speech_samples = x[is_speech]
    silence_samples = x[~is_speech]

    if len(speech_samples) == 0 or len(silence_samples) == 0:
        return 0.0
    sp_power = np.mean(speech_samples ** 2)
    sil_power = np.mean(silence_samples ** 2) + eps
    return float(10.0 * np.log10(sp_power / sil_power))


def evaluate_signal(
    signal: AudioSignal,
    predicted_boundaries: list[float],
    algorithm: str = "binary",
    threshold: float = 0.0,
) -> EvaluationMetrics:
    mae_ms: float = compute_mae(predicted_boundaries, signal.gt_boundaries)
    rmse_ms: float = compute_rmse(predicted_boundaries, signal.gt_boundaries)
    snr_db: float = estimate_snr_db(signal)

    if signal.is_phone:
        environment: str = "Phone"
    else:
        environment = "Studio"

    return EvaluationMetrics(
        signal_name=signal.name,
        environment=environment,
        algorithm=algorithm,
        threshold=threshold,
        mae_ms=mae_ms,
        rmse_ms=rmse_ms,
        predicted_boundaries=predicted_boundaries,
        gt_boundaries=signal.gt_boundaries,
        snr_db=snr_db,
    )


def evaluate_segmentation_result(
    result: SegmentationResult,
    signal: AudioSignal,
) -> EvaluationMetrics:
    return evaluate_signal(
        signal=signal,
        predicted_boundaries=result.predicted_boundaries,
        algorithm=result.algorithm,
        threshold=result.threshold_used,
    )


def summarize_metrics(
    metrics_list: list[EvaluationMetrics],
) -> dict[str, dict[str, float]]:
    phone_maes: list[float] = []
    phone_rmses: list[float] = []
    studio_maes: list[float] = []
    studio_rmses: list[float] = []
    all_maes: list[float] = []
    all_rmses: list[float] = []

    for m in metrics_list:
        all_maes.append(m.mae_ms)
        all_rmses.append(m.rmse_ms)

        if m.environment == "Phone":
            phone_maes.append(m.mae_ms)
            phone_rmses.append(m.rmse_ms)
        else:
            studio_maes.append(m.mae_ms)
            studio_rmses.append(m.rmse_ms)

    def _calc_average(values: list[float]) -> float:
        if len(values) == 0:
            return 0.0
        total: float = 0.0
        for val in values:
            total = total + val
        return float(total / len(values))

    summary: dict[str, dict[str, float]] = {
        "Phone": {
            "mean_mae_ms": _calc_average(phone_maes),
            "mean_rmse_ms": _calc_average(phone_rmses),
            "count": float(len(phone_maes)),
        },
        "Studio": {
            "mean_mae_ms": _calc_average(studio_maes),
            "mean_rmse_ms": _calc_average(studio_rmses),
            "count": float(len(studio_maes)),
        },
        "Overall": {
            "mean_mae_ms": _calc_average(all_maes),
            "mean_rmse_ms": _calc_average(all_rmses),
            "count": float(len(all_maes)),
        },
    }

    return summary


def print_evaluation_table(metrics_list: list[EvaluationMetrics]) -> None:
    separator: str = "=" * 99
    print(separator)
    header: str = (
        f"{'Tên File':<14} | {'Môi trường':<10} | {'SNR (dB)':<8} | {'Thuật toán':<10} | "
        f"{'Ngưỡng':<8} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'Số biên (Pred/GT)'}"
    )
    print(header)
    print("-" * 99)

    for m in metrics_list:
        boundary_info: str = f"{len(m.predicted_boundaries)}/{len(m.gt_boundaries)}"
        line: str = (
            f"{m.signal_name:<14} | "
            f"{m.environment:<10} | "
            f"{m.snr_db:<8.1f} | "
            f"{m.algorithm:<10} | "
            f"{m.threshold:<8.4f} | "
            f"{m.mae_ms:<10.2f} | "
            f"{m.rmse_ms:<10.2f} | "
            f"{boundary_info}"
        )
        print(line)

    print(separator)

    summary: dict[str, dict[str, float]] = summarize_metrics(metrics_list)
    print("\n=== TỔNG HỢP SAI SỐ TRUNG BÌNH ===")
    for env in ["Phone", "Studio", "Overall"]:
        info = summary[env]
        count_int = int(info["count"])
        print(
            f"• {env:<7}: MAE TB = {info['mean_mae_ms']:6.2f} ms | "
            f"RMSE TB = {info['mean_rmse_ms']:6.2f} ms (Số file: {count_int})"
        )
