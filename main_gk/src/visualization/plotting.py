from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluate.metrics import evaluate_signal
from src.features.extraction import extract_features
from src.features.framing import apply_framing
from src.models import AudioSignal, EvaluationMetrics, SegmentationResult


def plot_signal_segmentation(
    signal: AudioSignal,
    result: SegmentationResult,
    metrics: EvaluationMetrics | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    if metrics is None:
        metrics = evaluate_signal(
            signal=signal,
            predicted_boundaries=result.predicted_boundaries,
            algorithm=result.algorithm,
            threshold=result.threshold_used,
        )

    # 1. Trích xuất đặc trưng phù hợp theo thuật toán
    framed = apply_framing(signal)
    feats = extract_features(framed)
    if result.algorithm.lower() == "histogram":
        feature_vals = feats.log_ste
        feature_label = "Đặc trưng log(STE)"
        feature_ylabel = "log(STE)"
        y_min = float(np.min(feature_vals)) - 0.5
        y_max = float(np.max(feature_vals)) + 0.5
    else:
        feature_vals = feats.ste_norm
        feature_label = "Năng lượng STE chuẩn hóa"
        feature_ylabel = "STE chuẩn hóa"
        y_min = -0.02
        y_max = 1.05

    # 2. Tạo 2 đồ thị xếp chồng cùng trục thời gian
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 7), sharex=True)
    time_axis = np.linspace(0.0, signal.duration_sec, len(signal.signal))

    # --- ĐỒ THỊ 1: DẠNG SÓNG ÂM THANH & CÁC MỐC BIÊN ---
    ax1.plot(
        time_axis,
        signal.signal,
        color="#3498db",
        alpha=0.7,
        linewidth=0.8,
        label="Dạng sóng âm thanh",
    )

    # Tô màu nền vùng tiếng nói chuẩn Ground Truth (màu xanh lá nhạt)
    for i, (start_t, end_t) in enumerate(signal.gt_segments):
        ax1.axvspan(
            start_t,
            end_t,
            color="#2ecc71",
            alpha=0.2,
            label="Vùng tiếng nói GT" if i == 0 else None,
        )

    # Kẻ các mốc biên chuẩn Ground Truth (đường dọc màu đỏ nét đứt)
    for i, b in enumerate(signal.gt_boundaries):
        ax1.axvline(
            b,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label="Biên chuẩn GT" if i == 0 else None,
        )

    # Kẻ các mốc biên thuật toán tìm được (đường dọc màu xanh dương nét liền)
    for i, b in enumerate(result.predicted_boundaries):
        ax1.axvline(
            b,
            color="blue",
            linestyle="-",
            linewidth=1.8,
            label="Biên thuật toán" if i == 0 else None,
        )

    env_text = "Phone" if signal.is_phone else "Studio"
    snr_text = f" | SNR={metrics.snr_db:.1f}dB" if metrics.snr_db > 0 else ""
    ax1.set_title(
        f"{signal.name} ({env_text}{snr_text}) - Thuật toán: {result.algorithm.upper()} | "
        f"Ngưỡng T={result.threshold_used:.4f} | MAE={metrics.mae_ms:.1f}ms, RMSE={metrics.rmse_ms:.1f}ms",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Biên độ")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # --- ĐỒ THỊ 2: ĐƯỜNG ĐẶC TRƯNG NGẮN HẠN & ĐƯỜNG NGƯỠNG T ---
    ax2.plot(
        result.timestamps,
        feature_vals,
        color="#e67e22",
        linewidth=1.2,
        label=feature_label,
    )
    ax2.axhline(
        result.threshold_used,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"Ngưỡng T = {result.threshold_used:.4f}",
    )

    # Gióng các mốc biên xuống đồ thị đặc trưng
    for b in signal.gt_boundaries:
        ax2.axvline(b, color="red", linestyle="--", linewidth=1.4, alpha=0.6)
    for b in result.predicted_boundaries:
        ax2.axvline(b, color="blue", linestyle="-", linewidth=1.4, alpha=0.6)

    ax2.set_xlabel("Thời gian (giây)")
    ax2.set_ylabel(feature_ylabel)
    ax2.set_xlim(0.0, signal.duration_sec)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def visualize_test_dataset(
    test_signals: list[AudioSignal],
    results: list[SegmentationResult],
    metrics_list: list[EvaluationMetrics],
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for i in range(len(test_signals)):
        sig = test_signals[i]
        res = results[i]
        met = metrics_list[i]

        file_name = f"figure_{i + 1}_{sig.name}_{res.algorithm}.png"
        img_path = out_path / file_name

        fig = plot_signal_segmentation(
            signal=sig,
            result=res,
            metrics=met,
            save_path=img_path,
            show=False,
        )
        plt.close(fig)
        saved_paths.append(img_path)
    return saved_paths


def plot_comparison_all_algorithms(
    signal: AudioSignal,
    results_by_algo: dict[str, SegmentationResult],
    metrics_by_algo: dict[str, EvaluationMetrics],
    save_path: str | Path | None = None,
) -> plt.Figure:
    algos = list(results_by_algo.keys())
    fig, axes = plt.subplots(
        nrows=len(algos), ncols=1, figsize=(14, 3.2 * len(algos)), sharex=True
    )
    if len(algos) == 1:
        axes = [axes]

    time_axis = np.linspace(0.0, signal.duration_sec, len(signal.signal))

    for idx, algo in enumerate(algos):
        res = results_by_algo[algo]
        met = metrics_by_algo[algo]
        ax = axes[idx]

        ax.plot(time_axis, signal.signal, color="#7f8c8d", alpha=0.5, linewidth=0.7)
        for s, e in signal.gt_segments:
            ax.axvspan(s, e, color="#2ecc71", alpha=0.15)

        for i, b in enumerate(signal.gt_boundaries):
            ax.axvline(
                b,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="Biên GT (Đỏ)" if i == 0 else None,
            )
        for i, b in enumerate(res.predicted_boundaries):
            ax.axvline(
                b,
                color="blue",
                linestyle="-",
                linewidth=1.5,
                label="Biên dự đoán (Xanh)" if i == 0 else None,
            )

        ax.set_title(
            f"Thuật toán: {algo.upper()} | Ngưỡng T={res.threshold_used:.4f} | MAE={met.mae_ms:.1f}ms, RMSE={met.rmse_ms:.1f}ms",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_ylabel("Biên độ")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Thời gian (giây)")
    axes[-1].set_xlim(0.0, signal.duration_sec)
    plt.tight_layout()

    if save_path is not None:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, dpi=150, bbox_inches="tight")

    return fig
