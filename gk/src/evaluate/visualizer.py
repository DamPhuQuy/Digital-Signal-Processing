"""Visualizer Module for VAD Evaluation.

Cung cấp các công cụ trực quan hóa kết quả phân đoạn VAD theo yêu cầu đề bài:
1. plot_combined_segmentation_figure: Trực quan hóa toàn bộ 4 file kiểm thử trong cùng một đồ thị
   với các subplot (mặc định dạng lưới 2x2 tương ứng 4 góc: Trên-Trái, Trên-Phải, Dưới-Trái, Dưới-Phải,
   hoặc 4x2 với đường quyết định phân đoạn 0/1 riêng biệt).
2. plot_segmentation_figure: Vẽ đồ thị riêng lẻ cho một tín hiệu với 2 subplot (Waveform + STE và VAD 0/1).
3. position_figure_at_corner: Tùy chọn định vị cửa sổ riêng biệt tại 4 góc màn hình khi cần.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.models import (
    AudioSignal,
    EvaluationMetrics,
    FramedSignal,
    SegmentationResult,
    ShortTimeFeatures,
)


def get_screen_dimensions() -> Tuple[int, int]:
    """Lấy độ phân giải màn hình máy tính hiện tại (chiều rộng, chiều cao)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def position_figure_at_corner(fig: plt.Figure, corner_index: int) -> None:
    """Định vị cửa sổ Figure vào đúng 1 trong 4 góc màn hình theo yêu cầu thi.

    Thứ tự các góc:
        0: Góc trên - trái (Top-Left)
        1: Góc trên - phải (Top-Right)
        2: Góc dưới - trái (Bottom-Left)
        3: Góc dưới - phải (Bottom-Right)
    """
    screen_w, screen_h = get_screen_dimensions()
    half_w = screen_w // 2
    half_h = (screen_h - 70) // 2

    positions = [
        (0, 0),             # 0: Top-Left
        (half_w, 0),       # 1: Top-Right
        (0, half_h),       # 2: Bottom-Left
        (half_w, half_h),  # 3: Bottom-Right
    ]
    x, y = positions[corner_index % 4]

    try:
        mgr = fig.canvas.manager
        if hasattr(mgr, "window") and hasattr(mgr.window, "wm_geometry"):
            mgr.window.wm_geometry(f"{half_w}x{half_h}+{x}+{y}")
        elif hasattr(mgr, "window") and hasattr(mgr.window, "setGeometry"):
            mgr.window.setGeometry(x, y, half_w, half_h)
    except Exception:
        pass


def plot_combined_segmentation_figure(
    records: Sequence[Tuple[AudioSignal, FramedSignal, ShortTimeFeatures, SegmentationResult, EvaluationMetrics]],
    layout: str = "2x2",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Trực quan hóa toàn bộ kết quả phân đoạn VAD của các tín hiệu kiểm thử trong cùng một Figure với các subplot.

    Tham số:
        records: Danh sách các bộ dữ liệu đánh giá:
                 (signal, framed_signal, short_time_features, segmentation_result, evaluation_metrics).
        layout: Định dạng hiển thị subplot:
                - '2x2' (Mặc định): Lưới 2x2 tương ứng với 4 góc màn hình
                   (Góc trên-trái: file 1, Góc trên-phải: file 2,
                    Góc dưới-trái: file 3, Góc dưới-phải: file 4).
                   Tích hợp vùng phủ màu (fill_between) cho quyết định tiếng nói (Speech).
                - '4x2': Lưới 4 hàng x 2 cột. Mỗi file kiểm thử gồm 2 subplot con:
                   Cột 1: Waveform + STE + Ngưỡng T + Đường biên;
                   Cột 2: Quyết định phân đoạn VAD (0/1) sau lọc 200ms.
                - '4x1': 4 hàng ngang rộng hiển thị Waveform, STE, Ngưỡng và vùng tiếng nói.
        figsize: Kích thước của Figure (chiều rộng, chiều cao).
        title: Tiêu đề tổng thể (suptitle) của Figure.

    Trả về:
        plt.Figure: Đối tượng matplotlib Figure chứa tất cả các subplot.
    """
    num_signals = len(records)
    if num_signals == 0:
        raise ValueError("Danh sách records đánh giá không được để trống.")

    corner_labels = [
        "Góc trên - trái",
        "Góc trên - phải",
        "Góc dưới - trái",
        "Góc dưới - phải",
    ]

    if layout == "2x2":
        if figsize is None:
            figsize = (16, 9.5)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes_flat = axes.flatten()

        for idx, (sig, framed, feat, res, m) in enumerate(records):
            if idx >= len(axes_flat):
                break
            ax = axes_flat[idx]
            t_sig = np.arange(len(sig.signal)) / float(sig.fs)

            # 1. Vùng phủ màu tiếng nói (VAD = 1)
            speech_mask = (res.filtered_decisions == 1)
            ax.fill_between(
                framed.timestamps, -1.0, 1.0, where=speech_mask,
                color="palegreen", alpha=0.3, label="Vùng tiếng nói (VAD = 1)"
            )

            # 2. Waveform và STE chuẩn hóa
            ax.plot(t_sig, sig.signal, color="silver", linewidth=0.7, alpha=0.85, label="Waveform [-1, 1]")
            ax.plot(framed.timestamps, feat.ste_norm, color="royalblue", linewidth=1.2, label="Normalized STE")
            ax.axhline(
                res.threshold_used, color="darkorange", linestyle="--", linewidth=1.1,
                label=f"Ngưỡng T = {res.threshold_used:.4f}"
            )

            # 3. Đường biên Ground Truth (đỏ liền) và Dự đoán VAD (xanh nét đứt)
            labeled_red = False
            for gt_t in sig.gt_boundaries:
                lbl = "Ground Truth (Đỏ)" if not labeled_red else None
                ax.axvline(gt_t, color="crimson", linestyle="-", linewidth=1.5, alpha=0.9, label=lbl)
                labeled_red = True

            labeled_green = False
            for pred_t in res.predicted_boundaries:
                lbl = "Dự đoán VAD (Xanh)" if not labeled_green else None
                ax.axvline(pred_t, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.9, label=lbl)
                labeled_green = True

            c_label = corner_labels[idx] if idx < len(corner_labels) else f"Plot {idx+1}"
            ax.set_title(
                f"[{c_label}: {sig.name} ({sig.environment_name})]\n"
                f"Ngưỡng T = {res.threshold_used:.4f} | MAE = {m.mae_ms:.1f}ms | RMSE = {m.rmse_ms:.1f}ms | FER = {m.fer*100:.2f}% | F1 = {m.f1:.4f}",
                fontsize=9.5, fontweight="bold", pad=7
            )
            ax.set_xlabel("Thời gian (giây)", fontsize=9)
            ax.set_ylabel("Biên độ / STE", fontsize=9)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

        if title is None:
            title = (
                "TỔNG HỢP KẾT QUẢ PHÂN ĐOẠN TIẾNG NÓI VÀ KHOẢNG LẶNG (VAD) TRÊN 4 TÍN HIỆU KIỂM THỬ\n"
                "(Bố cục 4 góc: Trên-Trái: phone_F2 | Trên-Phải: phone_M2 | Dưới-Trái: studio_F2 | Dưới-Phải: studio_M2)"
            )
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    elif layout == "4x2":
        if figsize is None:
            figsize = (16, 11)
        fig, axes = plt.subplots(num_signals, 2, figsize=figsize, sharex="row")

        for idx, (sig, framed, feat, res, m) in enumerate(records):
            ax_wf = axes[idx, 0] if num_signals > 1 else axes[0]
            ax_vad = axes[idx, 1] if num_signals > 1 else axes[1]
            t_sig = np.arange(len(sig.signal)) / float(sig.fs)

            # Cột 1: Waveform và STE
            ax_wf.plot(t_sig, sig.signal, color="silver", linewidth=0.7, label="Waveform [-1, 1]")
            ax_wf.plot(framed.timestamps, feat.ste_norm, color="royalblue", linewidth=1.2, label="Normalized STE")
            ax_wf.axhline(
                res.threshold_used, color="darkorange", linestyle="--", linewidth=1.0,
                label=f"T = {res.threshold_used:.4f}"
            )

            labeled_red = False
            for gt_t in sig.gt_boundaries:
                lbl = "Ground Truth" if not labeled_red else None
                ax_wf.axvline(gt_t, color="crimson", linestyle="-", linewidth=1.5, alpha=0.85, label=lbl)
                labeled_red = True

            labeled_green = False
            for pred_t in res.predicted_boundaries:
                lbl = "Biên VAD" if not labeled_green else None
                ax_wf.axvline(pred_t, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.85, label=lbl)
                labeled_green = True

            c_label = corner_labels[idx] if idx < len(corner_labels) else f"File {idx+1}"
            ax_wf.set_title(
                f"[{c_label}] {sig.name} ({sig.environment_name}) | MAE: {m.mae_ms:.1f}ms | RMSE: {m.rmse_ms:.1f}ms",
                fontsize=9, fontweight="bold"
            )
            ax_wf.set_ylabel("Biên độ / STE", fontsize=8)
            ax_wf.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax_wf.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

            # Cột 2: Quyết định phân đoạn
            ax_vad.step(
                framed.timestamps, res.filtered_decisions, color="forestgreen", linewidth=1.3,
                where="mid", label="VAD Decision"
            )
            ax_vad.set_title(
                f"VAD Decision (Sau lọc 200ms) | FER: {m.fer*100:.2f}% | F1: {m.f1:.4f}",
                fontsize=9, fontweight="bold"
            )
            ax_vad.set_ylabel("Nhãn VAD", fontsize=8)
            ax_vad.set_ylim(-0.1, 1.2)
            ax_vad.set_yticks([0, 1])
            ax_vad.set_yticklabels(["Silence", "Speech"], fontsize=8)
            ax_vad.grid(True, linestyle=":", alpha=0.6)

        axes[-1, 0].set_xlabel("Thời gian (giây)", fontsize=9)
        axes[-1, 1].set_xlabel("Thời gian (giây)", fontsize=9)

        if title is None:
            title = "TỔNG HỢP KẾT QUẢ PHÂN ĐOẠN VAD TRÊN 4 TÍN HIỆU KIỂM THỬ"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    else:  # '4x1'
        if figsize is None:
            figsize = (16, 11)
        fig, axes = plt.subplots(num_signals, 1, figsize=figsize, sharex=False)

        for idx, (sig, framed, feat, res, m) in enumerate(records):
            ax = axes[idx] if num_signals > 1 else axes
            t_sig = np.arange(len(sig.signal)) / float(sig.fs)

            speech_mask = (res.filtered_decisions == 1)
            ax.fill_between(
                framed.timestamps, -1.0, 1.0, where=speech_mask,
                color="palegreen", alpha=0.3, label="Vùng tiếng nói (VAD = 1)"
            )

            ax.plot(t_sig, sig.signal, color="silver", linewidth=0.7, label="Waveform [-1, 1]")
            ax.plot(framed.timestamps, feat.ste_norm, color="royalblue", linewidth=1.2, label="Normalized STE")
            ax.axhline(
                res.threshold_used, color="darkorange", linestyle="--", linewidth=1.1,
                label=f"Ngưỡng T = {res.threshold_used:.4f}"
            )

            labeled_red = False
            for gt_t in sig.gt_boundaries:
                lbl = "Ground Truth" if not labeled_red else None
                ax.axvline(gt_t, color="crimson", linestyle="-", linewidth=1.5, alpha=0.9, label=lbl)
                labeled_red = True

            labeled_green = False
            for pred_t in res.predicted_boundaries:
                lbl = "Dự đoán (VAD)" if not labeled_green else None
                ax.axvline(pred_t, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.9, label=lbl)
                labeled_green = True

            c_label = corner_labels[idx] if idx < len(corner_labels) else f"File {idx+1}"
            ax.set_title(
                f"[{c_label}] File: {sig.name} ({sig.environment_name}) | Ngưỡng T = {res.threshold_used:.4f} | "
                f"MAE = {m.mae_ms:.1f}ms | RMSE = {m.rmse_ms:.1f}ms | FER = {m.fer*100:.2f}% | F1 = {m.f1:.4f}",
                fontsize=9.5, fontweight="bold"
            )
            ax.set_ylabel("Biên độ / STE", fontsize=9)
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8, ncol=6, framealpha=0.95)

        axes[-1].set_xlabel("Thời gian (giây)", fontsize=10)
        if title is None:
            title = "TỔNG HỢP KẾT QUẢ PHÂN ĐOẠN TIẾNG NÓI VÀ KHOẢNG LẶNG (VAD) TRÊN 4 TÍN HIỆU KIỂM THỬ"
        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.985)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_segmentation_figure(
    signal: AudioSignal,
    framed: FramedSignal,
    features: ShortTimeFeatures,
    result: SegmentationResult,
    metrics: EvaluationMetrics,
    corner_index: Optional[int] = None,
) -> plt.Figure:
    """Vẽ cửa sổ Figure chuẩn cho một tín hiệu kiểm thử theo đúng yêu cầu đề bài.

    Tham số:
        signal: Dữ liệu tín hiệu âm thanh AudioSignal.
        framed: Tín hiệu đã chia khung FramedSignal.
        features: Các đặc trưng năng lượng ngắn hạn ShortTimeFeatures.
        result: Kết quả phân đoạn SegmentationResult.
        metrics: Kết quả đánh giá sai số EvaluationMetrics.
        corner_index: Chỉ số góc màn hình (0-3) để tự động định vị cửa sổ.

    Trả về:
        plt.Figure: Đối tượng matplotlib Figure đã tạo.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True)
    try:
        fig.canvas.manager.set_window_title(f"VAD - {signal.name} ({signal.environment_name})")
    except Exception:
        pass

    # Trục thời gian tín hiệu liên tục
    t_sig = np.arange(len(signal.signal)) / float(signal.fs)

    # Subplot 1: Waveform và đường STE xếp chồng
    ax1.plot(t_sig, signal.signal, color="silver", alpha=0.8, linewidth=0.7, label="Waveform [-1, 1]")
    ax1.plot(framed.timestamps, features.ste_norm, color="royalblue", linewidth=1.2, label="Normalized STE")
    ax1.axhline(
        result.threshold_used, color="darkorange", linestyle="--", linewidth=1.0,
        label=f"Threshold T = {result.threshold_used:.4f}"
    )

    # Vẽ các đường biên Ground Truth (đỏ liền) và Dự đoán (xanh lá nét đứt)
    labeled_red = False
    for gt_t in signal.gt_boundaries:
        lbl = "Ground Truth (Red)" if not labeled_red else None
        ax1.axvline(gt_t, color="crimson", linestyle="-", linewidth=1.5, alpha=0.85, label=lbl)
        labeled_red = True

    labeled_green = False
    for pred_t in result.predicted_boundaries:
        lbl = "Predicted (Green)" if not labeled_green else None
        ax1.axvline(pred_t, color="forestgreen", linestyle="--", linewidth=1.5, alpha=0.85, label=lbl)
        labeled_green = True

    ax1.set_title(
        f"File: {signal.name} | Môi trường: {signal.environment_name} | "
        f"MAE: {metrics.mae_ms:.1f} ms | RMSE: {metrics.rmse_ms:.1f} ms | FER: {metrics.fer * 100:.2f}%",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_ylabel("Biên độ / STE", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # Subplot 2: Quyết định phân đoạn
    ax2.step(
        framed.timestamps, result.filtered_decisions, color="forestgreen", linewidth=1.5,
        where="mid", label="Thuật toán (Sau lọc 200ms)"
    )
    ax2.set_ylabel("Nhãn VAD (0/1)", fontsize=9)
    ax2.set_xlabel("Thời gian (giây)", fontsize=9)
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Silence (0)", "Speech (1)"])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if corner_index is not None:
        position_figure_at_corner(fig, corner_index)

    return fig


# Bí danh tương thích ngược
plot_vad_sample_figure = plot_segmentation_figure
plot_vad_result = plot_segmentation_figure
plot_combined_vad_figure = plot_combined_segmentation_figure
