#!/usr/bin/env python3
"""Chương trình chính: Phân đoạn tín hiệu tiếng nói và khoảng lặng (VAD).

Môn học: XỬ LÝ TÍN HIỆU SỐ (Digital Signal Processing - DSP)
Giảng viên hướng dẫn: Ninh Khánh Duy

Quy trình xử lý tuần tự theo kiến trúc 5 khối chuẩn (Pipeline Architecture):
    ┌────────────────────────────────────────────────────────────────────────┐
    │ KHỐI 1: PREPROCESSING (TIỀN XỬ LÝ TÍN HIỆU)                            │
    │   • Bước 1: Nạp dữ liệu âm thanh WAV & Ground Truth .lab (Train/Test)  │
    │   • Bước 2: Phân khung (25ms, dịch 10ms) + Nhân cửa sổ Hamming         │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ↓
    ┌────────────────────────────────────────────────────────────────────────┐
    │ KHỐI 2: FEATURE EXTRACTION (TRÍCH XUẤT ĐẶC TRƯNG)                      │
    │   • Bước 3: Tính năng lượng ngắn hạn (STE) + Chuẩn hóa Min-Max [0, 1]  │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ↓
    ┌────────────────────────────────────────────────────────────────────────┐
    │ KHỐI 3: THRESHOLD CALIBRATION (HUẤN LUYỆN TÌM NGƯỠNG TRÊN TRAIN)       │
    │   • Bước 4: Tối ưu hóa ngưỡng T (Binary Search / Histogram / Gaussian) │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ↓
    ┌────────────────────────────────────────────────────────────────────────┐
    │ KHỐI 4: INFERENCE & POST-PROCESSING (PHÂN ĐOẠN VAD & HẬU XỬ LÝ)        │
    │   • Bước 5: Chạy phân đoạn VAD trên TEST (So sánh STE với ngưỡng T)    │
    │   • Bước 6: Hậu xử lý (Lọc bỏ khoảng lặng ngắn < 200ms & trích biên)   │
    └───────────────────────────────────┬────────────────────────────────────┘
                                        ↓
    ┌────────────────────────────────────────────────────────────────────────┐
    │ KHỐI 5: EVALUATION & VISUALIZATION (ĐÁNH GIÁ & TRỰC QUAN HÓA)          │
    │   • Bước 7: Đánh giá sai số định lượng (MAE, RMSE, FER, F1-Score)      │
    │   • Bước 8: Trực quan hóa kết quả (Plot đồ thị tổng hợp 2x2 subplot)   │
    └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.models import AudioSignal
from src.pipeline import SpeechSegmenter
from src.evaluate import plot_combined_segmentation_figure, plot_segmentation_figure
from src.audio import load_dataset


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    data_path = root_dir / "data"

    train_dir = data_path / "TinHieuHuanLuyen" if (data_path / "TinHieuHuanLuyen").is_dir() else root_dir / "TinHieuHuanLuyen"
    test_dir = data_path / "TinHieuKiemThu" if (data_path / "TinHieuKiemThu").is_dir() else root_dir / "TinHieuKiemThu"

    # =========================================================================
    # KHỐI 1: PREPROCESSING (TIỀN XỬ LÝ TÍN HIỆU)
    #   - Bước 1: Nạp dữ liệu âm thanh và nhãn Ground Truth (Load Train/Test)
    #   - Bước 2: Phân khung & Cửa sổ hóa (Framing + Hamming)
    # =========================================================================
    print("=" * 86)
    print("      HỆ THỐNG PHÂN ĐOẠN TIẾNG NÓI VÀ KHOẢNG LẶNG (VOICE ACTIVITY DETECTION - VAD)")
    print("=" * 86)

    print("\n[KHỐI 1: PREPROCESSING - TIỀN XỬ LÝ TÍN HIỆU]")
    train_signals = load_dataset(train_dir)
    test_signals = load_dataset(test_dir)
    print(f"  • Bước 1 (Load Data):")
    print(f"    - Tập huấn luyện ({len(train_signals)} files): {[s.name for s in train_signals]}")
    print(f"    - Tập kiểm thử   ({len(test_signals)} files): {[s.name for s in test_signals]}")

    frame_len_ms = 25.0
    frame_shift_ms = 10.0
    min_silence_ms = 200.0

    print(f"  • Bước 2 (Framing + Windowing):")
    print(f"    - Độ dài khung: {frame_len_ms:.0f}ms (400 mẫu ở fs = 16000Hz)")
    print(f"    - Độ dịch khung: {frame_shift_ms:.0f}ms (160 mẫu, độ chồng lấp overlap 60%)")
    print(f"    - Cửa sổ phân tích: Hamming (giảm rò rỉ phổ)")

    # =========================================================================
    # KHỐI 2: FEATURE EXTRACTION (TRÍCH XUẤT ĐẶC TRƯNG NĂNG LƯỢNG)
    #   - Bước 3: Năng lượng ngắn hạn (STE) + Chuẩn hóa Min-Max [0, 1]
    # =========================================================================
    print("\n[KHỐI 2: FEATURE EXTRACTION - TRÍCH XUẤT ĐẶC TRƯNG]")
    print(f"  • Bước 3 (Short-Time Energy & Normalization):")
    print(f"    - Tính năng lượng ngắn hạn STE cho từng khung tín hiệu")
    print(f"    - Chuẩn hóa Min-Max đưa đặc trưng về dải [0, 1]")

    segmenter = SpeechSegmenter(
        frame_size_ms=frame_len_ms,
        frame_shift_ms=frame_shift_ms,
        window_type="hamming",
        min_silence_ms=min_silence_ms,
    )

    # =========================================================================
    # KHỐI 3: THRESHOLD CALIBRATION (HUẤN LUYỆN XÁC ĐỊNH NGƯỠNG)
    #   - Bước 4: Tối ưu hóa ngưỡng phân biệt T trên tập huấn luyện (TRAIN)
    # =========================================================================
    print("\n[KHỐI 3: THRESHOLD CALIBRATION - HUẤN LUYỆN XÁC ĐỊNH NGƯỠNG]")
    print(f"  • Bước 4 (Training on Train Signals):")
    thresholds = segmenter.find_thresholds(train_signals)
    print(f"    - Binary Search : Global = {thresholds.binary_global:.4f} | Phone = {thresholds.binary_phone:.4f} | Studio = {thresholds.binary_studio:.4f}")
    print(f"    - Histogram 2 đỉnh : Global = {thresholds.histogram_global:.4f} | Phone = {thresholds.get_threshold('histogram', is_phone=True):.4f} | Studio = {thresholds.get_threshold('histogram', is_phone=False):.4f}")
    print(f"    - Gaussian (GMM): Global = {thresholds.gaussian_global:.4f} | Phone = {thresholds.get_threshold('gaussian', is_phone=True):.4f} | Studio = {thresholds.get_threshold('gaussian', is_phone=False):.4f}")

    # =========================================================================
    # KHỐI 4: INFERENCE & POST-PROCESSING (PHÂN ĐOẠN VAD & HẬU XỬ LÝ)
    # KHỐI 5: EVALUATION (ĐÁNH GIÁ SAI SỐ ĐỊNH LƯỢNG)
    #   - Bước 5: Chạy phân đoạn VAD trên TEST
    #   - Bước 6: Lọc bỏ khoảng lặng ngắn < 200ms
    #   - Bước 7: Đánh giá sai số định lượng (MAE, RMSE, FER, F1-Score)
    # =========================================================================
    print("\n[KHỐI 4 & 5: INFERENCE, POST-PROCESSING & EVALUATION]")
    print(f"  • Bước 5 (VAD Inference) : Áp dụng ngưỡng T phân loại khung 0/1 trên tập kiểm thử")
    print(f"  • Bước 6 (Post-processing): Lọc bỏ khoảng lặng < {min_silence_ms:.0f}ms & trích xuất ranh giới biên")
    print(f"  • Bước 7 (Error Metrics)  : Tính toán MAE (ms), RMSE (ms), FER (%), F1-Score")

    algorithms = ["binary", "histogram", "gaussian"]
    algo_names = {
        "binary": "Binary Search",
        "histogram": "Histogram 2 đỉnh",
        "gaussian": "Gaussian (GMM)",
    }

    print("=" * 86)
    print("                BẢNG ĐÁNH GIÁ ĐỊNH LƯỢNG TRÊN TẬP KIỂM THỬ")
    print("=" * 86)

    best_eval_records = None

    for algo in algorithms:
        print(f"\n>>> THUẬT TOÁN: {algo_names[algo].upper()}")
        print("-" * 86)
        print(f"{'Tên file':<12} | {'Môi trường':<10} | {'Ngưỡng T':<10} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'FER (%)':<10} | {'F1-Score':<10}")
        print("-" * 86)

        eval_records = segmenter.evaluate_dataset(test_signals, algorithm=algo, use_adaptive=True)
        if algo == "binary":
            best_eval_records = eval_records

        mae_list, rmse_list, fer_list, f1_list = [], [], [], []
        for sig, framed, features, seg_res, m in eval_records:
            mae_list.append(m.mae_ms)
            rmse_list.append(m.rmse_ms)
            fer_list.append(m.fer * 100.0)
            f1_list.append(m.f1)
            print(f"{sig.name:<12} | {sig.environment_name:<10} | {seg_res.threshold_used:<10.4f} | {m.mae_ms:<10.2f} | {m.rmse_ms:<10.2f} | {m.fer * 100.0:<10.2f} | {m.f1:<10.4f}")

        print("-" * 86)
        print(f"{'TRUNG BÌNH':<12} | {'Toàn tập':<10} | {'-':<10} | {np.mean(mae_list):<10.2f} | {np.mean(rmse_list):<10.2f} | {np.mean(fer_list):<10.2f} | {np.mean(f1_list):<10.4f}")
    print("=" * 86)

    # =========================================================================
    # KHỐI 5: VISUALIZATION (TRỰC QUAN HÓA KẾT QUẢ)
    #   - Bước 8: Trực quan hóa toàn bộ 4 file kiểm thử trong cùng 1 đồ thị
    # =========================================================================
    print("\n[KHỐI 5: VISUALIZATION - TRỰC QUAN HÓA ĐỒ THỊ TỔNG HỢP]")
    print("  • Bước 8 (Plot Results): Vẽ đồ thị tổng hợp lưới 2x2 subplot tương ứng 4 góc màn hình...")
    if best_eval_records is not None:
        fig = plot_combined_segmentation_figure(best_eval_records, layout="2x2")

    if "--no-gui" not in sys.argv:
        print("    [!] Đang hiển thị Figure đồ thị tổng hợp (lưới 2x2 cho 4 file kiểm thử). Đóng cửa sổ để hoàn tất.")
        plt.show()
    else:
        print("    [!] Chế độ --no-gui: Đã hoàn tất tính toán và đánh giá định lượng thành công.")


if __name__ == "__main__":
    main()
