import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
from src.audio.dataset import load_dataset
from src.evaluate.metrics import evaluate_signal, summarize_metrics
from src.models import EvaluationMetrics, SegmentationResult
from src.segmentation.segment import segment_signal
from src.segmentation.threshold import calibrate_thresholds
from src.visualization.plotting import plot_comparison_all_algorithms, visualize_test_dataset


def main() -> None:
    output_dir = ROOT_DIR / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải tập dữ liệu huấn luyện và kiểm thử
    train_signals = load_dataset("TinHieuHuanLuyen")
    test_signals = load_dataset("TinHieuKiemThu")

    # 2. Hiệu chuẩn bộ ngưỡng cho cả 3 thuật toán
    thresholds = calibrate_thresholds(train_signals)

    algorithms = ["histogram", "binary", "gaussian"]
    results_by_algo: dict[str, list[SegmentationResult]] = {}
    metrics_by_algo: dict[str, list[EvaluationMetrics]] = {}
    summary_by_algo: dict[str, dict] = {}

    # 3. Chạy phân đoạn và đánh giá
    for algo in algorithms:
        r_list: list[SegmentationResult] = []
        m_list: list[EvaluationMetrics] = []
        for sig in test_signals:
            th_val = thresholds.get_threshold(algo, sig.is_phone)
            res = segment_signal(sig, th_val, algorithm=algo, min_silence_ms=200.0)
            met = evaluate_signal(sig, res.predicted_boundaries, algorithm=algo, threshold=th_val)
            r_list.append(res)
            m_list.append(met)
        results_by_algo[algo] = r_list
        metrics_by_algo[algo] = m_list
        summary_by_algo[algo] = summarize_metrics(m_list)

        # Lưu đồ thị từng thuật toán
        visualize_test_dataset(test_signals, r_list, m_list, output_dir=output_dir)

    # Lưu đồ thị so sánh xếp chồng cả 3 thuật toán
    for i, sig in enumerate(test_signals):
        res_dict = {algo: results_by_algo[algo][i] for algo in algorithms}
        met_dict = {algo: metrics_by_algo[algo][i] for algo in algorithms}
        comp_path = output_dir / f"comparison_{sig.name}.png"
        plot_comparison_all_algorithms(sig, res_dict, met_dict, save_path=comp_path)

    # Lưu báo cáo tổng hợp JSON
    report_json_path = ROOT_DIR / "reports" / "ket_qua_tong_hop.json"
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "thresholds": {
            "binary": {
                "global": thresholds.binary_threshold,
                "phone": thresholds.binary_threshold_phone,
                "studio": thresholds.binary_threshold_studio,
            },
            "histogram": {
                "global": thresholds.histogram_threshold,
                "phone": thresholds.histogram_threshold_phone,
                "studio": thresholds.histogram_threshold_studio,
            },
            "gaussian": {
                "global": thresholds.gaussian_threshold,
                "phone": thresholds.gaussian_threshold_phone,
                "studio": thresholds.gaussian_threshold_studio,
                "mean_speech": thresholds.mean_speech,
                "std_speech": thresholds.std_speech,
                "mean_silence": thresholds.mean_silence,
                "std_silence": thresholds.std_silence,
            },
        },
        "summary": summary_by_algo,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # 4. In bảng kết quả trực quan, ngắn gọn, dễ hiểu
    print("\n" + "=" * 96)
    print("                    KẾT QUẢ PHÂN ĐOẠN TIẾNG NÓI & KHOẢNG LẶNG (VAD)                    ")
    print("=" * 96)
    header = (
        f"{'Tên File':<12} {'Môi trường':<11} {'SNR (dB)':<10} "
        f"{'Histogram (MAE / RMSE)':<26} {'Binary (MAE)':<16} {'Gaussian (MAE)'}"
    )
    print(header)
    print("-" * 96)

    for i, sig in enumerate(test_signals):
        h_met = metrics_by_algo["histogram"][i]
        b_met = metrics_by_algo["binary"][i]
        g_met = metrics_by_algo["gaussian"][i]

        h_str = f"{h_met.mae_ms:4.1f} ms / {h_met.rmse_ms:4.1f} ms"
        b_str = f"{b_met.mae_ms:4.1f} ms"
        g_str = f"{g_met.mae_ms:4.1f} ms"

        print(
            f"{sig.name:<12} {h_met.environment:<11} {h_met.snr_db:4.1f} dB    "
            f"{h_str:<26} {b_str:<16} {g_str}"
        )

    print("-" * 96)
    # Tổng kết theo nhóm môi trường
    hp_mae, hp_rmse = summary_by_algo["histogram"]["Phone"]["mean_mae_ms"], summary_by_algo["histogram"]["Phone"]["mean_rmse_ms"]
    bp_mae = summary_by_algo["binary"]["Phone"]["mean_mae_ms"]
    gp_mae = summary_by_algo["gaussian"]["Phone"]["mean_mae_ms"]
    hp_str = f"{hp_mae:4.1f} ms / {hp_rmse:4.1f} ms"
    bp_str = f"{bp_mae:4.1f} ms"
    gp_str = f"{gp_mae:4.1f} ms"
    print(f"{'TB Phone':<12} {'Phone (2 file)':<21} {hp_str:<26} {bp_str:<16} {gp_str}")

    hs_mae, hs_rmse = summary_by_algo["histogram"]["Studio"]["mean_mae_ms"], summary_by_algo["histogram"]["Studio"]["mean_rmse_ms"]
    bs_mae = summary_by_algo["binary"]["Studio"]["mean_mae_ms"]
    gs_mae = summary_by_algo["gaussian"]["Studio"]["mean_mae_ms"]
    hs_str = f"{hs_mae:4.1f} ms / {hs_rmse:4.1f} ms"
    bs_str = f"{bs_mae:4.1f} ms"
    gs_str = f"{gs_mae:4.1f} ms"
    print(f"{'TB Studio':<12} {'Studio (2 file)':<21} {hs_str:<26} {bs_str:<16} {gs_str}")

    ha_mae, ha_rmse = summary_by_algo["histogram"]["Overall"]["mean_mae_ms"], summary_by_algo["histogram"]["Overall"]["mean_rmse_ms"]
    ba_mae = summary_by_algo["binary"]["Overall"]["mean_mae_ms"]
    ga_mae = summary_by_algo["gaussian"]["Overall"]["mean_mae_ms"]
    ha_str = f"{ha_mae:4.1f} ms / {ha_rmse:4.1f} ms"
    ba_str = f"{ba_mae:4.1f} ms"
    ga_str = f"{ga_mae:4.1f} ms"
    print(f"{'Toàn bộ':<12} {'Overall (4 file)':<21} {ha_str:<26} {ba_str:<16} {ga_str}")
    print("=" * 96)

    # In tóm tắt các giá trị ngưỡng tìm được
    print("\n[Ngưỡng phân đoạn hiệu chuẩn từ tập huấn luyện]:")
    print(f"  • Histogram : T_phone = {thresholds.histogram_threshold_phone:7.4f} | T_studio = {thresholds.histogram_threshold_studio:7.4f}  (Chung: T = {thresholds.histogram_threshold:7.4f})")
    print(f"  • Binary    : T_phone = {thresholds.binary_threshold_phone:7.4f} | T_studio = {thresholds.binary_threshold_studio:7.4f}  (Chung: T = {thresholds.binary_threshold:7.4f})")
    print(f"  • Gaussian  : T_phone = {thresholds.gaussian_threshold_phone:7.4f} | T_studio = {thresholds.gaussian_threshold_studio:7.4f}  (Chung: T = {thresholds.gaussian_threshold:7.4f})")

    print(f"\n[Hoàn tất]: Đã lưu 16 hình đồ thị vào {output_dir.relative_to(ROOT_DIR)}/ và số liệu JSON vào {report_json_path.relative_to(ROOT_DIR)}\n")


if __name__ == "__main__":
    main()
