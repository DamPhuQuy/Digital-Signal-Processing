# Module: `src/evaluate`
Đánh giá định lượng độ chính xác của thuật toán phân đoạn và đo lường mức nhiễu nền môi trường.

---

## 1. Các tệp thành phần
* **`metrics.py`**:
  - `estimate_snr_db`: Ước lượng tỷ số tín hiệu trên nhiễu SNR (dB) của file âm thanh bằng cách so sánh công suất trung bình giữa vùng Speech và Silence theo Ground Truth:
    $$SNR\text{ (dB)} = 10 \log_{10}\left(\frac{P_{\text{speech}}}{P_{\text{silence}}}\right)$$
  - `match_boundaries`: Ghép cặp biên dự đoán với biên Ground Truth gần nhất theo khoảng cách thời gian.
  - `compute_mae`: Tính sai số tuyệt đối trung bình (Mean Absolute Error) tính theo mili-giây (ms).
  - `compute_rmse`: Tính sai số căn bậc hai trung bình (Root Mean Squared Error) tính theo mili-giây (ms).
  - `evaluate_signal`: Tính toán tổng hợp MAE, RMSE, SNR cho 1 file kiểm thử và đóng gói vào `EvaluationMetrics`.
  - `summarize_metrics`: Tính trung bình MAE, RMSE phân nhóm theo môi trường `Phone`, `Studio` và `Overall`.
  - `print_evaluation_table`: In bảng kết quả chi tiết từng file và bảng tổng kết lên terminal.

---

## 2. Quy trình luồng làm việc (Pipeline)

```
AudioSignal (Chứa biên chuẩn GT) + Biên dự đoán (predicted_boundaries)
         │
         ▼
[metrics.py: evaluate_signal()]
         │
         ├──> [estimate_snr_db()]     --> Đo SNR môi trường thu âm
         ├──> [match_boundaries()]    --> Ghép 1-1 từng biên với mốc GT gần nhất
         ├──> [compute_mae()]         --> Tính MAE (ms)
         └──> [compute_rmse()]        --> Tính RMSE (ms)
         │
         ▼
Đối tượng EvaluationMetrics
         │
         ▼
[metrics.py: summarize_metrics()] / [print_evaluation_table()]
         │
         ▼
Báo cáo thống kê (Phone, Studio, Overall) hiển thị ra màn hình
```
