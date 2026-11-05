# Module: `src/visualization`
Trực quan hóa kết quả phân đoạn tín hiệu và xuất các hình vẽ báo cáo.

---

## 1. Các tệp thành phần
* **`plotting.py`**:
  - `plot_signal_segmentation`: Vẽ đồ thị kết quả 2 tầng cho một file kiểm thử:
    - **Tầng 1 (Trên - Dạng sóng âm thanh)**: Sóng $x(t)$, vùng tiếng nói chuẩn Ground Truth (nền xanh lá nhạt), các mốc biên chuẩn Ground Truth (đường dọc đỏ nét đứt), và các mốc biên thuật toán tìm được (đường dọc xanh dương nét liền). Tiêu đề hiển thị tên file, môi trường, SNR, thuật toán, ngưỡng $T$, MAE và RMSE.
    - **Tầng 2 (Dưới - Năng lượng STE)**: Đường năng lượng chuẩn hóa màu cam, đường ngưỡng ngang phân tách $T$, cùng các vạch biên đỏ và xanh gióng thẳng từ tầng trên xuống.
  - `visualize_test_dataset`: Vòng lặp duyệt qua toàn bộ danh sách file kiểm thử, gọi hàm vẽ và tự động lưu 4 hình ảnh `.png` vào thư mục `reports/figures/`.
  - `plot_comparison_all_algorithms`: Vẽ đồ thị so sánh dạng sóng và mốc biên của cả 3 thuật toán (Binary, Histogram, Gaussian) xếp chồng dọc trên cùng một file âm thanh.

---

## 2. Quy trình luồng làm việc (Pipeline)

```
Danh sách test_signals + Kết quả results + Chỉ số metrics_list
         │
         ▼
[plotting.py: visualize_test_dataset()]
         │
         ├──> Duyệt từng file kiểm thử (1 đến 4)
         │       │
         │       ▼
         │    [plot_signal_segmentation()]
         │       ├──> Vẽ Tầng 1: Sóng âm + Biên GT (Đỏ) + Biên thuật toán (Xanh)
         │       └──> Vẽ Tầng 2: STE + Đường ngưỡng ngang T
         │
         ▼
Xuất và lưu 4 file hình ảnh tại: reports/figures/figure_{i}_{name}_{algo}.png
```
