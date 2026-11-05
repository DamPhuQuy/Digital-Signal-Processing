# Đồ Án Giữa Kỳ Xử Lý Tín Hiệu Số (2026)

## Phân Đoạn Tín Hiệu Thành Tiếng Nói Và Khoảng Lặng (VAD)

Dự án cài đặt và đánh giá các thuật toán phân đoạn tiếng nói / khoảng lặng (Voice Activity Detection - VAD) dựa trên năng lượng ngắn hạn (Short-Time Energy - STE) chuẩn hóa:

1. **Binary Search (Ternary Search)**: Tìm ngưỡng tối ưu giảm thiểu số khung phân loại sai.
2. **Histogram**: Tìm cực tiểu cục bộ (đáy thung lũng) giữa hai đỉnh phân phối tiếng nói và khoảng lặng.
3. **Gaussian Mixture Model (1D)**: Ước lượng phân phối Gauss cho tiếng nói và khoảng lặng để tìm ngưỡng giao thoa.

---

## Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### 2. Chạy chương trình

Chạy script chính để duyệt qua 4 file kiểm thử và xuất các figure kết quả:

```bash
python main.py
```

*(Có thể truyền tham số để chạy thuật toán cụ thể: `python main.py binary`, `python main.py histogram`, `python main.py gaussian`)*

### 3. Xem kết quả

- Bảng sai số định lượng (MAE, RMSE) theo từng file và từng môi trường (Phone, Studio) sẽ hiển thị trực tiếp trên màn hình console.
- Các đồ thị trực quan hóa sẽ được lưu tại thư mục: `reports/figures/`.

---

## Cấu Trúc Dự Án

```text
main_gk/
├── requirements.txt
├── README.md
├── main.py                     # Script thực thi chính của đồ án
├── notebooks/
│   └── audio.ipynb
├── src/
│   ├── audio/                  # Đọc file wav, nạp dữ liệu nhãn .lab
│   ├── features/               # Phân khung (framing), trích xuất đặc trưng STE/MA
│   ├── segmentation/           # Thuật toán tìm ngưỡng (Binary, Hist, Gauss) & hậu xử lý
│   ├── evaluate/               # Đánh giá khớp biên, tính MAE, RMSE
│   ├── visualization/          # Vẽ đồ thị và lưu các figure
│   ├── models.py               # Định nghĩa các dataclass dữ liệu
│   └── pipeline.py             # Luồng điều phối huấn luyện và kiểm thử
├── TinHieuHuanLuyen/           # Dữ liệu huấn luyện (4 file wav + lab)
├── TinHieuKiemThu/             # Dữ liệu kiểm thử (4 file wav + lab)
└── reports/
    └── figures/                # Thư mục chứa các hình ảnh kết quả xuất ra
```
