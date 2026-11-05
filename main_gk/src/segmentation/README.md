# Module: `src/segmentation`
Cài đặt các thuật toán tìm ngưỡng phân tách Speech/Silence, phân loại khung và hậu xử lý.

---

## 1. Các tệp thành phần
* **`binary.py`**: Thuật toán Tìm kiếm nhị phân (Binary Search) theo tài liệu **CS425 (Hodgkinson, 2012) - Mục 2.1.3**:
  - `find_threshold_binary_search`: Thu hẹp nhị phân để cân bằng diện tích nhầm lẫn (*area of confusion*) trên vùng chồng lấn (*overlap*) theo công thức (2.8).
  - `find_threshold_binary`: Hàm interface nhận mảng STE và nhãn Ground Truth để tìm ngưỡng $T$.
* **`gauss.py`**: Thuật toán mô hình phân phối chuẩn Gauss:
  - Khảo sát các khung Speech và Silence trên tập huấn luyện để tính $(\mu_{\text{sp}}, \sigma_{\text{sp}}, \mu_{\text{sil}}, \sigma_{\text{sil}})$.
  - Tìm giao điểm tối ưu giữa 2 hàm mật độ xác suất làm ngưỡng $T$.
* **`histogram.py`**: Module dành riêng cho thuật toán Biểu đồ tần suất Histogram (Giannakopoulos, 2014) - *đang để trống chờ member phụ trách hoàn thiện*.
* **`threshold.py`**: Điều phối huấn luyện ngưỡng trên tập `TinHieuHuanLuyen`:
  - Hàm `calibrate_thresholds`: Tính ngưỡng phân tách riêng biệt cho môi trường **Phone** và **Studio** để thích nghi với mức nhiễu nền khác nhau.
  - Lưu trữ kết quả trong dataclass `Threshold`.
* **`segment.py`**: Thực hiện phân đoạn cho 1 tín hiệu kiểm thử:
  - Chia khung và trích xuất STE.
  - So sánh năng lượng từng khung với ngưỡng $T$ để tạo nhãn quyết định sơ bộ `raw_decisions`.
  - Gọi module hậu xử lý và trả về đối tượng `SegmentationResult`.
* **`postprocess.py`**: Hậu xử lý kết quả phân đoạn:
  - `filter_short_silence`: Loại bỏ các khoảng lặng ảo $< 200\text{ ms}$ theo đúng quy định đề bài.
  - `extract_boundaries_and_segments`: Trích xuất các mốc thời gian chuyển trạng thái (biên dự đoán) và danh sách các đoạn tín hiệu.

---

## 2. Quy trình luồng làm việc (Pipeline)

### Giai đoạn 1: Huấn luyện tìm ngưỡng (Training Phase)
```
Tập huấn luyện TinHieuHuanLuyen
         │
         ▼
[threshold.py: calibrate_thresholds()]
         ├──> [binary.py: find_threshold_binary()]   --> Ngưỡng Binary (Phone & Studio)
         ├──> [gauss.py: find_threshold_gaussian()]  --> Ngưỡng Gaussian (Phone & Studio)
         └──> [find_threshold_histogram()]           --> Ngưỡng Histogram (Phone & Studio)
         │
         ▼
Đối tượng Threshold (Chứa bộ tham số tối ưu)
```

### Giai đoạn 2: Phân đoạn tín hiệu kiểm thử (Inference Phase)
```
AudioSignal (File test) + Ngưỡng T phù hợp
         │
         ▼
[segment.py: segment_signal()]
         │
         ├──> So sánh: STE[i] >= T ? (1: Speech, 0: Silence) --> raw_decisions
         │
         ├──> [postprocess.py: filter_short_silence()]        --> Lọc khoảng lặng < 200ms
         │
         └──> [postprocess.py: extract_boundaries_and_segments()] --> Trích xuất mốc biên
         │
         ▼
Đối tượng SegmentationResult (Biên dự đoán + Nhãn sau lọc)
```
