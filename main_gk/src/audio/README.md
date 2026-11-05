# Module: `src/audio`
Quản lý nạp dữ liệu âm thanh và xử lý nhãn chuẩn Ground Truth.

---

## 1. Các tệp thành phần
* **`loader.py`**: Đọc file âm thanh `.wav` (sử dụng `scipy.io.wavfile`), chuẩn hóa biên độ về khoảng $[-1.0, 1.0]$, chuyển đổi kênh âm thanh về Mono nếu là tín hiệu đa kênh.
* **`labels.py`**: Phân tích cú pháp tệp nhãn `.lab` đi kèm:
  - Gom các nhãn `v` (voiced) và `uv` (unvoiced) thành `speech`, `sil` thành `silence`.
  - Trích xuất danh sách các đoạn tiếng nói (`gt_segments`), các mốc biên chuẩn (`gt_boundaries`) và thông số $F_0$.
  - Hàm `create_frame_labels`: Gán nhãn nhị phân ($0$: silence, $1$: speech) cho từng khung thời gian dựa trên vị trí tâm khung.
* **`dataset.py`**: Quét thư mục dữ liệu (`TinHieuHuanLuyen` hoặc `TinHieuKiemThu`), tự động ghép cặp `.wav` với `.lab` tương ứng và khởi tạo danh sách đối tượng `AudioSignal`.

---

## 2. Quy trình luồng làm việc (Pipeline)

```
Thư mục (*.wav + *.lab)
         │
         ▼
[dataset.py: load_dataset()]
         │
         ├──> [loader.py: load_audio()]   --> Đọc tín hiệu, lấy tần số lấy mẫu fs
         └──> [labels.py: parse_lab()]    --> Đọc nhãn, tách mốc biên GT
         │
         ▼
Danh sách đối tượng AudioSignal
```
