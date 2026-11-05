# Module: `src/features`
Chia khung tín hiệu ngắn hạn và trích xuất các đặc trưng âm học.

---

## 1. Các tệp thành phần
* **`framing.py`**: Chia tín hiệu âm thanh liên tục thành các khung ngắn hạn (Short-Time Frames):
  - Kích thước khung mặc định: $25\text{ ms}$ (`frame_size_ms`).
  - Bước dịch khung mặc định: $10\text{ ms}$ (`frame_shift_ms`).
  - Áp dụng hàm cửa sổ (`window_type`).
  - Tính toán chính xác vector mốc thời gian tại tâm từng khung (`timestamps`).
  - Trả về đối tượng `FrameSignal`.
* **`extraction.py`**: Trích xuất đặc trưng năng lượng ngắn hạn (Short-Time Energy - STE):
  - Tính năng lượng thô: $STE = \sum_{m} x^2[m]$.
  - Chuẩn hóa Min-Max về đoạn $[0.0, 1.0]$ (`ste_norm`).
  - Đóng gói vào đối tượng `ShortTimeFeatures`.
* **`pitch.py`**: Ước lượng đường cong cao độ $F_0$ (Pitch Tracking):
  - Sử dụng phương pháp hàm tự tương quan (*Autocorrelation*).
  - Tìm đỉnh cực đại trong khoảng trễ tương ứng $[60\text{ Hz}, 400\text{ Hz}]$.
  - Gán `NaN` cho các khung vô thanh hoặc khoảng lặng dựa trên ngưỡng độ tin cậy (*voicing confidence*).

---

## 2. Quy trình luồng làm việc (Pipeline)

```
AudioSignal (Mảng sóng âm + fs)
         │
         ▼
[framing.py: apply_framing()]
         │  --> Chia khung (25ms, bước nhảy 10ms)
         ▼
FrameSignal (Các ma trận khung + timestamps)
         │
         ▼
[extraction.py: extract_features()]
         │  --> Tính STE thô và chuẩn hóa Min-Max
         ▼
ShortTimeFeatures (Mảng năng lượng ste_norm dùng để phân tách VAD)
```
