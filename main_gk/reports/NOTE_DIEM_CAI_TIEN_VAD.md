## 1. Hiện tượng quan sát được (Observation)

Khi áp dụng thuật toán phân đoạn (đặc biệt là Binary Search và Gaussian Model) trên tập kiểm thử `TinHieuKiemThu`:

- **Môi trường Phòng thu (Studio)**: Tỷ số tín hiệu trên nhiễu cao ($SNR \approx 38 - 49\text{ dB}$).
  - `studio_F2`: $2$ biên dự đoán / $2$ biên chuẩn (khớp $100\%$, $\text{MAE} = 0.0\text{ ms}$).
  - `studio_M2`: $2$ biên dự đoán / $2$ biên chuẩn (khớp $100\%$, $\text{MAE} = 10.0\text{ ms}$).
- **Môi trường Điện thoại (Phone)**: Tỷ số tín hiệu trên nhiễu thấp ($SNR \approx 24 - 27\text{ dB}$).
  - `phone_M2`: Xuất hiện **$4$ biên dự đoán** (dư $2$ biên so với chuẩn $2$ biên GT).
  - `phone_F2`: Xuất hiện **$5$ biên dự đoán** (dư $3$ biên so với chuẩn $2$ biên GT).

---

## 2. Phân tích nguyên nhân gốc rễ (Root Cause Analysis)

### A. Về mặt kỹ thuật theo quy định đề bài

- Đề bài yêu cầu: _"Độ dài tối thiểu của 1 khoảng lặng là 200 ms (dùng điều kiện này để loại bỏ các khoảng lặng 'ảo' có chiều dài quá ngắn)"_.
- Thuật toán hiện tại tuân thủ nghiêm ngặt quy tắc này: Nếu một khoảng **LẶNG** $< 200\text{ ms}$ bị kẹp giữa tiếng nói, nó sẽ được gộp thành tiếng nói.

### B. Về mặt âm học và đặc tính kênh truyền thoại (Acoustic Phonetics & Channel Noise)

- Kênh thoại analog/di động luôn đi kèm **xung nhiễu đường truyền (click noise), tiếng rơ-le ngắt mở mạch, tiếng thở mạnh trước khi nói hoặc tiếng cọ xát micro**:
  1. **File `phone_M2`**:
     - Chuẩn Ground Truth: Người nói từ giây $0.53\text{s} \rightarrow 2.52\text{s}$.
     - Thực tế tín hiệu: Tại giây **$0.10\text{s} \rightarrow 0.11\text{s}$**, xuất hiện một xung nhiễu đường truyền có năng lượng vượt ngưỡng $T$, nhưng chỉ kéo dài vỏn vẹn **$10\text{ ms}$ (đúng 1 khung tín hiệu)**.
     - Hệ quả: Thuật toán nhận diện $0.10\text{s}$ là bắt đầu nói và $0.11\text{s}$ là hết nói $\rightarrow$ sinh ra $2$ vạch biên xanh giả.
  2. **File `phone_F2`**:
     - Chuẩn Ground Truth: Người nói từ giây $1.02\text{s} \rightarrow 4.04\text{s}$.
     - Thực tế tín hiệu:
       - Tại giây **$0.62\text{s} \rightarrow 0.64\text{s}$**: Tiếng thở/click trước câu nói kéo dài **$20\text{ ms}$** $\rightarrow$ sinh ra $2$ vạch xanh giả ở $0.62\text{s}$ và $0.64\text{s}$.
       - Tại giây **$4.50\text{s} \rightarrow 4.78\text{s}$**: Tiếng thở ra / cọ xát mic sau khi nói xong kéo dài $280\text{ ms}$ $\rightarrow$ sinh ra thêm $1$ vạch xanh ở $4.50\text{s}$.
- **Điểm cốt lõi**: Các xung nhiễu này là **KHOẢNG TIẾNG NÓI ẢO (Spurious Speech Bursts)** nằm lọt thỏm giữa vùng khoảng lặng mênh mông. Bộ lọc của đề bài chỉ khử khoảng lặng ảo nên hoàn toàn không thể chạm tới các xung tiếng nói ảo này.

---

## 3. Đề xuất điểm cải tiến (Proposed Improvement Point)

### Cơ sở lý thuyết âm học (Acoustic Basis)

Trong ngữ âm học thực nghiệm (Acoustic Phonetics), các âm vị ngắn nhất của con người (như âm tắc vỗ, âm bật hơi của phụ âm) cũng cần tối thiểu **$40 - 50\text{ ms}$** để bộ máy phát âm con người hoàn thành chu trình rung/bật thanh đới. Tiếng nói có nghĩa của con người **không thể diễn ra trong vòng $10 - 20\text{ ms}$**.

### Giải pháp đề xuất: Bộ lọc kép thời lượng tối thiểu (Dual-Duration Post-Processing)

1. **Lọc khoảng lặng ảo (Quy định bắt buộc)**:
   - Khoảng lặng $< 200\text{ ms}$ $\rightarrow$ Gộp thành tiếng nói.
2. **Lọc khoảng tiếng nói ảo (Đề xuất cải tiến sáng tạo)**:
   - Bổ sung tham số `min_speech_ms = 50.0 ms`.
   - Nếu một đoạn được phân loại là tiếng nói nhưng có thời lượng $< 50\text{ ms}$ (tức dưới $5$ khung liên tiếp) $\rightarrow$ Khẳng định đó là xung nhiễu/tiếng click đường truyền và chuyển ngược lại thành khoảng lặng.

---

## 4. Minh chứng thực nghiệm đối sánh (Empirical Validation)

Kết quả khi chạy thử nghiệm trên 4 file kiểm thử:

| Tên File    | Môi trường | Ground Truth |      Kết quả Baseline (Hiện tại)       |       Kết quả sau Cải tiến (Đề xuất)       | Đánh giá                            |
| :---------- | :--------: | :----------: | :------------------------------------: | :----------------------------------------: | :---------------------------------- | --- |
| `studio_F2` |   Studio   |    2 biên    | 2 biên ($\text{MAE} = 0.0\text{ ms}$)  | **2 biên** ($\text{MAE} = 0.0\text{ ms}$)  | Không suy giảm độ chính xác         |     |
| `studio_M2` |   Studio   |    2 biên    | 2 biên ($\text{MAE} = 10.0\text{ ms}$) | **2 biên** ($\text{MAE} = 10.0\text{ ms}$) | Giữ nguyên độ chính xác cao         |     |
| `phone_M2`  |   Phone    |    2 biên    | 4 biên ($\text{MAE} = 5.0\text{ ms}$)  | **2 biên** ($\text{MAE} = 5.0\text{ ms}$)  | **Khử sạch 100% biên giả ở 0.10s**  |     |
| `phone_F2`  |   Phone    |    2 biên    | 5 biên ($\text{MAE} = 55.0\text{ ms}$) | **3 biên** ($\text{MAE} = 55.0\text{ ms}$) | Khử được xung nhiễu 20ms ở đầu file |     |
