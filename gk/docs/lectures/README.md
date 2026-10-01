# Tổng Hợp Lý Thuyết: Chapter 6 - Speech Signal Processing

Tài liệu này đúc kết toàn bộ kiến thức cốt lõi từ slide bài giảng chính khóa **Chapter 6: Speech Signal Processing** của Giảng viên **Ninh Khánh Duy** (ĐH Bách Khoa / ĐH Quốc Gia).

---

## 1. Cơ Chế Phát Âm & Phân Loại Vùng Tín Hiệu

Tín hiệu tiếng nói trong tự nhiên được tạo ra từ luồng không khí đi từ phổi qua dây thanh quản và khoang cộng hưởng (họng, miệng, mũi):

```
Luồng khí từ phổi ---> Dây thanh âm (Vocal cords) ---> Các khoang cộng hưởng ---> Sóng âm thanh
```

Trong một file âm thanh thu âm, tín hiệu được phân chia thành 2 vùng chính:
1. **Khoảng lặng (Silence):** Vùng không có hoạt động phát âm của con người, chỉ chứa tiếng ồn môi trường (nhiễu nền).
2. **Tiếng nói (Speech):** Vùng có hoạt động phát âm các âm vị (phones), tiếp tục được chia thành:
   - **Âm hữu thanh (Voiced):** Dây thanh âm rung đều đặn $\rightarrow$ tín hiệu có tính tuần hoàn rất mạnh, năng lượng cao, chứa tần số cơ bản $F_0$ (Pitch). Ví dụ: các nguyên âm /a/, /e/, /i/, /o/, /u/.
   - **Âm vô thanh (Unvoiced):** Dây thanh âm không rung, luồng khí ma sát qua khe hẹp $\rightarrow$ tín hiệu giả ngẫu nhiên giống nhiễu trắng, năng lượng thấp, tập trung ở tần số cao. Ví dụ: các phụ âm xát /s/, /f/, /t/, /k/.

> [!IMPORTANT]
> **Ràng buộc bài toán VAD:** Theo yêu cầu đề bài, chúng ta **gộp cả âm hữu thanh (v) và âm vô thanh (uv) thành Tiếng Nói (Speech)**. Mục tiêu duy nhất là phân tách nhị phân giữa **Speech (1)** và **Silence (0)**.

---

## 2. Kỹ Thuật Chia Khung Ngắn Hạn (Framing & Windowing)

Tín hiệu tiếng nói mang tính bất dừng (non-stationary) trên toàn cục, nhưng có thể coi là **dừng cục bộ (quasi-stationary)** trong các khoảng thời gian rất ngắn ($10 - 30\text{ ms}$).

### 2.1. Tham số phân khung
- **Độ dài khung (Frame Length - $N$):** Thường chọn từ $20\text{ ms} - 30\text{ ms}$.
  $$N = \text{round}(T_{frame} \times F_s)$$
  *(Ví dụ: với tần số lấy mẫu $F_s = 16000\text{ Hz}$ và $T_{frame} = 20\text{ ms}$, thì $N = 320$ mẫu).*
- **Độ dịch khung (Hop Size / Frame Shift - $M$):** Thường chọn bằng $1/2$ hoặc $1/3$ chiều dài khung (ví dụ $10\text{ ms} = 160$ mẫu) để đảm bảo chuyển tiếp mượt mà.

### 2.2. Hàm cửa sổ (Window Functions)
Để giảm thiểu hiện tượng rò rỉ phổ (spectral leakage) và hiện tượng gián đoạn ở hai đầu khung:
- **Cửa sổ chữ nhật (Rectangular Window):** $w(m) = 1$ với $0 \le m \le N-1$.
- **Cửa sổ Hamming:** $w(m) = 0.54 - 0.46 \cos\left(\frac{2\pi m}{N-1}\right)$ với $0 \le m \le N-1$.

---

## 3. Các Hàm Đặc Trưng Miền Thời Gian (Time-Domain Features)

### 3.1. Năng Lượng Ngắn Hạn (Short-Time Energy - STE)
Đo tổng bình phương biên độ tín hiệu trong một khung:
$$E_n = \sum_{m=0}^{N-1} [x(n \cdot M + m) \cdot w(m)]^2$$
- **Đặc điểm:** Tăng cường mạnh các giá trị biên độ lớn. Năng lượng của tiếng nói (đặc biệt là âm hữu thanh) lớn hơn rất nhiều so với khoảng lặng/nhiễu nền.

### 3.2. Độ Lớn Trung Bình Ngắn Hạn (Short-Time Magnitude Average - MA)
Đo tổng giá trị tuyệt đối biên độ tín hiệu trong một khung:
$$MA_n = \sum_{m=0}^{N-1} |x(n \cdot M + m) \cdot w(m)|$$
- **Đặc điểm:** Không bình phương biên độ nên bớt nhạy cảm với các đỉnh biên độ đột biến, phản ánh mượt mà đường bao (envelope) của tín hiệu.

### 3.3. Chuẩn Hóa Vector Đặc Trưng (Feature Normalization)
Để đảm bảo ngưỡng phân đoạn có thể áp dụng nhất quán giữa các file khác nhau:
- **Min-Max Normalization về $[0, 1]$:**
  $$E_{norm}[n] = \frac{E[n] - \min(E)}{\max(E) - \min(E)}$$
- Hoặc biểu diễn trên thang logarit (dB) để nén dải động:
  $$E_{log}[n] = 10 \log_{10}(E[n] + \epsilon)$$
