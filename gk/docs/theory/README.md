# Cơ Sở Lý Thuyết & Thuật Toán Phân Đoạn Tiếng Nói / Khoảng Lặng (VAD)

Tài liệu này trình bày chi tiết bản chất toán học, thuật toán và phân tích kỹ thuật cho 03 phương pháp xác định ngưỡng phân đoạn trong bài toán Voice Activity Detection (VAD).

---

## 1. Bản Chất Bài Toán & Hàm Mục Tiêu

Cho tín hiệu âm thanh rời rạc $x[k]$ đã được phân thành chuỗi $K$ khung thời gian với hàm đặc trưng năng lượng chuẩn hóa $E[n] \in [0, 1]$. Bài toán cần tìm ngưỡng phân tách $\theta \in (0, 1)$ sao cho:
$$\hat{y}[n] = \begin{cases} 1 \text{ (Speech)} & \text{khi } E[n] \ge \theta \\ 0 \text{ (Silence)} & \text{khi } E[n] < \theta \end{cases}$$

---

## 2. Chi Tiết 3 Thuật Toán Xác Định Ngưỡng

### 2.1. Thuật Toán 1: Tìm Kiếm Nhị Phân (Binary Search)
- **Bản chất:** Có giám sát (Supervised Learning). Sử dụng nhãn Ground Truth từ các file `.lab` của tập huấn luyện.
- **Không gian tìm kiếm:** Dải ngưỡng $T \in [T_{min}, T_{max}]$, khởi tạo $[0.0, 1.0]$.
- **Hàm mất mát (Loss Function):** Tổng sai số chênh lệch mốc thời gian biên hoặc tỉ lệ phân loại sai khung (Frame Error Rate - FER) trên tập Train:
  $$\mathcal{L}(\theta) = \frac{1}{K} \sum_{n=1}^{K} |\hat{y}_\theta[n] - y_{true}[n]|$$
- **Cơ chế hoạt động:**
  1. Với bước lặp $i$, xét điểm chia giữa $T_{mid} = \frac{T_{left} + T_{right}}{2}$.
  2. Đo độ dốc hoặc đánh giá sai số tại 2 điểm lân cận $T_{mid} - \Delta$ và $T_{mid} + \Delta$.
  3. Thu hẹp không gian tìm kiếm về phía có sai số nhỏ hơn cho đến khi đạt độ chính xác $|T_{right} - T_{left}| < \epsilon$ hoặc sau $N_{iter}$ bước.
  4. Lấy ngưỡng $\theta_{binary}^*$ tốt nhất trên cả 4 file huấn luyện.

---

### 2.2. Thuật Toán 2: Dựa Trên Biểu Đồ Tần Suất (Histogram Analysis)
- **Bản chất:** Không giám sát (Unsupervised Learning). Hoàn toàn không cần nhãn ground truth.
- **Cơ sở khoa học:** Năng lượng của tín hiệu tiếng nói thực tế thường có phân bố **song đỉnh (Bimodal Distribution)**:
  - **Đỉnh 1 (Năng lượng thấp):** Đại diện cho các khung khoảng lặng và nhiễu nền môi trường.
  - **Đỉnh 2 (Năng lượng cao):** Đại diện cho các khung tiếng nói (nguyên âm, phụ âm hữu thanh).
- **Thuật toán tìm điểm lõm (Valley-Finding / Minimum Searching):**
  1. Lập biểu đồ tần suất gồm $B$ khoảng chia (ví dụ $B = 100$ bins) từ các giá trị $E[n]$:
     $$H(b) = \text{Histogram}(E[n]), \quad b = 1, \dots, B$$
  2. Áp dụng bộ lọc trung bình trượt (Moving Average Filter) với kích thước cửa sổ $W = 5$ để làm mịn biểu đồ, loại bỏ nhiễu răng cưa cục bộ:
     $$H_{smooth}(b) = \frac{1}{W} \sum_{j=-w}^{w} H(b+j)$$
  3. Xác định vị trí 2 đỉnh cục bộ lớn nhất: $\text{Peak}_1$ (nhiễu/silence) và $\text{Peak}_2$ (tiếng nói), với $\text{Bin}(\text{Peak}_1) < \text{Bin}(\text{Peak}_2)$.
  4. Tìm điểm cực tiểu cục bộ (thung lũng) nằm giữa 2 đỉnh này:
     $$b_{threshold} = \arg\min_{b \in [\text{Peak}_1, \text{Peak}_2]} H_{smooth}(b)$$
  5. Giá trị năng lượng tại bin $b_{threshold}$ chính là ngưỡng phân đoạn $\theta_{hist}$.

---

### 2.3. Thuật Toán 3: Thống Kê Phân Phối Chuẩn (Gaussian Statistics)
- **Bản chất:** Có giám sát (Parametric Supervised Learning).
- **Cơ sở khoa học:** Giả định năng lượng ngắn hạn của khung khoảng lặng và khung tiếng nói tuân theo hai phân phối chuẩn độc lập:
  $$p(E \mid \text{Silence}) \sim \mathcal{N}(\mu_{sil}, \sigma_{sil}^2)$$
  $$p(E \mid \text{Speech}) \sim \mathcal{N}(\mu_{sp}, \sigma_{sp}^2)$$
- **Quy trình ước lượng tham số (Parameter Estimation):**
  1. Dựa vào nhãn thời gian trong file `.lab` của 4 file huấn luyện, trích xuất:
     - Tập năng lượng của tất cả các khung silence $\rightarrow$ tính $(\mu_{sil}, \sigma_{sil})$.
     - Tập năng lượng của tất cả các khung speech $\rightarrow$ tính $(\mu_{sp}, \sigma_{sp})$.
  2. **Xác định ngưỡng giao thoa (Decision Boundary):**
     - *Cách 1: Quy tắc khoảng cách độ lệch chuẩn tương đương (Equal-std Distance Rule):*
       Tìm $\theta$ sao cho khoảng cách tính bằng số độ lệch chuẩn từ $\theta$ đến hai kỳ vọng là bằng nhau:
       $$\frac{\theta - \mu_{sil}}{\sigma_{sil}} = \frac{\mu_{sp} - \theta}{\sigma_{sp}} \implies \theta_{gauss} = \frac{\mu_{sil}\sigma_{sp} + \mu_{sp}\sigma_{sil}}{\sigma_{sil} + \sigma_{sp}}$$
     - *Cách 2: Giao điểm phân phối mật độ xác suất (Bayes Decision Threshold):*
       Giải phương trình $p(E \mid \text{Silence}) = p(E \mid \text{Speech})$.

---

## 3. Thuật Toán Hậu Xử Lý (Post-Processing Rule: Lọc Khoảng Lặng < 200 ms)

Trong quá trình phát âm, giữa các từ hoặc trong các phụ âm tắc (plosives như /p/, /t/, /k/) thường có những khoảng ngắt hơi rất ngắn ($20 - 80\text{ ms}$). Việc phân đoạn máy móc sẽ biến những đoạn này thành khoảng lặng giả định ("ảo").

- **Quy tắc bắt buộc từ đề bài:**
  Một khoảng lặng hợp lệ giữa hai câu nói phải kéo dài **tối thiểu $200\text{ ms}$**.
- **Thuật toán xử lý mảng nhị phân:**
  1. Xác định tất cả các đoạn liên tiếp mà $\hat{y}[n] == 0$ (Silence).
  2. Tính độ dài thời gian của đoạn: $\Delta t = \text{count} \times T_{shift}$.
  3. Nếu $\Delta t < 0.2\text{ giây}$ ($200\text{ ms}$), chuyển toàn bộ nhãn của đoạn này từ $0$ thành $1$ (Speech).

---

## 4. Công Thức Đánh Giá Sai Số Định Lượng (MAE & RMSE)

Để đánh giá mức độ chính xác của các đường biên phát hiện được so với Ground Truth:

Cho tập biên chuẩn Ground Truth $\mathcal{B}_{gt} = \{t_{gt}^{(1)}, t_{gt}^{(2)}, \dots, t_{gt}^{(M)}\}$ và tập biên tìm được $\mathcal{B}_{pred} = \{t_{pred}^{(1)}, t_{pred}^{(2)}, \dots, t_{pred}^{(M)}\}$.

- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{M} \sum_{i=1}^{M} |t_{pred}^{(i)} - t_{gt}^{(i)}| \times 1000 \quad (\text{ms})$$
- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{M} \sum_{i=1}^{M} (t_{pred}^{(i)} - t_{gt}^{(i)})^2} \times 1000 \quad (\text{ms})$$

---

## 5. Phân Tích Môi Trường Thu Âm & Tỉ Số Tín Hiệu Trên Nhiễu (SNR)

Dữ liệu thực nghiệm gồm 2 môi trường đặc thù:
1. **Môi trường Studio (`studio_*.wav`):** Phòng thu cách âm chuyên nghiệp, tỉ số SNR cao, nền nhiễu rất thấp. Biên độ và năng lượng phân tách cực kỳ rõ ràng giữa Speech và Silence $\rightarrow$ Sai số MAE/RMSE thường $< 20\text{ ms}$.
2. **Môi trường Điện thoại (`phone_*.wav`):** Băng thông bị bóp hẹp (thường $8\text{ kHz}$ hoặc nhiễu đường truyền), SNR thấp, nền nhiễu dao động lớn. Các âm vô thanh năng lượng thấp dễ bị chìm vào nhiễu $\rightarrow$ Dẫn đến sai lệch biên ở đầu hoặc đuôi từ phát âm.
