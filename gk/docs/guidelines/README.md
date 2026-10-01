# Quy Chế Báo Cáo, Thi & Tiêu Chuẩn Nộp Bài

Tài liệu này tổng hợp toàn bộ các quy định ngặt nghèo của môn học Xử lý tín hiệu số (DSP) liên quan đến phần báo cáo giữa kỳ, thuyết trình slide, demo chương trình và cấu trúc nộp bài.

---

## 1. Thời Gian & Kỹ Năng Thuyết Trình

- **Tổng thời lượng:** Đúng **4 phút / sinh viên** (Bấm giờ tự động, quá giờ sẽ bị cắt phần trình bày).
  - **3 phút:** Thuyết trình slide.
  - **1 phút:** Chạy demo chương trình trực tiếp trên máy tính.
- **Nội dung trọng tâm:**
  - **KHÔNG** nói lý thuyết suông hoặc giải thích công thức (giảng viên đã dạy trên lớp).
  - **TẬP TRUNG** vào sơ đồ khối giải pháp, phương pháp tìm ngưỡng, các đồ thị minh họa và bình luận so sánh kết quả thực nghiệm.
  - Phân tích rõ ràng: *đúng/sai ở đoạn nào, vì sao sai (ví dụ nhiễu SNR thấp ở môi trường điện thoại, mất phụ âm xát vô thanh,...)*.

---

## 2. Tiêu Chuẩn Thiết Kế Slide (Bắt Buộc)

- **Cấu trúc bài thuyết trình:**
  - **Slide 1:** Slide bìa (Cover) — Tên đề tài, Họ tên SV, Mã SV.
  - **Slide 2:** Sơ đồ khối giải pháp đề xuất (5 khối chuẩn).
  - **Slide 3:** Phương pháp xác định ngưỡng tối ưu trên 4 file huấn luyện (`TinHieuHuanLuyen`).
  - **Slide 4–5:** Đồ thị và kết quả thực nghiệm trên 4 file kiểm thử (`TinHieuKiemThu`).
  - **Slide 6:** Bảng sai số định lượng (MAE / RMSE) và so sánh giữa 2 môi trường (Studio vs Phone).
- **Quy tắc "<= 7 hàng, <= 10 từ":**
  - Số dòng trên 1 slide: $\le 7$ hàng/slide.
  - Số chữ trên 1 dòng: $\le 10$ từ/hàng.
  - Cỡ chữ: $\ge 18\text{ pt}$.
  - Màu chữ tương phản mạnh với nền (khuyến nghị chữ đen/xanh đậm trên nền trắng).
  - Hình vẽ và biểu đồ phải **to, sắc nét, có chú giải rõ ràng**.

---

## 3. Quy Chuẩn Mã Nguồn & Cách Demo

- **Quy tắc cấm Toolbox:**
  - Sinh viên phải **tự code hoàn toàn** các hàm tính toán xử lý tín hiệu.
  - Tuyệt đối **không dùng** các toolbox xử lý tín hiệu cấp cao (ví dụ: cấm dùng hàm tính STE, VAD có sẵn của các thư viện bên ngoài).
  - Chỉ được dùng hàm cơ bản: `numpy` (min, max, mean, std, sum, abs), `scipy.io.wavfile` để đọc ghi wav, `matplotlib` để vẽ đồ thị.
- **Tổ chức code:**
  - Khởi chạy duy nhất từ `main.py` (hoặc `main.m`).
  - Mỗi hàm phải có docstring rõ ràng mô tả chức năng, tham số đầu vào và đầu ra.
  - Từng block code $5 - 10$ dòng phải có chú thích (comment) giải thích logic.
- **Cơ chế Demo:**
  - Bấm Run `main.py` **đúng 1 lần duy nhất**.
  - Chương trình tự động chạy trên 4 file trong `TinHieuKiemThu` (`phone_F2`, `phone_M2`, `studio_F2`, `studio_M2`).
  - Xuất ra đồng thời **4 cửa sổ figure** và tự động ghim tại **4 góc của màn hình**:
    - Góc trên - trái: `phone_F2`
    - Góc trên - phải: `phone_M2`
    - Góc dưới - trái: `studio_F2`
    - Góc dưới - phải: `studio_M2`
  - Mỗi plot trên figure phải có: Title, nhãn trục X (thời gian tính bằng giây), nhãn trục Y, đường biên màu đỏ (Ground Truth) và đường biên màu xanh (Thuật toán tìm được).

---

## 4. Quy Cách Đóng Gói Nộp Bài

- Thư mục nộp bài đặt tên: `MaTheSV-HoTen` (Ví dụ: `21120000-NguyenVanA`).
- Bên trong thư mục chỉ chứa:
  - File PDF của slide thuyết trình.
  - Toàn bộ mã nguồn chương trình (`main.py`, `src/`, `tests/`).
- **Lưu ý:** Không đính kèm thư mục dữ liệu âm thanh (`TinHieuHuanLuyen`, `TinHieuKiemThu`) khi nộp bài để tiết kiệm dung lượng.
