# Quy tắc biến đổi đồ thị trên trục thời gian (Time Transformations in DSP)

---

## 1. Nguyên lý cốt lõi: Quy tắc "Ngược" (Inverse Rule)

Mọi phép toán tác động trực tiếp vào biến thời gian $n$ **BÊN TRONG DẤU NGOẶC** của $x[\dots]$ đều làm đồ thị biến đổi **NGƯỢC CHIỀU / NGHỊCH ĐẢO** trên trục hoành (trục thời gian):

* **Cộng ($+k$)** $\implies$ Dịch sang **TRÁI** (đến sớm).
* **Trừ ($-k$)** $\implies$ Dịch sang **PHẢI** (bị trễ).
* **Nhân ($\times a$)** $\implies$ **CO HẸP LẠI** (vứt bớt mẫu, $f_s$ giảm).
* **Chia ($\div a$)** $\implies$ **DÃN RỘNG RA** (thiếu mẫu, phải nội suy, $f_s$ tăng).

---

## 2. Các phép biến đổi cơ bản (Đơn lẻ)

| Phép biến đổi | Biểu thức | Tác động lên hình ảnh | Ý nghĩa vật lý / Ứng dụng |
| :--- | :---: | :--- | :--- |
| **Dịch trễ (Delay)** | $x[n - k]$<br>$(k > 0)$ | Dịch sang **PHẢI** $k$ đơn vị | Tín hiệu phát chậm đi $k$ mẫu |
| **Dịch sớm (Advance)** | $x[n + k]$<br>$(k > 0)$ | Dịch sang **TRÁI** $k$ đơn vị | Tín hiệu xuất hiện sớm hơn $k$ mẫu |
| **Đảo thời gian (Reversal)** | $x[-n]$ | **Lật đối xứng** qua trục tung $n = 0$ | Phát ngược băng (từ đuôi về đầu) |
| **Nén thời gian (Decimation)** | $x[a \cdot n]$<br>$(a > 1)$ | **Co lại** $a$ lần về phía gốc $0$ | Giảm $f_s$ đi $a$ lần (Downsampling) |
| **Dãn thời gian (Interpolation)** | $x\left[\dfrac{n}{a}\right]$<br>$(a > 1)$ | **Dãn ra** $a$ lần xa gốc $0$ | Tăng $f_s$ lên $a$ lần (Upsampling) |

> **Lưu ý:** Nếu có hệ số nhân bên ngoài $A \cdot x[n]$ (ví dụ $2 \cdot x[n]$), đây là thay đổi **biên độ** theo trục tung ($y$), không làm thay đổi trục thời gian ($n$).

---

## 3. Phép biến đổi tổ hợp: $y[n] = x[a \cdot n + b]$

Đây là dạng toán phổ biến nhất nhưng rất dễ nhầm lẫn thứ tự thực hiện giữa phép dịch và phép co/dãn.

### Cách 1: Đưa về dạng chuẩn nhân tử (Khuyên dùng khi vẽ hình)
Biến đổi biểu thức về dạng:
$$y[n] = x\left[ a \left( n + \frac{b}{a} \right) \right]$$

Thực hiện tuần tự 2 bước:
1. **Bước 1 (Dịch):** Dịch đồ thị $x[n]$ một đoạn $\dfrac{b}{a}$:
   - Sang **TRÁI** nếu $\dfrac{b}{a} > 0$.
   - Sang **PHẢI** nếu $\dfrac{b}{a} < 0$.
2. **Bước 2 (Co/Dãn & Lật):** 
   - Co lại $|a|$ lần nếu $|a| > 1$ (hoặc dãn ra nếu $|a| < 1$).
   - Nếu $a < 0$, lật ngược hình qua trục tung $n = 0$.

---

### Cách 2: "Mẹo tìm điểm mốc" (Nhanh nhất cho trắc nghiệm)
Không cần suy nghĩ thứ tự dịch hay co, chỉ cần tìm xem **các điểm mốc đặc trưng của hình gốc sẽ bay về tọa độ nào trên hình mới**:

> Muốn biết một mốc $n_{\text{cũ}}$ chuyển thành $n_{\text{mới}}$, giải phương trình:
> $$a \cdot n_{\text{mới}} + b = n_{\text{cũ}} \implies \mathbf{n_{\text{mới}} = \frac{n_{\text{cũ}} - b}{a}}$$

#### Ví dụ minh họa:
Cho tín hiệu $x[n]$ chỉ tồn tại khác không trong đoạn $[-2, 4]$. Hãy xác định khoảng tồn tại của $y[n] = x[-2n + 4]$:
- Với điểm đầu $n_{\text{cũ}} = -2$:
  $$-2n_{\text{mới}} + 4 = -2 \implies -2n_{\text{mới}} = -6 \implies n_{\text{mới}} = 3$$
- Với điểm cuối $n_{\text{cũ}} = 4$:
  $$-2n_{\text{mới}} + 4 = 4 \implies -2n_{\text{mới}} = 0 \implies n_{\text{mới}} = 0$$

👉 **Kết luận:** Tín hiệu $y[n]$ bị đảo chiều, co lại một nửa và chỉ tồn tại trong đoạn $[0, 3]$.

---

## 4. Mối liên hệ với Tần số lấy mẫu ($f_s$) trong Xử lý âm thanh

1. **Khi áp dụng $y[n] = x[2n]$ (Downsampling):**
   - Chỉ giữ lại các mẫu chỉ số chẵn $0, 2, 4, 6\dots$, vứt bỏ các mẫu lẻ $1, 3, 5\dots$
   - Khoảng cách thời gian giữa 2 mẫu liên tiếp tăng gấp đôi: $T' = 2T$.
   - Tần số lấy mẫu giảm đi một nửa: $f_s' = \dfrac{f_s}{2}$.
   - **Ảnh hưởng âm thanh:** Mất dải tần số cao, có thể xuất hiện méo chồng phổ (aliasing) nếu không lọc thông thấp trước.

2. **Khi áp dụng nội suy để tăng $f_s$ lên 2 lần (Upsampling):**
   - Chèn thêm các điểm trung bình cộng $y[2k+1] = \dfrac{1}{2}(x[k] + x[k+1])$ vào giữa các mẫu cũ.
   - Khoảng cách thời gian giữa 2 mẫu giảm một nửa: $T' = \dfrac{T}{2}$.
   - Tần số lấy mẫu tăng gấp đôi: $f_s' = 2 f_s$.
   - **Ảnh hưởng âm thanh:** Làm mịn dạng sóng, giảm tiếng chói gắt do răng cưa thô, nhưng không thể phục hồi dải tần cao đã mất từ trước.

---

## 5. Bảng tra cứu nhanh (Cheat Sheet)

```text
               TÍN HIỆU GỐC: x[n]
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   CỘNG / TRỪ        DẤU TRỪ      NHÂN / CHIA
  (Dịch hình)       (Lật hình)    (Co / Dãn hình)
       │               │               │
  x[n - k]: PHẢI     x[-n]: Lật    x[a*n]  : CO LẠI (a > 1)
  x[n + k]: TRÁI     qua n = 0     x[n / a]: DÃN RA (a > 1)
```
