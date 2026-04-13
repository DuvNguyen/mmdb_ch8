# Chương 8: Hệ quản trị cơ sở dữ liệu Âm thanh (Audio DB)
## Trích xuất Đặc trưng và Truy vấn Âm thanh

### 1. Bài toán chi tiết
Âm thanh là tín hiệu biến thiên theo thời gian. Để quản lý hiệu quả, ta không thể tìm kiếm trên dữ liệu thô (raw data) mà phải trích xuất các đặc trưng toán học biểu diễn nội dung âm thanh.

Yêu cầu bài tập:
- Phân đoạn audio thành các cửa sổ (windows).
- Trích xuất 4 đặc trưng: Cường độ (Intensity), Độ ồn (Loudness), Cao độ (Pitch), Độ sáng (Brightness).
- Chỉ mục hóa bằng **KD-Tree** để tìm K âm thanh giống nhất.

### 2. Các thành phần chính

#### a. Phân đoạn (Segmentation)
Chia file âm thanh thành các khung hình (frames) hoặc cửa sổ (windows) có kích thước cố định (ví dụ: 20ms - 50ms) để tính toán đặc trưng cục bộ.

#### b. Trích xuất đặc trưng (Feature Extraction)
1. **Intensity (I)**: Năng lượng trung bình của tín hiệu trong window.
2. **Loudness (L)**: Cường độ âm thanh cảm nhận được (thường là Log của cường độ).
3. **Pitch (P)**: Tần số cơ bản của tín hiệu (Fundamental Frequency). Được tính bằng phép tự tương quan (autocorrelation) hoặc FFT.
4. **Brightness (B)**: Đặc trưng cho độ "sáng" của âm sắc. Được tính bằng trọng tâm phổ (spectral centroid).

#### c. Chỉ mục KD-Tree
- Mỗi window được biểu diễn bằng một vector đặc trưng trong không gian N chiều.
- KD-Tree giúp tìm kiếm láng giềng gần nhất (Nearest Neighbor) cực nhanh, cho phép tìm các đoạn âm thanh có đặc trưng tương đồng nhất với mẫu truy vấn.

### 3. Phân biệt giọng Nam/Nữ
- **Giọng Nam**: Thường có Pitch thấp (80Hz - 180Hz).
- **Giọng Nữ**: Thường có Pitch cao (165Hz - 255Hz).
Dựa vào phân phối Pitch trong các vector trích xuất, hệ thống có thể phân loại giới tính của người nói.

### 4. Hướng dẫn chạy Demo
1. Mở thư mục `audio_db_system`.
2. Chạy lệnh: `python main.py`.
3. Giao diện hiện ra cho phép:
   - Load file `.wav` hoặc tạo âm thanh tổng hợp.
   - Trích xuất đặc trưng và xây dựng cây KD-Tree.
   - Nhập một đoạn âm thanh mẫu để tìm K đoạn tương đồng nhất trong database.
   - Xem biểu đồ trực quan hóa Pitch và Intensity.
