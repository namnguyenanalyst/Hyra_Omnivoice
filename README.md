# Hyra OmniVoice 🌍

Đây là phiên bản tùy chỉnh của mô hình **OmniVoice** - một mô hình Text-To-Speech (TTS) đa ngôn ngữ hỗ trợ hơn 600 ngôn ngữ, được thiết kế cho mục đích tối ưu hóa và triển khai dễ dàng hơn.

## 🚀 Tính Năng Chính

- **Hỗ trợ 600+ Ngôn ngữ:** Độ phủ ngôn ngữ lớn nhất trong các mô hình Zero-shot TTS.
- **Voice Cloning:** Sao chép giọng nói chất lượng cao chỉ với một đoạn âm thanh mẫu (3-10 giây).
- **Voice Design:** Tạo giọng nói theo các thuộc tính tùy chỉnh (giới tính, tuổi tác, cao độ, giọng vùng miền...).
- **Tốc độ cực nhanh:** Hỗ trợ suy luận với tốc độ nhanh gấp 40 lần thời gian thực (RTF 0.025).

## 🛠 Cài Đặt (Sử dụng GPU)

Dự án này đã được cấu hình sẵn `requirements.txt` để hỗ trợ cài đặt nhanh chóng (tương thích sẵn với hệ điều hành Linux sử dụng NVIDIA GPU có CUDA 12.8).

**Bước 1:** Khuyến nghị tạo một môi trường ảo (Virtual Environment hoặc Conda) trước khi cài đặt.

**Bước 2:** Chạy lệnh cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 🎮 Hướng Dẫn Sử Dụng Nhanh

Sau khi cài đặt xong, bạn có thể mở giao diện Web UI (Gradio) để trải nghiệm trực tiếp Voice Cloning và Voice Design:

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

Truy cập vào trình duyệt của bạn với địa chỉ `http://localhost:8001` (hoặc IP của máy chủ) để sử dụng.

## 💻 Sử dụng bằng Code Python

Bạn có thể tích hợp **Hyra OmniVoice** vào mã nguồn Python của bạn một cách dễ dàng. 

### 1. Voice Cloning (Sao chép giọng nói)
```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

# Tải mô hình
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)

# Thực hiện Voice Cloning
audio = model.generate(
    text="Xin chào, đây là hệ thống sao chép giọng nói của Hyra OmniVoice.",
    ref_audio="duong_dan_den_file_am_thanh_mau.wav",
) 

# Lưu file kết quả
sf.write("output.wav", audio[0], 24000)
```

### 2. Voice Design (Thiết kế giọng nói mới)
```python
# Tạo giọng đọc dựa trên mô tả (không cần file âm thanh mẫu)
audio = model.generate(
    text="Xin chào, đây là một ví dụ về thiết kế giọng nói.",
    instruct="female, high pitch, vietnamese",
)

sf.write("output_design.wav", audio[0], 24000)
```

---
*Dự án được phân nhánh và tùy biến từ mã nguồn gốc của [OmniVoice](https://github.com/k2-fsa/OmniVoice).*
