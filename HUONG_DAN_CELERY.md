# Tổng kết: Chuyển đổi Kiến trúc OmniVoice sang Celery + Redis

Tôi đã hoàn thành việc tái cấu trúc hệ thống của bạn từ mô hình Queue mặc định sang mô hình phân tán mạnh mẽ hơn. Dưới đây là tổng quan những gì đã được thực hiện và hướng dẫn cách bạn khởi động hệ thống mới.

## Các Thay Đổi Chính (What Changed)

1. **Thêm Thư Viện Mới:** Tôi đã chạy `uv add celery redis` để thêm 2 thư viện này vào môi trường dự án.
2. **Cấu Hình Redis (`docker-compose.yml`):** Tạo file để bạn có thể chạy Redis cực kỳ nhanh gọn thông qua Docker.
3. **Tạo Celery Worker (`celery_worker.py`):** 
   - Di chuyển toàn bộ code tải model AI (khá nặng) từ API sang file này.
   - Hàm `process_voice_job` đã được định nghĩa thành một task độc lập nhận việc từ Redis, có cấu hình `worker_prefetch_multiplier=1` giúp đảm bảo Worker chỉ nhận 1 job mỗi lần (bảo vệ GPU khỏi lỗi tràn VRAM).
4. **Viết lại API Server (`api.py`):**
   - Lược bỏ hoàn toàn model AI và background thread cũ. API Server giờ đây cực kỳ nhẹ và sẽ khởi động ngay lập tức.
   - Endpoint `/clone_voice` hiện tại chỉ làm nhiệm vụ tiền xử lý text với từ điển phát âm, đóng gói dữ liệu và đẩy vào Redis qua lệnh `celery_app.send_task()`.

---

## Hướng Dẫn Khởi Động Hệ Thống (How to Run)

Vì chúng ta đã tách hệ thống làm 2 mảnh (Microservices), từ nay bạn sẽ cần khởi động 3 thành phần (tốt nhất là dùng 3 cửa sổ Terminal khác nhau).

> [!IMPORTANT]
> **Khởi động theo đúng thứ tự dưới đây để đảm bảo mọi thứ trơn tru:**

### Bước 1: Khởi động Redis
Mở terminal, cd vào thư mục `OmniVoice` và chạy:
```bash
docker-compose up -d
```
*(Nếu bạn dùng `docker compose` bản mới, bỏ dấu gạch ngang).* Lệnh này sẽ tải và chạy Redis ngầm ở cổng 6379.

### Bước 2: Khởi động Celery Worker (Chứa AI Model)
Trong terminal thứ hai (nhớ activate môi trường `uv` hoặc `venv`), chạy lệnh sau để khởi động Worker:
```bash
python -m celery -A celery_worker worker --loglevel=info -P solo
```
*(Cờ `-P solo` trên Windows/Linux giúp tránh lỗi khi dùng Celery với PyTorch/GPU).* Bạn sẽ thấy Worker in ra log `[Celery] Đang tải model OmniVoice lên cuda...` và thông báo `[Celery] Model đã sẵn sàng!`.

### Bước 3: Khởi động FastAPI Server
Trong terminal thứ ba, chạy API Server y như trước đây:
```bash
uvicorn api:app --host 0.0.0.0 --port 8002
```
Server này giờ đây sẽ khởi động ngay lập tức do không phải tải model AI.

---
Bây giờ bạn có thể thử gọi API `/clone_voice` lại. FastAPI sẽ trả về `job_id` trong chưa tới 0.1 giây, và trên cửa sổ Terminal của Celery Worker bạn sẽ thấy nó tiếp nhận Job và bắt đầu clone giọng nói. Xong xuôi nó sẽ tự gọi lại Webhook như cũ!

Mở dashboard celery bằng flower:
```bash
python -m celery -A celery_worker flower
```

Nếu bị OOM: Xem xét kill các tiến trình chạy ngầm
```bash
pkill -f uvicorn
pkill -f celery
```