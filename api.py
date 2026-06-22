import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import json
import torch
import numpy as np
import soundfile as sf
import threading
import queue
import requests
import time
import traceback
from fastapi import Request
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device

app = FastAPI(title="OmniVoice Voice Cloning API", description="API for Zero-Shot Voice Cloning")

# Phục vụ file tĩnh (để backend tải file .wav về qua /output/...)
os.makedirs("outputs", exist_ok=True)
app.mount("/output", StaticFiles(directory="outputs"), name="output")

# Tải từ điển phát âm
DICT_FILE = "dictionary.json"

def load_dictionary():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_dictionary(data):
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

pronunciation_dict = load_dictionary()

# 1. Khởi tạo model khi start server
device = get_best_device()
print(f"Đang tải model OmniVoice lên {device}...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map=device,
    dtype=torch.float16,
    load_asr=True,
    local_files_only=True
)
print("Model đã sẵn sàng!")

# --- BACKGROUND WORKER ---
task_queue = queue.Queue(maxsize=10)

def process_voice_job(job_data: dict):
    job_id = job_data["job_id"]
    webhook_url = job_data["webhook_url"]
    webhook_secret = job_data["webhook_secret"]
    text = job_data["text"]
    temp_ref_audio_path = job_data["temp_ref_audio_path"]
    ref_text = job_data["ref_text"]
    language = job_data["language"]
    speed = job_data["speed"]
    num_step = job_data["num_step"]
    pause_period = job_data["pause_period"]
    pause_comma = job_data["pause_comma"]
    pause_semicolon = job_data["pause_semicolon"]
    pause_newline = job_data["pause_newline"]
    base_url = job_data["base_url"]
    
    audio_url = None
    error_message = None
    status = "failed"
    
    try:
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=temp_ref_audio_path,
            ref_text=ref_text,
        )
        
        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        
        import re
        import librosa
        
        local_text = text
        for word, replacement in pronunciation_dict.items():
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            local_text = pattern.sub(replacement, local_text)
            
        parts = re.split(r'([.,;\n]+)', local_text)
        final_y = []
        
        for i in range(0, len(parts), 2):
            chunk_text = parts[i].strip()
            delim = parts[i+1] if i+1 < len(parts) else ""
            
            if chunk_text:
                audio = model.generate(
                    text=chunk_text,
                    language=language,
                    generation_config=gen_config,
                    voice_clone_prompt=voice_clone_prompt,
                    speed=1.0,
                )
                chunk_y = audio[0].astype(np.float32)
                chunk_y, _ = librosa.effects.trim(chunk_y, top_db=40)
                final_y.append(chunk_y)
                
            if delim:
                pause_sec = 0.0
                if '.' in delim: pause_sec = pause_period
                elif '\n' in delim: pause_sec = pause_newline
                elif ';' in delim: pause_sec = pause_semicolon
                elif ',' in delim: pause_sec = pause_comma
                if pause_sec > 0:
                    silence_length = int(pause_sec * speed * model.sampling_rate)
                    final_y.append(np.zeros(silence_length, dtype=np.float32))
                    
        if len(final_y) > 0:
            y = np.concatenate(final_y)
        else:
            y = np.zeros(1, dtype=np.float32)
            
        if speed != 1.0:
            import pyrubberband as pyrb
            y = pyrb.time_stretch(y, model.sampling_rate, speed, rbargs={'-F': ''})
            
        waveform = (y * 32767).astype(np.int16)
        
        filename = f"{job_id}.wav"
        output_path = os.path.join("outputs", filename)
        sf.write(output_path, waveform, model.sampling_rate)
        
        audio_url = f"{base_url}output/{filename}"
        status = "success"
        
    except Exception as e:
        error_message = str(e)
        print(f"[Worker] Lỗi xử lý job {job_id}: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(temp_ref_audio_path):
            try:
                os.remove(temp_ref_audio_path)
            except Exception as e:
                print(f"[Worker] Lỗi xóa file tạm: {e}")
                
    payload = {
        "job_id": job_id,
        "status": status,
        "audio_url": audio_url,
        "error_message": error_message
    }
    headers = {"X-Webhook-Secret": webhook_secret, "Content-Type": "application/json"}
    
    print(f"[Worker] Job {job_id}: Đang chuẩn bị gọi Webhook tới URL: {webhook_url}")
    for attempt in range(1, 4):
        try:
            resp = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"[Worker] Job {job_id}: Gửi Webhook thành công.")
                break
            else:
                print(f"[Worker] Job {job_id}: Gửi Webhook thất bại (HTTP {resp.status_code}). Thử lại {attempt}/3...")
        except Exception as e:
            print(f"[Worker] Job {job_id}: Lỗi kết nối Webhook: {e}. Thử lại {attempt}/3...")
        
        if attempt < 3:
            time.sleep(5)
        else:
            print(f"[Worker] Job {job_id}: ĐÃ HỦY gửi Webhook sau 3 lần thử thất bại.")

def worker_loop():
    print("[Worker] Bắt đầu chạy ngầm...")
    while True:
        job_data = task_queue.get()
        if job_data is None:
            break
        print(f"[Worker] Bắt đầu xử lý job: {job_data['job_id']}")
        try:
            process_voice_job(job_data)
        except Exception as e:
            print(f"[Worker] Lỗi không mong muốn trong worker_loop: {e}")
            traceback.print_exc()
        finally:
            task_queue.task_done()
            print(f"[Worker] Hoàn thành job: {job_data['job_id']}")

worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()
# --- KẾT THÚC BACKGROUND WORKER ---

@app.get("/")
def ping():
    return {"status": "ok", "message": "OmniVoice XTTS is running"}

class AddDictRequest(BaseModel):
    words: str
    pronunciations: str

class DeleteDictRequest(BaseModel):
    words: List[str]

@app.get("/dictionary")
def get_dictionary():
    return [{"word": k, "pronunciation": v} for k, v in pronunciation_dict.items()]

@app.post("/dictionary")
def add_to_dictionary(req: AddDictRequest):
    words_list = [w.strip() for w in req.words.split(",")]
    prons_list = [p.strip() for p in req.pronunciations.split(",")]
    
    if len(words_list) != len(prons_list):
        raise HTTPException(status_code=400, detail="Số lượng từ và cách đọc không khớp.")
        
    for w, p in zip(words_list, prons_list):
        if w and p:
            pronunciation_dict[w] = p
            
    save_dictionary(pronunciation_dict)
    return {"status": "success", "message": f"Đã thêm {len(words_list)} từ."}

@app.delete("/dictionary")
def delete_from_dictionary(req: DeleteDictRequest):
    deleted = 0
    for w in req.words:
        if w in pronunciation_dict:
            del pronunciation_dict[w]
            deleted += 1
            
    if deleted > 0:
        save_dictionary(pronunciation_dict)
    return {"status": "success", "message": f"Đã xóa {deleted} từ."}

@app.post("/clone_voice")
def clone_voice(
    request: Request,
    text: str = Form(..., description="Văn bản cần đọc"),
    ref_audio: UploadFile = File(..., description="File audio mẫu (giọng cần clone)"),
    webhook_url: str = Form(..., description="URL để AI Server gọi trả kết quả về"),
    webhook_secret: str = Form(..., description="Mã bảo mật gửi kèm trong Header X-Webhook-Secret"),
    ref_text: str = Form(None, description="Nội dung của file audio mẫu (để trống sẽ tự động nhận diện)"),
    language: str = Form("Vietnamese", description="Ngôn ngữ (mặc định là Vietnamese)"),
    speed: float = Form(1.0, ge=0.25, le=1.25, description="Tốc độ đọc (0.25x - 1.25x)"),
    num_step: int = Form(32, description="Số bước Inference (càng cao càng tốt nhưng chậm)"),
    pause_period: float = Form(0.45, ge=0.0, description="Thời gian nghỉ sau dấu chấm (giây)"),
    pause_comma: float = Form(0.25, ge=0.0, description="Thời gian nghỉ sau dấu phẩy (giây)"),
    pause_semicolon: float = Form(0.3, ge=0.0, description="Thời gian nghỉ sau dấu chấm phẩy (giây)"),
    pause_newline: float = Form(0.6, ge=0.0, description="Thời gian nghỉ sau khi xuống dòng (giây)"),
):
    try:
        # Lưu file audio mẫu tải lên vào thư mục tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
            temp_ref_audio.write(ref_audio.file.read())
            temp_ref_audio_path = temp_ref_audio.name
            
        job_id = str(uuid.uuid4())
        base_url = str(request.base_url)
        
        # Đóng gói tác vụ
        job_data = {
            "job_id": job_id,
            "webhook_url": webhook_url,
            "webhook_secret": webhook_secret,
            "text": text,
            "temp_ref_audio_path": temp_ref_audio_path,
            "ref_text": ref_text,
            "language": language,
            "speed": speed,
            "num_step": num_step,
            "pause_period": pause_period,
            "pause_comma": pause_comma,
            "pause_semicolon": pause_semicolon,
            "pause_newline": pause_newline,
            "base_url": base_url
        }
        
        # Đẩy vào hàng đợi
        try:
            task_queue.put(job_data, block=False)
        except queue.Full:
            # Xóa file tạm vừa tạo nếu bị từ chối
            if os.path.exists(temp_ref_audio_path):
                os.remove(temp_ref_audio_path)
            raise HTTPException(status_code=429, detail="Hệ thống đang quá tải (vượt quá giới hạn hàng đợi). Vui lòng thử lại sau ít phút.")
        
        # Trả về JSON chứa job_id và trạng thái
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "pending"})
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lỗi khi nhận job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
