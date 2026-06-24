import os
from dotenv import load_dotenv

load_dotenv()

import re
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import json
import traceback
from fastapi import Request
from celery import Celery

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

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery('omnivoice_worker', broker=REDIS_URL)

@app.get("/")
def ping():
    return {"status": "ok", "message": "OmniVoice API Server is running (Celery Mode)"}

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
    ref_audio: List[UploadFile] = File(..., description="File audio mẫu (giọng cần clone)"),
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
    temp_ref_audio_paths = []
    try:
        # Lưu các file audio mẫu tải lên vào thư mục tạm
        for audio_file in ref_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
                temp_ref_audio.write(audio_file.file.read())
                temp_ref_audio_paths.append(temp_ref_audio.name)
            
        job_id = str(uuid.uuid4())
        base_url = str(request.base_url)
        
        # Tiền xử lý text với dictionary ở tầng API trước khi gửi sang Worker
        local_text = text
        for word, replacement in pronunciation_dict.items():
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            local_text = pattern.sub(replacement, local_text)
        
        # Đóng gói tác vụ
        job_data = {
            "job_id": job_id,
            "webhook_url": webhook_url,
            "webhook_secret": webhook_secret,
            "processed_text": local_text, # Gửi text đã được thay thế
            "temp_ref_audio_paths": temp_ref_audio_paths,
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
        
        # Đẩy vào Celery Task Queue
        try:
            celery_app.send_task("process_voice_job", args=[job_data])
        except Exception as e:
            # Xóa các file tạm vừa tạo nếu đẩy vào queue thất bại (vd Redis chết)
            for p in temp_ref_audio_paths:
                if os.path.exists(p):
                    os.remove(p)
            raise HTTPException(status_code=503, detail=f"Không thể kết nối đến Message Broker (Redis): {str(e)}")
        
        # Trả về JSON chứa job_id và trạng thái
        return JSONResponse(status_code=200, content={"job_id": job_id, "status": "pending"})
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Lỗi khi nhận job: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
