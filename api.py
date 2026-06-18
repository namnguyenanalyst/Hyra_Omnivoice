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
    text: str = Form(..., description="Văn bản cần đọc"),
    ref_audio: UploadFile = File(..., description="File audio mẫu (giọng cần clone)"),
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
        # Lưu file audio mẫu (reference audio) tải lên vào thư mục tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_ref_audio:
            temp_ref_audio.write(ref_audio.file.read())
            temp_ref_audio_path = temp_ref_audio.name
            
        # Tạo prompt từ audio mẫu
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=temp_ref_audio_path,
            ref_text=ref_text,
        )
        
        # Cấu hình các thông số tạo giọng nói
        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        
        import re
        import librosa
        
        # Áp dụng từ điển phát âm
        for word, replacement in pronunciation_dict.items():
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            text = pattern.sub(replacement, text)
        
        # Tiền xử lý: Tách văn bản thành các cụm theo dấu câu
        parts = re.split(r'([.,;\n]+)', text)
        final_y = []
        
        # Sinh audio cho từng cụm và chèn khoảng lặng
        for i in range(0, len(parts), 2):
            chunk_text = parts[i].strip()
            delim = parts[i+1] if i+1 < len(parts) else ""
            
            if chunk_text:
                # Gọi model để sinh audio ở tốc độ gốc (để đảm bảo chất lượng AI)
                audio = model.generate(
                    text=chunk_text,
                    language=language,
                    generation_config=gen_config,
                    voice_clone_prompt=voice_clone_prompt,
                    speed=1.0,
                )
                chunk_y = audio[0].astype(np.float32)
                
                # Cắt khoảng lặng thừa do AI tự sinh ra bằng thư viện librosa
                # top_db=40 là ngưỡng decibel tiêu chuẩn để cắt các âm thanh quá nhỏ (im lặng)
                chunk_y, _ = librosa.effects.trim(chunk_y, top_db=40)
                final_y.append(chunk_y)
                
            # Xử lý chèn khoảng lặng nhân tạo
            if delim:
                pause_sec = 0.0
                if '.' in delim:
                    pause_sec = pause_period
                elif '\n' in delim:
                    pause_sec = pause_newline
                elif ';' in delim:
                    pause_sec = pause_semicolon
                elif ',' in delim:
                    pause_sec = pause_comma
                    
                if pause_sec > 0:
                    # Tính toán số lượng frames cần thiết. Nhân với speed để khi RubberBand 
                    # tua nhanh/chậm thì khoảng lặng vẫn giữ đúng thời lượng người dùng set.
                    silence_length = int(pause_sec * speed * model.sampling_rate)
                    final_y.append(np.zeros(silence_length, dtype=np.float32))
                    
        # Nối tất cả các mảng lại với nhau
        if len(final_y) > 0:
            y = np.concatenate(final_y)
        else:
            y = np.zeros(1, dtype=np.float32)
        
        # Thay đổi tốc độ bằng thuật toán RubberBand chất lượng cao
        if speed != 1.0:
            import pyrubberband as pyrb
            # Sử dụng cờ '-F' (Formant preservation) 
            # để đảm bảo giữ nguyên tuyệt đối âm sắc gốc của giọng người.
            y = pyrb.time_stretch(y, model.sampling_rate, speed, rbargs={'-F': ''})
            
        # Chuyển đổi mảng numpy thành file âm thanh thực
        waveform = (y * 32767).astype(np.int16)
        
        # Lưu file vào thư mục public /outputs thay vì trả về trực tiếp
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.wav"
        output_path = os.path.join("outputs", filename)
        sf.write(output_path, waveform, model.sampling_rate)
        
        # Xoá file audio mẫu tạm thời
        os.remove(temp_ref_audio_path)
        
        # Trả về JSON chứa URL của file audio (chuẩn theo backend kỳ vọng)
        return {"audio_url": f"/output/{filename}"}
        
    except Exception as e:
        print(f"Lỗi khi xử lý: {e}")
        raise HTTPException(status_code=500, detail=str(e))
