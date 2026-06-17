import os
import uuid
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import soundfile as sf

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device

app = FastAPI(title="OmniVoice Voice Cloning API", description="API for Zero-Shot Voice Cloning")

# Phục vụ file tĩnh (để backend tải file .wav về qua /output/...)
os.makedirs("outputs", exist_ok=True)
app.mount("/output", StaticFiles(directory="outputs"), name="output")

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

@app.post("/clone_voice")
def clone_voice(
    text: str = Form(..., description="Văn bản cần đọc"),
    ref_audio: UploadFile = File(..., description="File audio mẫu (giọng cần clone)"),
    ref_text: str = Form(None, description="Nội dung của file audio mẫu (để trống sẽ tự động nhận diện)"),
    language: str = Form("Vietnamese", description="Ngôn ngữ (mặc định là Vietnamese)"),
    speed: float = Form(1.0, description="Tốc độ đọc"),
    num_step: int = Form(32, description="Số bước Inference (càng cao càng tốt nhưng chậm)"),
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
        
        # Gọi model để sinh audio
        audio = model.generate(
            text=text,
            language=language,
            generation_config=gen_config,
            voice_clone_prompt=voice_clone_prompt,
            speed=speed,
        )
        
        # Chuyển đổi mảng numpy thành file âm thanh thực
        waveform = (audio[0] * 32767).astype(np.int16)
        
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
