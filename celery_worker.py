import os
import re
import time
import traceback
import requests
import numpy as np
import soundfile as sf
import torch
from celery import Celery
from celery.signals import worker_process_init
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.common import get_best_device

celery_app = Celery(
    'omnivoice_worker',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Ho_Chi_Minh',
    enable_utc=True,
    worker_prefetch_multiplier=1, # Quan trọng để GPU worker chỉ nhận 1 task mỗi lần
)

# 1. Biến global chứa model AI
model = None

@worker_process_init.connect
def init_worker(**kwargs):
    global model
    device = get_best_device()
    print(f"\n[Celery Worker] Đang tải model OmniVoice lên {device}...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
        load_asr=True,
        local_files_only=True
    )
    print("[Celery Worker] Model đã sẵn sàng!\n")

@celery_app.task(name="process_voice_job", bind=True)
def process_voice_job(self, job_data: dict):
    global model

    job_id = job_data["job_id"]
    webhook_url = job_data["webhook_url"]
    webhook_secret = job_data["webhook_secret"]
    processed_text = job_data["processed_text"] # Text đã được xử lý dictionary ở API
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
        print(f"[Worker] Đang xử lý job: {job_id}")
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
        
        import librosa
        
        # Cắt chuỗi theo dấu câu
        parts = re.split(r'([.,;\n]+)', processed_text)
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
        os.makedirs("outputs", exist_ok=True)
        sf.write(output_path, waveform, model.sampling_rate)
        
        audio_url = f"{base_url}output/{filename}"
        status = "success"
        print(f"[Worker] Đã tạo xong audio cho job: {job_id}")
        
    except Exception as e:
        error_message = str(e)
        print(f"[Worker] Lỗi xử lý job {job_id}: {e}")
        traceback.print_exc()
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(temp_ref_audio_path):
            try:
                os.remove(temp_ref_audio_path)
            except Exception as e:
                print(f"[Worker] Lỗi xóa file tạm: {e}")
                
    # Gọi Webhook
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
    
    return payload
