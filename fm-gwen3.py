import subprocess
import numpy as np
import threading
import time
import sys
import torch
from qwen_asr import Qwen3ASRModel

# 1. Load Qwen3-ASR-0.6B (CPU optimized)
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    dtype=torch.float32,
    device_map="cpu",
)
print(">>> Qwen3-ASR-0.6B 載入成功！")

stop_event = threading.Event()

def playback_thread():
    cmd = "rtl_fm -f 88.1M -s 200k -r 48k -A fast -F 9 -E deemp -g 30 -l 0 | " \
          "tee >(pacat --format=s16le --rate=48000 --channels=1 --device=rtl_fm_sink) | " \
          "pacat --format=s16le --rate=48000 --channels=1"
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while not stop_event.is_set():
        if process.poll() is not None: break
        time.sleep(1)
    process.terminate()

def decode_thread():
    cmd = ["parec", "--format=s16le", "--rate=16000", "--channels=1", "--device=rtl_fm_sink.monitor"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    chunk_duration = 2
    sample_rate = 16000
    chunk_size = sample_rate * 2 * chunk_duration

    print(f"--- 實時粵語辨識開始 (Qwen3-ASR-0.6B) ---")

    try:
        while not stop_event.is_set():
            raw_audio = proc.stdout.read(chunk_size)
            t_start = time.perf_counter()
            
            if not raw_audio: continue

            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            results = model.transcribe(
                audio=(audio_np, sample_rate),
                language="Cantonese"
            )

            if results:
                trad_text = results[0].text.strip()

                t_end = time.perf_counter()
                latency = t_end - t_start

                if trad_text:
                    color = "\033[92m" if latency < 0.3 else "\033[93m"
                    reset = "\033[0m"
                    sys.stdout.write(f"\r[廣東話]: {trad_text:<30} | {color}延遲: {latency:.3f}s{reset}\n")
                    sys.stdout.flush()

    finally:
        proc.terminate()

if __name__ == "__main__":
    t_play = threading.Thread(target=playback_thread, daemon=True)
    t_decode = threading.Thread(target=decode_thread, daemon=True)

    t_play.start()
    t_decode.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        print("\n\033[91m已安全結束程式\033[0m")
        sys.exit(0)
