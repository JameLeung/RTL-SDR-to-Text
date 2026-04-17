# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import threading
import time
import sys
import os
from funasr import AutoModel
from opencc import OpenCC

# 1. 初始化繁簡轉換器 (香港模式)
cc = OpenCC('s2hk')

# 2. 模型初始化邏輯
# 我們嘗試使用 ModelScope 官方最新的粵語路徑
MODEL_ID = "iic/speech_UniASR_asr_2pass-cantonese-CHS-16k-common-vocab1468-tensorflow1-online"

REVISION = "v2.0.4" # 強制指定版本，避開 404

print(f"正在載入 WenetSpeech-Yue 模型: {MODEL_ID}...")

try:
    model = AutoModel(
        model=MODEL_ID,
        # revision=REVISION, # 如果依然 404，請取消這行的註釋
        device="cpu", # 如果你有 GPU，這裡改成 "cuda"
        disable_update=True,
        trust_remote_code=True
    )
    print(">>> WenetSpeech-Yue 載入成功！")
except Exception as e:
    print(f"\n[錯誤] 無法從 ModelScope 下載模型。")
    print(f"請嘗試手動安裝指令: pip install -U modelscope funasr")
    print(f"報錯詳情: {e}")
    sys.exit(1)

stop_event = threading.Event()

def playback_thread():
    """RTL-SDR 播放與音訊流轉發"""
    cmd = "rtl_fm -f 88.1M -s 200k -r 48k -A fast -F 9 -E deemp -g 30 -l 0 | " \
          "tee >(pacat --format=s16le --rate=48000 --channels=1 --device=rtl_fm_sink) | " \
          "pacat --format=s16le --rate=48000 --channels=1"
    
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stderr=subprocess.DEVNULL)
    while not stop_event.is_set():
        if process.poll() is not None: break
        time.sleep(1)
    process.terminate()

def decode_thread():
    """Paraformer 實時解碼邏輯"""
    # 擷取音訊 (16000Hz)
    cmd = ["parec", "--format=s16le", "--rate=16000", "--channels=1", "--device=rtl_fm_sink.monitor"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    sample_rate = 16000
    chunk_duration = 3.5 # Paraformer 在 3-4 秒的區塊表現最穩定
    chunk_size = int(sample_rate * 2 * chunk_duration) 

    print(f"--- 實時廣東話辨識中 (WenetSpeech-Yue) ---")

    try:
        while not stop_event.is_set():
            raw_audio = proc.stdout.read(chunk_size)
            if not raw_audio: continue

            t_start = time.perf_counter()
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 執行辨識
            res = model.generate(
                input=audio_np,
                batch_size_s=300,
                is_final=True
            )

            if res and len(res) > 0:
                raw_text = res[0]['text'].strip()
                if raw_text:
                    trad_text = cc.convert(raw_text)
                    latency = time.perf_counter() - t_start
                    
                    color = "\033[92m" if latency < 0.7 else "\033[93m"
                    reset = "\033[0m"
                    
                    sys.stdout.write(f"\r\033[K[廣東話]: {trad_text} | {color}延遲: {latency:.2f}s{reset}")
                    sys.stdout.flush()
                    if len(trad_text) > 15: print()

    finally:
        proc.terminate()

if __name__ == "__main__":
    t_play = threading.Thread(target=playback_thread, daemon=True)
    t_decode = threading.Thread(target=decode_thread, daemon=True)

    t_play.start()
    t_decode.start()

    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()
        print("\n\n\033[91m已終止辨識\033[0m")
        sys.exit(0)
