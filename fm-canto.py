# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import threading
import time
import sys
import torch
from opencc import OpenCC
from funasr import AutoModel
import os
import re
import logging

logging.getLogger("funasr").setLevel(logging.ERROR)

os.environ["TQDM_DISABLE"] = "1"

cc = OpenCC('s2hk')

MODEL_ID = "FunAudioLLM/SenseVoiceSmall"

print(f"正在載入 SenseVoice (FunAudioLLM) Cantonese 模型: {MODEL_ID}...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    model = AutoModel(
        model=MODEL_ID,
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        device=device,
        trust_remote_code=True,
        hub="hf"
    )
    print(">>> SenseVoice 載入成功！")
except Exception as e:
    print(f"載入失敗: {e}")
    sys.exit(1)

stop_event = threading.Event()

def playback_thread():
    cmd = "rtl_fm -f 88.1M -s 200k -r 48k -A fast -F 9 -E deemp -g 30 -l 0 | " \
          "tee >(pacat --format=s16le --rate=48000 --channels=1 --device=rtl_fm_sink) | " \
          "pacat --format=s16le --rate=48000 --channels=1"
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stderr=subprocess.DEVNULL)
    while not stop_event.is_set():
        if process.poll() is not None: break
        time.sleep(1)
    process.terminate()

def decode_thread():
    cmd = ["parec", "--format=s16le", "--rate=16000", "--channels=1", "--device=rtl_fm_sink.monitor"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    sample_rate = 16000
    chunk_duration = 2.5
    chunk_size = int(sample_rate * 2 * chunk_duration)

    print(f"--- 實時廣東話辨識啟動 (SenseVoice) ---")

    try:
        while not stop_event.is_set():
            raw_audio = proc.stdout.read(chunk_size)
            if not raw_audio: continue

            t_start = time.perf_counter()
            
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            res = model.generate(
                input=audio_np,
                language="yue",
                use_itn=True,
                batch_size=1,
                disable_pbar=True   #added the display
            )
            raw_text = res[0]["text"].strip() if res else ""
            clean = re.sub(r'<\|.*?\|>', '', raw_text).strip()
            trad_text = cc.convert(clean).rstrip('。')

            if raw_text:
                trad_text = cc.convert(raw_text)
                latency = time.perf_counter() - t_start
                
                color = "\033[92m" if latency < 0.6 else "\033[93m"
                reset = "\033[0m"
                
                sys.stdout.write(f"\r\033[K[廣東話]: {trad_text} | {color}延遲: {latency:.2f}s{reset}")
                sys.stdout.flush()
                if len(trad_text) > 10:
                    print()
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
        print("\n\n\033[91m已安全結束程式\033[0m")
        sys.exit(0)
