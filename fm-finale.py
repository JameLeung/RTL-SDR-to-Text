import subprocess
import numpy as np
import threading
import time
import sys
import re
from funasr import AutoModel
from opencc import OpenCC

# 1. 初始化繁簡轉換器 (s2hk: 簡體到香港繁體, 包含慣用語)
cc = OpenCC('s2hk')

# 2. 初始化優化：強制關閉所有不必要的偵測
print("正在啟動超高速繁體模式...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cpu", 
    trust_remote_code=True,
    disable_update=True,
    # 如果你的 CPU 支援，可以嘗試增加線程數（例如 4）
    # torch_config={"intra_op_num_threads": 4} 
)

stop_event = threading.Event()

def playback_thread():
    # 確保 stderr 被導向 DEVNULL 避免干擾控制台
    cmd = "rtl_fm -f 88.1M -s 200k -r 48k -A fast -F 9 -E deemp -g 30 -l 0 | " \
          "tee >(pacat --format=s16le --rate=48000 --channels=1 --device=rtl_fm_sink) | " \
          "pacat --format=s16le --rate=48000 --channels=1"
    
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash', stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while not stop_event.is_set():
        if process.poll() is not None: break
        time.sleep(1)
    process.terminate()

def decode_thread():
    # 擷取音訊
    cmd = ["parec", "--format=s16le", "--rate=16000", "--channels=1", "--device=rtl_fm_sink.monitor"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    chunk_duration = 2 
    sample_rate = 16000
    chunk_size = sample_rate * 2 * chunk_duration 

    # 預編譯正則，只保留內容
    tag_pattern = re.compile(r'<\|.*?\|>')

    print(f"--- 實時繁體辨識開始 (目標延遲: 0.1s-0.2s) ---")

    try:
        while not stop_event.is_set():
            raw_audio = proc.stdout.read(chunk_size)
            t_start = time.perf_counter() 
            
            if not raw_audio: continue

            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # 核心優化點：
            # 1. disable_pbar=True 減少 UI 開銷
            # 2. 如果延遲還是高，可嘗試設置 use_itn=False
            res = model.generate(
                input=audio_np,
                cache={},
                language="yue",
                use_itn=True, 
                disable_pbar=True
            )

            if res:
                # 1. 快速移除標籤
                raw_text = tag_pattern.sub('', res[0]['text']).strip()
                
                # 2. 極速繁體轉換
                trad_text = cc.convert(raw_text)
                
                t_end = time.perf_counter()
                latency = t_end - t_start

                if trad_text:
                    # 使用顏色標記延遲，方便觀察
                    color = "\033[92m" if latency < 0.2 else "\033[93m"
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
