# RTL-SDR-to-Text (STT Testing Script for English/Cantonese/Mandarin)

This project captures live audio from an RTL-SDR v4 dongle and performs real-time speech-to-text (STT) conversion to Cantonese (with support for English/Mandarin).
Feel free to modify the code as you wish.
Prerequisites

## RTL-SDR v4 USB dongle (available on Amazon or Taobao)
Installed drivers, rtl-fm, and GQRX (for testing)

## Tested Platforms

HKOS 42
Fedora 43
Ubuntu 22.04.4

## How It Works
The script pipes audio from rtl-fm to a virtual audio device. A separate thread decodes the stream using Whisper, WeNet, or SenseVoice.
Performance comparison:

Whisper (full): 2–3 s (removed – too slow)
WeNet: 1–2 s
SenseVoice Small: 0.2–0.3 s (recommended)

## Files

load.sh – Starts the virtual audio device
fm-finale.py – Best model (SenseVoice)
fm-finale-wenet.py – WeNet model
4. fm-finale.py     -   Sensevoice model with longer text captured - slightly more time delay


