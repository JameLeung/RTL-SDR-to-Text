# RTL-SDR-to-Text using ASR (STT Testing Script for English/Cantonese/Mandarin)

This project captures live audio from an RTL-SDR v4 dongle and performs real-time speech-to-text (STT) conversion to Cantonese (with support for English/Mandarin).

Feel free to modify the code as you wish.


## Prerequisites
RTL-SDR v4 USB dongle (available to buy in Amazon or Taobao)
Installed drivers, rtl-fm, and GQRX (for testing)

## Tested Platforms

HKOS 42

Fedora 43

Ubuntu 22.04.4

### What data you used ?
I use the radio tuner to collect the tranlsate character made and evaluate the performance of the systems , in terms of decoding time, and character accuracy reading on the fly.

I am Hongkongese and easy to spot if it is wrong.


## How It Works
The script pipes audio from rtl-fm to a virtual audio device. A separate thread decodes the stream using Whisper, WeNet, or SenseVoice.
Performance comparison:

SenseVoice Small: 0.2–0.3 s (recommended)
WeNet: 1–2 s 
Gwen: 1-2 s 
Whisper (full): 2–3 s (removed – too slow)



