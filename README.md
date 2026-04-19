# RTL-SDR-to-Text (STT testing script on English/Cantonese/Mandarin detection)

This model is to capture the voice from the device RTL-SDR v4 and generate cantonese text on screen.

You may take free will to chaneg the code, with your imagination.

## Pre-Procedure needed
RTL-SDR v4 USB token = can procure from Amazon or Taobao

Installed with device driver, rtl-fm and gqrx for testing

## Tested Platform 
* HKOS 42

* Fedora 43

* UBuntu 22.04.4

Don't ask me why I don't test in Windows. I just don't like it.

## Description of Works
The program is to output the virtual device via rtl-fm, creating the virtual device to capture and use another thread to decode it using Whisper or Sensevoice.

Tried and ofudn the Sensevoice is much faster.

Here is my finding,

* Whisper Full 2-3s (deleted as it was too slow)

* Wenet 1-2 s

* Sensevoice Small 0.2-0.3s

## File Description

1. load.sh     -  Start the virtual device
2. fm-finale.py      -  Sensevoice best model
3. fm-finale-wenet.py     -  Wenet model
4. fm-finale.py     -   Sensevoice model with longer text captured - slightly more time delay


