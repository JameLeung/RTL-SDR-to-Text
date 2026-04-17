pactl load-module module-remap-source source_name=rtl_fm_mic master=rtl_fm_sink.monitor channels=1 rate=48000
pactl load-module module-null-sink sink_name=rtl_fm_sink sink_properties=device.description=RTL_FM_Virtual_Mic
