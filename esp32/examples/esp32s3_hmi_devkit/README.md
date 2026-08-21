# Hey Robot demo for ESP32-S3-HMI-DevKit

English | [中文](README_CN.md)

This is a complete board-specific application using the Voicute SDK.

## Supported hardware

Tested on **ESP32-S3-HMI-DevKit**: ESP32-S3 at 240 MHz, 8 MB octal PSRAM, 16 MB flash, four MEMS microphones through ES7210, ES8311 audio output, a WS2812 status LED, and 16 kHz audio. The verified framework is **ESP-IDF 6.0.1**.

Different boards require changes to `main/bsp_board.c`, GPIO assignments, codec configuration, and possibly partition/PSRAM settings.

## Demo flow

1. Say **Hey Robot**.
2. Say Turn the light red, blue, green, or white.
3. The LED changes color and the application returns to wake-word mode.

`spiffs_content/hey_robot.tflite` and `main/head.h` are a matched pair and must be replaced together.

## Build and flash

From this directory in an ESP-IDF 6.0.x shell:

```bash
idf.py set-target esp32s3
python patch_led_strip.py
idf.py build
idf.py -p COM6 flash monitor
```

On Windows, `build.bat` performs target selection, the compatibility patch, and build. Update the COM port in `flash.bat` before flashing.
