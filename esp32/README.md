# Voicute for ESP32

English | [中文](README_CN.md)

Voicute is an offline custom wake-word SDK for ESP32. The reusable inference component is separated from the board-specific runnable example.

## Repository layout

```text
esp32/
├── sdk/voicute/                    Reusable ESP-IDF component
└── examples/esp32s3_hmi_devkit/    Complete tested application
```

The SDK accepts 16 kHz PCM and performs Mel extraction, INT8 TFLite inference, model-head post-processing, and wake-word detection. Microphone, codec, LED, ESP-SR AFE, and MultiNet handling belong to the example.

## Tested platform

| Item | Tested configuration |
|---|---|
| SoC | ESP32-S3, dual core, 240 MHz |
| Board | ESP32-S3-HMI-DevKit |
| Memory | 8 MB octal PSRAM, 16 MB flash |
| Audio | 4 MEMS microphones through ES7210; ES8311 output; 16 kHz PCM |
| Framework | ESP-IDF 6.0.1 |
| Runtime | esp-tflite-micro 1.3.5; ESP-SR MultiNet5 English |

Other ESP32-S3 boards can use the SDK after adapting their microphone/codec BSP and memory configuration. Other ESP32 chip families are not yet validated by this example.

## Included English demo

The matched model and head detect **Hey Robot**. After wake-up, say:

- Turn the light red
- Turn the light blue
- Turn the light green
- Turn the light white

See the [example guide](examples/esp32s3_hmi_devkit/README.md) or the [SDK guide](sdk/voicute/README.md).

## Measured performance

On the tested platform: Mel extraction is approximately 28.5–33 ms, TFLite `Invoke()` is approximately 154.9–155.7 ms, and the newest 98-frame window is scheduled every 160 ms. Results vary by model and target.
