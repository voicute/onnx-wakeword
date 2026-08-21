# Voicute ESP-IDF component

English | [中文](README_CN.md)

This is the reusable wake-word inference SDK. It contains no board BSP, microphone driver, LED behavior, command application, or demo model.

It provides 16 kHz PCM-to-Mel processing, INT8 TensorFlow Lite Micro inference, a configurable model-head callback, optional L1–L5 detection logic, and multi-model loading from SPIFFS.

Add this directory through `EXTRA_COMPONENT_DIRS`, or copy it into an ESP-IDF project's `components/` directory. See [API.md](API.md) and the [complete example](../../examples/esp32s3_hmi_devkit/README.md).
