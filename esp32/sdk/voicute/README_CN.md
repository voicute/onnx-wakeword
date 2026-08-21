# Voicute ESP-IDF 组件

[English](README.md) | 中文

本目录是可复用的唤醒词推理 SDK，不包含开发板 BSP、麦克风驱动、LED 逻辑、命令词应用和演示模型。

组件提供 16 kHz PCM 到 Mel 特征处理、INT8 TensorFlow Lite Micro 推理、可配置的模型 head 回调、可选 L1–L5 检测逻辑，以及从 SPIFFS 加载多个模型。

可以通过 `EXTRA_COMPONENT_DIRS` 引入，也可以复制到 ESP-IDF 工程的 `components/` 目录。接口见 [API.md](API.md)，完整运行方式见[板级示例](../../examples/esp32s3_hmi_devkit/README_CN.md)。
