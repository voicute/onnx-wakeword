# Voicute ESP32 离线唤醒词 SDK

[English](README.md) | 中文

Voicute 是面向 ESP32 的自定义离线唤醒词 SDK。可复用推理库与开发板相关的完整示例已经分离：

```text
esp32/
├── sdk/voicute/                    可复用 ESP-IDF 组件
└── examples/esp32s3_hmi_devkit/    可直接编译运行的完整示例
```

SDK 接收 16 kHz PCM，负责 Mel 特征、INT8 TFLite 推理、模型 head 后处理和唤醒判定。麦克风、Codec、LED、ESP-SR AFE 和 MultiNet 命令词属于板级示例。

## 已测试平台

| 项目 | 实测配置 |
|---|---|
| 芯片 | ESP32-S3，双核，240 MHz |
| 开发板 | ESP32-S3-HMI-DevKit |
| 存储 | 8 MB Octal PSRAM，16 MB Flash |
| 音频 | 4 颗 MEMS 麦克风、ES7210；ES8311 输出；16 kHz PCM |
| 开发框架 | ESP-IDF 6.0.1 |
| 推理环境 | esp-tflite-micro 1.3.5；ESP-SR MultiNet5 英文模型 |

SDK 可以移植到其他 ESP32-S3 开发板，但需要适配麦克风、Codec BSP 和内存配置。当前 Demo 尚未验证其他 ESP32 芯片系列。

## 内置英文演示

配套的模型和 `head.h` 检测 **Hey Robot**，唤醒后支持：

- Turn the light red
- Turn the light blue
- Turn the light green
- Turn the light white

编译烧录见[示例说明](examples/esp32s3_hmi_devkit/README_CN.md)，集成见 [SDK 文档](sdk/voicute/README_CN.md)。

## 实测性能

上述平台下，Mel 提取约 28.5–33 ms，TFLite `Invoke()` 约 154.9–155.7 ms，每 160 ms 调度最新的 98 帧窗口。性能会随模型和芯片变化。
