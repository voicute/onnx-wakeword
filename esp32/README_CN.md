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
| 推理环境 | esp-tflite-micro 1.4.1 + ESP-NN 1.4.1；ESP-SR MultiNet5 英文模型 |

SDK 可以移植到其他 ESP32-S3 开发板，但需要适配麦克风、Codec BSP 和内存配置。当前 Demo 尚未验证其他 ESP32 芯片系列。

## 内置英文演示

配套的模型和 `head.h` 检测 **Hey Robot**，唤醒后支持：

- Turn the light red
- Turn the light blue
- Turn the light green
- Turn the light white

编译烧录见[示例说明](examples/esp32s3_hmi_devkit/README_CN.md)，集成见 [SDK 文档](sdk/voicute/README_CN.md)。

## 实测性能

在 ESP32-S3 240 MHz、98 帧模型（esp-tflite-micro 1.4.1 + ESP-NN 1.4.1）下，Mel 提取约 29 ms；TFLite `Invoke()` 单独测量约 39.5–40.2 ms（平均约 39.7 ms），在完整 Demo 应用（ESP-SR AFE 喂音与 MultiNet 同板并行运行）中实测墙钟约 51 ms。随附的两个 98 帧模型（参考模型 `hey_robot` 与 `doubaodoubao`）均已上板实测，结果一致。性能会随模型、主频和芯片变化。

## 注意事项

- **换模型必须连 head.h 一起换**：`spiffs_content/*.tflite` 与 `main/head.h` 中的分类器权重、标定常数（`KWS_ABS_TEMP` / `KWS_FC_B`）是配套的，只换一边会导致打分错误。
- **加长词窗口要同步扩 arena**：`sdk/voicute/model_loader.h` 的 `TFLITE_ARENA_SIZE` 固定 64 KB，98 帧模型实际占用约 43 KB。`main/head.h` 中的 `KWS_MEL_TIME` 可配置（未定义时默认 98），调大（如长词用到 200 帧）时必须按比例扩大 arena，否则模型加载会因内存不足失败。
- **mel_time 不匹配即拒绝加载**：模型输入帧数与 `KWS_MEL_TIME` 不一致时启动即报错，没有静默降级路径——这是刻意设计，便于第一时间发现模型与固件版本错配。
- **组件下载需要网络**：`idf.py set-target` 会重新解析组件版本，`dependencies.lock` 锁定版本保证可复现。若机器通过系统代理上网，不要在构建脚本里设置 `no_proxy=*` 或清空代理环境变量，否则组件注册表解析会静默失败（表现为 "Component not found" / "no versions match"）。
- **`patch_led_strip.py` 必须在 `set-target` 之后执行**：set-target 会重新下载 `managed_components/`，把补丁覆盖掉。`build.bat` 已按正确顺序执行，手动操作时不要漏掉这一步。
