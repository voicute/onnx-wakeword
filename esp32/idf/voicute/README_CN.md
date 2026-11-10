# Voicute ESP-IDF 组件

[English](README.md) | 中文

可复用的 ESP-IDF 离线唤醒词 / 关键词识别运行时，已发布到 [Espressif Component Registry](https://components.espressif.com)（组件名 **`voicute/voicute`**）：

```yaml
# 你的工程 main/idf_component.yml
dependencies:
  voicute/voicute: "^1.0.0"
```

或者把本目录复制进工程的 `components/`，或通过 `EXTRA_COMPONENT_DIRS` 引入。可复制的最小示例工程在 [`examples/minimal/`](examples/minimal/)，完整板级示例在 [`../examples/esp32s3_hmi_devkit/`](../examples/)。

## 组件提供

16 kHz PCM 到 Mel 特征处理、INT8 TensorFlow Lite Micro 推理（ESP-NN 加速）、可配置的模型 head 回调、可选 L1–L5 检测逻辑，以及从 SPIFFS 加载多模型。接口见 [API.md](API.md)。

## 组件刻意不包含

- **不含模型** —— 带你自己的 `.tflite`（来自 [voicute.com](https://www.voicute.com) 或自行导出），烧到 SPIFFS 或编译进固件 C 数组
- **不含麦克风驱动** —— 你自己喂 16 kHz / 单声道 / int16 PCM，每个推理节拍一窗 `MEL_AUDIO_LEN`（16080 采样）+ 该窗 RMS
- **不含板级 BSP** —— 不含 codec（ES7210 等）、LED、按键
- **不含模型头权重** —— `postprocess` 回调由调用方提供，把 INT8 backbone 输出转成概率，必须与你模型的训练头匹配（见 [`examples/minimal/`](examples/minimal/) 的 `kws_postprocess.h` + `head_stub.h`）

## 实测目标

- **ESP32-S3**：完整硬件实测（本仓库全部板级验证都在 S3 上）
- 其他目标（含 ESP32-P4）：代码路径可用，**未经硬件实测**，不做支持承诺

## ESP-IDF 版本

以 **ESP-IDF v6.0** 构建验证（CMake + Component Manager 依赖）。

## L1–L5 在 ESP32 上的默认配置

MCU 推理远慢于手机/桌面，默认配置比移动端/网页端更精简：

| 层 | 用途 | ESP32 默认 |
| :---: | ---- | :---: |
| L1 | 连续帧门控 | **关**（帧率太低） |
| L2 | 峰值 ≫ 背景 | 开 |
| L3 | 1.5s 冷却 | 开 |
| L4 | burst 抑制 | 关 |
| L5 | 能量跳变（音乐/视频过滤，语后确认） | 开，`l5_delta = 200` |

通过 `recognizer_config_t` 调整。

## 许可证

MIT。
