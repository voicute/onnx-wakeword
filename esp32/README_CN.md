# Voicute ESP32 离线唤醒词 SDK

[English](README.md) | 中文

Voicute 是面向 ESP32 的自定义离线唤醒词 SDK。可复用推理库与开发板相关的完整示例已经分离：

```text
esp32/
├── sdk/voicute/                    可复用 ESP-IDF 组件
└── examples/esp32s3_hmi_devkit/    可直接编译运行的完整示例
```

SDK 接收 16 kHz PCM，负责 Mel 特征、INT8 TFLite 推理、模型 head 后处理和唤醒判定。麦克风、Codec、LED 和 ESP-SR AFE 属于板级示例。

## 已测试平台

| 项目 | 实测配置 |
|---|---|
| 芯片 | ESP32-S3，双核，240 MHz |
| 开发板 | ESP32-S3-HMI-DevKit |
| 存储 | 8 MB Octal PSRAM，16 MB Flash |
| 音频 | 4 颗 MEMS 麦克风、ES7210；ES8311 输出；16 kHz PCM |
| 开发框架 | ESP-IDF 6.0.1 |
| 推理环境 | esp-tflite-micro 1.4.1 + ESP-NN 1.4.1；ESP-SR AFE（MultiNet 已移除 —— 命令词改用我们自己的 KWS 模型） |

SDK 可以移植到其他 ESP32-S3 开发板，但需要适配麦克风、Codec BSP 和内存配置。当前 Demo 尚未验证其他 ESP32 芯片系列。

## 内置演示（德语，双自有模型）

示例在一块板子上分时复用跑两个我们自己的 KWS 模型（MultiNet 已完全移除）：

1. 说 **Hallo Lumo** 唤醒（单头 MultiProto，K=5）。
2. 6 秒内说 **Licht rot / Licht blau / Licht grün / Licht weiß**（多头，N=4 × K=3）。
3. LED 显示对应颜色，程序返回唤醒词检测状态。

两个模型都是 mel_time=150 导出（1.532s 窗口，32 mel），已上板验证：噪声环境（rms 150–190）和安静房间（rms 24–36）共 20+ 次唤醒（prob 0.73–1.000）全命中，四色命令零混淆。

编译烧录见[示例说明](examples/esp32s3_hmi_devkit/README_CN.md)（含模型目录说明和两份 head.h 的合并方法），集成见 [SDK 文档](sdk/voicute/README_CN.md)。

## 模型包（`onnx-wakeword/models/<语言>/`）

每个交付模型是一个 zip，内含完整配套（backbone + head），单个包即可完成集成：

```text
models/de/
├── hallolumo_r1_1_x654af9_tflite.zip   "Hallo Lumo" 唤醒模型
│     ├── hallo_lumo.tflite              int8 backbone（文件名 stem = 模型标识）
│     └── head.h                         float MultiProto 头（KWS_MEL_TIME 150, K=5）
├── licht_multi_v10.0_de_tflite.zip     "Licht rot/blau/grün/weiß" 命令模型
│     ├── licht_multi.tflite             int8 backbone
│     ├── head.h                         多头（KWS_HEAD_N 4, KWS_HEAD_K 3）
│     ├── model_info.json                mel 配置、词表、每词阈值
│     └── _report.txt                    训练/评估报告
├── hallolumo_r1_1_x654af9.zip          ONNX 版（Android/Web/Python 用；不含 head.h）
└── ...                                 旧 ONNX 产物留作参考
```

ESP32 用 `*_tflite.zip` 包。两份 `head.h` 不能直接一起编译（符号名冲突）—— 示例的 `main/_merge_heads.py` 会把它们合并成一个 `main/head.h`（`KWS_WAKE_*` / `KWS_CMD_*` 前缀）。这样合并的所有模型 mel 配置必须一致。详见[示例说明](examples/esp32s3_hmi_devkit/README_CN.md#head-怎么合并main_merge_headspy)。

## 实测性能

随附的 mel_time=150 模型（1.532s 窗口）在 ESP32-S3 240 MHz（esp-tflite-micro 1.4.1 + ESP-NN 1.4.1）下，Mel 提取约 47 ms、TFLite `Invoke()` 约 76 ms —— 每帧 121 ms，对 160 ms 的帧周期是实时有余量。此数据来自双模型完整 Demo 实测（ESP-SR AFE 在另一核运行；被门控禁用的模型直接跳过 Invoke，零开销）。性能会随模型、主频和芯片变化。

参考：旧 98 帧模型实测 Mel 约 29 ms + Invoke 约 40 ms（旧 MultiNet 同板并行时墙钟约 51 ms）。

## 注意事项

- **换模型必须连 head.h 一起换**：`spiffs_content/*.tflite` 与 `main/head.h` 中的分类器权重、标定常数（`KWS_ABS_TEMP` / `KWS_FC_B`）是配套的，只换一边会导致打分错误。
- **加长词窗口要同步扩 arena**：`sdk/voicute/model_loader.h` 的 `TFLITE_ARENA_SIZE` 固定 64 KB，98 帧模型实际占用约 43 KB。`main/head.h` 中的 `KWS_MEL_TIME` 可配置（未定义时默认 98），调大（如长词用到 200 帧）时必须按比例扩大 arena，否则模型加载会因内存不足失败。
- **mel_time 不匹配即拒绝加载**：模型输入帧数与 `KWS_MEL_TIME` 不一致时启动即报错，没有静默降级路径——这是刻意设计，便于第一时间发现模型与固件版本错配。
- **组件下载需要网络**：`idf.py set-target` 会重新解析组件版本，`dependencies.lock` 锁定版本保证可复现。若机器通过系统代理上网，不要在构建脚本里设置 `no_proxy=*` 或清空代理环境变量，否则组件注册表解析会静默失败（表现为 "Component not found" / "no versions match"）。
- **`patch_led_strip.py` 必须在 `set-target` 之后执行**：set-target 会重新下载 `managed_components/`，把补丁覆盖掉。`build.bat` 已按正确顺序执行，手动操作时不要漏掉这一步。
