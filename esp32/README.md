# Voicute Wake Word — ESP32 唤醒词检测

基于 ESP32-S3 的离线唤醒词检测引擎，支持自定义唤醒词、多模型同时检测、语音命令联动。

**测试平台**: ESP32-S3-HMI-DevKit (4 颗 MEMS 麦克风, ES7210 ADC, 8MB PSRAM, 16MB Flash)

## 特性

- **完全离线** — 无需网络，推理全程在 ESP32 上完成
- **自定义唤醒词** — 训练自己的唤醒词模型，替换 `head.h` + `.tflite` 即可
- **5 层检测管线 (L1-L5)** — 从 Android 端完整移植，适配 ESP32
- **MultiNet 语音命令** — 唤醒后进入命令模式，支持"灯光变红/蓝/绿/白"
- **多模型支持** — 最多 3 个唤醒词同时检测
- **概率平滑** — 5 帧滑动窗口 max，适应 ESP32 慢推理节奏
- **低误触发** — L5 能量跳变 + 后静音检测，过滤音乐/视频/TV 误触
- **WS2812 LED 反馈** — 唤醒闪烁、命令确认变色

## 硬件要求

| 组件 | 说明 |
|------|------|
| MCU | ESP32-S3 (需 PSRAM ≥ 8MB) |
| Flash | ≥ 16MB (存储模型 + MultiNet + 固件) |
| 麦克风 | 4 × MEMS 数字麦克风 (ES7210 ADC) |
| 音频 | 16kHz / 16bit / 单通道 |
| 开发板 | ESP32-S3-HMI-DevKit (或其他 ES7210 + ES8311 方案) |

如果你的硬件不同，需要修改 `main/bsp_board.c` 中的音频驱动。

## 目录结构

```
esp32/
├── README.md                    ← 本文件
├── CMakeLists.txt               ← 顶层 ESP-IDF 项目
├── sdkconfig.defaults           ← 默认 Kconfig 配置
├── partitions.csv               ← Flash 分区表
├── build.bat / flash.bat        ← Windows 编译/烧录脚本
│
├── components/
│   └── voicute/                  ← ★ 核心库 (通用, 不依赖具体模型)
│       ├── recognizer.cpp/h        推理管线 (Mel → TFLite → head → L1-L5)
│       ├── model_loader.cpp/h      TFLite 模型加载 (SPIFFS)
│       ├── detect_logic.c/h        L1-L5 检测管线 (与 Android 一致)
│       ├── mel_extractor.c/h       Mel 频谱提取 (纯 C, ESP-DSP FFT)
│       ├── kws_postprocess.h       MultiProto head 参考实现
│       ├── mel_filterbank.h        Mel 滤波器系数
│       ├── onnx_window.h           FFT 窗函数
│       ├── ring_buffer.h           双核无锁环形缓冲区
│       └── API.md                  API 详细文档
│
├── main/                        ← Demo 应用 (参考实现)
│   ├── main.cpp                    唤醒词 + MultiNet 语音命令完整 demo
│   ├── head.h                      模型 head 权重 (★ 用户须替换)
│   ├── bsp_board.c/h               板级支持 (ES7210 + ES8311 + I2S)
│   └── CMakeLists.txt / idf_component.yml
│
└── spiffs_content/              ← 模型存放目录
    └── README.md                   "把 .tflite 放这里"
```

## 快速开始

### 1. 准备模型

从训练管道导出两个文件:

| 文件 | 说明 |
|------|------|
| `your_wakeword.tflite` | TFLite backbone 模型 (INT8 量化) |
| `head.h` | MultiProto head 权重 (训练脚本自动生成) |

```bash
# 放模型
cp your_wakeword.tflite spiffs_content/

# 替换 head
cp head.h main/head.h
```

### 2. 修改配置 (可选)

打开 `main/main.cpp`，搜索 `WAKE_WORD`:

```cpp
#define WAKE_WORD  "你的唤醒词"   // 仅用于日志显示
```

调整检测参数 (同文件 `detect_loop` 函数内):

```cpp
recognizer_config_t cfg = {
    .model_path  = "/spiffs",
    .threshold   = 0.70f,     // 概率阈值 ★ 需要根据你的模型实测调
    .l1_enabled  = 0,         // 连续帧 (关)
    .l2_enabled  = 0,         // peak/bg 比 (可开)
    .l3_enabled  = 0,         // 1.5s 冷却 (可开)
    .l4_enabled  = 0,         // burst 阻断 (可开)
    .l5_enabled  = 1,         // 能量跳变 (建议开)
    .l5_delta    = 200.0f,    // RMS 跳变阈值 200-1200
    .postprocess = kws_postprocess,  // head 回调 (必填)
};
```

### 3. 编译

```bash
# Windows (双击或命令行)
build.bat

# 或直接用 idf.py
idf.py build
```

首次编译 ~5-10 分钟 (需下载依赖: esp-tflite-micro, esp-dsp, esp-sr)。

### 4. 烧录

```bash
# Windows (修改 COM 口)
flash.bat

# 或直接用 esptool
python -m esptool --chip esp32s3 -p COM5 -b 460800 \
  write-flash 0x10000 build/factory_01.bin 0x957000 build/models.bin
```

### 5. 监控日志

```bash
# 使用项目自带的监控脚本 (在仓库根目录)
python wake_monitor.py COM5

# 或使用 ESP-IDF 自带
idf.py monitor
```

对着设备说唤醒词，观察日志中的 `prob` 和 `rms` 值。

## 换自己的模型

只需提供**两个文件**:

### 1. `.tflite` 模型

放入 `spiffs_content/`，文件名 (不含扩展名) 即唤醒词标识符。

模型要求:
- 输入: `[1, 98, 32]` mel 频谱 (time-major, INT8 量化)
- 输出: `[1, 256]` backbone embedding (INT8)
- 算子: Conv2D, DepthwiseConv2D, Pad, Add, Mul, Reshape, Transpose, Slice, Sum, Concatenation
- SRNN / Causal DS-TCN backbone

### 2. `head.h`

MultiProto head 权重，由训练脚本自动生成。包含:
```c
#define KWS_HEAD_K 5       // 原型数量
#define KWS_HEAD_D 256     // 特征维度
KWS_ABS_TEMP               // 温度参数
KWS_FC_B, KWS_FC_W[K]     // 线性分类器
KWS_PROTO_NORM[K][D]      // L2-归一化原型向量
```

**head.h 必须和 .tflite 配套使用** (同一次训练导出)！

## L1-L5 检测管线

从 Android `DetectionLogic.java` v9.0 完整移植，C 实现。

| 层 | 作用 | 原理 | 建议 |
|----|------|------|------|
| **L1** | 连续帧确认 | N 帧连续超阈值 (MAX_GAP=2) | 关 — ESP32 推理 ~310ms/帧，唤醒词 ~500ms 只够 1 帧 |
| **L2** | 概率峰/底比 | 1500ms 窗口 peak / bg EMA > 3× | 可开 — 模型概率稳定时核心过滤器 |
| **L3** | 冷却 | 1.5s 内不重复触发 | 可开 — 防止一次说词多次回调 |
| **L4** | burst 阻断 | 3次/3秒 → 阻断 5 秒 | 按需 — 防止音箱循环播放误触 |
| **L5** | 能量跳变 (两阶段) | L5a: 当前 RMS > 历史 0.5-2s 最低 RMS + delta → 疑似人声<br>L5b: 等 400ms 后尾音 RMS 降回安静水平 → 确认唤醒 | **开** — 过滤音乐/视频/TV (核心) |

### L5 参数调优

L5 是最重要的误触发过滤器。只有一个参数 `l5_delta`:

```
l5_delta 越小 → 越灵敏 (安静环境用)
l5_delta 越大 → 越严格 (嘈杂环境用)
```

| 环境 | 建议 delta | 安静 RMS | 说话 RMS |
|------|-----------|---------|---------|
| 安静房间 | 200-400 | 20-50 | 200-500 |
| 普通室内 | 400-800 | 50-150 | 300-800 |
| 嘈杂/音乐 | 800-1200 | 100-300 | 500-1500 |

**调参方法**: 看日志中的 `L5 quiet/steady/JUMP` 输出。如果你说唤醒词时被 `L5 quiet` 或 `L5 steady` 拦截，说明 delta 太大，降低。如果总是 `L5 JUMP` 后触发，但你没说话，说明 delta 太小，增大。

## 麦克风增益

在 `main/bsp_board.h`:

```c
#define RECORD_VOLUME   (36.0)   // ES7210 麦克风增益 (dB)
```

| 增益 | 效果 |
|------|------|
| 24-30 dB | 近距离 (0-30cm)，低底噪 |
| 30-36 dB | 中等距离 (0.3-1m)，通用 |
| 36-42 dB | 远距离 (1m+)，底噪也会放大 |

## 语音命令 (MultiNet)

Demo 内置了 4 个中文语音命令，唤醒后生效:

| 命令 | 效果 | SDK Config |
|------|------|------------|
| "灯光变成红色" | LED 亮红 | `deng guang bian cheng hong se` |
| "灯光变成蓝色" | LED 亮蓝 | `deng guang bian cheng lan se` |
| "灯光变成绿色" | LED 亮绿 | `deng guang bian cheng lv se` |
| "灯光变成白色" | LED 亮白 | `deng guang bian cheng bai se` |

**修改/增加命令**: 编辑 `sdkconfig.defaults` 中的 `CONFIG_CN_SPEECH_COMMAND_ID*`，然后在 `main.cpp` 中:
1. `g_cmd_name()` 函数 — 修改命令显示名
2. `led_on_cmd()` 函数 — 修改 LED 行为
3. `esp_mn_commands_add()` 调用 — 添加拼音别名

**不需要语音命令?** 删除 `sdkconfig.defaults` 中的 `CONFIG_SR_*` 和 `CONFIG_CN_SPEECH_COMMAND_*` 配置，从 `main.cpp` 中删除 MultiNet 相关代码 (`mn_init()`, `g_multinet`, `g_mn_data`, `STATE_COMMAND` 分支)。

## TEST_MODE

`main.cpp` 顶部有一个 `TEST_MODE` 宏:

| 值 | 模式 | 说明 |
|----|------|------|
| 0 | 实时检测 (默认) | 麦克风 → 推理 → 唤醒 |
| 1 | PCM 测试 | 用 `test_pcm.h` 中的预录 PCM 做单帧推理 |
| 2 | PCM 录音 | 录制 5 秒音频，以十进制数字文本输出 (方便采集测试数据) |

## 常见问题

### Q: 安静时 prob 很高 (0.3-0.5)

这说明模型对底噪也有较高的输出。**方案 1**: 打开 L2 (peak/bg 比)，它会学习底噪概率并自动提高门槛。**方案 2**: 提高 threshold。

### Q: 唤醒词检测不到 (prob 始终 < threshold)

检查:
1. `head.h` 是否和 `.tflite` 配套
2. 麦克风增益是否太低 — 看日志中的 RMS 值
3. threshold 是否太高 — 从 0.3 开始试
4. 用 TEST_MODE 2 录音，在 PC 侧验证模型

### Q: 一直误触发

1. 先 clean 再编译 (build 缓存可能有问题)
2. 打开 L2 + L3
3. 增大 `l5_delta` 到 800+
4. 提高 threshold

### Q: 如何训练自己的唤醒词

训练管道在 `onnx-wakeword/` 目录的父级 (`wakeword-builder/`)，包含数据准备、模型训练、导出脚本。

### Q: 编译报错 "No module named esp_idf_monitor"

需要在 ESP-IDF 环境中运行。执行:
```bash
C:\esp\v6.0.1\esp-idf\export.bat
idf.py build
```

或直接双击 `build.bat` (已自动设置环境)。

## 依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| esp-tflite-micro | 1.3.5 | TFLite 推理引擎 |
| esp-dsp | ^1.4.0 | RFFT (Mel 频谱提取) |
| esp-sr | ^2.1.5 | MultiNet 语音命令 (可选) |
| esp_codec_dev | ^1.0.0 | ES7210/ES8311 音频驱动 |
| led_strip | ^2.5.0 | WS2812 LED 控制 |
| esp_io_expander_tca95xx_16bit | ^2.0.1 | IO 扩展 (HMI-DevKit) |


