# Voicute Wake Word API

Voicute 是 ESP32 唤醒词检测通用库。

## 架构

```
main.cpp (应用层)
  │
  │  注入: postprocess 回调
  ▼
recognizer (通用推理管线)
  ├─ mel_extractor    PCM → mel 声谱图 (16kHz, 32mel, 98帧)
  ├─ TFLite Invoke    INT8 backbone 推理 (esp-tflite-micro)
  ├─ postprocess()    模型 head → prob (调用方注入, 如 kws_postprocess)
  └─ detect_logic     L1-L5 检测管线 → VOICE_EVT_AWAKEN 回调
```

## 快速开始

```c
#include "recognizer.h"
#include "head.h"              // 你的模型 head 权重
#include "kws_postprocess.h"   // MultiProto head 实现 (或用你自己的)

// 1. 唤醒回调
static void on_wake(voice_event_t ev, voice_evt_data_t d, void *user) {
    if (ev == VOICE_EVT_AWAKEN)
        ESP_LOGI("APP", ">>> WAKE: %s", (const char *)user);
}

// 2. 初始化
void app_main() {
    recognizer_config_t cfg = {
        .model_path  = "/spiffs",        // SPIFFS 模型路径
        .threshold   = 0.70f,            // 概率阈值
        .l1_enabled  = 0,                // L1: 连续帧 (关 — ESP32 推理太慢)
        .l2_enabled  = 0,                // L2: peak/bg 比 > 3×
        .l3_enabled  = 0,                // L3: 1.5s 冷却
        .l4_enabled  = 0,                // L4: burst 阻断
        .l5_enabled  = 1,                // L5: 能量跳变 (过滤音乐/视频)
        .l5_delta    = 200.0f,           // L5 RMS 增量阈值
        .postprocess = kws_postprocess,  // ★ 必填: head 回调
    };

    recognizer_start(&cfg);
    recognizer_register_callback(0, on_wake, (void *)"你的唤醒词");

    // 3. 主循环
    while (1) {
        int16_t pcm[16192];  // 从麦克风采集
        float rms = compute_rms(pcm, 16192);
        int64_t now_ms = esp_timer_get_time() / 1000;

        // 非推理帧也记录 RMS 到 L5 历史 (用于能量跳变检测)
        recognizer_evaluate_silence(rms, now_ms);

        // 推理帧
        recognizer_run_frame(pcm, rms, now_ms);
    }
}
```

## API

### recognizer_start

```c
void recognizer_start(const recognizer_config_t *cfg);
```

初始化模型、mel 提取器、L1-L5 检测管线。调用一次。

### recognizer_run_frame

```c
void recognizer_run_frame(const int16_t *pcm, float rms, int64_t now_ms);
```

处理一帧音频 (16192 采样点)，运行完整推理 + 检测管线。触发时调用注册的回调。

| 参数 | 说明 |
|------|------|
| `pcm` | 16192 采样点, int16, 16kHz 单通道 |
| `rms` | 当前帧 RMS 值, `sqrt(mean(s²))` |
| `now_ms` | 当前时间毫秒, `esp_timer_get_time() / 1000` |

### recognizer_evaluate_silence

```c
void recognizer_evaluate_silence(float rms, int64_t now_ms);
```

**不运行推理**，仅将 RMS 记录到 L5 历史。用于推理帧之间的间隙帧，确保 L5 能量跳变检测有充足的历史数据。

推理帧 (~310ms 一次) 之外，每 ~50ms 调用一次，维持密集的 RMS 历史。

### recognizer_feed_rms

```c
void recognizer_feed_rms(float rms, int64_t now_ms);
```

同 `recognizer_evaluate_silence`，但同时在 L5b post-speech 检测中检查 pending 状态。用于高频率的音频数据（如 MultiNet 命令模式的 32ms 帧）。

### recognizer_register_callback

```c
void recognizer_register_callback(int idx, voice_event_callback_t cb, void *user);
```

注册唤醒回调。`idx` 为模型索引 (多模型时区分)。`user` 透传给回调。

### recognizer_get_last_prob

```c
float recognizer_get_last_prob(void);
```

返回最近一次推理的概率值 (用于监控/调试)。

### recognizer_config_t

```c
typedef struct {
    char  model_path[64];          // SPIFFS 模型路径, 如 "/spiffs"
    float threshold;               // 概率阈值 (0.0 到 1.0)
    int   l1_enabled;              // L1: 连续帧确认
    int   l2_enabled;              // L2: peak/bg 比值
    int   l3_enabled;              // L3: 1.5s 冷却
    int   l4_enabled;              // L4: burst 阻断
    int   l5_enabled;              // L5: 能量跳变 (两阶段)
    float l5_delta;                // L5 RMS 增量阈值 (0 = 使用默认 1200)

    kws_postprocess_fn postprocess;  // ★ 必填: 模型 head 回调
} recognizer_config_t;
```

### kws_postprocess_fn

```c
typedef float (*kws_postprocess_fn)(const int8_t *backbone_out,
                                     float out_scale, int out_zero);
```

模型 head 回调。TFLite Invoke 后调用，把 backbone INT8 输出转为概率 [0,1]。

| 参数 | 说明 |
|------|------|
| `backbone_out` | TFLite 输出张量 data (int8, 通常 [256]) |
| `out_scale` | 输出张量量化 scale |
| `out_zero` | 输出张量量化 zero_point |
| 返回 | 唤醒概率 [0, 1] |

参考实现见 [kws_postprocess.h](kws_postprocess.h) — MultiProto head (L2-norm + cosine + linear + sigmoid)。

### voice_event_t

```c
typedef enum {
    VOICE_EVT_AWAKEN,       // 唤醒词检测到
    VOICE_EVT_CMD,          // 语音命令识别 (预留)
    VOICE_EVT_CMD_TIMEOUT   // 命令超时 (预留)
} voice_event_t;
```

## 接入新模型

需要提供两样东西：

### 1. `.tflite` 模型文件

放入 `spiffs_content/` 目录，编译时自动打包到 SPIFFS 分区。

模型要求:
- 输入: `[1, 98, 32]` mel 频谱 (time-major, INT8)
- 输出: `[1, D]` backbone embedding (INT8, 通常 D=256)
- 算子: Conv2D, DepthwiseConv2D, Pad, Add, Mul, Reshape, Transpose, Slice, Sum, Concatenation

### 2. postprocess 实现

**如果模型最后一层已是 sigmoid** (输出单个 prob 值):
```c
float my_postprocess(const int8_t *out, float scale, int zp) {
    return ((float)out[0] - zp) * scale;
}
```

**如果模型输出 embedding** (如 [256] 向量, 需要用 MultiProto head):
```c
// 方式 A: 直接用 kws_postprocess.h (推荐)
#include "head.h"            // 训练导出的 MultiProto 权重
#include "kws_postprocess.h" // 标准 head 实现
// postprocess = kws_postprocess

// 方式 B: 自己实现
float my_postprocess(const int8_t *out, float scale, int zp) {
    // 1. dequantize: (out[i] - zp) * scale → float feat[256]
    // 2. L2-normalize feat
    // 3. cosine similarity to prototypes
    // 4. linear classifier + sigmoid
    // 5. return prob
}
```

## 模型加载

`model_loader_init()` 扫描 SPIFFS 目录中所有 `.tflite` 文件，自动加载。

- 文件名 (不含 `.tflite`) 作为 wake_word 标识符
- 支持最多 3 个模型同时检测 (`MAX_WAKE_WORDS`)
- 模型加载到 PSRAM，Tensor Arena 96KB 优先用内部 SRAM

## L1-L5 检测管线

从 Android `DetectionLogic.java` v9.0 完整移植到 C。

| 层 | 作用 | 原理 | ESP32 建议 |
|----|------|------|------------|
| **L1** | 连续帧确认 | N 帧连续超阈值 (MAX_GAP=2) | **关** — 推理 ~310ms/帧, 不够连续 |
| **L2** | 概率峰/底比 | 1500ms peak > bg EMA × 3 | **可开** — 核心概率过滤 |
| **L3** | 冷却 | 1.5s 不重复触发 | **可开** — 防双触 |
| **L4** | burst 阻断 | 3次/3s → 阻断 5s | **按需** — 防循环误触 |
| **L5** | 能量跳变 | 两阶段: L5a 跳变检测 → L5b 后静音确认 | **开** — 核心能量过滤 |

### L5 两阶段详解

**L5a — 能量跳变检测**
```
如果 curRms > preMin(0.5-2s前) + l5_delta → 疑似人声 → 进入 pending
否则 → 平稳噪声 → 拦截
```

**L5b — 后静音确认** (400ms 后触发)
```
如果 尾音 RMS < preMin × 2.5 → 说完了, 安静了 → 确认唤醒
否则 → 持续噪声 (音乐/视频) → 拦截
```

## 目录约定

```
esp32/
├── components/
│   └── voicute/       ← 通用库 (不依赖具体模型)
│       ├── recognizer.cpp/h     推理管线主轴
│       ├── model_loader.cpp/h   TFLite 模型加载
│       ├── mel_extractor.c/h    Mel 频谱提取
│       ├── detect_logic.c/h     L1-L5 检测管线
│       ├── kws_postprocess.h    MultiProto head 参考实现
│       └── ...
│
├── main/              ← 应用层 (Demo)
│   ├── main.cpp             主循环 + 唤醒回调
│   └── head.h               模型 head 权重 (★ 用户须替换)
│
└── spiffs_content/    ← 模型文件目录
    └── *.tflite
```
