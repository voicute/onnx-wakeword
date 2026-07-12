# Voicute Wake Word API

voicute 是 ESP32 唤醒词检测通用库，位于 `voice_engine/components/voicute/`。

## 架构

```
main.cpp (应用层)
  │
  │  注入: postprocess 回调 + 模型数据
  ▼
recognizer (通用推理管线)
  ├─ mel_extractor    PCM → mel 声谱图 (16kHz, 32mel, 98帧)
  ├─ TFLite Invoke    INT8 backbone 推理
  ├─ postprocess()    模型 head → prob (调用方注入)
  └─ detect_logic     L1-L5 检测管线 → WAKE 回调
```

## 快速开始

```c
#include "recognizer.h"
#include "my_head.h"      // 你的 postprocess 实现
#include "my_model.h"     // 你的模型 C 数组 (g_model_data, g_model_len)

// 1. 唤醒回调
static void on_wake(voice_event_t ev, voice_evt_data_t d, void *user) {
    if (ev == VOICE_EVT_AWAKEN)
        ESP_LOGI("APP", ">>> WAKE");
}

// 2. 初始化和主循环
void app_main() {
    recognizer_config_t cfg = {
        .model_path   = "/spiffs",       // SPIFFS 路径
        .threshold    = 0.40f,           // 概率阈值
        .l1_enabled   = 0,               // L1: 连续帧 (ESP32 推理太慢, 建议关)
        .l2_enabled   = 1,               // L2: peak/bg 比 > 3×
        .l3_enabled   = 1,               // L3: 1.5s 冷却
        .l4_enabled   = 0,               // L4: burst 检测
        .l5_enabled   = 1,               // L5: 能量跳变 (过滤音乐/视频)
        .l5_delta     = 200.0f,          // L5 RMS 增量阈值

        // ▼ 注入你的模型 ▼
        .postprocess     = my_postprocess,   // 必填: head 回调
        .compiled_model  = g_model_data,     // 编译模型 (和 SPIFFS 二选一)
        .compiled_len    = g_model_len,
    };

    recognizer_start(&cfg);
    recognizer_register_callback(0, on_wake, NULL);

    // 主循环: 每帧喂 16192 采样点 PCM (16kHz, int16)
    while (1) {
        // 从麦克风 / ring buffer 取 16192 samples
        int64_t now_ms = esp_timer_get_time() / 1000;
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

处理一帧音频，每帧调用一次。

| 参数 | 说明 |
|------|------|
| `pcm` | 16192 采样点, int16, 16kHz 单通道 |
| `rms` | 当前帧 RMS 值, `sqrt(mean(s²))` |
| `now_ms` | 当前时间毫秒, `esp_timer_get_time() / 1000` |

### recognizer_register_callback

```c
void recognizer_register_callback(int idx, voice_event_callback_t cb, void *user);
```

注册唤醒回调。`idx` 为模型索引（多模型时区分）。

### recognizer_config_t

```c
typedef struct {
    char  model_path[64];       // SPIFFS 模型路径
    float threshold;            // 概率阈值 (默认 0.40)
    int   l1_enabled;           // 连续帧确认
    int   l2_enabled;           // peak/bg 比值
    int   l3_enabled;           // 冷却 1.5s
    int   l4_enabled;           // burst 阻断
    int   l5_enabled;           // 能量跳变
    float l5_delta;             // L5 RMS 增量 (默认 200)

    kws_postprocess_fn postprocess;   // ▼ 必填: 模型 head
    const uint8_t *compiled_model;    // 编译模型数据
    size_t compiled_len;              // 编译模型长度
} recognizer_config_t;
```

### kws_postprocess_fn

```c
typedef float (*kws_postprocess_fn)(const int8_t *backbone_out,
                                     float out_scale, int out_zero);
```

模型 head 回调。TFLite Invoke 后调用，把 backbone 输出转为概率 [0,1]。

| 参数 | 说明 |
|------|------|
| `backbone_out` | TFLite 输出张量 data (int8) |
| `out_scale` | 输出张量量化 scale |
| `out_zero` | 输出张量量化 zero_point |
| 返回 | 唤醒概率 [0, 1] |

## 接入新模型

需要提供两样东西：

### 1. 模型文件 → C 数组

```bash
# 用 xxd 或 Python 把 .tflite 转成 C 数组
xxd -i your_model.tflite > model_data.h
```

生成的 `model_data.h`:
```c
unsigned char g_model_data[] = { 0x1c, 0x00, ... };
unsigned int  g_model_len = 12345;
```

### 2. postprocess 实现

取决于你的模型输出层：

**A. 模型已输出 prob (最后一层 sigmoid) → 一行就够了:**
```c
float my_postprocess(const int8_t *out, float scale, int zp) {
    return ((float)out[0] - zp) * scale;
}
```

**B. 模型输出 embedding (如 [256] 向量) → 需要实现完整 head:**
```c
float my_postprocess(const int8_t *out, float scale, int zp) {
    // 1. dequantize: (out[i] - zp) * scale → float feat[256]
    // 2. L2-normalize
    // 3. 你的 head 计算 (cosine / linear / sigmoid)
    // 4. return prob
}
```

可以参考 [manbo_kws_postprocess.h](voice_engine/components/voicute/manbo_kws_postprocess.h) 的 MultiProto head 实现。

## 模型加载优先级

1. SPIFFS 分区有 `.tflite` 文件 → 加载 SPIFFS 中的模型
2. SPIFFS 为空 + `compiled_model` 非空 → 加载编译模型
3. 都没有 → 启动失败

## L1-L5 检测管线说明

| 层 | 作用 | ESP32 建议 |
|----|------|------------|
| L1 | 连续 N 帧 > 阈值 | **建议关** (推理太慢 ~400ms/帧) |
| L2 | 1500ms 内 peak > bg×3 | **建议开** (核心过滤) |
| L3 | 1.5s 冷却防双触 | **建议开** |
| L4 | 3次/3秒 → 阻断5秒 | 按需 |
| L5 | RMS 跳变 > delta | **建议开** (过滤音乐误触) |

## 目录结构

```
voice_engine/components/voicute/   ← 通用库 (不依赖具体模型)
  recognizer.cpp/h                  推理管线
  model_loader.cpp/h                TFLite 模型加载
  mel_extractor.c/h                 mel 特征提取
  detect_logic.c/h                  L1-L5 检测
  manbo_kws_postprocess.h          manbo head (示例, 可选)
  model_data.h                     manbo 模型 (示例, 可选)
  manbo_head.h                     manbo head 权重 (示例, 可选)

main.cpp                           应用层, 注入模型
```
