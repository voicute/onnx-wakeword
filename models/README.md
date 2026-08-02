# Model Directory

Trained ONNX keyword spotting and wake word models.

## Shared Model

`melspectrogram.onnx` is the universal audio preprocessing module — **provided in this repo**, no additional download needed.

## Demo Models

Current models from [voicute.com](https://www.voicute.com):

### English (`en/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| Hey Friday | `hey_friday.onnx` | v9.3 |

> More languages available at [voicute.com](https://www.voicute.com). Supports **150+ languages**.

### Chinese (`zh/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| 曼波 | `manbo.onnx` | v9.3 |
| 曼波 (voice) | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `kaishibofang.onnx` | v9.3 |
| 来福 | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `gugugaga.onnx` | v9.3 |
| 小娜 / 你好小娜 / 小娜小娜 | `multi_xiaona.onnx` | v9.3-multi |

> **Voice edition (语音定制版)**: Standard TTS training + real user recordings (50x weight) + 80 epochs of training. Achieves ~17% lower false-trigger rate compared to the standard edition, with more stable recognition for specific user pronunciation patterns.

## How to Use

### Multi-keyword (single model, recommended)

One ONNX model outputs N keyword probabilities in a single inference. Model size: 130–167 KB for 2–10 keywords. Supports Android / Web / Python / ESP32.

```json
{
  "model_type": "multi_keyword",
  "keywords": ["hey friday", "turn on light"],
  "model_file": "model.onnx",
  "mel_time": 98,
  "n_mels": 32,
  "cons_frames": 2
}
```

### Single keyword (legacy)

```json
{
  "wake_word": "hey friday",
  "model_file": "model.onnx",
  "emb_frames": 1,
  "cons_frames": 3
}
```

## Version History

| Version | Changes |
|:-------:|---------|
| v9.3 | Reduced false-trigger rate, expanded voice coverage |
| v9.2 | Expanded training data diversity |
| v9.1 | Improved far-field recognition |
| v9.0 | New Causal TCN architecture |

---

## 中文说明

### 演示模型 (`zh/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| 曼波 | `manbo.onnx` | v9.3 |
| 曼波 (语音定制) | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `kaishibofang.onnx` | v9.3 |
| 来福 | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `gugugaga.onnx` | v9.3 |
| 小娜 / 你好小娜 / 小娜小娜 | `multi_xiaona.onnx` | v9.3-multi |

> **语音定制版 (voice)**: 标准 TTS 基础上加入真人录音 x50 权重 + 80 epoch 训练，误触发率比标准版低约 17%。

### 版本说明

| 版本 | 主要更新 |
|------|------|
| v9.3 | 优化误唤醒率 |
| v9.2 | 扩展训练数据 |
| v9.1 | 增强远场识别 |
| v9.0 | Causal TCN 新架构 |
