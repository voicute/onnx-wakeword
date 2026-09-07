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

> Custom models are available at [voicute.com](https://www.voicute.com). Currently supported languages: **Chinese, English, French, German, and Japanese**.

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

### False-trigger optimization pairs (`zh/`)

Before/after model pairs for three production keywords, measured on the same held-out corpus (15.4h speech/music/mixed; bare model, threshold 0.5, 40ms sliding window, counted per triggered file):

| Keyword | Before (baseline) | After (optimized) | Before → After (triggers/h) | Reduction | Recall |
|---------|---------|---------|---:|---:|---|
| 你好小娜 | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` | 221.9 → 10.0 | −95.5% | 100% → 100% |
| 小娜 | `xiaona_r0.onnx` | `xiaona_r1.onnx` | 329.5 → 7.3 | −97.8% | 98.5% → 98.3% |
| 豆包豆包 | `doubaodoubao_r0.onnx` | `doubaodoubao_r1.onnx` | 81.3 → 7.2 | −91.2% | 100% → 100% |

> Verify it yourself: load the baseline and the optimized model side by side, play music or a video — the baseline fires repeatedly, the optimized model stays quiet. **The baseline is for comparison only; use the optimized model in production.** False-trigger optimization (hard negative mining retrain) is currently production-verified on Chinese keywords; other languages are in development & testing.

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

### 误触发优化对比模型 (`zh/`)

三个生产关键词的优化前后模型对，同一留出语料实测（15.4 小时语音/音乐/混音；裸模型、阈值 0.5、40ms 滑窗、按触发文件计）：

| 关键词 | 优化前 (基线) | 优化后 (优化版) | 优化前 → 后 (次/小时) | 降幅 | 召回变化 |
|---------|---------|---------|---:|---:|---|
| 你好小娜 | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` | 221.9 → 10.0 | −95.5% | 100% → 100% |
| 小娜 | `xiaona_r0.onnx` | `xiaona_r1.onnx` | 329.5 → 7.3 | −97.8% | 98.5% → 98.3% |
| 豆包豆包 | `doubaodoubao_r0.onnx` | `doubaodoubao_r1.onnx` | 81.3 → 7.2 | −91.2% | 100% → 100% |

> 可自行验证：分别加载优化前基线与优化版，播放音乐或视频——基线频繁误触发，优化版保持安静。**基线仅作对比，正式使用请选优化版。**
> 误触发优化（难负样本挖掘重训）目前已在**中文**关键词上完成生产验证；其他语言开发测试中。

### 版本说明

| 版本 | 主要更新 |
|------|------|
| v9.3 | 优化误唤醒率 |
| v9.2 | 扩展训练数据 |
| v9.1 | 增强远场识别 |
| v9.0 | Causal TCN 新架构 |
