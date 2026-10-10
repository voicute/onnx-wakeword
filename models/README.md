# Model Directory

Trained ONNX keyword spotting and wake word models.

## Shared Model

`melspectrogram.onnx` is the universal audio preprocessing module — **provided in this repo**, no additional download needed.

## Demo Models

Current models from [voicute.com](https://www.voicute.com):

### English (`en/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| Hey Jarvis (optimized) | `heyjarvis_r1.onnx` | v9.3 |
| Hey Jarvis / Jarvis / Hi Jarvis (multi) | `multi_jarvis_v9.3.zip` | v9.3-multi |
| Hey Friday | `hey_friday.onnx` | v9.3 |

> Custom models are available at [voicute.com](https://www.voicute.com). Currently supported languages: **Chinese, English, French, German, Japanese, Spanish, and Hindi**.

### Japanese (`ja/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| みらい (optimized) | `mirai_r1.onnx` | v9.3 |

### German (`de/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| Apfelstrudel (optimized) | `apfelstrudel_r1.onnx` | v9.3 |
| Martina / Tina / Hey Tina (multi) | `multi_martina_v9.3.zip` | v9.3-multi |

### French (`fr/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| Salut Nova (optimized) | `salutnova_r1.onnx` | v9.3 |
| Monsieur Sadin | `monsieur_sadin.onnx` | v9.3 |
| Croissant | `croissant.onnx` | v9.3 |
| Lumière rouge / bleue / verte / blanche (multi, R0) | `multi_N4_f97a7e_v10.0.zip` | v10.0-multi |

### Spanish (`es/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| Hola Orion (optimized) | `Hola Orion_v10.0_basic_r1.zip` | v10.0 |
| Hola Orion (baseline) | `Hola Orion_v10.0_basic.zip` | v10.0 |
| Luz roja / azul / verde / blanca (multi, R0) | `multi_N4_990ee7_v10.0.zip` | v10.0-multi |

### Hindi (`hi/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| हैलो तारक (optimized) | `हैलो तारक_v10.0_basic_r1.zip` | v10.0 |
| हैलो तारक (baseline) | `हैलो तारक_v10.0_basic.zip` | v10.0 |
| लाल / नीली / हरी / सफ़ेद बत्ती (multi, R0) | `multi_N4_8a1d43_v10.0.zip` | v10.0-multi |

### Chinese (`zh/`)

| Keyword | Model File | Version |
|---------|-----------|:-------:|
| 曼波 | `manbo.onnx` | v9.3 |
| 曼波 (voice) | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `kaishibofang.onnx` | v9.3 |
| 来福 | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `gugugaga.onnx` | v9.3 |
| 小娜 / 你好小娜 / 小娜小娜 (multi) | `multi_xiaona_v9.3.zip` | v9.3-multi |
| 播放控制指令 ×10 (multi) | `multi_commands_v9.3.zip` | v9.3-multi |

> **Voice edition (语音定制版)**: Standard TTS training + weighted real user recordings. Achieves ~17% lower false-trigger rate compared to the standard edition, with more stable recognition for specific user pronunciation patterns.

### False-trigger optimization pairs (7 languages)

Before/after model pairs for production keywords in **all seven supported languages**. Each pair is measured on that keyword's own held-out negative corpus (same-language Common Voice read speech + music + household noise/silence, held out from training; corpus length per row), bare model, threshold 0.5, 40ms sliding window, counted per triggered file:

| Keyword | Language | Corpus | Before (baseline) | After (optimized) | Before → After (triggers/h) | Reduction | Recall |
|---------|:-------:|---:|---------|---------|---:|---:|---|
| 你好小娜 | ZH | 16.0h | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` | 74.5 → 3.5 | −95.3% | 97.8% → 100% |
| Hey Jarvis | EN | 26.9h | `heyjarvis_r0.onnx` | `heyjarvis_r1.onnx` | 44.6 → 4.8 | −89.2% | 100% → 99.0% |
| Salut Nova | FR | 22.6h | `salutnova_r0.onnx` | `salutnova_r1.onnx` | 222.7 → 4.5 | −98.0% | 100% → 100% |
| Apfelstrudel | DE | 24.4h | `apfelstrudel_r0.onnx` | `apfelstrudel_r1.onnx` | 265.3 → 2.7 | −99.0% | 97.0% → 98.5% |
| みらい | JA | 12.1h | `mirai_r0.onnx` | `mirai_r1.onnx` | 330.9 → 5.1 | −98.5% | 97.8% → 100% |
| Hola Orion | ES | 20.8h | `Hola Orion_v10.0_basic.zip` | `Hola Orion_v10.0_basic_r1.zip` | 456.7 → 2.9 | −99.4% | 99.8% → 99.3% |
| हैलो तारक | HI | 18.5h | `हैलो तारक_v10.0_basic.zip` | `हैलो तारक_v10.0_basic_r1.zip` | 212.2 → 3.3 | −98.5% | 99.5% → 99.3% |

> Verify it yourself: load the baseline and the optimized model side by side, play music or a video — the baseline fires repeatedly, the optimized model stays quiet. **The baseline is for comparison only; use the optimized model in production.** False-trigger optimization (hard negative mining retrain) is production-verified on all seven supported languages.

## Multi-keyword Demo Packages

One model, N keywords — a single inference outputs all keyword probabilities. Each extra keyword adds only ~0.8KB; packages load **directly as ZIP, no extraction needed** (Python and Web engines detect the ZIP automatically):

```python
engine.load('models/zh/multi_commands_v9.3.zip', 'models/melspectrogram.onnx')
```

**Multi wake word** — variants of one wake word in a single model:

| Directory | Package | Keywords | Size |
|:---:|---------|---------|:---:|
| `zh/` | `multi_xiaona_v9.3.zip` | 小娜 · 你好小娜 · 小娜小娜 | 135 KB |
| `en/` | `multi_jarvis_v9.3.zip` | Hey Jarvis · Jarvis · Hi Jarvis | 135 KB |
| `de/` | `multi_martina_v9.3.zip` | Martina · Tina · Hey Tina | 135 KB |

**Voice control** — ten media commands in a single model:

| Directory | Package | Keywords | Size |
|:---:|---------|---------|:---:|
| `zh/` | `multi_commands_v9.3.zip` | 播放 · 暂停 · 下一首 · 上一首 · 开始播放 · 停止播放 · 声音大一点 · 声音小一点 · 静音 · 继续播放 | 167 KB |

**Light control** — four color commands per language, one model per language:

| Directory | Package | Keywords | Size |
|:---:|---------|---------|:---:|
| `es/` | `multi_N4_990ee7_v10.0.zip` | Luz roja · Luz azul · Luz verde · Luz blanca | 139 KB |
| `fr/` | `multi_N4_f97a7e_v10.0.zip` | Lumière rouge · Lumière bleue · Lumière verte · Lumière blanche | 140 KB |
| `hi/` | `multi_N4_8a1d43_v10.0.zip` | लाल बत्ती · नीली बत्ती · हरी बत्ती · सफ़ेद बत्ती | 140 KB |

> All **multi-keyword demo packages are baseline (R0) models** — they have not gone through the false-trigger optimization flow yet. The optimization flow (negative scan → full retrain) **supports multi-keyword models too** and can produce optimized editions of these packages the same way (single-keyword optimized editions are available for comparison, see the pairs table above).

## How to Use

### Multi-keyword (single model, recommended)

One ONNX model outputs N keyword probabilities in a single inference. Model size: 130–167 KB for 2–10 keywords. Supports Android / Web / Python. See the demo packages above for ready-to-load examples.

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
| v10.0 | Expanded training data, improved long wake word support |
| v9.3 | Reduced false-trigger rate, expanded voice coverage |
| v9.2 | Expanded training data diversity |
| v9.1 | Improved far-field recognition |
| v9.0 | New Causal TCN architecture |

> Bundled demo models remain v9.3 (mel_time=98). v10.0 features apply to newly trained models; engines auto-handle both window lengths.

---

## 中文说明

### 演示模型 (多语言)

| 关键词 | 目录 | 模型文件 | 版本 |
|--------|:---:|---------|:---:|
| 曼波 | `zh/` | `manbo.onnx` | v9.3 |
| 曼波 (语音定制) | `zh/` | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `zh/` | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `zh/` | `kaishibofang.onnx` | v9.3 |
| 来福 | `zh/` | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `zh/` | `gugugaga.onnx` | v9.3 |
| 小娜 / 你好小娜 / 小娜小娜 (多关键词) | `zh/` | `multi_xiaona_v9.3.zip` | v9.3-multi |
| 播放控制指令 ×10 (多关键词) | `zh/` | `multi_commands_v9.3.zip` | v9.3-multi |
| Hey Friday | `en/` | `hey_friday.onnx` | v9.3 |
| Hey Jarvis / Jarvis / Hi Jarvis (多关键词) | `en/` | `multi_jarvis_v9.3.zip` | v9.3-multi |
| Monsieur Sadin | `fr/` | `monsieur_sadin.onnx` | v9.3 |
| Croissant | `fr/` | `croissant.onnx` | v9.3 |
| Martina / Tina / Hey Tina (多关键词) | `de/` | `multi_martina_v9.3.zip` | v9.3-multi |

> **语音定制版 (voice)**: 标准 TTS 基础上加入真人录音加权训练，误触发率比标准版低约 17%。

### 多关键词演示包

一个模型识别 N 个关键词——单次推理输出全部关键词的概率，每新增一个词只增加约 0.8KB。压缩包**直接加载即可，无需解压**（Python / Web 引擎自动识别 ZIP 格式）：

```python
engine.load('models/zh/multi_commands_v9.3.zip', 'models/melspectrogram.onnx')
```

**多唤醒词** —— 一个唤醒词的多种叫法收进同一个模型，用户怎么说都能唤醒：

| 目录 | 压缩包 | 关键词 | 大小 |
|:---:|---------|---------|:---:|
| `zh/` | `multi_xiaona_v9.3.zip` | 小娜 · 你好小娜 · 小娜小娜 | 135 KB |
| `en/` | `multi_jarvis_v9.3.zip` | Hey Jarvis · Jarvis · Hi Jarvis | 135 KB |
| `de/` | `multi_martina_v9.3.zip` | Martina · Tina · Hey Tina | 135 KB |

**语音控制** —— 十条播放器指令装进一个模型：

| 目录 | 压缩包 | 关键词 | 大小 |
|:---:|---------|---------|:---:|
| `zh/` | `multi_commands_v9.3.zip` | 播放 · 暂停 · 下一首 · 上一首 · 开始播放 · 停止播放 · 声音大一点 · 声音小一点 · 静音 · 继续播放 | 167 KB |

### 误触发优化对比模型 (5 种语言)

五个生产关键词的优化前后模型对（每种语言一对），每对用该词交付时的**词标语料**（同语言 Common Voice 朗读语音 + 音乐 + 家用噪声/静音，训练留出；时长见行内标注），裸模型、阈值 0.5、40ms 滑窗、按触发文件计：

| 关键词 | 语言 | 语料 | 优化前 (基线) | 优化后 (优化版) | 优化前 → 后 (次/小时) | 降幅 | 召回变化 |
|---------|:---:|---:|---------|---------|---:|---:|---|
| 你好小娜 | 中文 | 16.0h | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` | 74.5 → 3.5 | −95.3% | 97.8% → 100% |
| Hey Jarvis | 英语 | 26.9h | `heyjarvis_r0.onnx` | `heyjarvis_r1.onnx` | 44.6 → 4.8 | −89.2% | 100% → 99.0% |
| Salut Nova | 法语 | 22.6h | `salutnova_r0.onnx` | `salutnova_r1.onnx` | 222.7 → 4.5 | −98.0% | 100% → 100% |
| Apfelstrudel | 德语 | 24.4h | `apfelstrudel_r0.onnx` | `apfelstrudel_r1.onnx` | 265.3 → 2.7 | −99.0% | 97.0% → 98.5% |
| みらい | 日语 | 12.1h | `mirai_r0.onnx` | `mirai_r1.onnx` | 330.9 → 5.1 | −98.5% | 97.8% → 100% |

> 可自行验证：分别加载优化前基线与优化版，播放音乐或视频——基线频繁误触发，优化版保持安静。**基线仅作对比，正式使用请选优化版。**
> 误触发优化（难负样本挖掘重训）已在全部 **5 种支持语言**上完成生产验证。

### 版本说明

| 版本 | 主要更新 |
|------|------|
| v10.0 | 长词自适应识别窗, +35% 语速覆盖 |
| v9.3 | 优化误唤醒率 |
| v9.2 | 扩展训练数据 |
| v9.1 | 增强远场识别 |
| v9.0 | Causal TCN 新架构 |

> 当前内置演示模型仍为 v9.3（mel_time=98）。v10.0 特性适用于新训练的模型，引擎端自动兼容两种窗长。
