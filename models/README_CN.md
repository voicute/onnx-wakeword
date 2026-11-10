# 模型目录

[English](README.md)

训练好的 ONNX 唤醒词 / 关键词识别模型。

## 基线 (R0) 与误触发优化版 (R1)

每个生产关键词都以一对模型发布，文件后缀标明版本：

| 后缀 | 版本 | 说明 |
|:----:|------|------|
| `_r0` | **基线 (baseline)** | 第一版模型，未做误触发优化。召回余量略高，但误触发远多于优化版。用于前后对比，以及在您的硬件上觉得优化版难唤醒时的退路。 |
| `_r1` | **误触发优化版** | 用该词自己的误触发语料挖掘难负样本重训：**误触发减少约 89–99%，召回与基线相差不到 1 个百分点**。**正式产品请用这个版本。** |

文件无 `_r0`/`_r1` 后缀的（如 `manbo.onnx`、v9.3 `multi_*.zip`）是早于成对发布惯例的早期演示模型，属基线。下文表格沿用同一约定：*(优化版)* = R1，*(基线)* = R0，*(多关键词, R0)* = 尚未优化的多关键词包。实测前后对比数据见[误触发优化对比表](#误触发优化对比模型-7-种语言)。

## 共享模型

`melspectrogram.onnx` 是通用音频预处理模块 —— **已随仓库提供**，无需额外下载。

## 演示模型

来自 [voicute.com](https://www.voicute.com) 的当前模型：

### 中文 (`zh/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| 曼波 | `manbo.onnx` | v9.3 |
| 曼波 (语音定制) | `manbo_voice_model.onnx` | v9.3-voice |
| 你好电脑 | `nihaodiannao.onnx` | v9.3 |
| 开始播放 | `kaishibofang.onnx` | v9.3 |
| 来福 | `laifu.onnx` | v9.3 |
| 咕咕嘎嘎 | `gugugaga.onnx` | v9.3 |
| 小娜 / 你好小娜 / 小娜小娜 (多关键词) | `multi_xiaona_v9.3.zip` | v9.3-multi |
| 播放控制指令 ×10 (多关键词) | `multi_commands_v9.3.zip` | v9.3-multi |

> **语音定制版 (voice)**: 标准 TTS 基础上加入真人录音加权训练，误触发率比标准版低约 17%，对特定用户发音习惯识别更稳。

> 自定义模型可在 [voicute.com](https://www.voicute.com) 定制。当前支持语言：**中文、英语、法语、德语、日语、西班牙语、印地语**。

### 英语 (`en/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| Hey Jarvis (优化版) | `heyjarvis_r1.onnx` | v9.3 |
| Hey Jarvis / Jarvis / Hi Jarvis (多关键词) | `multi_jarvis_v9.3.zip` | v9.3-multi |
| Hey Friday | `hey_friday.onnx` | v9.3 |

### 日语 (`ja/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| みらい (优化版) | `mirai_r1.onnx` | v9.3 |

### 德语 (`de/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| Apfelstrudel (优化版) | `apfelstrudel_r1.onnx` | v9.3 |
| Martina / Tina / Hey Tina (多关键词) | `multi_martina_v9.3.zip` | v9.3-multi |

### 法语 (`fr/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| Salut Nova (优化版) | `salutnova_r1.onnx` | v9.3 |
| Monsieur Sadin | `monsieur_sadin.onnx` | v9.3 |
| Croissant | `croissant.onnx` | v9.3 |
| Lumière rouge / bleue / verte / blanche (多关键词, R0) | `multi_N4_f97a7e_v10.0.zip` | v10.0-multi |

### 西班牙语 (`es/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| Hola Orion (优化版) | `Hola Orion_v10.0_basic_r1.zip` | v10.0 |
| Hola Orion (基线) | `Hola Orion_v10.0_basic.zip` | v10.0 |
| Luz roja / azul / verde / blanca (多关键词, R0) | `multi_N4_990ee7_v10.0.zip` | v10.0-multi |

### 印地语 (`hi/`)

| 关键词 | 模型文件 | 版本 |
|--------|---------|:---:|
| हैलो तारक (优化版) | `हैलो तारक_v10.0_basic_r1.zip` | v10.0 |
| हैलो तारक (基线) | `हैलो तारक_v10.0_basic.zip` | v10.0 |
| लाल / नीली / हरी / सफ़ेद बत्ती (多关键词, R0) | `multi_N4_8a1d43_v10.0.zip` | v10.0-multi |

## 误触发优化对比模型 (7 种语言)

七个支持语言的生产关键词优化前后模型对。每对用该词交付时的**词标语料**（同语言 Common Voice 朗读语音 + 音乐 + 家用噪声/静音，训练留出；时长见行内标注），裸模型、阈值 0.5、40ms 滑窗、按触发文件计：

| 关键词 | 语言 | 语料 | 优化前 (基线) | 优化后 (优化版) | 优化前 → 后 (次/小时) | 降幅 | 召回变化 |
|--------|:---:|---:|---------|---------|---:|---:|---|
| 你好小娜 | 中文 | 16.0h | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` | 74.5 → 3.5 | −95.3% | 97.8% → 100% |
| Hey Jarvis | 英语 | 26.9h | `heyjarvis_r0.onnx` | `heyjarvis_r1.onnx` | 44.6 → 4.8 | −89.2% | 100% → 99.0% |
| Salut Nova | 法语 | 22.6h | `salutnova_r0.onnx` | `salutnova_r1.onnx` | 222.7 → 4.5 | −98.0% | 100% → 100% |
| Apfelstrudel | 德语 | 24.4h | `apfelstrudel_r0.onnx` | `apfelstrudel_r1.onnx` | 265.3 → 2.7 | −99.0% | 97.0% → 98.5% |
| みらい | 日语 | 12.1h | `mirai_r0.onnx` | `mirai_r1.onnx` | 330.9 → 5.1 | −98.5% | 97.8% → 100% |
| Hola Orion | 西班牙语 | 20.8h | `Hola Orion_v10.0_basic.zip` | `Hola Orion_v10.0_basic_r1.zip` | 456.7 → 2.9 | −99.4% | 99.8% → 99.3% |
| हैलो तारक | 印地语 | 18.5h | `हैलो तारक_v10.0_basic.zip` | `हैलो तारक_v10.0_basic_r1.zip` | 212.2 → 3.3 | −98.5% | 99.5% → 99.3% |

> 可自行验证：分别加载基线与优化版，播放音乐或视频——基线频繁误触发，优化版保持安静。**基线仅作对比，正式使用请选优化版。**
> 误触发优化（难负样本挖掘重训）已在全部 **7 种支持语言**上完成生产验证。

## ESP32 / TFLite 演示包 (优化版单关键词)

每个压缩包含 INT8 TFLite 骨干网络 + 匹配的浮点 MultiProto 头 (`head.h`) + `model_info.json`（mel 配置与阈值），即 ESP32 SDK 所需的全部文件。**与上面列出的优化版 ONNX 模型来自同一个训练 checkpoint**，因此 ESP32 上跑的模型与 Android/Web/Python 的 ONNX 演示行为一致。

| 目录 | 压缩包 | 关键词 | 识别窗 |
|:---:|---------|---------|:---:|
| `en/` | `heyjarvis_r1_tflite.zip` | Hey Jarvis | 98 帧 (1.012 s) |
| `zh/` | `nihaoxiaona_r1_tflite.zip` | 你好小娜 | 98 帧 (1.012 s) |
| `de/` | `apfelstrudel_r1_tflite.zip` | Apfelstrudel | 98 帧 (1.012 s) |
| `fr/` | `salutnova_r1_tflite.zip` | Salut Nova | 98 帧 (1.012 s) |
| `ja/` | `mirai_r1_tflite.zip` | みらい | 98 帧 (1.012 s) |
| `es/` | `hola_orion_r1_tflite.zip` | Hola Orion | 150 帧 (1.532 s) |
| `hi/` | `hallo_tarak_r1_tflite.zip` | हैलो तारक | 150 帧 (1.532 s) |

```text
<word>_r1_tflite.zip
├── <word>.tflite      INT8 骨干网络, NHWC 输入 [1, mel_time, 1, 32] (~66 KB)
├── head.h             浮点 MultiProto 头 (K=5) + mel 配置宏定义
└── model_info.json    mel_time、layout、阈值 —— 固件必须与之匹配
```

**`.tflite` 和 `head.h` 必须成对更换** —— 头部的校准常数与骨干网络配套。`KWS_MEL_TIME`（及 tensor arena）按 `model_info.json` 里的窗口设置；不匹配会在加载时快速失败。德语板上验证过的演示模型（`Hallo Lumo`、多关键词 `Licht …`）已随 `de/` 目录中的上述包一并提供。集成指南：[Arduino 路线](../esp32/arduino/Voicute/README.md) · [ESP-IDF 路线](../esp32/idf/voicute/README.md)。

### 基线版（优化前）ESP32 包

`_r0_tflite.zip` 是同样这些词的**基线模型**（召回余量略高、误触发更高），与上面的 `*_r0.onnx` / `*_v10.0_basic.zip` 同一 checkpoint 导出。产品追求安静选 R1 包；觉得 R1 难唤醒就换 R0 包。ZIP 结构相同，各语言窗口长度相同。

| 目录 | 压缩包 | 关键词 | 识别窗 |
|:---:|---------|---------|:---:|
| `en/` | `heyjarvis_r0_tflite.zip` | Hey Jarvis | 98 帧 (1.012 s) |
| `zh/` | `nihaoxiaona_r0_tflite.zip` | 你好小娜 | 98 帧 (1.012 s) |
| `de/` | `apfelstrudel_r0_tflite.zip` | Apfelstrudel | 98 帧 (1.012 s) |
| `fr/` | `salutnova_r0_tflite.zip` | Salut Nova | 98 帧 (1.012 s) |
| `ja/` | `mirai_r0_tflite.zip` | みらい | 98 帧 (1.012 s) |
| `es/` | `hola_orion_r0_tflite.zip` | Hola Orion | 150 帧 (1.532 s) |
| `hi/` | `hallo_tarak_r0_tflite.zip` | हैलो तारक | 150 帧 (1.532 s) |

## 多关键词演示包

一个模型识别 N 个关键词——单次推理输出全部关键词的概率，每新增一个词只增加约 0.8KB。压缩包**直接加载即可，无需解压**（Python / Web 引擎自动识别 ZIP 格式）：

```python
engine.load('models/zh/multi_commands_v9.3.zip', 'models/melspectrogram.onnx')
```

**多唤醒词** —— 一个唤醒词的多种叫法收进同一个模型：

| 目录 | 压缩包 | 关键词 | 大小 |
|:---:|---------|---------|:---:|
| `zh/` | `multi_xiaona_v9.3.zip` | 小娜 · 你好小娜 · 小娜小娜 | 135 KB |
| `en/` | `multi_jarvis_v9.3.zip` | Hey Jarvis · Jarvis · Hi Jarvis | 135 KB |
| `de/` | `multi_martina_v9.3.zip` | Martina · Tina · Hey Tina | 135 KB |

**语音控制** —— 十条播放器指令装进一个模型：

| 目录 | 压缩包 | 关键词 | 大小 |
|:---:|---------|---------|:---:|
| `zh/` | `multi_commands_v9.3.zip` | 播放 · 暂停 · 下一首 · 上一首 · 开始播放 · 停止播放 · 声音大一点 · 声音小一点 · 静音 · 继续播放 | 167 KB |

**灯控** —— 每种语言一个模型，四个颜色指令：

| 目录 | 压缩包 | 关键词 | 大小 |
|:---:|---------|---------|:---:|
| `es/` | `multi_N4_990ee7_v10.0.zip` | Luz roja · Luz azul · Luz verde · Luz blanca | 139 KB |
| `fr/` | `multi_N4_f97a7e_v10.0.zip` | Lumière rouge · Lumière bleue · Lumière verte · Lumière blanche | 140 KB |
| `hi/` | `multi_N4_8a1d43_v10.0.zip` | लाल बत्ती · नीली बत्ती · हरी बत्ती · सफ़ेद बत्ती | 140 KB |

> 所有**多关键词演示包都是基线 (R0) 模型**——尚未走误触发优化流程。优化流程（负样本扫描 → 全量重训）**同样支持多关键词模型**，可以按同样方式产出这些包的优化版（单关键词优化版已可对照，见上方对比表）。

## 使用方法

### 多关键词 (单模型, 推荐)

一个 ONNX 模型单次推理输出 N 个关键词的概率。模型大小：2–10 词 130–167 KB。支持 Android / Web / Python。可加载的示例见上方演示包。

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

### 单关键词 (旧式)

```json
{
  "wake_word": "hey friday",
  "model_file": "model.onnx",
  "emb_frames": 1,
  "cons_frames": 3
}
```

## 版本历史

| 版本 | 主要更新 |
|:---:|---------|
| v10.0 | 长词自适应识别窗, +35% 语速覆盖 |
| v9.3 | 优化误唤醒率 |
| v9.2 | 扩展训练数据 |
| v9.1 | 增强远场识别 |
| v9.0 | Causal TCN 新架构 |

> 内置演示模型仍为 v9.3（mel_time=98）。v10.0 特性适用于新训练的模型，引擎端自动兼容两种窗长。

