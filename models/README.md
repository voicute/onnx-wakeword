# Model Directory

[中文文档](README_CN.md)

Trained ONNX keyword spotting and wake word models.

## Baseline (R0) vs false-trigger optimized (R1)

Every production keyword ships as a pair, and the file suffix tells you which edition it is:

| Suffix | Edition | What it is |
|:------:|---------|------------|
| `_r0` | **Baseline** | The first-pass model, before false-trigger optimization. Recall margin is slightly higher, but it false-triggers far more often. Shipped for side-by-side comparison — and as a fallback if the optimized model feels hard to wake on your hardware. |
| `_r1` | **False-trigger optimized** | Retrained with hard negatives mined from that keyword's own false-trigger corpus: **~89–99% fewer false triggers, recall within ~1pp of baseline**. **Use this edition in production.** |

Tables below use the same convention: *(optimized)* = R1, *(baseline)* = R0, *(multi, R0)* = a multi-keyword package still in baseline form. Demo models without a suffix (e.g. `manbo.onnx`, the v9.3 `multi_*.zip` packages) predate the pair convention and are baseline models. See the [optimization pairs table](#false-trigger-optimization-pairs-7-languages) for measured before/after numbers.

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

## ESP32 / TFLite Demo Bundles (optimized single keyword)

Each bundle is a ZIP with the INT8 TFLite backbone, the matching float MultiProto head (`head.h`), and a `model_info.json` (mel config + threshold) — everything the ESP32 SDK needs. It is exported from the **same checkpoint as the optimized ONNX model** listed above, so the ESP32 build behaves identically to the Android/Web/Python ONNX demos.

| Directory | Package | Keyword | Window |
|:---:|---------|---------|:---:|
| `en/` | `heyjarvis_r1_tflite.zip` | Hey Jarvis | 98 frames (1.012 s) |
| `zh/` | `nihaoxiaona_r1_tflite.zip` | 你好小娜 | 98 frames (1.012 s) |
| `de/` | `apfelstrudel_r1_tflite.zip` | Apfelstrudel | 98 frames (1.012 s) |
| `fr/` | `salutnova_r1_tflite.zip` | Salut Nova | 98 frames (1.012 s) |
| `ja/` | `mirai_r1_tflite.zip` | みらい | 98 frames (1.012 s) |
| `es/` | `hola_orion_r1_tflite.zip` | Hola Orion | 150 frames (1.532 s) |
| `hi/` | `hallo_tarak_r1_tflite.zip` | हैलो तारक | 150 frames (1.532 s) |

```text
<word>_r1_tflite.zip
├── <word>.tflite      INT8 backbone, NHWC input [1, mel_time, 1, 32] (~66 KB)
├── head.h             float MultiProto head (K=5) + mel config defines
└── model_info.json    mel_time, layout, threshold — firmware must match these
```

**Always swap the `.tflite` and `head.h` together** — the head calibration constants are matched to the backbone. Set `KWS_MEL_TIME` (and the tensor arena) to the window in `model_info.json`; a mismatch fails fast at load. German board-verified demo models (`Hallo Lumo`, multi-keyword `Licht …`) already ship in `de/` alongside the bundle above. Integration guides: [Arduino route](../esp32/arduino/Voicute/README.md) · [ESP-IDF route](../esp32/idf/voicute/README.md).

### Baseline (pre-optimization) ESP32 bundles

The `_r0_tflite.zip` bundles are the **baseline (higher-recall, higher false-trigger) models** for the same keywords — exported from the same checkpoints as the `*_r0.onnx` / `*_v10.0_basic.zip` models above. Pick the R1 bundle for a quieter product; pick the R0 bundle if R1 feels hard to wake. Same ZIP layout, same window lengths per language.

| Directory | Package | Keyword | Window |
|:---:|---------|---------|:---:|
| `en/` | `heyjarvis_r0_tflite.zip` | Hey Jarvis | 98 frames (1.012 s) |
| `zh/` | `nihaoxiaona_r0_tflite.zip` | 你好小娜 | 98 frames (1.012 s) |
| `de/` | `apfelstrudel_r0_tflite.zip` | Apfelstrudel | 98 frames (1.012 s) |
| `fr/` | `salutnova_r0_tflite.zip` | Salut Nova | 98 frames (1.012 s) |
| `ja/` | `mirai_r0_tflite.zip` | みらい | 98 frames (1.012 s) |
| `es/` | `hola_orion_r0_tflite.zip` | Hola Orion | 150 frames (1.532 s) |
| `hi/` | `hallo_tarak_r0_tflite.zip` | हैलो तारक | 150 frames (1.532 s) |

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
