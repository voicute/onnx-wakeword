# onnx-wakeword — Offline Wake Word & Keyword Spotting Inference Engine

[中文文档](README_CN.md)

`KWS` · `Keyword Spotting` · `Wake Word` · `Custom Wake Word` · `ONNX` · `Edge AI` · `ESP32` · `Android` · `Offline` · `Privacy-First` · `Open Source`

> **100% offline · No audio upload · Model < 130KB · ESP32 / Android / Python / Web**

onnx-wakeword is an open-source, fully offline inference engine for **wake-word detection and keyword spotting (KWS)**.

Train your own custom wake word online ([voicute.com](https://www.voicute.com)), download the resulting model, and run it locally anywhere — browser, desktop, Android, ESP32, Home Assistant. **No audio is uploaded for inference.** Once downloaded, your model works entirely offline with zero ongoing cost.

It provides the complete runtime path from audio preprocessing and Mel feature extraction to model execution, matched-head evaluation, and wake-word detection logic. The runtime is designed primarily for keyword models using a **causal temporal convolutional network (Causal TCN)** with a matched classification or prototype-based head. Fully self-developed training pipeline — not affiliated with Porcupine, OpenWakeWord, or any other project.

The repository includes ONNX runtimes for Python, Web, and Android, plus an optimized INT8 TFLite runtime for ESP32-S3. All inference runs locally without uploading audio.

Inference Pipeline

```text
Audio → Mel features → Keyword TCN model → Matched classification/prototype head → Detection logic → Result
```

---

## Quick Links

| Platform | Directory | Entry Point |
|----------|-----------|-------------|
| **Web** | [`web/`](web/) | `wakeword.js` → `VoicuteWakeWord.create()` |
| **Python** | [`python/`](python/) | `wakeword_engine.py` → `WakeWordEngine()` |
| **Android** | [`android/`](android/) | `WakeWordEngine.java` |
| **ESP32** | [`esp32/`](esp32/) | ESP-IDF component |
| **Home Assistant** | [`wyoming/`](wyoming/) | `wyoming_voicute.py` → Wyoming protocol |

---

## Training & Deployment

onnx-wakeword is a two-part system: custom keyword models trained online, then a fully offline runtime.

1. [Train your own keyword](https://www.voicute.com) — enter any wake word (Chinese, English, Japanese, French, or German), platform generates TTS training data and trains the Causal TCN model (~30 min). Download the result.
2. Download the resulting `model.zip`, load it on any supported platform — browser, desktop, Android, ESP32, Home Assistant.

Your trained model runs completely offline from this point on — no API calls, no monthly fees, no telemetry. **Your audio never leaves your device during inference.** For testing before you generate a custom keyword, run the demo models included in `models/` (中文 / English / Deutsch / Français) using the same pipeline at zero cost.

---

## Features

- **Sub-130KB models** — 25K parameters, fits ESP32 INT8 flash
- **Multi-keyword** — detect 2–10+ keywords with a single model
- **Multi-language** — Chinese, English, Japanese, French, and German
- **5-layer anti-false-trigger** — consecutive frames, peak/background ratio, cooldown, burst detection, energy jump
- **ZIP packaging** — distribute model + config as a single file
- **Home Assistant** — native Wyoming protocol service, Docker image, and HA add-on

---

## Performance

Tested on **v9.3 models**.

| Metric | Value |
|--------|-------|
| ONNX size | ~128KB (FP32) / ~74KB (INT8) |
| Desktop inference | <5ms / frame |
| ESP32-S3 TFLite Invoke (included demo) | ~51ms / frame (live AFE demo) |

The isolated `Invoke()` baseline is about 39.5–40.2 ms; the published ~51 ms figure includes the live ESP-SR AFE pipeline and its scheduling/cache contention.

### Recall

Held-out Azure TTS, 400 samples × varied speed/pitch/volume; sliding-window peak detection:

| Keyword | Language | Recall |
|---------|:-------:|:------:|
| 你好小娜 (optimized) | ZH | 100% |
| Hey Jarvis (optimized) | EN | 99.0% |
| Salut Nova (optimized) | FR | 100% |
| Apfelstrudel (optimized) | DE | 98.5% |
| みらい (optimized) | JA | 100% |
| 小娜 (optimized) | ZH | 98.3% |
| 豆包豆包 (optimized) | ZH | 100% |
| Hey Robot | EN | 100% |
| サクラ (Sakura) | JA | 100% |
| Apfelstrudel (base) | DE | 99.4% |
| Monsieur Sadin | FR | 100% |
| Croissant | FR | 90.3% |

> "(optimized)" rows are false-trigger optimized models (hard-negative mining retrain, see below); unmarked rows are base demo models.

> Across 20 recently trained keywords (5 languages): recall 90.3%–100%, mean 98.8%, 20/20 ≥ 90%.

> Real-voice recall reaches 90%+ with 5 user recordings added during training.

### False trigger control

**False-trigger optimization: before vs after** (each keyword measured on its own held-out negative corpus — same-language Common Voice read speech + music + household noise/silence, held out from training; bare model, threshold 0.5, 40ms sliding window, counted per triggered file):

| Keyword | Corpus | Before (triggers/h) | After (triggers/h) | Reduction | Recall |
|---------|---:|---:|---:|---:|---|
| 你好小娜 | 16.0h | 74.5 | 3.5 | −95.3% | 97.8% → 100% |
| Hey Jarvis | 26.9h | 44.6 | 4.8 | −89.2% | 100% → 99.0% |
| Salut Nova | 22.6h | 222.7 | 4.5 | −98.0% | 100% → 100% |
| Apfelstrudel | 24.4h | 265.3 | 2.7 | −99.0% | 97.0% → 98.5% |
| みらい | 12.1h | 330.9 | 5.1 | −98.5% | 97.8% → 100% |

> Numbers above are measured on **keyword-specific held-out corpora** (same-language Common Voice read speech + music + noise/silence, held out from training) — deliberately realistic mixtures, close to **worst case**. On clean public read speech (AISHELL-1, 5000-clip sample) the optimized 你好小娜 drops to **0.8 triggers/hour** (baseline: 35.5).

> All tables above are **bare-model** numbers (detection layers off). The runtime anti-false-trigger layers cut the rate further — with L1 consecutive-frames only (on by default), the optimized 你好小娜 drops from 3.5 to **2.3 triggers/hour** (cons=2) or **1.0** (cons=3) on its 16-hour held-out corpus, before stacking any other layer (L3/L5/...).

> **False-trigger optimized edition**: once a keyword model is trained, one click starts the optimization — the trained model scans its negative audio corpus, finds the segments it wrongly scores as wake words (hard negatives), and retrains with them. The model learns from its own mistakes; **no user-reported audio is required**, and recall is preserved (−91% to −98% in a single round, table above).

**How to use false-trigger optimization**: after your keyword finishes training, open the Voicute console and follow:

```text
My Models → False-Trigger Optimization → Negative scan → Full retrain → Download the R1 model
```

The downloaded R1 model is the false-trigger optimized edition. It loads exactly like the base model (same `model_info.json` and runtime code — no code changes needed) and runs fully offline as usual.

Comparison model pairs included in this repo (`models/<lang>/`):

| Keyword | Language | Baseline (`*_r0.onnx`) | Optimized (`*_r1.onnx`) |
|---------|:-------:|---------|---------|
| 你好小娜 | ZH | `nihaoxiaona_r0.onnx` | `nihaoxiaona_r1.onnx` |
| Hey Jarvis | EN | `heyjarvis_r0.onnx` | `heyjarvis_r1.onnx` |
| Salut Nova | FR | `salutnova_r0.onnx` | `salutnova_r1.onnx` |
| Apfelstrudel | DE | `apfelstrudel_r0.onnx` | `apfelstrudel_r1.onnx` |
| みらい | JA | `mirai_r0.onnx` | `mirai_r1.onnx` |

> False-trigger optimization is production-verified on **all five supported languages** (Chinese, English, Japanese, French, German); the optimization flow (negative scan + full retrain) takes ~40–80 minutes in practice.

> Training takes ~30 minutes per keyword. Currently supports Chinese, English, Japanese, French, and German (5 languages).

---

## Multi-keyword Models

**One model recognizes multiple keywords**: a single inference outputs the probabilities of all keywords at once — no need to run a separate model per keyword. All keywords share one backbone; each extra keyword adds only ~0.8KB of head weights. A 3-keyword model is ~135KB, and even a 10-command model is only ~167KB.

### Multi wake word

Multiple ways of saying the same wake word, packed into one model — every variant wakes the device:

| Package | Keywords |
|---------|---------|
| `models/zh/multi_xiaona_v9.3.zip` | 小娜 · 你好小娜 · 小娜小娜 |
| `models/en/multi_jarvis_v9.3.zip` | Hey Jarvis · Jarvis · Hi Jarvis |
| `models/de/multi_martina_v9.3.zip` | Martina · Tina · Hey Tina |

### Voice control

Ten media commands in a single model:

| Package | Keywords |
|---------|---------|
| `models/zh/multi_commands_v9.3.zip` | 播放 · 暂停 · 下一首 · 上一首 · 开始播放 · 停止播放 · 声音大一点 · 声音小一点 · 静音 · 继续播放 |

ZIP packages load **directly, no extraction needed** (Python and Web engines detect ZIP automatically):

```python
engine.load('models/zh/multi_commands_v9.3.zip', 'models/melspectrogram.onnx')
```

The callback returns the matched keyword and its confidence (see the per-platform examples in Usage below). Custom multi-keyword models are trained at [voicute.com](https://www.voicute.com).

---

## Models

onnx-wakeword is an inference-only open-source project. A compatible keyword model and its matched classification head are required at runtime. Custom models are trained online at [voicute.com](https://www.voicute.com). See the Training & Deployment section above for the full workflow.

### Model files

You need two files:

| File | Purpose |
|------|---------|
| `melspectrogram.onnx` | Audio → mel spectrogram (provided in this repo) |
| `your_model.onnx` | A compatible keyword inference model |

Plus a `model_info.json`:

```json
{
  "model_type": "multi_keyword",
  "keywords": ["Hey Friday"],
  "model_file": "hey_friday.onnx",
  "mel_time": 98,
  "cons_frames": 3,
  "n_mels": 32
}
```

Multi-keyword:

```json
{
  "model_type": "multi_keyword",
  "keywords": ["turn on light", "turn off light", "volume up"],
  "model_file": "commands.onnx",
  "mel_time": 98,
  "cons_frames": 3,
  "n_mels": 32
}
```

Multi-keyword demo models are included in `models/` — see the **Multi-keyword Models** section above.

---

## Usage

### Web

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();
  await engine.load('model_info.json', 'melspectrogram.onnx');  // also supports ZIP
  engine.set_L1(true);
  await engine.start((word, prob) => {
    console.log(`Detected: ${word} (${(prob*100).toFixed(0)}%)`);
  });
</script>
```

### Python

```bash
pip install onnxruntime numpy pyaudio
```

```python
from wakeword_engine import WakeWordEngine

engine = WakeWordEngine()
engine.load('models/model_info.json', 'models/melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'Detected: {word}'))
```

**Mic test:** `python mic_test.py` (default model: xiaona/小娜). Models are searched in `models/zh/`, `models/en/`, etc. Use `--path /full/path/to/model.onnx` to specify a custom model file directly, or `--model manbo` for another built-in keyword.

### Android

```java
WakeWordEngine engine = new WakeWordEngine(context);
engine.load("model_info.json", "melspectrogram.onnx");
DetectionResult result = engine.process(audioChunk);
```

### Home Assistant (Wyoming)

A [Wyoming protocol](https://github.com/rhasspy/wyoming) service is included, so Home Assistant can use onnx-wakeword as a native wake-word engine — no add-on required, and it works on every HA install type (HAOS / Supervised / Container / Core).

**1. Run the service (Docker):**

```bash
docker run -d --name voicute-wakeword --restart unless-stopped --network host \
  -v /path/to/your/models:/models \
  voicute/voicute-wyoming:latest \
  --model-info /models/model_info.json --mel /app/models/melspectrogram.onnx
```

`melspectrogram.onnx` is bundled in the image — mount only your `model_info.json` + keyword `.onnx`.

**2. Add it to Home Assistant:**

Settings → Devices & services → Add Integration → **Wyoming Protocol** → host `IP` + port `10400`.

**3. Select the wake word** in Settings → Voice assistants → your assistant → Wake word.

Full guide (live-mic test, docker-compose, Home Assistant add-on): [`wyoming/README.md`](wyoming/README.md).

---

## Anti-False-Trigger Layers

5 independent layers. Enable progressively:

| Layer | Default | Purpose |
|:-----:|:-------:|---------|
| L1 | **ON** | Consecutive frames — filters transient clicks/noise |
| L3 | OFF | 1.5s cooldown — prevents duplicate triggers |
| L5 | OFF | Energy jump — blocks video/music playback |
| L2 | OFF | Peak/background ratio — prevents silence hallucination |
| L4 | OFF | Burst detection — blocks audio feedback loops |

> **Recommended:** Start with L1 only. Add L3 if double-triggering. Add L5 for noisy environments. L2/L4 rarely needed since v9.3.

---

## Repository Structure

```
onnx-wakeword/
├── android/     # Android (Java, ONNX Runtime)
├── web/         # Web (JavaScript, ONNX Runtime Web)
├── python/      # Linux / Windows / macOS (Python)
├── esp32/       # ESP32-S3/P4 (ESP-IDF)
├── wyoming/     # Home Assistant (Wyoming protocol service)
├── ha-addon/    # Home Assistant add-on
├── Dockerfile   # Docker image (voicute/voicute-wyoming)
└── models/      # Demo models, multi-keyword packages, false-trigger optimization pairs
```

---

## Version

**v9.3 (2026-06)** — Multi-keyword support, English keywords, false-trigger improvements.

See [models/README.md](models/README.md) for model changelog.
