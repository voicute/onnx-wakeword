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
| ESP32-S3 TFLite Invoke (included demo) | ~155ms / frame |

**Keyword recall** (held-out Azure TTS, 10 speakers × varied speed/pitch/volume; sliding-window peak detection):

| Keyword | Language | Recall |
|---------|:-------:|:------:|
| 小坦小坦 | ZH | 100% |
| 元宝元宝 | ZH | 100% |
| 你好琥珀 | ZH | 100% |
| 小黑 | ZH | 97.4% |
| Hey Robot | EN | 100% |
| サクラ (Sakura) | JA | 100% |
| Apfelstrudel | DE | 99.4% |
| Monsieur Sadin | FR | 100% |
| Croissant | FR | 90.3% |

> Across 20 recently trained keywords (5 languages): recall 90.3%–100%, mean 98.8%, 20/20 ≥ 90%.

> Real-voice recall reaches 90%+ with 5 user recordings added during training.

**False trigger resistance** (validated on ~25,000 held-out negatives across speech, music, noise, near-wake phrases; threshold 0.5):

| Metric | Value |
|--------|-------|
| Negative accuracy (validation set) | 96–98% |

> **Negative sample mining** (in development): For individual keywords experiencing false triggers, we plan to collect user-reported false trigger audio and feed it back into training as weighted hard negatives, continuously improving per-keyword accuracy.

> Training takes ~30 minutes per keyword. Currently supports Chinese, English, Japanese, French, and German (5 languages).

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

**Mic test:** `python mic_test.py` (default model: hey_limi). Models are searched in `models/zh/`, `models/en/`, etc. Use `--path /full/path/to/model.onnx` to specify a custom model file directly, or `--model manbo` for another built-in keyword.

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
└── models/      # Demo models
```

---

## Version

**v9.3 (2026-06)** — Multi-keyword support, English keywords, false-trigger improvements.

See [models/README.md](models/README.md) for model changelog.
