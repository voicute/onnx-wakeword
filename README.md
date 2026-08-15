# Voicute — Offline Keyword Spotting (KWS) & Wake Word Engine

[中文文档](README_CN.md)

`KWS` · `Keyword Spotting` · `Wake Word` · `ONNX` · `Edge AI` · `ESP32` · `Android` · `Offline` · `Open Source`

> **100% offline · Model < 130KB · No audio upload · ESP32 / Android / Python / Web**

Voicute is an open-source ONNX inference engine for **wake words and keyword spotting (KWS)**. It runs a **causal temporal convolutional network (Causal TCN, ~25K parameters)** with a mel-spectrogram frontend for on-device inference. TCN models are generated on the [Voicute platform](https://www.voicute.com) and exported as sub-130KB ONNX models that run **fully offline** on ESP32, Android, Web, or desktop. Currently supports Chinese, English, Japanese, French, and German.

---

## Quick Links

| Platform | Directory | Entry Point |
|----------|-----------|-------------|
| **Web** | [`web/`](web/) | `wakeword.js` → `VoicuteWakeWord.create()` |
| **Python** | [`python/`](python/) | `wakeword_engine.py` → `WakeWordEngine()` |
| **Android** | [`android/`](android/) | `WakeWordEngine.java` |
| **ESP32** | [`esp32/`](esp32/) | ESP-IDF component |

---

## Features

- **Sub-130KB models** — 25K parameters, fits ESP32 INT8 flash
- **Multi-keyword** — detect 2–10+ keywords with a single model
- **Multi-language** — Chinese, English, Japanese, French, and German
- **5-layer anti-false-trigger** — consecutive frames, peak/background ratio, cooldown, burst detection, energy jump
- **ZIP packaging** — distribute model + config as a single file

---

## Performance

Tested on **v9.3 models**.

| Metric | Value |
|--------|-------|
| ONNX size | ~128KB (FP32) / ~74KB (INT8) |
| Desktop inference | <5ms / frame |
| ESP32-S3 inference | <10ms / frame |

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

## Getting a Model

Models are trained on the [voicute.com](https://www.voicute.com) platform — type your keyword, get an ONNX model in minutes. **No audio upload required** — the model is generated from synthesized speech (TTS), so you can train a keyword without recording your voice.

- **Basic** — type a keyword, get a model with 90%+ recall in ~30 minutes. Trained entirely on synthesized speech; no audio upload, no recordings.
- **Voice enhancement** — add ~5 of your own recordings for pronunciation-challenged keywords (accents, children's voices, unusual pronunciations). Generalizes to other speakers and lifts real-voice recall.
- **Multi-keyword** — one model that detects 2–10+ keywords at once, sharing a single compact backbone instead of stacking separate models.

### Model files

You need two files:

| File | Purpose |
|------|---------|
| `melspectrogram.onnx` | Audio → mel spectrogram (provided in this repo) |
| `your_model.onnx` | The keyword model (trained on the Voicute platform) |

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

### Android

```java
WakeWordEngine engine = new WakeWordEngine(context);
engine.load("model_info.json", "melspectrogram.onnx");
DetectionResult result = engine.process(audioChunk);
```

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
└── models/      # Demo models
```

---

## Version

**v9.3 (2026-06)** — Multi-keyword support, English keywords, false-trigger improvements.

See [models/README.md](models/README.md) for model changelog.
