# Voicute — Offline Keyword Spotting & Wake Word Engine

[中文文档](README_CN.md)

`Keyword Spotting` · `Wake Word` · `KWS` · `ONNX` · `Edge AI` · `ESP32` · `Android` · `Offline` · `Open Source`

> **100% offline · Model < 130KB · No cloud dependency · ESP32 / Android / Python / Web**

Voicute is an open-source, cross-platform inference engine for custom keyword spotting and wake word detection. Bring your own ONNX model — or [generate one online](https://www.voicute.com) — and run **fully offline** on ESP32, Android, Web, or desktop.

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
- **Multi-language** — Chinese, English, and 150+ languages via custom training
- **5-layer anti-false-trigger** — consecutive frames, peak/background ratio, cooldown, burst detection, energy jump
- **ZIP packaging** — distribute model + config as a single file

---

## Performance

Tested on **v9.3 models** (~25K params, ~128KB ONNX, 2000 pos + 15000 neg training samples).

| Metric | Value |
|--------|-------|
| Parameters | ~25K |
| ONNX size | ~128KB (FP32) / ~25KB (INT8) |
| Desktop inference | <5ms / frame |
| ESP32-S3 inference | <10ms / frame |

**Keyword recall** (held-out Azure TTS, 10 speakers × varied speed/pitch/volume):

| Keyword | Language | Recall |
|---------|:-------:|:------:|
| 小坦小坦 | ZH | 96.7% |
| 打开灯光 | ZH | 94.2% |
| Cyclops | EN | 92.3% |
| Hey Friday | EN | 91.5% |
| Hey Limi | EN | 89.9% |

> Real-voice recall reaches 90%+ with 5–20 user recordings added during training.

**False trigger resistance** (20,000 negative WAVs across speech, music, noise, near-wake phrases; threshold 0.5, L1+L3 enabled):

| Metric | Value |
|--------|-------|
| Negative accuracy (static test set) | 93–96% |
| Real-world estimate (quiet indoor) | 0.1–0.3 / hour |
| With L5 enabled | ~0.1 / hour |

**Anti-false-trigger layers** (quiet-room ambient noise test):

| Configuration | FA / Hour | Reduction |
|:---|:---:|:---:|
| Off | ~2.5 | — |
| L1 | ~0.7 | -72% |
| L1 + L3 | ~0.2 | -91% |
| L1 + L3 + L5 | ~0.1 | -96% |

> Models are trained per-keyword with ~2,000 TTS positives + ~15,000 negatives.
> Training takes ~30 minutes per keyword. Supports Chinese, English, and 150+ languages.

---

## Getting a Model

### Online (recommended)

Go to [voicute.com](https://www.voicute.com), type your keyword, get an ONNX model in minutes.

### Train yourself

Any KWS framework that exports ONNX works:

- [OpenWakeWord](https://github.com/dscripka/openWakeWord)
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord)
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword)

### Model files

You need two files:

| File | Purpose |
|------|---------|
| `melspectrogram.onnx` | Audio → mel spectrogram (provided in this repo) |
| `your_model.onnx` | The keyword model (train or generate) |

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

**v9.3 (2026-06)** — Multi-keyword support, English wake words, false-trigger improvements.

See [models/README.md](models/README.md) for model changelog.
