# Voicute — Offline Wake Word / Keyword Spotting

`wake word` · `keyword spotting` · `KWS` · `voice trigger` · `custom wake word` · `offline` · `ONNX` · `edge inference` · `ESP32` · `Android` · `open source`

> **Runs fully offline · Model < 130KB · No audio uploaded · ESP32 / Android / Linux / Web**

---

## 🚧 English Wake Word Support — Coming Soon

**English wake word training and recognition is under active development.**

Currently the training pipeline and pre-trained models are optimized for **Chinese wake words** ( Mandarin, 2–4 character phrases ). English phoneme modeling, TTS data generation, and negative sample datasets are being prepared. If you need English wake word support, please watch this repository or contact us — we expect to ship initial English support in a future release.

In the meantime, you can use the existing ONNX runtime inference code on any platform (Android, Web, Linux, ESP32) with your own English models trained via [OpenWakeWord](https://github.com/dscripka/openWakeWord), [MicroWakeWord](https://github.com/kahrendt/microWakeWord), or any KWS framework that exports ONNX.

---

## What is this?

This repository provides **cross-platform, offline wake word / keyword spotting inference**. Bring your own ONNX model and run it on Android, Web, Linux, or ESP32 — no cloud, no internet, zero audio leaves the device.

## Web Demo

![Web screenshot](web/screenshot.png)

Built-in **false-trigger prevention panel** (5-layer detection toggles, L5 energy-jump slider, threshold control, confidence bar).

## Getting an ONNX Model

### Train your own

Any KWS framework that exports ONNX works:

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — open source, home assistant scenes, multi-wake-word
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — designed for ESP32-class MCUs
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — 11 architectures, models as small as 40KB
- Any ONNX-exportable KWS training pipeline

### Online generation

[voicute.com](https://www.voicute.com) — type a Chinese wake word, get an ONNX model. First one free.

## Model Files

Whichever tool you use, you need two files:

| File | Purpose |
|------|---------|
| `melspectrogram.onnx` | Audio → Mel spectrogram (shared, **provided in this repo**) |
| `your_model.onnx` | Wake word classifier (train or generate) |

Plus a `model_info.json` describing the configuration.

> Demo models are included in `models/`. See [models/README.md](models/README.md) for version info.

## Model Configuration

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "hey computer", "model_file": "hey_computer.onnx", "cons_frames": 3}
  ]
}
```

Multiple wake words:

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "turn on lights", "model_file": "lights.onnx", "cons_frames": 3},
    {"wake_word": "hey computer", "model_file": "computer.onnx", "cons_frames": 3}
  ]
}
```

## Directory Structure

```
onnx-wakeword/
├── android/     # Android (Java, ONNX Runtime)
├── web/         # Web (JavaScript, ONNX Runtime Web)
├── linux/       # Linux / Raspberry Pi (Python)
├── esp32/       # ESP32-S3/P4 (C, TFLite Micro)
└── models/      # Test models & configs (not committed to git)
```

## Platform Quick Start

| Platform | Directory | Entry Point |
|----------|-----------|-------------|
| Android | `android/` | `WakeWordEngine.java` |
| Web | `web/` | `wakeword.js` → `VoicuteWakeWord.create()` |
| Linux | `linux/` | `wakeword_engine.py` → `WakeWordEngine()` |
| ESP32 | `esp32/` | `recognizer.h` → `recognizer_start()` |

### Web

```html
<script src="onnxruntime-web/ort.min.js"></script>
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();
  // Supports local paths, remote URLs, and ZIP bundles
  await engine.load('model_info.json', 'melspectrogram.onnx');
  engine.set_L1(true);
  await engine.start((word, prob) => {
    console.log(`Detected: ${word} (${(prob*100).toFixed(0)}%)`);
  });
</script>
```

### Linux

```bash
pip install onnxruntime numpy pyaudio
```

```python
from wakeword_engine import WakeWordEngine
engine = WakeWordEngine()
engine.load('model_info.json', 'melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'{word}'))
```

### Android

Copy models to `assets/`, build & run.

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
```

### ESP32

```c
#include "recognizer.h"

recognizer_config_t cfg = {
    .model_path    = "/spiffs",
    .threshold     = 0.7f,
    .l5_jump_ratio = 5.0f,
    .l5_enabled    = 1,
};
recognizer_start(&audio_rb, &cfg);
```

## False-Trigger Prevention (L1–L5)

Five independent detection layers. Enable all five in production:

| Layer | Default | What it solves |
|:-----:|:-------:|----------------|
| L1 Consecutive frames | ON / 2–3 frames | Transient noises (keyboard clicks, chair creaks) |
| L2 Peak / background | OFF | Model hallucination on quiet-room noise |
| L3 Cooldown | OFF | Same phrase detected multiple times |
| L4 Burst block | OFF | Audio feedback loops causing rapid re-triggers |
| L5 Energy jump ratio | OFF | Continuous sounds (video playback, background music) |

L5 ratio is adjustable: `engine.set_L5_ratio(3.0)` (range 2.0–8.0×, default 3.0)

## How L5 Works

L5 checks whether the current audio is a **human voice burst** (isolated energy spike) vs. **continuous playback**:

1. **Pre-check** (L5a): Looks back 0.5–2.0s. If `curRms / minRms > jumpRatio`, it's likely a human speaking — set pending.
2. **Post-check** (L5b): Waits 700ms after the burst, then checks the trailing 300ms. If the energy dropped back down (`postMin < preMin × 2.5`), confirms it was an isolated utterance.
3. **Guard clauses**: Quiet-room skip (`preMin < 50 && rms < 80`), steady-noise block (`rms < preMin × jumpRatio`).

This makes L5 effective against music/video while remaining sensitive to real voice commands.

## Version History

**v9.3 (2026-06)**
- Web: Fixed double run() scheduling causing detection freeze
- Web: Added cache-control meta tags for WeChat WebView
- Web: Support model_type='tcn'
- Models upgraded to v9.3 with significantly reduced false triggers

Model update log: [models/README.md](models/README.md)

## License

Open source. See individual platform directories for details.
