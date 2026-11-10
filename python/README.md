# onnx-wakeword (Python package)

[中文说明](#中文说明)

Offline wake-word / keyword-spotting for Python — powered by ONNX Runtime, no cloud, models as small as ~130 KB.

- **pip installable**: `pip install onnx-wakeword`
- **Zero audio-library lock-in**: core inference is pure `numpy` + `onnxruntime`; microphone support is an optional extra
- **Bundled English demo model** ("Hey Jarvis", false-trigger-optimized edition) — works out of the box
- **Bring your own model**: any bundle from [voicute.com](https://www.voicute.com) (Chinese, English, French, German, Japanese, Spanish, Hindi) or trained yourself

## Supported Python versions

Python **3.9 – 3.13**, on Windows / Linux / macOS (x86_64 and arm64 — anything ONNX Runtime supports).

## Install

```bash
pip install onnx-wakeword                # core: inference only (numpy + onnxruntime)

# optional microphone support — pick ONE, you don't need both:
pip install "onnx-wakeword[audio]"        # pyaudio — for WakeWordEngine.start()
pip install "onnx-wakeword[sounddevice]"  # sounddevice — for mic_test.py / infer.py scripts
```

The core package never imports an audio library. `pyaudio` and `sounddevice` are deliberately separate extras so servers / CI can skip them.

## Quick start — bundled demo

```python
from onnx_wakeword import WakeWordEngine, get_demo_model

engine = WakeWordEngine()
engine.load(*get_demo_model())   # bundled "Hey Jarvis" model, absolute paths
engine.set_L1(True)              # consecutive-frames gate
engine.start(
    lambda word, probability, info: print(word, probability)
)
```

Or from your terminal:

```bash
onnx-wakeword-demo                        # microphone (needs the [audio] extra)
onnx-wakeword-demo --wav speech.wav       # offline scan of a WAV file — no mic needed
onnx-wakeword-demo --info /path/model_info.json --mel /path/melspectrogram.onnx
```

## Quick start — your own model

```python
from onnx_wakeword import WakeWordEngine

engine = WakeWordEngine()
# Absolute or relative paths both work; model files inside the JSON are
# resolved relative to the JSON file, never the current working directory.
engine.load(
    model_info_path="/absolute/path/model_info.json",
    mel_path="/absolute/path/melspectrogram.onnx",
)
engine.start(lambda word, prob, info: print(word, prob))
```

A model bundle is `model_info.json` + one or more keyword `.onnx` files + `melspectrogram.onnx` — or a single `.zip` containing all of them (the format voicute.com exports). See the [models directory](../models/) for free demo models in 7 languages.

## Detection layers (L1–L5)

All production detection logic is included and individually toggleable:

| Layer | Purpose | Default |
| :---: | ------- | :-----: |
| L1 | N consecutive frames above threshold (with brief-gap tolerance) | on |
| L2 | peak ≫ background (model-fluctuation filter) | off |
| L3 | cooldown against double-triggers | off |
| L4 | burst suppression (playback-loop filter) | off |
| L5 | energy-jump gate (video/music filter, post-speech confirmation) | off |

```python
engine.set_L2(True)
engine.set_L3(True)
engine.set_threshold(0.5)        # clamped to [0.3, 0.95]
engine.set_cooldown(1500)        # ms
```

`DetectionLogic` is also exported standalone for custom pipelines (see `mic_test.py` for the record/evaluate pattern).

## No microphone? Core still runs

Everything except `engine.start()` works without any audio hardware:

```python
result = engine.predict(audio)   # float32 array, int16 range, 16 kHz
detected = engine.detect(result['word'], result['prob'], result['cons_frames'])
```

## Repository scripts vs. package

The legacy scripts still run unchanged (`python wakeword_engine.py`, `python mic_test.py`, `python infer.py`, `python infer_tflite.py`) — they now import the package. Use the package API for new code.

## ONNX vs. TFLite

This package runs the **ONNX** models (`.onnx`) via onnxruntime — the right choice for Linux servers, Raspberry Pi-class SBCs, and desktops. The same models also export to TFLite for ESP32 microcontrollers (`infer_tflite.py`, and the [esp32/](../esp32/) directory); TFLite is not part of this pip package.

## License

MIT — the bundled demo model is trained in-house from synthetic speech and is redistributable. Custom wake-word models for your own keyword: [voicute.com](https://www.voicute.com).

---

## 中文说明

离线唤醒词 / 关键词识别 Python 包 — 基于 ONNX Runtime，无云端依赖，模型最小约 130 KB。

```bash
pip install onnx-wakeword                 # 核心推理（numpy + onnxruntime）
pip install "onnx-wakeword[audio]"        # 麦克风（pyaudio，engine.start() 用）
pip install "onnx-wakeword[sounddevice]"  # sounddevice（mic_test.py / infer.py 用）
```

```python
from onnx_wakeword import WakeWordEngine, get_demo_model

engine = WakeWordEngine()
engine.load(*get_demo_model())   # 内置英文 Demo「Hey Jarvis」
engine.start(lambda word, prob, info: print(word, prob))
```

命令行：

```bash
onnx-wakeword-demo                     # 麦克风监听
onnx-wakeword-demo --wav speech.wav    # 离线扫描 WAV，无需麦克风
```

- 支持 Python 3.9–3.13，Windows / Linux / macOS
- 自定义模型：`engine.load("你的/model_info.json", "你的/melspectrogram.onnx")`，支持绝对路径、ZIP 包
- L1–L5 检测层全部可开关（连帧门控 / 峰值背景比 / 冷却 / 防循环 / 能量跳变）
- 无麦克风环境下核心推理照常可用（`predict()` / `detect()` 纯 numpy）
- 核心包不依赖任何音频库；pyaudio 与 sounddevice 均为可选 extra，二选一即可
- 免费多语言 Demo 模型见 [models/](../models/)；训练自定义唤醒词模型见 [voicute.com](https://www.voicute.com)
- 许可证 MIT；内置 Demo 模型为自制 TTS 训练产物，可随包再分发
