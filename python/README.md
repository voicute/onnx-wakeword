# Python SDK — Keyword Spotting & Wake Word Engine

Real-time microphone keyword detection on Linux / Windows / macOS. Unified API with Web and Android.

---

## Install

```bash
pip install onnxruntime numpy pyaudio
```

## Quick Start

```python
from wakeword_engine import WakeWordEngine

engine = WakeWordEngine()
engine.load('models/model_info.json', 'models/melspectrogram.onnx')

# Configure detection layers
engine.set_L1(True)       # consecutive frames filter

# Start listening
engine.start(lambda word, prob, info: print(f'Detected: {word} ({prob:.0%})'))
```

## Mic Test

```bash
# Basic: L1 only, highest sensitivity
python mic_test.py

# Full: L1-L5 all enabled, lowest false-trigger
python mic_test.py --all

# Options
python mic_test.py --model hey_friday    # switch model
python mic_test.py --thr 0.6             # raise threshold
python mic_test.py --list-devices        # list audio devices
```

## L1-L5 Detection Layers

| Layer | Default | Purpose |
|:-----:|:-------:|---------|
| L1 | **ON** | Consecutive frames — filters transient clicks/noise |
| L3 | OFF | 1.5s cooldown — prevents duplicate triggers |
| L5 | OFF | Energy jump — blocks video/music playback |
| L2 | OFF | Peak/background ratio — prevents silence hallucination |
| L4 | OFF | Burst detection — blocks audio feedback loops |

> Start with L1 only. Add L3 if double-triggering. Add L5 for noisy environments.

## Raspberry Pi (32-bit ARM)

Use TFLite inference — see `infer_tflite.py`.

## Offline False-Trigger Benchmarking

Test false-accept rate (FA/h) using AISHELL-1 Chinese speech dataset:

```bash
# Test 2 hours of data, compare models
python bench_fa.py --aishell-dir /path/to/data_aishell/wav/test \
    --models manbo,manbo_voice,nihaodiannao --hours 2.0

# Per-layer comparison
python test_l2_fa.py --aishell-dir /path/to/data_aishell/wav/test \
    --model ../models/manbo.onnx --mel ../models/melspectrogram.onnx --max-files 500
```

---

## 中文说明

实时麦克风关键词检测，支持 Linux / Windows / macOS。API 与 Web/Android 统一。

### 安装

```bash
pip install onnxruntime numpy pyaudio
```

### 使用

```python
from wakeword_engine import WakeWordEngine

engine = WakeWordEngine()
engine.load('models/model_info.json', 'models/melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'检测到: {word} ({prob:.0%})'))
```

### L1-L5 检测层

| 层 | 推荐 | 说明 |
|:---:|:---:|------|
| L1 | 必开 | 连续帧过滤瞬态噪声 |
| L3 | 建议 | 1.5s 冷却防重复触发 |
| L5 | 按需 | 能量跳变防视频/音乐误触发 |
| L2 | 按需 | 峰值/背景比，防模型幻觉 |
| L4 | 按需 | 爆发封锁防回声回路 |
