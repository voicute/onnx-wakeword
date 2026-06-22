# Linux / macOS / Windows 命令行推理

Python SDK，麦克风实时唤醒词检测。API 与 Web/Android 统一。

## 安装

```bash
pip install onnxruntime numpy pyaudio
```

## 使用

```python
from wakeword_engine import WakeWordEngine

engine = WakeWordEngine()
engine.load('models/model_info.json', 'models/melspectrogram.onnx')

# 配置检测层
engine.set_L1(True)       # 连续帧过滤
engine.set_L5(True)       # 能量跳变
engine.set_L5_ratio(3.0)  # L5 倍数

# 实时监听
engine.start(lambda word, prob, info: print(f'检测到: {word} ({prob:.0%})'))
```

## 命令行 Demo

```bash
python wakeword_engine.py models/model_info.json models/melspectrogram.onnx
```

## API 完整对照

| Web JS | Linux Python |
|------|------|
| `VoicuteWakeWord.create()` | `WakeWordEngine()` |
| `engine.load(info, mel)` | `engine.load(info, mel)` |
| `engine.start(cb)` | `engine.start(cb)` |
| `engine.predict(audio)` | `engine.predict(audio)` |
| `engine.set_L1(v)` | `engine.set_L1(v)` |
| `engine.set_L5_ratio(v)` | `engine.set_L5_ratio(v)` |
| `engine.stop()` | `engine.stop()` |

## L1-L5 检测层

| 层 | 默认 | 说明 |
|:---:|:---:|------|
| L1 | on / 3帧 | 连续帧过滤瞬态噪声 |
| L2 | off | 峰值/背景比 |
| L3 | off | 1.5s 冷却 |
| L4 | off | 爆发封锁 |
| L5 | off | 能量跳变（防视频/音乐） |

## 32 位树莓派

32 位 ARM 用 TFLite 推理，见 `infer_tflite.py`。

## 离线误触发评测

用 AISHELL-1 中文语音数据集测试模型误触发率（FA/h），可独立开关每层检测层。

### 快速评测（推荐）

```bash
# 测试 2 小时 AISHELL 数据，对比多个模型
python bench_fa.py --aishell-dir /path/to/data_aishell/wav/test \
    --models manbo,manbo_voice,nihaodiannao --hours 2.0

# 参数
# --aishell-dir  (必填) AISHELL-1 wav 目录
# --models       模型名: manbo, manbo_voice, nihaodiannao, kaishibofang, gugugaga, laifu
# --hours        目标测试时长 (默认 2.0)
# --thr          检测阈值 (默认 0.5)
```

### 逐层对比测试

```bash
# 测试单个模型，对比各检测层效果
python test_l2_fa.py --aishell-dir /path/to/data_aishell/wav/test \
    --model ../models/manbo.onnx --mel ../models/melspectrogram.onnx --max-files 500
```

### 测试数据

使用 [AISHELL-1](https://www.openslr.org/33/) 中文语音数据集（不含唤醒词），滑窗推理，统计每小时误触发次数。测试脚本自动从 `data/negative/aishell/data_aishell/wav/test/` 读取 WAV 文件。
