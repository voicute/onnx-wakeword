# Linux / macOS / Windows 命令行推理

Python SDK，麦克风实时唤醒词检测。API 与 Web/Android 统一。

## 安装

```bash
pip install onnxruntime numpy sounddevice
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

## 快速测试（麦克风）

```bash
# 基础模式：只开 L1（连续帧），最高灵敏度
python mic_test.py

# 全开模式：L1-L5 全部启用，最低误触发
python mic_test.py --all

# 其他选项
python mic_test.py --model nihaodiannao    # 换模型
python mic_test.py --thr 0.6               # 提高阈值
python mic_test.py --list-devices           # 查看麦克风设备
```

## L1-L5 检测层

| 层 | 推荐 | 说明 | 什么时候开 |
|:---:|:---:|------|------|
| L1 | **必开** | 连续帧过滤瞬态噪声（键盘、椅子响） | 始终开启 |
| L3 | **建议开** | 1.5s 冷却防重复触发 | L1 不够时开 |
| L5 | 按需 | 能量跳变防视频/音乐误触发 | 音箱/嘈杂环境 |
| L2 | 按需 | 峰值/背景比，防模型幻觉 | 极安静环境有误触时 |
| L4 | 按需 | 爆发封锁防回声回路 | 喇叭播放回声导致误触时 |

> **建议流程**：先只开 L1 测识别率 → 有连击开 L3 → 有噪音误触发开 L5 → 还不行再开 L2/L4。不要一次全开，层数越多识别越慢越严格。

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
